"""Package the MLLM per-tile predictions as zero-shot ``g_fixed`` stores."""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent

DEFAULT_CSVS = [
    "outputs/runs/open_source_all_tiles_current.csv",
    "outputs/runs/proprietary_all_tiles_current.csv",
]
# Columns the store needs; the rest (NL response, request keys, ...) are ignored.
USECOLS = [
    "dataset", "task", "tileid",
    "ground_truth_label", "final_value",
    "model_label", "modelstr",
]
# Detector-inference knobs on a VLM config that would be misleading on an MLLM
# store; dropped when we copy the sibling config.
_DROP_CFG_KEYS = ("score_thresh", "capture_score_floor",
                  "text_prompt", "text_prompt_resolved")


def slugify(modelstr: str) -> str:
    """``org/Model-Name`` -> filename-safe slug, dropping the org prefix.

    Keeps alphanumerics, dots (version numbers stay readable), and hyphens;
    every other run collapses to a single ``-``.  The slug lands in run-dir
    names and in ``collect_sims.py``'s ``{class}__{model}.csv`` output name, so
    it must contain no ``/``.  ``model_label`` keeps the pretty display name.
    """
    name = modelstr.rsplit("/", 1)[-1].lower()
    name = re.sub(r"[^a-z0-9.]+", "-", name).strip("-.")
    if not name:
        raise ValueError(f"modelstr {modelstr!r} slugified to empty")
    return name


def find_reference_store(class_dir: Path, preferred: str) -> Optional[Path]:
    """A sibling VLM ``g_fixed`` store to source manifest + tiling config from.

    Prefers ``<preferred>__g_fixed`` (grounding_dino exists for every class);
    otherwise any ``*__g_fixed`` sibling with a manifest + config.
    """
    pref = class_dir / f"{preferred}__g_fixed"
    if (pref / "manifest.csv").exists() and (pref / "config.json").exists():
        return pref
    for cand in sorted(class_dir.glob("*__g_fixed")):
        if (cand / "manifest.csv").exists() and (cand / "config.json").exists():
            return cand
    return None


def build_g(sub: pd.DataFrame, manifest: pd.DataFrame, class_key: str) -> np.ndarray:
    """``final_value`` reindexed onto the manifest's ``tile_index`` order.

    Asserts the two alignment invariants (tile set + ground truth) so a
    misaligned drop can never yield a silently shifted ``g``.
    """
    man = manifest.sort_values("tile_index").reset_index(drop=True)
    sub = sub.sort_values("tileid").reset_index(drop=True)

    N = len(man)
    if len(sub) != N:
        raise SystemExit(
            f"{class_key}: {len(sub)} MLLM tiles but sibling manifest has {N}; "
            "the MLLM run does not cover the same pool -- refusing to write."
        )
    if not np.array_equal(sub["tileid"].to_numpy(), man["tile_index"].to_numpy()):
        raise SystemExit(
            f"{class_key}: tileid does not match tile_index row-for-row -- "
            "tile ordering diverged, refusing to write."
        )
    n_gt_bad = int((sub["ground_truth_label"].to_numpy() != man["gt_count"].to_numpy()).sum())
    if n_gt_bad:
        raise SystemExit(
            f"{class_key}: ground_truth_label != gt_count on {n_gt_bad} tiles -- "
            "the MLLM run's ground truth disagrees with the canonical manifest, "
            "refusing to write."
        )

    g = pd.to_numeric(sub["final_value"], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(g).all():
        raise SystemExit(
            f"{class_key}: {int((~np.isfinite(g)).sum())} non-finite final_value; "
            "clean or impute before building a g vector."
        )
    return g.astype(np.float32).reshape(1, N)


def write_store(store: Path, ref: Path, cfg: dict, g: np.ndarray) -> None:
    store.mkdir(parents=True, exist_ok=True)
    (store / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    shutil.copyfile(ref / "manifest.csv", store / "manifest.csv")
    # Written like ExperimentRecorder: g stacked over snapshots, one step at 0.
    buf = io.BytesIO()
    np.savez(buf, g=g, snapshot_steps=np.asarray([0], dtype=np.int64))
    (store / "snapshots.npz").write_bytes(buf.getvalue())


def make_config(ref_cfg: dict, class_key: str, slug: str, model_label: str,
                modelstr: str, g: np.ndarray, source_csv: str) -> dict:
    cfg = {k: v for k, v in ref_cfg.items() if k not in _DROP_CFG_KEYS}
    cfg["run_id"] = f"{slug}__g_fixed"
    cfg["regime"] = "g_fixed"
    cfg["class_key"] = class_key
    cfg["model"] = slug
    cfg["model_label"] = model_label
    cfg["modelstr"] = modelstr
    cfg["source_csv"] = source_csv
    # N / F_true are authoritative from the reused manifest (== ref_cfg's), and
    # F_model records this model's total predicted count for a quick eyeball.
    cfg["F_model"] = float(g.sum())
    return cfg


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", nargs="+", default=DEFAULT_CSVS,
                    help="MLLM prediction CSV(s) (default: the two current dumps)")
    ap.add_argument("--out-root", default="outputs/runs",
                    help="run-tree root the sibling VLM stores live under")
    ap.add_argument("--reference-model", default="grounding_dino",
                    help="which sibling __g_fixed store to source manifest/config from")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing __g_fixed store (default: skip)")
    ap.add_argument("--dry-run", action="store_true",
                    help="run every alignment check and report, but write nothing")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    out_root = (REPO / args.out_root) if not Path(args.out_root).is_absolute() else Path(args.out_root)

    frames = []
    for c in args.csv:
        p = (REPO / c) if not Path(c).is_absolute() else Path(c)
        df = pd.read_csv(p, usecols=USECOLS, low_memory=False)
        df["source_csv"] = Path(c).name
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    written, skipped, rows = 0, 0, []
    for (dataset, task, modelstr), sub in data.groupby(["dataset", "task", "modelstr"], sort=True):
        class_key = f"{dataset}/{task}"
        class_dir = out_root / dataset / task
        ref = find_reference_store(class_dir, args.reference_model)
        if ref is None:
            raise SystemExit(
                f"{class_key}: no sibling *__g_fixed store under {class_dir} to "
                "source a manifest/config from."
            )

        labels = sub["model_label"].dropna().unique()
        if len(labels) != 1:
            raise SystemExit(f"{class_key}/{modelstr}: model_label not unique: {list(labels)}")
        model_label = str(labels[0])
        slug = slugify(modelstr)

        manifest = pd.read_csv(ref / "manifest.csv")
        g = build_g(sub, manifest, class_key)  # asserts alignment invariants
        ref_cfg = json.loads((ref / "config.json").read_text(encoding="utf-8"))
        cfg = make_config(ref_cfg, class_key, slug, model_label, modelstr,
                          g, str(sub["source_csv"].iloc[0]))

        store = class_dir / f"{slug}__g_fixed"
        n_big = int((g[0] > 500).sum())
        rows.append((class_key, slug, g.shape[1], float(g.sum()), n_big))

        if store.exists() and not args.overwrite:
            skipped += 1
            continue
        if args.dry_run:
            continue
        write_store(store, ref, cfg, g)
        written += 1

    rep = pd.DataFrame(rows, columns=["class_key", "model", "N", "F_model", "g>500"])
    print(rep.to_string(index=False))
    print()
    print(rep.groupby("model").agg(classes=("N", "size"), tiles=("N", "sum"),
                                    outliers=("g>500", "sum")).to_string())
    print()
    action = "would write" if args.dry_run else "wrote"
    print(f"{action} {len(rows) - skipped} stores, skipped {skipped} existing "
          f"(--overwrite to replace); {rep['model'].nunique()} models x "
          f"{rep['class_key'].nunique()} classes")
    if not args.dry_run:
        print(f"next: python3 scripts/collect_sims.py --models <slug>   # fold into estimates_checkpoints.csv")


if __name__ == "__main__":
    sys.exit(main())
