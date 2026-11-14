"""Cross-seed model-path divergence, recomputed straight from the run snapshots."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_SEED_RE = re.compile(r"^(?P<base>.+)__s(?P<seed>\d+)$")
DETECTORS = ("frcnn_mobilenet", "frcnn_resnet50")


def collect_groups(runs_root: Path) -> dict:
    """(class, model, regime) -> {seed: (run_dir, config)} for detector runs."""
    groups: dict = defaultdict(dict)
    for cfg_path in sorted(runs_root.glob("*/*/*/config.json")):
        run = cfg_path.parent
        if not (run / "snapshots.npz").exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        if cfg.get("model") not in DETECTORS:
            continue
        m = _SEED_RE.match(run.name)
        if m:
            groups[(cfg["class_key"], cfg["model"], m["base"])][int(m["seed"])] = (run, cfg)
    return groups


def cell_rows(cls: str, model: str, regime: str, by_seed: dict) -> list[dict]:
    """Per-snapshot agreement rows for one (class, model, regime) group."""
    seeds = sorted(by_seed)
    gs, steps_per_seed = [], []
    for s in seeds:
        z = np.load(by_seed[s][0] / "snapshots.npz")
        gs.append(np.asarray(z["g"], np.float64))
        steps_per_seed.append(np.asarray(z["snapshot_steps"]))

    # Align on the steps every seed reached; seeds can stop at different depths.
    shared = sorted(set.intersection(*(set(map(int, st)) for st in steps_per_seed)))
    cfg = by_seed[seeds[0]][1]
    N, w = int(cfg["N"]), int(cfg["warmup_n"])

    rows = []
    for step in shared:
        G = np.stack([g[int(np.flatnonzero(st == step)[0])]
                      for g, st in zip(gs, steps_per_seed)])   # (n_seed, N)
        total = G.sum(axis=1)
        mean_total = float(total.mean())
        rows.append(dict(
            class_key=cls, model=model, regime=regime, step=step,
            frac_labeled=(w + step) / N, n_seeds=len(seeds),
            F_model_cv=(float(total.std(ddof=1) / mean_total)
                        if mean_total > 0 else np.nan),
        ))
    return rows


def report(df: pd.DataFrame) -> None:
    """The appendix's Limitations numbers, printed for eyeball verification."""
    lo, hi = df[df.frac_labeled <= 0.20], df[df.frac_labeled >= 0.30]
    per_task = lo.groupby("class_key").F_model_cv.max()
    print(f"\nF_model_cv below 20% budget: median {lo.F_model_cv.median():.3f}, "
          f"max {lo.F_model_cv.max():.2f}; "
          f"{int((per_task > 0.9).sum())} of {len(per_task)} tasks peak above 0.9")
    print(f"F_model_cv from 30% budget:  median {hi.F_model_cv.median():.3f}, "
          f"max {hi.F_model_cv.max():.2f}")
    print("\nworst tasks below 20% budget (max cross-seed CV, by model):")
    worst = (lo.groupby(["class_key", "model"]).F_model_cv.max()
               .sort_values(ascending=False).head(5))
    for (cls, model), v in worst.items():
        print(f"  {cls:<26} {model:<17} {v:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default="outputs/runs")
    ap.add_argument("--out", default="outputs/analysis/path_invariance/path_agreement.csv")
    args = ap.parse_args()

    groups = collect_groups(Path(args.runs_root))
    rows: list[dict] = []
    for (cls, model, regime), by_seed in sorted(groups.items()):
        if len(by_seed) >= 2:
            rows += cell_rows(cls, model, regime, by_seed)

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} rows over "
          f"{df.groupby(['class_key','model','regime']).ngroups} cells -> {out}")
    report(df)


if __name__ == "__main__":
    main()
