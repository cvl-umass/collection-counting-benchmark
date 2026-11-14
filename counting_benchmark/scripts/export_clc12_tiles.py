"""Export CLC-12 as two lightweight CSVs.

    clc12_tiles.csv                 one row per counting unit (tile): where it is
                                    and its ground-truth count
    clc12_zeroshot_predictions.csv  the frozen zero-shot count each of the 11
                                    off-the-shelf models predicts for that unit

Both are repackagings of files already under ``outputs/runs/``: the per-task
``manifest.csv`` and the ``*__g_fixed/snapshots.npz`` stores.  Nothing is
recomputed, and no imagery is read.

    python3 scripts/export_clc12_tiles.py            # -> ../clc12/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TILE_COLUMNS = [
    "task", "dataset", "class", "tier", "tile_index", "image_id", "image_path",
    "x0", "y0", "width", "height", "is_tiled", "gt_count",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-root", type=Path, default=ROOT / "outputs" / "runs")
    ap.add_argument("--selected", type=Path,
                    default=ROOT / "outputs" / "selected_classes.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT.parent / "clc12")
    args = ap.parse_args()

    tasks = json.loads(args.selected.read_text())["classes"]
    tiles, preds = [], []

    for task in tasks:
        key = task["class_key"]
        task_dir = args.runs_root / key

        m = pd.read_csv(task_dir / "manifest.csv")
        if not (m["tile_index"].to_numpy() == np.arange(len(m))).all():
            raise ValueError(f"{key}: tile_index is not 0..N-1 in order")
        if len(m) != task["n_units"] or int(m["gt_count"].sum()) != task["F_total"]:
            raise ValueError(f"{key}: manifest disagrees with selected_classes.json")

        m = m.rename(columns={"pw": "width", "ph": "height"})
        m.insert(0, "task", key)
        m["dataset"], m["class"], m["tier"] = task["dataset"], task["class"], task["level"]
        tiles.append(m[TILE_COLUMNS])

        p = m[["task", "tile_index"]].copy()
        for store in sorted(task_dir.glob("*__g_fixed")):
            cfg = json.loads((store / "config.json").read_text())
            g = np.load(store / "snapshots.npz")["g"]
            if g.shape != (1, len(m)) or cfg["N"] != len(m):
                raise ValueError(f"{store}: expected one prediction per unit")
            p[cfg["model"]] = g[0].astype(np.int64)
        preds.append(p)

    models = [c for c in preds[0].columns if c not in ("task", "tile_index")]
    for p in preds:
        if [c for c in p.columns if c not in ("task", "tile_index")] != models:
            raise ValueError("tasks do not share the same set of zero-shot models")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tiles_df, preds_df = pd.concat(tiles), pd.concat(preds)
    tiles_df.to_csv(args.out_dir / "clc12_tiles.csv", index=False)
    preds_df.to_csv(args.out_dir / "clc12_zeroshot_predictions.csv", index=False)

    print(f"{len(tiles_df)} units across {len(tasks)} tasks, {len(models)} models")
    print(f"wrote {args.out_dir / 'clc12_tiles.csv'}")
    print(f"wrote {args.out_dir / 'clc12_zeroshot_predictions.csv'}")


if __name__ == "__main__":
    main()
