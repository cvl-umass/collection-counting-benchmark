"""Per-class counting-regime statistics (empty fraction, density) under the frozen tiling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.tiled_dataset import CountingClassTiledDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index-root", default="../counting-datasets/counting/data")
    ap.add_argument("--raw-root", default="../counting-datasets/raw")
    ap.add_argument("--tile-json", default="outputs/suggested_tile_sizes.json")
    ap.add_argument("--pool-dir", default="data/image_pools")
    ap.add_argument("--out", default="outputs/regimes_v2.csv")
    ap.add_argument(
        "--whole-class",
        action="store_true",
        help="Ignore the pool lists; characterize the whole class instead of "
        "the budgeted pool rather than the budgeted subset.",
    )
    return ap.parse_args()


# Tier levels: a 1-D partition on empty-fraction (%∅), the dominant axis once
# the tiling normalizes apparent object size.  CV is reported per class as a
# within-level dispersion diagnostic, not a gate.  Ordered by emptiness.
_LEVELS = [("full", 0.05), ("partial", 0.60), ("sparse", 0.90), ("rare", 1.01)]
_LEVEL_ORDER = {name: i for i, (name, _) in enumerate(_LEVELS)}


def assign_level(pct0: float) -> str:
    """Bucket a class by empty-fraction into full/partial/sparse/rare."""
    for name, hi in _LEVELS:
        if pct0 < hi:
            return name
    return "rare"


def skewness(x: np.ndarray) -> float:
    """Fisher-Pearson g1 (population); appendix-only diagnostic."""
    x = x.astype(np.float64)
    n = x.size
    if n < 3:
        return float("nan")
    m = x.mean()
    s = x.std(ddof=0)
    if s == 0:
        return 0.0
    return float(((x - m) ** 3).mean() / s**3)


def find_pool_file(pool_dir: Path, class_key: str) -> Optional[Path]:
    """Map 'dota/harbor' -> data/image_pools/dota__harbor__*.txt (newest)."""
    stem = class_key.replace("/", "__")
    hits = sorted(pool_dir.glob(f"{stem}__*.txt"))
    return hits[-1] if hits else None


def stats_for_class(
    index: Any,
    class_key: str,
    spec: Dict[str, Any],
    *,
    raw_root: str,
    pool_file: Optional[Path],
) -> Dict[str, Any]:
    base_ds = index.load_class(
        class_key, split=None, load_images=False, on_missing_split="warn"
    )
    ds = CountingClassTiledDataset(
        base_ds,
        raw_root=raw_root,
        tile_size=int(spec["tile_size"]),
        tile_threshold=int(spec["tile_threshold"]),
        image_pool_list=str(pool_file) if pool_file else None,
        pool_record_dir=None,
        verbose=False,
    )

    counts = ds.gt_counts.astype(np.float64)
    n_units = counts.size
    mu = float(counts.mean())
    pct0 = float((counts == 0).mean())
    cv = float(counts.std(ddof=1) / mu) if mu > 0 and n_units > 1 else 0.0

    # obj% = median bbox area / median unit area * 100, scale-invariant, in
    # original pixels.  Unit area = pw*ph (full tiles -> tile_size^2; untiled
    # -> original image area).  Box areas read from the per-image cache the
    # dataset already populated during tiling.
    unit_areas = np.array(
        [float(e.pw) * float(e.ph) for e in ds._tile_index], dtype=np.float64
    )
    # Gather box areas only for images actually in the pool.
    box_areas: List[float] = []
    seen = {e.img_idx for e in ds._tile_index}
    for img_idx in seen:
        boxes = ds._get_boxes(img_idx)  # (N,4) xyxy original px
        if boxes.numel():
            w = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
            h = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
            box_areas.extend((w * h).tolist())
    med_box = float(np.median(box_areas)) if box_areas else float("nan")
    med_unit = float(np.median(unit_areas)) if unit_areas.size else float("nan")
    objpct = 100.0 * med_box / med_unit if med_unit and med_unit > 0 else float("nan")

    n_tiled = int(sum(1 for e in ds._tile_index if e.is_tiled))
    return {
        "class_key": class_key,
        "tile_size": int(spec["tile_size"]),
        "tile_threshold": int(spec["tile_threshold"]),
        "untiled": n_tiled == 0,
        "n_images": ds.num_images,
        "n_units": n_units,
        "F_total": int(ds.F_true),
        "mu": round(mu, 4),
        "pct0": round(pct0, 4),
        "cv": round(cv, 4),
        "skew": round(skewness(counts), 4),
        "objpct": round(objpct, 4),
        "count_max": int(counts.max()) if n_units else 0,
        "level": assign_level(pct0),
    }


def main() -> None:
    args = parse_args()
    specs = json.loads(Path(args.tile_json).read_text())
    index_mod = __import__("counting_dataset", fromlist=["CountingDatasetIndex"])
    index = index_mod.CountingDatasetIndex(root=Path(args.index_root))
    pool_dir = Path(args.pool_dir)

    rows: List[Dict[str, Any]] = []
    for class_key, spec in specs.items():
        pool_file = None if args.whole_class else find_pool_file(pool_dir, class_key)
        tag = pool_file.name if pool_file else "WHOLE-CLASS"
        print(f"[regime] {class_key:32s} ts={spec['tile_size']:>4} "
              f"tt={spec['tile_threshold']:>4}  pool={tag}", flush=True)
        rows.append(
            stats_for_class(
                index, class_key, spec, raw_root=args.raw_root, pool_file=pool_file
            )
        )

    rows.sort(key=lambda r: (_LEVEL_ORDER[r["level"]], r["pct0"]))

    # CSV
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["level", "class_key", "tile_size", "tile_threshold", "untiled",
            "n_images", "n_units", "F_total", "mu", "pct0", "cv", "skew",
            "objpct", "count_max"]
    with out.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    # Pretty table
    print("\n" + "=" * 110)
    print("COUNTING REGIMES  (per-class tiling, budgeted pools)")
    print("=" * 110)
    hdr = (f"{'level':<10}{'class':<28}{'ts':>5}{'unt':>5}{'units':>8}"
           f"{'F_total':>9}{'mu':>9}{'%0':>7}{'cv':>7}{'obj%':>7}{'cmax':>7}")
    last = None
    for r in rows:
        if r["level"] != last:
            print("-" * 110)
            last = r["level"]
            print(hdr)
        print(f"{r['level']:<10}{r['class_key']:<28}{r['tile_size']:>5}"
              f"{'Y' if r['untiled'] else 'n':>5}{r['n_units']:>8}{r['F_total']:>9}"
              f"{r['mu']:>9.2f}{r['pct0']:>7.2f}{r['cv']:>7.2f}"
              f"{r['objpct']:>7.2f}{r['count_max']:>7}")
    print("=" * 110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
