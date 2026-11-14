"""Freeze the benchmark class selection into ``outputs/selected_classes.json``."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

# Selection: level -> ordered class keys.  Bold/3rd-per-level picks documented
# Swap a key here and re-run to update the frozen selection.
SELECTION: Dict[str, List[str]] = {
    "full": [
        "malaria/red_blood_cell",   # dense anchor (mu=49, largest F)
        "malaria/trophozoite",      # canonical microscopy, moderate density
        "malaria/difficult",        # 3rd: highest-F of the mu~=1 malarias
    ],
    "partial": [
        "birds/bird",               # different domain, large F, low end of level
        "dota/tennis_court",        # 3rd: fills the %0 gap, untiled, structured
        "dota/harbor",              # best F in the transition level
    ],
    "sparse": [
        "dota/ship",                # high F + high CV, tiny objects
        "dota/small_vehicle",       # 3rd: tiniest raw objects / smallest tile
        "dota/plane",               # larger objects, COCO-recognizable
    ],
    "rare": [
        "kenyan_wildlife/giraffe",  # wildlife domain, highest-F rare class
        "dota/helicopter",          # DOTA rare, high CV
        "dota/storage_tank",        # 3rd: extreme-dispersion showcase (CV=14.2)
    ],
}

LEVEL_ORDER = ["full", "partial", "sparse", "rare"]

LEVEL_DEFINITIONS = {
    "full":    {"frac_empty": "< 0.05",       "meaning": "fully occupied; empty-filter cannot help"},
    "partial": {"frac_empty": "0.05 - 0.60",  "meaning": "partly occupied; transition level"},
    "sparse":  {"frac_empty": "0.60 - 0.90",  "meaning": "mostly empty"},
    "rare":    {"frac_empty": ">= 0.90",      "meaning": "objects are rare events; empty-filter dominates"},
}

# Convenience modality labels (the 'dataset' field is the authoritative key).
MODALITY = {
    "malaria": "microscopy",
    "dota": "satellite/aerial",
    "kenyan_wildlife": "aerial (wildlife)",
    "birds": "ground-level (wildlife)",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regimes", default="outputs/regimes_v2.csv")
    ap.add_argument("--tile-json", default="outputs/suggested_tile_sizes.json")
    ap.add_argument("--pool-dir", default="data/image_pools")
    ap.add_argument("--out", default="outputs/selected_classes.json")
    ap.add_argument("--tile-budget", type=int, default=5000)
    ap.add_argument("--pool-seed", type=int, default=0)
    return ap.parse_args()


def load_regimes(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open() as f:
        return {row["class_key"]: row for row in csv.DictReader(f)}


def find_pool_file(pool_dir: Path, class_key: str) -> Path:
    stem = class_key.replace("/", "__")
    hits = sorted(pool_dir.glob(f"{stem}__*.txt"))
    if not hits:
        raise FileNotFoundError(f"no pool list for {class_key} in {pool_dir}")
    return hits[-1]


def main() -> None:
    args = parse_args()
    regimes = load_regimes(Path(args.regimes))
    tiles = json.loads(Path(args.tile_json).read_text())
    pool_dir = Path(args.pool_dir)

    classes: List[Dict] = []
    for level in LEVEL_ORDER:
        rows = []
        for class_key in SELECTION[level]:
            if class_key not in regimes:
                raise KeyError(f"{class_key} missing from {args.regimes}")
            if class_key not in tiles:
                raise KeyError(f"{class_key} missing from {args.tile_json}")
            r = regimes[class_key]
            spec = tiles[class_key]
            dataset, name = class_key.split("/", 1)
            pool = find_pool_file(pool_dir, class_key)
            rows.append({
                "class_key": class_key,
                "dataset": dataset,
                "class": name,
                "level": level,
                "modality": MODALITY.get(dataset, dataset),
                "tile_size": int(spec["tile_size"]),
                "tile_threshold": int(spec["tile_threshold"]),
                "pool_list": str(pool).replace("\\", "/"),
                "n_units": int(r["n_units"]),
                "F_total": int(r["F_total"]),
                "frac_empty": float(r["pct0"]),
                "count_mean": float(r["mu"]),
                "cv": float(r["cv"]),
            })
        # order within level by emptiness, matching the regime table
        rows.sort(key=lambda d: d["frac_empty"])
        classes.extend(rows)

    doc = {
        "description": "The 12 tasks of the CLC-12 collection-level counting "
                       "benchmark.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_rationale": "3 classes per occupancy level "
                               "(full/partial/sparse/rare), gated on pool "
                               "F_total, spanning visual domains; tile budget "
                               "fixed at 5000 units.",
        "source": {
            "regimes_csv": args.regimes,
            "tile_sizes_json": args.tile_json,
            "pool_dir": args.pool_dir,
            "tile_budget_units": args.tile_budget,
            "pool_seed": args.pool_seed,
        },
        "level_definitions": LEVEL_DEFINITIONS,
        "n_classes": len(classes),
        "classes": classes,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")

    print(f"wrote {out}  ({len(classes)} classes)")
    for level in LEVEL_ORDER:
        keys = [c["class_key"] for c in classes if c["level"] == level]
        print(f"  {level:<8} {', '.join(keys)}")


if __name__ == "__main__":
    main()
