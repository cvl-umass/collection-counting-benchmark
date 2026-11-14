"""Per-class pre-loop tiling prep: suggest a tile size and freeze the image pool.

Runs before the main loop and so reads no unit labels: the tile size comes from
a small sample of object box sizes, and the pool from image dimensions alone.
Writes ``outputs/suggested_tile_sizes.json`` and ``data/image_pools/*.txt``.
The sizing rule is derived in the paper's tiling appendix.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Make the repo root importable regardless of the current directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.image_pool import select_image_pool  # noqa: E402  (path set up above)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # On a cluster, pass --index-root <DATA_ROOT>/counting/data; the default targets
    # the local sample checkout.
    ap.add_argument("--index-root", default="../counting-datasets/counting/data")
    ap.add_argument(
        "--class-key",
        default=None,
        help="Class to process, e.g. 'dota/small_vehicle'. If omitted, every "
        "policy-allowed class in the index is processed in one run.",
    )

    # --- how to pick the image pool (pick at most one; none = whole class) ---
    pool = ap.add_mutually_exclusive_group()
    pool.add_argument(
        "--tile-budget",
        type=int,
        default=None,
        help="BUILD a pool to this many estimation units (tiles, plus 1 per "
        "image kept whole) and freeze it to a pool-list file -- the normal "
        "pre-loop path; try 5000. Mutually exclusive with --pool-size / "
        "--pool-list; omit all three to size the whole class.",
    )
    pool.add_argument(
        "--pool-size",
        type=int,
        default=None,
        help="SIZE a given pool of this many randomly chosen images (seeded by "
        "--pool-seed) -- pass the experiment's pool_size. No pool-list file is "
        "written (the dataset reproduces the subsample from N + seed).",
    )
    pool.add_argument(
        "--pool-list",
        default=None,
        help="SIZE a given pool listed in this file (one image basename per "
        "line, '#' comments ok) -- the experiment's image_pool_list, or a pool "
        "frozen by an earlier --tile-budget run (verification).",
    )

    ap.add_argument(
        "--pool-seed",
        type=int,
        default=0,
        help="Seed for the --pool-size subsample (the experiment's pool_sample_seed).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the image permutation and the box sample. In budget mode "
        "this also seeds the pool selection; the frozen pool-list file is the "
        "reproducible artifact, this just makes the build deterministic.",
    )

    ap.add_argument(
        "--n-images",
        type=int,
        default=10,
        help="Min images to scan for the tile-size box sample (a floor, not a "
        "cap: scanning continues past it until --n-objects is met).",
    )
    ap.add_argument(
        "--n-objects",
        type=int,
        default=30,
        help="Objects to reveal for the tile-size box sample.",
    )

    # Tile-size formula + trust band (see module docstring).
    ap.add_argument(
        "--target-object-px",
        type=float,
        default=80.0,
        help="Apparent object size (longer side, px) the detector should see "
        "AFTER it resizes the tile. 80 sits upper-'medium' on COCO's bands, "
        "erring large so a typical class's small quartile clears the ~32px cliff.",
    )
    ap.add_argument(
        "--detector-input-px",
        type=float,
        default=800.0,
        help="Resolution the detector resizes a tile to (R). torchvision Faster "
        "R-CNN default min_size=800.",
    )
    ap.add_argument(
        "--round-to",
        type=int,
        default=16,
        help="Round the suggested tile size to this multiple (detector-friendly).",
    )
    ap.add_argument(
        "--tile-threshold-factor",
        type=float,
        default=1.5,
        help="Per-image tile boundary: tile an image whose longer side exceeds "
        "this many tile_sizes, else keep it whole. 1.5 keeps near-tile images "
        "whole (avoiding sliver grids) while still tiling larger outliers so "
        "their objects are not downscaled away.",
    )
    ap.add_argument(
        "--band-lo",
        type=float,
        default=32.0,
        help="Lower edge of the detector-friendly apparent band (small-object cliff).",
    )
    ap.add_argument(
        "--band-hi",
        type=float,
        default=150.0,
        help="Upper edge of the detector-friendly apparent band (advisory above it).",
    )

    ap.add_argument(
        "--pool-dir",
        default="data/image_pools",
        help="Directory for the frozen pool-list files (budget mode only).",
    )
    ap.add_argument(
        "--out",
        default="outputs/suggested_tile_sizes.json",
        help="JSON store to upsert each class's recommendation into.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Budget mode: rebuild even if this (class, budget, seed) pool-list "
        "file already exists. Ignored in the size-a-given-pool modes.",
    )
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Index / dataset
# ---------------------------------------------------------------------------


def build_index(args: argparse.Namespace):
    """Open the counting dataset index."""
    from counting_dataset import CountingDatasetIndex

    return CountingDatasetIndex(root=Path(args.index_root))


def load_class_ds(index: Any, class_key: str):
    """Load one class's base dataset (no pixels: annotations only)."""
    return index.load_class(
        class_key,
        split=None,
        load_images=False,
        on_missing_split="warn",
    )


# ---------------------------------------------------------------------------
# Tile-size sampling (shared by both modes)
# ---------------------------------------------------------------------------


def object_longer_sides(tgt: Dict[str, Any]) -> List[float]:
    """Return the longer side (px) of every valid instance box in one image.

    Reads the instance-role boxes the tiled dataset uses.  Geometry ``w``/``h``
    are in original-image pixels, so no resize/scale is involved.
    """
    sides: List[float] = []
    for ann in tgt.get("instances") or []:
        if ann.get("role", "instance") != "instance":
            continue
        geom = ann.get("geometry") or {}
        if not {"w", "h"}.issubset(geom):
            continue
        w, h = float(geom["w"]), float(geom["h"])
        if w > 1.0 and h > 1.0:
            sides.append(max(w, h))
    return sides


def collect_sample(
    base_ds: Any,
    meta: List[Dict[str, Any]],
    order: List[int],
    *,
    n_images: int,
    n_objects: int,
    seed: int,
) -> Tuple[List[float], List[str], int]:
    """Walk image indices in ``order``, revealing only a few boxes per image.
    
    Reads at most ``ceil(n_objects / n_images)`` randomly chosen boxes per image,
    never the whole image, so no image's full object count is learned.  Stops once
    both ``n_images`` and ``n_objects`` are met; if ``order`` is exhausted first, a
    top-up pass reveals more boxes from the already-visited images, round-robin.
    
    Returns ``(sides, used_names, n_visited)``, where ``n_visited`` counts the
    leading ``order`` entries actually read.  ``sides`` is empty if no usable box
    is found.
    """
    rng = np.random.default_rng(seed)
    per_image = max(1, -(-n_objects // max(1, n_images)))  # ceil-div

    sides: List[float] = []
    used_names: List[str] = []
    leftovers: List[List[float]] = []  # unrevealed boxes per visited image
    n_visited = 0
    for img_idx in order:
        n_visited += 1
        _, tgt = base_ds[img_idx]
        img_sides = object_longer_sides(tgt)
        if not img_sides:
            continue
        # Reveal only a few boxes from this image, chosen at random.
        take = min(per_image, len(img_sides))
        picks = rng.choice(len(img_sides), size=take, replace=False)
        chosen = {int(p) for p in picks}
        sides.extend(float(img_sides[p]) for p in picks)
        leftovers.append([float(s) for j, s in enumerate(img_sides)
                          if j not in chosen])
        used_names.append(Path(meta[img_idx]["path"]).name)
        if len(used_names) >= n_images and len(sides) >= n_objects:
            break

    # Top-up pass: only reached when ``order`` was exhausted short of
    # n_objects.  Reveal one more random box per image per round so no single
    # image dominates; still label-light (no image's full count is read) and
    # still confined to already-visited images.
    while len(sides) < n_objects and any(leftovers):
        for leftover in leftovers:
            if len(sides) >= n_objects:
                break
            if leftover:
                sides.append(leftover.pop(int(rng.integers(len(leftover)))))

    # The scan adds a whole image's boxes at a time, so it can overshoot
    # n_objects by up to per_image-1; trim to exactly the requested count.
    # Picks are already random, so the tail trim does not bias the median.
    sides = sides[:n_objects]
    return sides, used_names, n_visited


def tile_size_for_median(
    median: float, detector_input_px: float, target_object_px: float, round_to: int
) -> Tuple[int, float]:
    """The tile-size formula.

    ``raw = (R / target) * median``; the returned tile is ``raw`` snapped to a
    multiple of ``round_to`` (>= one step).  Both modes call this, so a pool
    priced under a budget uses the identical tile a size-a-given-pool re-run
    would report.
    """
    raw = (detector_input_px / target_object_px) * median
    step = max(1, round_to)
    return int(max(step, round(raw / step) * step)), raw


def tile_threshold_for(tile_size: int, factor: float) -> int:
    """Per-image tile/keep-whole boundary: tile an image whose longer side
    exceeds ``factor * tile_size``.  Decided image by image (never per class),
    so a class's large outliers tile even when its typical image stays whole.
    """
    return int(round(factor * tile_size))


def build_recommendation(
    sides: List[float],
    used_names: List[str],
    pool_img_long_sides: List[float],
    n_pool_images: int,
    args: argparse.Namespace,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Turn the sampled object sizes into a rounded tile-size recommendation.

    ``pool_img_long_sides`` is a CENSUS, not a sample: the longer side of every
    image in the (capped or budgeted) pool, read from free metadata.  Only the
    object sizes are sampled.  ``extra`` carries mode-specific provenance (the
    pool cap in size-a-given-pool mode, the tile budget in budget mode).
    """
    arr = np.asarray(sides, dtype=np.float64)
    median = float(np.median(arr))
    p25, p75 = (float(v) for v in np.percentile(arr, [25, 75]))
    cv = float(arr.std() / arr.mean()) if arr.mean() > 0 else 0.0

    tile_size, raw = tile_size_for_median(
        median, args.detector_input_px, args.target_object_px, args.round_to
    )

    # Band-check trust flag: at the *actual* (rounded) tile, an object of s px
    # appears at s * R / tile_size; the sampled middle 50% should land inside
    # the detector-friendly band.  Failure means no single tile size serves the
    # class's size spread -- more sampling won't fix it.
    apparent_p25 = p25 * args.detector_input_px / tile_size
    apparent_p75 = p75 * args.detector_input_px / tile_size
    band_ok = apparent_p25 >= args.band_lo and apparent_p75 <= args.band_hi

    # Per-image tile boundary (decided by the dataset, image by image): an image
    # is tiled iff its longer side exceeds tile_threshold (= factor * tile_size);
    # below that it stays whole, since tiling would only yield one tile plus thin
    # slivers.  The pool's median image side is recorded as a scale diagnostic.
    tile_threshold = tile_threshold_for(tile_size, args.tile_threshold_factor)
    median_img_long = (
        float(np.median(pool_img_long_sides)) if pool_img_long_sides else 0.0
    )

    rec = {
        "tile_size": tile_size,
        "tile_threshold": tile_threshold,
        "raw_tile_size": round(raw, 1),
        "median_longer_side_px": round(median, 1),
        "p25_longer_side_px": round(p25, 1),
        "p75_longer_side_px": round(p75, 1),
        "size_cv": round(cv, 3),
        "apparent_iqr_px": [round(apparent_p25, 1), round(apparent_p75, 1)],
        "band_px": [args.band_lo, args.band_hi],
        "band_ok": bool(band_ok),
        "median_image_longer_side_px": round(median_img_long, 1),
        "n_pool_images": n_pool_images,
        "n_images_used": len(used_names),
        "n_objects_used": len(sides),
        "image_names": used_names,
        "target_object_px": args.target_object_px,
        "detector_input_px": args.detector_input_px,
        "round_to": args.round_to,
        "tile_threshold_factor": args.tile_threshold_factor,
        "seed": args.seed,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        rec.update(extra)
    return rec


def upsert_store(out_path: Path, class_key: str, rec: Dict[str, Any]) -> None:
    """Append/replace this class's entry in the JSON store, keeping it sorted."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    store: Dict[str, Any] = {}
    if out_path.exists():
        store = json.loads(out_path.read_text(encoding="utf-8"))
    store[class_key] = rec
    out_path.write_text(
        json.dumps(dict(sorted(store.items())), indent=2) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Budget-mode machinery (only --tile-budget reaches this)
# ---------------------------------------------------------------------------


def units_for_image(w: int, h: int, tile_size: int, tile_threshold: int) -> int:
    """Estimation units one source image contributes (per image, matches the dataset).

    Mirrors CountingClassTiledDataset: an image whose longer side does not
    exceed ``tile_threshold`` is one whole-image unit; a larger image splits
    into a ``ceil(W/tile_size) x ceil(H/tile_size)`` grid.
    """
    if max(w, h) <= tile_threshold:
        return 1
    return math.ceil(w / tile_size) * math.ceil(h / tile_size)


def build_pool(
    perm: List[int],
    meta: List[Dict[str, Any]],
    tile_size: int,
    tile_threshold: int,
    budget: int,
    min_prefix: int,
) -> Tuple[List[int], int]:
    """Admit images down ``perm`` until the unit budget is met.

    Admit-THEN-price: each image is committed before its dimensions are read,
    so a priced image is always a pool member (no leak, no un-admitting).  Each
    image is priced individually (whole or a tile_size grid, by its own longer
    side), so the walk needs no class-level tiling verdict and nothing to
    revise.  The first ``min_prefix`` images (the tile-size sample) are always
    included even if the budget is already met.  Returns the admitted indices
    and their total unit count.
    """
    cum = 0
    positions: List[int] = []
    for pos, img_idx in enumerate(perm):
        cum += units_for_image(
            int(meta[img_idx]["width"]), int(meta[img_idx]["height"]),
            tile_size, tile_threshold,
        )
        positions.append(img_idx)
        if pos + 1 >= min_prefix and cum >= budget:
            break
    return positions, cum


def pool_list_path(class_key: str, args: argparse.Namespace) -> Path:
    """The pool-list filename for this (class, budget, seed) -- the cache key."""
    dataset, _, cls = class_key.partition("/")
    cls = cls or "unknown"
    return Path(args.pool_dir) / (
        f"{dataset}__{cls}__budget{args.tile_budget}_seed{args.seed}.txt"
    )


def write_pool_list(
    positions: List[int],
    meta: List[Dict[str, Any]],
    class_key: str,
    args: argparse.Namespace,
    tile_size: int,
    tile_threshold: int,
    n_units: int,
) -> Path:
    """Freeze the pool to a text file that doubles as an image_pool_list."""
    dataset, _, cls = class_key.partition("/")
    out = pool_list_path(class_key, args)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = [
        f"# dataset: {dataset}",
        f"# class: {cls or 'unknown'}",
        f"# tile_budget (units): {args.tile_budget}",
        f"# seed: {args.seed}",
        f"# tile_size: {tile_size}",
        f"# tile_threshold: {tile_threshold}",
        f"# images selected: {len(positions)} of {len(meta)}",
        f"# units in pool: {n_units}",
        "# Pass as image_pool_list / --pool-list to reproduce this pool exactly.",
    ]
    names = [Path(meta[i]["path"]).name for i in positions]
    out.write_text("\n".join(header + names) + "\n", encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# The two modes
# ---------------------------------------------------------------------------


def size_given_pool(
    index: Any, class_key: str, args: argparse.Namespace
) -> Optional[Dict[str, Any]]:
    """Size the tile for a GIVEN pool: --pool-list, --pool-size, or whole class.

    The pool is fixed up front (no budget walk), so the tile size is a plain
    one-pass estimate over a box sample of pool members.  Returns None if the
    pool yields no usable boxes.
    """
    base_ds = load_class_ds(index, class_key)
    meta = base_ds.image_metadata()

    # Apply the same selection (and seed) CountingClassTiledDataset will apply,
    # so the suggestion is derived from exactly the images the experiment tiles.
    pool = select_image_pool(
        meta,
        pool_size=args.pool_size,
        image_pool_list=args.pool_list,
        pool_sample_seed=args.pool_seed,
        verbose=True,
        log_prefix=f"[build_image_pools:{class_key}]",
    )

    # Walk the (frozen) pool in a seeded random order; the order only chooses
    # which boxes we see, never which images are pool members.
    rng = np.random.default_rng(args.seed)
    order = [pool[i] for i in rng.permutation(len(pool))]
    sides, used_names, _ = collect_sample(
        base_ds, meta, order,
        n_images=args.n_images, n_objects=args.n_objects, seed=args.seed,
    )
    if not sides:
        return None

    # Image dimensions are free metadata, so the median image side is an exact
    # census over the whole pool, not a sample.
    pool_img_long = [float(max(meta[i]["width"], meta[i]["height"])) for i in pool]
    extra = {
        "pool_size": args.pool_size,
        "pool_list": str(args.pool_list) if args.pool_list else None,
        "pool_seed": args.pool_seed,
    }
    return build_recommendation(sides, used_names, pool_img_long, len(pool), args, extra)


def build_budgeted_pool(
    index: Any, class_key: str, args: argparse.Namespace
) -> Optional[Dict[str, Any]]:
    """BUILD a budgeted pool + recommendation under the no-leak protocol.

    The only mode where the pool depends on the tile size, so it follows the
    seeded-permutation protocol in the module docstring.  Returns None if no
    boxes; also writes the frozen pool-list file and sets ``pool_list``.
    """
    base_ds = load_class_ds(index, class_key)
    meta = base_ds.image_metadata()
    rng = np.random.default_rng(args.seed)
    perm = [int(i) for i in rng.permutation(len(meta))]

    # Phase 1: tile size from the front of the permutation.  collect_sample
    # returns how many leading perm entries it visited -- those images are
    # committed pool members, so the box sample never leaks from outside.
    sides, used_names, k1 = collect_sample(
        base_ds, meta, perm,
        n_images=args.n_images, n_objects=args.n_objects, seed=args.seed,
    )
    if not sides:
        return None
    median = float(np.median(np.asarray(sides, dtype=np.float64)))
    tile_size, _ = tile_size_for_median(
        median, args.detector_input_px, args.target_object_px, args.round_to
    )
    tile_threshold = tile_threshold_for(tile_size, args.tile_threshold_factor)

    # Phase 2: grow the pool to the budget, pricing each admitted image by its
    # own dimensions (whole if <= tile_threshold, else a tile_size grid).  Per-
    # image pricing needs no class-level verdict, so cum is the final unit count
    # and there is nothing to reconcile after the walk.
    positions, n_units = build_pool(
        perm, meta, tile_size, tile_threshold, args.tile_budget, k1
    )

    pool_img_long = [float(max(meta[i]["width"], meta[i]["height"])) for i in positions]
    rec = build_recommendation(
        sides, used_names, pool_img_long, len(positions), args,
        extra={"tile_budget": args.tile_budget},
    )
    rec["n_tiles_total"] = n_units

    pool_file = write_pool_list(
        positions, meta, class_key, args, tile_size, tile_threshold, n_units
    )
    rec["pool_list"] = str(pool_file)
    return rec


# ---------------------------------------------------------------------------
# Reporting + main
# ---------------------------------------------------------------------------


def print_summary(class_key: str, rec: Dict[str, Any]) -> None:
    """Print the per-class recommendation block (adapts to the mode)."""
    mult = round(rec["detector_input_px"] / rec["target_object_px"], 2)
    print(f"class:    {class_key}")
    print(f"tile:     tile_size = {rec['tile_size']}  "
          f"(raw {rec['raw_tile_size']} = {mult} x median "
          f"{rec['median_longer_side_px']}px, rounded to {rec['round_to']})")
    print(f"tiling:   keep whole if longer side <= {rec['tile_threshold']}px "
          f"({rec['tile_threshold_factor']}x tile_size), else tile  "
          f"[pool median image {rec['median_image_longer_side_px']}px]")
    if "tile_budget" in rec:
        print(f"pool:     {rec['n_pool_images']} images -> {rec['n_tiles_total']} "
              f"units (budget {rec['tile_budget']})")
    else:
        if rec.get("pool_size") is not None:
            cap = f"capped by pool_size={rec['pool_size']}, seed={rec['pool_seed']}"
        elif rec.get("pool_list"):
            cap = f"capped by pool_list={rec['pool_list']}"
        else:
            cap = "uncapped (whole class)"
        print(f"pool:     {rec['n_pool_images']} images ({cap})")
    print(f"sampled:  {rec['n_objects_used']} objects from "
          f"{rec['n_images_used']} images")
    if not rec["band_ok"]:
        lo, hi = rec["band_px"]
        print(f"WARNING:  middle 50% of objects land at {rec['apparent_iqr_px']}px "
              f"apparent, outside the detector band [{lo:g}, {hi:g}] -- no single "
              "tile serves this size spread; expect weaker recall on the tails.")
    if "tile_budget" in rec:
        print(f"pool list -> {rec['pool_list']}")


def main() -> None:
    args = parse_args()
    index = build_index(args)
    budget_mode = args.tile_budget is not None

    if args.class_key is not None:
        targets = [args.class_key]
    else:
        classes = index.get_classes(apply_policy=True)
        targets = sorted(c["class_key"] for c in classes)
        if budget_mode:
            print(f"building pools for {len(targets)} class(es) at budget "
                  f"{args.tile_budget} from {args.index_root}\n")
        else:
            print(f"sizing {len(targets)} class(es) from {args.index_root}\n")

    out_path = Path(args.out)
    n_ok = 0
    for class_key in targets:
        # Budget mode caches on the frozen pool-list file; the size-a-given-pool
        # modes write no file, so they always (re)compute.
        if budget_mode and not args.force and pool_list_path(class_key, args).exists():
            print(f"[skip] {class_key}: pool list exists "
                  f"({pool_list_path(class_key, args)}); use --force to rebuild\n")
            continue
        try:
            rec = (build_budgeted_pool if budget_mode else size_given_pool)(
                index, class_key, args
            )
        except Exception as exc:  # one bad class shouldn't abort the sweep
            print(f"[skip] {class_key}: {exc}\n")
            continue
        if rec is None:
            print(f"[skip] {class_key}: no instance boxes found\n")
            continue
        upsert_store(out_path, class_key, rec)
        print_summary(class_key, rec)
        print()
        n_ok += 1

    tail = f"  (pool lists in {args.pool_dir}/)" if budget_mode else ""
    print(f"stored {n_ok}/{len(targets)} class(es) -> {out_path}{tail}")


if __name__ == "__main__":
    main()
