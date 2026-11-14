"""Read a cached per-unit feature matrix and align it to a tiled dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

TileKey = Tuple[Any, int, int]


def cache_dir_for(
    cache_root: Path, class_key: str, max_whole_image_side: int, tile_size: int
) -> Path:
    """Per-class cache directory: ``<root>/<dataset>/<class>_us<MWS>_ts<TS>``.

    The ``us`` token is the historical label for ``max_whole_image_side``; it is
    kept so caches built before the parameter rename still resolve.  This is the
    single source of truth for the layout -- the extractor and every reader call
    it, so the path convention never drifts.
    """
    dataset, class_name = class_key.split("/", 1)
    return cache_root / dataset / f"{class_name}_us{max_whole_image_side}_ts{tile_size}"


def tile_key(row: Dict[str, Any]) -> TileKey:
    """The pool-invariant identity of a unit, from one manifest row."""
    return (row["image_id"], int(row["x0"]), int(row["y0"]))


def load_cache(cache_dir: Path) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Load ``(features, manifest)`` from a cache dir; row i of each corresponds.

    Raises if the manifest is missing (a pre-manifest cache -- re-extract that
    class) or its length disagrees with the feature matrix.
    """
    cache_dir = Path(cache_dir)
    feats = np.load(cache_dir / "features.npy")
    man_path = cache_dir / "manifest.json"
    if not man_path.is_file():
        raise FileNotFoundError(
            f"{man_path} not found: this cache predates manifest.json. "
            "Re-run scripts/extract_dino_features.py for this class."
        )
    manifest = json.loads(man_path.read_text())
    if len(manifest) != feats.shape[0]:
        raise ValueError(
            f"manifest rows ({len(manifest)}) != feature rows "
            f"({feats.shape[0]}) in {cache_dir}"
        )
    return feats, manifest


def align_features(
    cache_dir: Path,
    target_manifest: Sequence[Dict[str, Any]],
    *,
    strict: bool = True,
) -> Tuple[np.ndarray, List[int]]:
    """Return features for ``target_manifest``'s units, in its row order.

    ``target_manifest`` is the consumer dataset's ``manifest()`` under any image
    pool.  Output row ``i`` is the cached feature for ``target_manifest[i]``,
    matched by ``(image_id, x0, y0)``.  Returns ``(features, missing)`` where
    ``missing`` lists target rows whose key was absent from the cache; with
    ``strict`` (default) a non-empty ``missing`` raises instead.

    Lookups are O(1): the key->row map is built once, so capping or reseeding
    the pool only changes which keys are requested, never the per-unit cost.
    Units the extractor skipped as sub-patch are present as zero rows here (they
    have a key), so they are *not* reported missing.
    """
    feats, src_manifest = load_cache(cache_dir)
    key_to_row: Dict[TileKey, int] = {
        tile_key(r): int(r["tile_index"]) for r in src_manifest
    }

    out = np.zeros((len(target_manifest), feats.shape[1]), dtype=feats.dtype)
    missing: List[int] = []
    for i, row in enumerate(target_manifest):
        src = key_to_row.get(tile_key(row))
        if src is None:
            missing.append(i)
        else:
            out[i] = feats[src]

    if strict and missing:
        raise KeyError(
            f"{len(missing)} of {len(target_manifest)} target units are not in "
            f"the cache {cache_dir} (first few target rows: {missing[:5]}). "
            "Was the cache built for the same class and tiling (tile_size, "
            "tile_threshold, max_whole_image_side)?"
        )
    return out, missing


def features_for_dataset(cache_dir: Path, ds: Any, *, strict: bool = True):
    """Convenience: align a cache to a live ``CountingClassTiledDataset``.

    ``out[s_t]`` is the cached feature for unit ``s_t`` of ``ds``, whatever image
    pool ``ds`` was built with.
    """
    return align_features(cache_dir, ds.manifest(), strict=strict)
