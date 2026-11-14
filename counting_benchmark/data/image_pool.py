"""Source-image pool selection, shared by tiling and tile-size suggestion."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

PoolListSpec = Union[str, Path, Sequence[str], None]


def resolve_include_names(image_pool_list: Union[str, Path, Sequence[str]]) -> set:
    """Normalize an include spec (file path or sequence) into a name set."""
    if isinstance(image_pool_list, (str, Path)):
        text = Path(image_pool_list).read_text(encoding="utf-8")
        lines = text.splitlines()
    else:
        lines = list(image_pool_list)
    names = set()
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            names.add(Path(line).name)
    return names


def select_image_pool(
    all_meta: List[Dict[str, Any]],
    *,
    pool_size: Optional[int] = None,
    image_pool_list: PoolListSpec = None,
    pool_sample_seed: int = 0,
    verbose: bool = True,
    log_prefix: str = "[image_pool]",
) -> List[int]:
    """Return the sorted base-dataset indices of the images in the pool.

    Args:
        all_meta: ``base_ds.image_metadata()`` rows, in base-dataset order.
        pool_size: If set, randomly subsample this many source images
            (seeded).  Mutually exclusive with ``image_pool_list``.
        image_pool_list: A path to a text file (one image filename per line,
            ``#`` comments allowed) or a sequence of image filenames; only
            these images are kept.  Matched by basename.
        pool_sample_seed: RNG seed for the ``pool_size`` subsample.
        verbose: Print a warning when include-list names are not found.
        log_prefix: Prefix for the warning line (callers pass their own tag).
    """
    if pool_size is not None and image_pool_list is not None:
        raise ValueError(
            "Pass at most one of pool_size / image_pool_list, not both."
        )
    if pool_size is not None and pool_size < 1:
        raise ValueError(f"pool_size must be >= 1, got {pool_size}")

    n = len(all_meta)

    if image_pool_list is not None:
        wanted = resolve_include_names(image_pool_list)
        picked = [
            i for i, m in enumerate(all_meta) if Path(m["path"]).name in wanted
        ]
        if not picked:
            raise ValueError(
                "image_pool_list matched no images in the base dataset."
            )
        found = {Path(all_meta[i]["path"]).name for i in picked}
        missing = wanted - found
        if missing and verbose:
            print(
                f"{log_prefix} WARNING: {len(missing)}/{len(wanted)} "
                f"include-list names were not found in the base dataset.",
                flush=True,
            )
        return sorted(picked)

    if pool_size is not None and pool_size < n:
        rng = np.random.default_rng(pool_sample_seed)
        picked = rng.choice(n, size=pool_size, replace=False)
        return sorted(int(i) for i in picked)

    return list(range(n))
