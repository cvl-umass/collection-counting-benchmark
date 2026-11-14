# Tiled dataset

How the benchmark turns a class of images into the flat pool of estimation units that the sequential labeling loop draws from.

## `CountingClassTiledDataset` (`data/tiled_dataset.py`)

Wraps a single-class `CountingClassDataset` and exposes a flat pool of fixed-size **estimation units** that the benchmark can score one at a time.

### Unit pool

Each unit is either:

- a **whole image**, downscaled to fit `max_whole_image_side` (longer side), when the image's longer side is `<= tile_threshold`, or
- a `tile_size x tile_size` **crop** of a larger image (non-overlapping grid; right/bottom tiles may be shorter at the image edge).

`tile_threshold` defaults to `max_whole_image_side`, i.e. tile anything that would otherwise need to be downscaled.

The pool is built once at construction.
Tile indices `0..len-1` are stable (ordered by image, then row, then column), so they can key an experiment-wide results cache.
`manifest()` returns one row per unit linking that index back to its source image and crop box.

### Ground-truth counts: center-point vs. overlap

Two different rules are used on purpose:

- **GT count (`gt_count`, `F_true`):** a box is assigned to the tile that contains its **center**. This partitions every image's boxes across its tiles with no double-counting, so per-tile counts sum to the image count and `sum(gt_count) == F_true`.
- **Training boxes (`target["boxes"]`):** every box that **overlaps** the tile is kept (clipped to the tile). The detector sees objects that straddle tile boundaries, but they don't inflate the count.

Whole-image units with no `aux_key` read their count directly from base metadata and skip the annotation parse.

### Image cap

To bound runtime, the source-image pool can be capped before tiling:

- `pool_size=N` randomly subsamples `N` images (seeded by `pool_sample_seed`, reproducible across builds).
- `image_pool_list=<path | list>` keeps only images whose basename appears in the list (text file with `#` comments, or a Python sequence).
- The two are mutually exclusive.

Capping drops whole images, so `F_true` is just the total over the retained pool.

A `pool_size` build writes the selected image names to `data/image_pools/<dataset>__<class>__pool<N>_seed<S>.txt` (`pool_record_dir`; set `None` to disable). The file has a provenance header (`#` comments: dataset, class, requested pool size, seed, selected/total counts) followed by one basename per line, and is itself a valid `image_pool_list`, so the exact pool is reproducible from the file alone without re-running the sampler. Content is deterministic per (class, pool size, seed), so re-builds don't churn the file. Explicit `image_pool_list` builds write no record (the list is already its own record).

The selection itself lives in `data/image_pool.py` (`select_image_pool`), **shared with `scripts/build_image_pools.py`**: capping defines which images exist in the experiment at all, so the tile-size/pool prep applies the identical selection (same spec, same seed) before sampling boxes or reading image dimensions. The prep script takes the experiment's cap values (`--pool-size` / `--pool-list` / `--pool-seed`), so both sides see the same pool.

### Caching and the grouped sampler

Tiles of the same parent image share one decode and one annotation parse:

- `_img_cache` &mdash; LRU of decoded PIL images, sized by `img_cache_size` (default 4; set 0 to disable).
- `_boxes_cache` &mdash; parsed original-coordinate boxes per base image, populated lazily on first access.

`TileGroupedSampler` yields all tiles of one source image consecutively, so with a non-zero `img_cache_size` each parent image hits disk once per epoch.
A plain shuffled `DataLoader` would interleave tiles across images and miss the cache on every call.

### Item format

`ds[tile_idx]` returns `(image_tensor, target, unit_meta)`:

- `image_tensor`: `CHW` float tensor via `TF.to_tensor`.
- `target`: torchvision detection dict (`boxes`, `labels` all `1`, `image_id`, `area`, `iscrowd`).
- `unit_meta`: `tile_index`, `img_idx`, `image_id`, `count_true`, `is_tiled`, `tile_coords=(x0, y0, pw, ph)`.

`collate_detection` serves as the loader's `collate_fn`; it keeps the per-sample lists intact instead of stacking.

### `aux_key`

When set, annotations are read from `tgt["aux"][aux_key]` instead of `tgt["instances"]` (e.g. DOTA OBB boxes exposed under an aux key).
In that mode, whole-image units also load annotations rather than trusting `meta["count"]`.
