"""Tiled, class-centric dataset: the flat pool of estimation units the loop draws from.

Layout and count semantics are documented in ``docs/tiled_dataset.md``.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from data.image_pool import select_image_pool
from utils.geometry import clamp_xyxy, scale_boxes_xyxy, xywh_to_xyxy
from utils.image import load_image_rgb, resize_pil_keep_aspect
from utils.paths import resolve_image_path

_LOG = "[CountingClassTiledDataset]"

# Floor for a rendered tile's side. Remainder strips at an image's right/bottom
# edge can be as thin as 1px; below this the detector's aspect-preserving resize
# can collapse a dimension to 0. Padding (not re-tiling) keeps unit keys/counts.
_MIN_TILE_PX = 32


def collate_detection(
    batch: List[Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]],
) -> Tuple[List[torch.Tensor], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collate function for detection batches: keeps per-sample lists intact."""
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    metas = [b[2] for b in batch]
    return images, targets, metas


@dataclass(slots=True)
class _TileEntry:
    """Metadata for a single estimation unit (a tile or a full image)."""

    img_idx: int  # index into the base CountingClassDataset
    x0: int  # tile left edge, in original-image pixels
    y0: int  # tile top edge, in original-image pixels
    pw: int  # tile width  (< tile_size at the image's right edge)
    ph: int  # tile height (< tile_size at the image's bottom edge)
    is_tiled: bool  # True = crop of a large image; False = whole image
    gt_count: int  # ground-truth count for this unit


class CountingClassTiledDataset(torch.utils.data.Dataset):
    """PyTorch dataset that tiles a single-class counting dataset into units.

    Each item is ``(image_tensor, target_dict, unit_meta)`` where ``target_dict``
    is in torchvision detection format (single foreground class, label 1) and
    ``unit_meta`` carries the unit's identity and ground-truth count.

    The unit pool is fixed at construction time.  Tile indices ``0..len-1`` are
    stable (ordered by image, then row, then column) so they can key a results
    cache for the whole experiment.
    """

    def __init__(
        self,
        base_ds: Any,
        *,
        raw_root: Union[str, Path],
        max_whole_image_side: int = 1536,
        tile_size: int = 512,
        tile_threshold: Optional[int] = None,
        pool_size: Optional[int] = None,
        image_pool_list: Union[str, Path, Sequence[str], None] = None,
        pool_sample_seed: int = 0,
        pool_record_dir: Union[str, Path, None] = "data/image_pools",
        aux_key: Optional[str] = None,
        img_cache_size: int = 4,
        verbose: bool = True,
    ) -> None:
        """Args:
            base_ds: A `CountingClassDataset` loaded with ``load_images=False``.
            raw_root: Root directory used to resolve image paths from the index.
            max_whole_image_side: Downscale cap (longer side, px) for whole images --
                a memory guardrail, never an upscale.
            tile_size: Square tile side length, in original-image pixels.
            tile_threshold: Tile any image whose longer side exceeds this.  Defaults
                to ``max_whole_image_side``.
            pool_size: If set, randomly subsample this many source images before
                tiling.  Mutually exclusive with ``image_pool_list``.
            image_pool_list: Path to a text file (one image filename per line, ``#``
                comments allowed) or a sequence of filenames; only these images are
                used, matched by basename.  Mutually exclusive with ``pool_size``.
            pool_sample_seed: RNG seed for the ``pool_size`` subsample.
            pool_record_dir: Where to record the images a ``pool_size`` subsample
                selected.  The record is itself a valid ``image_pool_list``, so the
                pool is reproducible from the file alone.  None disables it; explicit
                ``image_pool_list`` builds write no record.
            aux_key: If set, read annotations from ``tgt["aux"][aux_key]`` instead of
                ``tgt["instances"]`` (e.g. DOTA OBB boxes).
            img_cache_size: Decoded parent images held in an LRU cache, so
                consecutive tiles of one image decode it once.  0 disables.
            verbose: Print a build summary.
        """
        if not hasattr(base_ds, "image_metadata"):
            raise TypeError(
                "base_ds must be a class-centric CountingClassDataset "
                "(no image_metadata() method found)."
            )
        self.base_ds = base_ds
        self.raw_root = Path(raw_root)
        self.max_whole_image_side = int(max_whole_image_side)
        self.tile_size = int(tile_size)
        self.tile_threshold = (
            int(tile_threshold)
            if tile_threshold is not None
            else self.max_whole_image_side
        )
        self.pool_size = pool_size
        self.pool_sample_seed = int(pool_sample_seed)
        self.aux_key = aux_key
        self.verbose = bool(verbose)

        # LRU cache of decoded parent images, keyed by resolved path string.
        self._img_cache_size = int(img_cache_size)
        self._img_cache: "OrderedDict[str, Image.Image]" = OrderedDict()

        # Parsed original-coordinate boxes, keyed by base_ds index.  Populated
        # during tile-index construction and lazily on first item access, so an
        # image's annotation JSON is parsed at most once.
        self._boxes_cache: Dict[int, torch.Tensor] = {}

        # Per-base-image metadata, keyed by base_ds index (selected images only).
        # The pool cap is shared logic (data/image_pool.py): the tile-size
        # suggester applies the identical selection so it never reasons about
        # images this dataset will drop.
        all_meta: List[Dict[str, Any]] = base_ds.image_metadata()
        selected = select_image_pool(
            all_meta,
            pool_size=pool_size,
            image_pool_list=image_pool_list,
            pool_sample_seed=self.pool_sample_seed,
            verbose=self.verbose,
            log_prefix=_LOG,
        )
        self._img_meta: Dict[int, Dict[str, Any]] = {i: all_meta[i] for i in selected}

        # Transparency record: persist which images a random cap selected, so
        # the exact pool can be shared/reproduced without sharing RNG details.
        if pool_size is not None and pool_record_dir is not None:
            self._record_pool(selected, len(all_meta), pool_record_dir)

        # Build the flat unit index.
        self._tile_index: List[_TileEntry] = self._build_tile_index(selected)

        # Group tile indices by source image (used by TileGroupedSampler).
        self.image_groups: Dict[int, List[int]] = {}
        for tile_idx, entry in enumerate(self._tile_index):
            self.image_groups.setdefault(entry.img_idx, []).append(tile_idx)

        self._gt_counts = np.array(
            [e.gt_count for e in self._tile_index], dtype=np.int64
        )
        self.F_true: int = int(self._gt_counts.sum())

        if self.verbose:
            n_tiled = sum(1 for e in self._tile_index if e.is_tiled)
            n_empty = int((self._gt_counts == 0).sum())
            print(
                f"{_LOG} {len(self._tile_index)} units from "
                f"{len(self._img_meta)} images "
                f"({n_tiled} tiles, {len(self._tile_index) - n_tiled} whole "
                f"images, {n_empty} empty).  F_true = {self.F_true}.",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Pool transparency record
    # ------------------------------------------------------------------

    def _record_pool(
        self, selected: List[int], n_total: int, record_dir: Union[str, Path]
    ) -> None:
        """Write the ``pool_size``-selected image names to a text record.

        Content is deterministic for a given (class, pool_size, seed), so the
        file only changes when the pool does.  The record doubles as an
        ``image_pool_list``: feeding it back reproduces this exact pool with
        no RNG involved.
        """
        class_key = str(getattr(self.base_ds, "class_key", "unknown/unknown"))
        dataset, _, cls = class_key.partition("/")
        cls = cls or "unknown"
        out = Path(record_dir) / (
            f"{dataset}__{cls}__pool{self.pool_size}"
            f"_seed{self.pool_sample_seed}.txt"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# dataset: {dataset}",
            f"# class: {cls}",
            f"# pool_size (requested): {self.pool_size}",
            f"# pool_sample_seed: {self.pool_sample_seed}",
            f"# images selected: {len(selected)} of {n_total}",
            "# Pass this file as image_pool_list / --pool-list to reproduce "
            "this pool exactly.",
        ]
        lines += [Path(self._img_meta[i]["path"]).name for i in selected]
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if self.verbose:
            print(f"{_LOG} pool record -> {out}", flush=True)

    # ------------------------------------------------------------------
    # Tile index construction
    # ------------------------------------------------------------------

    def _build_tile_index(self, selected: List[int]) -> List[_TileEntry]:
        """Build the flat unit index over the selected source images.

        Whole images smaller than ``tile_threshold`` skip the SQLite annotation
        query and read their count straight from image metadata -- unless
        ``aux_key`` is set, where the count must come from the aux annotations.
        Tiled images always load annotations to assign boxes to tiles.
        """
        entries: List[_TileEntry] = []
        for img_idx in selected:
            meta = self._img_meta[img_idx]
            w, h = int(meta["width"]), int(meta["height"])
            is_tiled = max(w, h) > self.tile_threshold

            if not is_tiled and self.aux_key is None:
                entries.append(
                    _TileEntry(img_idx, 0, 0, w, h, False, int(meta["count"]))
                )
                continue

            boxes = self._get_boxes(img_idx)
            if is_tiled:
                entries.extend(self._tile_one_image(img_idx, w, h, boxes))
            else:
                entries.append(
                    _TileEntry(img_idx, 0, 0, w, h, False, int(boxes.shape[0]))
                )
        return entries

    def _tile_one_image(
        self, img_idx: int, w: int, h: int, boxes: torch.Tensor
    ) -> List[_TileEntry]:
        """Split one large image into a non-overlapping grid of tile units."""
        ts = self.tile_size
        n_cols = math.ceil(w / ts)
        n_rows = math.ceil(h / ts)

        # Assign each box to the tile containing its center (center-point rule).
        if boxes.shape[0] > 0:
            cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
            cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
            col = (cx / ts).long().clamp(0, n_cols - 1)
            row = (cy / ts).long().clamp(0, n_rows - 1)
            flat = row * n_cols + col
            counts = torch.bincount(flat, minlength=n_rows * n_cols).view(
                n_rows, n_cols
            )
        else:
            counts = torch.zeros((n_rows, n_cols), dtype=torch.long)

        entries: List[_TileEntry] = []
        for r in range(n_rows):
            y0 = r * ts
            ph = min(ts, h - y0)
            for c in range(n_cols):
                x0 = c * ts
                pw = min(ts, w - x0)
                entries.append(
                    _TileEntry(img_idx, x0, y0, pw, ph, True, int(counts[r, c].item()))
                )
        return entries

    # ------------------------------------------------------------------
    # Annotation parsing
    # ------------------------------------------------------------------

    def _get_boxes(self, img_idx: int) -> torch.Tensor:
        """Return image *img_idx*'s annotation boxes in original-image xyxy.

        Parsed from the base dataset once per image, then cached; later calls
        are dict lookups.  The returned tensor is shared -- callers must treat
        it as read-only (``__getitem__`` always works on copies of it).
        """
        cached = self._boxes_cache.get(img_idx)
        if cached is not None:
            return cached
        meta = self._img_meta[img_idx]
        _, tgt = self.base_ds[img_idx]
        boxes = self._parse_boxes(tgt, int(meta["width"]), int(meta["height"]))
        self._boxes_cache[img_idx] = boxes
        return boxes

    def _parse_boxes(self, tgt: Dict[str, Any], img_w: int, img_h: int) -> torch.Tensor:
        """Return all valid annotation boxes as an (N, 4) xyxy tensor.

        Boxes are clamped to image bounds; degenerate boxes (either side <= 1px)
        are dropped.

        tgt: Target object returned by CountingClassDataset.
        """
        if self.aux_key is not None:
            anns = (tgt.get("aux") or {}).get(self.aux_key, [])
            skip_role = True
        else:
            anns = tgt.get("instances") or []
            skip_role = False

        parsed: List[List[float]] = []
        for ann in anns:
            if not skip_role and ann.get("role", "instance") != "instance":
                continue
            geom = ann.get("geometry") or {}
            if not {"x", "y", "w", "h"}.issubset(geom):
                continue
            box = clamp_xyxy(xywh_to_xyxy(geom), img_w, img_h)
            if (box[2] - box[0]) > 1.0 and (box[3] - box[1]) > 1.0:
                parsed.append(box)

        if parsed:
            return torch.tensor(parsed, dtype=torch.float32)
        return torch.zeros((0, 4), dtype=torch.float32)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._tile_index)

    @property
    def num_images(self) -> int:
        """Number of source images contributing units to the pool."""
        return len(self._img_meta)

    @property
    def gt_counts(self) -> np.ndarray:
        """Ground-truth count of every unit, as an (N,) int64 array."""
        return self._gt_counts

    def get_gt_count(self, tile_idx: int) -> int:
        """Return the ground-truth count for unit *tile_idx* (no disk I/O)."""
        return self._tile_index[tile_idx].gt_count

    def manifest(self) -> List[Dict[str, Any]]:
        """Return one descriptor row per unit, ordered by tile index.

        Pairs a unit's stable tile index with its source image and crop box, so
        a results cache can be interpreted without re-running the dataset.
        """
        rows: List[Dict[str, Any]] = []
        for tile_idx, e in enumerate(self._tile_index):
            meta = self._img_meta[e.img_idx]
            rows.append(
                {
                    "tile_index": tile_idx,
                    "img_idx": e.img_idx,
                    "image_id": meta["image_id"],
                    "image_path": meta["path"],
                    "x0": e.x0,
                    "y0": e.y0,
                    "pw": e.pw,
                    "ph": e.ph,
                    "is_tiled": e.is_tiled,
                    "gt_count": e.gt_count,
                }
            )
        return rows

    # ------------------------------------------------------------------
    # Item loading
    # ------------------------------------------------------------------

    def _load_image(self, img_or_path: Any) -> Image.Image:
        """Load a parent image as RGB, serving from the LRU cache when possible."""
        if isinstance(img_or_path, Image.Image):
            return img_or_path.convert("RGB")

        key = str(resolve_image_path(str(img_or_path), self.raw_root))
        if self._img_cache_size > 0 and key in self._img_cache:
            self._img_cache.move_to_end(key)
            return self._img_cache[key]

        img = load_image_rgb(Path(key))
        if self._img_cache_size > 0:
            self._img_cache[key] = img
            self._img_cache.move_to_end(key)
            if len(self._img_cache) > self._img_cache_size:
                self._img_cache.popitem(last=False)
        return img

    def __getitem__(
        self, tile_idx: int
    ) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
        entry = self._tile_index[tile_idx]
        meta = self._img_meta[entry.img_idx]

        img = self._load_image(meta["path"])
        boxes = self._get_boxes(entry.img_idx)

        if entry.is_tiled:
            img, boxes = self._crop_tile(img, boxes, entry)
        else:
            img, boxes = self._resize_full(img, boxes)

        n = boxes.shape[0]
        wh = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
            boxes[:, 3] - boxes[:, 1]
        ).clamp(min=0)
        target: Dict[str, Any] = {
            "boxes": boxes,
            "labels": torch.ones((n,), dtype=torch.int64),
            "image_id": torch.tensor([tile_idx], dtype=torch.int64),
            "area": wh,
            "iscrowd": torch.zeros((n,), dtype=torch.int64),
        }
        unit_meta: Dict[str, Any] = {
            "tile_index": tile_idx,
            "img_idx": entry.img_idx,
            "image_id": meta["image_id"],
            "count_true": entry.gt_count,
            "is_tiled": entry.is_tiled,
            "tile_coords": (entry.x0, entry.y0, entry.pw, entry.ph),
        }
        return TF.to_tensor(img), target, unit_meta

    def _crop_tile(
        self, img: Image.Image, boxes: torch.Tensor, entry: _TileEntry
    ) -> Tuple[Image.Image, torch.Tensor]:
        """Crop the tile region; keep every overlapping box (training signal).

        Training boxes use overlap, not the center-point rule, so the model
        also learns from objects that straddle a tile boundary.
        """
        x0, y0, pw, ph = entry.x0, entry.y0, entry.pw, entry.ph
        crop = img.crop((x0, y0, x0 + pw, y0 + ph))
        # Fixed-grid tiling leaves a thin remainder strip whenever an image side
        # isn't a multiple of tile_size; a 1px-wide strip (e.g. dota/harbor with
        # width % ts == 1) collapses to width 0 under the detector's
        # aspect-preserving resize (1333/long_side) and crashes interpolation.
        # Zero-pad the short side up to a small floor so the resize can't
        # degenerate. Boxes are tile-local (top-left origin) and counts come
        # from the index, so the unit's (image_id, x0, y0) key and GT count are
        # untouched -- the DINOv3 cache and F_true stay aligned.
        if pw < _MIN_TILE_PX or ph < _MIN_TILE_PX:
            padded = Image.new(crop.mode, (max(pw, _MIN_TILE_PX), max(ph, _MIN_TILE_PX)))
            padded.paste(crop, (0, 0))
            crop = padded
        if boxes.shape[0] == 0:
            return crop, torch.zeros((0, 4), dtype=torch.float32)

        x2, y2 = float(x0 + pw), float(y0 + ph)
        overlap = (
            (boxes[:, 2] > float(x0))
            & (boxes[:, 0] < x2)
            & (boxes[:, 3] > float(y0))
            & (boxes[:, 1] < y2)
        )
        b = boxes[overlap].clone()
        if b.shape[0] > 0:
            b[:, 0] -= x0
            b[:, 2] -= x0
            b[:, 1] -= y0
            b[:, 3] -= y0
            b[:, 0].clamp_(0.0, float(pw))
            b[:, 2].clamp_(0.0, float(pw))
            b[:, 1].clamp_(0.0, float(ph))
            b[:, 3].clamp_(0.0, float(ph))
            keep = (b[:, 2] - b[:, 0] > 1.0) & (b[:, 3] - b[:, 1] > 1.0)
            b = b[keep]
        return crop, b

    def _resize_full(
        self, img: Image.Image, boxes: torch.Tensor
    ) -> Tuple[Image.Image, torch.Tensor]:
        """Downscale a whole image to fit ``max_whole_image_side``, scaling boxes.

        Returns a fresh box tensor, never the shared ``_boxes_cache`` entry.
        """
        resized, scale = resize_pil_keep_aspect(img, self.max_whole_image_side)
        return resized, scale_boxes_xyxy(boxes, scale).clone()

    def __repr__(self) -> str:
        return (
            f"CountingClassTiledDataset(units={len(self)}, "
            f"images={self.num_images}, F_true={self.F_true}, "
            f"tile_size={self.tile_size}, max_whole_image_side={self.max_whole_image_side})"
        )


class TileGroupedSampler(torch.utils.data.Sampler):
    """Sampler that yields all tiles of one source image consecutively.

    With a non-zero ``img_cache_size`` on the dataset, this guarantees each
    parent image is decoded from disk once per epoch.  A plain shuffled loader
    would interleave tiles from different images and miss the cache every call.
    """

    def __init__(
        self,
        dataset: CountingClassTiledDataset,
        *,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        # Stable base order: images sorted by base index, tiles in pool order.
        self._groups: List[List[int]] = [
            dataset.image_groups[img_idx] for img_idx in sorted(dataset.image_groups)
        ]
        self._shuffle = bool(shuffle)
        self._seed = int(seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Vary the shuffle order; call once before each training epoch."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return sum(len(g) for g in self._groups)

    def __iter__(self):
        order = list(range(len(self._groups)))
        if self._shuffle:
            np.random.default_rng(self._seed + self._epoch).shuffle(order)
        for i in order:
            yield from self._groups[i]
