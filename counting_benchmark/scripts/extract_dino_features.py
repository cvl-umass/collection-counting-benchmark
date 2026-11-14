"""Pre-compute and cache per-unit DINOv3 features, used by the spot-check filter."""

from __future__ import annotations

import os

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Make the repo root importable regardless of the current directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.feature_cache import cache_dir_for  # noqa: E402  (path set up above)

# Root of the counting-datasets checkout.  Override with --index-root /
# --raw-root, or set COUNTING_DATA_ROOT; the fallback assumes the two
# repositories are checked out side by side.
_DATA_ROOT = os.environ.get(
    "COUNTING_DATA_ROOT",
    next((p for p in ("../counting_datasets", "../counting-datasets")
          if os.path.isdir(p)), "../counting_datasets"))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tile-sizes",
        default="outputs/suggested_tile_sizes.json",
        help="Suggestion store mapping class_key -> {tile_size, tile_threshold, ...}.",
    )
    ap.add_argument("--index-root", default=f"{_DATA_ROOT}/counting/data")
    ap.add_argument("--raw-root", default=f"{_DATA_ROOT}/raw")
    ap.add_argument(
        "--cache-root",
        default="data/dinov3",
        help="Root for the per-class feature caches.",
    )
    ap.add_argument(
        "--class-key",
        default=None,
        help="Process only this exact class_key (e.g. dota/small_vehicle).",
    )
    ap.add_argument(
        "--class-filter",
        nargs="*",
        default=None,
        metavar="SUBSTR",
        help="Process only classes whose class_key contains one of these "
        "substrings (e.g. --class-filter malaria zebra).",
    )
    ap.add_argument(
        "--dino-model",
        default="facebook/dinov3-vitl16-pretrain-lvd1689m",
        help="HuggingFace model ID for the DINOv3 feature extractor.",
    )
    ap.add_argument(
        "--max-whole-image-side",
        type=int,
        default=1536,
        help="Downscale cap (longer side) for whole, non-tiled units -- a "
        "memory guardrail, never an upscale.",
    )
    ap.add_argument(
        "--dino-input-max-side",
        type=int,
        default=1024,
        help="Cap on a unit's longer side fed to DINOv3 (downscale only), "
        "rounded down to a multiple of 16 (the patch size). DINO-stage only: "
        "applies to every unit (tiles and whole images), and is distinct from "
        "the dataset's --max-whole-image-side (whole images only). Keeps large "
        "per-class tiles from exploding ViT attention.",
    )
    ap.add_argument("--batch-size", type=int, default=8, help="DataLoader batch size.")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument(
        "--device",
        default=None,
        help="torch device (default: cuda if available, else cpu).",
    )
    ap.add_argument(
        "--amp",
        action="store_true",
        help="Run DINOv3 in autocast (fp16/bf16) on CUDA for speed.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract even if a valid features.npy already exists.",
    )
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Tiling derivation from the suggestion store
# ---------------------------------------------------------------------------


def tiling_for(
    spec: Dict[str, Any], max_whole_image_side: int
) -> Tuple[int, int, int]:
    """Map a suggestion entry to ``(max_whole_image_side, tile_size, tile_threshold)``.

    ``tile_size`` sets the grid; ``tile_threshold`` (per image) decides whether
    each image is tiled or kept whole -- both come straight from the store.
    ``max_whole_image_side`` is the shared whole-image downscale cap.  Older
    stores without a ``tile_threshold`` fall back to ``tile_size`` (tile
    anything past one tile).
    """
    ts = int(spec["tile_size"])
    threshold = int(spec.get("tile_threshold", ts))
    return max_whole_image_side, ts, threshold


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def build_dataset(args: argparse.Namespace, class_key: str, spec: Dict[str, Any]):
    """Recreate the tile pool for one class under the suggestion's tiling."""
    from counting_dataset import CountingDatasetIndex

    from data import CountingClassTiledDataset

    mws, ts, threshold = tiling_for(spec, args.max_whole_image_side)
    index = CountingDatasetIndex(root=Path(args.index_root))
    base = index.load_class(
        class_key,
        split=None,
        load_images=False,
        min_count=1,
        on_missing_split="warn",
    )
    ds = CountingClassTiledDataset(
        base,
        raw_root=args.raw_root,
        max_whole_image_side=mws,
        tile_size=ts,
        tile_threshold=threshold,
        verbose=True,
    )
    return ds, (mws, ts, threshold)


# ---------------------------------------------------------------------------
# DINOv3 feature extractor
# ---------------------------------------------------------------------------


class DinoExtractor:
    """Loads DINOv3 once and turns a unit image tensor into a feature vector."""

    def __init__(self, model_name: str, device: torch.device, max_side: int) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ImportError(
                "Install transformers to extract DINOv3 features "
                "(pip install transformers)."
            ) from exc

        print(f"[extract_dino] loading {model_name} ...", flush=True)
        t0 = time.time()
        proc = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.max_side = (max_side // 16) * 16
        mean = np.asarray(proc.image_mean, dtype=np.float32)
        std = np.asarray(proc.image_std, dtype=np.float32)
        self._mean = torch.from_numpy(mean).view(3, 1, 1).to(device)
        self._std = torch.from_numpy(std).view(3, 1, 1).to(device)
        self.feat_dim = int(getattr(self.model.config, "hidden_size", 0)) or None
        print(
            f"[extract_dino] loaded in {time.time() - t0:.1f}s "
            f"(hidden_size={self.feat_dim}, max_side={self.max_side})",
            flush=True,
        )

    def _prep(self, img: torch.Tensor) -> Optional[torch.Tensor]:
        """Normalise, cap the longer side, crop to a multiple of 16. (3,H,W)->(1,3,H',W')."""
        img = img.to(self.device, non_blocking=True)
        img = (img - self._mean) / self._std
        h, w = img.shape[1], img.shape[2]

        longer = max(h, w)
        if longer > self.max_side:
            scale = self.max_side / float(longer)
            nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0), size=(nh, nw), mode="bilinear", align_corners=False
            ).squeeze(0)
            h, w = nh, nw

        h16, w16 = (h // 16) * 16, (w // 16) * 16
        if h16 == 0 or w16 == 0:
            return None  # unit smaller than one patch; caller leaves a zero row
        return img[:, :h16, :w16].unsqueeze(0)

    @torch.inference_mode()
    def features(self, img: torch.Tensor, use_amp: bool) -> Optional[np.ndarray]:
        pixel_values = self._prep(img)
        if pixel_values is None:
            return None
        if use_amp and self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = self.model(pixel_values=pixel_values)
        else:
            out = self.model(pixel_values=pixel_values)
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:  # fall back to mean of patch tokens
            pooled = out.last_hidden_state.mean(dim=1)
        return pooled[0].float().cpu().numpy()


# ---------------------------------------------------------------------------
# Per-class extraction
# ---------------------------------------------------------------------------


def extract_one(
    class_key: str,
    spec: Dict[str, Any],
    extractor: DinoExtractor,
    args: argparse.Namespace,
) -> None:
    ds, (mws, ts, threshold) = build_dataset(args, class_key, spec)
    n = len(ds)
    out_dir = cache_dir_for(Path(args.cache_root), class_key, mws, ts)
    feat_path = out_dir / "features.npy"

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
        pin_memory=(extractor.device.type == "cuda"),
    )

    feat_dim = extractor.feat_dim
    feats: Optional[np.ndarray] = None
    n_skipped = 0
    t0 = time.time()
    n_done = 0
    for images, metas in loader:
        for img, meta in zip(images, metas):
            tile_idx = int(meta["tile_index"])
            vec = extractor.features(img, args.amp)
            if vec is None:
                n_skipped += 1
            else:
                if feats is None:  # lazily size the matrix from the first vector
                    feat_dim = feat_dim or int(vec.shape[0])
                    feats = np.zeros((n, feat_dim), dtype=np.float32)
                feats[tile_idx] = vec
            n_done += 1
            if n_done % 200 == 0:
                rate = n_done / max(time.time() - t0, 1e-6)
                print(
                    f"    {n_done}/{n} units  (~{(n - n_done) / max(rate, 1e-6):.0f}s left)",
                    flush=True,
                )

    if feats is None:
        feats = np.zeros((n, feat_dim or 0), dtype=np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(feat_path, feats)
    # Row-aligned keys so the cache can be re-mapped to a different image pool
    # (cap / reseed) without re-extracting -- see data/feature_cache.py.
    (out_dir / "manifest.json").write_text(json.dumps(ds.manifest()) + "\n")
    meta = {
        "class_key": class_key,
        "n_units": n,
        "feat_dim": int(feats.shape[1]),
        "F_true": int(ds.F_true),
        "n_units_skipped_too_small": n_skipped,
        "max_whole_image_side": mws,
        "tile_size": ts,
        "tile_threshold": threshold,
        "dino_model": args.dino_model,
        "dino_input_max_side": extractor.max_side,
        "suggested_tile_size": int(spec["tile_size"]),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"  saved {feats.shape} -> {feat_path}  "
        f"({time.time() - t0:.1f}s, {n_skipped} units too small)",
        flush=True,
    )


def _collate(batch):
    """Keep only (image_tensor, meta) per sample; targets are unused here."""
    return [b[0] for b in batch], [b[2] for b in batch]


# ---------------------------------------------------------------------------
# Selection + main
# ---------------------------------------------------------------------------


def select_classes(
    store: Dict[str, Any], class_key: Optional[str], class_filter: Optional[List[str]]
) -> List[str]:
    keys = list(store.keys())
    if class_key is not None:
        if class_key not in store:
            sys.exit(f"ERROR: {class_key!r} not found in the suggestion store.")
        return [class_key]
    if class_filter:
        keys = [k for k in keys if any(s in k for s in class_filter)]
        if not keys:
            sys.exit(f"ERROR: no class_key matched filter {class_filter}.")
    return keys


def main() -> None:
    args = parse_args()
    store_path = Path(args.tile_sizes)
    if not store_path.is_file():
        sys.exit(f"ERROR: --tile-sizes not found: {store_path}")
    store: Dict[str, Any] = json.loads(store_path.read_text())

    classes = select_classes(store, args.class_key, args.class_filter)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[extract_dino] device={device}; {len(classes)} class(es) to consider")

    # Plan: show what is cached vs pending before loading the (heavy) model.
    pending: List[str] = []
    for ck in classes:
        mws, ts, _ = tiling_for(store[ck], args.max_whole_image_side)
        feat_path = cache_dir_for(Path(args.cache_root), ck, mws, ts) / "features.npy"
        if feat_path.exists() and not args.overwrite:
            print(f"  CACHED  {ck}  -> {feat_path}")
        else:
            pending.append(ck)
            print(f"  PENDING {ck}  (max_whole_side={mws}, ts={ts})")

    if not pending:
        print("[extract_dino] nothing to do (use --overwrite to force).")
        return

    extractor = DinoExtractor(args.dino_model, device, args.dino_input_max_side)
    for i, ck in enumerate(pending, 1):
        print(f"\n=== [{i}/{len(pending)}] {ck} ===", flush=True)
        try:
            extract_one(ck, store[ck], extractor, args)
        except Exception as exc:  # keep going across classes
            print(f"  ERROR on {ck}: {exc}", flush=True)

    print("\n[extract_dino] done.")


if __name__ == "__main__":
    main()
