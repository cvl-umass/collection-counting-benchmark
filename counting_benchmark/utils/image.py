"""Image loading and resizing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from PIL import Image


def load_image_rgb(path: Path) -> Image.Image:
    """Open an image file and convert it to RGB."""
    return Image.open(path).convert("RGB")


def resize_pil_keep_aspect(
    pil: Image.Image, target_size: int
) -> Tuple[Image.Image, float]:
    """Resize *pil* so its longer side equals *target_size*, preserving aspect.

    Returns (resized_image, scale_factor).  If the image already fits within
    *target_size*, the original image is returned unchanged with scale 1.0.
    """
    w, h = pil.size
    longest = max(w, h)
    if longest <= target_size:
        return pil, 1.0
    scale = float(target_size) / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return pil.resize((new_w, new_h), resample=Image.BILINEAR), scale
