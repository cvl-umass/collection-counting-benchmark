"""Box geometry helpers for object detection."""

from __future__ import annotations

from typing import Dict, List

import torch


def xywh_to_xyxy(box: Dict[str, float]) -> List[float]:
    """Convert a {x, y, w, h} geometry dict to an [x0, y0, x1, y1] list."""
    x = float(box["x"])
    y = float(box["y"])
    w = float(box["w"])
    h = float(box["h"])
    return [x, y, x + w, y + h]


def clamp_xyxy(box: List[float], width: int, height: int) -> List[float]:
    """Clamp [x0, y0, x1, y1] to image bounds and ensure x0<=x1, y0<=y1."""
    x0, y0, x1, y1 = box
    x0 = max(0.0, min(x0, float(width)))
    x1 = max(0.0, min(x1, float(width)))
    y0 = max(0.0, min(y0, float(height)))
    y1 = max(0.0, min(y1, float(height)))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [x0, y0, x1, y1]


def scale_boxes_xyxy(boxes: torch.Tensor, scale: float) -> torch.Tensor:
    """Scale an (N, 4) xyxy box tensor by a constant factor."""
    if scale == 1.0:
        return boxes
    return boxes * float(scale)
