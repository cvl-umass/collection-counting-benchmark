"""GroundingDINO -- zero-shot open-vocabulary object detector."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from models.base import CountingModel

try:
    from transformers import AutoProcessor, GroundingDinoForObjectDetection
except ImportError:  # pragma: no cover
    AutoProcessor = GroundingDinoForObjectDetection = None  # type: ignore[assignment,misc]


_CHECKPOINT = "IDEA-Research/grounding-dino-base"
# Tiny edge tiles (e.g., 3px wide from DOTA tiling) crash the transformers
# top-k proposal step.  Skip; their pred stays 0.
_MIN_INFERENCE_SIDE = 32


class GroundingDINOModel(CountingModel):
    """GroundingDINO wrapped in the CountingModel interface."""

    trainable = False
    needs_text_prompt = True

    def __init__(self, device: torch.device, class_name: str) -> None:
        if GroundingDinoForObjectDetection is None:
            raise ImportError(
                "transformers>=4.38 is required for GroundingDINOModel "
                "(pip install transformers)."
            )
        self._processor = AutoProcessor.from_pretrained(_CHECKPOINT)
        self._model = GroundingDinoForObjectDetection.from_pretrained(_CHECKPOINT)
        self._model.to(device)
        self._model.eval()
        self._device = device
        # GroundingDINO requires the prompt to end with a period.
        self._text = class_name.strip().rstrip(".") + "."

    def predict(
        self,
        eval_loader: DataLoader,
        device: torch.device,
        *,
        score_thresh: float = 0.5,
        capture_floor: float = 0.05,
    ):
        n_total = len(eval_loader.dataset)
        preds = np.zeros(n_total, dtype=np.float32)
        per_unit_scores = [np.zeros(0, dtype=np.float32) for _ in range(n_total)]

        with torch.no_grad():
            for images, _targets, metas in eval_loader:
                pil_imgs = [to_pil_image(img.clamp(0, 1)) for img in images]
                valid = [
                    (img, m) for img, m in zip(pil_imgs, metas)
                    if min(img.size) >= _MIN_INFERENCE_SIDE
                ]
                if not valid:
                    continue
                pil_v, metas_v = zip(*valid)

                inputs = self._processor(
                    images=list(pil_v),
                    text=[self._text] * len(pil_v),
                    return_tensors="pt",
                    padding=True,
                ).to(device)

                outputs = self._model(**inputs)
                # outputs.logits: (B, Q, text_seq_len); max over text tokens
                # then sigmoid is the per-query confidence.
                scores = outputs.logits.sigmoid().max(dim=-1).values  # (B, Q)
                for s_i, meta in zip(scores, metas_v):
                    tile_idx = int(meta["tile_index"])
                    s_i = s_i.detach().cpu().numpy().astype(np.float32)
                    preds[tile_idx] = float((s_i > score_thresh).sum())
                    # Capture only above the floor: the ~900 queries are mostly
                    # near-zero background otherwise.
                    per_unit_scores[tile_idx] = s_i[s_i >= capture_floor]

        return preds, per_unit_scores
