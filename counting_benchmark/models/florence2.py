"""Florence-2 -- zero-shot vision-language model for object detection."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from models.base import CountingModel

try:
    from transformers import AutoProcessor, Florence2ForConditionalGeneration
except ImportError:  # pragma: no cover
    AutoProcessor = Florence2ForConditionalGeneration = None  # type: ignore[assignment,misc]


_CHECKPOINT = "florence-community/Florence-2-large"
_TASK = "<OPEN_VOCABULARY_DETECTION>"
_MIN_INFERENCE_SIDE = 32


class Florence2Model(CountingModel):
    """Florence-2 wrapped in the CountingModel interface."""

    trainable = False
    needs_text_prompt = True

    def __init__(self, device: torch.device, class_name: str) -> None:
        if Florence2ForConditionalGeneration is None:
            raise ImportError(
                "transformers>=4.45 is required for Florence2Model "
                "(pip install transformers timm einops)."
            )
        self._processor = AutoProcessor.from_pretrained(_CHECKPOINT)
        self._model = Florence2ForConditionalGeneration.from_pretrained(
            _CHECKPOINT,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
        )
        if device.type != "cuda":
            self._model.to(device)
        self._model.eval()
        self._device = device
        self._class_name = class_name

    def predict(
        self,
        eval_loader: DataLoader,
        device: torch.device,
        *,
        score_thresh: float = 0.5,  # unused; Florence-2 emits boxes via generation
        capture_floor: float = 0.05,  # unused; generation yields no confidences
    ):
        n_total = len(eval_loader.dataset)
        preds = np.zeros(n_total, dtype=np.float32)
        # Florence-2 generates boxes via text and exposes no per-box confidence,
        # so the soft-detection score arrays are empty for every unit.
        per_unit_scores = [np.zeros(0, dtype=np.float32) for _ in range(n_total)]

        # Generation is sequential; process one image at a time to avoid
        # variable-length output padding complexity.
        with torch.no_grad():
            for images, _targets, metas in eval_loader:
                for img_tensor, meta in zip(images, metas):
                    pil_img = to_pil_image(img_tensor.clamp(0, 1))
                    if min(pil_img.size) < _MIN_INFERENCE_SIDE:
                        continue

                    prompt = _TASK + self._class_name
                    inputs = self._processor(
                        text=prompt, images=pil_img, return_tensors="pt",
                    ).to(device)

                    generated = self._model.generate(
                        **inputs, max_new_tokens=1024, num_beams=3,
                    )
                    raw = self._processor.batch_decode(
                        generated, skip_special_tokens=False,
                    )[0]
                    parsed = self._processor.post_process_generation(
                        raw, task=_TASK, image_size=pil_img.size,
                    )
                    bboxes = parsed.get(_TASK, {}).get("bboxes", [])
                    tile_idx = int(meta["tile_index"])
                    preds[tile_idx] = float(len(bboxes))

        return preds, per_unit_scores
