"""OWLv2 -- zero-shot open-vocabulary object detector."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from models.base import CountingModel

try:
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
except ImportError:  # pragma: no cover
    Owlv2ForObjectDetection = Owlv2Processor = None  # type: ignore[assignment,misc]


_CHECKPOINT = "google/owlv2-large-patch14-ensemble"
# Tiny edge tiles cause channel-dimension ambiguity in the HF processor
# (e.g. a 3px-wide DOTA edge tile with H == 3 collapses with C == 3).
_MIN_INFERENCE_SIDE = 32


class OWLv2Model(CountingModel):
    """OWLv2 wrapped in the CountingModel interface."""

    trainable = False
    needs_text_prompt = True

    def __init__(self, device: torch.device, class_name: str) -> None:
        if Owlv2ForObjectDetection is None:
            raise ImportError(
                "transformers is required for OWLv2Model "
                "(pip install transformers)."
            )
        self._processor = Owlv2Processor.from_pretrained(_CHECKPOINT)
        self._model = Owlv2ForObjectDetection.from_pretrained(_CHECKPOINT)
        self._model.to(device)
        self._model.eval()
        self._device = device
        # OWLv2 performs best with descriptive prompts.
        self._text = f"a photo of a {class_name}"

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
                    (img, m, orig)
                    for img, m, orig in zip(pil_imgs, metas, images)
                    if min(img.size) >= _MIN_INFERENCE_SIDE
                ]
                if not valid:
                    continue
                pil_v, metas_v, images_v = zip(*valid)

                target_sizes = [(img.shape[1], img.shape[2]) for img in images_v]
                texts = [[self._text]] * len(pil_v)

                inputs = self._processor(
                    text=texts,
                    images=list(pil_v),
                    return_tensors="pt",
                ).to(device)

                outputs = self._model(**inputs)
                # Post-process at the capture floor (not score_thresh) so the
                # sub-threshold band is retained; the count re-applies
                # score_thresh and stays identical to the old behavior.
                results = self._processor.post_process_grounded_object_detection(
                    outputs,
                    threshold=capture_floor,
                    target_sizes=target_sizes,
                    text_labels=texts,
                )
                for result, meta in zip(results, metas_v):
                    tile_idx = int(meta["tile_index"])
                    scores = result["scores"].detach().cpu().numpy().astype(np.float32)
                    preds[tile_idx] = float((scores >= score_thresh).sum())
                    per_unit_scores[tile_idx] = scores

        return preds, per_unit_scores
