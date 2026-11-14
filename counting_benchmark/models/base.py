"""Abstract base class for counting / detection models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


class CountingModel(ABC):
    """Unified interface for detection and counting models."""

    #: Whether the model supports gradient-based fine-tuning.
    trainable: bool = True

    #: Whether the model requires a text class prompt for inference.
    needs_text_prompt: bool = False

    @abstractmethod
    def predict(
        self,
        eval_loader: DataLoader,
        device: torch.device,
        *,
        score_thresh: float = 0.5,
        capture_floor: float = 0.05,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Run inference once over *eval_loader*; return ``(counts, scores)``.

        ``counts`` is a ``(N,)`` float32 array indexed by ``meta['tile_index']``:
        the number of detections with confidence ``>= score_thresh`` per unit
        (the hard count that feeds the sampler).  ``scores`` is a length-``N``
        list whose entry ``i`` is the 1-D float32 array of per-box confidences
        ``>= capture_floor`` for unit ``i`` -- the raw signal for post-hoc soft
        detection.  Both are produced in a single inference pass.  Units the
        model skips (too small) or with no detections get count 0 and an empty
        score array.  Models without per-box confidences (Florence-2) return
        empty score arrays.
        """

    def predict_counts(
        self,
        eval_loader: DataLoader,
        device: torch.device,
        *,
        score_thresh: float = 0.5,
    ) -> np.ndarray:
        """Convenience wrapper: the ``(N,)`` hard count only (see :meth:`predict`)."""
        counts, _ = self.predict(eval_loader, device, score_thresh=score_thresh)
        return counts

    def fit(
        self,
        dataset,
        revealed_indices: List[int],
        device: torch.device,
        **train_kwargs: Any,
    ) -> List[float]:
        """Fine-tune on *revealed_indices*.  Default: no-op for zero-shot models.

        Trainable subclasses override with an explicit signature accepting
        ``batch_size``, ``num_workers``, ``epochs``, ``lr``, and
        ``backbone_lr_factor``.
        """
        return []

    def configure_unfreeze(self, level: Any) -> None:
        """Configure which layers are trainable.  The meaning of *level* is
        model-specific; see subclasses for valid values."""
        return None

    def state_dict(self) -> Dict[str, Any]:
        """Serializable state for checkpointing.  Empty for zero-shot models."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore from a checkpoint dict.  No-op for zero-shot models."""
        return None
