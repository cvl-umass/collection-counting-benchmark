"""Faster R-CNN counting models (ResNet-50 V2 + MobileNetV3-Large backbones)."""

from __future__ import annotations

import math
from typing import List

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from data import collate_detection
from models.base import CountingModel

_NUM_CLASSES = 2          # binary detection: background + foreground
_DETECTIONS_PER_IMG = 300


class _FasterRCNNBase(CountingModel):
    """Shared Faster R-CNN logic.  Concrete subclasses build a backbone and
    define the valid unfreeze levels."""

    trainable = True
    needs_text_prompt = False

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self._model = model
        self._device = device

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

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
        # torchvision floors detections at box_score_thresh=0.05 already, so the
        # capture_floor (default 0.05) keeps essentially every emitted box.
        per_unit_scores: List[np.ndarray] = [
            np.zeros(0, dtype=np.float32) for _ in range(n_total)
        ]
        self._model.eval()
        with torch.no_grad():
            for images, _targets, metas in eval_loader:
                images = [im.to(device) for im in images]
                outs = self._model(images)
                for out, meta in zip(outs, metas):
                    tile_idx = int(meta["tile_index"])
                    scores = out.get("scores")
                    if scores is None:
                        continue
                    scores = scores.detach().cpu().numpy().astype(np.float32)
                    preds[tile_idx] = float((scores >= score_thresh).sum())
                    per_unit_scores[tile_idx] = scores[scores >= capture_floor]
        return preds, per_unit_scores

    # ------------------------------------------------------------------
    # Fine-tuning
    # ------------------------------------------------------------------

    def fit(
        self,
        dataset,
        revealed_indices: List[int],
        device: torch.device,
        *,
        batch_size: int,
        num_workers: int,
        epochs: float,
        lr: float,
        backbone_lr_factor: float = 0.1,
    ) -> List[float]:
        """Fine-tune on the revealed subset; returns per-step training losses."""
        if not revealed_indices:
            return []

        train_loader = DataLoader(
            Subset(dataset, revealed_indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_detection,
            pin_memory=True,
        )
        steps_per_epoch = max(1, math.ceil(len(revealed_indices) / batch_size))
        max_steps = max(1, int(round(float(epochs) * steps_per_epoch)))
        optimizer = self._make_optimizer(lr, backbone_lr_factor)

        self._model.train()
        losses: List[float] = []
        step = 0
        while step < max_steps:
            for images, targets, _metas in train_loader:
                images = [im.to(device) for im in images]
                targets = [
                    {k: v.to(device) for k, v in t.items()} for t in targets
                ]
                loss_dict = self._model(images, targets)
                loss = sum(loss_dict.values())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
                step += 1
                if step >= max_steps:
                    break
        return losses

    def _make_optimizer(
        self, lr: float, backbone_lr_factor: float
    ) -> torch.optim.Optimizer:
        """SGD with backbone params on a lower learning rate."""
        backbone_params = [
            p for n, p in self._model.named_parameters()
            if p.requires_grad and "backbone" in n
        ]
        head_params = [
            p for n, p in self._model.named_parameters()
            if p.requires_grad and "backbone" not in n
        ]
        return torch.optim.SGD(
            [
                {"params": head_params, "lr": lr},
                {"params": backbone_params, "lr": lr * backbone_lr_factor},
            ],
            momentum=0.9,
            weight_decay=1e-4,
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return self._model.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self._model.load_state_dict(state_dict)

    def configure_unfreeze(self, level) -> None:
        raise NotImplementedError  # subclasses define the level table


# ---------------------------------------------------------------------------
# ResNet-50 FPN V2
# ---------------------------------------------------------------------------


class FasterRCNNResNet50(_FasterRCNNBase):
    """Faster R-CNN ResNet-50 FPN V2 (COCO_V1 weights).

    Unfreeze levels:

      ``roi_heads``  Backbone fully frozen; train RPN + RoI heads only.
      ``stride32``   Additionally unfreeze ``backbone.fpn`` and the stride-32
                     stage of the backbone body (``backbone.body.layer4``).
                     Default for the benchmark, since the target classes
                     differ substantially from COCO.
    """

    _LEVELS = ("roi_heads", "stride32")

    def __init__(self, device: torch.device) -> None:
        weights = (
            torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        )
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=weights
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes=_NUM_CLASSES
        )
        model.roi_heads.detections_per_img = _DETECTIONS_PER_IMG
        model.to(device)
        super().__init__(model, device)

    def configure_unfreeze(self, level: str) -> None:
        if level not in self._LEVELS:
            raise ValueError(
                f"unknown FasterRCNNResNet50 unfreeze level {level!r}; "
                f"valid: {self._LEVELS}"
            )
        for p in self._model.backbone.parameters():
            p.requires_grad = False
        if level == "stride32":
            for p in self._model.backbone.fpn.parameters():
                p.requires_grad = True
            for p in self._model.backbone.body.layer4.parameters():
                p.requires_grad = True


# ---------------------------------------------------------------------------
# MobileNetV3-Large FPN
# ---------------------------------------------------------------------------


class FasterRCNNMobileNet(_FasterRCNNBase):
    """Faster R-CNN with MobileNetV3-Large FPN backbone (COCO_V1 weights).

    Unfreeze levels:

      ``roi_heads``  Backbone fully frozen; train RPN + RoI heads only.
      ``stride32``   Additionally unfreeze ``backbone.fpn`` and the stride-32
                     stage of the backbone body -- the last four children of
                     ``backbone.body`` (inverted-residual blocks 13..16 of
                     MobileNetV3-Large; the stage that produces stride-32
                     feature maps).  Default for the benchmark, since the
                     target classes differ substantially from COCO.

    `backbone.body` is an ``IntermediateLayerGetter`` (a ``ModuleDict``
    subclass) wrapping ``mobilenet.features``; its registered children are
    the original inverted-residual feature blocks.
    """

    _LEVELS = ("roi_heads", "stride32")

    def __init__(self, device: torch.device) -> None:
        weights = (
            torchvision.models.detection
            .FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
        )
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights=weights
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes=_NUM_CLASSES
        )
        model.roi_heads.detections_per_img = _DETECTIONS_PER_IMG
        model.to(device)
        super().__init__(model, device)

    def configure_unfreeze(self, level: str) -> None:
        if level not in self._LEVELS:
            raise ValueError(
                f"unknown FasterRCNNMobileNet unfreeze level {level!r}; "
                f"valid: {self._LEVELS}"
            )
        for p in self._model.backbone.parameters():
            p.requires_grad = False
        if level == "stride32":
            for p in self._model.backbone.fpn.parameters():
                p.requires_grad = True
            for block in list(self._model.backbone.body.children())[-4:]:
                for p in block.parameters():
                    p.requires_grad = True
