"""Model factory.

The driver builds a single model via :func:`build_model` and then calls
:meth:`CountingModel.configure_unfreeze`.  Zero-shot models receive the text
prompt at construction; trainable models do not need one.
"""

from __future__ import annotations

import torch

from models.base import CountingModel


def build_model(args, device: torch.device) -> CountingModel:
    """Construct a counting model from the run's CLI arguments.

    Recognised values of ``args.model``:

      ``frcnn_resnet50``   Faster R-CNN with ResNet-50 FPN V2 backbone.
      ``frcnn_mobilenet``  Faster R-CNN with MobileNetV3-Large FPN backbone.
      ``grounding_dino``   GroundingDINO zero-shot detector.
      ``owlv2``            OWLv2 zero-shot detector.
      ``florence2``        Florence-2 zero-shot detector.
    """
    name = args.model.lower()

    if name == "frcnn_resnet50":
        from models.faster_rcnn import FasterRCNNResNet50
        return FasterRCNNResNet50(device)

    if name == "frcnn_mobilenet":
        from models.faster_rcnn import FasterRCNNMobileNet
        return FasterRCNNMobileNet(device)

    if name == "grounding_dino":
        from models.grounding_dino import GroundingDINOModel
        return GroundingDINOModel(device, class_name=_resolve_text_prompt(args))

    if name == "owlv2":
        from models.owlv2 import OWLv2Model
        return OWLv2Model(device, class_name=_resolve_text_prompt(args))

    if name == "florence2":
        from models.florence2 import Florence2Model
        return Florence2Model(device, class_name=_resolve_text_prompt(args))

    raise ValueError(
        f"Unknown model {name!r}.  Valid: frcnn_resnet50, frcnn_mobilenet, "
        f"grounding_dino, owlv2, florence2."
    )


def _resolve_text_prompt(args) -> str:
    """Derive the text prompt for zero-shot models from CLI args.

    Priority:
      1. ``--text-prompt`` if explicitly set.
      2. The last segment of ``--class-key`` with underscores replaced by spaces
         (e.g. ``malaria/red_blood_cell`` → ``red blood cell``).
      3. ``"object"`` as a generic fallback.
    """
    prompt = getattr(args, "text_prompt", None)
    if prompt:
        return prompt
    class_key = getattr(args, "class_key", None)
    if class_key:
        return class_key.split("/")[-1].replace("_", " ")
    return "object"
