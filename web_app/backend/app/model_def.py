"""ConvNeXt-Tiny binary classifier definition matching the training notebook."""
from __future__ import annotations

import timm
import torch.nn as nn


def create_model() -> nn.Module:
    """Create the exact timm architecture used during training.

    The checkpoint supplies all learned weights, so pretrained=False is required
    at inference time and avoids downloading ImageNet weights.
    """
    return timm.create_model(
        "convnext_tiny",
        pretrained=False,
        num_classes=1,
        in_chans=3,
    )
