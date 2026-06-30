"""
face_model.py
-------------
Loads pretrained torchvision models and replaces their classification heads
with a new linear layer sized to NUM_CLASSES.

Supported architectures
-----------------------
"vit_b_16"            ViT-B/16 pretrained on ImageNet-1K (IMAGENET1K_V1,
                      224×224).  Head gets an extra Dropout layer before the
                      Linear to regularise the large pretrained backbone.
"mobilenet_v3_small"  MobileNetV3-Small (lightweight baseline).  Already has
                      a Dropout in its classifier; we only swap the final
                      Linear.

Both models expect ImageNet-normalised 224×224 inputs — the same preprocessing
that get_transforms() in face_dataset.py produces.
"""

from __future__ import annotations

from typing import List, Dict

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ViT_B_16_Weights,
)

# IMAGENET1K_SWAG_E2E_V1 expects 384×384 — use V1 (224×224) instead.
_VIT_WEIGHTS = ViT_B_16_Weights.IMAGENET1K_V1

from src.data.face_dataset import NUM_CLASSES


# ── Public API ────────────────────────────────────────────────────────────────

def load_face_model(
    model_name: str,
    num_classes: int = NUM_CLASSES,
    head_dropout: float = 0.3,
    device: str = "cpu",
) -> nn.Module:
    """
    Returns a model with its head replaced and ready for fine-tuning.

    Parameters
    ----------
    model_name : str
        One of "vit_b_16" or "mobilenet_v3_small".
    num_classes : int
        Number of output classes (default = 7 for MELD).
    head_dropout : float
        Dropout probability inserted before the ViT classification Linear.
        (MobileNet already has Dropout in its classifier; this param is
        ignored for that architecture.)
    device : str
        "cuda" or "cpu".
    """
    if model_name == "vit_b_16":
        model = _build_vit_b_16(num_classes, head_dropout)
    elif model_name == "mobilenet_v3_small":
        model = _build_mobilenet_v3_small(num_classes)
    else:
        raise ValueError(
            f"Unknown face model: '{model_name}'. "
            "Choose 'vit_b_16' or 'mobilenet_v3_small'."
        )
    return model.to(device)


def get_param_groups(
    model: nn.Module,
    model_name: str,
    head_lr: float,
    backbone_lr_mult: float = 0.1,
) -> List[Dict]:
    """
    Returns two AdamW parameter groups for differential learning rates:
      - backbone : head_lr × backbone_lr_mult  (pretrained layers, slow)
      - head     : head_lr                     (new classifier, fast)

    Parameters
    ----------
    model            : the model returned by load_face_model
    model_name       : "vit_b_16" or "mobilenet_v3_small"
    head_lr          : learning rate for the new classification head
    backbone_lr_mult : multiplier applied to get the backbone LR (default 0.1)
    """
    if model_name == "vit_b_16":
        head_prefix = "heads"
    elif model_name == "mobilenet_v3_small":
        head_prefix = "classifier"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    backbone_params = [p for n, p in model.named_parameters()
                       if not n.startswith(head_prefix)]
    head_params     = [p for n, p in model.named_parameters()
                       if n.startswith(head_prefix)]

    return [
        {"params": backbone_params, "lr": head_lr * backbone_lr_mult},
        {"params": head_params,     "lr": head_lr},
    ]


def freeze_backbone(model: nn.Module, model_name: str) -> None:
    """Freezes all parameters except the classification head."""
    head_prefix = "heads" if model_name == "vit_b_16" else "classifier"
    for name, p in model.named_parameters():
        if not name.startswith(head_prefix):
            p.requires_grad_(False)


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreezes all parameters."""
    for p in model.parameters():
        p.requires_grad_(True)


def count_params(model: nn.Module) -> dict:
    """Returns total and trainable parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# ── Private builders ──────────────────────────────────────────────────────────

def _build_vit_b_16(num_classes: int, head_dropout: float) -> nn.Module:
    """
    ViT-B/16 with IMAGENET1K_V1 pretrained weights (224×224 input).
    Replaces model.heads.head with Dropout → Linear(768, num_classes).
    All parameters start trainable; use freeze_backbone() to freeze the
    encoder during the warm-up phase.
    """
    model = models.vit_b_16(weights=_VIT_WEIGHTS)
    in_features = model.heads.head.in_features          # 768
    model.heads.head = nn.Sequential(
        nn.Dropout(p=head_dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def _build_mobilenet_v3_small(num_classes: int) -> nn.Module:
    """
    MobileNetV3-Small with IMAGENET1K_V1 pretrained weights.
    Original classifier: Sequential(Linear 576→1024, Hardswish, Dropout, Linear 1024→1000)
    We replace only the final Linear: 1024 → num_classes.
    The existing Dropout(0.2) in the classifier is preserved.
    """
    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features      # 1024
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
