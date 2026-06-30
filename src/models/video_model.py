"""
video_model.py
--------------
Model builders, head modifications, and training utilities for the video
modality.

Supported architectures
-----------------------
"mvit_v2_s"  — Multiscale Vision Transformer V2-Small (torchvision).
               Expects input: (B, 3, T=16, H=224, W=224).
               Original head: Dropout(0) → Linear(768, 400).  forward() calls x=x[:,0] first.
               Modified head: Dropout(head_dropout) → Linear(768, num_classes).

"s3d"        — Separable 3D Convolutions (torchvision).
               Expects input: (B, 3, T≥8, H=224, W=224).
               Original classifier: Sequential(Dropout(0.5), Conv3d(1024,400,1)).
               Modified classifier: Sequential(Dropout(head_dropout), Conv3d(1024, num_classes, 1)).

Both models are loaded with their ImageNet (kinetics) pre-trained weights.

Anti-overfitting helpers
------------------------
get_video_param_groups  — two AdamW groups: backbone at lr×mult, head at lr.
freeze_video_backbone   — freezes all params except the classification head.
unfreeze_video_backbone — restores requires_grad=True for all params.
count_params            — returns total / trainable param counts.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from torchvision.models.video import (
    MViT_V2_S_Weights,
    S3D_Weights,
    mvit_v2_s,
    s3d,
)

from src.data.video_dataset import NUM_CLASSES


# ── Internal builders ─────────────────────────────────────────────────────────

def _build_mvit_v2_s(
    num_classes:  int,
    head_dropout: float,
) -> nn.Module:
    """
    Loads MViT-V2-S with Kinetics-400 weights and replaces the classification
    head Linear with Dropout → Linear(768, num_classes).
    """
    weights = MViT_V2_S_Weights.DEFAULT
    model   = mvit_v2_s(weights=weights)

    # The actual MViT head is: Sequential(Dropout(0, inplace=True), Linear(768, 400))
    # MViT.forward() does x = x[:, 0] (CLS token extraction) *before* calling self.head,
    # so the head always receives a 2D tensor (B, embed_dim).
    # Do NOT carry over old_head[1] (Linear(768→400)) — it would collapse features to 400-dim,
    # causing a shape mismatch with the new Linear(768, num_classes).
    # Replace the entire head with a fresh Dropout + Linear for num_classes.
    old_head = model.head   # nn.Sequential
    in_feats = old_head[-1].in_features   # 768

    model.head = nn.Sequential(
        nn.Dropout(p=head_dropout),
        nn.Linear(in_feats, num_classes),
    )
    return model


def _build_s3d(
    num_classes:  int,
    head_dropout: float,
) -> nn.Module:
    """
    Loads S3D with Kinetics-400 weights and replaces the classifier head
    Conv3d with Dropout(head_dropout) → Conv3d(1024, num_classes, kernel_size=1).
    """
    weights = S3D_Weights.DEFAULT
    model   = s3d(weights=weights)

    # model.classifier = Sequential(Dropout(0.5), Conv3d(1024, 400, 1))
    in_channels = model.classifier[1].in_channels   # 1024

    model.classifier = nn.Sequential(
        nn.Dropout(p=head_dropout),
        nn.Conv3d(in_channels, num_classes, kernel_size=1),
    )
    return model


# ── Public factory ────────────────────────────────────────────────────────────

_BUILDERS = {
    "mvit_v2_s": _build_mvit_v2_s,
    "s3d":       _build_s3d,
}


def load_video_model(
    model_name:   str,
    num_classes:  int   = NUM_CLASSES,
    head_dropout: float = 0.3,
    device:       str   = "cpu",
) -> nn.Module:
    """
    Instantiates *model_name* with pretrained weights and a custom head.

    Parameters
    ----------
    model_name   : one of "mvit_v2_s", "s3d"
    num_classes  : number of output logits (7 for MELD)
    head_dropout : dropout probability in the classification head
    device       : "cuda" or "cpu"

    Returns
    -------
    nn.Module ready for training / inference on *device*.
    """
    if model_name not in _BUILDERS:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from {list(_BUILDERS)}"
        )
    model = _BUILDERS[model_name](num_classes, head_dropout)
    model.to(device)
    return model


# ── Head identification ───────────────────────────────────────────────────────

_HEAD_ATTR = {
    "mvit_v2_s": "head",
    "s3d":       "classifier",
}

def _get_head_params(model: nn.Module, model_name: str) -> List[nn.Parameter]:
    """Returns a list of parameters belonging to the classification head."""
    head_attr = _HEAD_ATTR[model_name]
    head_module = getattr(model, head_attr)
    return list(head_module.parameters())


def _get_backbone_params(model: nn.Module, model_name: str) -> List[nn.Parameter]:
    """Returns a list of parameters NOT in the classification head."""
    head_attr   = _HEAD_ATTR[model_name]
    head_module = getattr(model, head_attr)
    head_ids    = {id(p) for p in head_module.parameters()}
    return [p for p in model.parameters() if id(p) not in head_ids]


# ── Anti-overfitting helpers ──────────────────────────────────────────────────

def get_video_param_groups(
    model:            nn.Module,
    model_name:       str,
    head_lr:          float,
    backbone_lr_mult: float = 0.1,
) -> List[Dict]:
    """
    Returns two AdamW parameter groups:
      • backbone : lr = head_lr × backbone_lr_mult
      • head     : lr = head_lr

    Pass the result directly to AdamW:
        optimizer = AdamW(get_video_param_groups(model, name, lr, mult), weight_decay=…)
    """
    return [
        {
            "params": _get_backbone_params(model, model_name),
            "lr":     head_lr * backbone_lr_mult,
            "name":   "backbone",
        },
        {
            "params": _get_head_params(model, model_name),
            "lr":     head_lr,
            "name":   "head",
        },
    ]


def freeze_video_backbone(model: nn.Module, model_name: str) -> None:
    """
    Freezes all parameters except those in the classification head.
    Call before the training loop starts to implement backbone freeze warm-up.
    """
    head_attr   = _HEAD_ATTR[model_name]
    head_module = getattr(model, head_attr)
    head_ids    = {id(p) for p in head_module.parameters()}

    for p in model.parameters():
        if id(p) not in head_ids:
            p.requires_grad = False


def unfreeze_video_backbone(model: nn.Module) -> None:
    """Restores requires_grad=True for all model parameters."""
    for p in model.parameters():
        p.requires_grad = True


def count_params(model: nn.Module) -> Dict[str, int]:
    """Returns {'total': …, 'trainable': …} parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
