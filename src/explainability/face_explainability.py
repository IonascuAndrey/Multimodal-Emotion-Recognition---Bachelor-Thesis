"""
face_explainability.py
----------------------
Three attribution methods for ViT-B/16 and MobileNetV3-Small face-emotion
classifiers.

Attention Rollout  (ViT-B/16 only)
    Traces how attention flows from the [CLS] token back through all 12
    transformer layers.  For each layer the per-head attention matrix is
    averaged, an identity is added (residual connection), and the result is
    row-normalised; all layers are then multiplied together.  The CLS row of
    the final product gives the importance of each of the 196 patches, which
    is reshaped to 14×14 and upsampled to the input resolution.  Because it
    reads the model's own attention mechanism rather than gradients or feature
    SVDs, it is robust to the global token-mixing that makes EigenCAM and
    GradCAM spatially noisy on Vision Transformers.

EigenCAM  (pytorch-grad-cam)
    Highlights spatially important image regions using the first principal
    component (SVD) of the target layer's feature maps.  Used for
    MobileNetV3-Small (targets model.features[-1]).

Integrated Gradients  (Captum)
    Attributes the predicted class probability to individual pixels by
    integrating gradients of the model output along the straight-line path
    from a black (all-zeros) baseline to the actual normalised image.
    Channel-wise absolute values are summed to produce a 2-D magnitude map;
    the signed channel sum is also returned for directional analysis.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ── Target-layer resolution ────────────────────────────────────────────────────

def _resolve_target_layers(model: nn.Module, model_name: str):
    """
    Return (target_layers, reshape_transform) for pytorch-grad-cam.

    ViT-B/16
        Layer  : model.encoder.layers[-1]  (full last transformer block output)
        Reshape: removes the CLS token and folds (B, 196, 768) →
                 (B, 768, 14, 14) so standard spatial averaging can be applied.
        Note   : Targeting the block *output* (after attention + residuals) keeps
                 the gradient path short (block → encoder.ln → CLS → head) and
                 avoids the global token-mixing that makes gradients spatially
                 noisy when hooking ln_1 (pre-attention LayerNorm).

    MobileNetV3-Small
        Layer  : model.features[-1]  (standard spatial feature map)
        Reshape: None  (no reshape needed)
    """
    if model_name == "vit_b_16":
        target_layers = [model.encoder.layers[-1]]

        def reshape_transform(tensor, height: int = 14, width: int = 14):
            # tensor: (B, 1 + num_patches, D)  — index 0 is the CLS token
            result = tensor[:, 1:, :].reshape(
                tensor.size(0), height, width, tensor.size(2)
            )
            return result.transpose(2, 3).transpose(1, 2)  # (B, D, H, W)

        return target_layers, reshape_transform

    elif model_name == "mobilenet_v3_small":
        return [model.features[-1]], None

    else:
        raise ValueError(
            f"Unknown face model: '{model_name}'. "
            "Expected 'vit_b_16' or 'mobilenet_v3_small'."
        )


# ── FaceExplainer ──────────────────────────────────────────────────────────────

class FaceExplainer:
    """
    Explainability wrapper for ViT-B/16 and MobileNetV3-Small face classifiers.

    Parameters
    ----------
    model      : loaded face model (output of load_face_model / _load_checkpoint)
    model_name : 'vit_b_16' or 'mobilenet_v3_small'
    device     : 'cuda' or 'cpu'
    """

    def __init__(self, model: nn.Module, model_name: str, device: str) -> None:
        self.model      = model.to(device).eval()
        self.model_name = model_name
        self.device     = device

        self._target_layers, self._reshape = _resolve_target_layers(model, model_name)
        self._ig = IntegratedGradients(self._forward_fn)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _forward_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Wrapper that returns raw logits for Captum."""
        return self.model(x)

    # ── Prediction ─────────────────────────────────────────────────────────────

    def predict(self, image_tensor: torch.Tensor) -> Tuple[int, np.ndarray]:
        """
        Forward pass on a single (1, 3, H, W) normalised tensor.

        Returns
        -------
        predicted_class : int
        probabilities   : float32 array of shape (num_classes,)
        """
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            logits = self.model(image_tensor)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return int(probs.argmax()), probs

    # ── Grad-CAM ───────────────────────────────────────────────────────────────

    def grad_cam(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Compute an EigenCAM saliency map.

        EigenCAM uses the first principal component (SVD) of the target layer's
        feature maps instead of gradients, making it robust to the global
        token-mixing noise that degrades GradCAM on ViT.  pytorch-grad-cam
        handles upsampling internally; the returned map is already resized to
        the input spatial resolution.

        Parameters
        ----------
        image_tensor : (1, 3, H, W) ImageNet-normalised tensor
        target_class : class to explain (defaults to model argmax)

        Returns
        -------
        cam_map      : float32 (H, W) in [0, 1]
        target_class : class index actually used
        """
        image_tensor = image_tensor.to(self.device)
        if target_class is None:
            pred, _ = self.predict(image_tensor)
            target_class = pred

        targets = [ClassifierOutputTarget(target_class)]
        with EigenCAM(
            model=self.model,
            target_layers=self._target_layers,
            reshape_transform=self._reshape,
        ) as cam:
            cam_map = cam(input_tensor=image_tensor, targets=targets)
        # cam_map: (1, H, W) in [0, 1] — already upsampled to input resolution
        return cam_map[0].astype(np.float32), target_class

    # ── Attention Rollout ──────────────────────────────────────────────────────

    def attention_rollout(
        self,
        image_tensor: torch.Tensor,
        head_fusion: str = "mean",
        discard_ratio: float = 0.9,
    ) -> np.ndarray:
        """
        Compute an Attention Rollout saliency map for ViT-B/16.

        Forward hooks capture the raw per-head attention weight matrix
        (B, num_heads, 197, 197) from every transformer layer without
        re-running the model.  The hooks call module.forward() directly
        (bypassing __call__) so they do not recurse.

        Rollout algorithm per layer
        ---------------------------
        1. Fuse heads (mean / max / min) → (197, 197)
        2. Zero out the lowest `discard_ratio` fraction of weights (noise
           reduction — Chefer et al. 2021).
        3. Add identity matrix to model the residual connection.
        4. Row-normalise.
        5. Left-multiply the running rollout matrix.

        The CLS row (index 0) of the final matrix gives the importance of
        each of the 196 patch tokens.  This is reshaped to 14×14 and
        upsampled to the input resolution via bilinear interpolation.

        Parameters
        ----------
        image_tensor  : (1, 3, H, W) ImageNet-normalised tensor
        head_fusion   : how to aggregate attention heads —
                        'mean' (default), 'max', or 'min'
        discard_ratio : fraction of lowest-weight connections zeroed out
                        before rollout to reduce noise (default 0.9)

        Returns
        -------
        saliency : float32 (H, W) in [0, 1], upsampled from 14×14
        """
        if self.model_name != "vit_b_16":
            raise ValueError("attention_rollout is only supported for vit_b_16.")

        image_tensor = image_tensor.to(self.device)
        attn_list: list = []
        hooks = []

        def _make_hook():
            def _hook(module, inp, _out):
                # inp = (query, key, value) — call .forward() directly to
                # request attention weights without triggering hooks again.
                q, k, v = inp[0], inp[1], inp[2]
                with torch.no_grad():
                    _, w = module.forward(
                        q, k, v,
                        need_weights=True,
                        average_attn_weights=False,
                    )
                # w: (B, num_heads, seq_len, seq_len)
                attn_list.append(w.detach().cpu())
            return _hook

        for layer in self.model.encoder.layers:
            hooks.append(
                layer.self_attention.register_forward_hook(_make_hook())
            )

        with torch.no_grad():
            _ = self.model(image_tensor)

        for h in hooks:
            h.remove()

        num_tokens = attn_list[0].shape[-1]       # 197 = 1 CLS + 196 patches
        rollout    = torch.eye(num_tokens)         # identity as starting point

        for attn in attn_list:
            A = attn[0]                            # (num_heads, 197, 197)

            # 1. Fuse heads
            if head_fusion == "mean":
                A = A.mean(dim=0)
            elif head_fusion == "max":
                A = A.max(dim=0).values
            elif head_fusion == "min":
                A = A.min(dim=0).values
            else:
                raise ValueError(f"Unknown head_fusion: '{head_fusion}'")
            # A: (197, 197)

            # 2. Discard lowest-attention connections
            flat      = A.flatten()
            threshold = flat.kthvalue(int(flat.numel() * discard_ratio)).values
            A[A < threshold] = 0.0

            # 3. Add identity (residual)
            A = A + torch.eye(num_tokens)

            # 4. Row-normalise
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)

            # 5. Accumulate
            rollout = A @ rollout

        # CLS row → patch importances (drop the CLS token itself at index 0)
        mask = rollout[0, 1:].reshape(14, 14).numpy().astype(np.float32)

        # Normalise to [0, 1]
        lo, hi = mask.min(), mask.max()
        if hi > lo:
            mask = (mask - lo) / (hi - lo)

        # Upsample to input resolution
        h_in, w_in = image_tensor.shape[-2], image_tensor.shape[-1]
        mask_t  = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # (1,1,14,14)
        mask_up = F.interpolate(
            mask_t, size=(h_in, w_in), mode="bilinear", align_corners=False
        )
        return mask_up.squeeze().numpy().astype(np.float32)

    # ── Integrated Gradients ───────────────────────────────────────────────────

    def integrated_gradients(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        n_steps: int = 50,
        internal_batch_size: int = 4,
        baseline: str = "black",
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Compute pixel-level Integrated Gradients attributions.

        Parameters
        ----------
        image_tensor        : (1, 3, H, W) ImageNet-normalised tensor
        target_class        : class to attribute towards; defaults to argmax
        n_steps             : integration steps (↑ accuracy, ↑ compute)
        internal_batch_size : Captum interpolation batch size
        baseline            : 'black' (all-zero image) or 'mean'
                              (constant ImageNet-mean pixel)

        Returns
        -------
        ig_magnitude : float32 (H, W) — |attribution| summed across RGB
        ig_signed    : float32 (H, W) — signed attribution summed across RGB
        target_class : class index actually used
        """
        image_tensor = image_tensor.to(self.device).float()

        if target_class is None:
            with torch.no_grad():
                logits = self.model(image_tensor)
            target_class = int(logits.argmax(dim=-1))

        if baseline == "black":
            base = torch.zeros_like(image_tensor)
        elif baseline == "mean":
            mean = torch.tensor(
                [0.485, 0.456, 0.406], device=self.device
            ).reshape(1, 3, 1, 1).expand_as(image_tensor)
            base = mean
        else:
            raise ValueError(f"Unknown baseline: '{baseline}'")

        attributions = self._ig.attribute(
            inputs=image_tensor,
            baselines=base,
            target=target_class,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
        )
        # attributions: (1, 3, H, W)
        attr         = attributions.squeeze(0).cpu().detach().numpy()  # (3, H, W)
        ig_magnitude = np.abs(attr).sum(axis=0).astype(np.float32)    # (H, W)
        ig_signed    = attr.sum(axis=0).astype(np.float32)            # (H, W)
        return ig_magnitude, ig_signed, target_class
