"""
utils.py
--------
Shared helpers for the explainability module.

  - Output-directory management
  - Test-sample selection  (one correct + one incorrect per emotion class)
  - Colour-map and normalisation helpers
  - Plot functions:
      plot_token_attribution      — horizontal bar chart for text tokens
      plot_waveform_attribution   — waveform + per-sample IG heatmap
      plot_temporal_occlusion     — occlusion score bar chart over time
      plot_gradcam_overlay        — original image | Grad-CAM blended image
      plot_ig_face                — original image | IG magnitude | IG signed
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


# ── Canonical emotion class names (MELD 7-class) ──────────────────────────────

MELD_CLASSES = [
    "Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"
]


# ── Output directory ──────────────────────────────────────────────────────────

# Relative to notebooks/ (same convention as other EXP_ROOT_* constants)
EXP_ROOT_EXPLAIN = "../experiments/explainability"


def make_explain_dir(modality: str, model_slug: str) -> str:
    """
    Create and return  experiments/explainability/{modality}/{model_slug}/.

    Parameters
    ----------
    modality   : 'text', 'audio', or 'face'
    model_slug : slugified model name, e.g. 'bert_base_uncased'
    """
    path = os.path.join(EXP_ROOT_EXPLAIN, modality, model_slug)
    os.makedirs(path, exist_ok=True)
    return path


# ── Sample selection ──────────────────────────────────────────────────────────

def select_explanation_samples(
    labels: List[int],
    preds: List[int],
    id2label: Dict[int, str],
    n_correct: int = 1,
    n_incorrect: int = 1,
    seed: int = 42,
) -> Dict[str, List[Tuple[int, bool]]]:
    """
    For each emotion class, randomly select up to *n_correct* correctly-
    classified and *n_incorrect* misclassified test-set indices.

    Parameters
    ----------
    labels     : ground-truth integer labels for all test samples
    preds      : predicted integer labels  (same order)
    id2label   : mapping int → emotion string
    n_correct  : samples per class that the model got right
    n_incorrect: samples per class that the model got wrong
    seed       : RNG seed for reproducibility

    Returns
    -------
    Dict keyed by emotion name.  Each value is a list of
    (sample_index, is_correct) tuples.
    """
    rng        = np.random.default_rng(seed)
    labels_arr = np.array(labels)
    preds_arr  = np.array(preds)
    correct    = labels_arr == preds_arr

    result: Dict[str, List[Tuple[int, bool]]] = {}
    for cls_id, cls_name in id2label.items():
        cls_mask      = labels_arr == cls_id
        correct_idx   = np.where(cls_mask &  correct)[0].tolist()
        incorrect_idx = np.where(cls_mask & ~correct)[0].tolist()

        rng.shuffle(correct_idx)
        rng.shuffle(incorrect_idx)

        chosen = (
            [(int(i), True)  for i in correct_idx[:n_correct]]
            + [(int(i), False) for i in incorrect_idx[:n_incorrect]]
        )
        if chosen:
            result[cls_name] = chosen

    return result


# ── Colour helpers ────────────────────────────────────────────────────────────

def _rdbur():
    """Red–White–Blue diverging colormap (positive=red, negative=blue)."""
    return plt.cm.RdBu_r


def normalise_attributions(attr: np.ndarray) -> np.ndarray:
    """Scale *attr* to [–1, 1] by dividing by max absolute value."""
    amax = np.abs(attr).max()
    return attr / (amax + 1e-8)


# ── Token attribution (text) ──────────────────────────────────────────────────

def plot_token_attribution(
    tokens: List[str],
    scores: np.ndarray,
    title: str,
    save_path: str,
    method_label: str = "Attribution score",
) -> None:
    """
    Horizontal bar chart of per-token attribution scores.

    Bars are coloured on a red–white–blue scale:
      red  = token increases predicted probability  (positive score)
      blue = token decreases predicted probability  (negative score)

    Parameters
    ----------
    tokens       : list of subword tokens (including special tokens)
    scores       : float array of shape (seq_len,)
    title        : plot title
    save_path    : output PNG path
    method_label : x-axis label (e.g. 'IG attribution', 'Occlusion score')
    """
    scores_norm = normalise_attributions(scores)
    cmap        = _rdbur()
    colours     = cmap((scores_norm + 1) / 2)   # map [–1,1] → [0,1]

    n      = len(tokens)
    fig_h  = max(3, n * 0.35)
    fig, ax = plt.subplots(figsize=(8, fig_h))

    y = np.arange(n)
    ax.barh(y, scores, color=colours, edgecolor="grey", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(tokens, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(method_label)
    ax.set_title(title, fontsize=11, pad=8)

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=mcolors.Normalize(vmin=-1, vmax=1)
    )
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Normalised score", shrink=0.6)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Waveform + IG attribution (audio) ────────────────────────────────────────

def plot_waveform_attribution(
    waveform: np.ndarray,
    attr_per_sample: np.ndarray,
    sr: int,
    title: str,
    save_path: str,
    n_display_bins: int = 200,
) -> None:
    """
    Two-panel figure: raw waveform (top) + binned attribution heatmap (bottom).

    Parameters
    ----------
    waveform        : 1-D float array (preprocessed, normalised waveform)
    attr_per_sample : 1-D float array of same length as *waveform*
    sr              : sample rate in Hz
    title           : figure title
    save_path       : output PNG path
    n_display_bins  : number of time bins for the attribution panel
    """
    n = len(waveform)
    t = np.linspace(0, n / sr, n)

    # Bin attribution into n_display_bins equal-width segments
    bin_edges   = np.linspace(0, n, n_display_bins + 1).astype(int)
    bin_centers = np.array(
        [(bin_edges[i] + bin_edges[i + 1]) / 2 / sr for i in range(n_display_bins)]
    )
    binned_attr = np.array(
        [attr_per_sample[bin_edges[i]:bin_edges[i + 1]].mean()
         for i in range(n_display_bins)]
    )

    vmax = max(np.abs(binned_attr).max(), 1e-8)
    norm    = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap    = _rdbur()
    colours = cmap(norm(binned_attr))
    widths  = np.diff(np.linspace(0, t[-1], n_display_bins + 1))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))

    ax1.plot(t, waveform, color="steelblue", linewidth=0.5, alpha=0.8)
    ax1.set_ylabel("Amplitude")
    ax1.set_title(title, fontsize=11)
    ax1.set_xlim(0, t[-1])
    ax1.set_xticklabels([])

    ax2.bar(
        bin_centers, binned_attr,
        width=widths * 0.9, color=colours, edgecolor="none",
    )
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("IG attribution")
    ax2.set_xlim(0, t[-1])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax2, label="Attribution", orientation="vertical", shrink=0.8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_temporal_occlusion(
    scores: np.ndarray,
    total_duration_s: float,
    title: str,
    save_path: str,
) -> None:
    """
    Bar chart of per-segment occlusion scores over time.

    A positive score means silencing that segment hurt the prediction
    → that window is temporally important.

    Parameters
    ----------
    scores            : 1-D float array of length n_segments
    total_duration_s  : total audio duration in seconds
    title             : plot title
    save_path         : output PNG path
    """
    n       = len(scores)
    seg_dur = total_duration_s / n
    centers = np.arange(n) * seg_dur + seg_dur / 2

    vmin = min(scores.min(), -1e-8)
    vmax = max(scores.max(),  1e-8)
    norm    = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap    = _rdbur()
    colours = cmap(norm(scores))

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(centers, scores, width=seg_dur * 0.85, color=colours, edgecolor="none")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Prob. drop when silenced")
    ax.set_title(title, fontsize=11)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Score", shrink=0.8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Grad-CAM overlay (face) ───────────────────────────────────────────────────

def plot_gradcam_overlay(
    image_np: np.ndarray,
    cam_map: np.ndarray,
    title: str,
    save_path: str,
    alpha: float = 0.45,
) -> None:
    """
    Two-panel figure: original face image | Grad-CAM heatmap blended on image.

    Parameters
    ----------
    image_np  : (H, W, 3) array, uint8 or float [0, 1]
    cam_map   : (H, W) float array in [0, 1] from pytorch-grad-cam
    title     : figure title (suptitle)
    save_path : output PNG path
    alpha     : heatmap opacity in the blended panel
    """
    from PIL import Image as _PIL

    if image_np.dtype != np.uint8:
        img_u8 = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
    else:
        img_u8 = image_np

    # Resize CAM to match image resolution
    h, w = img_u8.shape[:2]
    cam_resized = np.array(
        _PIL.fromarray((cam_map * 255).astype(np.uint8)).resize(
            (w, h), _PIL.BILINEAR
        )
    ) / 255.0

    heatmap = plt.cm.jet(cam_resized)[..., :3]        # (H, W, 3) float
    blended = (1 - alpha) * img_u8 / 255.0 + alpha * heatmap
    blended = np.clip(blended, 0, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(img_u8);  ax1.set_title("Input", fontsize=9);    ax1.axis("off")
    ax2.imshow(blended); ax2.set_title("Grad-CAM", fontsize=9); ax2.axis("off")

    fig.suptitle(title, fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Integrated Gradients face overlay ────────────────────────────────────────

def plot_ig_face(
    image_np: np.ndarray,
    ig_magnitude: np.ndarray,
    ig_signed: np.ndarray,
    title: str,
    save_path: str,
) -> None:
    """
    Three-panel figure:
      1. Original face image
      2. IG magnitude heatmap  (hot colourmap — brighter = more attribution)
      3. Signed IG heatmap     (RdBu_r — red = positive, blue = negative)

    Parameters
    ----------
    image_np     : (H, W, 3) array, uint8 or float [0, 1]
    ig_magnitude : (H, W) — sum of |attribution| across RGB channels
    ig_signed    : (H, W) — signed attribution summed across RGB channels
    title        : figure suptitle
    save_path    : output PNG path
    """
    if image_np.dtype != np.uint8:
        img_u8 = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
    else:
        img_u8 = image_np

    amax = max(np.abs(ig_signed).max(), 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_u8)
    axes[0].set_title("Input", fontsize=9)
    axes[0].axis("off")

    im1 = axes[1].imshow(ig_magnitude, cmap="hot")
    axes[1].set_title("IG magnitude", fontsize=9)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(ig_signed, cmap="RdBu_r", vmin=-amax, vmax=amax)
    axes[2].set_title("IG signed", fontsize=9)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── ImageNet denormalisation (for display) ────────────────────────────────────

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalize_imagenet(tensor_chw) -> np.ndarray:
    """
    Reverse ImageNet normalisation on a (3, H, W) tensor or array.

    Returns a uint8 (H, W, 3) array suitable for imshow / PIL.
    """
    import torch
    if isinstance(tensor_chw, torch.Tensor):
        arr = tensor_chw.cpu().numpy()
    else:
        arr = np.asarray(tensor_chw)

    img = arr.transpose(1, 2, 0)                           # (H, W, 3)
    img = img * _IMAGENET_STD + _IMAGENET_MEAN             # undo normalisation
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)
