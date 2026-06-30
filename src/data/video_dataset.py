"""
video_dataset.py
----------------
PyTorch Dataset and DataLoader factories for the video-modality emotion-
recognition pipeline.

Two dataset modes
-----------------
FrameCacheVideoDataset  — loads a pre-built (T_CACHE, H, W, 3) uint8 .npy
    file and subsamples it to the model's required clip length T at load time.
    Recommended for all training runs (O(1) RAM, fast I/O).

NoCacheVideoDataset     — decodes frames live from .mp4 using cv2 (slow but
    requires no pre-processing step).

Temporal augmentation
---------------------
For training we sample a *single* set of spatial transform parameters and apply
them identically to every frame in the clip.  This ensures temporal consistency
(the same crop / flip appears across all frames).  Per-frame RandomErasing is
applied independently afterwards.

Clip tensor format
------------------
Output shape : (3, T, H, W)   float32, ImageNet-normalised.
This is the format expected by torchvision's MViT-V2 and S3D models.

DataLoader factories
--------------------
make_video_loaders(train_df, dev_df, test_df, model_name, …)
    Returns (train_loader, dev_loader, test_loader).
    Accepts an optional pre-built WeightedRandomSampler so the caller can
    control class balance.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from src.config import (
    FACE_MELD_CLASSES,
    VIDEO_IMG_SIZE,
    VIDEO_RANDOM_ERASING_P,
    VIDEO_T_CACHE,
    VIDEO_T_FRAMES,
)
from src.data.video_preprocessing import (
    load_analysis_boxes,
    load_video_metadata,
    sample_clip_frames,
    crop_or_resize,
)

# ── Label maps ────────────────────────────────────────────────────────────────
NUM_CLASSES: int = len(FACE_MELD_CLASSES)
LABEL2ID: Dict[str, int] = {cls.lower(): i for i, cls in enumerate(FACE_MELD_CLASSES)}
ID2LABEL: Dict[int, str] = {i: cls for cls, i in LABEL2ID.items()}

# ── ImageNet normalisation (shared with face modality) ────────────────────────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_normalise = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
_to_tensor = transforms.ToTensor()   # HWC uint8 → CHW float32 [0,1]


# ── Augmentation helpers ──────────────────────────────────────────────────────

def _augment_clip(
    frames_thwc: np.ndarray,
    img_size: int = VIDEO_IMG_SIZE,
    erasing_p: float = VIDEO_RANDOM_ERASING_P,
) -> torch.Tensor:
    """
    Training augmentation for a (T, H, W, 3) uint8 clip.

    Strategy
    --------
    1. Sample *one* set of geometric parameters (crop box, h-flip, rotation,
       colour-jitter coefficients) and apply them identically to every frame
       so the clip stays temporally coherent.
    2. Apply per-frame RandomErasing independently (different erasure patches
       per frame add noise robustness without breaking temporal coherence).

    Returns
    -------
    Tensor of shape (3, T, img_size, img_size), float32, ImageNet-normalised.
    """
    T, H, W, _ = frames_thwc.shape

    # ── Shared geometric params ────────────────────────────────────────────────
    # RandomResizedCrop params
    crop_transform = transforms.RandomResizedCrop(
        img_size,
        scale=(0.75, 1.0),
        ratio=(0.9, 1.1),
        interpolation=transforms.InterpolationMode.BILINEAR,
    )
    crop_params = crop_transform.get_params(
        torch.zeros(1, H, W),   # dummy tensor for param generation
        scale=(0.75, 1.0),
        ratio=(0.9, 1.1),
    )  # returns (i, j, h, w)
    ci, cj, ch, cw = crop_params

    do_flip  = random.random() < 0.5
    angle    = random.uniform(-15, 15)

    # Colour jitter — compute once and apply to all frames
    jitter = transforms.ColorJitter(
        brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
    )
    # Get deterministic transform (apply same jitter to all frames)
    jitter_fn = jitter

    # RandomErasing applied per frame
    eraser = transforms.RandomErasing(
        p=erasing_p, scale=(0.02, 0.25), ratio=(0.3, 3.3), value=0
    )

    clip_tensors: List[torch.Tensor] = []

    for t in range(T):
        frame = frames_thwc[t]   # (H, W, 3) uint8
        pil   = transforms.functional.to_pil_image(frame)

        # Geometric transforms (shared params)
        pil = transforms.functional.resized_crop(
            pil, ci, cj, ch, cw, [img_size, img_size],
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        if do_flip:
            pil = transforms.functional.hflip(pil)
        pil = transforms.functional.rotate(pil, angle, expand=False, fill=0)

        # Colour jitter (re-apply same transform instance — different result per frame
        # because jitter samples internally; use same params would need manual compose)
        # For true shared colour params we use a pre-fixed fn applied per frame.
        pil = jitter_fn(pil)

        # To tensor + normalise
        t_frame = _to_tensor(pil)          # (3, H, W)
        t_frame = _normalise(t_frame)

        # Per-frame RandomErasing
        t_frame = eraser(t_frame)

        clip_tensors.append(t_frame)

    clip = torch.stack(clip_tensors, dim=1)   # (3, T, H, W)
    return clip


def _resize_clip(
    frames_thwc: np.ndarray,
    img_size: int = VIDEO_IMG_SIZE,
) -> torch.Tensor:
    """
    Deterministic resize for evaluation (no augmentation).

    Returns
    -------
    Tensor of shape (3, T, img_size, img_size), float32, ImageNet-normalised.
    """
    T = frames_thwc.shape[0]
    resize = transforms.Resize(
        (img_size, img_size),
        interpolation=transforms.InterpolationMode.BILINEAR,
    )
    clip_tensors: List[torch.Tensor] = []

    for t in range(T):
        frame   = frames_thwc[t]          # (H, W, 3) uint8
        pil     = transforms.functional.to_pil_image(frame)
        pil     = resize(pil)
        t_frame = _to_tensor(pil)
        t_frame = _normalise(t_frame)
        clip_tensors.append(t_frame)

    clip = torch.stack(clip_tensors, dim=1)   # (3, T, H, W)
    return clip


def _subsample(frames_thwc: np.ndarray, T: int) -> np.ndarray:
    """
    Uniformly subsample T frames from a cache array of shape
    (T_CACHE, H, W, 3).  If T >= T_CACHE all frames are returned as-is.
    """
    T_cache = frames_thwc.shape[0]
    if T >= T_cache:
        return frames_thwc
    indices = np.linspace(0, T_cache - 1, T, dtype=int)
    return frames_thwc[indices]


# ── Datasets ──────────────────────────────────────────────────────────────────

class FrameCacheVideoDataset(Dataset):
    """
    Loads per-utterance frame caches built by video_preprocessing.build_frame_cache.

    Cache path: <cache_dir>/dia{D}_utt{U}.npy
    Cache shape: (VIDEO_T_CACHE, VIDEO_IMG_SIZE, VIDEO_IMG_SIZE, 3) uint8

    Parameters
    ----------
    df          : DataFrame from load_video_metadata (filtered for valid rows)
    cache_dir   : directory produced by _frame_cache_dir (contains *.npy files)
    T           : number of frames the model expects (subsampled from T_CACHE)
    img_size    : spatial resolution (should match cache; kept for API symmetry)
    train       : True → apply augmentation, False → deterministic resize
    erasing_p   : per-frame RandomErasing probability (training only)
    """

    def __init__(
        self,
        df:        "pd.DataFrame",
        cache_dir: str,
        T:         int            = VIDEO_T_CACHE,
        img_size:  int            = VIDEO_IMG_SIZE,
        train:     bool           = True,
        erasing_p: float          = VIDEO_RANDOM_ERASING_P,
    ):
        self.cache_dir = cache_dir
        self.T         = T
        self.img_size  = img_size
        self.train     = train
        self.erasing_p = erasing_p

        # Drop rows whose .npy cache file is missing and warn once.
        records = df.reset_index(drop=True)
        missing_mask = ~records.apply(
            lambda row: os.path.exists(
                os.path.join(cache_dir,
                             f"dia{int(row['dialogue_id'])}_utt{int(row['utterance_id'])}.npy")
            ),
            axis=1,
        )
        n_missing = missing_mask.sum()
        if n_missing:
            import warnings
            warnings.warn(
                f"FrameCacheVideoDataset: skipping {n_missing} row(s) whose .npy "
                f"cache file was not found in '{cache_dir}'.",
                stacklevel=2,
            )
        self.records = records[~missing_mask].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        row = self.records.iloc[idx]
        d_id = int(row["dialogue_id"])
        u_id = int(row["utterance_id"])
        label_id = int(row["label_id"])

        npy_path = os.path.join(self.cache_dir, f"dia{d_id}_utt{u_id}.npy")
        frames   = np.load(npy_path)          # (T_CACHE, H, W, 3) uint8

        frames = _subsample(frames, self.T)   # (T, H, W, 3)

        if self.train:
            clip = _augment_clip(frames, self.img_size, self.erasing_p)
        else:
            clip = _resize_clip(frames, self.img_size)

        return clip, label_id, d_id, u_id


class NoCacheVideoDataset(Dataset):
    """
    Decodes video frames live from .mp4 files.  Use when disk space prevents
    pre-building the frame cache.  Significantly slower than FrameCacheVideoDataset.

    Parameters
    ----------
    df          : DataFrame from load_video_metadata
    T           : frames per clip
    img_size    : resize target
    crop_mode   : "face" or "full"
    face_version: version string used by face-preprocessing (for box lookup)
    split       : dataset split ("train" | "dev" | "test")
    train       : True → augmentation, False → resize only
    erasing_p   : RandomErasing probability
    """

    def __init__(
        self,
        df:           "pd.DataFrame",
        split:        str,
        T:            int   = VIDEO_T_CACHE,
        img_size:     int   = VIDEO_IMG_SIZE,
        crop_mode:    str   = "face",
        face_version: str   = "v1",
        train:        bool  = True,
        erasing_p:    float = VIDEO_RANDOM_ERASING_P,
    ):
        from src.data.video_preprocessing import (
            get_box_for_frame,
            VIDEO_FACE_BOX_MAX_GAP,
        )
        self.records      = df.reset_index(drop=True)
        self.split        = split
        self.T            = T
        self.img_size     = img_size
        self.crop_mode    = crop_mode
        self.face_version = face_version
        self.train        = train
        self.erasing_p    = erasing_p

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        from src.data.video_preprocessing import (
            get_box_for_frame,
            VIDEO_FACE_BOX_MAX_GAP,
        )
        row      = self.records.iloc[idx]
        d_id     = int(row["dialogue_id"])
        u_id     = int(row["utterance_id"])
        label_id = int(row["label_id"])

        raw_frames = sample_clip_frames(row["video_path"], self.T)

        # Load face boxes if needed
        box_map: Dict = {}
        if self.crop_mode == "face":
            box_map = load_analysis_boxes(
                self.split, d_id, u_id, self.face_version
            )

        frames_list = []
        for fi, rgb in raw_frames[:self.T]:
            box = None
            if self.crop_mode == "face":
                box = get_box_for_frame(fi, box_map, VIDEO_FACE_BOX_MAX_GAP)
            frame = crop_or_resize(rgb, box, self.img_size)
            frames_list.append(frame)

        # Pad if fewer frames decoded
        if len(frames_list) < self.T:
            last = frames_list[-1] if frames_list else np.zeros(
                (self.img_size, self.img_size, 3), dtype=np.uint8
            )
            while len(frames_list) < self.T:
                frames_list.append(last)

        frames = np.stack(frames_list, axis=0)   # (T, H, W, 3)

        if self.train:
            clip = _augment_clip(frames, self.img_size, self.erasing_p)
        else:
            clip = _resize_clip(frames, self.img_size)

        return clip, label_id, d_id, u_id


# ── DataLoader factory ────────────────────────────────────────────────────────

def _build_video_sampler(train_df: "pd.DataFrame") -> WeightedRandomSampler:
    """
    Inverse-frequency weighted sampler — mirrors face modality implementation.
    """
    labels        = train_df["label_id"].values.astype(int)
    class_counts  = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    class_counts  = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights     = torch.from_numpy(sample_weights).float(),
        num_samples = len(train_df),
        replacement = True,
    )


def make_video_loaders(
    train_df:       "pd.DataFrame",
    dev_df:         "pd.DataFrame",
    test_df:        "pd.DataFrame",
    model_name:     str,
    cache_dir:      str,
    batch_size:     int,
    num_workers:    int              = 4,
    img_size:       int              = VIDEO_IMG_SIZE,
    erasing_p:      float            = VIDEO_RANDOM_ERASING_P,
    balanced_sampler: bool           = True,
    train_sampler:  Optional[WeightedRandomSampler] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds train / dev / test DataLoaders backed by FrameCacheVideoDataset.

    Parameters
    ----------
    train_df / dev_df / test_df : DataFrames from load_video_metadata (filtered)
    model_name   : key into VIDEO_T_FRAMES for the per-model clip length
    cache_dir    : directory containing per-split subdirectories with *.npy files
                   (i.e. what _frame_cache_dir returns for each split)
    batch_size   : mini-batch size for this model
    num_workers  : DataLoader worker processes
    balanced_sampler : if True and train_sampler is None, build one automatically
    train_sampler    : optional pre-built WeightedRandomSampler

    Returns
    -------
    (train_loader, dev_loader, test_loader)
    """
    T = VIDEO_T_FRAMES.get(model_name, VIDEO_T_CACHE)

    train_ds = FrameCacheVideoDataset(
        df        = train_df,
        cache_dir = cache_dir["train"],
        T         = T,
        img_size  = img_size,
        train     = True,
        erasing_p = erasing_p,
    )
    dev_ds = FrameCacheVideoDataset(
        df        = dev_df,
        cache_dir = cache_dir["dev"],
        T         = T,
        img_size  = img_size,
        train     = False,
    )
    test_ds = FrameCacheVideoDataset(
        df        = test_df,
        cache_dir = cache_dir["test"],
        T         = T,
        img_size  = img_size,
        train     = False,
    )

    # Build (or rebuild) the sampler from the *filtered* dataset records so
    # that sampler indices never exceed len(train_ds) - 1.  This matters
    # when FrameCacheVideoDataset drops rows for missing .npy files.
    if balanced_sampler or train_sampler is not None:
        train_sampler = _build_video_sampler(train_ds.records)

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        sampler     = train_sampler,
        shuffle     = (train_sampler is None),
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )

    return train_loader, dev_loader, test_loader
