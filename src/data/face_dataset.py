"""
face_dataset.py
---------------
Dataset, transforms, per-image .npy cache builder, and DataLoaders for the
face modality.

Reads `metadata.csv` files produced by face_preprocessing.py.
Each row represents ONE saved face crop (frame-level).

Training DataLoader  → frame-level samples (label = utterance emotion,
                        weight = score_comb^power clipped to [weight_min, 1.0]).
Eval DataLoader      → same frame-level samples but grouped so the eval loop
                        can average probabilities across the K frames that
                        belong to the same utterance before computing metrics.

Per-image .npy cache
--------------------
Location : <cache_dir>/npy/<version>/<split>/<image_stem>.npy
Each file : uint8 array (img_size, img_size, 3) — one face crop.

No data is loaded into RAM at dataset construction time.  Each __getitem__
reads exactly one ~150 KB file, so RAM usage is O(batch_size) and
num_workers can be set freely without multiplying RAM.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import (
    FACE_CACHE_DIR,
    FACE_IMG_SIZE,
    FACE_MELD_CLASSES,
    FACE_OUTPUT_ROOT,
    FACE_RANDOM_ERASING_P,
    FACE_SCORE_WEIGHT_MIN,
    FACE_SCORE_WEIGHT_NORMALIZE,
    FACE_SCORE_WEIGHT_POWER,
    FACE_TRAIN_BATCH_SIZE,
    FACE_USE_CACHE,
    FACE_USE_SCORE_WEIGHTS,
    VERSION,
)

# ── Label map (fixed for all face experiments) ────────────────────────────────
LABEL2ID: Dict[str, int] = {cls: i for i, cls in enumerate(FACE_MELD_CLASSES)}
ID2LABEL: Dict[int, str] = {i: cls for cls, i in LABEL2ID.items()}
NUM_CLASSES = len(FACE_MELD_CLASSES)


# ── ImageNet normalisation (used by both ViT and MobileNet pretrained weights) ─
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(train: bool) -> transforms.Compose:
    """
    Returns torchvision transform pipelines.

    Training  : resize → random crop → hflip → colour jitter → rotation →
                ToTensor → Normalise → RandomErasing.
    Eval      : resize → ToTensor → Normalise (no augmentation).

    RandomErasing (train only) randomly blacks out a rectangular patch of
    the face, preventing the model from over-relying on specific facial
    regions (eyes, mouth, etc.).
    """
    if train:
        return transforms.Compose([
            transforms.Resize((FACE_IMG_SIZE + 16, FACE_IMG_SIZE + 16)),
            transforms.RandomCrop(FACE_IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.3, hue=0.08),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            transforms.RandomErasing(p=FACE_RANDOM_ERASING_P,
                                     scale=(0.02, 0.25),
                                     ratio=(0.3, 3.3)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((FACE_IMG_SIZE, FACE_IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])


# ── Metadata loading ──────────────────────────────────────────────────────────

def load_face_metadata(split: str, version: str = VERSION) -> pd.DataFrame:
    """
    Reads the metadata.csv for *split* and keeps only rows with status='ok'.
    Adds a `label_id` column and drops rows whose emotion is not in LABEL2ID.
    """
    meta_path = os.path.join(FACE_OUTPUT_ROOT, version, split, "metadata.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"No metadata.csv at {meta_path}. "
            "Run 03_preprocess_face.ipynb first."
        )
    df = pd.read_csv(meta_path)
    df = df[df["status"] == "ok"].copy()
    df = df[df["emotion_norm"].isin(LABEL2ID)].copy()
    df["label_id"] = df["emotion_norm"].map(LABEL2ID)
    df = df.reset_index(drop=True)
    return df


# ── Per-image .npy cache ──────────────────────────────────────────────────────
# Stored separately from the legacy NPZ files under a dedicated "npy/" subfolder
# so both can coexist without conflict.

def _cache_dir_path(split: str, version: str, cache_dir: str) -> str:
    """Returns the directory that holds the per-image .npy files for *split*.

    Layout: <cache_dir>/npy/<version>/<split>/
    This is intentionally separate from the NPZ files at
    <cache_dir>/<version>/<split>_pixels.npz.
    """
    return os.path.join(cache_dir, "npy", version, split)


def _npy_filename(img_path: str) -> str:
    """Derives the .npy filename from the PNG image path (basename, swap ext)."""
    return os.path.splitext(os.path.basename(img_path))[0] + ".npy"


def build_cache(
    df: pd.DataFrame,
    split: str,
    version: str = VERSION,
    cache_dir: str = FACE_CACHE_DIR,
    img_size: int = FACE_IMG_SIZE,
    force: bool = False,
) -> str:
    """
    Decodes every PNG in *df* to a uint8 (img_size, img_size, 3) array and
    saves it as an individual .npy file inside:
        <cache_dir>/npy/<version>/<split>/<image_stem>.npy

    RAM usage : O(1) — one image decoded and written at a time.
    Resumable : files that already exist are skipped (unless force=True).

    Returns the cache directory path.
    """
    out_dir = _cache_dir_path(split, version, cache_dir)
    os.makedirs(out_dir, exist_ok=True)

    pending = [
        row["image_path"] for _, row in df.iterrows()
        if force or not os.path.exists(
            os.path.join(out_dir, _npy_filename(row["image_path"]))
        )
    ]

    if not pending:
        print(f"[cache] {split} already fully cached ({len(df)} files) → {out_dir}")
        return out_dir

    print(f"[cache] Building {split} .npy cache → {out_dir} "
          f"({len(pending)} remaining / {len(df)} total) …")

    failed = 0
    for img_path in pending:
        npy_path = os.path.join(out_dir, _npy_filename(img_path))
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((img_size, img_size), Image.BILINEAR)
            np.save(npy_path, np.array(img, dtype=np.uint8))
        except Exception as e:
            print(f"  [warn] Could not load {img_path}: {e}")
            failed += 1

    written = len(pending) - failed
    print(f"[cache] Done. {written} files written, {failed} failed.")
    return out_dir


# ── Dataset ───────────────────────────────────────────────────────────────────

class FaceEmotionDataset(Dataset):
    """
    Frame-level dataset.  Each __getitem__ returns:
        image   : FloatTensor (3, H, W)  — normalised face crop
        label   : int                    — emotion class index
        weight  : float                  — sample weight for loss
        d_id    : int                    — dialogue_id   (for utterance grouping)
        u_id    : int                    — utterance_id  (for utterance grouping)

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_face_metadata — one row per saved frame.
    transform : torchvision transform
        Applied to each PIL image.
    use_score_weights : bool
        If True, compute frame weights from score_comb.
    score_weight_power : float
        Exponent applied to score_comb before clipping.
    score_weight_min : float
        Floor weight so low-scoring frames still contribute.
    score_weight_normalize : bool
        Normalise weights globally so their mean = 1.
    use_cache : bool
        Load pixel data from the per-image .npy cache instead of PNG files.
    cache_dir : str | None
        Path to the per-split .npy directory (returned by build_cache).
        Required when use_cache=True.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: transforms.Compose,
        use_score_weights: bool = FACE_USE_SCORE_WEIGHTS,
        score_weight_power: float = FACE_SCORE_WEIGHT_POWER,
        score_weight_min: float = FACE_SCORE_WEIGHT_MIN,
        score_weight_normalize: bool = FACE_SCORE_WEIGHT_NORMALIZE,
        use_cache: bool = FACE_USE_CACHE,
        cache_dir: Optional[str] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform

        # Weights
        if use_score_weights and "score_comb" in df.columns:
            raw = df["score_comb"].fillna(0.0).values.astype(np.float32)
            w = np.clip(np.power(raw, score_weight_power), score_weight_min, 1.0)
            if score_weight_normalize and w.mean() > 0:
                w = w / w.mean()
            self.weights = w
        else:
            self.weights = np.ones(len(df), dtype=np.float32)

        # Store only the directory path — no data is loaded here.
        # Each __getitem__ reads one ~150 KB .npy file on demand.
        # Workers receive only this path string via pickle, so RAM stays O(1).
        self._cache_dir: Optional[str] = None
        if use_cache and cache_dir:
            if os.path.isdir(cache_dir):
                self._cache_dir = cache_dir
            else:
                print(f"[FaceEmotionDataset] Cache dir not found at {cache_dir}; "
                      "falling back to PNG loading.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["image_path"]

        if self._cache_dir is not None:
            npy_path = os.path.join(self._cache_dir, _npy_filename(img_path))
            if os.path.exists(npy_path):
                arr = np.load(npy_path)              # uint8 (H, W, 3)
                img = Image.fromarray(arr, mode="RGB")
            else:
                img = Image.open(img_path).convert("RGB")
        else:
            img = Image.open(img_path).convert("RGB")

        image  = self.transform(img)
        label  = int(row["label_id"])
        weight = float(self.weights[idx])
        d_id   = int(row["dialogue_id"])
        u_id   = int(row["utterance_id"])

        return image, label, weight, d_id, u_id


# ── DataLoaders ───────────────────────────────────────────────────────────────

def make_face_loaders(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = FACE_TRAIN_BATCH_SIZE,
    num_workers: int = 4,
    use_cache: bool = FACE_USE_CACHE,
    cache_dir: str = FACE_CACHE_DIR,
    version: str = VERSION,
    use_score_weights: bool = FACE_USE_SCORE_WEIGHTS,
    score_weight_power: float = FACE_SCORE_WEIGHT_POWER,
    score_weight_min: float = FACE_SCORE_WEIGHT_MIN,
    score_weight_normalize: bool = FACE_SCORE_WEIGHT_NORMALIZE,
    train_sampler=None,          # optional WeightedRandomSampler (disables shuffle)
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds train / dev / test DataLoaders.

    Train loader  : augmentation transforms, score weights.
                    If train_sampler is provided it is used instead of shuffle.
    Dev/Test loaders : deterministic, no augmentation.
    """
    split_dfs = {"train": train_df, "dev": dev_df, "test": test_df}

    # Locate (or build) per-split .npy cache directories.
    cache_dirs: Dict[str, Optional[str]] = {}
    for split, df in split_dfs.items():
        if use_cache:
            cdir = _cache_dir_path(split, version, cache_dir)
            if not os.path.isdir(cdir):
                build_cache(df, split, version=version, cache_dir=cache_dir)
            cache_dirs[split] = cdir
        else:
            cache_dirs[split] = None

    def _make_loader(split: str, train: bool) -> DataLoader:
        ds = FaceEmotionDataset(
            df=split_dfs[split],
            transform=get_transforms(train),
            use_score_weights=(use_score_weights if train else False),
            score_weight_power=score_weight_power,
            score_weight_min=score_weight_min,
            score_weight_normalize=score_weight_normalize,
            use_cache=use_cache,
            cache_dir=cache_dirs[split],
        )
        if train and train_sampler is not None:
            return DataLoader(
                ds,
                batch_size=batch_size,
                sampler=train_sampler,   # sampler handles ordering; shuffle must be False
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
            )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=train,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=train,
        )

    return (
        _make_loader("train", train=True),
        _make_loader("dev",   train=False),
        _make_loader("test",  train=False),
    )
