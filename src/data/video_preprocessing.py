"""
video_preprocessing.py
-----------------------
Utilities for extracting, cropping, and caching video clips for the
video-modality emotion-recognition pipeline.

Frame cache layout
------------------
    <VIDEO_CACHE_DIR>/npy/<version>/<split>/dia{D}_utt{U}.npy
    dtype  : uint8
    shape  : (VIDEO_T_CACHE, VIDEO_IMG_SIZE, VIDEO_IMG_SIZE, 3)

One file per utterance, T=VIDEO_T_CACHE frames sampled uniformly.
Models that need fewer frames (e.g. S3D with T=8) subsample at load time.

Face-crop mode
--------------
Re-uses the face bounding boxes already stored in the face-preprocessing
analysis NPZs:
    <FACE_OUTPUT_ROOT>/<version>/<split>/analysis/dia{D}_utt{U}.npz
    keys: frame_idxs (F,), bboxes (F,4) — format [x1, y1, x2, y2]

For each sampled frame, the nearest detected box (by frame index) within
VIDEO_FACE_BOX_MAX_GAP frames is used.  If none exists, the full frame is
used instead.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None
    F = None
    _CUDA_AVAILABLE = False

try:
    import decord
    decord.gpu(0) if _CUDA_AVAILABLE else None
    _DECORD_AVAILABLE = True
except (ImportError, Exception):
    decord = None
    _DECORD_AVAILABLE = False

from src.config import (
    DEV_CSV,
    FACE_OUTPUT_ROOT,
    FACE_VIDEO_DIRS,
    LABEL_REMAP,
    FACE_MELD_CLASSES,
    TEST_CSV,
    TRAIN_CSV,
    VERSION,
    VIDEO_CACHE_DIR,
    VIDEO_CROP_MODE,
    VIDEO_FACE_BOX_MAX_GAP,
    VIDEO_IMG_SIZE,
    VIDEO_T_CACHE,
)

# ── Label map (shared with face modality) ─────────────────────────────────────
_LABEL2ID: Dict[str, int] = {cls.lower(): i for i, cls in enumerate(FACE_MELD_CLASSES)}

_SPLIT_CSV = {"train": TRAIN_CSV, "dev": DEV_CSV, "test": TEST_CSV}


# ── Metadata loading ──────────────────────────────────────────────────────────

def load_video_metadata(split: str, version: str = VERSION) -> pd.DataFrame:
    """
    Reads the MELD CSV for *split*, normalises emotion labels, and adds:
        dialogue_id, utterance_id, emotion_norm, label_id,
        video_path, video_exists

    Rows whose video file does not exist or whose emotion is not in
    FACE_MELD_CLASSES are kept in the DataFrame but flagged via
    video_exists=False / missing label_id so callers can filter as needed.
    """
    csv_path  = _SPLIT_CSV[split]
    video_dir = FACE_VIDEO_DIRS[split]

    df = pd.read_csv(csv_path)

    # Normalise column names
    df = df.rename(columns={
        "Dialogue_ID":  "dialogue_id",
        "Utterance_ID": "utterance_id",
        "Emotion":      "emotion",
    })

    # Normalise emotion labels
    df["emotion_norm"] = (
        df["emotion"].str.lower()
                     .str.strip()
                     .replace(LABEL_REMAP)
    )

    # Video paths
    df["video_path"] = df.apply(
        lambda r: os.path.join(
            video_dir,
            f"dia{int(r.dialogue_id)}_utt{int(r.utterance_id)}.mp4"
        ),
        axis=1,
    )
    df["video_exists"] = df["video_path"].apply(os.path.isfile)

    # Label IDs (only for known emotions)
    mask = df["emotion_norm"].isin(_LABEL2ID)
    df.loc[mask, "label_id"] = (
        df.loc[mask, "emotion_norm"].map(_LABEL2ID).astype(int)
    )

    df = df.reset_index(drop=True)
    return df


# ── Face bounding-box helpers ─────────────────────────────────────────────────

def load_analysis_boxes(
    split: str,
    d_id: int,
    u_id: int,
    version: str = VERSION,
) -> Dict[int, Tuple[int, int, int, int]]:
    """
    Loads the face bounding boxes saved by the face-preprocessing pipeline
    for utterance (d_id, u_id) in *split*.

    Returns a dict  {frame_idx: (x1, y1, x2, y2)}.
    Returns {} if the analysis NPZ does not exist.
    """
    npz_path = os.path.join(
        FACE_OUTPUT_ROOT, version, split, "analysis",
        f"dia{d_id}_utt{u_id}.npz",
    )
    if not os.path.isfile(npz_path):
        return {}

    data = np.load(npz_path, allow_pickle=False)
    frame_idxs = data["frame_idxs"].astype(int)   # (F,)
    bboxes     = data["bboxes"].astype(int)        # (F, 4)  [x1, y1, x2, y2]

    return {int(fi): tuple(bboxes[i].tolist()) for i, fi in enumerate(frame_idxs)}


def get_box_for_frame(
    frame_idx: int,
    box_map: Dict[int, Tuple],
    max_gap: int = VIDEO_FACE_BOX_MAX_GAP,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Returns the bounding box for *frame_idx* from *box_map*.

    Exact match is preferred.  Otherwise the nearest frame index within
    *max_gap* frames is used.  Returns None if no suitable box is found.
    """
    if not box_map:
        return None
    if frame_idx in box_map:
        return box_map[frame_idx]
    nearest = min(box_map.keys(), key=lambda k: abs(k - frame_idx))
    if abs(nearest - frame_idx) <= max_gap:
        return box_map[nearest]
    return None


# ── Frame extraction ──────────────────────────────────────────────────────────

def sample_clip_frames(
    video_path: str,
    T: int,
    use_gpu: bool = _CUDA_AVAILABLE,
) -> List[Tuple[int, np.ndarray]]:
    """
    Uniformly samples *T* frames from the video at *video_path*.
    
    GPU acceleration: uses Decord on GPU if available (2-3x faster than OpenCV).
    Falls back to OpenCV if Decord is unavailable.

    Returns a list of (frame_index, rgb_array) tuples.
    Returns [] if the video cannot be opened or has no frames.
    """
    # Try GPU-accelerated Decord first
    if use_gpu and _DECORD_AVAILABLE and decord is not None:
        try:
            vr = decord.VideoReader(video_path, ctx=decord.gpu(0))
            total = len(vr)
            if total <= 0:
                return []
            
            indices = np.linspace(0, total - 1, T, dtype=int)
            vr.seek(0)
            frames: List[Tuple[int, np.ndarray]] = []
            
            for idx in indices:
                frame_tensor = vr[int(idx)]
                rgb = frame_tensor.asnumpy()
                frames.append((int(idx), rgb))
            
            return frames
        except Exception:
            pass
    
    # Fallback to CPU-based OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, T, dtype=int)
    frames: List[Tuple[int, np.ndarray]] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((int(idx), rgb))

    cap.release()
    return frames


def crop_or_resize(
    frame_rgb: np.ndarray,
    box: Optional[Tuple[int, int, int, int]],
    img_size: int = VIDEO_IMG_SIZE,
    use_gpu: bool = _CUDA_AVAILABLE,
) -> np.ndarray:
    """
    If *box* is provided, crops the frame to the face region then resizes
    to (img_size, img_size).  Otherwise resizes the full frame.

    Returns a (img_size, img_size, 3) uint8 array.
    
    GPU acceleration: uses torch on GPU if available (30-50% faster).
    """
    if box is not None:
        x1, y1, x2, y2 = box
        h, w = frame_rgb.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            frame_rgb = frame_rgb[y1:y2, x1:x2]

    # GPU acceleration with torch
    if use_gpu and torch is not None and _CUDA_AVAILABLE:
        # Convert to torch tensor on GPU
        frame_t = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().cuda()  # (3, H, W)
        frame_t = frame_t.unsqueeze(0)  # (1, 3, H, W) for batch
        
        # Resize using bilinear interpolation on GPU
        resized = F.interpolate(
            frame_t,
            size=(img_size, img_size),
            mode='bilinear',
            align_corners=False
        )
        
        # Convert back to CPU numpy
        result = (resized.squeeze(0).permute(1, 2, 0).cpu().clamp(0, 255).byte().numpy())
        return result
    else:
        # Fallback to CPU PIL
        pil = Image.fromarray(frame_rgb).resize(
            (img_size, img_size), Image.BILINEAR
        )
        return np.array(pil, dtype=np.uint8)


# ── Frame cache ───────────────────────────────────────────────────────────────

def _frame_cache_dir(
    split: str,
    version: str,
    cache_dir: str,
) -> str:
    """Returns the directory holding per-utterance .npy files for *split*."""
    return os.path.join(cache_dir, "npy", version, split)


def build_frame_cache(
    split: str,
    version: str          = VERSION,
    cache_dir: str        = VIDEO_CACHE_DIR,
    crop_mode: str        = VIDEO_CROP_MODE,
    img_size: int         = VIDEO_IMG_SIZE,
    T_cache: int          = VIDEO_T_CACHE,
    face_box_max_gap: int = VIDEO_FACE_BOX_MAX_GAP,
    force: bool           = False,
    use_gpu: bool         = _CUDA_AVAILABLE,
) -> str:
    """
    Builds the per-utterance frame cache for *split*.

    For each utterance, samples T_cache frames uniformly, optionally crops
    to the speaker's face region, resizes to img_size × img_size, and saves
    as a (T_cache, img_size, img_size, 3) uint8 .npy file.

    GPU acceleration:
      - Frame decoding: Decord (GPU) if available, else OpenCV (CPU)
      - Frame resizing: PyTorch (GPU) if available, else PIL (CPU)

    RAM usage : O(1) — one utterance decoded and written at a time.
    Resumable : already-written files are skipped (unless force=True).

    Returns the cache directory path.
    """
    out_dir = _frame_cache_dir(split, version, cache_dir)
    os.makedirs(out_dir, exist_ok=True)

    meta = load_video_metadata(split, version)
    meta = meta[meta["video_exists"] & meta["emotion_norm"].isin(_LABEL2ID)]

    pending = [
        row for _, row in meta.iterrows()
        if force or not os.path.isfile(
            os.path.join(
                out_dir,
                f"dia{int(row.dialogue_id)}_utt{int(row.utterance_id)}.npy"
            )
        )
    ]

    if not pending:
        print(f"[video_cache] {split} already fully cached "
              f"({len(meta)} utterances) → {out_dir}")
        return out_dir

    gpu_info = f" [GPU: {'ON' if use_gpu and _CUDA_AVAILABLE else 'OFF'}]" if use_gpu else ""
    print(f"[video_cache] Building {split} frame cache → {out_dir}{gpu_info}")
    print(f"  {len(pending)} remaining / {len(meta)} total | "
          f"T={T_cache}, size={img_size}, crop={crop_mode}")

    failed = written = 0

    for row in pending:
        d_id = int(row.dialogue_id)
        u_id = int(row.utterance_id)
        npy_path = os.path.join(out_dir, f"dia{d_id}_utt{u_id}.npy")

        try:
            raw_frames = sample_clip_frames(row.video_path, T_cache, use_gpu=use_gpu)
            if not raw_frames:
                raise ValueError("No frames decoded")

            # Load face boxes if crop_mode == "face"
            box_map = {}
            if crop_mode == "face":
                box_map = load_analysis_boxes(split, d_id, u_id, version)

            clip = np.zeros(
                (T_cache, img_size, img_size, 3), dtype=np.uint8
            )
            for t, (fi, rgb) in enumerate(raw_frames[:T_cache]):
                box = get_box_for_frame(fi, box_map, face_box_max_gap) \
                      if crop_mode == "face" else None
                clip[t] = crop_or_resize(rgb, box, img_size, use_gpu=use_gpu)

            # Pad last frame if fewer frames were decoded
            for t in range(len(raw_frames), T_cache):
                clip[t] = clip[len(raw_frames) - 1]

            np.save(npy_path, clip)
            written += 1

        except Exception as exc:
            print(f"  [warn] dia{d_id}_utt{u_id}: {exc}")
            failed += 1

    print(f"[video_cache] Done. {written} written, {failed} failed.")
    return out_dir
