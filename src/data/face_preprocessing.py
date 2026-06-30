"""
Face preprocessing pipeline for MELD video clips.

For each utterance the pipeline:
  1. Samples frames at a fixed stride.
  2. Detects all faces with MTCNN.
  3. Selects the speaker's face (largest face; disambiguated by InceptionResnetV1
     speaker-embedding similarity when multiple faces are present).
  4. Runs EmotiEffLib FER to score each frame:
         score = gt_prob_renorm × max_prob_renorm
  5. Applies a short Gaussian smoothing window over the score curve.
  6. Picks K frames via scipy peak detection (greedy fallback if too few peaks).
  7. Saves the selected face crops as PNG files.
  8. Saves a per-utterance .npz with the full per-frame analysis.
"""

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from tqdm import tqdm

from src.config import (
    DEVICE,
    FACE_AFEW8_TO_MELD7,
    FACE_EMBEDDING_EMA,
    FACE_EXPAND_SCALE,
    FACE_FER_MODEL,
    FACE_FRAME_STRIDE,
    FACE_IDX_TO_CLASS_8,
    FACE_K_FRAMES,
    FACE_MELD_CLASSES,
    FACE_MIN_FACE_PX,
    FACE_MIN_GAP_S,
    FACE_OUTPUT_ROOT,
    FACE_SAVE_EVERY,
    FACE_SMOOTH_SIGMA_S,
    LABEL_COL,
    LABEL_REMAP,
    VERSION,
)

# Lazy import to avoid hard dependency at module load time
try:
    from emotiefflib.facial_analysis import EmotiEffLibRecognizer
except ImportError:
    EmotiEffLibRecognizer = None  # type: ignore

# ── Label helpers ─────────────────────────────────────────────────────────────

_MELD_CLASS_TO_IDX7: Dict[str, int] = {c: i for i, c in enumerate(FACE_MELD_CLASSES)}

# Capitalised mapping used to normalise CSV emotion strings
_NORMALISE_MAP = {
    "anger":    "Anger",
    "disgust":  "Disgust",
    "fear":     "Fear",
    "happiness": "Happiness",
    "joy":      "Happiness",   # MELD variant
    "neutral":  "Neutral",
    "sadness":  "Sadness",
    "surprise": "Surprise",
}


def normalise_emotion(raw: str) -> str:
    """Lower-case, strip, remap 'joy' → 'Happiness', return Title-cased label."""
    key = str(raw).strip().lower()
    if key not in _NORMALISE_MAP:
        raise ValueError(f"Unknown emotion label: '{raw}'")
    return _NORMALISE_MAP[key]


# ── 8-class → 7-class remapping ───────────────────────────────────────────────

def remap_8_to_7(probs_8: np.ndarray) -> np.ndarray:
    """
    Drop Contempt (index 1 in the AFEW model) and renormalise to 7 MELD classes.

    Args:
        probs_8: Probability array of shape (8,).

    Returns:
        Probability array of shape (7,) summing to 1.
    """
    probs_7 = probs_8[FACE_AFEW8_TO_MELD7]          # (7,)
    total   = probs_7.sum()
    return probs_7 / total if total > 0 else probs_7


# ── Face-box helpers ──────────────────────────────────────────────────────────

def pick_largest_box(boxes_xyxy: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Return the bounding box with the largest area from an (N, 4) array.
    Returns None if boxes_xyxy is None or empty.
    """
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return None
    areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
    return boxes_xyxy[int(np.argmax(areas))]


def expand_and_clip_box(
    box: np.ndarray,
    img_w: int,
    img_h: int,
    scale: float = FACE_EXPAND_SCALE,
) -> Tuple[int, int, int, int]:
    """
    Expand a bounding box by *scale* around its centre and clip to image bounds.

    Returns:
        (x1, y1, x2, y2) as integers, or (0, 0, 0, 0) if the result is degenerate.
    """
    x1, y1, x2, y2 = box.astype(float)
    bw, bh = x2 - x1, y2 - y1
    cx, cy = x1 + bw / 2.0, y1 + bh / 2.0
    bw2, bh2 = bw * scale, bh * scale

    nx1 = int(max(0, math.floor(cx - bw2 / 2.0)))
    ny1 = int(max(0, math.floor(cy - bh2 / 2.0)))
    nx2 = int(min(img_w, math.ceil(cx + bw2 / 2.0)))
    ny2 = int(min(img_h, math.ceil(cy + bh2 / 2.0)))

    if nx2 <= nx1 or ny2 <= ny1:
        return (0, 0, 0, 0)
    return nx1, ny1, nx2, ny2


# ── Speaker embedding tracking ────────────────────────────────────────────────

class SpeakerEmbeddingTracker:
    """
    Maintains one running-mean face embedding per speaker name.

    Uses cosine similarity (dot product of L2-normalised vectors) to match
    an incoming set of face embeddings to a known speaker. Updates the
    reference embedding via exponential moving average after each match.

    When a speaker is seen for the first time (no reference yet), the
    largest face (index 0 in the sorted list) is chosen as the default.
    """

    def __init__(self, ema_alpha: float = FACE_EMBEDDING_EMA):
        self.ema_alpha  = ema_alpha
        self._embeddings: Dict[str, np.ndarray] = {}

    def match(self, speaker: str, embeddings: np.ndarray) -> int:
        """
        Return the index of the face most similar to *speaker*'s reference.

        Args:
            speaker:    Speaker name from the CSV.
            embeddings: (N, 512) L2-normalised face embeddings, sorted
                        largest-face-first.

        Returns:
            Index into *embeddings* (0 = largest face if no prior reference).
        """
        if speaker not in self._embeddings or len(embeddings) == 1:
            return 0  # default: largest face

        ref  = self._embeddings[speaker]        # (512,)
        sims = embeddings @ ref                  # (N,) cosine similarity
        return int(np.argmax(sims))

    def update(self, speaker: str, embedding: np.ndarray):
        """
        Update the reference embedding for *speaker* using EMA.
        Re-normalises after blending to keep the reference on the unit sphere.
        """
        if speaker not in self._embeddings:
            self._embeddings[speaker] = embedding.copy()
        else:
            blended = self.ema_alpha * self._embeddings[speaker] + (1.0 - self.ema_alpha) * embedding
            norm    = np.linalg.norm(blended)
            self._embeddings[speaker] = blended / norm if norm > 0 else blended

    def has(self, speaker: str) -> bool:
        return speaker in self._embeddings


def get_face_embedding(
    pil_img: Image.Image,
    box: np.ndarray,
    resnet: InceptionResnetV1,
    device: str,
) -> np.ndarray:
    """
    Crop *box* from *pil_img*, resize to 160×160, and return an L2-normalised
    512-dim embedding from InceptionResnetV1.

    Returns a zero vector on failure (too-small crop, etc.).
    """
    x1, y1, x2, y2 = [max(0, int(b)) for b in box]
    face_pil = pil_img.crop((x1, y1, x2, y2))
    if face_pil.width < 1 or face_pil.height < 1:
        return np.zeros(512, dtype=np.float32)

    face_pil  = face_pil.resize((160, 160), Image.BILINEAR)
    face_np   = np.array(face_pil, dtype=np.float32)
    face_np   = (face_np - 127.5) / 128.0               # normalise to [-1, 1]
    face_t    = torch.from_numpy(face_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = resnet(face_t).cpu().numpy()[0]             # (512,)

    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


# ── Frame selection ───────────────────────────────────────────────────────────

def smooth_scores(
    scores: np.ndarray,
    frame_idxs: List[int],
    fps: float,
    sigma_s: float = FACE_SMOOTH_SIGMA_S,
) -> np.ndarray:
    """
    Apply a Gaussian kernel to *scores* with width *sigma_s* seconds.
    The kernel width is converted to frames using *fps*.

    Returns the smoothed score array (same length as *scores*).
    """
    sigma_frames = max(0.5, sigma_s * fps / FACE_FRAME_STRIDE)
    return gaussian_filter1d(scores.astype(np.float64), sigma=sigma_frames).astype(np.float32)


def select_frames(
    frame_idxs: List[int],
    scores_smooth: np.ndarray,
    k: int,
    min_gap_frames: int,
) -> List[int]:
    """
    Select up to *k* array indices from *frame_idxs* using peak detection.

    Strategy:
      1. Run scipy find_peaks on *scores_smooth* with *min_gap_frames* distance.
      2. Take the top-k peaks by score.
      3. If fewer than k peaks, fill greedily (descending score, gap-enforced).
      4. If still fewer than k, fill without gap constraint.

    Returns:
        List of indices into *frame_idxs* / *scores_smooth*.
    """
    n = len(frame_idxs)
    if n <= k:
        return list(range(n))

    # ── Step 1: peak detection ────────────────────────────────────────────────
    peaks, _ = find_peaks(
        scores_smooth,
        distance=min_gap_frames,
        prominence=1e-3,
    )

    # Sort peaks by descending score
    if len(peaks) > 0:
        peaks = peaks[np.argsort(-scores_smooth[peaks])]

    selected: List[int] = list(peaks[:k].tolist())

    # ── Step 2: greedy fill (gap-enforced) ───────────────────────────────────
    if len(selected) < k:
        order = np.argsort(-scores_smooth)
        for idx in order:
            idx = int(idx)
            if idx in selected:
                continue
            fi = frame_idxs[idx]
            if all(abs(fi - frame_idxs[s]) >= min_gap_frames for s in selected):
                selected.append(idx)
                if len(selected) == k:
                    break

    # ── Step 3: fill without gap constraint ──────────────────────────────────
    if len(selected) < k:
        order = np.argsort(-scores_smooth)
        for idx in order:
            if int(idx) not in selected:
                selected.append(int(idx))
                if len(selected) == k:
                    break

    return selected


# ── Per-utterance processing ──────────────────────────────────────────────────

def process_utterance(
    d_id: int,
    u_id: int,
    emotion_norm: str,
    speaker: Optional[str],
    video_path: str,
    image_out_dir: str,
    analysis_out_dir: str,
    split: str,
    mtcnn: MTCNN,
    resnet: InceptionResnetV1,
    fer: "EmotiEffLibRecognizer",
    speaker_tracker: SpeakerEmbeddingTracker,
    k_frames: int           = FACE_K_FRAMES,
    frame_stride: int       = FACE_FRAME_STRIDE,
    min_gap_s: float        = FACE_MIN_GAP_S,
    expand_scale: float     = FACE_EXPAND_SCALE,
    min_face_px: int        = FACE_MIN_FACE_PX,
    smooth_sigma_s: float   = FACE_SMOOTH_SIGMA_S,
) -> List[Dict[str, Any]]:
    """
    Process a single utterance video and return metadata rows.

    Returns a list of dicts — one per saved frame (status='ok'), or a single
    dict with status='missing_video' / 'no_faces_detected' / 'error:<msg>'.
    """
    # ── Open video ────────────────────────────────────────────────────────────
    cap         = cv2.VideoCapture(video_path)
    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    min_gap_frames = max(1, int(round(fps * min_gap_s)))

    emo_idx = _MELD_CLASS_TO_IDX7[emotion_norm]

    # Accumulators (per-frame data collected during first pass)
    frame_idxs:   List[int]   = []
    times_s:      List[float] = []
    scores_raw:   List[float] = []   # gt_prob_renorm
    scores_comb:  List[float] = []   # gt_prob_renorm * max_prob_renorm
    probs7_list:  List[np.ndarray] = []
    pred_labels:  List[str]   = []
    pred_probs:   List[float] = []
    bboxes:       List[Tuple[int, int, int, int]] = []

    frame_i = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_stride > 1 and (frame_i % frame_stride != 0):
                frame_i += 1
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img_h, img_w = frame_rgb.shape[:2]
            pil_img = Image.fromarray(frame_rgb)

            # ── Detect all faces ──────────────────────────────────────────────
            boxes, confs = mtcnn.detect(pil_img)

            if boxes is None or len(boxes) == 0:
                frame_i += 1
                continue

            # Sort boxes by area descending (largest = most prominent face first)
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            sort_order = np.argsort(-areas)
            boxes = boxes[sort_order]

            # ── Speaker-aware face selection ──────────────────────────────────
            if speaker is not None and len(boxes) > 1:
                # Compute embeddings for all detected faces
                embeddings = np.stack([
                    get_face_embedding(pil_img, b, resnet, DEVICE)
                    for b in boxes
                ])                                           # (N, 512)
                chosen_idx = speaker_tracker.match(speaker, embeddings)
                speaker_tracker.update(speaker, embeddings[chosen_idx])
            else:
                chosen_idx = 0   # largest face

            chosen_box = boxes[chosen_idx]

            # ── Expand and quality-check ──────────────────────────────────────
            x1, y1, x2, y2 = expand_and_clip_box(chosen_box, img_w, img_h, expand_scale)
            if (x2 - x1) < min_face_px or (y2 - y1) < min_face_px:
                frame_i += 1
                continue

            face_rgb = frame_rgb[y1:y2, x1:x2]

            # ── FER inference ─────────────────────────────────────────────────
            # fer.predict_emotions returns (list_of_labels, ndarray (N, n_classes))
            _, probs_raw = fer.predict_emotions(face_rgb, logits=False)
            probs_8 = probs_raw[0].astype(np.float32)       # (8,)
            probs_7 = remap_8_to_7(probs_8)                 # (7,)

            gt_prob   = float(probs_7[emo_idx])
            max_prob  = float(probs_7.max())
            pred_idx  = int(np.argmax(probs_7))
            pred_lbl  = FACE_MELD_CLASSES[pred_idx]
            score_raw  = gt_prob
            score_comb = gt_prob * max_prob

            frame_idxs.append(frame_i)
            times_s.append(frame_i / fps)
            scores_raw.append(score_raw)
            scores_comb.append(score_comb)
            probs7_list.append(probs_7)
            pred_labels.append(pred_lbl)
            pred_probs.append(max_prob)
            bboxes.append((x1, y1, x2, y2))

            frame_i += 1

    finally:
        cap.release()

    # ── No faces found ────────────────────────────────────────────────────────
    if not frame_idxs:
        return [{
            "split": split, "dialogue_id": d_id, "utterance_id": u_id,
            "emotion_norm": emotion_norm, "video_path": video_path,
            "fps": fps, "total_frames": total_frames,
            "status": "no_faces_detected",
        }]

    # ── Save per-utterance analysis NPZ ──────────────────────────────────────
    scores_smooth = smooth_scores(
        np.array(scores_comb, dtype=np.float32), frame_idxs, fps, smooth_sigma_s
    )

    npz_path = os.path.join(analysis_out_dir, f"dia{d_id}_utt{u_id}.npz")
    np.savez_compressed(
        npz_path,
        frame_idxs   = np.array(frame_idxs,  dtype=np.int32),
        times_s      = np.array(times_s,      dtype=np.float32),
        scores_raw   = np.array(scores_raw,   dtype=np.float32),
        scores_comb  = np.array(scores_comb,  dtype=np.float32),
        scores_smooth= scores_smooth,
        probs7       = np.stack(probs7_list,  dtype=np.float32),  # (F, 7)
        pred_labels  = np.array(pred_labels,  dtype="S"),
        pred_probs   = np.array(pred_probs,   dtype=np.float32),
        bboxes       = np.array(bboxes,       dtype=np.int32),
        emotion_norm = np.array([emotion_norm.encode()]),
        fps          = np.array([fps],        dtype=np.float32),
    )

    # ── Select K frames ───────────────────────────────────────────────────────
    picked_indices = select_frames(frame_idxs, scores_smooth, k_frames, min_gap_frames)

    # ── Extract and save selected crops (second video pass) ──────────────────
    cap2 = cv2.VideoCapture(video_path)
    rows_out: List[Dict[str, Any]] = []

    try:
        for rank, idx in enumerate(picked_indices):
            fi              = frame_idxs[idx]
            x1, y1, x2, y2 = bboxes[idx]

            cap2.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame_bgr = cap2.read()
            if not ok:
                continue

            face_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)[y1:y2, x1:x2]
            face_pil = Image.fromarray(face_rgb)

            score_milli = int(round(scores_comb[idx] * 1000))
            img_name    = f"dia{d_id}_utt{u_id}_r{rank}_f{fi:05d}_sc{score_milli:04d}.png"
            img_path    = os.path.join(image_out_dir, img_name)
            face_pil.save(img_path, format="PNG")

            rows_out.append({
                "split":          split,
                "dialogue_id":    d_id,
                "utterance_id":   u_id,
                "speaker":        speaker,
                "emotion_norm":   emotion_norm,
                "video_path":     video_path,
                "fps":            fps,
                "total_frames":   total_frames,
                "frame_idx":      fi,
                "time_s":         times_s[idx],
                "selection_rank": rank,
                "score_raw":      scores_raw[idx],
                "score_comb":     scores_comb[idx],
                "score_smooth":   float(scores_smooth[idx]),
                "pred_label":     pred_labels[idx],
                "pred_prob":      pred_probs[idx],
                "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                "image_path":     img_path,
                "analysis_npz":   npz_path,
                "status":         "ok",
            })
    finally:
        cap2.release()

    return rows_out


# ── Main entry point ──────────────────────────────────────────────────────────

def run_all_splits(
    splits: Optional[List[str]] = None,
    version: str                = VERSION,
    k_frames: int               = FACE_K_FRAMES,
    frame_stride: int           = FACE_FRAME_STRIDE,
    min_gap_s: float            = FACE_MIN_GAP_S,
    expand_scale: float         = FACE_EXPAND_SCALE,
    min_face_px: int            = FACE_MIN_FACE_PX,
    smooth_sigma_s: float       = FACE_SMOOTH_SIGMA_S,
    max_utterances: Optional[int] = None,
    save_every: int             = FACE_SAVE_EVERY,
) -> Dict[str, str]:
    """
    Run the full face preprocessing pipeline for *splits* (default: all three).

    Each split gets its own subdirectory under FACE_OUTPUT_ROOT / version:
        <output_root>/<version>/<split>/images/
        <output_root>/<version>/<split>/analysis/
        <output_root>/<version>/<split>/metadata.csv

    The speaker embedding tracker is reset between splits so speakers from
    different splits don't interfere (though it would likely not hurt).

    Args:
        splits:           List of split names to process, e.g. ['train', 'dev', 'test'].
                          Defaults to all three.
        version:          Output version string (creates a new folder).
        k_frames:         Frames to save per utterance.
        frame_stride:     Sample every Nth frame.
        min_gap_s:        Minimum seconds between selected frames.
        expand_scale:     Face box expansion factor.
        min_face_px:      Minimum face dimension (px) after expansion.
        smooth_sigma_s:   Gaussian smoothing window (seconds).
        max_utterances:   Cap for debugging (None = process all).
        save_every:       Flush metadata CSV every N utterances.

    Returns:
        Dict mapping split name → metadata CSV path.
    """
    from src.config import FACE_CSV_PATHS, FACE_VIDEO_DIRS

    if splits is None:
        splits = ["train", "dev", "test"]

    if EmotiEffLibRecognizer is None:
        raise ImportError("emotiefflib is not installed. Run: pip install 'emotiefflib[torch]'")

    # ── Load models once (shared across all splits) ───────────────────────────
    print("Loading MTCNN …")
    mtcnn  = MTCNN(keep_all=True, device=DEVICE)

    print("Loading InceptionResnetV1 (VGGFace2) …")
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

    print(f"Loading FER model: {FACE_FER_MODEL} …")
    fer    = EmotiEffLibRecognizer(engine="torch", model_name=FACE_FER_MODEL, device=DEVICE)

    metadata_paths: Dict[str, str] = {}

    for split in splits:
        print(f"\n{'='*60}")
        print(f"  Processing split: {split}")
        print(f"{'='*60}")

        csv_path   = FACE_CSV_PATHS[split]
        video_dir  = FACE_VIDEO_DIRS[split]

        # ── Output directories ────────────────────────────────────────────────
        split_root   = os.path.join(FACE_OUTPUT_ROOT, version, split)
        image_out    = os.path.join(split_root, "images")
        analysis_out = os.path.join(split_root, "analysis")
        metadata_csv = os.path.join(split_root, "metadata.csv")
        os.makedirs(image_out,    exist_ok=True)
        os.makedirs(analysis_out, exist_ok=True)

        metadata_paths[split] = metadata_csv

        # ── Load CSV ──────────────────────────────────────────────────────────
        df = pd.read_csv(csv_path)

        # Robust column resolution
        cols_lower = {c.lower(): c for c in df.columns}
        col_d = cols_lower.get("dialogue_id",  cols_lower.get("dialogue_id"))
        col_u = cols_lower.get("utterance_id", cols_lower.get("utterance_id"))
        col_e = cols_lower.get("emotion",      None)
        col_s = cols_lower.get("speaker",      None)

        if any(c is None for c in [col_d, col_u, col_e]):
            raise ValueError(f"[{split}] CSV missing Dialogue_ID / Utterance_ID / Emotion column.")

        df["emotion_norm"] = df[col_e].apply(normalise_emotion)

        # ── Resume support: skip already-processed utterances ─────────────────
        processed_keys = set()
        if os.path.exists(metadata_csv):
            old_df = pd.read_csv(metadata_csv)
            if "status" in old_df.columns:
                processed_keys = set(
                    zip(old_df["dialogue_id"].astype(int), old_df["utterance_id"].astype(int))
                )
            print(f"  [resume] {len(processed_keys)} utterances already done — skipping.")

        # ── Per-split speaker tracker (fresh for each split) ──────────────────
        speaker_tracker = SpeakerEmbeddingTracker(ema_alpha=FACE_EMBEDDING_EMA)

        rows_buffer: List[Dict] = []
        n_done = 0

        def _flush(rows: List[Dict], csv_path: str):
            if not rows:
                return
            out = pd.DataFrame(rows)
            header = not os.path.exists(csv_path)
            out.to_csv(csv_path, mode="a", header=header, index=False)
            rows.clear()

        rows = list(df.itertuples(index=False))
        if max_utterances is not None:
            rows = rows[:max_utterances]

        try:
            for r in tqdm(rows, desc=split):
                d_id = int(getattr(r, col_d))
                u_id = int(getattr(r, col_u))

                if (d_id, u_id) in processed_keys:
                    continue

                emo_norm = getattr(r, "emotion_norm")
                speaker  = str(getattr(r, col_s)) if col_s else None

                # Consistent filename across all splits: dia{d}_utt{u}.mp4
                video_path = os.path.join(video_dir, f"dia{d_id}_utt{u_id}.mp4")

                if not os.path.exists(video_path):
                    rows_buffer.append({
                        "split": split, "dialogue_id": d_id, "utterance_id": u_id,
                        "emotion_norm": emo_norm, "speaker": speaker,
                        "video_path": video_path, "status": "missing_video",
                    })
                    processed_keys.add((d_id, u_id))
                    n_done += 1
                    continue

                try:
                    result_rows = process_utterance(
                        d_id=d_id, u_id=u_id,
                        emotion_norm=emo_norm, speaker=speaker,
                        video_path=video_path,
                        image_out_dir=image_out,
                        analysis_out_dir=analysis_out,
                        split=split,
                        mtcnn=mtcnn, resnet=resnet, fer=fer,
                        speaker_tracker=speaker_tracker,
                        k_frames=k_frames, frame_stride=frame_stride,
                        min_gap_s=min_gap_s, expand_scale=expand_scale,
                        min_face_px=min_face_px, smooth_sigma_s=smooth_sigma_s,
                    )
                    rows_buffer.extend(result_rows)

                except Exception as exc:
                    rows_buffer.append({
                        "split": split, "dialogue_id": d_id, "utterance_id": u_id,
                        "emotion_norm": emo_norm, "speaker": speaker,
                        "video_path": video_path, "status": f"error:{exc}",
                    })

                processed_keys.add((d_id, u_id))
                n_done += 1

                # Free GPU memory accumulation after each utterance
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if n_done % save_every == 0:
                    _flush(rows_buffer, metadata_csv)

        finally:
            # Always flush whatever is buffered — handles KeyboardInterrupt and crashes
            _flush(rows_buffer, metadata_csv)
            print(f"\n  [{split}] Flushed. Metadata → {metadata_csv}")

    # Final memory cleanup
    del mtcnn, resnet, fer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metadata_paths
