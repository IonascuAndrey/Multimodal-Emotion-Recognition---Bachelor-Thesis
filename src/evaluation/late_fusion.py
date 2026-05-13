"""
late_fusion.py
--------------
Late (decision-level) fusion of the Text, Audio, and Face unimodal classifiers
trained on MELD.

Two strategies are implemented
-------------------------------
1. Average Ensembling
   Simple arithmetic mean of all modalities' softmax probability vectors.
   Applied with equal weights (1/N each).

2. Weighted Average Ensembling
   Weighted mean with per-modality weights that sum to 1.0.
   Weights can be set manually or found automatically via exhaustive grid
   search on the dev set (maximises dev macro-F1).

All modality outputs are re-aligned to the canonical label order defined by
FACE_MELD_CLASSES before any fusion:
    0=Anger  1=Disgust  2=Fear  3=Happiness  4=Neutral  5=Sadness  6=Surprise

Usage
-----
See notebooks/30_late_fusion.ipynb for a complete walkthrough, or call
run_late_fusion() directly.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.config import (
    AUDIO_BATCH_SIZE,
    AUDIO_DEV_DIR,
    AUDIO_TEST_DIR,
    BATCH_SIZE,
    DEVICE,
    FACE_CACHE_DIR,
    FACE_MELD_CLASSES,
    FACE_TRAIN_BATCH_SIZE,
    FACE_USE_CACHE,
    MAX_AUDIO_S,
    MAX_LEN,
    SAMPLE_RATE,
    TRIM_END_S,
    TRIM_START_S,
    VERSION,
)
from src.data.audio_preprocessing import AudioCollator, AudioEmotionDataset
from src.data.face_dataset import (
    ID2LABEL as FACE_ID2LABEL,
    LABEL2ID as FACE_LABEL2ID,
    NUM_CLASSES,
    FaceEmotionDataset,
    _cache_dir_path,
    build_cache,
    get_transforms,
)
from src.models.face_model import load_face_model
from src.evaluation.plots import save_confusion_matrix
from src.training.utils import write_json


# ── Canonical label order ─────────────────────────────────────────────────────
# All per-modality probability vectors are re-ordered to this space before fusion.
_CLASSES   = FACE_MELD_CLASSES                               # 7 strings, Title-case
_N         = NUM_CLASSES                                     # 7
_CANON_L2I = {c.lower(): i for i, c in enumerate(_CLASSES)} # lowercase key → canonical idx
_CANON_I2L = {i: c for i, c in enumerate(_CLASSES)}         # canonical idx → label string

# Type aliases
ProbsMap = Dict[Tuple[int, int], np.ndarray]  # {(dialogue_id, utterance_id): softmax (7,)}
LabelMap = Dict[Tuple[int, int], int]          # {(dialogue_id, utterance_id): canonical_label}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_remap(id2label_train: Dict[int, str]) -> np.ndarray:
    """
    Build a remapping array so that canonical_probs[canonical_idx] = model_probs[model_idx].

    Parameters
    ----------
    id2label_train : {model_output_idx: emotion_string}  (from config.json)

    Returns
    -------
    remap : ndarray (N,) where  remap[model_idx] = canonical_idx
    """
    remap = np.zeros(_N, dtype=int)
    for model_idx, emotion_str in id2label_train.items():
        remap[int(model_idx)] = _CANON_L2I[str(emotion_str).lower()]
    return remap


def _apply_remap(probs: np.ndarray, remap: np.ndarray) -> np.ndarray:
    """
    Reorder a probability vector from model label space to canonical label space.

    Parameters
    ----------
    probs : (N,) array in model output order
    remap : (N,) array where remap[model_idx] = canonical_idx
    """
    out = np.zeros(_N, dtype=np.float32)
    for model_idx in range(_N):
        out[remap[model_idx]] = probs[model_idx]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Per-modality utterance-level inference
#     Each function returns:
#       probs_map : {(dialogue_id, utterance_id): canonical softmax (7,)}
#       label_map : {(dialogue_id, utterance_id): canonical label int}
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_text_utterance_probs(
    run_dir:       str,
    csv_path:      str,
    device:        str = DEVICE,
    batch_size:    int = BATCH_SIZE,
    max_len:       int = MAX_LEN,
) -> Tuple[ProbsMap, LabelMap]:
    """
    Run text model inference on a MELD CSV and return per-utterance softmax
    probabilities aligned to the canonical label order.

    Parameters
    ----------
    run_dir  : path to a train_one() run directory (contains model/ + config.json)
    csv_path : MELD CSV with columns Dialogue_ID, Utterance_ID, Utterance, Emotion

    Returns
    -------
    probs_map : {(dialogue_id, utterance_id): softmax_probs (7,)}
    label_map : {(dialogue_id, utterance_id): canonical_label_int}
    """
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    id2label_train: Dict[int, str] = {int(k): v for k, v in cfg["id2label"].items()}
    remap = _build_remap(id2label_train)

    model_path = os.path.join(run_dir, "model")
    model     = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    df = pd.read_csv(csv_path)
    df["Emotion"] = df["Emotion"].astype(str).str.lower().str.strip().replace({"joy": "happiness"})
    df = df.dropna(subset=["Utterance", "Emotion", "Dialogue_ID", "Utterance_ID"])

    texts    = df["Utterance"].tolist()
    d_ids    = df["Dialogue_ID"].astype(int).tolist()
    u_ids    = df["Utterance_ID"].astype(int).tolist()
    emotions = df["Emotion"].tolist()

    probs_map: ProbsMap = {}
    label_map: LabelMap = {}

    for start in range(0, len(texts), batch_size):
        end   = min(start + batch_size, len(texts))
        enc   = tokenizer(
            texts[start:end],
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        )
        enc    = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits                          # (B, N_train)
        probs  = torch.softmax(logits, dim=-1).cpu().numpy() # (B, N_train)

        for i in range(end - start):
            key     = (d_ids[start + i], u_ids[start + i])
            emotion = emotions[start + i]
            probs_map[key] = _apply_remap(probs[i], remap)
            if emotion in _CANON_L2I:
                label_map[key] = _CANON_L2I[emotion]

    return probs_map, label_map


@torch.no_grad()
def get_audio_utterance_probs(
    run_dir:   str,
    df:        pd.DataFrame,
    audio_dir: str = AUDIO_TEST_DIR,
    device:    str = DEVICE,
) -> Tuple[ProbsMap, LabelMap]:
    """
    Run audio model inference on *df* and return per-utterance softmax
    probabilities aligned to the canonical label order.

    Parameters
    ----------
    run_dir   : path to a train_one_audio() run directory
    df        : DataFrame from load_audio_split() — must have 'filename' and 'label'
    audio_dir : directory that contains the .wav files

    Notes
    -----
    Per-utterance keys are parsed from the filename pattern dia{D}_utt{U}.wav.
    The DataLoader uses shuffle=False so sample ordering matches df row order.
    """
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    id2label_train: Dict[int, str] = {int(k): v for k, v in cfg["id2label"].items()}
    remap = _build_remap(id2label_train)

    model_path        = os.path.join(run_dir, "model")
    model             = AutoModelForAudioClassification.from_pretrained(model_path).to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    collator          = AudioCollator(feature_extractor, SAMPLE_RATE)
    model.eval()

    loader = DataLoader(
        AudioEmotionDataset(
            df,
            audio_dir=audio_dir,
            sample_rate=SAMPLE_RATE,
            max_audio_s=MAX_AUDIO_S,
            trim_start_s=TRIM_START_S,
            trim_end_s=TRIM_END_S,
        ),
        batch_size=AUDIO_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )

    # Pre-parse (d_id, u_id) from each filename — order matches shuffle=False iteration
    _pat = re.compile(r"dia(\d+)_utt(\d+)", re.IGNORECASE)
    keys: List[Optional[Tuple[int, int]]] = []
    for fname in df["filename"].tolist():
        m = _pat.search(fname)
        keys.append((int(m.group(1)), int(m.group(2))) if m else None)

    probs_map: ProbsMap = {}
    label_map: LabelMap = {}
    sample_idx = 0

    for batch in loader:
        batch_d = {
            k: (v.to(device) if torch.is_tensor(v) else torch.tensor(v).to(device))
            for k, v in batch.items()
        }
        logits = model(**batch_d).logits
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()  # (B, N_train)
        labels = batch["labels"].cpu().numpy()                 # (B,)

        for i in range(len(labels)):
            key = keys[sample_idx]
            if key is not None:
                probs_map[key] = _apply_remap(probs[i], remap)
                label_map[key] = int(remap[int(labels[i])])   # training label → canonical
            sample_idx += 1

    return probs_map, label_map


@torch.no_grad()
def get_face_utterance_probs(
    run_dir:     str,
    df:          pd.DataFrame,
    device:      str  = DEVICE,
    batch_size:  int  = FACE_TRAIN_BATCH_SIZE,
    use_cache:   bool = FACE_USE_CACHE,
    cache_dir:   str  = FACE_CACHE_DIR,
    version:     str  = VERSION,
    num_workers: int  = 4,
) -> Tuple[ProbsMap, LabelMap]:
    """
    Run face model inference on *df* and return per-utterance softmax
    probabilities (averaged over K crops per utterance) aligned to the
    canonical label order.

    The face model is already trained in canonical label order, so no
    index remapping is applied to its outputs.

    Parameters
    ----------
    run_dir : path to a train_one_face() run directory
    df      : DataFrame from load_face_metadata("test" / "dev")
    """
    model_dir = os.path.join(run_dir, "model")
    with open(os.path.join(model_dir, "arch.txt")) as f:
        model_name = f.read().strip()

    model = load_face_model(model_name, num_classes=NUM_CLASSES, device=device)
    state = torch.load(os.path.join(model_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(state)
    model.eval()

    cdir = None
    if use_cache:
        cdir = _cache_dir_path("test", version, cache_dir)
        if not os.path.isdir(cdir):
            # Try "dev" path as fallback (caller may pass a dev df)
            cdir_dev = _cache_dir_path("dev", version, cache_dir)
            cdir = cdir_dev if os.path.isdir(cdir_dev) else cdir
            if not os.path.isdir(cdir):
                build_cache(df, "test", version=version, cache_dir=cache_dir)

    ds = FaceEmotionDataset(
        df=df,
        transform=get_transforms(train=False),
        use_score_weights=False,
        use_cache=use_cache,
        cache_dir=cdir,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    utt_probs: Dict[Tuple[int, int], List[np.ndarray]] = {}
    utt_label: LabelMap = {}

    for images, labels, _weights, d_ids, u_ids in loader:
        images = images.to(device)
        with torch.amp.autocast(device, enabled=(device == "cuda")):
            logits = model(images)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()  # (B, 7) — already canonical

        for i in range(len(labels)):
            key = (int(d_ids[i]), int(u_ids[i]))
            utt_probs.setdefault(key, []).append(probs[i])
            utt_label[key] = int(labels[i])  # already canonical

    probs_map: ProbsMap = {
        key: np.mean(utt_probs[key], axis=0)
        for key in utt_label
    }
    return probs_map, utt_label


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Fusion
# ─────────────────────────────────────────────────────────────────────────────

def fuse_probs(
    named_probs: Dict[str, ProbsMap],
    weights:     Optional[Dict[str, float]] = None,
) -> ProbsMap:
    """
    Weighted average of per-modality probability maps.

    Only utterances present in ALL modalities contribute to the output.

    Parameters
    ----------
    named_probs : {"text": probs_map, "audio": probs_map, "face": probs_map}
    weights     : {"text": w_t, "audio": w_a, "face": w_f} — must sum to 1.0.
                  None → equal weights (1 / number of modalities).

    Returns
    -------
    fused : {(dialogue_id, utterance_id): fused_probs (7,)}
    """
    modalities = list(named_probs.keys())
    n_mod      = len(modalities)
    if weights is None:
        weights = {m: 1.0 / n_mod for m in modalities}

    common_keys: set = set(named_probs[modalities[0]].keys())
    for m in modalities[1:]:
        common_keys &= set(named_probs[m].keys())

    fused: ProbsMap = {}
    for key in common_keys:
        combined = np.zeros(_N, dtype=np.float32)
        for m in modalities:
            combined += weights[m] * named_probs[m][key]
        fused[key] = combined
    return fused


def score_fusion(
    fused_probs: ProbsMap,
    label_map:   LabelMap,
    id2label:    Optional[Dict[int, str]] = None,
) -> Dict:
    """
    Compute classification metrics from a fused probability map.

    Parameters
    ----------
    fused_probs : output of fuse_probs()
    label_map   : {(d_id, u_id): canonical_label_int}
    id2label    : {0: "Anger", …, 6: "Surprise"} — defaults to _CANON_I2L

    Returns
    -------
    dict with: accuracy, macro_f1, weighted_f1, num_utterances,
               preds, labels, report, cm
    """
    if id2label is None:
        id2label = _CANON_I2L

    common_keys = sorted(set(fused_probs.keys()) & set(label_map.keys()))
    preds  = [int(np.argmax(fused_probs[k])) for k in common_keys]
    labels = [label_map[k]                    for k in common_keys]

    label_names = [id2label[i] for i in range(_N)]
    acc  = float(accuracy_score(labels, preds))
    mf1  = float(f1_score(labels, preds, average="macro",    zero_division=0))
    wf1  = float(f1_score(labels, preds, average="weighted", zero_division=0))
    report = classification_report(
        labels, preds,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=list(range(_N)))

    return {
        "accuracy":         round(acc, 6),
        "macro_f1":         round(mf1, 6),
        "weighted_f1":      round(wf1, 6),
        "num_utterances":   len(common_keys),
        "preds":            preds,
        "labels":           labels,
        "report":           report,
        "cm":               cm,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Weight optimisation (grid search on the dev set)
# ─────────────────────────────────────────────────────────────────────────────

def grid_search_weights(
    named_probs_dev: Dict[str, ProbsMap],
    label_map_dev:   LabelMap,
    n_steps:         int = 11,
    verbose:         bool = True,
) -> Dict[str, float]:
    """
    Find the modality weights that maximise dev macro-F1 by exhaustive search
    over a simplex grid.

    Parameters
    ----------
    named_probs_dev : {"text": …, "audio": …, "face": …} on the DEV set
    n_steps         : grid resolution per axis (11 → 0.0, 0.1, …, 1.0).
                      With 3 modalities this gives 66 candidate triplets.

    Returns
    -------
    best_weights : {"text": w_t, "audio": w_a, "face": w_f}
    """
    modalities  = list(named_probs_dev.keys())
    n_mod       = len(modalities)
    step        = 1.0 / (n_steps - 1)
    grid        = [round(i * step, 10) for i in range(n_steps)]

    best_f1      = -1.0
    best_weights = {m: 1.0 / n_mod for m in modalities}

    if n_mod == 2:
        m0, m1 = modalities
        for w0 in grid:
            w1 = round(1.0 - w0, 10)
            if w1 < -1e-9:
                continue
            w1 = max(w1, 0.0)
            w  = {m0: w0, m1: w1}
            mf1 = score_fusion(fuse_probs(named_probs_dev, w), label_map_dev)["macro_f1"]
            if mf1 > best_f1:
                best_f1, best_weights = mf1, w

    elif n_mod == 3:
        m0, m1, m2 = modalities
        for w0 in grid:
            for w1 in grid:
                w2 = round(1.0 - w0 - w1, 10)
                if w2 < -1e-9:
                    continue
                w2 = max(w2, 0.0)
                w  = {m0: w0, m1: w1, m2: w2}
                mf1 = score_fusion(fuse_probs(named_probs_dev, w), label_map_dev)["macro_f1"]
                if mf1 > best_f1:
                    best_f1, best_weights = mf1, w
    else:
        raise ValueError(
            f"grid_search_weights supports 2 or 3 modalities, got {n_mod}"
        )

    if verbose:
        print(f"  Grid search — best dev macro-F1 : {best_f1:.4f}")
        print(f"               best weights       : {best_weights}")
    return best_weights


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_late_fusion(
    # Unimodal run directories
    text_run_dir:  str,
    audio_run_dir: str,
    face_run_dir:  str,
    # Test data
    test_csv_path: str,
    test_audio_df: pd.DataFrame,
    test_face_df:  pd.DataFrame,
    # Dev data (needed only when do_grid_search=True)
    dev_csv_path:  Optional[str]          = None,
    dev_audio_df:  Optional[pd.DataFrame] = None,
    dev_face_df:   Optional[pd.DataFrame] = None,
    # Fusion strategy
    weights:        Optional[Dict[str, float]] = None,
    do_grid_search: bool                       = True,
    # Output
    output_dir:    Optional[str]          = None,
    # Infrastructure
    device:        str = DEVICE,
    audio_test_dir: str = AUDIO_TEST_DIR,
    audio_dev_dir:  str = AUDIO_DEV_DIR,
) -> Dict:
    """
    Full late-fusion pipeline.

    Steps
    -----
    1. Run per-modality inference on the test set (and dev set for grid search).
    2. Average ensembling  — equal weights (1/3 each).
    3. Weighted ensembling — weights from grid search on dev set, or manual.
    4. Write JSON metrics + confusion matrix PNGs to output_dir.

    Parameters
    ----------
    text_run_dir / audio_run_dir / face_run_dir
        Directories returned by the unimodal training functions.
        Each must contain  model/  and  config.json.
    test_csv_path
        MELD test CSV (Dialogue_ID, Utterance_ID, Utterance, Emotion).
    test_audio_df / test_face_df
        DataFrames for audio and face test data.
    dev_csv_path / dev_audio_df / dev_face_df
        Same for the dev set — required when do_grid_search=True.
    weights
        Explicit modality weights.  Ignored when do_grid_search=True and
        dev data is provided.  Falls back to equal weights if None and
        do_grid_search=False.
    output_dir
        Directory to write artefacts.  Created if it does not exist.

    Returns
    -------
    {
        "average_ensemble":  score_fusion result dict (equal weights),
        "weighted_ensemble": score_fusion result dict (best/manual weights),
        "weights":           weights used for the weighted ensemble,
    }
    """
    id2label = _CANON_I2L

    # ── Test inference ────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Late Fusion — test-set inference")
    print("=" * 60)

    print("\n[1/3] Text …")
    text_probs_test, text_labels_test = get_text_utterance_probs(
        text_run_dir, test_csv_path, device=device
    )
    print(f"      {len(text_probs_test)} utterances")

    print("[2/3] Audio …")
    audio_probs_test, audio_labels_test = get_audio_utterance_probs(
        audio_run_dir, test_audio_df, audio_test_dir, device=device
    )
    print(f"      {len(audio_probs_test)} utterances")

    print("[3/3] Face …")
    face_probs_test, face_labels_test = get_face_utterance_probs(
        face_run_dir, test_face_df, device=device
    )
    print(f"      {len(face_probs_test)} utterances")

    named_probs_test = {
        "text":  text_probs_test,
        "audio": audio_probs_test,
        "face":  face_probs_test,
    }
    # Use text labels as ground truth (all utterances present; face may be missing some)
    label_map_test = text_labels_test

    # ── Average ensembling ────────────────────────────────────────────────────
    print("\n── Average Ensemble (equal weights) ─────────────────────")
    fused_avg  = fuse_probs(named_probs_test, weights=None)
    result_avg = score_fusion(fused_avg, label_map_test, id2label)
    _print_metrics(result_avg, "Average Ensemble (test)")

    # ── Weighted ensembling ───────────────────────────────────────────────────
    if do_grid_search and weights is None:
        if dev_csv_path is None or dev_audio_df is None or dev_face_df is None:
            raise ValueError(
                "dev_csv_path, dev_audio_df, and dev_face_df are all required "
                "when do_grid_search=True."
            )
        print("\n── Grid Search on dev set ───────────────────────────────")
        print("  Dev-set inference …")
        text_probs_dev,  _ = get_text_utterance_probs(
            text_run_dir, dev_csv_path, device=device
        )
        audio_probs_dev, _ = get_audio_utterance_probs(
            audio_run_dir, dev_audio_df, audio_dev_dir, device=device
        )
        face_probs_dev,  face_labels_dev = get_face_utterance_probs(
            face_run_dir, dev_face_df, device=device
        )
        # Use text labels for dev ground truth as well
        text_probs_dev_tmp, text_labels_dev = get_text_utterance_probs(
            text_run_dir, dev_csv_path, device=device
        )
        named_probs_dev = {
            "text":  text_probs_dev,
            "audio": audio_probs_dev,
            "face":  face_probs_dev,
        }
        weights = grid_search_weights(named_probs_dev, text_labels_dev)

    if weights is None:
        weights = {"text": 1 / 3, "audio": 1 / 3, "face": 1 / 3}

    print(f"\n── Weighted Ensemble  weights={_fmt_weights(weights)} ──────────")
    fused_wt  = fuse_probs(named_probs_test, weights)
    result_wt = score_fusion(fused_wt, label_map_test, id2label)
    _print_metrics(result_wt, "Weighted Ensemble (test)")

    # ── Save artefacts ────────────────────────────────────────────────────────
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        label_names = [id2label[i] for i in range(_N)]

        _save_results(
            result_avg,
            prefix="avg_ensemble",
            output_dir=output_dir,
            label_names=label_names,
            title="Late Fusion — Average Ensemble (test)",
        )
        _save_results(
            result_wt,
            prefix="weighted_ensemble",
            output_dir=output_dir,
            label_names=label_names,
            title=f"Late Fusion — Weighted Ensemble (test)  w={_fmt_weights(weights)}",
        )
        write_json(os.path.join(output_dir, "weights.json"), weights)
        print(f"\nArtefacts saved to  {output_dir}")

    return {
        "average_ensemble":  result_avg,
        "weighted_ensemble": result_wt,
        "weights":           weights,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal display / IO helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_metrics(result: Dict, label: str = "") -> None:
    if label:
        print(f"  {label}")
    print(f"  Utterances  : {result['num_utterances']}")
    print(f"  Accuracy    : {result['accuracy']:.4f}")
    print(f"  Macro F1    : {result['macro_f1']:.4f}")
    print(f"  Weighted F1 : {result['weighted_f1']:.4f}")


def _fmt_weights(w: Dict[str, float]) -> str:
    return "{" + ", ".join(f"{k}:{v:.2f}" for k, v in w.items()) + "}"


def _save_results(
    result:      Dict,
    prefix:      str,
    output_dir:  str,
    label_names: List[str],
    title:       str,
) -> None:
    """Write metrics JSON, classification report JSON, and confusion matrix PNG."""
    metrics_only = {
        k: v for k, v in result.items()
        if k not in ("preds", "labels", "report", "cm")
    }
    write_json(os.path.join(output_dir, f"{prefix}_metrics.json"), metrics_only)
    write_json(os.path.join(output_dir, f"{prefix}_report.json"), result["report"])
    save_confusion_matrix(
        np.array(result["cm"]),
        label_names,
        os.path.join(output_dir, f"{prefix}_cm.png"),
        title=title,
    )
