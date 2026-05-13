"""
evaluate_video.py
-----------------
Loads the best checkpoint for a video model run and performs utterance-level
evaluation on the test set.

Utterance-level aggregation
----------------------------
For each utterance, multiple clips may be present in the DataLoader (if the
clip-sampling produces multiple windows, or simply all frames are batched
together).  Softmax probabilities are averaged across all clips/batches for
the same (dialogue_id, utterance_id), then argmax gives the prediction.
This mirrors the strategy in train_video.py.

Outputs (written to run_dir)
-----------------------------
classification_report_test.json
confusion_matrix_test.png
eval_metrics.json
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from src.config import (
    DEVICE,
    VIDEO_IMG_SIZE,
    VIDEO_RANDOM_ERASING_P,
    VIDEO_T_FRAMES,
    VIDEO_TRAIN_BATCH_SIZE,
    VERSION,
)
from src.data.video_dataset import (
    ID2LABEL,
    LABEL2ID,
    NUM_CLASSES,
    FrameCacheVideoDataset,
)
from src.evaluation.plots import save_confusion_matrix
from src.models.video_model import count_params, load_video_model
from src.training.utils import write_json


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_video_checkpoint(run_dir: str, device: str) -> nn.Module:
    """
    Reads arch.txt from the run's model/ directory, instantiates the correct
    architecture, and loads the best checkpoint weights.
    """
    model_dir = os.path.join(run_dir, "model")
    arch_file = os.path.join(model_dir, "arch.txt")
    ckpt_file = os.path.join(model_dir, "best_model.pt")

    if not os.path.exists(arch_file):
        raise FileNotFoundError(
            f"arch.txt not found in {model_dir}. "
            "Did you run train_one_video successfully?"
        )
    with open(arch_file) as f:
        model_name = f.read().strip()

    model = load_video_model(model_name, num_classes=NUM_CLASSES, device=device)
    state = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def evaluate_video(
    model:  nn.Module,
    loader: DataLoader,
    device: str,
) -> Dict:
    """
    Utterance-level evaluation.

    Returns dict with keys: preds, labels, utterance_keys
    (all as lists, parallel-indexed).
    """
    utt_probs: Dict[Tuple[int, int], list] = {}
    utt_label: Dict[Tuple[int, int], int]  = {}

    for clips, labels, d_ids, u_ids in loader:
        clips  = clips.to(device)

        with torch.amp.autocast(device, enabled=(device == "cuda")):
            logits = model(clips)

        # S3D outputs (B, C, t', h', w') — reduce spatial dims
        if logits.dim() > 2:
            logits = logits.mean(dim=[2, 3, 4])

        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        for i in range(len(labels)):
            key = (int(d_ids[i]), int(u_ids[i]))
            utt_probs.setdefault(key, []).append(probs[i])
            utt_label[key] = int(labels[i])

    preds, gts, keys = [], [], []
    for key in utt_label:
        avg_p = np.mean(utt_probs[key], axis=0)
        preds.append(int(np.argmax(avg_p)))
        gts.append(utt_label[key])
        keys.append(key)

    return {"preds": preds, "labels": gts, "utterance_keys": keys}


def measure_video_inference_ms(
    model:       nn.Module,
    loader:      DataLoader,
    device:      str,
    num_batches: int = 50,
) -> float:
    """
    Measures inference latency per clip (ms/clip) averaged over *num_batches*.
    """
    model.eval()
    times = []
    for i, (clips, *_) in enumerate(loader):
        if i >= num_batches:
            break
        clips = clips.to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(clips)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / len(clips) * 1000)

    return float(np.mean(times)) if times else float("nan")


# ── Full evaluation pipeline ──────────────────────────────────────────────────

def run_full_video_evaluation(
    run_dir:     str,
    test_df:     pd.DataFrame,
    cache_dir:   str,           # per-split test cache dir (contains *.npy)
    device:      str = DEVICE,
    batch_size:  int = None,    # None → use VIDEO_TRAIN_BATCH_SIZE[model_name]
    num_workers: int = 4,
    img_size:    int = VIDEO_IMG_SIZE,
    version:     str = VERSION,
) -> Dict:
    """
    Loads the best checkpoint, evaluates on test_df, writes artefacts, and
    returns a metrics dict.

    Parameters
    ----------
    run_dir    : path returned by train_one_video
    test_df    : DataFrame from load_video_metadata("test") — filtered for valid rows
    cache_dir  : the test-split cache directory returned by build_frame_cache
    """
    # Load config saved during training
    cfg        = json.load(open(os.path.join(run_dir, "config.json")))
    model_name = cfg["model_name"]
    T          = VIDEO_T_FRAMES.get(model_name, cfg.get("T_frames", 16))

    if batch_size is None:
        batch_size = VIDEO_TRAIN_BATCH_SIZE.get(model_name, 4)

    print(f"  Loading checkpoint from {run_dir}")
    model = _load_video_checkpoint(run_dir, device)

    # Build test DataLoader
    test_ds = FrameCacheVideoDataset(
        df        = test_df,
        cache_dir = cache_dir,
        T         = T,
        img_size  = img_size,
        train     = False,      # no augmentation
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )

    # Utterance-level predictions
    result = evaluate_video(model, test_loader, device)
    preds  = result["preds"]
    labels = result["labels"]

    # Metrics
    label_names = [ID2LABEL[i] for i in range(NUM_CLASSES)]
    acc    = float(accuracy_score(labels, preds))
    mf1    = float(f1_score(labels, preds, average="macro",    zero_division=0))
    wf1    = float(f1_score(labels, preds, average="weighted", zero_division=0))
    report = classification_report(
        labels, preds,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))

    # Inference speed
    infer_ms = measure_video_inference_ms(model, test_loader, device)

    # ── Write artefacts ───────────────────────────────────────────────────────
    write_json(os.path.join(run_dir, "classification_report_test.json"), report)

    cm_path = os.path.join(run_dir, "confusion_matrix_test.png")
    save_confusion_matrix(cm, label_names, cm_path,
                          title=f"Video — {model_name} (utterance-level)")

    params     = count_params(model)
    eval_metrics = {
        "model_name":                  model_name,
        "accuracy":                    round(acc, 6),
        "macro_f1":                    round(mf1, 6),
        "weighted_f1":                 round(wf1, 6),
        "num_test_utterances":         len(set(result["utterance_keys"])),
        "num_test_clips":              len(test_df),
        "inference_ms_per_clip_bs1":   round(infer_ms, 3),
        "total_params":                params["total"],
    }
    write_json(os.path.join(run_dir, "eval_metrics.json"), eval_metrics)

    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Macro F1    : {mf1:.4f}")
    print(f"  Weighted F1 : {wf1:.4f}")
    print(f"  Infer ms/clip: {infer_ms:.2f}")

    return {**eval_metrics, "preds": preds, "labels": labels}
