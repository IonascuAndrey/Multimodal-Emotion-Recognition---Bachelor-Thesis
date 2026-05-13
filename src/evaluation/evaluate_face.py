"""
evaluate_face.py
----------------
Loads the best checkpoint for a face model run and performs utterance-level
evaluation on the test set.

Utterance-level aggregation
----------------------------
K frame crops per utterance are run through the model independently.
Their softmax probabilities are averaged, then argmax gives the utterance
prediction.  This mirrors the strategy used in train_face.py so evaluation
is consistent with how the model was trained.

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
from typing import Dict, Optional, Tuple

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
    FACE_CACHE_DIR,
    FACE_IMG_SIZE,
    FACE_TRAIN_BATCH_SIZE,
    FACE_USE_CACHE,
    VERSION,
)
from src.data.face_dataset import (
    ID2LABEL,
    LABEL2ID,
    NUM_CLASSES,
    FaceEmotionDataset,
    get_transforms,
    _cache_dir_path,
    _npy_filename,
    build_cache,
)
from src.evaluation.plots import save_confusion_matrix
from src.models.face_model import load_face_model
from src.training.utils import write_json


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_checkpoint(run_dir: str, device: str) -> nn.Module:
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
            "Did you run train_one_face successfully?"
        )
    with open(arch_file) as f:
        model_name = f.read().strip()

    model = load_face_model(model_name, num_classes=NUM_CLASSES, device=device)
    state = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def evaluate_face(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Dict:
    """
    Utterance-level evaluation.

    Returns
    -------
    dict with keys: preds, labels, utterance_keys
    (all as lists, parallel-indexed)
    """
    utt_probs: Dict[Tuple[int, int], list] = {}
    utt_label: Dict[Tuple[int, int], int]  = {}

    for images, labels, weights, d_ids, u_ids in loader:
        images = images.to(device)
        with torch.amp.autocast(device, enabled=(device == "cuda")):
            logits = model(images)
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


def measure_face_inference_ms(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    num_batches: int = 200,
) -> float:
    """
    Measures inference latency at batch-size 1 equivalent (ms per sample).
    Runs *num_batches* batches from *loader* and averages.
    """
    model.eval()
    times = []
    for i, (images, *_) in enumerate(loader):
        if i >= num_batches:
            break
        images = images.to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(images)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / len(images) * 1000)

    return float(np.mean(times)) if times else float("nan")


# ── Full evaluation pipeline ──────────────────────────────────────────────────

def run_full_face_evaluation(
    run_dir: str,
    test_df: pd.DataFrame,
    device: str = DEVICE,
    use_cache: bool = FACE_USE_CACHE,
    cache_dir: str = FACE_CACHE_DIR,
    version: str = VERSION,
    batch_size: int = FACE_TRAIN_BATCH_SIZE,
    num_workers: int = 4,
) -> Dict:
    """
    Loads the best checkpoint, evaluates on test_df, writes artefacts, and
    returns a metrics dict.

    Parameters
    ----------
    run_dir    : path returned by train_one_face
    test_df    : DataFrame from load_face_metadata("test")
    """
    # Load config saved during training
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    model_name = cfg["model_name"]

    print(f"  Loading checkpoint from {run_dir}")
    model = _load_checkpoint(run_dir, device)

    # Build test DataLoader
    cdir = None
    if use_cache:
        cdir = _cache_dir_path("test", version, cache_dir)
        if not os.path.isdir(cdir):
            print("  Test cache not found — building …")
            build_cache(test_df, "test", version=version, cache_dir=cache_dir)

    test_ds = FaceEmotionDataset(
        df=test_df,
        transform=get_transforms(train=False),
        use_score_weights=False,          # no weighting at eval time
        use_cache=use_cache,
        cache_dir=cdir,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Utterance-level predictions
    result = evaluate_face(model, test_loader, device)
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

    # Inference speed (bs=1 approximation using existing loader)
    infer_ms = measure_face_inference_ms(model, test_loader, device)

    # ── Write artefacts ───────────────────────────────────────────────────────
    write_json(os.path.join(run_dir, "classification_report_test.json"), report)

    cm_path = os.path.join(run_dir, "confusion_matrix_test.png")
    save_confusion_matrix(cm, label_names, cm_path,
                          title=f"Face — {model_name} (utterance-level)")

    eval_metrics = {
        "model_name":                  model_name,
        "accuracy":                    round(acc, 6),
        "macro_f1":                    round(mf1, 6),
        "weighted_f1":                 round(wf1, 6),
        "num_test_utterances":         len(set(result["utterance_keys"])),
        "num_test_frames":             len(test_df),
        "inference_ms_per_sample_bs1": round(infer_ms, 3),
    }
    write_json(os.path.join(run_dir, "eval_metrics.json"), eval_metrics)

    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Macro F1    : {mf1:.4f}")
    print(f"  Weighted F1 : {wf1:.4f}")
    print(f"  Infer ms/sample: {infer_ms:.2f}")

    return {**eval_metrics, "preds": preds, "labels": labels}
