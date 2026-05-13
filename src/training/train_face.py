"""
train_face.py
-------------
Training loop for the face modality.

Anti-overfitting measures
--------------------------
1. Backbone freeze warm-up
   The pretrained backbone is frozen for the first FACE_FREEZE_EPOCHS epochs.
   Only the new classification head is trained.  After that, the entire model
   is fine-tuned with differential learning rates (see below).

2. Differential learning rates
   Two AdamW parameter groups:
     • backbone : lr × FACE_BACKBONE_LR_MULT  (e.g. 1e-5)
     • head     : lr                           (e.g. 1e-4)
   Prevents the pretrained features from being overwritten too quickly.

3. Label smoothing
   CrossEntropyLoss with label_smoothing=FACE_LABEL_SMOOTHING prevents
   overconfident predictions and acts as a soft regulariser.

4. Score-weighted cross-entropy
   loss = mean( w_i * CE(logits_i, label_i) )
   where w_i = clip(score_comb_i^power, weight_min, 1.0).

5. Class-balanced sampling
   WeightedRandomSampler draws training samples so each emotion class appears
   roughly equally often per epoch, combating MELD's heavy class imbalance.

6. Stronger augmentation
   get_transforms(train=True) applies RandomErasing on top of the standard
   crop / flip / colour jitter / rotation pipeline.

Other design choices
--------------------
* Mixed-precision (torch.amp) on CUDA, fp32 on CPU.
* AdamW + OneCycleLR (separate schedules for each param group).
* Best checkpoint selected by dev macro-F1.
* Output layout:
    experiments/face_unimodal/<version>/<slug>/
        model/          — best_model.pt  +  arch.txt
        history.csv
        config.json
        train_metrics.json
        power_samples.csv
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
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.config import (
    DEVICE,
    EXP_ROOT_FACE,
    FACE_BACKBONE_LR_MULT,
    FACE_BALANCED_SAMPLER,
    FACE_CACHE_DIR,
    FACE_FREEZE_EPOCHS,
    FACE_GRAD_CLIP_NORM,
    FACE_HEAD_DROPOUT,
    FACE_IMG_SIZE,
    FACE_LABEL_SMOOTHING,
    FACE_SCORE_WEIGHT_MIN,
    FACE_SCORE_WEIGHT_NORMALIZE,
    FACE_SCORE_WEIGHT_POWER,
    FACE_TRAIN_BATCH_SIZE,
    FACE_TRAIN_EPOCHS,
    FACE_TRAIN_LR,
    FACE_USE_CACHE,
    FACE_USE_SCORE_WEIGHTS,
    FACE_WARMUP_RATIO,
    FACE_WEIGHT_DECAY,
    VERSION,
)
from src.data.face_dataset import (
    ID2LABEL,
    LABEL2ID,
    NUM_CLASSES,
    load_face_metadata,
    make_face_loaders,
)
from src.models.face_model import (
    count_params,
    freeze_backbone,
    get_param_groups,
    load_face_model,
    unfreeze_backbone,
)
from src.training.utils import (
    NvidiaPowerMonitor,
    safe_mkdir_run_dir,
    set_seed,
    slugify,
    write_json,
)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _weighted_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Element-wise cross-entropy (with optional label smoothing) multiplied by
    per-sample score weights.  Returns the mean over the batch.
    """
    ce = nn.functional.cross_entropy(
        logits, labels,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    return (ce * weights).mean()


def _build_sampler(train_df: pd.DataFrame) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler that draws samples with probability
    inversely proportional to class frequency, so each emotion class appears
    roughly equally often each epoch.
    """
    labels = train_df["label_id"].values
    class_counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    class_counts = np.maximum(class_counts, 1)           # avoid div-by-zero
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(train_df),
        replacement=True,
    )


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    label_smoothing: float = 0.0,
) -> Dict:
    """
    Utterance-aware evaluation:
      - Collects (d_id, u_id, softmax_probs) for every frame.
      - Groups frames by (d_id, u_id) and averages softmax probs.
      - Argmax of averaged probs is the utterance prediction.
    Returns accuracy, macro_f1, weighted_f1, loss.
    """
    model.eval()
    utt_probs: Dict[Tuple[int, int], list] = {}
    utt_label: Dict[Tuple[int, int], int]  = {}
    total_loss = 0.0
    n_batches  = 0

    for images, labels, weights, d_ids, u_ids in loader:
        images  = images.to(device)
        labels  = labels.to(device)
        weights = weights.to(device)

        with torch.amp.autocast(device, enabled=(device == "cuda")):
            logits = model(images)

        loss = _weighted_ce_loss(logits, labels, weights, label_smoothing)
        total_loss += loss.item()
        n_batches  += 1

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for i in range(len(labels)):
            key = (int(d_ids[i]), int(u_ids[i]))
            utt_probs.setdefault(key, []).append(probs[i])
            utt_label[key] = int(labels[i].cpu())

    preds, gts = [], []
    for key in utt_label:
        avg_p = np.mean(utt_probs[key], axis=0)
        preds.append(int(np.argmax(avg_p)))
        gts.append(utt_label[key])

    preds = np.array(preds)
    gts   = np.array(gts)
    acc   = float((preds == gts).mean())
    mf1   = float(f1_score(gts, preds, average="macro",    zero_division=0))
    wf1   = float(f1_score(gts, preds, average="weighted", zero_division=0))

    return {
        "loss":        total_loss / max(n_batches, 1),
        "accuracy":    acc,
        "macro_f1":    mf1,
        "weighted_f1": wf1,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def train_one_face(
    model_name: str,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    # Hyperparameter overrides (default = config.py values)
    batch_size: int          = FACE_TRAIN_BATCH_SIZE,
    epochs: int              = FACE_TRAIN_EPOCHS,
    lr: float                = FACE_TRAIN_LR,
    weight_decay: float      = FACE_WEIGHT_DECAY,
    warmup_ratio: float      = FACE_WARMUP_RATIO,
    grad_clip: float         = FACE_GRAD_CLIP_NORM,
    label_smoothing: float   = FACE_LABEL_SMOOTHING,
    backbone_lr_mult: float  = FACE_BACKBONE_LR_MULT,
    freeze_epochs: int       = FACE_FREEZE_EPOCHS,
    head_dropout: float      = FACE_HEAD_DROPOUT,
    balanced_sampler: bool   = FACE_BALANCED_SAMPLER,
    use_score_weights: bool  = FACE_USE_SCORE_WEIGHTS,
    score_power: float       = FACE_SCORE_WEIGHT_POWER,
    score_min: float         = FACE_SCORE_WEIGHT_MIN,
    score_normalize: bool    = FACE_SCORE_WEIGHT_NORMALIZE,
    use_cache: bool          = FACE_USE_CACHE,
    cache_dir: str           = FACE_CACHE_DIR,
    version: str             = VERSION,
    device: str              = DEVICE,
    seed: int                = 42,
    num_workers: int         = 4,
) -> str:
    """
    Fine-tunes *model_name* and saves the best checkpoint.

    Returns the run directory path.
    """
    set_seed(seed)

    run_dir   = safe_mkdir_run_dir(os.path.join(EXP_ROOT_FACE, version, slugify(model_name)))
    model_dir = os.path.join(run_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Face model      : {model_name}")
    print(f"  Run dir         : {run_dir}")
    print(f"  Device          : {device}")
    print(f"  Freeze epochs   : {freeze_epochs}")
    print(f"  Backbone LR     : {lr * backbone_lr_mult:.2e}  |  Head LR: {lr:.2e}")
    print(f"  Label smoothing : {label_smoothing}")
    print(f"  Balanced sampler: {balanced_sampler}")
    print(f"{'='*60}")

    # ── Sampler ───────────────────────────────────────────────────────────────
    sampler = _build_sampler(train_df) if balanced_sampler else None

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, dev_loader, _ = make_face_loaders(
        train_df=train_df,
        dev_df=dev_df,
        test_df=test_df,
        batch_size=batch_size,
        num_workers=num_workers,
        use_cache=use_cache,
        cache_dir=cache_dir,
        version=version,
        use_score_weights=use_score_weights,
        score_weight_power=score_power,
        score_weight_min=score_min,
        score_weight_normalize=score_normalize,
        train_sampler=sampler,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = load_face_model(
        model_name,
        num_classes=NUM_CLASSES,
        head_dropout=head_dropout,
        device=device,
    )
    params = count_params(model)
    print(f"  Params          : {params['trainable']:,} trainable / "
          f"{params['total']:,} total")

    # ── Freeze backbone for warm-up phase ─────────────────────────────────────
    if freeze_epochs > 0:
        freeze_backbone(model, model_name)
        trainable_after_freeze = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(f"  Backbone frozen : {trainable_after_freeze:,} params active "
              f"for first {freeze_epochs} epoch(s).")

    # ── Optimiser + schedule ──────────────────────────────────────────────────
    # Always create both param groups; backbone group has zero gradient while
    # frozen — it becomes active automatically when requires_grad is restored.
    param_groups  = get_param_groups(model, model_name, lr, backbone_lr_mult)
    optimizer     = AdamW(param_groups, weight_decay=weight_decay)
    total_steps   = len(train_loader) * epochs
    scheduler     = OneCycleLR(
        optimizer,
        max_lr=[lr * backbone_lr_mult, lr],   # one per param group
        total_steps=total_steps,
        pct_start=warmup_ratio,
        anneal_strategy="cos",
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # ── Power monitor ─────────────────────────────────────────────────────────
    power_monitor = NvidiaPowerMonitor(interval_s=5.0)
    power_monitor.start()

    # ── Training loop ─────────────────────────────────────────────────────────
    history = []
    best_f1 = -1.0
    t_start = time.time()

    for epoch in range(1, epochs + 1):

        # Unfreeze backbone after warm-up phase
        if epoch == freeze_epochs + 1:
            unfreeze_backbone(model)
            trainable_now = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n  [Epoch {epoch}] Backbone unfrozen — "
                  f"{trainable_now:,} params now active.")

        model.train()
        ep_loss = 0.0
        ep_n    = 0

        for images, labels, weights, _d, _u in train_loader:
            images  = images.to(device)
            labels  = labels.to(device)
            weights = weights.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device, enabled=(device == "cuda")):
                logits = model(images)
                loss   = _weighted_ce_loss(
                    logits, labels, weights, label_smoothing
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            ep_loss += loss.item()
            ep_n    += 1

        train_loss  = ep_loss / max(ep_n, 1)
        dev_metrics = _evaluate(model, dev_loader, device, label_smoothing)

        row = {
            "epoch":        epoch,
            "train_loss":   round(train_loss, 6),
            "dev_loss":     round(dev_metrics["loss"], 6),
            "dev_acc":      round(dev_metrics["accuracy"], 6),
            "dev_macro_f1": round(dev_metrics["macro_f1"], 6),
            "dev_wt_f1":    round(dev_metrics["weighted_f1"], 6),
            "frozen":       epoch <= freeze_epochs,
        }
        history.append(row)

        frozen_tag = " [frozen]" if epoch <= freeze_epochs else ""
        print(
            f"  [Epoch {epoch:02d}/{epochs}]{frozen_tag} "
            f"train_loss={train_loss:.4f}  "
            f"dev_loss={dev_metrics['loss']:.4f}  "
            f"dev_macro_f1={dev_metrics['macro_f1']:.4f}"
        )

        if dev_metrics["macro_f1"] > best_f1:
            best_f1 = dev_metrics["macro_f1"]
            torch.save(model.state_dict(),
                       os.path.join(model_dir, "best_model.pt"))
            print(f"    ↳ New best dev macro-F1: {best_f1:.4f} — checkpoint saved.")

    elapsed = time.time() - t_start
    power_monitor.stop()

    # ── Persist artefacts ─────────────────────────────────────────────────────
    pd.DataFrame(history).to_csv(
        os.path.join(run_dir, "history.csv"), index=False
    )

    cfg = {
        "model_name":             model_name,
        "version":                version,
        "epochs":                 epochs,
        "batch_size":             batch_size,
        "lr":                     lr,
        "backbone_lr_mult":       backbone_lr_mult,
        "freeze_epochs":          freeze_epochs,
        "weight_decay":           weight_decay,
        "warmup_ratio":           warmup_ratio,
        "grad_clip":              grad_clip,
        "label_smoothing":        label_smoothing,
        "head_dropout":           head_dropout,
        "balanced_sampler":       balanced_sampler,
        "img_size":               FACE_IMG_SIZE,
        "use_score_weights":      use_score_weights,
        "score_weight_power":     score_power,
        "score_weight_min":       score_min,
        "score_weight_normalize": score_normalize,
        "use_cache":              use_cache,
        "num_classes":            NUM_CLASSES,
        "label2id":               LABEL2ID,
        "id2label":               {str(k): v for k, v in ID2LABEL.items()},
        "seed":                   seed,
        "device":                 device,
        "num_train_frames":       len(train_df),
        "num_dev_frames":         len(dev_df),
    }
    write_json(os.path.join(run_dir, "config.json"), cfg)

    with open(os.path.join(model_dir, "arch.txt"), "w") as f:
        f.write(model_name)

    pwr = power_monitor.stats()
    train_metrics = {
        "model_name":          model_name,
        "best_dev_macro_f1":   round(best_f1, 6),
        "total_params":        params["total"],
        "trainable_params":    params["trainable"],
        "training_time_s":     round(elapsed, 1),
        "avg_gpu_power_w":     pwr.get("avg_power_w"),
        "energy_kwh":          pwr.get("energy_kwh"),
    }
    write_json(os.path.join(run_dir, "train_metrics.json"), train_metrics)

    print(f"\n  Finished in {elapsed/60:.1f} min.  Best dev macro-F1: {best_f1:.4f}")
    print(f"  Artefacts → {run_dir}")
    return run_dir
