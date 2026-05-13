"""
train_video.py
--------------
Training loop for the video modality (MViT-V2-S, S3D).

Anti-overfitting measures (mirrors face modality)
--------------------------------------------------
1. Backbone freeze warm-up
   Head-only training for first VIDEO_FREEZE_EPOCHS epochs; then full
   fine-tuning with differential learning rates.

2. Differential learning rates
   Two AdamW param groups:
     • backbone : lr × VIDEO_BACKBONE_LR_MULT  (default: 1e-5)
     • head     : lr                           (default: 1e-4)

3. Label smoothing
   CrossEntropyLoss with label_smoothing=VIDEO_LABEL_SMOOTHING.

4. Class-balanced sampling
   WeightedRandomSampler with inverse class-frequency weights.

5. Temporal-consistent augmentation + per-frame RandomErasing
   Applied in video_dataset._augment_clip.

6. Gradient accumulation
   Effective batch kept at 16 regardless of per-step batch size:
     MViT: batch=4 × accum=4   S3D: batch=8 × accum=2

Other design choices
--------------------
* Mixed-precision (torch.amp) on CUDA, fp32 on CPU.
* AdamW + OneCycleLR (separate max_lr per param group).
* Best checkpoint selected by dev utterance-level macro-F1.
* Output layout:
    experiments/video_unimodal/<version>/<slug>/
        model/      — best_model.pt  + arch.txt
        history.csv
        config.json
        train_metrics.json
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
    EXP_ROOT_VIDEO,
    VIDEO_BACKBONE_LR_MULT,
    VIDEO_BALANCED_SAMPLER,
    VIDEO_FREEZE_EPOCHS,
    VIDEO_GRAD_ACCUM_STEPS,
    VIDEO_GRAD_CLIP_NORM,
    VIDEO_HEAD_DROPOUT,
    VIDEO_IMG_SIZE,
    VIDEO_LABEL_SMOOTHING,
    VIDEO_RANDOM_ERASING_P,
    VIDEO_T_FRAMES,
    VIDEO_TRAIN_BATCH_SIZE,
    VIDEO_TRAIN_EPOCHS,
    VIDEO_TRAIN_LR,
    VIDEO_WARMUP_RATIO,
    VIDEO_WEIGHT_DECAY,
    VERSION,
)
from src.data.video_dataset import (
    ID2LABEL,
    LABEL2ID,
    NUM_CLASSES,
    _build_video_sampler,
    make_video_loaders,
)
from src.models.video_model import (
    count_params,
    freeze_video_backbone,
    get_video_param_groups,
    load_video_model,
    unfreeze_video_backbone,
)
from src.training.utils import (
    NvidiaPowerMonitor,
    safe_mkdir_run_dir,
    set_seed,
    slugify,
    write_json,
)


# ── Loss ──────────────────────────────────────────────────────────────────────

def _ce_loss(
    logits:          torch.Tensor,
    labels:          torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Standard cross-entropy with optional label smoothing."""
    return nn.functional.cross_entropy(
        logits, labels, label_smoothing=label_smoothing
    )


# ── Utterance-level evaluation ────────────────────────────────────────────────

@torch.no_grad()
def _evaluate(
    model:           nn.Module,
    loader:          DataLoader,
    device:          str,
    label_smoothing: float = 0.0,
) -> Dict:
    """
    Utterance-aware evaluation:
      - Collects (d_id, u_id, softmax_probs) for every clip.
      - Groups clips by (d_id, u_id) and averages softmax probs.
      - Argmax of averaged probs → utterance prediction.
    Returns accuracy, macro_f1, weighted_f1, loss.
    """
    model.eval()
    utt_probs: Dict[Tuple[int, int], list] = {}
    utt_label: Dict[Tuple[int, int], int]  = {}
    total_loss = 0.0
    n_batches  = 0

    for clips, labels, d_ids, u_ids in loader:
        clips  = clips.to(device)   # (B, 3, T, H, W)
        labels = labels.to(device)

        with torch.amp.autocast(device, enabled=(device == "cuda")):
            logits = model(clips)

        # S3D outputs (B, num_classes, 1, 1, 1) — squeeze spatial dims
        if logits.dim() > 2:
            logits = logits.mean(dim=[2, 3, 4])

        loss = _ce_loss(logits, labels, label_smoothing)
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

def train_one_video(
    model_name:       str,
    train_df:         pd.DataFrame,
    dev_df:           pd.DataFrame,
    test_df:          pd.DataFrame,
    cache_dirs:       Dict[str, str],   # {"train": path, "dev": path, "test": path}
    # Hyperparameter overrides (defaults from config.py)
    batch_size:       int   = None,     # None → use VIDEO_TRAIN_BATCH_SIZE[model_name]
    grad_accum:       int   = None,     # None → use VIDEO_GRAD_ACCUM_STEPS[model_name]
    epochs:           int   = VIDEO_TRAIN_EPOCHS,
    lr:               float = VIDEO_TRAIN_LR,
    weight_decay:     float = VIDEO_WEIGHT_DECAY,
    warmup_ratio:     float = VIDEO_WARMUP_RATIO,
    grad_clip:        float = VIDEO_GRAD_CLIP_NORM,
    label_smoothing:  float = VIDEO_LABEL_SMOOTHING,
    backbone_lr_mult: float = VIDEO_BACKBONE_LR_MULT,
    freeze_epochs:    int   = VIDEO_FREEZE_EPOCHS,
    head_dropout:     float = VIDEO_HEAD_DROPOUT,
    balanced_sampler: bool  = VIDEO_BALANCED_SAMPLER,
    erasing_p:        float = VIDEO_RANDOM_ERASING_P,
    img_size:         int   = VIDEO_IMG_SIZE,
    version:          str   = VERSION,
    device:           str   = DEVICE,
    seed:             int   = 42,
    num_workers:      int   = 4,
) -> str:
    """
    Fine-tunes *model_name* on video clips and saves the best checkpoint.

    Parameters
    ----------
    cache_dirs : dict with keys "train", "dev", "test" mapping to the
                 per-split frame-cache directories returned by build_frame_cache.

    Returns
    -------
    Path to the run directory.
    """
    set_seed(seed)

    # Resolve per-model defaults
    if batch_size is None:
        batch_size = VIDEO_TRAIN_BATCH_SIZE[model_name]
    if grad_accum is None:
        grad_accum = VIDEO_GRAD_ACCUM_STEPS[model_name]

    T = VIDEO_T_FRAMES[model_name]

    run_dir   = safe_mkdir_run_dir(
        os.path.join(EXP_ROOT_VIDEO, version, slugify(model_name))
    )
    model_dir = os.path.join(run_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Video model     : {model_name}")
    print(f"  Run dir         : {run_dir}")
    print(f"  Device          : {device}")
    print(f"  T (frames/clip) : {T}")
    print(f"  Batch size      : {batch_size}  |  Grad accum: {grad_accum}  "
          f"(eff. batch={batch_size * grad_accum})")
    print(f"  Freeze epochs   : {freeze_epochs}")
    print(f"  Backbone LR     : {lr * backbone_lr_mult:.2e}  |  Head LR: {lr:.2e}")
    print(f"  Label smoothing : {label_smoothing}")
    print(f"  Balanced sampler: {balanced_sampler}")
    print(f"{'='*60}")

    # ── Sampler ───────────────────────────────────────────────────────────────
    sampler = _build_video_sampler(train_df) if balanced_sampler else None

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, dev_loader, _ = make_video_loaders(
        train_df        = train_df,
        dev_df          = dev_df,
        test_df         = test_df,
        model_name      = model_name,
        cache_dir       = cache_dirs,
        batch_size      = batch_size,
        num_workers     = num_workers,
        img_size        = img_size,
        erasing_p       = erasing_p,
        balanced_sampler= False,   # handled externally via train_sampler
        train_sampler   = sampler,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = load_video_model(
        model_name,
        num_classes  = NUM_CLASSES,
        head_dropout = head_dropout,
        device       = device,
    )
    params = count_params(model)
    print(f"  Params          : {params['trainable']:,} trainable / "
          f"{params['total']:,} total")

    # ── Backbone freeze warm-up ───────────────────────────────────────────────
    if freeze_epochs > 0:
        freeze_video_backbone(model, model_name)
        trainable_frozen = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(f"  Backbone frozen : {trainable_frozen:,} params active "
              f"for first {freeze_epochs} epoch(s).")

    # ── Optimiser + schedule ──────────────────────────────────────────────────
    param_groups = get_video_param_groups(model, model_name, lr, backbone_lr_mult)
    optimizer    = AdamW(param_groups, weight_decay=weight_decay)

    # OneCycleLR uses total optimizer steps (after gradient accumulation)
    steps_per_epoch = len(train_loader)   # batches (before accumulation)
    total_steps     = steps_per_epoch * epochs

    scheduler = OneCycleLR(
        optimizer,
        max_lr         = [lr * backbone_lr_mult, lr],
        total_steps    = total_steps,
        pct_start      = warmup_ratio,
        anneal_strategy= "cos",
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

        # Unfreeze backbone after warm-up
        if epoch == freeze_epochs + 1:
            unfreeze_video_backbone(model)
            trainable_now = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            print(f"\n  [Epoch {epoch}] Backbone unfrozen — "
                  f"{trainable_now:,} params now active.")

        model.train()
        ep_loss = 0.0
        ep_n    = 0

        optimizer.zero_grad()

        for step, (clips, labels, _d, _u) in enumerate(train_loader, 1):
            clips  = clips.to(device)    # (B, 3, T, H, W)
            labels = labels.to(device)

            with torch.amp.autocast(device, enabled=(device == "cuda")):
                logits = model(clips)

                # S3D outputs (B, C, t', h', w') — reduce spatial dims
                if logits.dim() > 2:
                    logits = logits.mean(dim=[2, 3, 4])

                loss = _ce_loss(logits, labels, label_smoothing)
                loss = loss / grad_accum   # scale for accumulation

            scaler.scale(loss).backward()

            if step % grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Step the scheduler every batch (OneCycleLR expects this)
            scheduler.step()

            ep_loss += loss.item() * grad_accum   # un-scale for logging
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
        "model_name":        model_name,
        "version":           version,
        "epochs":            epochs,
        "batch_size":        batch_size,
        "grad_accum":        grad_accum,
        "effective_batch":   batch_size * grad_accum,
        "T_frames":          T,
        "img_size":          img_size,
        "lr":                lr,
        "backbone_lr_mult":  backbone_lr_mult,
        "freeze_epochs":     freeze_epochs,
        "weight_decay":      weight_decay,
        "warmup_ratio":      warmup_ratio,
        "grad_clip":         grad_clip,
        "label_smoothing":   label_smoothing,
        "head_dropout":      head_dropout,
        "balanced_sampler":  balanced_sampler,
        "erasing_p":         erasing_p,
        "num_classes":       NUM_CLASSES,
        "label2id":          LABEL2ID,
        "id2label":          {str(k): v for k, v in ID2LABEL.items()},
        "seed":              seed,
        "device":            device,
        "num_train_clips":   len(train_df),
        "num_dev_clips":     len(dev_df),
    }
    write_json(os.path.join(run_dir, "config.json"), cfg)

    with open(os.path.join(model_dir, "arch.txt"), "w") as f:
        f.write(model_name)

    pwr = power_monitor.stats()
    train_metrics = {
        "model_name":        model_name,
        "best_dev_macro_f1": round(best_f1, 6),
        "total_params":      params["total"],
        "trainable_params":  params["trainable"],
        "training_time_s":   round(elapsed, 1),
        "avg_gpu_power_w":   pwr.get("avg_power_w"),
        "energy_kwh":        pwr.get("energy_kwh"),
    }
    write_json(os.path.join(run_dir, "train_metrics.json"), train_metrics)

    print(f"\n  Finished in {elapsed/60:.1f} min.  Best dev macro-F1: {best_f1:.4f}")
    print(f"  Artefacts → {run_dir}")
    return run_dir
