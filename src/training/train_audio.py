import os
import time
from typing import Dict

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, get_linear_schedule_with_warmup

from src.config import (
    AUDIO_BATCH_SIZE,
    AUDIO_TRAIN_DIR,
    AUDIO_DEV_DIR,
    AUDIO_TEST_DIR,
    DEVICE,
    EPOCHS,
    EXP_ROOT_AUDIO,
    GRAD_CLIP_NORM,
    LR,
    MAX_AUDIO_S,
    SAMPLE_RATE,
    TRIM_END_S,
    TRIM_START_S,
    VERSION,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from src.data.audio_preprocessing import make_audio_loaders
from src.evaluation.metrics import evaluate
from src.models.audio_model import count_params, load_audio_model
from src.training.utils import (
    NvidiaPowerMonitor,
    safe_mkdir_run_dir,
    slugify,
    write_json,
)


def train_one_audio(
    model_name: str,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    trim_start_s: float = TRIM_START_S,
    trim_end_s: float = TRIM_END_S,
    max_audio_s: float = MAX_AUDIO_S,
) -> str:
    """
    Fine-tune a HuggingFace audio-classification model on the MELD training set.

    Saves the best checkpoint (by dev macro-F1) plus training history,
    run config, power samples, and a training-summary metrics file.

    Args:
        model_name:    HuggingFace model id (e.g. 'facebook/wav2vec2-base').
        train_df:      Training DataFrame (must have 'filename' and 'label').
        dev_df:        Validation DataFrame.
        test_df:       Test DataFrame (used only for DataLoader construction).
        label2id:      Emotion string → int mapping.
        id2label:      Int → emotion string mapping.
        trim_start_s:  Seconds to remove from the start of each clip.
        trim_end_s:    Seconds to remove from the end of each clip.
        max_audio_s:   Maximum clip duration in seconds (clips are truncated).

    Returns:
        run_dir: Path to the directory where all artefacts were saved.
    """
    model_slug = slugify(model_name)
    run_dir    = safe_mkdir_run_dir(os.path.join(EXP_ROOT_AUDIO, VERSION, model_slug))
    os.makedirs(os.path.join(run_dir, "model"), exist_ok=True)

    # ── Save run config ───────────────────────────────────────────────────────
    write_json(os.path.join(run_dir, "config.json"), {
        "model_name":    model_name,
        "version":       VERSION,
        "run_dir":       run_dir,
        "sample_rate":   SAMPLE_RATE,
        "max_audio_s":   max_audio_s,
        "trim_start_s":  trim_start_s,
        "trim_end_s":    trim_end_s,
        "batch_size":    AUDIO_BATCH_SIZE,
        "epochs":        EPOCHS,
        "lr":            LR,
        "weight_decay":  WEIGHT_DECAY,
        "warmup_ratio":  WARMUP_RATIO,
        "device":        DEVICE,
        "torch_version": torch.__version__,
        "label2id":      label2id,
        "id2label":      id2label,
    })

    # ── Feature extractor + DataLoaders ──────────────────────────────────────
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    train_loader, dev_loader, _ = make_audio_loaders(
        train_df=train_df,
        dev_df=dev_df,
        test_df=test_df,
        audio_dirs=(AUDIO_TRAIN_DIR, AUDIO_DEV_DIR, AUDIO_TEST_DIR),
        feature_extractor=feature_extractor,
        batch_size=AUDIO_BATCH_SIZE,
        sample_rate=SAMPLE_RATE,
        max_audio_s=max_audio_s,
        trim_start_s=trim_start_s,
        trim_end_s=trim_end_s,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model       = load_audio_model(model_name, len(label2id), label2id, id2label, DEVICE)
    params_info = count_params(model)

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Mixed precision ───────────────────────────────────────────────────────
    use_amp = torch.cuda.is_available()
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── GPU power monitoring ──────────────────────────────────────────────────
    power_mon = NvidiaPowerMonitor(interval_s=1.0)
    power_mon.start()
    t_start = time.time()

    history_rows      = []
    best_dev_macro_f1 = -1.0
    best_epoch        = -1

    try:
        patience = 5  
        no_improvement_counter = 0
        for epoch in range(1, EPOCHS + 1):
            model.train()
            epoch_loss = 0.0

            for batch in tqdm(train_loader, desc=f"{model_slug} | epoch {epoch}/{EPOCHS}"):
                batch = {
                    k: v.to(DEVICE) if torch.is_tensor(v) else torch.tensor(v).to(DEVICE)
                    for k, v in batch.items()
                }

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(**batch)
                    loss    = outputs.loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                epoch_loss += loss.item()

            train_loss  = epoch_loss / max(1, len(train_loader))
            dev_metrics = evaluate(model, dev_loader, DEVICE)

            row = {
                "epoch":           epoch,
                "train_loss":      float(train_loss),
                "dev_loss":        dev_metrics["loss"],
                "dev_accuracy":    dev_metrics["accuracy"],
                "dev_macro_f1":    dev_metrics["macro_f1"],
                "dev_weighted_f1": dev_metrics["weighted_f1"],
            }
            history_rows.append(row)
            pd.DataFrame(history_rows).to_csv(os.path.join(run_dir, "history.csv"), index=False)

            # Save checkpoint if this is the best dev macro-F1 so far
            if dev_metrics["macro_f1"] > best_dev_macro_f1:
                best_dev_macro_f1 = dev_metrics["macro_f1"]
                best_epoch        = epoch
                model.save_pretrained(os.path.join(run_dir, "model"))
                feature_extractor.save_pretrained(os.path.join(run_dir, "model"))
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1
                print(f"No prgress in {no_improvement_counter} epochs.")
            if no_improvement_counter >= patience:
                print(f"Early stopping at epoch {epoch}!")
                break 

    finally:
        t_end = time.time()
        power_mon.stop()
        power_mon.save_csv(os.path.join(run_dir, "power_samples.csv"))

    # ── Training summary ──────────────────────────────────────────────────────
    write_json(os.path.join(run_dir, "train_metrics.json"), {
        "model_name":        model_name,
        "params":            params_info,
        "best_epoch":        best_epoch,
        "best_dev_macro_f1": best_dev_macro_f1,
        "train_time_s":      float(t_end - t_start),
        "power":             power_mon.stats(),
        "trim_start_s":      trim_start_s,
        "trim_end_s":        trim_end_s,
        "max_audio_s":       max_audio_s,
    })

    print(f"\n[DONE] {model_name}  →  {run_dir}")
    print(f"  Best dev macro-F1 : {best_dev_macro_f1:.4f}  (epoch {best_epoch})")
    return run_dir
