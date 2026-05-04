import os
import time
from typing import Dict

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.config import (
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
    EXP_ROOT,
    GRAD_CLIP_NORM,
    LR,
    MAX_LEN,
    VERSION,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from src.data.text_preprocessing import make_loaders
from src.evaluation.metrics import evaluate
from src.models.text_model import count_params, load_model
from src.training.utils import (
    NvidiaPowerMonitor,
    safe_mkdir_run_dir,
    slugify,
    write_json,
)


def train_one(
    model_name: str,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
) -> str:
    """
    Fine-tune a HuggingFace classifier on the MELD training set.

    Saves the best checkpoint (by dev macro-F1) plus training history,
    run config, power samples, and a summary metrics file.

    Args:
        model_name:  HuggingFace model id (e.g. 'bert-base-uncased').
        train_df:    Training DataFrame (must have 'text' and 'label' columns).
        dev_df:      Validation DataFrame.
        test_df:     Test DataFrame (used only for DataLoader construction).
        label2id:    Emotion string → int mapping.
        id2label:    Int → emotion string mapping.

    Returns:
        run_dir: Path to the directory where all artefacts were saved.
    """
    model_slug = slugify(model_name)
    run_dir    = safe_mkdir_run_dir(os.path.join(EXP_ROOT, VERSION, model_slug))
    os.makedirs(os.path.join(run_dir, "model"), exist_ok=True)

    # ── Save run config ───────────────────────────────────────────────────────
    write_json(os.path.join(run_dir, "config.json"), {
        "model_name":   model_name,
        "version":      VERSION,
        "run_dir":      run_dir,
        "max_len":      MAX_LEN,
        "batch_size":   BATCH_SIZE,
        "epochs":       EPOCHS,
        "lr":           LR,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "device":       DEVICE,
        "torch_version": torch.__version__,
        "label2id":     label2id,
        "id2label":     id2label,
    })

    # ── Tokeniser + DataLoaders ───────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_loader, dev_loader, _ = make_loaders(train_df, dev_df, test_df, tokenizer)

    # ── Model ─────────────────────────────────────────────────────────────────
    model       = load_model(model_name, len(label2id), label2id, id2label, DEVICE)
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
        for epoch in range(1, EPOCHS + 1):
            model.train()
            epoch_loss = 0.0

            for batch in tqdm(train_loader, desc=f"{model_slug} | epoch {epoch}/{EPOCHS}"):
                batch = {
                    k: torch.tensor(v).to(DEVICE) if not torch.is_tensor(v) else v.to(DEVICE)
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
                tokenizer.save_pretrained(os.path.join(run_dir, "model"))

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
    })

    print(f"\n[DONE] {model_name}  →  {run_dir}")
    print(f"  Best dev macro-F1 : {best_dev_macro_f1:.4f}  (epoch {best_epoch})")
    return run_dir
