import os
import time
from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from src.config import BATCH_SIZE
from src.data.text_preprocessing import TextEmotionDataset
from src.evaluation.metrics import evaluate
from src.evaluation.plots import save_confusion_matrix
from src.training.utils import write_json


# ── Inference speed ───────────────────────────────────────────────────────────

def measure_inference_ms_per_sample(
    model,
    loader: DataLoader,
    device: str,
    num_batches: int = 200,
) -> float:
    """
    Measure average forward-pass latency per sample (expects batch size = 1).

    Uses torch.cuda.synchronize() for accurate GPU timing.

    Args:
        model:       Trained model in eval mode.
        loader:      DataLoader with batch_size=1.
        device:      'cuda' or 'cpu'.
        num_batches: How many batches to time (after a warmup phase).

    Returns:
        Mean inference time in milliseconds, or nan if loader is empty.
    """
    model.eval()
    times = []
    it    = iter(loader)

    # Warmup — lets the GPU reach steady-state clocks and warms caches.
    for _ in range(10):
        try:
            batch = next(it)
        except StopIteration:
            break
        batch = {
            k: torch.tensor(v).to(device) if not torch.is_tensor(v) else v.to(device)
            for k, v in batch.items()
        }
        _ = model(**batch)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for _ in range(num_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        batch = {
            k: torch.tensor(v).to(device) if not torch.is_tensor(v) else v.to(device)
            for k, v in batch.items()
        }
        t0 = time.time()
        _ = model(**batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000.0)

    return float(np.mean(times)) if times else float("nan")


# ── Full test-set evaluation ──────────────────────────────────────────────────

def run_full_evaluation(
    run_dir: str,
    test_df,
    id2label: Dict[int, str],
    device: str,
    num_inference_batches: int = 200,
) -> Dict[str, Any]:
    """
    Load the best saved checkpoint from *run_dir*, evaluate on *test_df*, and
    write the following files into *run_dir*:
        - classification_report_test.json
        - confusion_matrix_test.png

    Args:
        run_dir:               Directory that contains a 'model/' subdirectory.
        test_df:               Test DataFrame with 'text' and 'label' columns.
        id2label:              Mapping int → emotion string.
        device:                'cuda' or 'cpu'.
        num_inference_batches: Batches to time for the speed measurement.

    Returns:
        Dict with test metrics and inference speed.
    """
    model_path = os.path.join(run_dir, "model")
    model      = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer  = AutoTokenizer.from_pretrained(model_path)
    collator   = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    # Full-batch test loader for metric computation
    test_loader = DataLoader(
        TextEmotionDataset(test_df, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )
    test_metrics = evaluate(model, test_loader, device)

    # Classification report
    label_names = [id2label[i] for i in range(len(id2label))]
    report = classification_report(
        test_metrics["labels"],
        test_metrics["preds"],
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    write_json(os.path.join(run_dir, "classification_report_test.json"), report)

    # Confusion matrix figure
    cm = confusion_matrix(test_metrics["labels"], test_metrics["preds"])
    model_name = os.path.basename(run_dir).replace("_", "-")
    save_confusion_matrix(
        cm,
        labels=label_names,
        path=os.path.join(run_dir, "confusion_matrix_test.png"),
        title=f"Confusion Matrix (Test) — {model_name}",
    )

    # Inference speed at batch size 1
    bs1_loader = DataLoader(
        TextEmotionDataset(test_df, tokenizer),
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
    )
    infer_ms = measure_inference_ms_per_sample(model, bs1_loader, device, num_inference_batches)

    return {
        "loss":                       test_metrics["loss"],
        "accuracy":                   test_metrics["accuracy"],
        "macro_f1":                   test_metrics["macro_f1"],
        "weighted_f1":                test_metrics["weighted_f1"],
        "inference_ms_per_sample_bs1": infer_ms,
    }
