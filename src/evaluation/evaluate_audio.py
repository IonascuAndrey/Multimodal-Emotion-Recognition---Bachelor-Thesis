import os
import time
from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from src.config import AUDIO_BATCH_SIZE, AUDIO_TEST_DIR, MAX_AUDIO_S, SAMPLE_RATE, TRIM_END_S, TRIM_START_S
from src.data.audio_preprocessing import AudioCollator, AudioEmotionDataset
from src.evaluation.metrics import evaluate
from src.evaluation.plots import save_confusion_matrix
from src.training.utils import write_json


# ── Inference speed ───────────────────────────────────────────────────────────

def measure_audio_inference_ms(
    model,
    loader: DataLoader,
    device: str,
    num_batches: int = 100,
) -> float:
    """
    Measure average forward-pass latency per sample (expects batch size = 1).

    Args:
        model:       Trained model in eval mode.
        loader:      DataLoader with batch_size=1.
        device:      'cuda' or 'cpu'.
        num_batches: How many batches to time after warmup.

    Returns:
        Mean inference time in milliseconds, or nan if loader is empty.
    """
    model.eval()
    times = []
    it    = iter(loader)

    # Warmup
    for _ in range(5):
        try:
            batch = next(it)
        except StopIteration:
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else torch.tensor(v).to(device)
                 for k, v in batch.items()}
        _ = model(**batch)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for _ in range(num_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else torch.tensor(v).to(device)
                 for k, v in batch.items()}
        t0 = time.time()
        _ = model(**batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000.0)

    return float(np.mean(times)) if times else float("nan")


# ── Full test-set evaluation ──────────────────────────────────────────────────

def run_full_audio_evaluation(
    run_dir: str,
    test_df,
    id2label: Dict[int, str],
    device: str,
    audio_test_dir: str = AUDIO_TEST_DIR,
    trim_start_s: float = TRIM_START_S,
    trim_end_s: float = TRIM_END_S,
    max_audio_s: float = MAX_AUDIO_S,
    num_inference_batches: int = 100,
) -> Dict[str, Any]:
    """
    Load the best saved audio checkpoint from *run_dir*, evaluate on *test_df*,
    and write the following files into *run_dir*:
        - classification_report_test.json
        - confusion_matrix_test.png
        - eval_metrics.json

    Args:
        run_dir:               Directory that contains a 'model/' subdirectory.
        test_df:               Test DataFrame with 'filename' and 'label' columns.
        id2label:              Mapping int → emotion string.
        device:                'cuda' or 'cpu'.
        audio_test_dir:        Directory with test .wav files.
        trim_start_s:          Seconds to cut from the start of each clip.
        trim_end_s:            Seconds to cut from the end of each clip.
        max_audio_s:           Maximum clip duration used during evaluation.
        num_inference_batches: Batches to time for the speed measurement.

    Returns:
        Dict with test metrics and inference speed.
    """
    model_path        = os.path.join(run_dir, "model")
    model             = AutoModelForAudioClassification.from_pretrained(model_path).to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    collator          = AudioCollator(feature_extractor, SAMPLE_RATE)

    ds_kwargs = dict(
        audio_dir=audio_test_dir,
        sample_rate=SAMPLE_RATE,
        max_audio_s=max_audio_s,
        trim_start_s=trim_start_s,
        trim_end_s=trim_end_s,
    )

    # Full-batch test loader for metric computation
    test_loader = DataLoader(
        AudioEmotionDataset(test_df, **ds_kwargs),
        batch_size=AUDIO_BATCH_SIZE,
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
        AudioEmotionDataset(test_df, **ds_kwargs),
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
    )
    infer_ms = measure_audio_inference_ms(model, bs1_loader, device, num_inference_batches)

    results = {
        "loss":                        test_metrics["loss"],
        "accuracy":                    test_metrics["accuracy"],
        "macro_f1":                    test_metrics["macro_f1"],
        "weighted_f1":                 test_metrics["weighted_f1"],
        "inference_ms_per_sample_bs1": infer_ms,
    }
    write_json(os.path.join(run_dir, "eval_metrics.json"), results)
    return results
