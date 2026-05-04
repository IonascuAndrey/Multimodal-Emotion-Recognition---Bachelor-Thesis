from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


@torch.no_grad()
def evaluate(model, loader, device: str) -> Dict[str, Any]:
    """
    Run inference over *loader* and compute loss + classification metrics.

    Args:
        model:   A HuggingFace model that returns an object with .loss and .logits.
        loader:  DataLoader yielding tokenised batches (must include 'labels').
        device:  'cuda' or 'cpu'.

    Returns:
        Dict with keys:
            loss         – mean cross-entropy over all batches
            accuracy     – overall accuracy
            macro_f1     – macro-averaged F1
            weighted_f1  – weighted-averaged F1
            preds        – np.ndarray of predicted class indices
            labels       – np.ndarray of ground-truth class indices
    """
    model.eval()
    losses     = []
    all_preds  = []
    all_labels = []

    for batch in loader:
        batch = {
            k: torch.tensor(v).to(device) if not torch.is_tensor(v) else v.to(device)
            for k, v in batch.items()
        }
        outputs = model(**batch)

        losses.append(outputs.loss.item())
        preds  = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        labels = batch["labels"].cpu().numpy()

        all_preds.append(preds)
        all_labels.append(labels)

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return {
        "loss":        float(np.mean(losses)),
        "accuracy":    float(accuracy_score(all_labels, all_preds)),
        "macro_f1":    float(f1_score(all_labels, all_preds, average="macro")),
        "weighted_f1": float(f1_score(all_labels, all_preds, average="weighted")),
        "preds":       all_preds,
        "labels":      all_labels,
    }
