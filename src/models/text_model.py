from typing import Dict

from transformers import AutoModelForSequenceClassification


def load_model(
    model_name: str,
    num_labels: int,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    device: str,
):
    """
    Load a HuggingFace sequence-classification model and move it to device.

    Args:
        model_name:  HuggingFace model identifier (e.g. 'bert-base-uncased').
        num_labels:  Number of emotion classes.
        label2id:    Emotion string → int id.
        id2label:    Int id → emotion string.
        device:      'cuda' or 'cpu'.

    Returns:
        Model on the requested device.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )
    return model.to(device)


def count_params(model) -> Dict[str, int]:
    """Return total and trainable parameter counts for a model."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": int(total), "trainable_params": int(trainable)}
