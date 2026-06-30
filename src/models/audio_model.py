from typing import Dict

from transformers import AutoModelForAudioClassification

# count_params is identical for any PyTorch model — re-export from text_model
from src.models.text_model import count_params  # noqa: F401


def load_audio_model(
    model_name: str,
    num_labels: int,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    device: str,
):
    """
    Load a HuggingFace audio-classification model and move it to device.

    Works with wav2vec2-based models (``facebook/wav2vec2-base``) and
    DistilHuBERT (``ntu-spml/distilhubert``).

    Args:
        model_name:  HuggingFace model identifier.
        num_labels:  Number of emotion classes.
        label2id:    Emotion string → int id.
        id2label:    Int id → emotion string.
        device:      'cuda' or 'cpu'.

    Returns:
        Model on the requested device.
    """
    model = AutoModelForAudioClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,   # classifier head is always re-initialised
    )
    return model.to(device)
