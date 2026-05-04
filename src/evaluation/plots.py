from typing import List

import matplotlib.pyplot as plt
import numpy as np


def save_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    path: str,
    title: str,
):
    """
    Render a confusion matrix and save it to *path*.

    Args:
        cm:     Square array (rows = true class, cols = predicted class).
        labels: Class names in the same order as rows/columns.
        path:   Output file path, e.g. 'results/figures/cm_bert.png'.
        title:  Plot title shown at the top of the figure.
    """
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
