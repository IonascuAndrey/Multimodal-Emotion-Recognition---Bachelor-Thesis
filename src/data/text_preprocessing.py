from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding

from src.config import BATCH_SIZE, LABEL_COL, LABEL_REMAP, MAX_LEN, TEXT_COL


# ── Loading ───────────────────────────────────────────────────────────────────

def load_split(
    csv_path: str,
    label2id: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Load a MELD CSV split, normalise labels, and optionally encode them.

    Args:
        csv_path:  Path to the CSV file.
        label2id:  Emotion → int mapping. When provided, a 'label' column is
                   added and rows with unseen labels are dropped.

    Returns:
        DataFrame with columns: text, emotion[, label]
    """
    df = pd.read_csv(csv_path)

    if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(
            f"CSV '{csv_path}' must contain columns: {TEXT_COL}, {LABEL_COL}. "
            f"Found: {list(df.columns)}"
        )

    df = df[[TEXT_COL, LABEL_COL]].dropna().copy()
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.lower().str.strip()
    df[LABEL_COL] = df[LABEL_COL].replace(LABEL_REMAP)
    df.rename(columns={TEXT_COL: "text", LABEL_COL: "emotion"}, inplace=True)

    if label2id is not None:
        df["label"] = df["emotion"].map(label2id)
        df = df.dropna(subset=["label"]).copy()
        df["label"] = df["label"].astype(int)

    return df


def build_label_map(
    train_df: pd.DataFrame,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build label ↔ id mappings from the training split only (avoids leakage).

    Returns:
        label2id: {"anger": 0, "disgust": 1, ...}
        id2label: {0: "anger", 1: "disgust", ...}
    """
    labels_sorted = sorted(train_df["emotion"].unique().tolist())
    label2id = {lab: i for i, lab in enumerate(labels_sorted)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label


# ── Dataset ───────────────────────────────────────────────────────────────────

class TextEmotionDataset(Dataset):
    """
    PyTorch Dataset that tokenises utterances on-the-fly.
    Padding is deferred to DataCollatorWithPadding so every batch is
    padded only to its own longest sequence.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = MAX_LEN):
        self.texts     = df["text"].tolist()
        self.labels    = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            return_tensors=None,
        )
        enc["labels"] = self.labels[idx]
        return enc


# ── DataLoaders ───────────────────────────────────────────────────────────────

def make_loaders(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer,
    batch_size: int = BATCH_SIZE,
    max_len: int = MAX_LEN,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build DataLoaders for train / dev / test splits.

    Returns:
        (train_loader, dev_loader, test_loader)
    """
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    train_ds = TextEmotionDataset(train_df, tokenizer, max_len)
    dev_ds   = TextEmotionDataset(dev_df,   tokenizer, max_len)
    test_ds  = TextEmotionDataset(test_df,  tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collator)
    dev_loader   = DataLoader(dev_ds,   batch_size=batch_size, shuffle=False, collate_fn=collator)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collator)

    return train_loader, dev_loader, test_loader
