import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor

from src.config import (
    AUDIO_BATCH_SIZE,
    LABEL_COL,
    LABEL_REMAP,
    MAX_AUDIO_S,
    SAMPLE_RATE,
    TRIM_END_S,
    TRIM_START_S,
)


# ── CSV → audio DataFrame ─────────────────────────────────────────────────────

def load_audio_split(
    csv_path: str,
    audio_dir: str,
    label2id: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Load a MELD CSV split and build a DataFrame that links each row to its
    corresponding audio file via Dialogue_ID / Utterance_ID.

    The expected filename pattern is ``dia{Dialogue_ID}_utt{Utterance_ID}.wav``.
    Rows whose audio file is missing from *audio_dir* are dropped with a warning.

    Args:
        csv_path:  Path to the MELD CSV (train / dev / test).
        audio_dir: Directory that contains the .wav files for this split.
        label2id:  Emotion → int mapping. When provided, a 'label' column is
                   added and rows with unseen emotions are dropped.

    Returns:
        DataFrame with columns: filename, emotion[, label]
    """
    df = pd.read_csv(csv_path)

    required = {"Dialogue_ID", "Utterance_ID", LABEL_COL}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV '{csv_path}' is missing columns: {missing_cols}")

    # Build the wav filename from dialogue / utterance IDs
    df["filename"] = df.apply(
        lambda r: f"dia{int(r['Dialogue_ID'])}_utt{int(r['Utterance_ID'])}.wav",
        axis=1,
    )

    # Drop rows whose audio file doesn't exist in the split directory
    exists_mask = df["filename"].apply(
        lambda f: os.path.isfile(os.path.join(audio_dir, f))
    )
    n_missing = (~exists_mask).sum()
    if n_missing:
        print(f"  Warning: {n_missing} audio file(s) not found in '{audio_dir}' — dropped.")
    df = df[exists_mask].copy()

    # Normalise emotion labels (lower-case, remap joy → happiness)
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.lower().str.strip().replace(LABEL_REMAP)
    df.rename(columns={LABEL_COL: "emotion"}, inplace=True)

    if label2id is not None:
        df["label"] = df["emotion"].map(label2id)
        df = df.dropna(subset=["label"]).copy()
        df["label"] = df["label"].astype(int)
        return df[["filename", "emotion", "label"]].reset_index(drop=True)

    return df[["filename", "emotion"]].reset_index(drop=True)


# ── Dataset ───────────────────────────────────────────────────────────────────

class AudioEmotionDataset(Dataset):
    """
    PyTorch Dataset that loads raw waveforms on-the-fly.

    Each item is a dict with:
        ``input_values``: 1-D numpy float32 array of raw PCM samples.
        ``labels``:       integer class index.

    The feature extractor (and padding) is applied later in ``AudioCollator``
    so every batch is padded only to its own longest waveform.

    Trim parameters cut a fixed number of seconds from the start/end of every
    clip *before* truncation — useful to remove studio laughter tracks.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: str,
        sample_rate: int = SAMPLE_RATE,
        max_audio_s: float = MAX_AUDIO_S,
        trim_start_s: float = TRIM_START_S,
        trim_end_s: float = TRIM_END_S,
    ):
        self.df          = df.reset_index(drop=True)
        self.audio_dir   = audio_dir
        self.sample_rate = sample_rate
        self.max_samples = int(max_audio_s * sample_rate)
        self.trim_start  = int(trim_start_s * sample_rate)   # samples to remove from start
        self.trim_end    = int(trim_end_s   * sample_rate)   # samples to remove from end

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row      = self.df.iloc[idx]
        wav_path = os.path.join(self.audio_dir, row["filename"])

        # Load waveform (returns tensor of shape [channels, samples])
        waveform, sr = torchaudio.load(wav_path)

        # Resample to target rate if the file differs
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)

        # Convert stereo → mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Flatten to 1-D
        waveform = waveform.squeeze(0)   # shape: (n_samples,)

        # ── Trim ─────────────────────────────────────────────────────────────
        start = self.trim_start
        end   = len(waveform) - self.trim_end if self.trim_end > 0 else len(waveform)
        # Guard: if trim removes everything, keep a single silent sample
        if end <= start:
            waveform = waveform[:1]
        else:
            waveform = waveform[start:end]

        # ── Truncate to max length ────────────────────────────────────────────
        if len(waveform) > self.max_samples:
            waveform = waveform[: self.max_samples]

        return {
            "input_values": waveform.numpy().astype(np.float32),
            "labels":       int(row["label"]),
        }


# ── Collator ──────────────────────────────────────────────────────────────────

class AudioCollator:
    """
    Pads a list of variable-length waveforms to the longest in the batch using
    the HuggingFace feature extractor, then stacks labels.

    Pass an instance of this class as ``collate_fn`` to DataLoader.
    """

    def __init__(self, feature_extractor, sample_rate: int = SAMPLE_RATE):
        self.feature_extractor = feature_extractor
        self.sample_rate       = sample_rate

    def __call__(self, batch):
        waveforms = [item["input_values"] for item in batch]
        labels    = [item["labels"]       for item in batch]

        encoded = self.feature_extractor(
            waveforms,
            sampling_rate=self.sample_rate,
            padding=True,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded


# ── DataLoaders ───────────────────────────────────────────────────────────────

def make_audio_loaders(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    audio_dirs: Tuple[str, str, str],   # (train_dir, dev_dir, test_dir)
    feature_extractor,
    batch_size: int = AUDIO_BATCH_SIZE,
    sample_rate: int = SAMPLE_RATE,
    max_audio_s: float = MAX_AUDIO_S,
    trim_start_s: float = TRIM_START_S,
    trim_end_s: float = TRIM_END_S,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build DataLoaders for train / dev / test audio splits.

    Args:
        train_df, dev_df, test_df: DataFrames from ``load_audio_split``.
        audio_dirs:   Tuple of (train_dir, dev_dir, test_dir) paths.
        feature_extractor: HuggingFace feature extractor for the model.
        batch_size:   Samples per batch (keep low — waveforms are large).
        sample_rate:  Target sample rate in Hz.
        max_audio_s:  Waveforms longer than this are truncated (seconds).
        trim_start_s: Seconds to cut from the start of every clip.
        trim_end_s:   Seconds to cut from the end of every clip.

    Returns:
        (train_loader, dev_loader, test_loader)
    """
    collator = AudioCollator(feature_extractor, sample_rate)

    ds_kwargs = dict(
        sample_rate=sample_rate,
        max_audio_s=max_audio_s,
        trim_start_s=trim_start_s,
        trim_end_s=trim_end_s,
    )
    train_dir, dev_dir, test_dir = audio_dirs

    train_ds = AudioEmotionDataset(train_df, train_dir, **ds_kwargs)
    dev_ds   = AudioEmotionDataset(dev_df,   dev_dir,   **ds_kwargs)
    test_ds  = AudioEmotionDataset(test_df,  test_dir,  **ds_kwargs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collator)
    dev_loader   = DataLoader(dev_ds,   batch_size=batch_size, shuffle=False, collate_fn=collator)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collator)

    return train_loader, dev_loader, test_loader
