"""
intermediate_fusion.py
----------------------
Feature-level (intermediate) fusion of the Text, Audio, and Face unimodal
classifiers trained on MELD.

Strategy
--------
1.  Extract penultimate embeddings from each frozen backbone by hooking the
    input to each model's final classification Linear.

    Embedding dimensions (per model):
        BERT / DistilBERT         768  (pooled CLS → model.classifier)
        wav2vec2 / DistilHuBERT   256  (projected mean-pool → model.classifier)
        ViT-B/16                  768  (CLS token → model.heads.head[-1])
        MobileNetV3-Small        1024  (classifier feat → model.classifier[-1])

2.  Cache embeddings to disk (one .npz per modality per split) so extraction
    only runs once.

3.  Align utterances across modalities and feed them to the fusion model
    as a per-modality dictionary {"text": (B, D_t), "audio": ...}.

4.  Train a small **ProjectionFusionMLP** on top of the embeddings:

        # per-modality projection (LayerNorm + Linear + GELU + Dropout)
        text  → Linear(D_t, D_proj) → LN → GELU → Dropout
        audio → Linear(D_a, D_proj) → LN → GELU → Dropout
        face  → Linear(D_f, D_proj) → LN → GELU → Dropout

        # fusion + classifier
        concat → Linear(3·D_proj, 256) → LN → GELU → Dropout
              → Linear(256, 7)

    The per-modality projection re-scales each modality to a common subspace
    (D_proj=128) so the fusion head sees comparable feature magnitudes —
    without this step text/face dominate the audio signal (which has a
    different distribution and smaller dim).

    Backbones are kept frozen throughout; only the projection + classifier
    are trained.

5.  Evaluate on the test set and write artefacts.

Compared to the original implementation, this version:
  * Adds per-modality LayerNorm + projection to align scales.
  * Adds modality dropout during training (full-modality masking).
  * Uses balanced class weights in the loss.
  * Uses AdamW + cosine LR with warmup and patience-based early stopping.
  * Returns gating/projection statistics so the notebook can render
    explainability plots.

Usage
-----
See notebooks/31_intermediate_fusion.ipynb.
"""

from __future__ import annotations

import json
import os
import re
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.config import (
    AUDIO_BATCH_SIZE,
    AUDIO_DEV_DIR,
    AUDIO_TEST_DIR,
    BATCH_SIZE,
    DEVICE,
    FACE_CACHE_DIR,
    FACE_MELD_CLASSES,
    FACE_TRAIN_BATCH_SIZE,
    FACE_USE_CACHE,
    MAX_AUDIO_S,
    MAX_LEN,
    SAMPLE_RATE,
    TRIM_END_S,
    TRIM_START_S,
    VERSION,
)
from src.data.audio_preprocessing import AudioCollator, AudioEmotionDataset
from src.data.face_dataset import (
    ID2LABEL as FACE_ID2LABEL,
    LABEL2ID as FACE_LABEL2ID,
    NUM_CLASSES,
    FaceEmotionDataset,
    _cache_dir_path,
    build_cache,
    get_transforms,
)
from src.models.face_model import load_face_model
from src.evaluation.plots import save_confusion_matrix
from src.training.utils import write_json


# ── Canonical label order ─────────────────────────────────────────────────────
_CLASSES   = FACE_MELD_CLASSES
_N         = NUM_CLASSES
_CANON_L2I = {c.lower(): i for i, c in enumerate(_CLASSES)}
_CANON_I2L = {i: c for i, c in enumerate(_CLASSES)}

# Type aliases
EmbMap   = Dict[Tuple[int, int], np.ndarray]  # {(d_id, u_id): embedding (D,)}
LabelMap = Dict[Tuple[int, int], int]          # {(d_id, u_id): canonical label}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_remap(id2label_train: Dict[int, str]) -> np.ndarray:
    remap = np.zeros(_N, dtype=int)
    for model_idx, emotion_str in id2label_train.items():
        remap[int(model_idx)] = _CANON_L2I[str(emotion_str).lower()]
    return remap


@contextmanager
def _capture_input(module: nn.Module):
    """
    Context manager that captures the input tensor of *module* on the next
    forward pass.  Yields a list that will be populated after the forward call.
    """
    captured: List[torch.Tensor] = []
    handle = module.register_forward_pre_hook(
        lambda m, inp: captured.append(inp[0].detach().cpu())
    )
    try:
        yield captured
    finally:
        handle.remove()


def _find_head_linear(model: nn.Module, source: str, model_type: str) -> nn.Module:
    """
    Return the final classification Linear layer so we can hook its input.

    Parameters
    ----------
    source     : "hf" for HuggingFace models, "tv" for torchvision
    model_type : HF model_type string or torchvision model name
    """
    if source == "hf":
        # All HF sequence/audio classification models expose .classifier
        clf = model.classifier
        if isinstance(clf, nn.Linear):
            return clf
        if isinstance(clf, nn.Sequential):
            return clf[-1]
        # Last Linear child (e.g. DistilBERT's classifier is directly a Linear)
        for layer in reversed(list(clf.children())):
            if isinstance(layer, nn.Linear):
                return layer
        return clf  # fallback

    elif source == "tv":
        if model_type == "vit_b_16":
            # model.heads.head = Sequential(Dropout, Linear(768, 7))
            return model.heads.head[-1]
        elif model_type == "mobilenet_v3_small":
            # model.classifier = Sequential(..., Linear(1024, 7))
            return model.classifier[-1]
        else:
            raise ValueError(f"Unknown torchvision model type: {model_type}")

    raise ValueError(f"Unknown source: {source}")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Embedding extraction (backbone frozen, hook on head input)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_text_embeddings(
    run_dir:    str,
    csv_path:   str,
    device:     str = DEVICE,
    batch_size: int = BATCH_SIZE,
    max_len:    int = MAX_LEN,
) -> Tuple[EmbMap, LabelMap]:
    """
    Extract penultimate embeddings from a trained text classification model.

    Hooks the input to model.classifier (the final Linear) to obtain the
    task-specific representation learned during fine-tuning.
    """
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    id2label_train = {int(k): v for k, v in cfg["id2label"].items()}
    remap = _build_remap(id2label_train)

    hf_cfg_path = os.path.join(run_dir, "model", "config.json")
    with open(hf_cfg_path) as f:
        hf_cfg = json.load(f)
    model_type = hf_cfg.get("model_type", "bert")

    model_path = os.path.join(run_dir, "model")
    model     = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    head_linear = _find_head_linear(model, source="hf", model_type=model_type)

    df = pd.read_csv(csv_path)
    df["Emotion"] = df["Emotion"].astype(str).str.lower().str.strip().replace({"joy": "happiness"})
    df = df.dropna(subset=["Utterance", "Emotion", "Dialogue_ID", "Utterance_ID"])

    texts    = df["Utterance"].tolist()
    d_ids    = df["Dialogue_ID"].astype(int).tolist()
    u_ids    = df["Utterance_ID"].astype(int).tolist()
    emotions = df["Emotion"].tolist()

    emb_map: EmbMap   = {}
    label_map: LabelMap = {}

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        enc = tokenizer(
            texts[start:end],
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with _capture_input(head_linear) as captured:
            model(**enc)

        embs = captured[0].numpy()  # (B, D)

        for i in range(end - start):
            key     = (d_ids[start + i], u_ids[start + i])
            emotion = emotions[start + i]
            emb_map[key] = embs[i]
            if emotion in _CANON_L2I:
                label_map[key] = _CANON_L2I[emotion]

    return emb_map, label_map


@torch.no_grad()
def extract_audio_embeddings(
    run_dir:   str,
    df:        pd.DataFrame,
    audio_dir: str = AUDIO_TEST_DIR,
    device:    str = DEVICE,
) -> Tuple[EmbMap, LabelMap]:
    """
    Extract penultimate embeddings from a trained audio classification model.
    """
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    id2label_train = {int(k): v for k, v in cfg["id2label"].items()}
    remap = _build_remap(id2label_train)

    hf_cfg_path = os.path.join(run_dir, "model", "config.json")
    with open(hf_cfg_path) as f:
        hf_cfg = json.load(f)
    model_type = hf_cfg.get("model_type", "wav2vec2")

    model_path        = os.path.join(run_dir, "model")
    model             = AutoModelForAudioClassification.from_pretrained(model_path).to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    collator          = AudioCollator(feature_extractor, SAMPLE_RATE)
    model.eval()

    head_linear = _find_head_linear(model, source="hf", model_type=model_type)

    loader = DataLoader(
        AudioEmotionDataset(
            df,
            audio_dir=audio_dir,
            sample_rate=SAMPLE_RATE,
            max_audio_s=MAX_AUDIO_S,
            trim_start_s=TRIM_START_S,
            trim_end_s=TRIM_END_S,
        ),
        batch_size=AUDIO_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )

    _pat = re.compile(r"dia(\d+)_utt(\d+)", re.IGNORECASE)
    keys: List[Optional[Tuple[int, int]]] = []
    for fname in df["filename"].tolist():
        m = _pat.search(fname)
        keys.append((int(m.group(1)), int(m.group(2))) if m else None)

    emb_map: EmbMap   = {}
    label_map: LabelMap = {}
    sample_idx = 0

    for batch in loader:
        batch_d = {
            k: (v.to(device) if torch.is_tensor(v) else torch.tensor(v).to(device))
            for k, v in batch.items()
        }
        with _capture_input(head_linear) as captured:
            model(**batch_d)

        embs   = captured[0].numpy()          # (B, D)
        labels = batch["labels"].cpu().numpy()

        for i in range(len(labels)):
            key = keys[sample_idx]
            if key is not None:
                emb_map[key]   = embs[i]
                label_map[key] = int(remap[int(labels[i])])
            sample_idx += 1

    return emb_map, label_map


@torch.no_grad()
def extract_face_embeddings(
    run_dir:     str,
    df:          pd.DataFrame,
    device:      str  = DEVICE,
    batch_size:  int  = FACE_TRAIN_BATCH_SIZE,
    use_cache:   bool = FACE_USE_CACHE,
    cache_dir:   str  = FACE_CACHE_DIR,
    version:     str  = VERSION,
    num_workers: int  = 4,
) -> Tuple[EmbMap, LabelMap]:
    """
    Extract penultimate embeddings from a trained face classification model.
    Frame-level embeddings are averaged per utterance.
    """
    model_dir  = os.path.join(run_dir, "model")
    with open(os.path.join(model_dir, "arch.txt")) as f:
        model_name = f.read().strip()

    model = load_face_model(model_name, num_classes=NUM_CLASSES, device=device)
    state = torch.load(os.path.join(model_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(state)
    model.eval()

    head_linear = _find_head_linear(model, source="tv", model_type=model_name)

    # Resolve cache split: try "test", then "dev"
    cdir = None
    if use_cache:
        for split_name in ("test", "dev", "train"):
            candidate = _cache_dir_path(split_name, version, cache_dir)
            if os.path.isdir(candidate):
                cdir = candidate
                break
        if cdir is None:
            build_cache(df, "test", version=version, cache_dir=cache_dir)
            cdir = _cache_dir_path("test", version, cache_dir)

    ds = FaceEmotionDataset(
        df=df,
        transform=get_transforms(train=False),
        use_score_weights=False,
        use_cache=use_cache,
        cache_dir=cdir,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    utt_embs: Dict[Tuple[int, int], List[np.ndarray]] = {}
    utt_label: LabelMap = {}

    for images, labels, _weights, d_ids, u_ids in loader:
        images = images.to(device)
        with torch.amp.autocast(device, enabled=(device == "cuda")):
            with _capture_input(head_linear) as captured:
                model(images)

        embs = captured[0].numpy()  # (B, D)

        for i in range(len(labels)):
            key = (int(d_ids[i]), int(u_ids[i]))
            utt_embs.setdefault(key, []).append(embs[i])
            utt_label[key] = int(labels[i])

    emb_map: EmbMap = {
        key: np.mean(utt_embs[key], axis=0)
        for key in utt_label
    }
    return emb_map, utt_label


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Disk cache  (extracted once, reused across training runs)
# ─────────────────────────────────────────────────────────────────────────────

def save_embedding_cache(
    emb_map:   EmbMap,
    label_map: LabelMap,
    path:      str,
) -> None:
    common_keys = sorted(set(emb_map.keys()) & set(label_map.keys()))
    d_ids  = np.array([k[0] for k in common_keys], dtype=np.int32)
    u_ids  = np.array([k[1] for k in common_keys], dtype=np.int32)
    embs   = np.array([emb_map[k] for k in common_keys], dtype=np.float32)
    labels = np.array([label_map[k] for k in common_keys], dtype=np.int32)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, d_ids=d_ids, u_ids=u_ids,
                        embeddings=embs, labels=labels)
    print(f"  Saved {len(common_keys)} embeddings  (dim={embs.shape[1]})  → {path}")


def load_embedding_cache(path: str) -> Tuple[EmbMap, LabelMap]:
    data   = np.load(path)
    d_ids  = data["d_ids"]
    u_ids  = data["u_ids"]
    embs   = data["embeddings"]
    labels = data["labels"]

    emb_map: EmbMap   = {}
    label_map: LabelMap = {}
    for i in range(len(d_ids)):
        key = (int(d_ids[i]), int(u_ids[i]))
        emb_map[key]   = embs[i]
        label_map[key] = int(labels[i])
    return emb_map, label_map


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Fusion Dataset  (per-modality dict, not concatenated)
# ─────────────────────────────────────────────────────────────────────────────

class FusionDataset(Dataset):
    """
    Aligns per-modality embedding maps by utterance key.

    Returns a (features_dict, label) tuple where features_dict maps each
    modality name to its tensor.  This makes it easy for the model to apply
    different transformations to each modality (e.g. per-modality projection
    or gating).

    Only utterances present in ALL modalities are included.  Ground-truth
    labels come from *label_map* (typically the text label map).
    """

    def __init__(
        self,
        named_embs: Dict[str, EmbMap],
        label_map:  LabelMap,
    ):
        modalities = list(named_embs.keys())
        common_keys: set = set(named_embs[modalities[0]].keys())
        for m in modalities[1:]:
            common_keys &= set(named_embs[m].keys())
        common_keys &= set(label_map.keys())

        self.keys       = sorted(common_keys)
        self.modalities = modalities
        self.named_embs = named_embs
        self.label_map  = label_map

        sample_key = self.keys[0]
        # Per-modality dim and total concat dim (useful for legacy code)
        self.dims    = {m: int(named_embs[m][sample_key].shape[0]) for m in modalities}
        self.emb_dim = sum(self.dims.values())

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int):
        key = self.keys[idx]
        feats = {
            m: torch.from_numpy(np.asarray(self.named_embs[m][key])).float()
            for m in self.modalities
        }
        y = torch.tensor(self.label_map[key], dtype=torch.long)
        return feats, y


def _collate_dict(batch):
    """Default collate for FusionDataset: stack features per modality."""
    feats_list, ys = zip(*batch)
    modalities = list(feats_list[0].keys())
    feats = {m: torch.stack([f[m] for f in feats_list], dim=0) for m in modalities}
    y     = torch.stack(ys, dim=0)
    return feats, y


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Fusion model — per-modality projection + small fusion head
# ─────────────────────────────────────────────────────────────────────────────

class ProjectionFusionMLP(nn.Module):
    """
    Per-modality projection followed by concatenation and a small MLP head.

    For each modality m:
        h_m = Linear(D_m, D_proj)(LayerNorm(x_m))
        h_m = Dropout(GELU(h_m))

    Then:
        z = concat([h_m for m in modalities])      # (B, M·D_proj)
        z = Dropout(GELU(LayerNorm(Linear(z))))    # (B, D_hidden)
        logits = Linear(z)                          # (B, num_classes)

    Compared to the original `FusionMLP`, the per-modality projection puts
    all modalities on the same scale and the same dimension, preventing one
    modality (here: text/face vs the smaller audio embedding) from
    dominating purely through magnitude or feature count.

    Modality dropout
    ----------------
    During training, with probability *modality_dropout*, each modality is
    fully zeroed out (independently per sample).  Forces the head to learn
    to rely on every modality at least some of the time.

    Parameters
    ----------
    in_dims          : per-modality input dim, e.g. {"text": 768, ...}
    d_proj           : projection dim per modality
    d_hidden         : fusion-head bottleneck dim
    num_classes      : number of output classes
    dropout          : dropout probability inside the head
    modality_dropout : probability that an entire modality is zeroed during
                       training (0.0 disables).
    """

    def __init__(
        self,
        in_dims:           Dict[str, int],
        d_proj:            int   = 128,
        d_hidden:          int   = 256,
        num_classes:       int   = _N,
        dropout:           float = 0.5,
        modality_dropout:  float = 0.2,
    ):
        super().__init__()
        self.modalities       = list(in_dims.keys())
        self.d_proj           = d_proj
        self.modality_dropout = modality_dropout

        # Per-modality projection branches.
        self.projections = nn.ModuleDict({
            m: nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d_proj),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for m, d in in_dims.items()
        })

        in_fusion = d_proj * len(in_dims)
        self.head = nn.Sequential(
            nn.Linear(in_fusion, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, num_classes),
        )

    def forward(
        self,
        feats: Dict[str, torch.Tensor],
        return_proj: bool = False,
    ) -> torch.Tensor:
        proj = {m: self.projections[m](feats[m]) for m in self.modalities}

        if self.training and self.modality_dropout > 0.0:
            # Sample per-sample, per-modality masks; rescale to keep expectation.
            B = next(iter(proj.values())).size(0)
            device = next(iter(proj.values())).device
            scale  = 1.0 / max(1e-6, 1.0 - self.modality_dropout)
            for m in self.modalities:
                mask = (torch.rand(B, 1, device=device) >= self.modality_dropout).float()
                proj[m] = proj[m] * mask * scale

        z      = torch.cat([proj[m] for m in self.modalities], dim=-1)
        logits = self.head(z)

        if return_proj:
            return logits, proj
        return logits


# Backwards-compatible alias for old code that imports `FusionMLP`.
class FusionMLP(ProjectionFusionMLP):
    """
    Backward-compatible wrapper around :class:`ProjectionFusionMLP`.

    Accepts the legacy ``in_dim`` argument (total concatenated dim).  In that
    case we approximate per-modality dims as equal splits — but new code
    should pass an `in_dims` dict instead.
    """

    def __init__(
        self,
        in_dim:           Optional[int]              = None,
        in_dims:          Optional[Dict[str, int]]   = None,
        num_classes:      int   = _N,
        dropout:          float = 0.5,
        d_proj:           int   = 128,
        d_hidden:         int   = 256,
        modality_dropout: float = 0.2,
    ):
        if in_dims is None:
            assert in_dim is not None, "Provide either in_dims or in_dim."
            # Legacy entry point — treat as a single 'all' modality.
            in_dims = {"all": int(in_dim)}

        super().__init__(
            in_dims=in_dims,
            d_proj=d_proj,
            d_hidden=d_hidden,
            num_classes=num_classes,
            dropout=dropout,
            modality_dropout=modality_dropout,
        )

    # Legacy `forward(x)` that took a single concatenated tensor.
    def forward(
        self,
        x_or_feats,
        return_proj: bool = False,
    ):
        if isinstance(x_or_feats, dict):
            return super().forward(x_or_feats, return_proj=return_proj)
        return super().forward({"all": x_or_feats}, return_proj=return_proj)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Training & Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_class_weights(train_ds: FusionDataset, num_classes: int) -> torch.Tensor:
    """Balanced inverse-frequency class weights from the train dataset labels."""
    labels = np.array([train_ds.label_map[k] for k in train_ds.keys], dtype=np.int64)
    classes_present = np.unique(labels)
    w = compute_class_weight("balanced", classes=classes_present, y=labels)
    weights = np.ones(num_classes, dtype=np.float32)
    for cls, wv in zip(classes_present, w):
        weights[int(cls)] = float(wv)
    return torch.tensor(weights, dtype=torch.float32)


def train_fusion_mlp(
    mlp:               nn.Module,
    train_ds:          FusionDataset,
    dev_ds:            FusionDataset,
    epochs:            int   = 80,
    batch_size:        int   = 128,
    lr:                float = 5e-4,
    weight_decay:      float = 1e-2,
    label_smoothing:   float = 0.05,
    device:            str   = DEVICE,
    num_workers:       int   = 0,
    use_class_weights: bool  = True,
    patience:          int   = 12,
    warmup_epochs:     int   = 3,
    grad_clip:         float = 1.0,
    verbose:           bool  = True,
) -> Tuple[nn.Module, List[Dict]]:
    """
    Train the fusion model and return (best_model, training_history).

    Improvements over the original loop:
      * Balanced class weights in CE loss (helps macro-F1 on imbalanced MELD).
      * Linear warmup → cosine decay over the remaining epochs.
      * Patience-based early stopping on dev macro-F1.
      * Gradient clipping.
    """
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=_collate_dict,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_dict,
    )

    mlp = mlp.to(device)
    optimizer = AdamW(mlp.parameters(), lr=lr, weight_decay=weight_decay)

    # Warmup + cosine schedule, per-step.
    total_steps    = max(1, len(train_loader) * epochs)
    warmup_steps   = max(1, len(train_loader) * warmup_epochs)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=lr * 0.05,
    )

    class_weights = (
        _compute_class_weights(train_ds, num_classes=_N).to(device)
        if use_class_weights else None
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing,
    )

    best_f1, best_state, patience_ctr = -1.0, None, 0
    history: List[Dict] = []
    step = 0

    for epoch in range(1, epochs + 1):
        mlp.train()
        ep_loss, ep_n = 0.0, 0

        for feats, y in train_loader:
            feats = {m: v.to(device) for m, v in feats.items()}
            y     = y.to(device)
            logits = mlp(feats)
            loss   = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(mlp.parameters(), grad_clip)
            optimizer.step()

            # LR schedule: linear warmup then cosine
            step += 1
            if step <= warmup_steps:
                for pg in optimizer.param_groups:
                    pg["lr"] = lr * step / warmup_steps
            else:
                cosine.step()

            ep_loss += loss.item()
            ep_n    += 1

        dev_metrics = _evaluate_mlp(mlp, dev_loader, device)
        row = {
            "epoch":        epoch,
            "train_loss":   round(ep_loss / max(ep_n, 1), 6),
            "dev_loss":     round(dev_metrics["loss"], 6),
            "dev_acc":      round(dev_metrics["accuracy"], 6),
            "dev_macro_f1": round(dev_metrics["macro_f1"], 6),
            "lr":           round(optimizer.param_groups[0]["lr"], 7),
        }
        history.append(row)

        if dev_metrics["macro_f1"] > best_f1 + 1e-5:
            best_f1      = dev_metrics["macro_f1"]
            best_state   = {k: v.detach().clone() for k, v in mlp.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
            print(
                f"  [Epoch {epoch:03d}/{epochs}]  "
                f"train_loss={row['train_loss']:.4f}  "
                f"dev_loss={row['dev_loss']:.4f}  "
                f"dev_macro_f1={row['dev_macro_f1']:.4f}  "
                f"(best={best_f1:.4f})"
            )

        if patience_ctr >= patience:
            if verbose:
                print(f"  Early stop @ epoch {epoch} (no improvement for {patience} epochs)")
            break

    if best_state:
        mlp.load_state_dict(best_state)
    if verbose:
        print(f"\n  Best dev macro-F1: {best_f1:.4f}")
    return mlp, history


@torch.no_grad()
def _evaluate_mlp(
    mlp:    nn.Module,
    loader: DataLoader,
    device: str,
) -> Dict:
    """Compute loss + metrics on a DataLoader."""
    mlp.eval()
    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels, total_loss, n = [], [], 0.0, 0

    for feats, y in loader:
        feats = {m: v.to(device) for m, v in feats.items()}
        y     = y.to(device)
        logits = mlp(feats)
        total_loss += criterion(logits, y).item()
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y.cpu().numpy())
        n += 1

    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return {
        "loss":        total_loss / max(n, 1),
        "accuracy":    float(accuracy_score(labels, preds)),
        "macro_f1":    float(f1_score(labels, preds, average="macro",    zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "preds":       preds,
        "labels":      labels,
    }


def evaluate_fusion_mlp(
    mlp:      nn.Module,
    test_ds:  FusionDataset,
    id2label: Optional[Dict[int, str]] = None,
    device:   str = DEVICE,
) -> Dict:
    """Evaluate the fusion model on a test FusionDataset."""
    if id2label is None:
        id2label = _CANON_I2L

    loader  = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=_collate_dict)
    metrics = _evaluate_mlp(mlp, loader, device)

    label_names = [id2label[i] for i in range(_N)]
    report = classification_report(
        metrics["labels"], metrics["preds"],
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(
        metrics["labels"], metrics["preds"],
        labels=list(range(_N)),
    )
    return {
        "accuracy":       round(metrics["accuracy"],    6),
        "macro_f1":       round(metrics["macro_f1"],    6),
        "weighted_f1":    round(metrics["weighted_f1"], 6),
        "num_utterances": len(test_ds),
        "preds":          metrics["preds"].tolist(),
        "labels":         metrics["labels"].tolist(),
        "report":         report,
        "cm":             cm,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Modality ablation (explainability)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def modality_ablation(
    mlp:      nn.Module,
    test_ds:  FusionDataset,
    device:   str = DEVICE,
) -> Dict[str, Dict]:
    """
    For each modality, zero out its features at inference and re-score the
    test set.  The resulting drop in macro-F1 is a simple proxy for that
    modality's contribution.

    Returns a dict like::

        {
            "full":          {"macro_f1": 0.51, ...},
            "no_text":       {"macro_f1": 0.41, ...},
            "no_audio":      {"macro_f1": 0.50, ...},
            "no_face":       {"macro_f1": 0.48, ...},
            "text_only":     {"macro_f1": 0.38, ...},
            "audio_only":    {"macro_f1": 0.22, ...},
            "face_only":     {"macro_f1": 0.20, ...},
        }
    """
    mlp.eval()
    modalities = list(test_ds.modalities)
    loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=_collate_dict)

    def run(mask_modalities=None, keep_only=None):
        """Run inference with selected modalities zeroed."""
        preds_all, labels_all = [], []
        for feats, y in loader:
            feats_d = {m: feats[m].to(device) for m in modalities}
            for m in modalities:
                if mask_modalities and m in mask_modalities:
                    feats_d[m] = torch.zeros_like(feats_d[m])
                if keep_only and m != keep_only:
                    feats_d[m] = torch.zeros_like(feats_d[m])
            logits = mlp(feats_d)
            preds_all.append(logits.argmax(1).cpu().numpy())
            labels_all.append(y.numpy())
        preds  = np.concatenate(preds_all)
        labels = np.concatenate(labels_all)
        return {
            "accuracy":    float(accuracy_score(labels, preds)),
            "macro_f1":    float(f1_score(labels, preds, average="macro",    zero_division=0)),
            "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        }

    out = {"full": run()}
    for m in modalities:
        out[f"no_{m}"]    = run(mask_modalities={m})
        out[f"{m}_only"]  = run(keep_only=m)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_intermediate_fusion(
    # Unimodal run directories (best model per modality)
    text_run_dir:  str,
    audio_run_dir: str,
    face_run_dir:  str,
    # Data
    train_csv_path: str,
    dev_csv_path:   str,
    test_csv_path:  str,
    train_audio_df: pd.DataFrame,
    dev_audio_df:   pd.DataFrame,
    test_audio_df:  pd.DataFrame,
    train_face_df:  pd.DataFrame,
    dev_face_df:    pd.DataFrame,
    test_face_df:   pd.DataFrame,
    # MLP hyperparameters
    mlp_epochs:           int   = 80,
    mlp_batch_size:       int   = 128,
    mlp_lr:               float = 5e-4,
    mlp_weight_decay:     float = 1e-2,
    mlp_label_smoothing:  float = 0.05,
    mlp_dropout:          float = 0.5,
    mlp_d_proj:           int   = 128,
    mlp_d_hidden:         int   = 256,
    mlp_modality_dropout: float = 0.2,
    mlp_patience:         int   = 12,
    # Cache / output
    cache_dir:  Optional[str] = None,
    output_dir: Optional[str] = None,
    # Infrastructure
    device:          str = DEVICE,
    audio_train_dir: str = None,
    audio_dev_dir:   str = AUDIO_DEV_DIR,
    audio_test_dir:  str = AUDIO_TEST_DIR,
) -> Dict:
    """
    Full intermediate-fusion pipeline.

    1. Extract (or load from cache) penultimate embeddings for each modality.
    2. Build FusionDataset instances by aligning embeddings.
    3. Train a ProjectionFusionMLP with class weights + modality dropout,
       early-stopping on dev macro-F1.
    4. Evaluate on the test set, run modality-ablation explainability, and
       write artefacts.
    """
    from src.config import AUDIO_TRAIN_DIR
    if audio_train_dir is None:
        audio_train_dir = AUDIO_TRAIN_DIR
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if cache_dir is None:
        cache_dir = os.path.join(output_dir or ".", "cache")
    os.makedirs(cache_dir, exist_ok=True)

    id2label = _CANON_I2L

    # ── Step 1: Extract / load embeddings ────────────────────────────────────
    def _get_embs(extractor_fn, cache_name, *args, **kwargs):
        cache_path = os.path.join(cache_dir, cache_name + ".npz")
        if os.path.exists(cache_path):
            print(f"  Loading cache: {cache_name}")
            return load_embedding_cache(cache_path)
        print(f"  Extracting  : {cache_name}")
        emb_map, label_map = extractor_fn(*args, **kwargs)
        save_embedding_cache(emb_map, label_map, cache_path)
        return emb_map, label_map

    print("=" * 60)
    print("  Intermediate Fusion — embedding extraction")
    print("=" * 60)

    text_emb_train, text_lbl_train = _get_embs(
        extract_text_embeddings, "text_train",
        text_run_dir, train_csv_path, device
    )
    text_emb_dev, text_lbl_dev = _get_embs(
        extract_text_embeddings, "text_dev",
        text_run_dir, dev_csv_path, device
    )
    text_emb_test, text_lbl_test = _get_embs(
        extract_text_embeddings, "text_test",
        text_run_dir, test_csv_path, device
    )

    audio_emb_train, _ = _get_embs(
        extract_audio_embeddings, "audio_train",
        audio_run_dir, train_audio_df, audio_train_dir, device
    )
    audio_emb_dev, _ = _get_embs(
        extract_audio_embeddings, "audio_dev",
        audio_run_dir, dev_audio_df, audio_dev_dir, device
    )
    audio_emb_test, _ = _get_embs(
        extract_audio_embeddings, "audio_test",
        audio_run_dir, test_audio_df, audio_test_dir, device
    )

    face_emb_train, _ = _get_embs(
        extract_face_embeddings, "face_train",
        face_run_dir, train_face_df, device
    )
    face_emb_dev, _ = _get_embs(
        extract_face_embeddings, "face_dev",
        face_run_dir, dev_face_df, device
    )
    face_emb_test, _ = _get_embs(
        extract_face_embeddings, "face_test",
        face_run_dir, test_face_df, device
    )

    # ── Step 2: Build datasets ────────────────────────────────────────────────
    named_train = {"text": text_emb_train, "audio": audio_emb_train, "face": face_emb_train}
    named_dev   = {"text": text_emb_dev,   "audio": audio_emb_dev,   "face": face_emb_dev}
    named_test  = {"text": text_emb_test,  "audio": audio_emb_test,  "face": face_emb_test}

    train_ds = FusionDataset(named_train, text_lbl_train)
    dev_ds   = FusionDataset(named_dev,   text_lbl_dev)
    test_ds  = FusionDataset(named_test,  text_lbl_test)

    in_dims  = train_ds.dims
    total_dim = train_ds.emb_dim
    print(f"\n  Embedding dims : {in_dims}  →  total = {total_dim}")
    print(f"  Train utterances: {len(train_ds)}   Dev: {len(dev_ds)}   Test: {len(test_ds)}")

    # ── Step 3: Train MLP ─────────────────────────────────────────────────────
    print(f"\n  Training Projection Fusion MLP  (d_proj={mlp_d_proj}, d_hidden={mlp_d_hidden})")
    mlp = ProjectionFusionMLP(
        in_dims          = in_dims,
        d_proj           = mlp_d_proj,
        d_hidden         = mlp_d_hidden,
        num_classes      = _N,
        dropout          = mlp_dropout,
        modality_dropout = mlp_modality_dropout,
    )
    n_params = sum(p.numel() for p in mlp.parameters())
    print(f"  MLP params: {n_params:,}")

    t0 = time.time()
    mlp, history = train_fusion_mlp(
        mlp,
        train_ds,
        dev_ds,
        epochs            = mlp_epochs,
        batch_size        = mlp_batch_size,
        lr                = mlp_lr,
        weight_decay      = mlp_weight_decay,
        label_smoothing   = mlp_label_smoothing,
        device            = device,
        use_class_weights = True,
        patience          = mlp_patience,
    )
    elapsed = time.time() - t0
    print(f"  Training time: {elapsed/60:.1f} min")

    # ── Step 4: Evaluate on test set ──────────────────────────────────────────
    print("\n  Evaluating on test set …")
    test_metrics = evaluate_fusion_mlp(mlp, test_ds, id2label, device)
    print(f"  Accuracy    : {test_metrics['accuracy']:.4f}")
    print(f"  Macro F1    : {test_metrics['macro_f1']:.4f}")
    print(f"  Weighted F1 : {test_metrics['weighted_f1']:.4f}")

    # ── Explainability: modality ablation ─────────────────────────────────────
    print("\n  Modality ablation (zeroing out modalities at inference) …")
    ablation = modality_ablation(mlp, test_ds, device=device)
    for k, v in ablation.items():
        print(f"    {k:14s}  macro_f1={v['macro_f1']:.4f}  acc={v['accuracy']:.4f}")

    # ── Save artefacts ────────────────────────────────────────────────────────
    if output_dir:
        label_names = [id2label[i] for i in range(_N)]

        torch.save(mlp.state_dict(), os.path.join(output_dir, "fusion_mlp.pt"))

        write_json(os.path.join(output_dir, "test_metrics.json"), {
            k: v for k, v in test_metrics.items()
            if k not in ("preds", "labels", "report", "cm")
        })
        write_json(os.path.join(output_dir, "classification_report.json"),
                   test_metrics["report"])
        write_json(os.path.join(output_dir, "modality_ablation.json"), ablation)
        save_confusion_matrix(
            np.array(test_metrics["cm"]),
            label_names,
            os.path.join(output_dir, "confusion_matrix_test.png"),
            title="Intermediate Fusion — Projection MLP (test)",
        )
        pd.DataFrame(history).to_csv(
            os.path.join(output_dir, "history.csv"), index=False
        )
        write_json(os.path.join(output_dir, "config.json"), {
            "fusion_type":         "intermediate_projection_mlp",
            "text_run_dir":        text_run_dir,
            "audio_run_dir":       audio_run_dir,
            "face_run_dir":        face_run_dir,
            "emb_dims":            in_dims,
            "total_dim":           total_dim,
            "mlp_epochs":          mlp_epochs,
            "mlp_batch_size":      mlp_batch_size,
            "mlp_lr":              mlp_lr,
            "mlp_weight_decay":    mlp_weight_decay,
            "mlp_label_smoothing": mlp_label_smoothing,
            "mlp_dropout":         mlp_dropout,
            "mlp_d_proj":          mlp_d_proj,
            "mlp_d_hidden":        mlp_d_hidden,
            "mlp_modality_dropout": mlp_modality_dropout,
            "mlp_patience":        mlp_patience,
            "mlp_total_params":    int(n_params),
            "training_time_s":     round(elapsed, 1),
        })
        print(f"\n  Artefacts saved to  {output_dir}")

    return {
        "test_metrics":   test_metrics,
        "history":        history,
        "emb_dims":       in_dims,
        "mlp":            mlp,
        "ablation":       ablation,
        "training_time_s": elapsed,
        "num_params":     int(n_params),
    }
