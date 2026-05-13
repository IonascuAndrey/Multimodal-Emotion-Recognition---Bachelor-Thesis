"""
gated_fusion.py
---------------
Gated multimodal fusion of the Text, Audio, and Face unimodal classifiers
trained on MELD.

Two variants are implemented:

* **GatedFusionLight** — per-modality projection followed by a single soft
  gating layer that produces one scalar gate per modality and per sample.
  This is a faithful 3-way generalisation of the *Gated Multimodal Unit*
  (Arevalo et al., 2017) with a softmax over modalities, plus a small MLP
  classifier on top of the gated sum.  Inexpensive, very interpretable.

* **GatedFusionHeavy** — per-modality projection + a 1-layer multi-head
  cross-modal Transformer encoder over the three modality tokens; the
  resulting context-aware tokens go through a sigmoid feature-wise gate
  before being concatenated and classified.  Captures pairwise
  interactions between modalities.

Both variants reuse the cached penultimate embeddings produced by the
intermediate-fusion notebook, so no backbone re-inference is needed.

Usage
-----
See notebooks/32_gated_fusion.ipynb.
"""

from __future__ import annotations

import os
import time
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
from torch.utils.data import DataLoader

from src.config import (
    AUDIO_DEV_DIR,
    AUDIO_TEST_DIR,
    DEVICE,
)
from src.data.face_dataset import NUM_CLASSES
from src.evaluation.intermediate_fusion import (
    FusionDataset,
    _CANON_I2L,
    _collate_dict,
    _compute_class_weights,
    _evaluate_mlp,
    evaluate_fusion_mlp,
    extract_audio_embeddings,
    extract_face_embeddings,
    extract_text_embeddings,
    load_embedding_cache,
    modality_ablation,
    save_embedding_cache,
    train_fusion_mlp,
)
from src.evaluation.plots import save_confusion_matrix
from src.training.utils import write_json


_N = NUM_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Light gated fusion
# ─────────────────────────────────────────────────────────────────────────────

class GatedFusionLight(nn.Module):
    """
    Light gated fusion (Gated Multimodal Unit, GMU-style).

    Architecture::

        for each modality m:
            h_m = tanh( Linear(LayerNorm(x_m), d_proj) )
        z   = softmax_over_modalities(  Linear( concat(h_1..h_M), M ) )
        g   = sum_m  z_m * h_m                          # (B, d_proj)
        out = Dropout(GELU(LayerNorm(Linear(g, d_hidden))))
        out = Linear(out, num_classes)

    * Gates are per-sample, per-modality scalars in [0, 1] summing to 1.
    * The model exposes the gates via ``forward(..., return_gates=True)``
      so the notebook can plot them per class.

    Parameters
    ----------
    in_dims   : per-modality input dim, e.g. {"text": 768, ...}
    d_proj    : projection dim per modality (default 128)
    d_hidden  : classifier bottleneck (default 128)
    dropout   : dropout in projection + classifier
    """

    def __init__(
        self,
        in_dims:     Dict[str, int],
        d_proj:      int   = 128,
        d_hidden:    int   = 128,
        num_classes: int   = _N,
        dropout:     float = 0.4,
    ):
        super().__init__()
        self.modalities = list(in_dims.keys())
        self.d_proj     = d_proj

        # Modality encoders (LN + Linear + tanh, GMU style).
        self.encoders = nn.ModuleDict({
            m: nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d_proj),
                nn.Tanh(),
            )
            for m, d in in_dims.items()
        })

        # Gating network: shared concat → M logits.
        self.gate_proj = nn.Sequential(
            nn.Linear(d_proj * len(in_dims), d_proj),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_proj, len(in_dims)),
        )

        # Classifier on the fused vector.
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_proj),
            nn.Linear(d_proj, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, num_classes),
        )

    def forward(
        self,
        feats: Dict[str, torch.Tensor],
        return_gates: bool = False,
    ):
        h = {m: self.encoders[m](feats[m]) for m in self.modalities}   # (B, d_proj)
        cat = torch.cat([h[m] for m in self.modalities], dim=-1)        # (B, M·d_proj)

        gate_logits = self.gate_proj(cat)                              # (B, M)
        gates       = F.softmax(gate_logits, dim=-1)                   # (B, M)

        # Weighted sum across modalities.
        stacked = torch.stack([h[m] for m in self.modalities], dim=1)  # (B, M, d_proj)
        fused   = (gates.unsqueeze(-1) * stacked).sum(dim=1)            # (B, d_proj)

        logits = self.classifier(fused)
        if return_gates:
            return logits, gates
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Heavy gated fusion (cross-modal attention + feature-wise gate)
# ─────────────────────────────────────────────────────────────────────────────

class GatedFusionHeavy(nn.Module):
    """
    Heavy gated fusion.

    Each modality is projected to a common dim and treated as a token in a
    short sequence (length = number of modalities).  A 1- or 2-layer
    Transformer encoder attends across the three modality tokens, allowing
    the model to learn pairwise interactions (e.g. "text says *fine* but
    face shows anger → conflict, downweight text").

    After cross-modal attention each modality token passes through a
    feature-wise sigmoid gate, the gated tokens are concatenated and a
    small MLP produces the logits.

    Architecture::

        proj_m = Linear(LayerNorm(x_m), d_model) + pos_embed_m
        tokens = stack(proj_text, proj_audio, proj_face)         # (B, M, d_model)
        tokens = TransformerEncoder(tokens, num_layers=L)         # (B, M, d_model)
        gates  = sigmoid( Linear(tokens, d_model) )               # per-feature gate
        gated  = tokens * gates
        out    = Dropout(GELU(LayerNorm(Linear(flatten(gated), d_hidden))))
        out    = Linear(out, num_classes)

    The per-feature sigmoid gate is exposed via
    ``forward(..., return_gates=True)`` so the notebook can summarise
    per-modality gate activations.

    Parameters
    ----------
    in_dims     : per-modality input dim
    d_model     : common token dim (default 256)
    nhead       : attention heads (default 4)
    num_layers  : transformer encoder layers (default 2)
    d_ff        : transformer FFN dim (default 512)
    d_hidden    : classifier bottleneck (default 256)
    dropout     : dropout in transformer + classifier
    """

    def __init__(
        self,
        in_dims:     Dict[str, int],
        d_model:     int   = 256,
        nhead:       int   = 4,
        num_layers:  int   = 2,
        d_ff:        int   = 512,
        d_hidden:    int   = 256,
        num_classes: int   = _N,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.modalities = list(in_dims.keys())
        self.d_model    = d_model
        M               = len(in_dims)

        # Per-modality projection to a common dim.
        self.projections = nn.ModuleDict({
            m: nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d_model),
            )
            for m, d in in_dims.items()
        })

        # Learned modality positional embedding (one vector per modality).
        self.modality_embed = nn.Parameter(torch.zeros(M, d_model))
        nn.init.normal_(self.modality_embed, std=0.02)

        # Cross-modal Transformer encoder.
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = d_ff,
            dropout         = dropout,
            batch_first     = True,
            activation      = "gelu",
            norm_first      = True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Feature-wise sigmoid gate per token.
        self.gate = nn.Linear(d_model, d_model)

        # Classification head over flattened, gated modality tokens.
        self.classifier = nn.Sequential(
            nn.LayerNorm(M * d_model),
            nn.Linear(M * d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, num_classes),
        )

    def forward(
        self,
        feats: Dict[str, torch.Tensor],
        return_gates: bool = False,
    ):
        proj = [self.projections[m](feats[m]) for m in self.modalities]  # M × (B, d_model)
        tokens = torch.stack(proj, dim=1)                                 # (B, M, d_model)
        tokens = tokens + self.modality_embed.unsqueeze(0)                # (B, M, d_model)
        ctx    = self.encoder(tokens)                                     # (B, M, d_model)

        gates  = torch.sigmoid(self.gate(ctx))                            # (B, M, d_model)
        gated  = ctx * gates                                              # (B, M, d_model)

        flat   = gated.reshape(gated.size(0), -1)                         # (B, M·d_model)
        logits = self.classifier(flat)

        if return_gates:
            # Reduce per-feature gates to per-modality scalar mean for plotting.
            modality_gate_mean = gates.mean(dim=-1)                       # (B, M)
            return logits, modality_gate_mean
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Gating-statistics utility
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_gates(
    model:    nn.Module,
    test_ds:  FusionDataset,
    device:   str = DEVICE,
) -> Dict:
    """
    Run the model on the test set and collect per-sample gating values.

    Returns
    -------
    {
        "modalities":     [...],
        "gates":          (N, M) float32,
        "labels":         (N,) int32,
        "preds":          (N,) int32,
        "mean_by_class":  (C, M) float32 — average gate per class
        "mean_overall":   (M,) float32   — average gate over all samples
    }
    """
    model.eval()
    loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=_collate_dict)
    all_gates, all_labels, all_preds = [], [], []

    for feats, y in loader:
        feats_d = {m: feats[m].to(device) for m in test_ds.modalities}
        out = model(feats_d, return_gates=True)
        if isinstance(out, tuple):
            logits, gates = out
        else:
            logits, gates = out, None
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(y.numpy())
        if gates is not None:
            all_gates.append(gates.cpu().numpy())

    if not all_gates:
        return {}

    gates  = np.concatenate(all_gates, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0).astype(np.int32)
    preds  = np.concatenate(all_preds, axis=0).astype(np.int32)

    C = _N
    M = gates.shape[1]
    mean_by_class = np.zeros((C, M), dtype=np.float32)
    for c in range(C):
        mask = labels == c
        if mask.any():
            mean_by_class[c] = gates[mask].mean(axis=0)

    return {
        "modalities":     list(test_ds.modalities),
        "gates":          gates,
        "labels":         labels,
        "preds":          preds,
        "mean_by_class":  mean_by_class,
        "mean_overall":   gates.mean(axis=0).astype(np.float32),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_gated_fusion(
    # Unimodal run directories
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
    # Fusion variant
    variant:        str = "light",     # "light" | "heavy"
    # Hyperparameters
    epochs:         int   = 80,
    batch_size:     int   = 128,
    lr:             float = 5e-4,
    weight_decay:   float = 1e-2,
    label_smoothing: float = 0.05,
    dropout:        float = 0.4,
    d_proj:         int   = 128,      # light only
    d_hidden:       int   = 256,
    d_model:        int   = 256,      # heavy only
    nhead:          int   = 4,        # heavy only
    num_layers:     int   = 2,        # heavy only
    d_ff:           int   = 512,      # heavy only
    patience:       int   = 12,
    # Cache / output
    cache_dir:  Optional[str] = None,
    output_dir: Optional[str] = None,
    # Infra
    device:          str = DEVICE,
    audio_train_dir: str = None,
    audio_dev_dir:   str = AUDIO_DEV_DIR,
    audio_test_dir:  str = AUDIO_TEST_DIR,
) -> Dict:
    """
    Full gated-fusion pipeline.

    Mirrors :func:`run_intermediate_fusion` but trains a gated fusion model
    (light or heavy) on top of the cached per-modality embeddings.
    """
    assert variant in ("light", "heavy"), \
        f"variant must be 'light' or 'heavy', got {variant!r}"

    from src.config import AUDIO_TRAIN_DIR
    if audio_train_dir is None:
        audio_train_dir = AUDIO_TRAIN_DIR
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if cache_dir is None:
        cache_dir = os.path.join(output_dir or ".", "cache")
    os.makedirs(cache_dir, exist_ok=True)

    id2label = _CANON_I2L

    # ── Step 1: Extract / load embeddings (reuse the intermediate-fusion cache) ──
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
    print(f"  Gated Fusion ({variant.upper()}) — embedding extraction")
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
    print(f"\n  Embedding dims : {in_dims}")
    print(f"  Train utterances: {len(train_ds)}   Dev: {len(dev_ds)}   Test: {len(test_ds)}")

    # ── Step 3: Build model ───────────────────────────────────────────────────
    if variant == "light":
        model = GatedFusionLight(
            in_dims     = in_dims,
            d_proj      = d_proj,
            d_hidden    = d_hidden,
            num_classes = _N,
            dropout     = dropout,
        )
    else:
        model = GatedFusionHeavy(
            in_dims     = in_dims,
            d_model     = d_model,
            nhead       = nhead,
            num_layers  = num_layers,
            d_ff        = d_ff,
            d_hidden    = d_hidden,
            num_classes = _N,
            dropout     = dropout,
        )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Gated Fusion ({variant}) params: {n_params:,}")

    # ── Step 4: Train ─────────────────────────────────────────────────────────
    t0 = time.time()
    model, history = train_fusion_mlp(
        model,
        train_ds,
        dev_ds,
        epochs            = epochs,
        batch_size        = batch_size,
        lr                = lr,
        weight_decay      = weight_decay,
        label_smoothing   = label_smoothing,
        device            = device,
        use_class_weights = True,
        patience          = patience,
    )
    elapsed = time.time() - t0
    print(f"  Training time: {elapsed/60:.1f} min")

    # ── Step 5: Evaluate ──────────────────────────────────────────────────────
    print("\n  Evaluating on test set …")
    test_metrics = evaluate_fusion_mlp(model, test_ds, id2label, device)
    print(f"  Accuracy    : {test_metrics['accuracy']:.4f}")
    print(f"  Macro F1    : {test_metrics['macro_f1']:.4f}")
    print(f"  Weighted F1 : {test_metrics['weighted_f1']:.4f}")

    # ── Explainability: gates + ablation ──────────────────────────────────────
    print("\n  Collecting gate statistics on test set …")
    gate_stats = collect_gates(model, test_ds, device=device)

    print("  Modality ablation on test set …")
    ablation = modality_ablation(model, test_ds, device=device)
    for k, v in ablation.items():
        print(f"    {k:14s}  macro_f1={v['macro_f1']:.4f}  acc={v['accuracy']:.4f}")

    # ── Save artefacts ────────────────────────────────────────────────────────
    if output_dir:
        label_names = [id2label[i] for i in range(_N)]

        torch.save(model.state_dict(), os.path.join(output_dir, "gated_fusion.pt"))

        write_json(os.path.join(output_dir, "test_metrics.json"), {
            k: v for k, v in test_metrics.items()
            if k not in ("preds", "labels", "report", "cm")
        })
        write_json(os.path.join(output_dir, "classification_report.json"),
                   test_metrics["report"])
        write_json(os.path.join(output_dir, "modality_ablation.json"), ablation)

        # Save gating stats (only numeric arrays / lists in JSON, full in NPZ).
        if gate_stats:
            np.savez_compressed(
                os.path.join(output_dir, "gates.npz"),
                gates         = gate_stats["gates"],
                labels        = gate_stats["labels"],
                preds         = gate_stats["preds"],
                mean_by_class = gate_stats["mean_by_class"],
                mean_overall  = gate_stats["mean_overall"],
                modalities    = np.array(gate_stats["modalities"]),
            )
            write_json(os.path.join(output_dir, "gates_summary.json"), {
                "modalities":    gate_stats["modalities"],
                "mean_overall":  gate_stats["mean_overall"].tolist(),
                "mean_by_class": {
                    id2label[c]: gate_stats["mean_by_class"][c].tolist()
                    for c in range(_N)
                },
            })

        save_confusion_matrix(
            np.array(test_metrics["cm"]),
            label_names,
            os.path.join(output_dir, "confusion_matrix_test.png"),
            title=f"Gated Fusion — {variant} (test)",
        )
        pd.DataFrame(history).to_csv(
            os.path.join(output_dir, "history.csv"), index=False
        )
        write_json(os.path.join(output_dir, "config.json"), {
            "fusion_type":       f"gated_fusion_{variant}",
            "variant":           variant,
            "text_run_dir":      text_run_dir,
            "audio_run_dir":     audio_run_dir,
            "face_run_dir":      face_run_dir,
            "emb_dims":          in_dims,
            "epochs":            epochs,
            "batch_size":        batch_size,
            "lr":                lr,
            "weight_decay":      weight_decay,
            "label_smoothing":   label_smoothing,
            "dropout":           dropout,
            "d_proj":            d_proj,
            "d_hidden":          d_hidden,
            "d_model":           d_model,
            "nhead":             nhead,
            "num_layers":        num_layers,
            "d_ff":              d_ff,
            "patience":          patience,
            "total_params":      int(n_params),
            "training_time_s":   round(elapsed, 1),
        })
        print(f"\n  Artefacts saved to  {output_dir}")

    return {
        "test_metrics":    test_metrics,
        "history":         history,
        "emb_dims":        in_dims,
        "model":           model,
        "gate_stats":      gate_stats,
        "ablation":        ablation,
        "training_time_s": elapsed,
        "num_params":      int(n_params),
        "variant":         variant,
    }
