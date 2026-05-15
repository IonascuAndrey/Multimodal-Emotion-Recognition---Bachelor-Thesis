"""
text_explainability.py
----------------------
Two attribution methods for BERT / DistilBERT emotion classifiers.

LayerIntegratedGradients (Captum)
    Attributes the predicted class probability to individual input tokens by
    integrating gradients of the model output w.r.t. the token-embedding layer
    along a straight-line path from an all-[PAD] baseline to the actual input.
    Attribution vectors are summed over the embedding dimension so each token
    receives a single scalar score.

Token Occlusion
    Each token is replaced with [MASK] in turn.  The drop in predicted
    probability for the target class is the occlusion importance of that token:

        score_i = P(target | original) − P(target | token_i → [MASK])

    Positive values indicate tokens that actively push the model toward the
    predicted emotion; negative values indicate tokens that suppress it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from captum.attr import LayerIntegratedGradients
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import MAX_LEN


class TextExplainer:
    """
    Explainability wrapper for BERT / DistilBERT sequence-classification models.

    Parameters
    ----------
    model     : fine-tuned AutoModelForSequenceClassification in eval mode
    tokenizer : matching AutoTokenizer
    device    : 'cuda' or 'cpu'
    """

    def __init__(
        self,
        model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        device: str,
    ) -> None:
        self.model     = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device    = device

        # Resolve the embedding layer that LayerIG will hook into.
        if hasattr(model, "bert"):
            emb_layer = model.bert.embeddings
        elif hasattr(model, "distilbert"):
            emb_layer = model.distilbert.embeddings
        else:
            raise ValueError(
                f"Unsupported model type: {type(model).__name__}. "
                "Expected a BERT or DistilBERT-based model."
            )
        self._lig = LayerIntegratedGradients(self._forward_fn, emb_layer)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _forward_fn(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Wrapper that returns raw logits for Captum."""
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits

    def _encode(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenise *text* and move tensors to self.device."""
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN,
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    # ── Prediction ─────────────────────────────────────────────────────────────

    def predict(self, text: str) -> Tuple[int, np.ndarray]:
        """
        Forward pass on *text*.

        Returns
        -------
        predicted_class : int
        probabilities   : float32 array of shape (num_classes,)
        """
        enc = self._encode(text)
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return int(probs.argmax()), probs

    # ── Layer Integrated Gradients ─────────────────────────────────────────────

    def integrated_gradients(
        self,
        text: str,
        target_class: Optional[int] = None,
        n_steps: int = 50,
        internal_batch_size: int = 8,
    ) -> Tuple[List[str], np.ndarray, int]:
        """
        Compute per-token LayerIntegratedGradients attributions.

        The baseline is a sequence of [PAD] tokens of the same length as the
        input.  Token attributions are the sum of the embedding-dimension IG
        scores, giving one scalar per token.

        Parameters
        ----------
        text                : input utterance string
        target_class        : class to attribute towards; defaults to argmax
        n_steps             : integration steps (↑ accuracy, ↑ compute)
        internal_batch_size : Captum splits the n_steps into batches of this size

        Returns
        -------
        tokens       : list of subword tokens including [CLS] / [SEP]
        attr_scores  : float32 array of shape (seq_len,)
        target_class : class index actually used
        """
        enc            = self._encode(text)
        input_ids      = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        if target_class is None:
            with torch.no_grad():
                logits = self.model(**enc).logits
            target_class = int(logits.argmax(dim=-1))

        pad_id       = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        baseline_ids = torch.full_like(input_ids, pad_id)

        attributions = self._lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            target=target_class,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
        )
        # attributions: (1, seq_len, hidden_size) → sum over embedding dim
        attr_scores = attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()

        tokens = self.tokenizer.convert_ids_to_tokens(
            input_ids.squeeze(0).cpu().tolist()
        )
        return tokens, attr_scores.astype(np.float32), target_class

    # ── Token Occlusion ───────────────────────────────────────────────────────

    def token_occlusion(
        self,
        text: str,
        target_class: Optional[int] = None,
    ) -> Tuple[List[str], np.ndarray, int]:
        """
        Measure the importance of each token by replacing it with [MASK] and
        recording the drop in predicted probability for *target_class*.

            score_i = P(target | original) − P(target | token_i masked)

        Parameters
        ----------
        text         : input utterance string
        target_class : class to monitor; defaults to the model's argmax

        Returns
        -------
        tokens           : list of subword tokens
        occlusion_scores : float32 array of shape (seq_len,)
        target_class     : class index actually used
        """
        enc            = self._encode(text)
        input_ids      = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        if target_class is None:
            target_class = int(probs.argmax())
        original_prob = probs[target_class].item()

        mask_id = (
            self.tokenizer.mask_token_id
            or self.tokenizer.unk_token_id
            or 0
        )
        tokens   = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().tolist())
        seq_len  = input_ids.shape[1]
        scores   = np.zeros(seq_len, dtype=np.float32)

        for i in range(seq_len):
            masked_ids = input_ids.clone()
            masked_ids[0, i] = mask_id
            with torch.no_grad():
                logits_m = self.model(
                    input_ids=masked_ids, attention_mask=attention_mask
                ).logits
            prob_m    = torch.softmax(logits_m, dim=-1).squeeze(0)[target_class].item()
            scores[i] = original_prob - prob_m

        return tokens, scores, target_class
