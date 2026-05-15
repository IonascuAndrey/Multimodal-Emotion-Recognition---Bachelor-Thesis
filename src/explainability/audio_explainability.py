"""
audio_explainability.py
-----------------------
Two attribution methods for wav2vec2 / DistilHuBERT emotion classifiers.

Integrated Gradients (Captum)
    Attributes the predicted class probability to individual samples in the
    preprocessed waveform, using a silent (all-zeros) baseline.  Raw sample-
    level scores are returned alongside a binned visualisation via
    utils.plot_waveform_attribution.

Temporal Occlusion
    The preprocessed waveform is divided into N equal-length segments; each is
    replaced with silence in turn.  The importance of segment i is:

        score_i = P(target | original) − P(target | segment_i silenced)

    Positive scores indicate temporally important windows.

Both methods work directly on the feature-extractor output (normalised
input_values tensor), which is the actual model input and avoids any ambiguity
from raw PCM encoding.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from captum.attr import IntegratedGradients
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from src.config import MAX_AUDIO_S, SAMPLE_RATE, TRIM_END_S, TRIM_START_S


class AudioExplainer:
    """
    Explainability wrapper for wav2vec2 / DistilHuBERT classifiers.

    Parameters
    ----------
    model             : fine-tuned AutoModelForAudioClassification
    feature_extractor : matching AutoFeatureExtractor
    device            : 'cuda' or 'cpu'
    sample_rate       : audio sample rate in Hz (must match training, default 16 000)
    trim_start_s      : seconds trimmed from the start of each clip
                        (must match the value used during training)
    trim_end_s        : seconds trimmed from the end of each clip
    max_audio_s       : maximum clip duration; longer clips are truncated
    """

    def __init__(
        self,
        model: AutoModelForAudioClassification,
        feature_extractor: AutoFeatureExtractor,
        device: str,
        sample_rate: int = SAMPLE_RATE,
        trim_start_s: float = TRIM_START_S,
        trim_end_s: float = TRIM_END_S,
        max_audio_s: float = MAX_AUDIO_S,
    ) -> None:
        self.model             = model.to(device).eval()
        self.feature_extractor = feature_extractor
        self.device            = device
        self.sample_rate       = sample_rate
        self.trim_start        = int(trim_start_s * sample_rate)
        self.trim_end          = int(trim_end_s   * sample_rate)
        self.max_samples       = int(max_audio_s  * sample_rate)

        self._ig = IntegratedGradients(self._forward_fn)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _forward_fn(self, input_values: torch.Tensor) -> torch.Tensor:
        """Wrapper that returns raw logits for Captum."""
        return self.model(input_values=input_values).logits

    def preprocess(self, waveform: np.ndarray) -> torch.Tensor:
        """
        Apply the same trim → truncate → feature-extract pipeline used during
        training and return a (1, n_samples) float tensor on self.device.

        Parameters
        ----------
        waveform : 1-D float32 numpy array at self.sample_rate
        """
        wav = waveform.copy().astype(np.float32)

        # Trim start / end
        start = self.trim_start
        end   = len(wav) - self.trim_end if self.trim_end > 0 else len(wav)
        wav   = wav[start:end] if end > start else wav[:1]

        # Truncate
        if len(wav) > self.max_samples:
            wav = wav[: self.max_samples]

        # Feature extraction (normalisation; no padding needed for a single clip)
        features = self.feature_extractor(
            wav,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=False,
        )
        return features["input_values"].to(self.device)  # (1, n_samples)

    # ── Prediction ─────────────────────────────────────────────────────────────

    def predict(self, waveform: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Run a forward pass on *waveform*.

        Returns
        -------
        predicted_class : int
        probabilities   : float32 array of shape (num_classes,)
        """
        input_values = self.preprocess(waveform)
        with torch.no_grad():
            logits = self.model(input_values=input_values).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return int(probs.argmax()), probs

    # ── Integrated Gradients ───────────────────────────────────────────────────

    def integrated_gradients(
        self,
        waveform: np.ndarray,
        target_class: Optional[int] = None,
        n_steps: int = 50,
        internal_batch_size: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Compute sample-level Integrated Gradients attributions.

        The baseline is a silent waveform (all zeros after feature extraction).

        Parameters
        ----------
        waveform            : raw 1-D float32 waveform at self.sample_rate
        target_class        : class to attribute towards; defaults to argmax
        n_steps             : integration steps
        internal_batch_size : Captum interpolation batch size

        Returns
        -------
        attr        : float32 array of shape (n_samples,) — IG per sample
        input_proc  : float32 array of shape (n_samples,) — preprocessed waveform
        target_class : class index actually used
        """
        input_values = self.preprocess(waveform).float()   # (1, n_samples)

        if target_class is None:
            with torch.no_grad():
                logits = self.model(input_values=input_values).logits
            target_class = int(logits.argmax(dim=-1))

        baseline     = torch.zeros_like(input_values)
        attributions = self._ig.attribute(
            inputs=input_values,
            baselines=baseline,
            target=target_class,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
        )
        attr       = attributions.squeeze(0).cpu().detach().numpy().astype(np.float32)
        input_proc = input_values.squeeze(0).cpu().detach().numpy().astype(np.float32)
        return attr, input_proc, target_class

    # ── Temporal Occlusion ─────────────────────────────────────────────────────

    def temporal_occlusion(
        self,
        waveform: np.ndarray,
        n_segments: int = 20,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Zero out each of *n_segments* equal-length temporal windows in turn
        and measure the drop in predicted class probability.

            score_i = P(target | original) − P(target | segment_i silenced)

        Parameters
        ----------
        waveform     : raw 1-D float32 waveform at self.sample_rate
        n_segments   : number of non-overlapping windows to test
        target_class : class to monitor; defaults to the model's argmax

        Returns
        -------
        scores       : float32 array of shape (n_segments,)
        input_proc   : float32 array of shape (n_samples,) — preprocessed waveform
        target_class : class index actually used
        """
        input_values = self.preprocess(waveform)   # (1, n_samples)
        n_samples    = input_values.shape[1]

        with torch.no_grad():
            probs_orig = torch.softmax(
                self.model(input_values=input_values).logits, dim=-1
            ).squeeze(0)

        if target_class is None:
            target_class = int(probs_orig.argmax())
        original_prob = probs_orig[target_class].item()

        seg_len = max(n_samples // n_segments, 1)
        scores  = np.zeros(n_segments, dtype=np.float32)

        for i in range(n_segments):
            start = i * seg_len
            end   = start + seg_len if i < n_segments - 1 else n_samples
            masked = input_values.clone()
            masked[0, start:end] = 0.0
            with torch.no_grad():
                prob_m = torch.softmax(
                    self.model(input_values=masked).logits, dim=-1
                ).squeeze(0)[target_class].item()
            scores[i] = original_prob - prob_m

        input_proc = input_values.squeeze(0).cpu().numpy().astype(np.float32)
        return scores, input_proc, target_class
