"""
explainability
--------------
Unimodal attribution methods for the MELD emotion-recognition models.

  Text  : LayerIntegratedGradients (Captum) + Token Occlusion
  Audio : IntegratedGradients on normalised waveform (Captum) + Temporal Occlusion
  Face  : Grad-CAM (pytorch-grad-cam) + IntegratedGradients (Captum)

Usage example::

    from src.explainability import TextExplainer, AudioExplainer, FaceExplainer
"""

from src.explainability.text_explainability  import TextExplainer
from src.explainability.audio_explainability import AudioExplainer
from src.explainability.face_explainability  import FaceExplainer

__all__ = ["TextExplainer", "AudioExplainer", "FaceExplainer"]
