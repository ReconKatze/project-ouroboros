"""
chimera.perception — parallel perceptual subsystem for Amore.

Three loops run alongside the core cognition loop:
  1. VisionLoop  (~5–15 Hz)   — webcam → encoder → predictor → error → PerceptionPacket
  2. AudioLoop   (continuous) — mic → ASR/prosody → predictor → error → AudioPacket
  3. FusionLoop  (event-driven) — high-surprise events → episodic/controller decisions

All loops feed compact summary packets into the core; raw frames/waveforms never reach
the language head.  Signal flow mirrors the text prediction-error doctrine: predict the
next latent, measure honest surprise, gate the write.

See dump1.txt (perception stack design) for full rationale.
"""

from .packets import AudioPacket, PerceptionPacket
from .vision_loop import VisionLoop
from .audio_loop import AudioLoop
from .fusion import FusionLoop

__all__ = [
    "PerceptionPacket",
    "AudioPacket",
    "VisionLoop",
    "AudioLoop",
    "FusionLoop",
]
