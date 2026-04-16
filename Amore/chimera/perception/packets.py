"""
Interface contracts for the perception loops.

These dataclasses are the *only* thing that crosses the boundary between
the perception subsystem and the Amore core.  Raw frames and waveforms
never leave the perception process.

Design principle (from dump1.txt):
  "The contract is more important than the vision model itself."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class TrackedEntity:
    """A persistent object or person identified across multiple frames."""
    entity_id: str                        # stable across ticks (UUID or integer string)
    label: str                            # e.g. "person", "face", "object"
    confidence: float                     # 0–1
    embedding: Optional[torch.Tensor]     # [d_vis] visual identity embedding, if available
    last_seen_tick: int                   # perception loop tick counter
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerceptionPacket:
    """
    Compact summary emitted by VisionLoop each tick.

    Only this packet crosses into the Amore core — not raw frames.

    Fields
    ------
    tick : int
        Monotonic perception-loop tick counter.
    scene_latent : Tensor [d_vis]
        Compressed scene representation from the vision encoder.
    predicted_latent : Tensor [d_vis] | None
        Predicted scene latent from the temporal predictor (P_vis).
        None on the very first tick (no prior to condition on).
    visual_surprise : float
        Scalar prediction error norm: ||z_vis - z_hat_vis||.
        Feeds into the shared epsilon accounting alongside text error.
    surprise_map : Tensor [...] | None
        Spatially resolved surprise if the encoder supports it (e.g., patch-level ViT).
    entities : list[TrackedEntity]
        Tracked objects/persons identified in this frame.
    salient_events : list[str]
        Human-readable event flags, e.g. "face_appeared", "motion_spike",
        "person_left", "identity_change".
    frame_index : int
        Raw webcam frame counter (for debugging/alignment).
    """
    tick: int
    scene_latent: torch.Tensor
    predicted_latent: Optional[torch.Tensor]
    visual_surprise: float
    surprise_map: Optional[torch.Tensor]
    entities: List[TrackedEntity]
    salient_events: List[str]
    frame_index: int


@dataclass
class AudioPacket:
    """
    Compact summary emitted by AudioLoop each audio window.

    Fields
    ------
    tick : int
        Monotonic audio-loop tick counter.
    transcript_chunk : str
        ASR output for this window (may be partial/incremental).
    speaker_id : str | None
        Diarized speaker label ("speaker_0", "speaker_1", ...) or None if silent.
    prosody_latent : Tensor [d_pros]
        Compressed prosodic features (pitch contour, speaking rate, intensity,
        pause structure, hesitation patterns).
    ambient_latent : Tensor [d_amb]
        Compressed ambient sound representation (non-speech audio context).
    predicted_prosody : Tensor [d_pros] | None
        Predicted prosody latent from P_aud. None on first tick.
    auditory_surprise : float
        Scalar prediction error for the audio stream.
        Contributes to global epsilon accounting.
    salient_audio_events : list[str]
        Event flags, e.g. "speaker_switch", "interruption", "sudden_silence",
        "unknown_voice", "door_opened", "cadence_broken".
    window_start_ms : int
        Start of this audio window in ms from session start.
    window_duration_ms : int
        Duration of this audio window in ms.
    """
    tick: int
    transcript_chunk: str
    speaker_id: Optional[str]
    prosody_latent: torch.Tensor
    ambient_latent: torch.Tensor
    predicted_prosody: Optional[torch.Tensor]
    auditory_surprise: float
    salient_audio_events: List[str]
    window_start_ms: int
    window_duration_ms: int


@dataclass
class FusedPerceptionState:
    """
    Running perceptual world state maintained by the FusionLoop.
    Passed to the Amore core on every cognition tick.
    """
    latest_vision: Optional[PerceptionPacket]
    latest_audio: Optional[AudioPacket]
    # Combined prediction error: lambda_vis * vis_surprise + lambda_aud * aud_surprise
    combined_surprise: float
    # Accumulated salient events since last cognition tick
    pending_events: List[str]
    # All currently tracked entities
    active_entities: List[TrackedEntity]
