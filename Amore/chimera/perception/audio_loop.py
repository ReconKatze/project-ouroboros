"""
AudioLoop — parallel microphone perception loop.

Runs continuously in a background thread.  Each audio window:
  1. Capture audio chunk from microphone
  2. ASR transcription (Whisper-style)
  3. Speaker diarization (who is speaking)
  4. Prosody feature extraction (pitch, rate, intensity, pauses, hesitation)
  5. Ambient sound embedding (non-speech context)
  6. Predict next prosody latent via P_aud
  7. Compute auditory prediction error
  8. Emit AudioPacket to thread-safe queue

Architecture (from dump1.txt):
  z_t_aud = E_aud(x_t_aud)
  z_hat_t_aud = P_aud(z_{t-1}_aud, h_{t-1})
  e_t_aud = z_t_aud - z_hat_t_aud
  → auditory surprise contributes to global epsilon accounting

Signal doctrine: don't stream full transcripts into context.
Send a compact AudioPacket that updates the persistent core state.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Optional

import torch
import torch.nn as nn

from .packets import AudioPacket

# Optional heavy dependencies
try:
    import numpy as np
    _NP_AVAILABLE = True
except ImportError:
    _NP_AVAILABLE = False

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False


def _require_audio_deps() -> None:
    if not _NP_AVAILABLE:
        raise ImportError("numpy is required for AudioLoop. pip install numpy")
    if not _SD_AVAILABLE:
        raise ImportError("sounddevice is required for AudioLoop. pip install sounddevice")


class ProsodyEncoder(nn.Module):
    """
    Encodes a window of raw prosodic features into a compact latent.

    Input: hand-crafted features vector (pitch mean/std, rate, intensity, pauses, etc.)
    Output: [d_pros] embedding.

    Phase 1 placeholder — replace with a trained encoder (e.g., wav2vec2 top layer
    or a small learned MLP over OpenSMILE features).
    """
    def __init__(self, n_features: int = 32, d_pros: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.GELU(),
            nn.Linear(128, d_pros),
        )
        self.n_features = n_features

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: [n_features] → [d_pros]"""
        return self.mlp(features)


class AmbientEncoder(nn.Module):
    """
    Encodes ambient (non-speech) audio into a compact latent.
    Phase 1: shallow MLP over mel-filterbank features.
    Phase 2: replace with a small audio classification encoder.
    """
    def __init__(self, n_mel: int = 40, d_amb: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_mel, 64),
            nn.GELU(),
            nn.Linear(64, d_amb),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: [n_mel] → [d_amb]"""
        return self.mlp(mel)


class AudioTemporalPredictor(nn.Module):
    """
    Predicts the next prosody latent from the previous one.
    Mirrors TemporalPredictor in vision_loop.py.
    """
    def __init__(self, d_pros: int, hidden_dim: int = 128):
        super().__init__()
        self.gru = nn.GRUCell(d_pros, hidden_dim)
        self.proj = nn.Linear(hidden_dim, d_pros)
        self.hidden_dim = hidden_dim
        self._h: Optional[torch.Tensor] = None

    def forward(self, z_pros: torch.Tensor) -> torch.Tensor:
        z = z_pros.unsqueeze(0)
        if self._h is None:
            self._h = torch.zeros(1, self.hidden_dim, device=z.device, dtype=z.dtype)
        self._h = self.gru(z, self._h)
        return self.proj(self._h).squeeze(0)

    def reset(self) -> None:
        self._h = None


class SpeakerTracker:
    """
    Minimal speaker diarization: assigns persistent IDs based on voice embedding
    similarity across windows.  Phase 1 placeholder.
    """
    def __init__(self, similarity_threshold: float = 0.80):
        self.threshold = similarity_threshold
        self._speakers: dict = {}  # label → embedding

    def assign(self, embedding: Optional[torch.Tensor]) -> Optional[str]:
        if embedding is None:
            return None
        best_label: Optional[str] = None
        best_sim = 0.0
        for label, emb in self._speakers.items():
            sim = float(torch.nn.functional.cosine_similarity(
                embedding.unsqueeze(0), emb.unsqueeze(0)
            ))
            if sim > best_sim and sim >= self.threshold:
                best_sim = sim
                best_label = label

        if best_label is None:
            best_label = f"speaker_{len(self._speakers)}"

        self._speakers[best_label] = embedding
        return best_label


class AudioLoop:
    """
    Parallel microphone perception loop.

    Parameters
    ----------
    prosody_encoder : ProsodyEncoder
        Encodes prosodic feature vectors to latents.
    ambient_encoder : AmbientEncoder
        Encodes ambient audio to latents.
    d_pros : int
        Prosody latent dimensionality.
    asr_fn : callable | None
        Function (audio_np: np.ndarray, sample_rate: int) → str.
        If None, transcription is skipped (useful during Phase 1 passive training).
    device : str
        Torch device.
    sample_rate : int
        Microphone sample rate in Hz.
    window_ms : int
        Audio window duration in milliseconds.
    max_queue_size : int
        Maximum AudioPackets buffered before oldest are dropped.
    lambda_aud : float
        Surprise scaling coefficient for global epsilon accounting.
    predictor_weights : str | None
        Path to saved AudioTemporalPredictor weights.
    """

    def __init__(
        self,
        prosody_encoder: ProsodyEncoder,
        ambient_encoder: AmbientEncoder,
        d_pros: int,
        asr_fn: Optional[Callable] = None,
        device: str = "cpu",
        sample_rate: int = 16000,
        window_ms: int = 500,
        max_queue_size: int = 64,
        lambda_aud: float = 1.0,
        predictor_weights: Optional[str] = None,
    ):
        _require_audio_deps()
        self.prosody_encoder = prosody_encoder.to(device)
        self.prosody_encoder.eval()
        self.ambient_encoder = ambient_encoder.to(device)
        self.ambient_encoder.eval()
        self.d_pros = d_pros
        self.asr_fn = asr_fn
        self.device = device
        self.sample_rate = sample_rate
        self.window_ms = window_ms
        self.window_samples = int(sample_rate * window_ms / 1000)
        self.lambda_aud = lambda_aud

        self.predictor = AudioTemporalPredictor(d_pros).to(device)
        if predictor_weights:
            self.predictor.load_state_dict(torch.load(predictor_weights, map_location=device))
        self.predictor.eval()

        self.speaker_tracker = SpeakerTracker()
        self.queue: queue.Queue[AudioPacket] = queue.Queue(maxsize=max_queue_size)

        self._tick = 0
        self._session_start_ms = int(time.time() * 1000)
        self._prev_prosody: Optional[torch.Tensor] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._on_packet: Optional[Callable[[AudioPacket], None]] = None

    def set_packet_hook(self, fn: Callable[[AudioPacket], None]) -> None:
        self._on_packet = fn

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="AudioLoop")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def get_packet(self, timeout: float = 0.1) -> Optional[AudioPacket]:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        import sounddevice as _sd
        import numpy as np

        while not self._stop_event.is_set():
            window_start_ms = int(time.time() * 1000) - self._session_start_ms

            try:
                audio_np = _sd.rec(
                    self.window_samples,
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                )
                _sd.wait()
            except Exception:
                time.sleep(self.window_ms / 1000)
                continue

            audio_np = audio_np.flatten()
            packet = self._process_window(audio_np, window_start_ms)
            self._tick += 1

            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            self.queue.put(packet)

            if self._on_packet:
                self._on_packet(packet)

    @torch.no_grad()
    def _process_window(self, audio_np, window_start_ms: int) -> AudioPacket:
        import numpy as np

        # --- ASR ---
        transcript = ""
        if self.asr_fn is not None:
            try:
                transcript = self.asr_fn(audio_np, self.sample_rate) or ""
            except Exception:
                transcript = ""

        # --- Prosody features (Phase 1: simple heuristics) ---
        rms = float(np.sqrt(np.mean(audio_np ** 2)))
        # Zero-crossing rate as rough pitch proxy
        zcr = float(np.mean(np.abs(np.diff(np.sign(audio_np)))) / 2)
        # Pauses: fraction of near-silent frames
        silence_frac = float(np.mean(np.abs(audio_np) < 0.01))

        n_feat = self.prosody_encoder.n_features
        raw_features = torch.zeros(n_feat, dtype=torch.float32)
        raw_features[0] = rms
        raw_features[1] = zcr
        raw_features[2] = silence_frac
        raw_features = raw_features.to(self.device)

        prosody_latent = self.prosody_encoder(raw_features)  # [d_pros]

        # --- Ambient sound embedding (Phase 1: mel-based) ---
        mel_n = self.ambient_encoder.mlp[0].in_features
        mel_features = torch.zeros(mel_n, dtype=torch.float32).to(self.device)
        # Placeholder: use RMS across frequency bands as a rough mel proxy
        chunk_size = max(1, len(audio_np) // mel_n)
        for i in range(mel_n):
            seg = audio_np[i * chunk_size: (i + 1) * chunk_size]
            mel_features[i] = float(np.sqrt(np.mean(seg ** 2))) if len(seg) > 0 else 0.0
        ambient_latent = self.ambient_encoder(mel_features)  # [d_amb]

        # --- Auditory prediction error ---
        predicted_prosody: Optional[torch.Tensor] = None
        surprise = 0.0
        if self._prev_prosody is not None:
            predicted_prosody = self.predictor(self._prev_prosody)
            error = prosody_latent - predicted_prosody
            surprise = float(error.norm().item()) * self.lambda_aud
        self._prev_prosody = prosody_latent.clone()

        # --- Speaker tracking (placeholder: no voice embedding yet) ---
        speaker_id = self.speaker_tracker.assign(None) if rms > 0.005 else None

        # --- Salient event detection ---
        salient: list[str] = []
        if speaker_id is not None and self._prev_prosody is not None and surprise > 1.5:
            salient.append("prosody_spike")
        if silence_frac > 0.9 and rms < 0.002:
            salient.append("sudden_silence")
        if transcript and len(transcript) > 0 and rms > 0.02:
            salient.append("speech_detected")

        return AudioPacket(
            tick=self._tick,
            transcript_chunk=transcript,
            speaker_id=speaker_id,
            prosody_latent=prosody_latent.cpu(),
            ambient_latent=ambient_latent.cpu(),
            predicted_prosody=predicted_prosody.cpu() if predicted_prosody is not None else None,
            auditory_surprise=surprise,
            salient_audio_events=salient,
            window_start_ms=window_start_ms,
            window_duration_ms=self.window_ms,
        )
