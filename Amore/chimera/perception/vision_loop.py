"""
VisionLoop — parallel webcam perception loop.

Runs at ~5–15 Hz in a background thread.  Each tick:
  1. Grab frame from webcam
  2. Encode to visual latent z_vis via frozen vision encoder (SigLIP / DINOv2 / CLIP)
  3. Predict next latent z_hat_vis via small temporal predictor P_vis
  4. Compute prediction error e_vis = z_vis - z_hat_vis
  5. Detect salient entities and events
  6. Emit PerceptionPacket to a thread-safe queue

Consumers (FusionLoop / Amore core) read from the queue at their own rate.

Architecture (from dump1.txt):
  z_t_vis = E_vis(frame_t)
  z_hat_t_vis = P_vis(z_{t-1}_vis, h_{t-1})
  e_t_vis = z_t_vis - z_hat_t_vis
  e_tilde_t_vis = A_att_vis * e_t_vis - friction_vis
  → only the gated visual error writes into the main recurrent state

The VisionLoop does NOT write into the main state — it produces packets.
State integration is the FusionLoop's / core's responsibility.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from .packets import PerceptionPacket, TrackedEntity

# Optional heavy dependencies — imported lazily so the rest of the codebase
# works even when OpenCV / transformers aren't installed.
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


def _require_cv2() -> None:
    if not _CV2_AVAILABLE:
        raise ImportError(
            "cv2 is required for VisionLoop. Install with: pip install opencv-python"
        )


class TemporalPredictor(nn.Module):
    """
    Small GRU that predicts the next visual latent from the previous one.
    Trained separately (Phase 1 perception training).
    """
    def __init__(self, d_vis: int, hidden_dim: int = 256):
        super().__init__()
        self.gru = nn.GRUCell(d_vis, hidden_dim)
        self.proj = nn.Linear(hidden_dim, d_vis)
        self.hidden_dim = hidden_dim
        self._h: Optional[torch.Tensor] = None

    def forward(self, z_vis: torch.Tensor) -> torch.Tensor:
        """z_vis: [d_vis] → predicted next latent [d_vis]."""
        z = z_vis.unsqueeze(0)  # [1, d_vis]
        if self._h is None:
            self._h = torch.zeros(1, self.hidden_dim, device=z.device, dtype=z.dtype)
        self._h = self.gru(z, self._h)
        return self.proj(self._h).squeeze(0)

    def reset(self) -> None:
        self._h = None


class SimpleEntityTracker:
    """
    Lightweight entity tracker based on embedding similarity.
    Assigns stable IDs to detected objects/persons across frames.
    Phase 1 placeholder — can be replaced with SORT/ByteTrack/DeepSORT.
    """
    def __init__(self, similarity_threshold: float = 0.85, max_age: int = 30):
        self.threshold = similarity_threshold
        self.max_age = max_age
        self._tracks: Dict[str, Dict] = {}  # entity_id → {embedding, last_tick, label}

    def update(
        self,
        detections: List[Dict],  # each: {"label": str, "embedding": Tensor, "confidence": float}
        tick: int,
    ) -> List[TrackedEntity]:
        matched: List[TrackedEntity] = []
        used_ids: set = set()

        for det in detections:
            emb = det["embedding"]
            best_id: Optional[str] = None
            best_sim: float = 0.0

            for eid, track in self._tracks.items():
                if eid in used_ids:
                    continue
                sim = float(torch.nn.functional.cosine_similarity(
                    emb.unsqueeze(0), track["embedding"].unsqueeze(0)
                ))
                if sim > best_sim and sim >= self.threshold:
                    best_sim = sim
                    best_id = eid

            if best_id is None:
                best_id = str(uuid.uuid4())[:8]

            self._tracks[best_id] = {
                "embedding": emb,
                "last_tick": tick,
                "label": det["label"],
            }
            used_ids.add(best_id)
            matched.append(TrackedEntity(
                entity_id=best_id,
                label=det["label"],
                confidence=det["confidence"],
                embedding=emb,
                last_seen_tick=tick,
            ))

        # Prune stale tracks
        stale = [eid for eid, t in self._tracks.items() if tick - t["last_tick"] > self.max_age]
        for eid in stale:
            del self._tracks[eid]

        return matched


class VisionLoop:
    """
    Parallel webcam perception loop.

    Parameters
    ----------
    encoder : nn.Module
        Frozen vision encoder.  Must accept a [1, C, H, W] float32 tensor and return
        a [d_vis] feature tensor.  Use SigLIP / DINOv2 / CLIP; frozen at first.
    d_vis : int
        Visual latent dimensionality (must match encoder output).
    device : str
        Torch device.
    target_hz : float
        Target tick rate.  Actual rate depends on camera and encoder speed.
    camera_index : int
        OpenCV camera index (0 = default webcam).
    max_queue_size : int
        Maximum PerceptionPackets buffered before oldest are dropped.
    lambda_vis : float
        Surprise scaling coefficient for global epsilon accounting.
    predictor_weights : str | None
        Path to saved TemporalPredictor weights.  If None, predictor starts untrained.
    """

    def __init__(
        self,
        encoder: nn.Module,
        d_vis: int,
        device: str = "cpu",
        target_hz: float = 10.0,
        camera_index: int = 0,
        max_queue_size: int = 32,
        lambda_vis: float = 1.0,
        predictor_weights: Optional[str] = None,
    ):
        _require_cv2()
        self.encoder = encoder.to(device)
        self.encoder.eval()
        self.d_vis = d_vis
        self.device = device
        self.target_hz = target_hz
        self.camera_index = camera_index
        self.lambda_vis = lambda_vis

        self.predictor = TemporalPredictor(d_vis).to(device)
        if predictor_weights:
            self.predictor.load_state_dict(torch.load(predictor_weights, map_location=device))
        self.predictor.eval()

        self.tracker = SimpleEntityTracker()
        self.queue: queue.Queue[PerceptionPacket] = queue.Queue(maxsize=max_queue_size)

        self._tick = 0
        self._frame_index = 0
        self._prev_latent: Optional[torch.Tensor] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # External hooks (optional): called on every packet for logging / routing
        self._on_packet: Optional[Callable[[PerceptionPacket], None]] = None

    def set_packet_hook(self, fn: Callable[[PerceptionPacket], None]) -> None:
        """Register a callback invoked (in the perception thread) on every packet."""
        self._on_packet = fn

    def start(self) -> None:
        """Start the perception loop in a background thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="VisionLoop")
        self._thread.start()

    def stop(self) -> None:
        """Signal the loop to stop and join the thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def get_packet(self, timeout: float = 0.1) -> Optional[PerceptionPacket]:
        """Retrieve the next PerceptionPacket, or None if the queue is empty."""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}")

        period = 1.0 / self.target_hz
        try:
            while not self._stop_event.is_set():
                t0 = time.monotonic()
                ret, frame = cap.read()
                if not ret:
                    time.sleep(period)
                    continue

                self._frame_index += 1
                packet = self._process_frame(frame)
                self._tick += 1

                # Drop oldest if queue is full
                if self.queue.full():
                    try:
                        self.queue.get_nowait()
                    except queue.Empty:
                        pass
                self.queue.put(packet)

                if self._on_packet:
                    self._on_packet(packet)

                elapsed = time.monotonic() - t0
                sleep_time = period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            cap.release()

    @torch.no_grad()
    def _process_frame(self, frame) -> PerceptionPacket:
        import cv2 as _cv2

        # Resize and normalize frame
        rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
        resized = _cv2.resize(rgb, (224, 224))
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)

        # Encode to visual latent
        z_vis = self.encoder(tensor)
        if z_vis.dim() > 1:
            z_vis = z_vis.squeeze(0)  # ensure [d_vis]

        # Predict next latent and compute surprise
        z_hat: Optional[torch.Tensor] = None
        surprise = 0.0
        if self._prev_latent is not None:
            z_hat = self.predictor(self._prev_latent)
            error = z_vis - z_hat
            surprise = float(error.norm().item()) * self.lambda_vis

        self._prev_latent = z_vis.clone()

        # Entity detection — placeholder: no detections until a detector is wired
        entities = self.tracker.update([], tick=self._tick)

        # Salient event detection
        salient_events: list[str] = []
        if surprise > 2.0:  # threshold tunable
            salient_events.append("motion_spike")

        return PerceptionPacket(
            tick=self._tick,
            scene_latent=z_vis.cpu(),
            predicted_latent=z_hat.cpu() if z_hat is not None else None,
            visual_surprise=surprise,
            surprise_map=None,
            entities=entities,
            salient_events=salient_events,
            frame_index=self._frame_index,
        )
