"""
FusionLoop — event-driven episodic/controller loop.

Runs slower than the perception loops, triggered by high-surprise events.
Merges the latest PerceptionPacket and AudioPacket into a FusedPerceptionState
that the Amore core reads on each cognition tick.

Decision logic (from dump1.txt):
  Triggered by: high visual surprise, person detected, face/gaze/gesture change,
  user enters/leaves, repeated or identity-relevant events.

  Decides:
    - write memory?
    - ask a question?
    - attend harder?
    - mark a relational update?
    - ignore as transient noise?

This loop is the boundary between perception and cognition.  It does NOT modify
model state — it produces a FusedPerceptionState that the core consumes.

Combined surprise formula (from dump1.txt):
  epsilon_total = lambda_txt * epsilon_txt + lambda_vis * epsilon_vis + lambda_aud * epsilon_aud
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .packets import AudioPacket, FusedPerceptionState, PerceptionPacket, TrackedEntity


# Thresholds tunable via FusionConfig
SURPRISE_WRITE_THRESHOLD = 2.0    # combined surprise → flag "should_write_memory"
SURPRISE_ATTEND_THRESHOLD = 1.0   # combined surprise → flag "attend_harder"
PERSON_ABSENCE_TICKS = 60          # ticks without a person → "user_left" event


@dataclass
class FusionConfig:
    lambda_vis: float = 1.0
    lambda_aud: float = 1.0
    surprise_write_threshold: float = SURPRISE_WRITE_THRESHOLD
    surprise_attend_threshold: float = SURPRISE_ATTEND_THRESHOLD
    person_absence_ticks: int = PERSON_ABSENCE_TICKS


@dataclass
class FusionDecision:
    """
    Signals produced by the FusionLoop for the core cognition system.
    Attached to each FusedPerceptionState.
    """
    should_write_memory: bool = False
    should_attend_harder: bool = False
    ask_clarifying_question: bool = False
    relational_update: bool = False
    ignore: bool = False
    reason: str = ""


@dataclass
class FusedPerceptionStateWithDecision:
    state: FusedPerceptionState
    decision: FusionDecision


class FusionLoop:
    """
    Merges VisionLoop and AudioLoop outputs into FusedPerceptionState.

    Parameters
    ----------
    vision_queue : queue.Queue[PerceptionPacket]
        The vision loop's output queue.  Pass `loop.queue`.
    audio_queue : queue.Queue[AudioPacket]
        The audio loop's output queue.  Pass `loop.queue`.
    config : FusionConfig
    on_state : callable | None
        Called (in fusion thread) on every new fused state.
        Signature: (FusedPerceptionStateWithDecision) → None.
    poll_interval : float
        Seconds between fusion ticks when no new packets arrive.
    """

    def __init__(
        self,
        vision_queue: Optional[queue.Queue] = None,
        audio_queue: Optional[queue.Queue] = None,
        config: Optional[FusionConfig] = None,
        on_state: Optional[Callable[[FusedPerceptionStateWithDecision], None]] = None,
        poll_interval: float = 0.1,
    ):
        self.vision_queue = vision_queue
        self.audio_queue = audio_queue
        self.config = config or FusionConfig()
        self.on_state = on_state
        self.poll_interval = poll_interval

        self._latest_vision: Optional[PerceptionPacket] = None
        self._latest_audio: Optional[AudioPacket] = None
        self._pending_events: List[str] = []
        self._last_person_tick: int = 0
        self._person_was_present: bool = False

        self._output_queue: queue.Queue[FusedPerceptionStateWithDecision] = queue.Queue(maxsize=16)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="FusionLoop")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def get_state(self, timeout: float = 0.05) -> Optional[FusedPerceptionStateWithDecision]:
        try:
            return self._output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        while not self._stop_event.is_set():
            # Drain new packets
            updated_vision = False
            updated_audio = False

            if self.vision_queue:
                while True:
                    try:
                        pkt = self.vision_queue.get_nowait()
                        self._ingest_vision(pkt)
                        updated_vision = True
                    except queue.Empty:
                        break

            if self.audio_queue:
                while True:
                    try:
                        pkt = self.audio_queue.get_nowait()
                        self._ingest_audio(pkt)
                        updated_audio = True
                    except queue.Empty:
                        break

            if updated_vision or updated_audio:
                fused = self._build_fused_state()
                if self._output_queue.full():
                    try:
                        self._output_queue.get_nowait()
                    except queue.Empty:
                        pass
                self._output_queue.put(fused)
                if self.on_state:
                    self.on_state(fused)

            time.sleep(self.poll_interval)

    def _ingest_vision(self, pkt: PerceptionPacket) -> None:
        self._latest_vision = pkt
        self._pending_events.extend(pkt.salient_events)

        # Person tracking
        person_present = any(e.label in ("person", "face") for e in pkt.entities)
        if person_present:
            self._last_person_tick = pkt.tick
            if not self._person_was_present:
                self._pending_events.append("person_appeared")
            self._person_was_present = True
        elif self._person_was_present:
            absence = pkt.tick - self._last_person_tick
            if absence >= self.config.person_absence_ticks:
                self._pending_events.append("user_left")
                self._person_was_present = False

    def _ingest_audio(self, pkt: AudioPacket) -> None:
        self._latest_audio = pkt
        self._pending_events.extend(pkt.salient_audio_events)

    def _build_fused_state(self) -> FusedPerceptionStateWithDecision:
        vis_surprise = self._latest_vision.visual_surprise if self._latest_vision else 0.0
        aud_surprise = self._latest_audio.auditory_surprise if self._latest_audio else 0.0
        combined = (
            self.config.lambda_vis * vis_surprise
            + self.config.lambda_aud * aud_surprise
        )

        entities: List[TrackedEntity] = (
            self._latest_vision.entities if self._latest_vision else []
        )

        pending = list(self._pending_events)
        self._pending_events.clear()

        state = FusedPerceptionState(
            latest_vision=self._latest_vision,
            latest_audio=self._latest_audio,
            combined_surprise=combined,
            pending_events=pending,
            active_entities=entities,
        )

        decision = self._decide(combined, pending, entities)

        return FusedPerceptionStateWithDecision(state=state, decision=decision)

    def _decide(
        self,
        combined_surprise: float,
        events: List[str],
        entities: List[TrackedEntity],
    ) -> FusionDecision:
        write = combined_surprise >= self.config.surprise_write_threshold
        attend = combined_surprise >= self.config.surprise_attend_threshold
        relational = any(
            e in events for e in ("person_appeared", "user_left", "identity_change")
        )
        ask = "cadence_broken" in events or "prosody_spike" in events

        if not (write or attend or relational or ask):
            return FusionDecision(ignore=True, reason="below thresholds")

        reason_parts = []
        if write:
            reason_parts.append(f"surprise={combined_surprise:.2f}")
        if relational:
            reason_parts.append("relational_event")
        if ask:
            reason_parts.append("cadence_or_prosody")

        return FusionDecision(
            should_write_memory=write,
            should_attend_harder=attend,
            ask_clarifying_question=ask,
            relational_update=relational,
            ignore=False,
            reason=", ".join(reason_parts),
        )
