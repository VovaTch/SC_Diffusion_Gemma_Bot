"""
Coordinator tying S0 ticks, S1 reactive loop, and S2 deliberative loop together.

Guarantees:
- ``on_tick(GameState)`` is always non-blocking (<5 ms) and immediately returns
  the latest merged ``CommandBundle`` so S0 runs concurrently at 24 FPS.
- S1 (every ~2.5 s) and S2 (every ~30 s or on escalation) run in background
  worker threads sharing a ``PriorityGate`` (S1 priority 0, S2 priority 1).
- Any vLLM outage or validation error degrades gracefully to the last valid
  directive and deterministic Protoss Zealot + Dragoon opening.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from starcraft_diffusion_gemma.brain.gate import GatedTransport, PriorityGate
from starcraft_diffusion_gemma.brain.s1 import S1Agent
from starcraft_diffusion_gemma.brain.s2 import S2ActionState, S2Agent, S2Result
from starcraft_diffusion_gemma.llm.client import (
    DEFAULT_MODEL,
    DiffusionGemmaClient,
    DiffusionGemmaError,
    Transport,
)
from starcraft_diffusion_gemma.state.models import (
    CommandBundle,
    EmergencyAction,
    GameState,
    S1Output,
    S2Directive,
    TacticalPosture,
    TilePoint,
)
from starcraft_diffusion_gemma.vision.minimap import minimap_legend, render_game_minimap

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoordinatorConfig:
    """Cadence and scheduling parameters for S1 and S2."""

    s1_interval_secs: float = 2.5
    s2_interval_secs: float = 30.0
    s2_min_escalation_gap_secs: float = 8.0
    enable_llm: bool = True


class BotCoordinator:
    """Owns shared game state, background S1/S2 threads, and S0 command synthesis."""

    def __init__(
        self,
        transport: Transport,
        config: CoordinatorConfig | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self._config = config or CoordinatorConfig()
        self._gate = PriorityGate()
        self._s1_agent = S1Agent(
            DiffusionGemmaClient(
                GatedTransport(transport, self._gate, priority=0), model=model
            )
        )
        self._s2_agent = S2Agent(
            DiffusionGemmaClient(
                GatedTransport(transport, self._gate, priority=1), model=model
            )
        )

        self._lock = threading.Lock()
        self._latest_state: GameState | None = None
        self._latest_minimap_png: bytes | None = None
        self._action_state = S2ActionState()
        self._directive = S2Directive()
        self._last_s1: S1Output | None = None
        self._recent_s1: deque[S1Output] = deque(maxlen=5)
        self._last_s2_result: S2Result | None = None

        self._last_s1_at: float | None = None
        self._last_s2_at: float | None = None
        self._last_s1_latency: float | None = None
        self._last_s1_error: str | None = None
        self._last_s2_error: str | None = None
        self._enemy_base_known = False
        self._revision = 0

        self._stop_event = threading.Event()
        self._s2_wake_event = threading.Event()
        self._pending_s2_reason: str = "periodic"
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        """Start background S1 and S2 worker threads (idempotent)."""
        if not self._config.enable_llm or self._threads:
            return
        self._stop_event.clear()
        t1 = threading.Thread(target=self._s1_loop, name="s1-worker", daemon=True)
        t2 = threading.Thread(target=self._s2_loop, name="s2-worker", daemon=True)
        t1.start()
        t2.start()
        self._threads = [t1, t2]

    def stop(self, timeout: float = 2.0) -> None:
        """Signal background S1/S2 worker threads to exit."""
        self._stop_event.set()
        self._s2_wake_event.set()
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads.clear()

    def on_tick(self, state: GameState) -> CommandBundle:
        """
        Ingest a fresh BWAPI ``GameState`` and immediately return ``CommandBundle``.

        Never waits on vLLM; safe to call on every host ``/tick`` request.

        Args:
            state (GameState): Latest game snapshot from the JBWAPI bridge.

        Returns:
            CommandBundle: Merged S0/S1/S2 commands to execute concurrently.
        """
        now = time.monotonic()
        with self._lock:
            self._latest_state = state
            # Auto-escalate to S2 the first time enemy base is scouted.
            if state.enemy_base is not None and not self._enemy_base_known:
                self._enemy_base_known = True
                self._request_s2_locked("event:enemy_base_discovered", now)
            return self._build_bundle_locked(state, now)

    def _request_s2_locked(self, reason: str, now: float) -> None:
        """Trigger an out-of-band S2 run if past the escalation debounce gap."""
        if (
            self._last_s2_at is not None
            and (now - self._last_s2_at) < self._config.s2_min_escalation_gap_secs
        ):
            return
        self._pending_s2_reason = reason
        self._s2_wake_event.set()

    def _default_target_locked(self, state: GameState) -> TilePoint | None:
        """Fallback army target when S1 has not yet emitted a target."""
        if self._directive.attack_target is not None:
            return self._directive.attack_target
        if state.choke_point is not None:
            return state.choke_point
        if state.natural_base is not None:
            return state.natural_base
        return state.our_base

    def _build_bundle_locked(self, state: GameState, now: float) -> CommandBundle:
        """Merge S0 build-order step, S1 reactive output, and S2 directive."""
        active_step = self._action_state.build_order.next_step(state)
        posture: TacticalPosture = (
            self._last_s1.posture
            if self._last_s1 is not None
            else self._directive.posture_hint
        )
        army_target: TilePoint | None = (
            self._last_s1.army_target
            if self._last_s1 is not None
            else self._default_target_locked(state)
        )
        emergency: EmergencyAction = (
            self._last_s1.emergency_action
            if self._last_s1 is not None
            else EmergencyAction.NONE
        )
        return CommandBundle(
            revision=self._revision,
            frame_issued=state.frame,
            posture=posture,
            army_target=army_target,
            emergency_action=emergency,
            target_probes=self._action_state.target_probes,
            target_gateways=self._action_state.target_gateways,
            zealot_ratio=self._action_state.zealot_ratio,
            dragoon_ratio=self._action_state.dragoon_ratio,
            want_singularity_charge=self._action_state.want_singularity_charge,
            queued_structures=list(self._action_state.queued_structures),
            expand_target=self._action_state.expand_target,
            active_build_step=active_step,
            s1_age_secs=(
                round(now - self._last_s1_at, 2)
                if self._last_s1_at is not None
                else None
            ),
            s2_age_secs=(
                round(now - self._last_s2_at, 2)
                if self._last_s2_at is not None
                else None
            ),
        )

    def step_s1_once(self) -> S1Output | None:
        """Run one S1 reactive step against the latest state (testable synchronously)."""
        with self._lock:
            state = self._latest_state
            if state is None:
                return None
            directive = self._directive
            build_status = self._action_state.build_order.status_text(state)
            prev_target = (
                self._last_s1.army_target
                if self._last_s1 is not None
                else self._default_target_locked(state)
            )

        minimap = render_game_minimap(state, army_target=prev_target)
        png_buf = io.BytesIO()
        minimap.save(png_buf, format="PNG")
        legend = minimap_legend(state)

        start = time.perf_counter()
        try:
            out = self._s1_agent.step(state, minimap, legend, directive, build_status)
        except (DiffusionGemmaError, Exception) as exc:  # noqa: BLE001
            logger.warning("S1 step failed: %s", exc)
            with self._lock:
                self._last_s1_error = str(exc)
            return None

        now = time.monotonic()
        with self._lock:
            self._latest_minimap_png = png_buf.getvalue()
            self._last_s1 = out
            self._recent_s1.append(out)
            self._last_s1_at = now
            self._last_s1_latency = time.perf_counter() - start
            self._last_s1_error = None
            self._revision += 1
            if out.escalate_to_s2:
                reason = f"s1_escalation:{out.escalation_reason or 'unspecified'}"
                self._request_s2_locked(reason, now)
        return out

    def step_s2_once(self, reason: str = "periodic") -> S2Result | None:
        """Run one two-step S2 strategic turn against the latest state."""
        with self._lock:
            state = self._latest_state
            if state is None:
                return None
            directive = self._directive
            recent_s1 = list(self._recent_s1)
            prev_target = (
                self._last_s1.army_target
                if self._last_s1 is not None
                else self._default_target_locked(state)
            )
            # Snapshot action_state for tool execution outside the lock; reconcile on commit.
            working_action_state = S2ActionState(
                build_order=self._action_state.build_order,
                queued_structures=list(self._action_state.queued_structures),
                target_gateways=self._action_state.target_gateways,
                target_probes=self._action_state.target_probes,
                zealot_ratio=self._action_state.zealot_ratio,
                dragoon_ratio=self._action_state.dragoon_ratio,
                want_singularity_charge=self._action_state.want_singularity_charge,
                expand_target=self._action_state.expand_target,
            )

        minimap = render_game_minimap(state, army_target=prev_target)
        legend = minimap_legend(state)

        try:
            result = self._s2_agent.plan(
                state=state,
                minimap=minimap,
                legend=legend,
                action_state=working_action_state,
                recent_s1=recent_s1,
                current_directive=directive,
                reason=reason,
            )
        except (DiffusionGemmaError, Exception) as exc:  # noqa: BLE001
            logger.warning("S2 plan failed: %s", exc)
            with self._lock:
                self._last_s2_error = str(exc)
            return None

        now = time.monotonic()
        with self._lock:
            self._directive = result.directive
            # Merge directive fields + tool mutations into live action_state.
            self._action_state.queued_structures = (
                working_action_state.queued_structures
            )
            self._action_state.target_gateways = result.directive.target_gateways
            self._action_state.target_probes = result.directive.target_probes
            self._action_state.zealot_ratio = result.directive.zealot_ratio
            self._action_state.dragoon_ratio = result.directive.dragoon_ratio
            self._action_state.want_singularity_charge = (
                result.directive.want_singularity_charge
            )
            if result.directive.want_expand and result.directive.expand_target:
                self._action_state.expand_target = result.directive.expand_target
            else:
                self._action_state.expand_target = working_action_state.expand_target
            self._last_s2_result = result
            self._last_s2_at = now
            self._last_s2_error = None
            self._revision += 1
        return result

    def _s1_loop(self) -> None:
        while not self._stop_event.is_set():
            self.step_s1_once()
            self._stop_event.wait(self._config.s1_interval_secs)

    def _s2_loop(self) -> None:
        while not self._stop_event.is_set():
            triggered = self._s2_wake_event.wait(self._config.s2_interval_secs)
            if self._stop_event.is_set():
                break
            self._s2_wake_event.clear()
            with self._lock:
                reason = self._pending_s2_reason if triggered else "periodic"
                self._pending_s2_reason = "periodic"
            self.step_s2_once(reason=reason)

    def latest_minimap_png(self) -> bytes | None:
        """Return the most recently rendered minimap PNG bytes, if any."""
        with self._lock:
            return self._latest_minimap_png

    def debug_snapshot(self) -> dict[str, Any]:
        """Return diagnostic state for the host ``/debug`` endpoint."""
        now = time.monotonic()
        with self._lock:
            bundle = (
                self._build_bundle_locked(self._latest_state, now).model_dump()
                if self._latest_state is not None
                else None
            )
            return {
                "revision": self._revision,
                "has_state": self._latest_state is not None,
                "frame": self._latest_state.frame if self._latest_state else None,
                "command_bundle": bundle,
                "last_s1": self._last_s1.model_dump() if self._last_s1 else None,
                "last_s1_latency_secs": self._last_s1_latency,
                "last_s1_error": self._last_s1_error,
                "directive": self._directive.model_dump(),
                "last_s2_latency_secs": (
                    self._last_s2_result.latency_secs if self._last_s2_result else None
                ),
                "last_s2_tools": (
                    [
                        {"name": c.name, "arguments": c.arguments}
                        for c in self._last_s2_result.tool_calls
                    ]
                    if self._last_s2_result
                    else []
                ),
                "last_s2_error": self._last_s2_error,
            }
