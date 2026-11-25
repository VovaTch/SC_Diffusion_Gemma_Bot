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
import math
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from starcraft_diffusion_gemma.brain.gate import GatedTransport, PriorityGate
from starcraft_diffusion_gemma.brain.s1 import S1Agent
from starcraft_diffusion_gemma.brain.s2 import (
    S2ActionState,
    S2Agent,
    S2Result,
    S2RunCapture,
)
from starcraft_diffusion_gemma.brain.traces import (
    S1Trace,
    S2Trace,
    TracesSnapshot,
)
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
    """Cadence, scheduling, and trace logging parameters for S1 and S2."""

    s1_interval_secs: float = 2.5
    s2_interval_secs: float = 30.0
    s2_min_escalation_gap_secs: float = 8.0
    enable_llm: bool = True
    log_dir: Path | None = None
    max_s1_traces: int = 50
    max_s2_traces: int = 30
    max_images: int = 80


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
        self._last_building_counts: dict[str, int] = {}
        self._last_stray_escalation_at: float | None = None
        self._last_push_escalation_at: float | None = None
        self._revision = 0

        self._trace_seq = 0
        self._s1_traces: deque[S1Trace] = deque(maxlen=self._config.max_s1_traces)
        self._s2_traces: deque[S2Trace] = deque(maxlen=self._config.max_s2_traces)
        self._images: OrderedDict[str, bytes] = OrderedDict()
        self._log_lock = threading.Lock()
        if self._config.log_dir is not None:
            (self._config.log_dir / "images").mkdir(parents=True, exist_ok=True)

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
            self._reconcile_macro_progress_locked(state)
            # Auto-escalate to S2 the first time enemy base is scouted.
            if state.enemy_base is not None and not self._enemy_base_known:
                self._enemy_base_known = True
                self._request_s2_locked("event:enemy_base_discovered", now)
            self._check_tactical_escalations_locked(state, now)
            return self._build_bundle_locked(state, now)

    def _reconcile_macro_progress_locked(self, state: GameState) -> None:
        """Clear completed expand_target and drain queued_structures as S0 builds them."""
        exp = self._action_state.expand_target
        if exp is not None:
            has_nexus_near = any(
                b.unit_type.lower() == "nexus"
                and math.hypot(b.x - exp.x, b.y - exp.y) <= 10.0
                for b in state.our_buildings
            )
            if has_nexus_near:
                self._action_state.expand_target = None
                self._directive = self._directive.model_copy(
                    update={"want_expand": False, "expand_target": None}
                )
                self._revision += 1

        counts = state.building_counts()
        if self._action_state.queued_structures and self._last_building_counts:
            for btype, count in counts.items():
                delta = count - self._last_building_counts.get(btype, 0)
                while delta > 0:
                    idx = next(
                        (
                            i
                            for i, s in enumerate(self._action_state.queued_structures)
                            if s.value == btype
                        ),
                        None,
                    )
                    if idx is None:
                        break
                    self._action_state.queued_structures.pop(idx)
                    delta -= 1
        self._last_building_counts = counts

    def _check_tactical_escalations_locked(
        self, state: GameState, now: float
    ) -> None:
        """Auto-escalate to S2 when stray units wander/chase or army hits push threshold."""
        active_posture = (
            self._last_s1.posture
            if self._last_s1 is not None
            else self._directive.posture_hint
        )
        active_target = (
            self._last_s1.army_target
            if self._last_s1 is not None
            else self._default_target_locked(state)
        )
        # 1. Alert S2 if combat units drift far from target while not pushing (e.g. SCV chase).
        if active_posture != TacticalPosture.PUSH:
            strays = state.stray_combat_units(
                reference=active_target, max_dist_tiles=20.0
            )
            if strays and (
                self._last_stray_escalation_at is None
                or (now - self._last_stray_escalation_at) >= 20.0
            ):
                self._last_stray_escalation_at = now
                self._request_s2_locked("event:stray_units_chasing", now)

        # 2. Alert S2 when army reaches 1-base early-push or 2-base timing-push size.
        if self._directive.posture_hint != TacticalPosture.PUSH:
            nexuses = state.building_counts().get("nexus", 1)
            combat_count = len(state.combat_units())
            threshold = 5 if nexuses <= 1 else 10
            if combat_count >= threshold and (
                self._last_push_escalation_at is None
                or (now - self._last_push_escalation_at) >= 30.0
            ):
                self._last_push_escalation_at = now
                self._request_s2_locked("event:army_ready_to_push", now)

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
            leash_army=self._directive.leash_army,
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

    def _store_image_locked(self, image_id: str, png_bytes: bytes) -> None:
        """Store a rendered minimap PNG in the bounded in-memory LRU buffer."""
        self._images[image_id] = png_bytes
        self._images.move_to_end(image_id)
        while len(self._images) > self._config.max_images:
            self._images.popitem(last=False)

    def _persist_trace(
        self,
        trace: S1Trace | S2Trace,
        png_bytes: bytes | None,
    ) -> None:
        """Append trace JSONL and write PNG to ``log_dir`` when configured."""
        log_dir = self._config.log_dir
        if log_dir is None:
            return
        try:
            with self._log_lock:
                if trace.image_id and png_bytes is not None:
                    img_path = log_dir / "images" / f"{trace.image_id}.png"
                    img_path.write_bytes(png_bytes)
                with (log_dir / "traces.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(trace.model_dump_json() + "\n")
        except OSError as exc:
            logger.warning("Failed to persist trace %s: %s", trace.id, exc)

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
        png_bytes = png_buf.getvalue()
        legend = minimap_legend(state)
        sys_prompt, user_prompt = self._s1_agent.build_prompts(
            state, legend, directive, build_status, army_target=prev_target
        )

        start = time.perf_counter()
        out: S1Output | None = None
        err_msg: str | None = None
        try:
            out = self._s1_agent.step(
                state, minimap, legend, directive, build_status, army_target=prev_target
            )
        except (DiffusionGemmaError, Exception) as exc:  # noqa: BLE001
            logger.warning("S1 step failed: %s", exc)
            err_msg = str(exc)

        latency = time.perf_counter() - start
        now = time.monotonic()
        wall_ts = time.time()
        with self._lock:
            self._trace_seq += 1
            trace_id = self._trace_seq
            image_id = f"s1_{trace_id:05d}_f{state.frame}"
            self._latest_minimap_png = png_bytes
            self._store_image_locked(image_id, png_bytes)
            trace = S1Trace(
                id=trace_id,
                timestamp=wall_ts,
                frame=state.frame,
                latency_secs=round(latency, 3),
                image_id=image_id,
                system_prompt=sys_prompt,
                user_prompt=user_prompt,
                output=out,
                error=err_msg,
            )
            self._s1_traces.append(trace)
            if out is not None:
                self._last_s1 = out
                self._recent_s1.append(out)
                self._last_s1_at = now
                self._last_s1_latency = latency
                self._last_s1_error = None
                self._revision += 1
                if out.escalate_to_s2:
                    reason = f"s1_escalation:{out.escalation_reason or 'unspecified'}"
                    self._request_s2_locked(reason, now)
            else:
                self._last_s1_error = err_msg

        self._persist_trace(trace, png_bytes)
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
        png_buf = io.BytesIO()
        minimap.save(png_buf, format="PNG")
        png_bytes = png_buf.getvalue()
        legend = minimap_legend(state)

        capture = S2RunCapture()
        start = time.perf_counter()
        result: S2Result | None = None
        err_msg: str | None = None
        try:
            result = self._s2_agent.plan(
                state=state,
                minimap=minimap,
                legend=legend,
                action_state=working_action_state,
                recent_s1=recent_s1,
                current_directive=directive,
                reason=reason,
                capture=capture,
                army_target=prev_target,
            )
        except (DiffusionGemmaError, Exception) as exc:  # noqa: BLE001
            logger.warning("S2 plan failed: %s", exc)
            err_msg = str(exc)

        latency = (
            result.latency_secs if result is not None else (time.perf_counter() - start)
        )
        now = time.monotonic()
        wall_ts = time.time()
        with self._lock:
            self._trace_seq += 1
            trace_id = self._trace_seq
            image_id = f"s2_{trace_id:05d}_f{state.frame}"
            self._latest_minimap_png = png_bytes
            self._store_image_locked(image_id, png_bytes)
            trace = S2Trace(
                id=trace_id,
                timestamp=wall_ts,
                frame=state.frame,
                reason=reason,
                latency_secs=round(latency, 3),
                image_id=image_id,
                step1_system_prompt=capture.step1_system_prompt,
                step1_user_prompt=capture.step1_user_prompt,
                step1_reasoning=capture.step1_reasoning,
                step1_summary=capture.step1_summary,
                tool_calls=list(capture.tool_traces),
                step2_system_prompt=capture.step2_system_prompt,
                step2_user_prompt=capture.step2_user_prompt,
                directive=result.directive if result is not None else None,
                error=err_msg,
            )
            self._s2_traces.append(trace)
            if result is not None:
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
                    self._action_state.expand_target = (
                        working_action_state.expand_target
                    )
                self._last_s2_result = result
                self._last_s2_at = now
                self._last_s2_error = None
                self._revision += 1
            else:
                self._last_s2_error = err_msg

        self._persist_trace(trace, png_bytes)
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

    def get_image_png(self, image_id: str) -> bytes | None:
        """Return the exact minimap PNG associated with ``image_id``, if retained."""
        if not image_id or "/" in image_id or "\\" in image_id or ".." in image_id:
            return None
        with self._lock:
            png = self._images.get(image_id)
        if png is not None:
            return png
        if self._config.log_dir is not None:
            img_path = self._config.log_dir / "images" / f"{image_id}.png"
            if img_path.is_file():
                try:
                    return img_path.read_bytes()
                except OSError:
                    return None
        return None

    def traces_snapshot(self, since_id: int = 0, limit: int = 50) -> TracesSnapshot:
        """Return incremental S1/S2 trace history and current command bundle."""
        now = time.monotonic()
        bounded_limit = max(1, min(limit, 200))
        with self._lock:
            bundle = (
                self._build_bundle_locked(self._latest_state, now)
                if self._latest_state is not None
                else None
            )
            s1_list = [t for t in self._s1_traces if t.id > since_id][-bounded_limit:]
            s2_list = [t for t in self._s2_traces if t.id > since_id][-bounded_limit:]
            return TracesSnapshot(
                revision=self._revision,
                has_state=self._latest_state is not None,
                frame=self._latest_state.frame if self._latest_state else None,
                command_bundle=bundle,
                latest_trace_id=self._trace_seq,
                s1_traces=s1_list,
                s2_traces=s2_list,
            )

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
