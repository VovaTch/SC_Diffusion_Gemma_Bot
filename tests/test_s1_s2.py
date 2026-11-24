"""Tests for S1 structured output, S2 two-step tool + directive flow, and priority gate."""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from starcraft_diffusion_gemma.brain.coordinator import (
    BotCoordinator,
    CoordinatorConfig,
)
from starcraft_diffusion_gemma.brain.gate import PriorityGate
from starcraft_diffusion_gemma.llm.client import DiffusionGemmaError, JsonDict
from starcraft_diffusion_gemma.state.models import (
    EmergencyAction,
    GamePhase,
    GameState,
    ProtossStructure,
    S1Output,
    S2Directive,
    TacticalPosture,
    ThreatLevel,
    TilePoint,
    UnitSnapshot,
)


class FakeTransport:
    """Scripted Transport that records requests and returns queued chat responses."""

    def __init__(self, responses: list[JsonDict | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, JsonDict]] = []

    def post(self, path: str, body: JsonDict) -> JsonDict:
        self.calls.append((path, body))
        if not self._responses:
            raise AssertionError("FakeTransport ran out of scripted responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _chat_reply(
    content: str, tool_calls: list[dict[str, Any]] | None = None
) -> JsonDict:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "choices": [{"message": msg, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }


def _midgame_state() -> GameState:
    return GameState(
        frame=4800,
        map_name="(2)Benzene.scx",
        map_width=128,
        map_height=128,
        minerals=420,
        gas=180,
        supply_used=26,
        supply_total=33,
        our_base=TilePoint(x=16, y=112),
        choke_point=TilePoint(x=32, y=96),
        enemy_base=TilePoint(x=112, y=16),
        our_units=[
            UnitSnapshot(id=1, unit_type="zealot", x=34, y=92, hp=100, shields=60),
            UnitSnapshot(id=2, unit_type="dragoon", x=32, y=94, hp=100, shields=80),
        ],
        our_buildings=[
            UnitSnapshot(id=10, unit_type="nexus", x=16, y=112, hp=750, shields=750),
            UnitSnapshot(id=11, unit_type="gateway", x=22, y=106, hp=500, shields=500),
            UnitSnapshot(id=12, unit_type="gateway", x=26, y=106, hp=500, shields=500),
        ],
        enemy_units=[
            UnitSnapshot(id=50, unit_type="marine", x=60, y=68, hp=40),
        ],
    )


def test_s1_step_sends_image_and_updates_command_bundle() -> None:
    s1_payload = S1Output(
        posture=TacticalPosture.DEFEND_NATURAL,
        army_target=TilePoint(x=48, y=80),
        emergency_action=EmergencyAction.EMERGENCY_ZEALOT,
        threat_level=ThreatLevel.HIGH,
        escalate_to_s2=True,
        escalation_reason="Marine push approaching natural",
    )
    transport = FakeTransport([_chat_reply(s1_payload.model_dump_json())])
    coord = BotCoordinator(transport, CoordinatorConfig(enable_llm=False))
    coord.on_tick(_midgame_state())

    out = coord.step_s1_once()
    assert out is not None
    assert out.posture == TacticalPosture.DEFEND_NATURAL
    assert out.army_target == TilePoint(x=48, y=80)

    # Verify multimodal image_url was included in the S1 request.
    _, req_body = transport.calls[0]
    user_parts = req_body["messages"][-1]["content"]
    assert any(p.get("type") == "image_url" for p in user_parts)

    bundle = coord.on_tick(_midgame_state())
    assert bundle.posture == TacticalPosture.DEFEND_NATURAL
    assert bundle.emergency_action == EmergencyAction.EMERGENCY_ZEALOT
    assert bundle.army_target == TilePoint(x=48, y=80)
    assert bundle.revision == 1


def test_s2_two_step_turn_executes_tools_and_emits_directive() -> None:
    tool_call_reply = _chat_reply(
        "",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "queue_structure",
                    "arguments": json.dumps({"structure": "gateway", "count": 1}),
                },
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "set_unit_mix",
                    "arguments": json.dumps(
                        {
                            "zealot_ratio": 0.3,
                            "dragoon_ratio": 0.7,
                            "target_probes": 26,
                            "target_gateways": 3,
                        }
                    ),
                },
            },
        ],
    )
    tool_summary_reply = _chat_reply("Queued 3rd gateway and shifted to 70% Dragoons.")
    directive_payload = S2Directive(
        phase=GamePhase.MID_PUSH,
        posture_hint=TacticalPosture.PUSH,
        attack_target=TilePoint(x=112, y=16),
        target_gateways=3,
        target_probes=26,
        zealot_ratio=0.3,
        dragoon_ratio=0.7,
        want_singularity_charge=True,
        notes="Push once 3rd gateway Dragoons finish.",
    )
    transport = FakeTransport(
        [
            tool_call_reply,
            tool_summary_reply,
            _chat_reply(directive_payload.model_dump_json()),
        ]
    )
    coord = BotCoordinator(transport, CoordinatorConfig(enable_llm=False))
    coord.on_tick(_midgame_state())

    res = coord.step_s2_once(reason="periodic")
    assert res is not None
    assert len(res.tool_calls) == 2
    assert res.directive.phase == GamePhase.MID_PUSH
    assert res.directive.target_gateways == 3

    bundle = coord.on_tick(_midgame_state())
    assert bundle.target_gateways == 3
    assert bundle.target_probes == 26
    assert bundle.dragoon_ratio == 0.7
    assert ProtossStructure.GATEWAY in bundle.queued_structures
    assert bundle.posture == TacticalPosture.PUSH


def test_llm_failure_degrades_gracefully_without_breaking_s0() -> None:
    transport = FakeTransport([DiffusionGemmaError("vLLM offline")])
    coord = BotCoordinator(transport, CoordinatorConfig(enable_llm=False))
    state = _midgame_state()
    bundle_before = coord.on_tick(state)
    assert coord.step_s1_once() is None
    bundle_after = coord.on_tick(state)
    assert bundle_after.posture == bundle_before.posture
    assert coord.debug_snapshot()["last_s1_error"] is not None


def test_priority_gate_runs_s1_ahead_of_queued_s2() -> None:
    gate = PriorityGate()
    order: list[str] = []
    holder_entered = threading.Event()
    release_holder = threading.Event()

    def holder() -> None:
        with gate.acquire(priority=1):
            holder_entered.set()
            release_holder.wait(timeout=2.0)

    def waiter(label: str, priority: int) -> None:
        with gate.acquire(priority=priority):
            order.append(label)

    t_hold = threading.Thread(target=holder)
    t_hold.start()
    assert holder_entered.wait(timeout=1.0)

    t_s2 = threading.Thread(target=waiter, args=("s2", 1))
    t_s1 = threading.Thread(target=waiter, args=("s1", 0))
    t_s2.start()
    time.sleep(0.05)  # ensure S2 enters queue before S1
    t_s1.start()
    time.sleep(0.05)
    release_holder.set()

    t_hold.join(timeout=2.0)
    t_s2.join(timeout=2.0)
    t_s1.join(timeout=2.0)
    assert order == ["s1", "s2"]
