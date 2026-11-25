"""Tests for S1/S2 trace capture, --log-dir persistence, /debug/traces, and PySide6 dashboard."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from starcraft_diffusion_gemma.brain.coordinator import (
    BotCoordinator,
    CoordinatorConfig,
)
from starcraft_diffusion_gemma.brain.traces import TracesSnapshot
from starcraft_diffusion_gemma.llm.client import DiffusionGemmaError
from starcraft_diffusion_gemma.server.app import create_app, parse_args
from starcraft_diffusion_gemma.state.models import GameState, TilePoint, UnitSnapshot


class ScriptedTransport:
    """Fake transport returning queued responses or raising exceptions."""

    def __init__(self, responses: list[dict[str, Any] | Exception]) -> None:
        self._responses = list(responses)

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        del path, body
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _chat_reply(
    content: str,
    tool_calls: list[dict[str, Any]] | None = None,
    reasoning: str | None = None,
) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    if reasoning is not None:
        msg["reasoning"] = reasoning
    return {"choices": [{"message": msg, "finish_reason": "stop"}], "usage": {}}


def _sample_state(frame: int = 2400) -> GameState:
    return GameState(
        frame=frame,
        minerals=320,
        gas=140,
        supply_used=24,
        supply_total=34,
        our_base=TilePoint(x=16, y=104),
        choke_point=TilePoint(x=32, y=88),
        enemy_base=TilePoint(x=112, y=20),
        our_units=[
            UnitSnapshot(id=1, unit_type="Protoss_Zealot", x=30, y=90, hp=100),
            UnitSnapshot(id=2, unit_type="Protoss_Dragoon", x=32, y=88, hp=100),
        ],
    )


def test_traces_capture_prompts_images_tools_errors_and_log_dir(tmp_path: Path) -> None:
    s1_json = json.dumps(
        {
            "posture": "push",
            "army_target": {"x": 100, "y": 24},
            "emergency_action": "none",
            "threat_level": "low",
            "escalate_to_s2": False,
            "escalation_reason": "",
        }
    )
    s2_tool = _chat_reply(
        "",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "queue_structure",
                    "arguments": json.dumps({"structure": "gateway", "count": 1}),
                },
            }
        ],
    )
    s2_summary = _chat_reply(
        "Added third gateway for mid-game push.",
        reasoning="Economy supports a 3rd gateway before pushing.",
    )
    s2_directive = _chat_reply(
        json.dumps(
            {
                "phase": "mid_push",
                "posture_hint": "push",
                "attack_target": {"x": 112, "y": 20},
                "target_gateways": 3,
                "target_probes": 24,
                "zealot_ratio": 0.4,
                "dragoon_ratio": 0.6,
                "want_singularity_charge": True,
                "want_expand": False,
                "expand_target": None,
                "notes": "Push natural",
            }
        )
    )
    transport = ScriptedTransport(
        [
            _chat_reply(s1_json),
            s2_tool,
            s2_summary,
            s2_directive,
            DiffusionGemmaError("vLLM offline"),
        ]
    )
    log_dir = tmp_path / "match_logs"
    coord = BotCoordinator(
        transport,
        config=CoordinatorConfig(enable_llm=False, log_dir=log_dir),
    )
    coord.on_tick(_sample_state(2400))

    assert coord.step_s1_once() is not None
    assert coord.step_s2_once(reason="periodic") is not None
    # Trigger a failed S1 turn and confirm error trace is still captured.
    assert coord.step_s1_once() is None

    snap = coord.traces_snapshot(since_id=0, limit=50)
    assert snap.latest_trace_id == 3
    assert len(snap.s1_traces) == 2
    assert len(snap.s2_traces) == 1

    t1 = snap.s1_traces[0]
    assert "System 1" in t1.system_prompt
    assert "frame 2400" in t1.user_prompt
    assert t1.output is not None and t1.output.posture.value == "push"
    assert t1.image_id is not None
    assert coord.get_image_png(t1.image_id) is not None

    t2 = snap.s2_traces[0]
    assert "System 2" in t2.step1_system_prompt
    assert "Macro tool calls executed in Step 1" in t2.step2_user_prompt
    assert t2.step1_reasoning == "Economy supports a 3rd gateway before pushing."
    assert len(t2.tool_calls) == 1
    assert t2.tool_calls[0].name == "queue_structure"
    assert '"ok": true' in t2.tool_calls[0].result
    assert t2.directive is not None and t2.directive.target_gateways == 3
    assert t2.image_id is not None

    t3 = snap.s1_traces[1]
    assert t3.output is None
    assert t3.error is not None and "vLLM offline" in t3.error

    # Verify incremental cursor works.
    inc = coord.traces_snapshot(since_id=2, limit=50)
    assert [t.id for t in inc.s1_traces] == [3]
    assert inc.s2_traces == []

    # Verify --log-dir persistence.
    jsonl_lines = (log_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(jsonl_lines) == 3
    assert (log_dir / "images" / f"{t1.image_id}.png").is_file()
    assert (log_dir / "images" / f"{t2.image_id}.png").is_file()


def test_debug_traces_and_image_endpoints(tmp_path: Path) -> None:
    s1_json = json.dumps(
        {
            "posture": "hold_choke",
            "army_target": {"x": 32, "y": 88},
            "emergency_action": "none",
            "threat_level": "medium",
            "escalate_to_s2": False,
            "escalation_reason": "",
        }
    )
    coord = BotCoordinator(
        ScriptedTransport([_chat_reply(s1_json)]),
        config=CoordinatorConfig(enable_llm=False, log_dir=tmp_path),
    )
    app = create_app(coordinator=coord)

    with TestClient(app) as client:
        client.post("/tick", json=_sample_state(1800).model_dump())
        coord.step_s1_once()

        res = client.get("/debug/traces", params={"since_id": 0, "limit": 20})
        assert res.status_code == 200
        snap = TracesSnapshot.model_validate(res.json())
        assert len(snap.s1_traces) == 1
        img_id = snap.s1_traces[0].image_id
        assert img_id is not None

        img_res = client.get(f"/debug/images/{img_id}.png")
        assert img_res.status_code == 200
        assert img_res.headers["content-type"] == "image/png"
        assert img_res.content.startswith(b"\x89PNG")

        missing = client.get("/debug/images/nonexistent.png")
        assert missing.status_code == 404

    opts = parse_args(["--log-dir", str(tmp_path)])
    assert opts.log_dir == tmp_path


def test_pyside6_dashboard_window_offscreen(tmp_path: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from starcraft_diffusion_gemma.dashboard import DashboardWindow, load_log_dir

    s1_json = json.dumps(
        {
            "posture": "defend",
            "army_target": {"x": 20, "y": 100},
            "emergency_action": "none",
            "threat_level": "low",
            "escalate_to_s2": False,
            "escalation_reason": "",
        }
    )
    coord = BotCoordinator(
        ScriptedTransport([_chat_reply(s1_json)]),
        config=CoordinatorConfig(enable_llm=False, log_dir=tmp_path),
    )
    coord.on_tick(_sample_state(900))
    coord.step_s1_once()

    s1_list, s2_list, images = load_log_dir(tmp_path)
    assert len(s1_list) == 1
    assert len(s2_list) == 0
    assert len(images) == 1

    app = QApplication.instance() or QApplication([])
    assert app is not None
    win = DashboardWindow(log_dir=tmp_path, start_poller=False)
    assert win.s1_pane._history.count() == 1
    win.close()
