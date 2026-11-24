"""Tests for the non-blocking FastAPI policy server."""

from __future__ import annotations

from fastapi.testclient import TestClient

from starcraft_diffusion_gemma.brain.coordinator import (
    BotCoordinator,
    CoordinatorConfig,
)
from starcraft_diffusion_gemma.llm.client import JsonDict
from starcraft_diffusion_gemma.server.app import create_app
from starcraft_diffusion_gemma.state.models import (
    GameState,
    S1Output,
    TacticalPosture,
    TilePoint,
)


class _DummyTransport:
    def post(self, path: str, body: JsonDict) -> JsonDict:
        out = S1Output(
            posture=TacticalPosture.HOLD_CHOKE,
            army_target=TilePoint(x=32, y=96),
        )
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": out.model_dump_json()},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }


def test_tick_and_debug_endpoints() -> None:
    coord = BotCoordinator(_DummyTransport(), CoordinatorConfig(enable_llm=False))
    app = create_app(coordinator=coord)

    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/debug/minimap.png").status_code == 404

        state = GameState(
            frame=240,
            minerals=150,
            supply_used=8,
            supply_total=9,
            our_base=TilePoint(x=16, y=112),
            choke_point=TilePoint(x=32, y=96),
        )
        resp = client.post("/tick", json=state.model_dump())
        assert resp.status_code == 200
        bundle = resp.json()
        assert bundle["frame_issued"] == 240
        assert bundle["active_build_step"]["target"] == "pylon"

        # Trigger one S1 step so minimap PNG is cached and served at /debug/minimap.png
        assert coord.step_s1_once() is not None
        png_resp = client.get("/debug/minimap.png")
        assert png_resp.status_code == 200
        assert png_resp.headers["content-type"] == "image/png"
        assert png_resp.content.startswith(b"\x89PNG")

        dbg = client.get("/debug").json()
        assert dbg["has_state"] is True
        assert dbg["last_s1"]["posture"] == "hold_choke"
