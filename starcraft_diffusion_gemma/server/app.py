"""
Non-blocking FastAPI policy server for the StarCraft DiffusionGemma bot.

Run from the project root:
    uv run python -m starcraft_diffusion_gemma.server.app --host 0.0.0.0 --port 8080

The JBWAPI S0 bridge running inside ``sc_net`` Docker POSTs ``GameState``
snapshots to ``http://172.18.0.1:8080/tick`` and immediately receives the
latest ``CommandBundle`` without blocking on vLLM inference.
"""

from __future__ import annotations

import argparse
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response

from starcraft_diffusion_gemma.brain.coordinator import (
    BotCoordinator,
    CoordinatorConfig,
)
from starcraft_diffusion_gemma.brain.traces import TracesSnapshot
from starcraft_diffusion_gemma.llm.client import DEFAULT_MODEL, HttpTransport
from starcraft_diffusion_gemma.state.models import CommandBundle, GameState


def create_app(
    coordinator: BotCoordinator | None = None,
    vllm_url: str = "http://localhost:8000/v1/",
    model: str = DEFAULT_MODEL,
    config: CoordinatorConfig | None = None,
) -> FastAPI:
    """
    Create the FastAPI policy server application.

    Args:
        coordinator (BotCoordinator | None): Optional pre-built coordinator
            (useful for tests with a fake transport).
        vllm_url (str): Local vLLM OpenAI-compatible API root.
        model (str): Served DiffusionGemma model id.
        config (CoordinatorConfig | None): S1/S2 cadence settings.

    Returns:
        FastAPI: Configured ASGI application.
    """
    coord = coordinator or BotCoordinator(
        HttpTransport(base_url=vllm_url),
        config=config,
        model=model,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        coord.start()
        try:
            yield
        finally:
            coord.stop()

    app = FastAPI(title="StarCraft DiffusionGemma Policy Server", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/tick", response_model=CommandBundle)
    def tick(state: GameState) -> CommandBundle:
        return coord.on_tick(state)

    @app.get("/debug")
    def debug() -> dict[str, Any]:
        return coord.debug_snapshot()

    @app.get("/debug/minimap.png")
    def debug_minimap() -> Response:
        png = coord.latest_minimap_png()
        if png is None:
            raise HTTPException(status_code=404, detail="No minimap rendered yet")
        return Response(content=png, media_type="image/png")

    @app.get("/debug/traces", response_model=TracesSnapshot)
    def debug_traces(
        since_id: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> TracesSnapshot:
        return coord.traces_snapshot(since_id=since_id, limit=limit)

    @app.get("/debug/images/{image_id}.png")
    def debug_image(image_id: str) -> Response:
        png = coord.get_image_png(image_id)
        if png is None:
            raise HTTPException(status_code=404, detail=f"Image {image_id!r} not found")
        return Response(content=png, media_type="image/png")

    return app


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags for launching the host policy server."""
    parser = argparse.ArgumentParser(
        description="Run the host S1/S2 policy server for the Protoss bot."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1/")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--s1-interval", type=float, default=2.5)
    parser.add_argument("--s2-interval", type=float, default=30.0)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Optional directory to persist traces.jsonl and sent minimap PNGs",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable background S1/S2 LLM loops (pure S0 build-order mode)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Launch uvicorn serving the policy server."""
    opts = parse_args(argv)
    cfg = CoordinatorConfig(
        s1_interval_secs=opts.s1_interval,
        s2_interval_secs=opts.s2_interval,
        enable_llm=not opts.no_llm,
        log_dir=opts.log_dir,
    )
    app = create_app(vllm_url=opts.vllm_url, model=opts.model, config=cfg)
    uvicorn.run(app, host=opts.host, port=opts.port)


if __name__ == "__main__":
    main()
