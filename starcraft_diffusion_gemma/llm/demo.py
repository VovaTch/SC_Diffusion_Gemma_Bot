"""
End-to-end capability check against a live DiffusionGemma server.

Run from the project root (server must be up, see docs/diffusion_gemma.md):
    uv run python -m starcraft_diffusion_gemma.llm.demo
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from starcraft_diffusion_gemma.llm import (
    DiffusionGemmaClient,
    HttpTransport,
    Question,
    ToolCall,
    image_part,
    tool_spec,
    user,
)

TILES = 128  # map size in build tiles
SCALE = 4  # pixels per tile
OUR_BASE = (16, 112)
ENEMY_BASE = (112, 16)
ENEMY_ARMY = (60, 70)
EXPANSIONS = [(16, 64), (112, 64), (64, 112), (64, 16)]


def render_minimap() -> Image.Image:
    """
    Draw a synthetic top-down minimap with a labelled tile grid.

    Returns:
        Image.Image: 512x512 RGB minimap.
    """
    px = TILES * SCALE
    img = Image.new("RGB", (px, px), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    for t in range(0, TILES + 1, 16):
        draw.line([(t * SCALE, 0), (t * SCALE, px)], fill=(70, 70, 70))
        draw.line([(0, t * SCALE), (px, t * SCALE)], fill=(70, 70, 70))
        draw.text((t * SCALE + 2, 2), str(t), fill=(200, 200, 200))
        draw.text((2, t * SCALE + 2), str(t), fill=(200, 200, 200))

    def box(center: tuple[int, int], half: int, color: tuple[int, int, int]) -> None:
        x, y = center[0] * SCALE, center[1] * SCALE
        draw.rectangle([x - half, y - half, x + half, y + half], fill=color)

    for exp in EXPANSIONS:
        box(exp, 6, (60, 140, 255))  # blue = free mineral field
    box(OUR_BASE, 14, (40, 200, 60))  # green = our base
    box(ENEMY_BASE, 14, (220, 40, 40))  # red = enemy base
    for dx, dy in [(0, 0), (3, 1), (-2, 3), (1, -3), (4, 4)]:
        a = (ENEMY_ARMY[0] + dx, ENEMY_ARMY[1] + dy)
        draw.ellipse(
            [a[0] * SCALE - 3, a[1] * SCALE - 3, a[0] * SCALE + 3, a[1] * SCALE + 3],
            fill=(255, 120, 120),
        )  # pink dots = enemy units
    return img


LEGEND = (
    "Minimap of a 128x128-tile StarCraft map. Grid lines every 16 tiles, "
    "labelled with tile coordinates (x right, y down). Green square = our "
    "base, red square = enemy base, blue squares = free mineral expansions, "
    "pink dots = enemy units."
)


class Point(BaseModel):
    """Tile coordinate."""

    x: int = Field(ge=0, le=TILES)
    y: int = Field(ge=0, le=TILES)


class MapReading(BaseModel):
    """Structured reading of the minimap."""

    our_base: Point
    enemy_base: Point
    enemy_army: Point | None = Field(description="centre of enemy units, if any")
    free_expansions: list[Point]
    summary: str = Field(description="one sentence")


class Unit(StrEnum):
    """Trainable Zerg units."""

    DRONE = "drone"
    ZERGLING = "zergling"
    HYDRALISK = "hydralisk"
    OVERLORD = "overlord"


class Structure(StrEnum):
    """Buildable Zerg structures."""

    HATCHERY = "hatchery"
    SUNKEN = "sunken_colony"
    SPIRE = "spire"


class Train(BaseModel):
    unit: Unit
    count: int = Field(ge=1, le=24)


class Build(BaseModel):
    structure: Structure
    x: int
    y: int


class Move(BaseModel):
    x: int
    y: int


TOOL_ARGS: dict[str, type[BaseModel]] = {
    "train": Train,
    "build": Build,
    "attack_move": Move,
}
TOOL_DOCS = {
    "train": "Queue unit production at hatcheries.",
    "build": "Place a structure at a tile coordinate.",
    "attack_move": "Send the army to attack-move to a tile.",
}
TOOL_NAMES = set(TOOL_ARGS)
TOOLS = [tool_spec(n, TOOL_DOCS[n], m) for n, m in TOOL_ARGS.items()]


def near(p: Point | None, ref: tuple[int, int], tol: int = 12) -> bool:
    """
    Check a point lies within ``tol`` tiles of ``ref`` on both axes.

    Args:
        p (Point | None): Predicted point.
        ref (tuple[int, int]): Ground truth.
        tol (int): Tolerance in tiles.

    Returns:
        bool: True if close enough.
    """
    return p is not None and abs(p.x - ref[0]) <= tol and abs(p.y - ref[1]) <= tol


def timed(label: str, fn: Callable[[], Any], check: Callable[[Any], bool]) -> Any:
    """
    Run a probe, print timing, output and PASS/FAIL.

    Args:
        label (str): Probe name.
        fn (Callable[[], Any]): Probe.
        check (Callable[[Any], bool]): Success predicate on the result.

    Returns:
        Any: Probe result (or the exception raised).
    """
    start = time.perf_counter()
    try:
        out = fn()
        ok = check(out)
    except Exception as exc:  # noqa: BLE001 - report every failure
        out, ok = exc, False
    secs = time.perf_counter() - start
    shown = out.model_dump() if isinstance(out, BaseModel) else out
    print(f"\n[{'PASS' if ok else 'FAIL'}] {label} ({secs:.2f}s)\n  {shown}")
    return out


def main() -> None:
    """Run every capability probe."""
    client = DiffusionGemmaClient(HttpTransport())
    img = image_part(render_minimap())
    render_minimap().save("docs/minimap_demo.png")

    timed(
        "image -> text",
        lambda: client.chat([user(img, LEGEND, "Describe the situation.")]).content,
        lambda s: "red" in s.lower() or "enemy" in s.lower(),
    )
    reading_ok: Callable[[MapReading], bool] = lambda r: near(
        r.enemy_base, ENEMY_BASE
    ) and near(r.our_base, OUR_BASE)
    timed(
        "image -> structured (prompted JSON)",
        lambda: client.structured([user(img, LEGEND, "Read the map.")], MapReading),
        reading_ok,
    )
    timed(
        "image -> structured (forced tool call)",
        lambda: client.structured_via_tool(
            [user(img, LEGEND, "Read the map.")], MapReading
        ),
        reading_ok,
    )

    order = (
        "We are Zerg with 300 minerals, 14/18 supply, 12 zerglings. "
        "Defend against the incoming enemy units, then counterattack the "
        "enemy base. Use the tools; call as many as needed."
    )
    msgs = [user(img, LEGEND, order)]
    timed(
        "image -> tool calls (auto, single turn)",
        lambda: client.chat(msgs, tools=TOOLS).tool_calls,
        lambda cs: len(cs) > 0 and all(c.name in TOOL_NAMES for c in cs),
    )

    def execute(call: ToolCall) -> str:
        args = TOOL_ARGS[call.name].model_validate(call.arguments)
        return json.dumps({"ok": True, "executed": args.model_dump(mode="json")})

    timed(
        "agent loop: tool calls -> results -> final answer",
        lambda: (lambda r: (r[0].content, [(c.name, c.arguments) for c in r[1]]))(
            client.run_tools(msgs, TOOLS, execute)
        ),
        lambda out: len(out[0]) > 0 and len(out[1]) > 0,
    )

    timed(
        "decisions (joint, text-only)",
        lambda: client.decide(
            "Zerg vs Terran, minute 5. Enemy has 2 barracks, no factory; "
            "8 marines approaching our natural.",
            [
                Question(
                    "choice",
                    "plan",
                    "Best response?",
                    {
                        "drone_hard": "greedy economy",
                        "sunken_defense": "build sunken colonies",
                        "mutalisk_tech": "rush lair and spire",
                    },
                ),
                Question("predicate", "under_attack", "Is an attack imminent?"),
                Question(
                    "score",
                    "threat",
                    "Threat level",
                    {"low": "", "mid": "", "high": ""},
                ),
            ],
        ),
        lambda a: len(a) == 3 and a[0]["choice"] == "sunken_defense",
    )


if __name__ == "__main__":
    main()
