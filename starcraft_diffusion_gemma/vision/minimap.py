"""
Render a live BWAPI ``GameState`` into a 512x512 grid-labelled minimap PNG.

Follows the visual conventions verified in ``docs/diffusion_gemma.md`` §3.1:
- 16-tile coordinate grid with edge labels (x right, y down, build-tile units).
- High-contrast flat colours matched to an explicit prompt legend.
"""

from __future__ import annotations

from PIL import Image, ImageDraw

from starcraft_diffusion_gemma.state.models import GameState, TilePoint

MINIMAP_PX = 512
GRID_STEP_TILES = 16

# High-contrast flat palette for DiffusionGemma vision.
BG_COLOR = (28, 28, 32)
GRID_COLOR = (72, 72, 78)
LABEL_COLOR = (210, 210, 215)
OUR_BASE_COLOR = (40, 210, 70)
OUR_BLDG_COLOR = (40, 150, 120)
OUR_ARMY_COLOR = (80, 230, 255)
OUR_PROBE_COLOR = (220, 210, 80)
ENEMY_BASE_COLOR = (225, 45, 45)
ENEMY_UNIT_COLOR = (255, 120, 150)
FREE_EXP_COLOR = (60, 140, 255)
TARGET_COLOR = (255, 165, 0)


def minimap_legend(state: GameState) -> str:
    """
    Build the text legend describing the rendered minimap for DiffusionGemma.

    Args:
        state (GameState): Game state whose map dimensions are referenced.

    Returns:
        str: Concise legend prompt fragment.
    """
    return (
        f"Minimap of a {state.map_width}x{state.map_height}-tile StarCraft map. "
        f"Grid lines every {GRID_STEP_TILES} tiles, labelled in tile coordinates "
        "(x right, y down). Green square = our Nexus/base, teal squares = our "
        "buildings, bright cyan dots = our army (Zealots/Dragoons), yellow dots = "
        "our Probes, red squares = enemy base/buildings, pink dots = enemy units, "
        "blue squares = free mineral expansions, orange cross = current army target."
    )


def render_game_minimap(
    state: GameState,
    army_target: TilePoint | None = None,
    size_px: int = MINIMAP_PX,
) -> Image.Image:
    """
    Render ``state`` to a square RGB minimap image with a labelled tile grid.

    Args:
        state (GameState): Current BWAPI game state.
        army_target (TilePoint | None): Optional active army target marker.
        size_px (int): Output image width and height in pixels (default 512).

    Returns:
        Image.Image: Rendered RGB minimap.
    """
    img = Image.new("RGB", (size_px, size_px), BG_COLOR)
    draw = ImageDraw.Draw(img)

    w = max(state.map_width, 1)
    h = max(state.map_height, 1)

    def to_px(tx: int, ty: int) -> tuple[int, int]:
        px = min(size_px - 1, max(0, round(tx * size_px / w)))
        py = min(size_px - 1, max(0, round(ty * size_px / h)))
        return px, py

    def draw_box(tx: int, ty: int, half: int, color: tuple[int, int, int]) -> None:
        cx, cy = to_px(tx, ty)
        draw.rectangle([cx - half, cy - half, cx + half, cy + half], fill=color)

    def draw_dot(tx: int, ty: int, radius: int, color: tuple[int, int, int]) -> None:
        cx, cy = to_px(tx, ty)
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color)

    for tx in range(0, w + 1, GRID_STEP_TILES):
        x, _ = to_px(tx, 0)
        draw.line([(x, 0), (x, size_px)], fill=GRID_COLOR)
        if tx < w:
            draw.text((x + 2, 2), str(tx), fill=LABEL_COLOR)
    for ty in range(0, h + 1, GRID_STEP_TILES):
        _, y = to_px(0, ty)
        draw.line([(0, y), (size_px, y)], fill=GRID_COLOR)
        if ty < h:
            draw.text((2, y + 2), str(ty), fill=LABEL_COLOR)

    for exp in state.expansions:
        if exp.is_occupied_by_us:
            draw_box(exp.x, exp.y, 10, OUR_BASE_COLOR)
        elif exp.is_occupied_by_enemy:
            draw_box(exp.x, exp.y, 10, ENEMY_BASE_COLOR)
        else:
            draw_box(exp.x, exp.y, 6, FREE_EXP_COLOR)

    if state.our_base is not None:
        draw_box(state.our_base.x, state.our_base.y, 12, OUR_BASE_COLOR)
    if state.enemy_base is not None:
        draw_box(state.enemy_base.x, state.enemy_base.y, 12, ENEMY_BASE_COLOR)

    for bldg in state.our_buildings:
        is_nexus = "nexus" in bldg.unit_type.lower()
        draw_box(
            bldg.x,
            bldg.y,
            10 if is_nexus else 5,
            OUR_BASE_COLOR if is_nexus else OUR_BLDG_COLOR,
        )

    for bldg in state.enemy_buildings:
        draw_box(bldg.x, bldg.y, 7, ENEMY_BASE_COLOR)

    for unit in state.our_units:
        if "probe" in unit.unit_type.lower():
            draw_dot(unit.x, unit.y, 2, OUR_PROBE_COLOR)
        else:
            draw_dot(unit.x, unit.y, 3, OUR_ARMY_COLOR)

    for unit in state.enemy_units:
        draw_dot(unit.x, unit.y, 3, ENEMY_UNIT_COLOR)

    if army_target is not None:
        cx, cy = to_px(army_target.x, army_target.y)
        arm = 7
        draw.line([(cx - arm, cy), (cx + arm, cy)], fill=TARGET_COLOR, width=2)
        draw.line([(cx, cy - arm), (cx, cy + arm)], fill=TARGET_COLOR, width=2)

    return img
