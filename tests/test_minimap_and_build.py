"""Tests for Protoss Zealot + Dragoon build order and minimap rendering."""

from __future__ import annotations

from starcraft_diffusion_gemma.protoss.build_order import BuildOrderTracker
from starcraft_diffusion_gemma.state.models import (
    BaseLocation,
    GameState,
    ProtossStructure,
    ProtossUnit,
    ProtossUpgrade,
    TilePoint,
    UnitSnapshot,
)
from starcraft_diffusion_gemma.vision.minimap import (
    MINIMAP_PX,
    minimap_legend,
    render_game_minimap,
)


def _sample_state() -> GameState:
    return GameState(
        frame=2400,
        map_name="(2)Benzene.scx",
        map_width=128,
        map_height=112,
        minerals=250,
        gas=100,
        supply_used=14,
        supply_total=17,
        our_base=TilePoint(x=16, y=96),
        enemy_base=TilePoint(x=112, y=16),
        our_units=[
            UnitSnapshot(id=1, unit_type="probe", x=18, y=95, hp=20, shields=20),
            UnitSnapshot(id=2, unit_type="zealot", x=30, y=80, hp=100, shields=60),
        ],
        our_buildings=[
            UnitSnapshot(id=10, unit_type="nexus", x=16, y=96, hp=750, shields=750),
            UnitSnapshot(id=11, unit_type="pylon", x=22, y=92, hp=300, shields=300),
            UnitSnapshot(id=12, unit_type="gateway", x=24, y=88, hp=500, shields=500),
        ],
        enemy_units=[
            UnitSnapshot(id=99, unit_type="marine", x=64, y=56, hp=40),
        ],
        expansions=[
            BaseLocation(x=16, y=56),
            BaseLocation(x=112, y=56),
        ],
    )


def test_build_order_advances_through_opening() -> None:
    tracker = BuildOrderTracker()
    state = GameState(frame=100, supply_used=8)
    first = tracker.next_step(state)
    assert first is not None
    assert first.action == "build"
    assert first.target == ProtossStructure.PYLON

    # Satisfy Pylon + Gateway + Assimilator + Zealot + Cybernetics Core
    state = GameState(
        frame=3000,
        supply_used=16,
        our_buildings=[
            UnitSnapshot(id=1, unit_type="pylon", x=10, y=10, hp=300),
            UnitSnapshot(id=2, unit_type="gateway", x=12, y=10, hp=500),
            UnitSnapshot(id=3, unit_type="assimilator", x=14, y=10, hp=450),
            UnitSnapshot(id=4, unit_type="cybernetics_core", x=16, y=10, hp=500),
        ],
        our_units=[
            UnitSnapshot(id=5, unit_type="zealot", x=20, y=20, hp=100),
        ],
    )
    nxt = tracker.next_step(state)
    assert nxt is not None
    assert nxt.target == ProtossStructure.PYLON
    assert nxt.count == 2

    # Satisfy Singularity Charge as in-progress
    state.upgrades_in_progress.append(ProtossUpgrade.SINGULARITY_CHARGE)
    state.our_buildings.extend(
        [
            UnitSnapshot(id=6, unit_type="pylon", x=10, y=14, hp=300),
            UnitSnapshot(id=7, unit_type="gateway", x=12, y=14, hp=500),
        ]
    )
    nxt = tracker.next_step(state)
    assert nxt is not None
    assert nxt.action == "train"
    assert nxt.target == ProtossUnit.DRAGOON


def test_minimap_renders_expected_dimensions_and_legend() -> None:
    state = _sample_state()
    img = render_game_minimap(state, army_target=TilePoint(x=48, y=64))
    assert img.size == (MINIMAP_PX, MINIMAP_PX)
    assert img.mode == "RGB"
    legend = minimap_legend(state)
    assert "128x112" in legend
    assert "orange cross" in legend
