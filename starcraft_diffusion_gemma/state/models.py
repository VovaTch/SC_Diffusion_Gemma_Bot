"""
Pydantic schemas shared across the JBWAPI S0 bridge, S1 fast loop, and S2 planner.

Intentionally restricted to the initial Protoss Zealot + Dragoon scope so
DiffusionGemma's structured outputs and tool calls stay compact (<256 tokens
for S1) and strongly constrained via ``StrEnum`` and numeric bounds.
"""

from __future__ import annotations

from collections import Counter
from enum import StrEnum
import math
from typing import Literal

from pydantic import BaseModel, Field


class ProtossUnit(StrEnum):
    """Trainable Protoss units in the initial Zealot + Dragoon scope."""

    PROBE = "probe"
    ZEALOT = "zealot"
    DRAGOON = "dragoon"


class ProtossStructure(StrEnum):
    """Buildable Protoss structures in the initial Zealot + Dragoon scope."""

    NEXUS = "nexus"
    PYLON = "pylon"
    GATEWAY = "gateway"
    ASSIMILATOR = "assimilator"
    CYBERNETICS_CORE = "cybernetics_core"


class ProtossUpgrade(StrEnum):
    """Researchable Protoss upgrades in the initial scope."""

    SINGULARITY_CHARGE = "singularity_charge"


class TacticalPosture(StrEnum):
    """Immediate army posture executed by S0 and adjusted by S1/S2."""

    DEFEND_MAIN = "defend_main"
    DEFEND_NATURAL = "defend_natural"
    HOLD_CHOKE = "hold_choke"
    RALLY = "rally"
    PUSH = "push"
    RETREAT = "retreat"


class EmergencyAction(StrEnum):
    """Fast reactive override that S1 can request from S0."""

    NONE = "none"
    PULL_PROBES = "pull_probes"
    EMERGENCY_ZEALOT = "emergency_zealot"
    EMERGENCY_PYLON = "emergency_pylon"


class ThreatLevel(StrEnum):
    """Immediate threat assessment emitted by S1."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GamePhase(StrEnum):
    """High-level strategic game phase set by S2."""

    OPENING = "opening"
    EARLY_MID = "early_mid"
    MID_PUSH = "mid_push"
    DEFENSIVE = "defensive"


class TilePoint(BaseModel):
    """Build-tile coordinate (x right, y down)."""

    x: int = Field(ge=0, le=256)
    y: int = Field(ge=0, le=256)


class UnitSnapshot(BaseModel):
    """Lightweight per-unit or per-building record sent by the BWAPI bridge."""

    id: int
    unit_type: str
    x: int = Field(ge=0, le=256)
    y: int = Field(ge=0, le=256)
    hp: int = Field(ge=0)
    shields: int = Field(default=0, ge=0)
    is_completed: bool = True
    is_under_attack: bool = False
    is_idle: bool = False


class BaseLocation(BaseModel):
    """Candidate base/expansion location on the map."""

    x: int = Field(ge=0, le=256)
    y: int = Field(ge=0, le=256)
    is_start_location: bool = False
    is_occupied_by_us: bool = False
    is_occupied_by_enemy: bool = False


class GameState(BaseModel):
    """Periodic game-state snapshot POSTed from the JBWAPI bridge."""

    frame: int = Field(default=0, ge=0)
    map_name: str = ""
    map_width: int = Field(default=128, ge=32, le=256)
    map_height: int = Field(default=128, ge=32, le=256)
    minerals: int = Field(default=50, ge=0)
    gas: int = Field(default=0, ge=0)
    supply_used: int = Field(default=4, ge=0)
    supply_total: int = Field(default=9, ge=0)
    mineral_workers: int = Field(default=0, ge=0)
    gas_workers: int = Field(default=0, ge=0)
    idle_workers: int = Field(default=0, ge=0)

    our_base: TilePoint | None = None
    natural_base: TilePoint | None = None
    choke_point: TilePoint | None = None
    enemy_base: TilePoint | None = None

    our_units: list[UnitSnapshot] = Field(default_factory=list)
    our_buildings: list[UnitSnapshot] = Field(default_factory=list)
    enemy_units: list[UnitSnapshot] = Field(default_factory=list)
    enemy_buildings: list[UnitSnapshot] = Field(default_factory=list)
    expansions: list[BaseLocation] = Field(default_factory=list)

    upgrades_completed: list[str] = Field(default_factory=list)
    upgrades_in_progress: list[str] = Field(default_factory=list)

    @property
    def game_seconds(self) -> float:
        """Approximate elapsed game seconds at BW Fastest (24 frames/s)."""
        return self.frame / 24.0

    def unit_counts(self) -> dict[str, int]:
        """Count our units by normalized lowercase type name."""
        return dict(Counter(u.unit_type.lower() for u in self.our_units))

    def building_counts(self, completed_only: bool = False) -> dict[str, int]:
        """Count our structures by normalized lowercase type name."""
        return dict(
            Counter(
                b.unit_type.lower()
                for b in self.our_buildings
                if not completed_only or b.is_completed
            )
        )

    def enemy_unit_counts(self) -> dict[str, int]:
        """Count currently visible enemy units by normalized type name."""
        return dict(Counter(u.unit_type.lower() for u in self.enemy_units))

    def enemy_building_counts(self) -> dict[str, int]:
        """Count known/visible enemy buildings by normalized type name."""
        return dict(Counter(b.unit_type.lower() for b in self.enemy_buildings))

    def combat_units(self) -> list[UnitSnapshot]:
        """Return our non-worker combat units."""
        return [u for u in self.our_units if u.unit_type.lower() != "probe"]

    def stray_combat_units(
        self,
        reference: TilePoint | None = None,
        max_dist_tiles: float = 20.0,
    ) -> list[UnitSnapshot]:
        """Return combat units farther than ``max_dist_tiles`` from ``reference``."""
        ref = reference or self.choke_point or self.natural_base or self.our_base
        if ref is None:
            return []
        return [
            u
            for u in self.combat_units()
            if math.hypot(u.x - ref.x, u.y - ref.y) > max_dist_tiles
        ]

    def compact_summary(self, army_target: TilePoint | None = None) -> str:
        """
        Render a dense text summary suitable for S1/S2 prompts.

        Args:
            army_target (TilePoint | None): Optional active army target for stray detection.

        Returns:
            str: Multi-line summary of time, economy, bases, army spread, and enemies.
        """
        mins = int(self.game_seconds // 60)
        secs = int(self.game_seconds % 60)
        units = self.unit_counts()
        bldgs = self.building_counts()
        enemies = self.enemy_unit_counts()
        e_bldgs = self.enemy_building_counts()
        under_attack = sum(
            1 for u in (*self.our_units, *self.our_buildings) if u.is_under_attack
        )
        lines = [
            f"Time: {mins:02d}:{secs:02d} (frame {self.frame}) | "
            f"Minerals: {self.minerals} | Gas: {self.gas} | "
            f"Supply: {self.supply_used}/{self.supply_total}",
            f"Our units: {units or 'none'} | Under attack: {under_attack}",
            f"Our buildings: {bldgs or 'none'}",
            f"Visible enemy units: {enemies or 'none'}",
            f"Visible enemy buildings: {e_bldgs or 'none'}",
        ]
        if self.mineral_workers or self.gas_workers or self.idle_workers:
            lines.append(
                f"Workers: mineral={self.mineral_workers}, "
                f"gas={self.gas_workers}, idle={self.idle_workers}"
            )
        base_parts: list[str] = []
        if self.our_base is not None:
            base_parts.append(f"our=({self.our_base.x},{self.our_base.y})")
        if self.natural_base is not None:
            base_parts.append(f"natural=({self.natural_base.x},{self.natural_base.y})")
        if self.choke_point is not None:
            base_parts.append(f"choke=({self.choke_point.x},{self.choke_point.y})")
        if self.enemy_base is not None:
            base_parts.append(f"enemy=({self.enemy_base.x},{self.enemy_base.y})")
        free_exps = [
            e
            for e in self.expansions
            if not e.is_occupied_by_us and not e.is_occupied_by_enemy
        ]
        if free_exps:
            if self.our_base is not None:
                ob = self.our_base
                free_exps.sort(key=lambda e: math.hypot(e.x - ob.x, e.y - ob.y))
            exp_str = ", ".join(f"({e.x},{e.y})" for e in free_exps[:4])
            base_parts.append(f"free_exp=[{exp_str}]")
        if base_parts:
            lines.append("Bases: " + " | ".join(base_parts))

        combat = self.combat_units()
        if combat:
            cx = round(sum(u.x for u in combat) / len(combat))
            cy = round(sum(u.y for u in combat) / len(combat))
            ref = army_target or self.choke_point or self.natural_base or self.our_base
            strays = self.stray_combat_units(reference=ref, max_dist_tiles=20.0)
            army_line = f"Army ({len(combat)} combat): centroid=({cx},{cy})"
            if strays:
                enemy_workers = [
                    e
                    for e in self.enemy_units
                    if e.unit_type.lower() in {"scv", "probe", "drone"}
                ]
                chasing_worker = any(
                    math.hypot(u.x - ew.x, u.y - ew.y) <= 12.0
                    for u in strays
                    for ew in enemy_workers
                )
                sample = ", ".join(f"{u.unit_type}@({u.x},{u.y})" for u in strays[:3])
                note = " [CHASING ENEMY WORKER]" if chasing_worker else ""
                army_line += f" | STRAY: {len(strays)} >20t from target ({sample}){note}"
            lines.append(army_line)

        if self.enemy_units:
            e_pos = ", ".join(
                f"{e.unit_type}@({e.x},{e.y})" for e in self.enemy_units[:5]
            )
            lines.append(f"Enemy positions: {e_pos}")

        if self.upgrades_completed or self.upgrades_in_progress:
            lines.append(
                f"Upgrades done: {self.upgrades_completed} | "
                f"in progress: {self.upgrades_in_progress}"
            )
        return "\n".join(lines)


class BuildStep(BaseModel):
    """One milestone in a Protoss opening build order."""

    supply: int = Field(ge=0, le=200)
    action: Literal["build", "train", "upgrade"]
    target: str
    count: int = Field(default=1, ge=1, le=12)
    note: str = ""


class UnitMix(BaseModel):
    """Desired Zealot vs. Dragoon composition and Probe cap."""

    zealot_ratio: float = Field(default=0.4, ge=0.0, le=1.0)
    dragoon_ratio: float = Field(default=0.6, ge=0.0, le=1.0)
    target_probes: int = Field(default=22, ge=4, le=60)


class S1Output(BaseModel):
    """
    Fast reactive tactical decision emitted by S1 (<256 tokens).

    Validated via prompt-injected JSON schema (``DiffusionGemmaClient.structured``).
    """

    posture: TacticalPosture = Field(
        description="Immediate army posture to execute right now"
    )
    army_target: TilePoint = Field(
        description="Tile coordinate (x, y) to defend, rally at, or attack-move to"
    )
    emergency_action: EmergencyAction = Field(
        default=EmergencyAction.NONE,
        description="Urgent override if under immediate threat or supply blocked",
    )
    threat_level: ThreatLevel = Field(
        default=ThreatLevel.NONE,
        description="Immediate threat severity near our bases or army",
    )
    escalate_to_s2: bool = Field(
        default=False,
        description="True only if strategic plan/build order must be re-evaluated now",
    )
    escalation_reason: str | None = Field(
        default=None,
        description="Short reason (<=12 words) when escalate_to_s2 is true",
    )


class S2Directive(BaseModel):
    """
    Slow deliberative strategic directive emitted by S2.

    Read by S1 on every reactive tick and merged into the S0 CommandBundle.
    """

    phase: GamePhase = Field(
        default=GamePhase.OPENING, description="Current strategic phase"
    )
    posture_hint: TacticalPosture = Field(
        default=TacticalPosture.HOLD_CHOKE,
        description="Default posture for S1 when no immediate emergency overrides it",
    )
    attack_target: TilePoint | None = Field(
        default=None, description="Strategic attack or pressure target tile, if any"
    )
    target_gateways: int = Field(
        default=2, ge=1, le=8, description="Desired total Gateway count"
    )
    target_probes: int = Field(
        default=22, ge=8, le=60, description="Desired Probe count"
    )
    zealot_ratio: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Target Zealot share of army"
    )
    dragoon_ratio: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Target Dragoon share of army"
    )
    want_singularity_charge: bool = Field(
        default=True,
        description="Whether to research Dragoon range at Cybernetics Core",
    )
    leash_army: bool = Field(
        default=True,
        description=(
            "True to recall/leash stray combat units back to army_target when not pushing "
            "(prevents chasing a lone SCV across the map); False to allow free pursuit/scouting"
        ),
    )
    want_expand: bool = Field(
        default=False, description="Whether to take a natural/third Nexus expansion"
    )
    expand_target: TilePoint | None = Field(
        default=None,
        description="Tile coordinate for next Nexus expansion if expanding",
    )
    notes: str = Field(
        default="", description="Concise strategic guidance for S1 (1-2 sentences)"
    )


class CommandBundle(BaseModel):
    """
    Latest merged command bundle returned immediately by host ``/tick`` to S0.

    S0 executes this continuously at 24 FPS without blocking on LLM calls.
    """

    revision: int = 0
    frame_issued: int = 0
    posture: TacticalPosture = TacticalPosture.HOLD_CHOKE
    army_target: TilePoint | None = None
    emergency_action: EmergencyAction = EmergencyAction.NONE
    target_probes: int = 22
    target_gateways: int = 2
    zealot_ratio: float = 0.4
    dragoon_ratio: float = 0.6
    want_singularity_charge: bool = True
    leash_army: bool = True
    queued_structures: list[ProtossStructure] = Field(default_factory=list)
    expand_target: TilePoint | None = None
    active_build_step: BuildStep | None = None
    s1_age_secs: float | None = None
    s2_age_secs: float | None = None
