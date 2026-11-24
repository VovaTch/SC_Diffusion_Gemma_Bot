"""
Initial Protoss Zealot + Dragoon build order and progress tracker.

Provides a clean, deterministic opening (1-Gate Core into 2/3-Gate Zealot +
Dragoon with Singularity Charge) that S0 can follow immediately from frame 0,
and that S2 can inspect or adjust via tools as the game develops.
"""

from __future__ import annotations

from starcraft_diffusion_gemma.state.models import (
    BuildStep,
    GameState,
    ProtossStructure,
    ProtossUnit,
    ProtossUpgrade,
)

DEFAULT_ZEALOT_DRAGOON_BUILD: tuple[BuildStep, ...] = (
    BuildStep(
        supply=8,
        action="build",
        target=ProtossStructure.PYLON,
        count=1,
        note="First pylon near nexus/choke",
    ),
    BuildStep(
        supply=10,
        action="build",
        target=ProtossStructure.GATEWAY,
        count=1,
        note="First gateway",
    ),
    BuildStep(
        supply=12,
        action="build",
        target=ProtossStructure.ASSIMILATOR,
        count=1,
        note="Gas for cybernetics core and dragoons",
    ),
    BuildStep(
        supply=13,
        action="train",
        target=ProtossUnit.ZEALOT,
        count=1,
        note="Initial zealot for early defense/scouting",
    ),
    BuildStep(
        supply=14,
        action="build",
        target=ProtossStructure.CYBERNETICS_CORE,
        count=1,
        note="Unlock dragoons and singularity charge",
    ),
    BuildStep(
        supply=16,
        action="build",
        target=ProtossStructure.PYLON,
        count=2,
        note="Second pylon before supply block",
    ),
    BuildStep(
        supply=18,
        action="build",
        target=ProtossStructure.GATEWAY,
        count=2,
        note="Second gateway for zealot/dragoon production",
    ),
    BuildStep(
        supply=19,
        action="upgrade",
        target=ProtossUpgrade.SINGULARITY_CHARGE,
        count=1,
        note="Dragoon range upgrade",
    ),
    BuildStep(
        supply=20,
        action="train",
        target=ProtossUnit.DRAGOON,
        count=1,
        note="First dragoon",
    ),
    BuildStep(
        supply=22,
        action="build",
        target=ProtossStructure.PYLON,
        count=3,
        note="Third pylon for sustained gateway production",
    ),
    BuildStep(
        supply=28,
        action="build",
        target=ProtossStructure.GATEWAY,
        count=3,
        note="Third gateway for mid-game timing push",
    ),
)


def default_zealot_dragoon_build() -> list[BuildStep]:
    """Return a mutable copy of the default Zealot + Dragoon build order."""
    return [step.model_copy() for step in DEFAULT_ZEALOT_DRAGOON_BUILD]


def _step_satisfied(step: BuildStep, state: GameState) -> bool:
    """Check whether ``state`` already satisfies ``step``."""
    target = step.target.lower()
    if step.action == "build":
        return state.building_counts().get(target, 0) >= step.count
    if step.action == "train":
        return state.unit_counts().get(target, 0) >= step.count
    if step.action == "upgrade":
        done = {u.lower() for u in state.upgrades_completed}
        prog = {u.lower() for u in state.upgrades_in_progress}
        return target in done or target in prog
    return False


class BuildOrderTracker:
    """Tracks progress through an ordered list of Protoss opening steps."""

    def __init__(self, steps: list[BuildStep] | None = None) -> None:
        self._steps: list[BuildStep] = (
            list(steps) if steps is not None else default_zealot_dragoon_build()
        )
        self._completed_indices: set[int] = set()

    @property
    def steps(self) -> list[BuildStep]:
        """Current build order steps."""
        return list(self._steps)

    def replace(self, steps: list[BuildStep]) -> None:
        """Replace remaining build order steps (invoked by S2)."""
        self._steps = list(steps)
        self._completed_indices.clear()

    def next_step(self, state: GameState) -> BuildStep | None:
        """
        Advance completed steps against ``state`` and return the next pending step.

        Args:
            state (GameState): Latest BWAPI game state.

        Returns:
            BuildStep | None: Next uncompleted step, or None if the opening is done.
        """
        for idx, step in enumerate(self._steps):
            if idx in self._completed_indices:
                continue
            if _step_satisfied(step, state):
                self._completed_indices.add(idx)
                continue
            return step
        return None

    def status_text(self, state: GameState) -> str:
        """
        Format a concise build-order progress string for S1/S2 prompts.

        Args:
            state (GameState): Latest BWAPI game state.

        Returns:
            str: Compact progress summary.
        """
        nxt = self.next_step(state)
        done = len(self._completed_indices)
        total = len(self._steps)
        if nxt is None:
            return f"Build order complete ({done}/{total}); free macro mode."
        ready = (
            "READY" if state.supply_used >= nxt.supply else f"at {nxt.supply} supply"
        )
        return (
            f"Build order {done}/{total} done | Next: "
            f"{nxt.action} {nxt.target} (x{nxt.count}, {ready})"
        )
