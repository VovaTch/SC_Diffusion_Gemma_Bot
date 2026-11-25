"""
System 1 (S1): fast, reactive multimodal tactical loop.

Runs every ~2-3 seconds against the latest rendered minimap, compact state
summary, and active S2 strategic directive. Emits a compact ``S1Output``
(<256 tokens so DiffusionGemma completes in a single canvas block, ~0.9-1.1 s).
"""

from __future__ import annotations

from PIL import Image

from starcraft_diffusion_gemma.llm.client import (
    DiffusionGemmaClient,
    image_part,
    system,
    user,
)
from starcraft_diffusion_gemma.state.models import (
    GameState,
    S1Output,
    S2Directive,
    TilePoint,
)

S1_SYSTEM_PROMPT = (
    "You are System 1 (fast tactical controller) for a StarCraft: Brood War "
    "Protoss Zealot + Dragoon bot. Read the minimap image, numeric state, and "
    "System 2 strategic directive. Decide the immediate army posture, exact "
    "tile target (x, y), any urgent emergency action, and whether to escalate "
    "to System 2 right now. "
    "Tactical rules: "
    "(1) Follow S2's posture_hint and attack_target when S2 orders a push, "
    "unless our base is under immediate attack. "
    "(2) Do not chase or over-react to a single scouting worker (SCV/Probe/Drone); "
    "keep threat_level low/none and keep army_target on our choke/natural/rally. "
    "(3) If STRAY combat units are reported far from target (e.g. chasing a worker) "
    "while not pushing, escalate to S2 so S2 can decide whether to leash/recall them. "
    "(4) Only escalate when the strategic plan must change (unexpected rush, "
    "stray/chasing units, enemy expansion/tech found, or army wiped). "
    "Keep fields concise."
)


def _clamp_point(point: TilePoint, state: GameState) -> TilePoint:
    """Clamp predicted tile coordinates to the map bounds."""
    return TilePoint(
        x=min(max(point.x, 0), state.map_width),
        y=min(max(point.y, 0), state.map_height),
    )


class S1Agent:
    """Fast reactive multimodal controller backed by DiffusionGemma."""

    def __init__(
        self,
        client: DiffusionGemmaClient,
        max_tokens: int = 256,
        retries: int = 1,
    ) -> None:
        self._client = client
        self._max_tokens = max_tokens
        self._retries = retries

    def build_prompts(
        self,
        state: GameState,
        legend: str,
        directive: S2Directive,
        build_status: str,
        army_target: TilePoint | None = None,
    ) -> tuple[str, str]:
        """Build the ``(system_prompt, user_prompt)`` pair sent to DiffusionGemma."""
        directive_text = (
            f"S2 Directive: phase={directive.phase}, "
            f"posture_hint={directive.posture_hint}, "
            f"attack_target={directive.attack_target.model_dump() if directive.attack_target else None}, "
            f"leash_army={directive.leash_army}, "
            f"target_gateways={directive.target_gateways}, "
            f"mix={directive.zealot_ratio:.0%}Z/{directive.dragoon_ratio:.0%}D, "
            f"notes={directive.notes or 'follow opening'}"
        )
        prompt = "\n".join(
            [
                legend,
                "",
                state.compact_summary(army_target=army_target),
                build_status,
                directive_text,
                "",
                "Choose the immediate tactical posture and tile target.",
            ]
        )
        return S1_SYSTEM_PROMPT, prompt

    def step(
        self,
        state: GameState,
        minimap: Image.Image,
        legend: str,
        directive: S2Directive,
        build_status: str,
        army_target: TilePoint | None = None,
    ) -> S1Output:
        """
        Produce one validated ``S1Output`` from the current minimap and state.

        Args:
            state (GameState): Latest BWAPI game state.
            minimap (Image.Image): Rendered 512x512 grid-labelled minimap.
            legend (str): Text legend matching ``minimap``.
            directive (S2Directive): Active strategic directive from S2.
            build_status (str): One-line Protoss build-order progress string.
            army_target (TilePoint | None): Optional active army target for stray detection.

        Returns:
            S1Output: Validated reactive decision with clamped coordinates.
        """
        sys_prompt, prompt = self.build_prompts(
            state, legend, directive, build_status, army_target=army_target
        )
        messages = [
            system(sys_prompt),
            user(image_part(minimap), prompt),
        ]
        out = self._client.structured(
            messages,
            S1Output,
            max_tokens=self._max_tokens,
            retries=self._retries,
        )
        return out.model_copy(
            update={"army_target": _clamp_point(out.army_target, state)}
        )
