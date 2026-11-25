"""
System 2 (S2): slow, deliberative strategic planner (two-step turn).

Runs every ~30 seconds or on S1 escalation:
1. Tool-calling turn (``client.run_tools``) using flat Pydantic tool schemas
   to mutate macro state (structure queue, Zealot/Dragoon mix, upgrades,
   expansion target).
2. Structured directive turn (``client.structured``) producing a validated
   ``S2Directive`` consumed by S1 and merged into S0's ``CommandBundle``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from starcraft_diffusion_gemma.brain.traces import ToolCallTrace
from starcraft_diffusion_gemma.llm.client import (
    DiffusionGemmaClient,
    ToolCall,
    image_part,
    system,
    tool_spec,
    user,
)
from starcraft_diffusion_gemma.protoss.build_order import BuildOrderTracker
from starcraft_diffusion_gemma.state.models import (
    GameState,
    ProtossStructure,
    ProtossUpgrade,
    S1Output,
    S2Directive,
    TilePoint,
)

S2_TOOLS_SYSTEM_PROMPT = (
    "You are System 2 (strategic macro planner) for a StarCraft: Brood War "
    "Protoss Zealot + Dragoon bot. Inspect the minimap, economy, bases, build order, "
    "stray-unit status, and recent System 1 tactical reports. "
    "Strategic guidelines: "
    "(1) Expansion: YOU decide when to expand. When ready to take a natural or third "
    "Nexus (typically ~20-28 supply after Cybernetics Core or when floating minerals), "
    "call set_expand_target(enable=True, x, y) using the exact natural/free_exp tile "
    "coordinates from Bases, and raise target_probes (e.g. 36-44 on 2 bases) and "
    "target_gateways (4-6 on 2 bases). S0 will reserve 400 minerals and build the Nexus. "
    "(2) Do NOT call queue_structure for buildings already in the opening build order "
    "or already listed in Queued structures. "
    "(3) Attacking: Be proactive, not passive. If attacking BEFORE expanding (1 Nexus), "
    "push early with 4-6+ combat units. If you expanded FIRST (2+ Nexuses), defend until "
    "you have a stronger timing force (8-12+ combat units, ideally with Singularity Charge), "
    "then push the enemy base. "
    "(4) Stray/chasing units: Notice any STRAY units (e.g. chasing a lone enemy SCV across "
    "the map) and decide in your directive whether to leash/recall them (leash_army=True) "
    "or allow pursuit (leash_army=False). "
    "Call tools ONLY when changing macro settings, then briefly summarize."
)

S2_DIRECTIVE_SYSTEM_PROMPT = (
    "You are System 2 finalizing the strategic directive for System 1 and the "
    "S0 Protoss Zealot + Dragoon executor. Emit a single valid S2Directive JSON "
    "object reflecting the current game state and any macro tool actions just taken. "
    "Set phase='mid_push', posture_hint='push', and attack_target=<enemy_base coords> "
    "when ready to attack (1-base early push at 4-6+ combat units before expanding, or "
    "2-base timing push at 8-12+ combat units after expanding). "
    "Set leash_army=True to recall stray/chasing units back to the army when not pushing, "
    "or leash_army=False only if you intentionally want units to chase or scout."
)


class QueueStructureArgs(BaseModel):
    """Queue one or more Protoss structures for S0 to construct."""

    structure: ProtossStructure
    count: int = Field(default=1, ge=1, le=4)


class SetUnitMixArgs(BaseModel):
    """Update desired Zealot/Dragoon ratio, Probe cap, and Gateway target."""

    zealot_ratio: float = Field(ge=0.0, le=1.0)
    dragoon_ratio: float = Field(ge=0.0, le=1.0)
    target_probes: int = Field(ge=8, le=60)
    target_gateways: int = Field(ge=1, le=8)


class SetUpgradeArgs(BaseModel):
    """Enable or disable researching a Protoss upgrade."""

    upgrade: ProtossUpgrade
    enabled: bool = True


class SetExpandTargetArgs(BaseModel):
    """Enable or disable expanding to a specific tile coordinate."""

    enable: bool
    x: int = Field(ge=0, le=256)
    y: int = Field(ge=0, le=256)


class InspectBuildOrderArgs(BaseModel):
    """Inspect the remaining Protoss opening build-order steps."""


S2_TOOL_ARGS: dict[str, type[BaseModel]] = {
    "queue_structure": QueueStructureArgs,
    "set_unit_mix": SetUnitMixArgs,
    "set_upgrade": SetUpgradeArgs,
    "set_expand_target": SetExpandTargetArgs,
    "inspect_build_order": InspectBuildOrderArgs,
}

S2_TOOL_DOCS: dict[str, str] = {
    "queue_structure": "Queue a Protoss structure (pylon, gateway, assimilator, cybernetics_core, nexus) for S0 to build.",
    "set_unit_mix": "Set desired Zealot vs Dragoon production ratio, target Probes, and target Gateways.",
    "set_upgrade": "Enable or disable researching Singularity Charge (Dragoon range).",
    "set_expand_target": "Enable or disable taking a Nexus expansion at tile (x, y).",
    "inspect_build_order": "Read the remaining steps in the current Protoss opening build order.",
}

S2_TOOLS = [
    tool_spec(name, S2_TOOL_DOCS[name], model) for name, model in S2_TOOL_ARGS.items()
]


@dataclass
class S2ActionState:
    """Mutable macro settings updated by S2 tools and read into ``CommandBundle``."""

    build_order: BuildOrderTracker = field(default_factory=BuildOrderTracker)
    queued_structures: list[ProtossStructure] = field(default_factory=list)
    target_gateways: int = 2
    target_probes: int = 22
    zealot_ratio: float = 0.4
    dragoon_ratio: float = 0.6
    want_singularity_charge: bool = True
    expand_target: TilePoint | None = None

    def apply_tool(self, call: ToolCall, state: GameState) -> str:
        """
        Validate and apply a single S2 tool call, returning JSON result text.

        Args:
            call (ToolCall): Parsed tool call from DiffusionGemma.
            state (GameState): Current game state (used for map bounds/status).

        Returns:
            str: JSON-encoded execution result sent back as the tool message.
        """
        schema = S2_TOOL_ARGS.get(call.name)
        if schema is None:
            return json.dumps({"ok": False, "error": f"unknown tool: {call.name}"})
        try:
            args = schema.model_validate(call.arguments)
        except ValidationError as exc:
            return json.dumps({"ok": False, "error": str(exc)[:400]})

        if isinstance(args, QueueStructureArgs):
            self.queued_structures.extend([args.structure] * args.count)
            # Keep queue bounded so repeated calls never blow up S0.
            self.queued_structures = self.queued_structures[-8:]
            return json.dumps(
                {
                    "ok": True,
                    "queued": [s.value for s in self.queued_structures],
                }
            )

        if isinstance(args, SetUnitMixArgs):
            total = args.zealot_ratio + args.dragoon_ratio
            if total > 0:
                self.zealot_ratio = round(args.zealot_ratio / total, 2)
                self.dragoon_ratio = round(args.dragoon_ratio / total, 2)
            self.target_probes = args.target_probes
            self.target_gateways = args.target_gateways
            return json.dumps(
                {
                    "ok": True,
                    "zealot_ratio": self.zealot_ratio,
                    "dragoon_ratio": self.dragoon_ratio,
                    "target_probes": self.target_probes,
                    "target_gateways": self.target_gateways,
                }
            )

        if isinstance(args, SetUpgradeArgs):
            if args.upgrade == ProtossUpgrade.SINGULARITY_CHARGE:
                self.want_singularity_charge = args.enabled
            return json.dumps(
                {
                    "ok": True,
                    "upgrade": args.upgrade.value,
                    "enabled": args.enabled,
                }
            )

        if isinstance(args, SetExpandTargetArgs):
            if args.enable:
                self.expand_target = TilePoint(
                    x=min(max(args.x, 0), state.map_width),
                    y=min(max(args.y, 0), state.map_height),
                )
            else:
                self.expand_target = None
            return json.dumps(
                {
                    "ok": True,
                    "expand_target": (
                        self.expand_target.model_dump() if self.expand_target else None
                    ),
                }
            )

        if isinstance(args, InspectBuildOrderArgs):
            return json.dumps(
                {
                    "ok": True,
                    "status": self.build_order.status_text(state),
                    "steps": [s.model_dump() for s in self.build_order.steps],
                }
            )

        return json.dumps({"ok": False, "error": "unhandled tool"})


@dataclass
class S2RunCapture:
    """Mutable capture bag populated progressively during ``S2Agent.plan``."""

    step1_system_prompt: str = ""
    step1_user_prompt: str = ""
    step1_reasoning: str | None = None
    step1_summary: str = ""
    tool_traces: list[ToolCallTrace] = field(default_factory=list)
    step2_system_prompt: str = ""
    step2_user_prompt: str = ""


@dataclass(frozen=True)
class S2Result:
    """Output of one complete two-step S2 planning cycle."""

    directive: S2Directive
    tool_calls: list[ToolCall]
    tool_summary: str
    latency_secs: float
    tool_traces: list[ToolCallTrace] = field(default_factory=list)
    step1_reasoning: str | None = None


class S2Agent:
    """Two-step deliberative strategic planner (tools -> structured directive)."""

    def __init__(
        self,
        client: DiffusionGemmaClient,
        max_tool_steps: int = 3,
        tool_max_tokens: int = 384,
        directive_max_tokens: int = 512,
        retries: int = 1,
    ) -> None:
        self._client = client
        self._max_tool_steps = max_tool_steps
        self._tool_max_tokens = tool_max_tokens
        self._directive_max_tokens = directive_max_tokens
        self._retries = retries

    def plan(
        self,
        state: GameState,
        minimap: Image.Image,
        legend: str,
        action_state: S2ActionState,
        recent_s1: list[S1Output],
        current_directive: S2Directive,
        reason: str = "periodic",
        capture: S2RunCapture | None = None,
        army_target: TilePoint | None = None,
    ) -> S2Result:
        """
        Execute a two-step S2 strategic turn.

        Step 1 runs ``client.run_tools`` to inspect/mutate ``action_state``.
        Step 2 runs ``client.structured`` to emit a validated ``S2Directive``.

        Args:
            state (GameState): Latest BWAPI game state.
            minimap (Image.Image): Rendered 512x512 minimap.
            legend (str): Minimap colour/grid legend.
            action_state (S2ActionState): Mutable macro state modified by tools.
            recent_s1 (list[S1Output]): Recent S1 outputs for tactical context.
            current_directive (S2Directive): Previously active S2 directive.
            reason (str): Trigger reason (e.g. "periodic", "s1_escalation:...").
            capture (S2RunCapture | None): Optional bag to record prompts/tools even on error.

        Returns:
            S2Result: New directive, executed tool calls, summary, and latency.
        """
        start = time.perf_counter()
        cap = capture if capture is not None else S2RunCapture()
        img = image_part(minimap)
        s1_lines = (
            [
                f"- posture={s.posture}, target=({s.army_target.x},{s.army_target.y}), "
                f"threat={s.threat_level}, emergency={s.emergency_action}, "
                f"escalate={s.escalate_to_s2} ({s.escalation_reason or '-'})"
                for s in recent_s1[-3:]
            ]
            if recent_s1
            else ["- none yet"]
        )
        queued_str = (
            ", ".join(s.value for s in action_state.queued_structures)
            if action_state.queued_structures
            else "none"
        )
        expand_str = (
            f"({action_state.expand_target.x},{action_state.expand_target.y})"
            if action_state.expand_target is not None
            else "none"
        )
        context_text = "\n".join(
            [
                legend,
                "",
                f"Trigger: {reason}",
                state.compact_summary(army_target=army_target),
                action_state.build_order.status_text(state),
                f"Queued structures: {queued_str} | Pending expand_target: {expand_str}",
                f"Current directive: {current_directive.model_dump_json()}",
                "Recent S1 outputs:",
                *s1_lines,
            ]
        )

        # Step 1: Tool calling for macro mutations.
        step1_user = (
            context_text
            + "\n\nUse tools if macro adjustments are needed, then summarize."
        )
        cap.step1_system_prompt = S2_TOOLS_SYSTEM_PROMPT
        cap.step1_user_prompt = step1_user
        tool_msgs = [
            system(S2_TOOLS_SYSTEM_PROMPT),
            user(img, step1_user),
        ]
        tool_results_log: list[str] = []

        def execute(call: ToolCall) -> str:
            res = action_state.apply_tool(call, state)
            tool_results_log.append(f"{call.name}({call.arguments}) -> {res}")
            cap.tool_traces.append(
                ToolCallTrace(
                    id=call.id,
                    name=call.name,
                    arguments=dict(call.arguments),
                    result=res,
                )
            )
            return res

        final_turn, calls = self._client.run_tools(
            tool_msgs,
            S2_TOOLS,
            execute,
            max_steps=self._max_tool_steps,
            max_tokens=self._tool_max_tokens,
        )
        cap.step1_reasoning = final_turn.reasoning
        cap.step1_summary = final_turn.content

        # Step 2: Emit validated structured S2Directive.
        tools_recap = (
            "\n".join(tool_results_log) if tool_results_log else "No tool calls made."
        )
        step2_user = "\n".join(
            [
                context_text,
                "",
                "Macro tool calls executed in Step 1:",
                tools_recap,
                f"Step 1 summary: {final_turn.content or '(none)'}",
                "",
                "Emit the updated S2Directive JSON.",
            ]
        )
        cap.step2_system_prompt = S2_DIRECTIVE_SYSTEM_PROMPT
        cap.step2_user_prompt = step2_user
        directive_msgs = [
            system(S2_DIRECTIVE_SYSTEM_PROMPT),
            user(img, step2_user),
        ]
        directive = self._client.structured(
            directive_msgs,
            S2Directive,
            max_tokens=self._directive_max_tokens,
            retries=self._retries,
        )
        return S2Result(
            directive=directive,
            tool_calls=calls,
            tool_summary=final_turn.content,
            latency_secs=time.perf_counter() - start,
            tool_traces=list(cap.tool_traces),
            step1_reasoning=final_turn.reasoning,
        )
