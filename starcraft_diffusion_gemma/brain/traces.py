"""
Structured trace models for System 1 (fast decision) and System 2 (longer thinking).

Used by ``BotCoordinator`` for bounded in-memory history, optional ``--log-dir``
JSONL + PNG persistence, the ``/debug/traces`` endpoint, and the PySide6 dashboard.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from starcraft_diffusion_gemma.state.models import CommandBundle, S1Output, S2Directive


class ToolCallTrace(BaseModel):
    """Single S2 tool invocation with its parsed arguments and JSON result."""

    id: str = ""
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str = ""


class S1Trace(BaseModel):
    """Recorded System 1 (fast decision) inference turn."""

    id: int
    kind: Literal["s1"] = "s1"
    timestamp: float
    frame: int
    latency_secs: float
    image_id: str | None = None
    system_prompt: str
    user_prompt: str
    output: S1Output | None = None
    error: str | None = None


class S2Trace(BaseModel):
    """Recorded System 2 (longer thinking) two-step strategic turn."""

    id: int
    kind: Literal["s2"] = "s2"
    timestamp: float
    frame: int
    reason: str = "periodic"
    latency_secs: float
    image_id: str | None = None
    step1_system_prompt: str
    step1_user_prompt: str
    step1_reasoning: str | None = None
    step1_summary: str = ""
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    step2_system_prompt: str = ""
    step2_user_prompt: str = ""
    directive: S2Directive | None = None
    error: str | None = None


class TracesSnapshot(BaseModel):
    """Incremental snapshot returned by ``GET /debug/traces``."""

    revision: int
    has_state: bool
    frame: int | None = None
    command_bundle: CommandBundle | None = None
    latest_trace_id: int = 0
    s1_traces: list[S1Trace] = Field(default_factory=list)
    s2_traces: list[S2Trace] = Field(default_factory=list)
