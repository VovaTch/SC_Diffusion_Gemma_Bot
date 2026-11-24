"""
Thin client for DiffusionGemma served by vLLM (OpenAI-compatible API).

Covers the four capabilities a planning bot needs:
image input, native tool calls, structured (Pydantic) outputs and
logprob-based decisions via vLLM's ``/v1/decisions`` endpoint.
See ``docs/diffusion_gemma.md`` for why each works the way it does.
"""

from __future__ import annotations

import base64
import io
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx
from PIL import Image
from pydantic import BaseModel, ValidationError

DEFAULT_MODEL = "RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic"
JsonDict = dict[str, Any]
Message = JsonDict


class DiffusionGemmaError(RuntimeError):
    """Server returned an error, or the model output could not be used."""


class Transport(Protocol):
    """Sends a JSON body to an API path and returns the JSON reply."""

    def post(self, path: str, body: JsonDict) -> JsonDict:
        """
        POST a JSON body.

        Args:
            path (str): Path relative to the API root, e.g. "chat/completions".
            body (JsonDict): Request body.

        Returns:
            JsonDict: Parsed JSON response.
        """
        ...


class HttpTransport:
    """httpx-backed ``Transport`` against a vLLM server."""

    def __init__(
        self, base_url: str = "http://localhost:8000/v1/", timeout: float = 120.0
    ) -> None:
        """
        Args:
            base_url (str): API root, must end with "/v1/".
            timeout (float): Per-request timeout in seconds.
        """
        self._http = httpx.Client(base_url=base_url, timeout=timeout)

    def post(self, path: str, body: JsonDict) -> JsonDict:
        """
        POST a JSON body, raising on HTTP errors.

        Args:
            path (str): Path relative to the API root.
            body (JsonDict): Request body.

        Returns:
            JsonDict: Parsed JSON response.
        """
        resp = self._http.post(path, json=body)
        if resp.status_code >= 400:
            raise DiffusionGemmaError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        return resp.json()


def text_part(text: str) -> JsonDict:
    """
    Build a text content part.

    Args:
        text (str): Text.

    Returns:
        JsonDict: OpenAI content part.
    """
    return {"type": "text", "text": text}


def image_part(image: Image.Image | bytes | Path) -> JsonDict:
    """
    Build an inline (base64 PNG data URL) image content part.

    Args:
        image (Image.Image | bytes | Path): PIL image, encoded bytes or file.

    Returns:
        JsonDict: OpenAI ``image_url`` content part.
    """
    if isinstance(image, Path):
        image = image.read_bytes()
    if isinstance(image, Image.Image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image = buf.getvalue()
    url = "data:image/png;base64," + base64.b64encode(image).decode()
    return {"type": "image_url", "image_url": {"url": url}}


def user(*parts: JsonDict | str) -> Message:
    """
    Build a user message from text and/or image parts.

    Args:
        *parts (JsonDict | str): Content parts; strings become text parts.

    Returns:
        Message: Chat message.
    """
    content = [text_part(p) if isinstance(p, str) else p for p in parts]
    return {"role": "user", "content": content}


def system(text: str) -> Message:
    """
    Build a system message.

    Args:
        text (str): System prompt.

    Returns:
        Message: Chat message.
    """
    return {"role": "system", "content": text}


def tool_spec(name: str, description: str, params: type[BaseModel]) -> JsonDict:
    """
    Build an OpenAI tool definition from a Pydantic model.

    Args:
        name (str): Function name.
        description (str): What the tool does (the model reads this).
        params (type[BaseModel]): Argument schema.

    Returns:
        JsonDict: Tool entry for the ``tools`` request field.
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": inline_refs(params.model_json_schema()),
        },
    }


def inline_refs(schema: JsonDict) -> JsonDict:
    """
    Replace ``$ref`` pointers with their ``$defs`` bodies.

    The Gemma chat template renders tool parameters flat; nested models
    behind ``$ref`` are invisible to the model, so it guesses their shape.

    Args:
        schema (JsonDict): Pydantic JSON schema (may contain "$defs").

    Returns:
        JsonDict: Equivalent schema without "$defs"/"$ref".
    """
    defs: JsonDict = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].rsplit("/", 1)[-1]])
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


@dataclass(frozen=True)
class ToolCall:
    """One parsed tool call."""

    id: str
    name: str
    arguments: JsonDict


def assistant_tool_calls(calls: list[ToolCall]) -> Message:
    """
    Build the assistant message that echoes tool calls back to the server.

    Args:
        calls (list[ToolCall]): Calls the model made.

    Returns:
        Message: Assistant message with ``tool_calls``.
    """
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
            }
            for c in calls
        ],
    }


@dataclass(frozen=True)
class ChatResult:
    """Parsed chat completion."""

    content: str
    reasoning: str | None
    tool_calls: list[ToolCall]
    finish_reason: str
    usage: JsonDict = field(default_factory=dict)


# Leaked empty-thinking header when the server runs without --reasoning-parser.
_THOUGHT_PREFIX = re.compile(r"^\s*thought\s*\n")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def extract_json(text: str) -> str:
    """
    Cut the outermost JSON object out of model text.

    Args:
        text (str): Raw model output (may have fences / prose).

    Returns:
        str: Substring from the first "{" to the last "}".
    """
    text = _FENCE.sub("", _THOUGHT_PREFIX.sub("", text.strip()))
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise DiffusionGemmaError(f"no JSON object in output: {text[:200]!r}")
    return text[start : end + 1]


class DiffusionGemmaClient:
    """
    High-level DiffusionGemma API.

    Example:
        client = DiffusionGemmaClient(HttpTransport())
        plan = client.structured([user(image_part(img), "Plan.")], Plan)
    """

    def __init__(self, transport: Transport, model: str = DEFAULT_MODEL) -> None:
        """
        Args:
            transport (Transport): Request sender (inject a fake for tests).
            model (str): Served model id.
        """
        self._transport = transport
        self._model = model

    def chat(
        self,
        messages: list[Message],
        tools: list[JsonDict] | None = None,
        tool_choice: str | JsonDict = "auto",
        max_tokens: int = 512,
        thinking: bool = False,
    ) -> ChatResult:
        """
        Run one chat completion.

        Diffusion models reject temperature/seed/min_p/logit_bias/
        response_format, so none are exposed here.

        Args:
            messages (list[Message]): Conversation.
            tools (list[JsonDict] | None): Tool definitions (see ``tool_spec``).
            tool_choice (str | JsonDict): "auto", "none", "required" or
                {"type": "function", "function": {"name": ...}}.
            max_tokens (int): Generation cap (rounded up to 256-token blocks).
            thinking (bool): Enable the Gemma thinking channel.

        Returns:
            ChatResult: Text, reasoning and parsed tool calls.
        """
        body: JsonDict = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        if tools:
            body |= {"tools": tools, "tool_choice": tool_choice}
        reply = self._transport.post("chat/completions", body)
        choice = reply["choices"][0]
        msg = choice["message"]
        calls = [
            ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=json.loads(tc["function"]["arguments"] or "{}"),
            )
            for tc in msg.get("tool_calls") or []
        ]
        return ChatResult(
            content=_THOUGHT_PREFIX.sub("", msg.get("content") or ""),
            reasoning=msg.get("reasoning") or msg.get("reasoning_content"),
            tool_calls=calls,
            finish_reason=choice["finish_reason"],
            usage=reply.get("usage", {}),
        )

    def run_tools(
        self,
        messages: list[Message],
        tools: list[JsonDict],
        execute: Callable[[ToolCall], str],
        max_steps: int = 4,
        max_tokens: int = 512,
    ) -> tuple[ChatResult, list[ToolCall]]:
        """
        Agent loop: call tools until the model answers in plain text.

        Args:
            messages (list[Message]): Conversation start.
            tools (list[JsonDict]): Tool definitions.
            execute (Callable[[ToolCall], str]): Runs one call, returns result text.
            max_steps (int): Max model turns.
            max_tokens (int): Generation cap per turn.

        Returns:
            tuple[ChatResult, list[ToolCall]]: Last model turn, all calls made.
        """
        convo = list(messages)
        made: list[ToolCall] = []
        result = self.chat(convo, tools=tools, max_tokens=max_tokens)
        for _ in range(max_steps - 1):
            if not result.tool_calls:
                break
            made += result.tool_calls
            convo += [assistant_tool_calls(result.tool_calls)]
            convo += [
                {"role": "tool", "tool_call_id": c.id, "content": execute(c)}
                for c in result.tool_calls
            ]
            result = self.chat(convo, tools=tools, max_tokens=max_tokens)
        return result, made

    def structured[M: BaseModel](
        self,
        messages: list[Message],
        schema: type[M],
        max_tokens: int = 512,
        retries: int = 2,
    ) -> M:
        """
        Get a validated Pydantic object via prompt-injected JSON schema.

        vLLM has no constrained decoding for diffusion LMs, so the schema is
        put in a system prompt and the reply is validated; on failure the
        validation error is fed back and the model retries.

        Args:
            messages (list[Message]): Conversation (images allowed).
            schema (type[M]): Target Pydantic model.
            max_tokens (int): Generation cap.
            retries (int): Extra attempts after a parse/validation failure.

        Returns:
            M: Validated instance of ``schema``.
        """
        instruction = system(
            "Reply with exactly one JSON object and nothing else: no prose, no "
            "markdown fences. It must validate against this JSON schema:\n"
            + json.dumps(schema.model_json_schema())
        )
        convo = [instruction, *messages]
        error = ""
        for _ in range(retries + 1):
            text = self.chat(convo, max_tokens=max_tokens).content
            try:
                return schema.model_validate_json(extract_json(text))
            except (ValidationError, DiffusionGemmaError) as exc:
                error = str(exc)
                convo = [
                    *convo,
                    {"role": "assistant", "content": text},
                    user(f"Invalid: {error[:800]}\nReply with corrected JSON only."),
                ]
        raise DiffusionGemmaError(f"structured output failed: {error[:500]}")

    def structured_via_tool[M: BaseModel](
        self,
        messages: list[Message],
        schema: type[M],
        max_tokens: int = 512,
    ) -> M:
        """
        Get a Pydantic object by forcing a single tool call whose arguments
        follow ``schema`` (native Gemma tool syntax, parsed server-side).

        Args:
            messages (list[Message]): Conversation (images allowed).
            schema (type[M]): Target Pydantic model.
            max_tokens (int): Generation cap.

        Returns:
            M: Validated instance of ``schema``.
        """
        name = "submit_" + schema.__name__.lower()
        doc = schema.__doc__ or f"Submit the {schema.__name__}."
        result = self.chat(
            messages,
            tools=[tool_spec(name, doc.strip(), schema)],
            tool_choice={"type": "function", "function": {"name": name}},
            max_tokens=max_tokens,
        )
        if not result.tool_calls:
            raise DiffusionGemmaError(f"no tool call; content={result.content!r}")
        return schema.model_validate(result.tool_calls[0].arguments)

    def decide(
        self,
        context: str,
        questions: list[Question],
        joint: bool = True,
    ) -> list[JsonDict]:
        """
        Answer multiple-choice / yes-no / score questions from label
        probabilities (``/v1/decisions``). Text-only input.

        Args:
            context (str): Situation description.
            questions (list[Question]): Questions to answer.
            joint (bool): Read all questions in one pass (faster; diffusion only).

        Returns:
            list[JsonDict]: One answer per question, with probabilities.
        """
        body = {
            "model": self._model,
            "input": context,
            "questions": [q.to_api() for q in questions],
            "joint": joint,
        }
        return self._transport.post("decisions", body)["answers"]


@dataclass(frozen=True)
class Question:
    """
    A ``/v1/decisions`` question.

    Args:
        kind (Literal["choice", "predicate", "score"]): Question type.
        name (str): Answer key.
        instructions (str): The question text.
        options (dict[str, str]): choice: value -> description; score: ordered
            level label -> description; predicate: unused.
    """

    kind: Literal["choice", "predicate", "score"]
    name: str
    instructions: str
    options: dict[str, str] = field(default_factory=dict)

    def to_api(self) -> JsonDict:
        """
        Serialize for the decisions endpoint.

        Returns:
            JsonDict: API question object.
        """
        q: JsonDict = {
            "type": self.kind,
            "name": self.name,
            "instructions": self.instructions,
        }
        if self.kind == "choice":
            q["choices"] = [
                {"value": v, "description": d} for v, d in self.options.items()
            ]
        elif self.kind == "score":
            q["levels"] = [
                {"label": v, "description": d} for v, d in self.options.items()
            ]
        return q
