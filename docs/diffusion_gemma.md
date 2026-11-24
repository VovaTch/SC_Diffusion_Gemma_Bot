# DiffusionGemma for the StarCraft bot: what works and how

Record of the approach that works for getting **image input**, **tool calls**,
**structured output** and **probability-scored decisions** out of DiffusionGemma
served locally with vLLM. Client code: `starcraft_diffusion_gemma/llm/client.py`.
End-to-end check: `starcraft_diffusion_gemma/llm/demo.py`.

## 1. Serving

Hardware: 2x RTX A5000 (24 GB, sm_86). Server project: `~/code/my/diffusion_gemma_server`
(uv venv, vLLM 0.33.0). Model: `RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic`
(26B MoE, 4B active, FP8; on sm_86 runs as Marlin weight-only FP8).

### Start the server

```bash
cd ~/code/my/diffusion_gemma_server && ./serve.sh --enforce-eager
```

Ready when `curl -s localhost:8000/v1/models` returns the model (~80 s from cold,
weights cached). Log: `~/code/my/diffusion_gemma_server/logs/vllm.log`.
Stop: Ctrl+C in that terminal (stragglers: `pkill -f VLLM::`).

`serve.sh` runs:

```bash
uv run vllm serve RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic \
    --tensor-parallel-size 2 \
    --disable-custom-all-reduce \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.78 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 1024 \
    --limit-mm-per-prompt '{"image": 4, "video": 0}' \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --host 127.0.0.1 --port 8000 \
    --enforce-eager
```

Env overrides: `MAX_LEN`, `GPU_UTIL`, `MAX_SEQS`, `MAX_BATCH_TOKENS`, `HOST`, `PORT`.
For the bot running in Docker, start with `HOST=0.0.0.0 ./serve.sh --enforce-eager`
(and reach it at the docker bridge IP, e.g. `172.17.0.1:8000`).

### Flags that matter (and why)

| Flag | Why |
|---|---|
| `--disable-custom-all-reduce` | **Required.** vLLM's custom all-reduce deadlocks on this 2-GPU box (both GPUs 100 %, endless "No available shared memory broadcast block", no error, forever). NCCL works. |
| `--max-num-batched-tokens 1024` | Sampler materialises logits for every canvas token x 262k vocab. 2496 caused an OOM at warmup. |
| `--gpu-memory-utilization 0.78` | Leaves room for that logits spike + Marlin repack. |
| `--enforce-eager` | Known-good. Compiled mode untested since the all-reduce fix; may be faster (first start compiles for minutes). |
| `--enable-auto-tool-choice --tool-call-parser gemma4` | Turns the model's native `<\|tool_call>call:name{...}<tool_call\|>` text into OpenAI `tool_calls`. Without it, any `tools` request with `auto`/named choice is HTTP 400. |
| `--reasoning-parser gemma4` | Strips the empty thinking header. Without it every reply starts with `thought\n` (the client strips it defensively anyway). |

Benign log noise: "CUDACachingAllocator ... OOM" warnings right after weight load
(Marlin repack retries).

### Speed (eager, measured)

* Single stream ~125-130 tok/s. Output arrives in 256-token canvas blocks
  (~2 s per block); short answers (<256 tok) come back in ~0.4-1.5 s total.
* Aggregate throughput plateaus ~130 tok/s at 4-16 concurrent requests -> batching
  buys nothing; keep bot requests sequential and short.
* Image prompt costs 280 tokens (fixed, 512x512 input tested).

## 2. Hard constraints of diffusion LMs in vLLM

vLLM rejects (HTTP 400) for diffusion models: `temperature`, `top_p`-style knobs
like `min_p`, `seed`, `min_tokens`, `logit_bias`, `bad_words`, `allowed_token_ids`,
and **all structured-output constraints** (`response_format` json_schema/json_object,
`structured_outputs`):

> "Structured outputs are not yet supported for diffusion language models. Remove
> the structured output constraint (e.g. `response_format`, `structured_outputs`)
> from the request."

OK to send: `max_tokens`, `ignore_eos`, `stream`, `tools`, `tool_choice`,
`chat_template_kwargs` (`{"enable_thinking": true|false}`).
No sampling knobs: outputs are near-deterministic but can still differ slightly
between runs of the same prompt. For diversity, vary the prompt.

## 3. Capabilities and the approach that works

All examples use the client:

```python
from starcraft_diffusion_gemma.llm import DiffusionGemmaClient, HttpTransport, image_part, user
client = DiffusionGemmaClient(HttpTransport("http://localhost:8000/v1/"))
```

### 3.1 Image input — works natively

Standard OpenAI content part with a base64 PNG data URL
(`image_part(PIL.Image | bytes | Path)`). Up to 4 images per prompt (server limit).

What made the map readable (see `render_minimap()` in `demo.py`, output in
`docs/minimap_demo.png`):

* Draw a **labelled coordinate grid** (lines every 16 tiles with numbers on the
  edges) and state the coordinate system in text ("x right, y down, tile units").
* Use **flat, distinct colours** and give a **legend in the prompt**
  (green = our base, red = enemy base, blue = free expansions, pink dots = enemy units).
* Result: free-text description correct; JSON reading had **every coordinate exactly
  right** (bases, 4 expansions), enemy army within ~6 tiles.

### 3.2 Structured output — prompt-injected JSON schema + validate + retry

Constrained decoding is unavailable (section 2), so `client.structured(messages, Model)`:

1. Prepends a system message: "Reply with exactly one JSON object and nothing else
   ... must validate against this JSON schema: <Model.model_json_schema()>".
2. Cuts the outermost `{...}` from the reply (tolerates fences / prose / `thought\n`).
3. `Model.model_validate_json`. On failure, appends the bad reply + the validation
   error as a user turn and retries (default 2 retries).

Measured: image + schema -> valid `MapReading` on first try, 0.94 s.
Tips: keep schemas flat-ish, use `Field(description=...)`, enums (`StrEnum`) and
bounds (`ge/le`) — they appear in the schema the model reads and pydantic enforces them.

Alternative: `client.structured_via_tool(messages, Model)` forces one tool call
whose parameters are the model's schema (needs the tool parser). Works (1.0 s) once
`$ref`s are inlined (see 3.3), but coordinates were less precise than prompted JSON
(expansions read as x=48 instead of 64) and values come back loosely typed
(e.g. `"64"` strings, pydantic coerces). **Prefer prompted JSON (`structured`) for
rich nested objects; use tools for actions with flat arguments.**

### 3.3 Tool calls — native Gemma syntax + vLLM `gemma4` parser

The model natively emits Gemma-4 tool syntax. Raw output observed before the parser
was enabled (special tokens stripped):

```
call:train_units{count:1,unit:overlord}call:train_units{count:6,unit:zergling}call:attack{x:120,y:40}
```

i.e. correct names, correct args, multiple calls in one turn. With
`--enable-auto-tool-choice --tool-call-parser gemma4` vLLM converts it to standard
`message.tool_calls` (`client.chat(..., tools=TOOLS).tool_calls -> list[ToolCall]`).
Define tools from pydantic models with `tool_spec(name, description, Model)`.

Notes:
* The gemma4 parser has `supports_required_and_named = False`: for
  `tool_choice="required"` / named it does **not** add guided JSON (which diffusion
  would reject anyway); it just lets the model emit native syntax. So a forced tool
  is a strong hint, not a guarantee -> always check `tool_calls` is non-empty.
* **Inline `$ref`s.** Pydantic puts nested models in `$defs` + `$ref`; the Gemma
  chat template renders parameters flat, so the model never sees the nested
  shape and invents one (observed: `"enemy_base": "(112,16)"`, lists instead of
  objects, missing fields). `tool_spec()` runs `inline_refs()` on the schema —
  after that nested tool arguments validated fine.
* Use enums (`StrEnum`) for unit/structure names — the model sticks to them.
* Always validate arguments with the pydantic model before executing
  (`TOOL_ARGS[call.name].model_validate(call.arguments)` in `demo.py`).
* **Multi-step / agent loop** — `client.run_tools(messages, tools, execute)`:
  send back `assistant_tool_calls(calls)` (assistant message with `tool_calls`,
  `content: None`) plus one `{"role": "tool", "tool_call_id": id, "content": result}`
  per call, ask again, repeat until the reply has no tool calls. The model **does
  issue further calls after seeing results** (observed: `train` + `attack_move`,
  then another `train`, then a text summary), so don't assume the second turn is
  text — loop with a `max_steps` cap.

Measured (with parser): single turn image + 3 tools -> 2 correct calls
(`train{unit: zergling, count: 10}`, `attack_move{x: 48, y: 64}` = enemy army
position) in 0.59 s; full agent loop (3 calls + final text) 2.2 s.

### 3.4 Decisions — `/v1/decisions` (vLLM extension, diffusion-aware)

Best tool for discrete high-level choices. Model reads the context and the
server returns **probabilities** of option labels (A/B/C, yes/no, 0-9) — one
forward read, no free text, no parsing. Question types:

* `choice`: pick one of up to 26 values (with descriptions) -> `choice`,
  `probabilities`, `confidence`.
* `predicate`: yes/no -> `probability` of yes.
* `score`: ordered levels (2-10) -> expected `score` + distribution.

`joint: true` (diffusion only) reads all questions in one pass: 3 questions in
**0.2-0.4 s** (vs ~12 s cold non-joint). Limitation: **text-only input** (images
rejected). Pattern: image -> `structured()` map reading -> serialize into text
context -> `decide()`.

```python
from starcraft_diffusion_gemma.llm import Question
client.decide("Zerg vs Terran, minute 5. 8 marines approaching our natural.", [
    Question("choice", "plan", "Best response?",
             {"drone_hard": "greedy economy", "sunken_defense": "build sunkens"}),
    Question("predicate", "under_attack", "Is an attack imminent?"),
    Question("score", "threat", "Threat level", {"low": "", "mid": "", "high": ""}),
])
```

## 4. Suggested bot architecture (high-level planning)

```
JBWAPI bridge (Docker, every N frames)
  -> POST game state + rendered minimap to FastAPI policy server (host)
       1. render minimap (grid + legend colours) from unit/base positions
       2. client.structured(image + state text, SituationReport)   ~1 s
       3. client.decide(report text, strategic questions)          ~0.3 s
       4. client.chat(image + report + decision, tools=MACRO_TOOLS) ~1 s
       -> list of tool calls = macro commands (train/build/attack_move/expand)
  <- bridge executes commands with low-level scripted micro
```

* Call the LLM every few seconds of game time, not every frame (~1-2.5 s per plan).
* Keep outputs < 256 tokens (one canvas block) for lowest latency.
* Validate every tool call argument with pydantic before executing.
* Server down / bad output -> fall back to a scripted default policy.

## 5. Verification log

Run: `uv run python -m starcraft_diffusion_gemma.llm.demo` (from this repo root).

Without tool parser (2026-11-13):

| Probe | Result | Time |
|---|---|---|
| image -> text | PASS | 1.44 s |
| image -> structured (prompted JSON) | PASS, all coords exact | 0.94 s |
| image -> structured (forced tool) | 400: needs `--tool-call-parser` | - |
| image -> tool calls (auto) | 400: needs `--enable-auto-tool-choice` | - |
| decisions (joint, 3 questions) | PASS, `sunken_defense` 0.996 | 0.20 s |
| `response_format` json_schema | 400: unsupported for diffusion LMs | - |
| raw tools in prompt (`tool_choice: none`) | model emitted correct native calls | 0.38 s |

With `serve.sh` as in section 1 (tool + reasoning parser), 2026-11-14 — **all PASS**:

| Probe | Result | Time |
|---|---|---|
| image -> text | PASS, no `thought\n` leak | 1.26 s |
| image -> structured (prompted JSON) | PASS, bases exact, 3/4 expansions | 1.09 s |
| image -> structured (forced tool, before `inline_refs`) | FAIL, invented nested shapes | 3.38 s |
| image -> structured (forced tool, with `inline_refs`) | PASS, bases exact, expansions off by 16 | 1.00 s |
| image -> tool calls (auto) | PASS, `train` + `attack_move` at enemy army | 0.59 s |
| agent loop (calls -> results -> answer) | PASS, 3 calls then text summary | 2.21 s |
| decisions (joint, 3 questions) | PASS, `sunken_defense` 0.997, attack 0.98, threat high | 0.24 s |

Known quirks: the model's free-text coordinates drift (e.g. enemy base "(96, 16)"
in prose while JSON says (112, 16)) — always take coordinates from structured
output, never from prose. Results vary slightly between runs even with identical
prompts (prefix-cache / batching effects).
