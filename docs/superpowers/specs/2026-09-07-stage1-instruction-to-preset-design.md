# Stage 1: instruction-to-preset — design

## Context

Stage 1 of the roadmap (`CLAUDE.md`, "A four-stage roadmap"): an LLM parses
a natural-language instruction ("make a squelchy, resonant acid bass...")
directly into a structured CV recipe — a value for each of the instrument's
CV channels — with no training involved. This is deliberately the fastest,
lowest-risk stage, and it doubles as a qualitative "does this sound right"
check for everything built in Stages 2-3.

**Provider-neutral by explicit instruction.** The initial design assumed
the Anthropic API directly. Corrected: build this LLM-agnostic — an
abstraction with (at minimum) an Anthropic-backed implementation and a
local-model implementation, since a local model is a real near-term
possibility, not a hypothetical. This mirrors a pattern this project
already committed to for exactly this reason: `CVBackend` abstracts VCV
Rack away so a future Expert Sleepers backend is a new implementation, not
a rewrite. The same shape applies here — the *rest* of the pipeline
(recipe application, verification) must not care which LLM produced the
recipe.

**A real, current gap, not addressed by this stage**: this machine has no
Anthropic API credentials configured (no `ANTHROPIC_API_KEY`, no `ant`
CLI). The Anthropic-backed implementation is written and unit-tested
against mocked responses regardless — this stage does not block on
credentials existing, but the live "say a thing, hear a thing"
verification does. `openai` and `pydantic` are already installed; only
`anthropic` needs adding.

Out of scope for this stage: any actual training (Stages 2-3's job),
touching the sequencer/LFO patch or feature-aggregation code (Stage 0's
already-shipped, unmodified foundation), automated pass/fail judgment of
whether the result "sounds like" the instruction (not rigorously checkable
from 6 aggregated numbers — this stage reports real measurements for a
human to judge, it doesn't grade itself).

## Architecture

```
 instruction (text) ──▶ InstructionParser.parse_recipe(instruction, channels)
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
   AnthropicInstructionParser         LocalInstructionParser
   (anthropic SDK,                   (openai SDK pointed at
    messages.parse + a               any OpenAI-compatible
    dynamically-built                 endpoint -- vLLM, Ollama,
    Pydantic schema)                  llama.cpp server, etc.)
              │                               │
              └───────────────┬───────────────┘
                               ▼
                    dict[channel_name, float in [0,1]]
                               │
                               ▼
                  apply via backend.set_cv() per channel
                               │
                               ▼
              capture real audio, extract_features_aggregated
                               │
                               ▼
        report instruction + recipe + measured profile (no auto-grading)
```

### `InstructionParser` abstract interface

```python
class InstructionParser(ABC):
    def parse_recipe(self, instruction: str, channels: dict[str, str]) -> dict[str, float]:
        """channels maps channel name -> human-readable description (from
        the patch config). Returns a value in [0, 1] for every key in
        `channels`, no more, no fewer."""
```

Both implementations build their schema/prompt from the same
`channels: dict[str, str]` at call time — nothing about a specific patch's
channel set or count is hardcoded in either implementation. This is the
same discipline as `CVBackend`: patch-specific facts live in config, never
in the parser code.

### Config extension: per-channel descriptions

`configs/*.yaml` gains an optional `description` field per channel, used to
build the LLM system prompt. Example addition to
`configs/sequencer_test.yaml` (existing `cc` fields unchanged):

```yaml
channels:
  vco_freq:
    cc: 1
    description: "Base pitch of the bass voice. Low = deep bass, high = higher register."
  vcf_cutoff:
    cc: 2
    description: "Filter brightness. Low = dark/muffled, high = bright/harsh."
  vca_level:
    cc: 3
    description: "Overall loudness."
  seq_tempo:
    cc: 4
    description: "Speed of the looping 8-step sequence. Low = slow, high = fast."
  sweep_rate:
    cc: 5
    description: "Speed of the continuous filter sweep (separate from the sequence tempo)."
  sweep_depth:
    cc: 6
    description: "How much the continuous sweep moves the filter. Low = subtle, high = dramatic."
  filter_env_amount:
    cc: 7
    description: "Intensity of the per-note filter pluck/squelch triggered by each sequencer step. This is the main 'squelchy acid' control."
  vcf_resonance:
    cc: 8
    description: "Filter resonance/sharpness. Low = smooth, high = sharp, whistling, squelchy."
```

`load_channel_config` (in `backends/vcv_rack.py`) already parses this YAML
and currently returns a 2-tuple, `(midi_port_name, channel_cc)`. It grows a
third return value, `channel_descriptions: dict[str, str]` (empty string
for any channel with no `description` key, so a config without any
descriptions — `configs/bridge_test.yaml`, `configs/minimoog_test.yaml` —
still loads without error). This is a real, if small, signature change:
`VCVRackBackend.__init__`'s one existing call site
(`midi_port_name, self._channel_cc = load_channel_config(config_path)`)
needs a one-line update to unpack three values instead of two. Descriptions
are per-patch, hand-written once when each config is authored (matching how
CC numbers themselves are hand-written) — not generated or inferred.

### `AnthropicInstructionParser`

Uses `anthropic.Anthropic()` (zero-arg — resolves credentials from the
environment per the SDK's own resolution order, never a hardcoded key) and
`client.messages.parse(model=..., output_format=<dynamic Pydantic model>)`.
The Pydantic model is built per-call via `pydantic.create_model`, one
`float` field per channel with `Field(ge=0.0, le=1.0)`, so the schema is
never stale relative to whatever `channels` dict is passed in. Default
model: `claude-opus-5` (this project's default choice unless told
otherwise), overridable via constructor argument. System prompt: a fixed
preamble (what this instrument is, how to interpret the instruction)
followed by the channel descriptions, built fresh per call from the passed
`channels` dict — not cached across different channel sets.

### `LocalInstructionParser`

Uses `openai.OpenAI(base_url=..., api_key=...)` — `base_url` and `model`
are required constructor arguments (no default local server assumed;
`api_key` defaults to a placeholder string like `"not-needed"` since most
local servers don't check it) — and
`client.chat.completions.create(response_format={"type": "json_schema",
"json_schema": {...}})` with the same per-call schema shape as the
Anthropic path (translated to plain JSON Schema rather than a Pydantic
class, since the `openai` SDK's structured-output helper is
Pydantic-native only in newer versions — pass the raw schema for
portability across server implementations with uneven client-library
support).

**Local servers vary in how strictly they enforce a JSON schema** (vLLM's
structured-output support differs from Ollama's, which differs from a bare
llama.cpp server) — this implementation must not simply trust that the
returned text is valid, schema-conforming JSON. After getting the
response text, parse it with `json.loads` and validate it against the same
dynamically-built Pydantic model used by the Anthropic path (shared
validation logic, not duplicated) — a genuine parse or validation failure
raises a clear, specific exception (e.g. `RecipeParseError`) naming what
was wrong, rather than silently clamping or guessing. This is a real,
expected failure mode for a local model in a way it mostly isn't for
Anthropic's native structured outputs, and the caller needs to be able to
tell the difference between "got a recipe" and "the local model produced
garbage."

### Recipe application + verification script

`instruction_to_preset.py`, CLI: an instruction string (positional arg)
plus flags choosing the parser (`--llm anthropic|local`, `--model`,
`--base-url` for local) and the backend config (`--config
configs/sequencer_test.yaml`, defaulting to the current Stage 0 patch).
Steps: build the chosen `InstructionParser`, call `parse_recipe` with
`backend.channels()` paired against the config's descriptions, apply each
returned value via `backend.set_cv(channel, value)`, sleep a settle time,
capture a real aggregation window, and print the instruction, the full
recipe, and the measured `[mean, std] x [loudness, brightness, pitch]`
profile. No pass/fail judgment — this is a report for a human to read
against what they asked for, per this stage's explicit "no automated
grading" scope boundary.

**A real, small refactor this stage should make, not defer a third time**:
the "read N blocks spanning `aggregate_window_s` seconds, call
`extract_features_aggregated`" logic currently exists as near-identical
inline code in both `collect_sweep_dataset` and `run_control_loop` — the
whole-branch review flagged this duplication as a Minor, explicitly-not-
worth-fixing-yet finding when there were only two copies. This script
would be a third copy, which tips YAGNI's own calculus: extract a shared
helper (e.g. `read_aggregated_window(backend, sample_rate,
aggregate_window_s) -> np.ndarray`, a natural home is alongside
`extract_features_aggregated` in `features.py`, though it needs
`backend.read_audio_block()` so it isn't a pure feature-extraction
function — decide the exact module placement at plan time) and have all
three call sites use it. This is a mechanical, low-risk extraction (the
logic doesn't change, only where it lives), not a design change to
`collect_sweep_dataset`/`run_control_loop`'s own behavior or signatures.

## Global constraints

- No new `ttnn`/hardware dependency — this stage doesn't touch Tenstorrent
  hardware at all (no `gozer` lease needed for anything in this stage).
- CV values stay `[0, 1]` at every interface boundary, same as every prior
  stage.
- Channel identity, CC numbers, *and now descriptions* live in per-patch
  YAML, never hardcoded in Python.
- Neither `InstructionParser` implementation hardcodes a channel list or
  count — both build their schema/prompt from whatever `channels` dict is
  passed at call time.
- Follow the `claude-api` skill's guidance for any Anthropic-specific code
  (current model IDs, `client.messages.parse`, zero-arg credential
  resolution, no hardcoded keys) — do not mix Anthropic SDK calls and
  OpenAI SDK calls in the same file; keep the two parser implementations
  in separate modules, unified only through the shared abstract interface
  and the shared Pydantic-validation helper.

## Verification plan

1. **Both parsers have real unit tests against mocked responses** (patch
   the Anthropic/OpenAI client construction, not internal parser
   methods): a well-formed mocked response round-trips to the expected
   dict; an out-of-range value (e.g. `1.5`) is rejected by Pydantic
   validation, not silently clamped; `LocalInstructionParser` additionally
   gets a malformed-JSON-response test asserting `RecipeParseError`, since
   that's a real, expected local-model failure mode the Anthropic path is
   less exposed to.
2. **The dynamic schema-building is tested against varying channel
   counts** (e.g. a 3-channel and an 8-channel `channels` dict in the same
   test module) to confirm neither implementation has a hidden assumption
   about a fixed channel set.
3. **Config loading's new `description` field has a real test** —
   confirms `load_channel_config` returns descriptions correctly and that
   a channel with no `description` key doesn't break loading (backward
   compatible with `configs/bridge_test.yaml`/`configs/minimoog_test.yaml`,
   neither of which need this field added).
4. **Live verification is explicitly deferred, not skipped-and-forgotten**:
   document in `CLAUDE.md` that `instruction_to_preset.py` has not been run
   live against either a real Anthropic call or a real local model server,
   pending credentials/a running local server, and that this is a known,
   explicit gap — not silently left untested-looking-tested.

Full existing test suite (46 non-hardware + 4 hardware) must stay green;
this stage adds tests, doesn't touch anything hardware-marked.
