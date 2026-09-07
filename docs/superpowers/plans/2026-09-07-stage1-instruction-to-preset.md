# Stage 1: Instruction-to-Preset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An LLM parses a natural-language instruction into a structured CV recipe for the current instrument, applies it, and reports what was actually measured — with the LLM provider swappable (Anthropic today, a local OpenAI-compatible model later) behind one shared interface.

**Architecture:** An `InstructionParser` abstract interface (mirroring `CVBackend`'s own pattern) with two implementations — `AnthropicInstructionParser` and `LocalInstructionParser` — unified only through the interface and a shared Pydantic-validation helper that builds a dynamic per-call schema from whatever channel set the current patch defines. A small CLI script applies the returned recipe and reports real measured audio; no automated pass/fail grading.

**Tech Stack:** Python 3.12, `anthropic` (new dependency, this plan), `openai` (already installed), `pydantic` (already installed), the existing `CVBackend`/`VCVRackBackend`/`features.py` stack.

**Spec:** `docs/superpowers/specs/2026-09-07-stage1-instruction-to-preset-design.md`

## Global Constraints

- No `ttnn`/hardware dependency anywhere in this plan — no `gozer` lease needed for any task.
- CV values stay `[0, 1]` at every interface boundary.
- Channel identity, CC numbers, and now descriptions live in per-patch YAML, never hardcoded in Python.
- Neither `InstructionParser` implementation may hardcode a channel list, count, or name — both build their schema/prompt from the `channels: dict[str, str]` passed to `parse_recipe` at call time.
- Anthropic-specific code follows the `claude-api` skill's current guidance: model ID `claude-opus-5` (this project's default, override only if asked), `client.messages.parse()` for structured output, zero-arg `anthropic.Anthropic()` client construction (resolves credentials from the environment — never a hardcoded key, never assume a key is unset just because `ANTHROPIC_API_KEY` isn't in `os.environ`; the SDK's own resolution order covers `ant auth login` profiles too).
- Never mix the Anthropic SDK and the OpenAI SDK in the same file — the two parser implementations live in separate modules.
- Live verification (a real Anthropic call, a real local model server) is out of scope for every task below — this machine has no Anthropic credentials configured and no local model server running. Every task's tests use mocked client responses. This is a known, explicit, documented gap (Task 6), not an oversight.

---

### Task 1: Extract the shared windowed-read helper

**Files:**
- Modify: `features.py`
- Modify: `data_collection.py`
- Modify: `control_loop.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `CVBackend.read_audio_block()`/`sample_rate()`/`block_size()` (existing).
- Produces: `features.read_aggregated_window(backend: CVBackend, sample_rate: int, aggregate_window_s: float) -> np.ndarray` — returns the 6-dim aggregated feature vector directly (reads the blocks internally, calls `extract_features_aggregated`). `collect_sweep_dataset` and `run_control_loop` both call this instead of inlining the block-count/read-loop themselves. Their own signatures (including the `aggregate_window_s` parameter each already has) are unchanged — this is a pure internal refactor, not a behavior change.

This closes a duplication the Stage 0 whole-branch review flagged as a Minor, explicitly-deferred finding when there were only 2 copies of this logic — Stage 3's `instruction_to_preset.py` (Task 6) would make it a 3rd copy, which is why this plan finally extracts it rather than deferring again.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features.py addition
def test_read_aggregated_window_reads_multiple_blocks():
    from backends.base import FakeCVBackend
    from features import read_aggregated_window

    backend = FakeCVBackend(channel_names=["a"], sample_rate=48000, block_size=1024)
    result = read_aggregated_window(backend, sample_rate=48000, aggregate_window_s=0.1)
    assert result.shape == (6,)
    # 0.1s * 48000 / 1024 = 4.6875 -> ceil to 5 blocks
    assert len(backend.set_cv_calls) == 0  # this helper only reads, never writes CV
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_features.py -k read_aggregated_window -v`
Expected: FAIL with `ImportError`/`AttributeError` (`read_aggregated_window` not defined).

- [ ] **Step 3: Implement `read_aggregated_window` in `features.py`**

```python
import math

def read_aggregated_window(backend, sample_rate: int, aggregate_window_s: float) -> np.ndarray:
    """Read a window of audio blocks spanning roughly aggregate_window_s
    seconds from `backend` and reduce it to the 6-dim aggregated feature
    vector via extract_features_aggregated. A single instantaneous block
    can't represent a looping/evolving pattern (a sequencer, an LFO sweep)
    -- this is the shared read used by both data_collection.py's sweep
    collection and control_loop.py's per-iteration read, so the two never
    drift out of sync on how a window is sized."""
    blocks_per_window = max(1, math.ceil(aggregate_window_s * sample_rate / backend.block_size()))
    blocks = [backend.read_audio_block() for _ in range(blocks_per_window)]
    return extract_features_aggregated(blocks, sample_rate)
```

(`backend`'s type is `CVBackend` — add the import and type hint; avoid a circular import by checking whether `features.py` can import from `backends.base` cleanly, since `backends/vcv_rack.py` already imports from `features.py`'s sibling modules elsewhere in this codebase — if a circular import surfaces, use a `TYPE_CHECKING`-guarded import for the type hint only, per standard Python practice, and leave the runtime code untyped-but-correct.)

- [ ] **Step 4: Update `data_collection.py`/`control_loop.py` to call it**

In `collect_sweep_dataset`, replace:
```python
    blocks_per_window = max(1, math.ceil(aggregate_window_s * sample_rate / backend.block_size()))
    ...
        blocks = [backend.read_audio_block() for _ in range(blocks_per_window)]
        feature_array[i] = extract_features_aggregated(blocks, sample_rate)
```
with:
```python
        feature_array[i] = read_aggregated_window(backend, sample_rate, aggregate_window_s)
```
(remove the now-unused `blocks_per_window` precompute and the `math`/`extract_features_aggregated` imports if nothing else in the file still needs them — check before removing). Same substitution in `run_control_loop`'s loop body. Both files import `read_aggregated_window` from `features`.

- [ ] **Step 5: Run tests to verify everything passes**

Run: `python3 -m pytest tests/test_features.py tests/test_data_collection.py tests/test_control_loop.py -v`
Expected: all PASS, including every pre-existing test in these three files unchanged in behavior.

- [ ] **Step 6: Commit**

```bash
git add features.py data_collection.py control_loop.py tests/test_features.py
git commit -m "Extract shared read_aggregated_window helper, dedupe two call sites"
```

---

### Task 2: Per-channel descriptions in config

**Files:**
- Modify: `backends/vcv_rack.py`
- Modify: `configs/sequencer_test.yaml`
- Test: `tests/test_vcv_rack_backend.py` (or wherever `load_channel_config` is currently tested — check for the existing test file name before creating a new one)

**Interfaces:**
- Consumes: nothing new.
- Produces: `load_channel_config(config_path: str) -> tuple[str, dict[str, int], dict[str, str]]` — third return value is `channel_descriptions`, mapping channel name to its `description` string (empty string `""` if the channel's YAML entry has no `description` key, so configs without any descriptions still load without error). `VCVRackBackend.__init__` is updated to unpack all three and expose the descriptions (add a `channel_descriptions() -> dict[str, str]` method to `VCVRackBackend`, returning what was loaded — not part of the abstract `CVBackend` interface, since not every future backend need have LLM-facing descriptions; callers that need them check `hasattr` or are written specifically against `VCVRackBackend`).

- [ ] **Step 1: Write the failing tests**

```python
def test_load_channel_config_returns_descriptions():
    from backends.vcv_rack import load_channel_config
    # Use an existing fixture config or write a small temp YAML inline via tmp_path
    midi_port, channel_cc, descriptions = load_channel_config("configs/sequencer_test.yaml")
    assert descriptions["vco_freq"] == "Base pitch of the bass voice. Low = deep bass, high = higher register."
    assert set(descriptions.keys()) == set(channel_cc.keys())

def test_load_channel_config_handles_missing_descriptions(tmp_path):
    from backends.vcv_rack import load_channel_config
    config_path = tmp_path / "no_descriptions.yaml"
    config_path.write_text(
        'midi_port_name: "test"\n'
        "channels:\n"
        "  foo:\n"
        "    cc: 1\n"
    )
    _, channel_cc, descriptions = load_channel_config(str(config_path))
    assert descriptions["foo"] == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest -k "load_channel_config" -v`
Expected: FAIL (current function returns a 2-tuple; the description text in the first test doesn't exist yet in `configs/sequencer_test.yaml` either).

- [ ] **Step 3: Implement**

```python
def load_channel_config(config_path: str) -> tuple[str, dict[str, int], dict[str, str]]:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    channel_cc = {name: entry["cc"] for name, entry in raw["channels"].items()}
    channel_descriptions = {
        name: entry.get("description", "") for name, entry in raw["channels"].items()
    }
    return raw["midi_port_name"], channel_cc, channel_descriptions
```

Update `VCVRackBackend.__init__`:
```python
        midi_port_name, self._channel_cc, self._channel_descriptions = load_channel_config(config_path)
```
Add:
```python
    def channel_descriptions(self) -> dict[str, str]:
        return self._channel_descriptions
```

Add descriptions to every channel in `configs/sequencer_test.yaml` (exact text — write descriptions that would actually help an LLM reason about instructions like "squelchy, resonant acid bass... looping sequence... sweeping subtly on the filter"):

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

Do NOT add `description` fields to `configs/bridge_test.yaml` or `configs/minimoog_test.yaml` — they predate this feature and this task's own test (Step 1) specifically covers the no-description case, so leaving them as-is is the correct backward-compatibility check, not an oversight.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest -v` (full non-hardware suite)
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add backends/vcv_rack.py configs/sequencer_test.yaml tests/
git commit -m "Add per-channel descriptions to config, for the LLM-facing instruction parser"
```

---

### Task 3: `InstructionParser` interface + shared validation

**Files:**
- Create: `instruction_parser/__init__.py` (empty)
- Create: `instruction_parser/base.py`
- Create: `instruction_parser/schema.py`
- Test: `tests/test_instruction_parser_schema.py`

**Interfaces:**
- Produces:
  - `instruction_parser.base.InstructionParser` — an `ABC` with one abstract method: `parse_recipe(self, instruction: str, channels: dict[str, str]) -> dict[str, float]`.
  - `instruction_parser.base.RecipeParseError(Exception)` — raised when a parser's underlying LLM response can't be turned into a valid recipe (malformed JSON, out-of-range values, missing/extra keys).
  - `instruction_parser.schema.build_recipe_model(channels: dict[str, str]) -> type[pydantic.BaseModel]` — given a channel-name-to-description dict, returns a dynamically-constructed Pydantic model with one `float` field per channel name, each constrained `ge=0.0, le=1.0`, `additionalProperties` effectively forbidden (Pydantic's default `extra="forbid"` config, set explicitly — don't rely on the default in case a project-wide Pydantic config changes it later).
  - `instruction_parser.schema.validate_recipe_json(raw_json: str, channels: dict[str, str]) -> dict[str, float]` — parses `raw_json` with `json.loads`, validates it against `build_recipe_model(channels)`, and returns a plain `dict[str, float]` (via the validated model's own field values) on success; raises `RecipeParseError` with a specific, actionable message (what was wrong: bad JSON, a value out of `[0,1]`, a missing/extra key — name which) on any failure. Both `AnthropicInstructionParser` (Task 4) and `LocalInstructionParser` (Task 5) call this for validation, so the exact same rules apply to both regardless of how reliably their underlying model conforms to the requested schema.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_instruction_parser_schema.py
import json
import pytest
from instruction_parser.base import RecipeParseError
from instruction_parser.schema import build_recipe_model, validate_recipe_json

CHANNELS = {"vco_freq": "pitch", "vca_level": "loudness"}

def test_build_recipe_model_has_one_field_per_channel():
    Model = build_recipe_model(CHANNELS)
    instance = Model(vco_freq=0.5, vca_level=0.9)
    assert instance.vco_freq == 0.5
    assert instance.vca_level == 0.9

def test_build_recipe_model_rejects_out_of_range():
    Model = build_recipe_model(CHANNELS)
    with pytest.raises(Exception):  # pydantic.ValidationError
        Model(vco_freq=1.5, vca_level=0.5)

def test_validate_recipe_json_round_trips():
    raw = json.dumps({"vco_freq": 0.2, "vca_level": 0.8})
    result = validate_recipe_json(raw, CHANNELS)
    assert result == {"vco_freq": 0.2, "vca_level": 0.8}

def test_validate_recipe_json_rejects_malformed_json():
    with pytest.raises(RecipeParseError):
        validate_recipe_json("not json at all {{{", CHANNELS)

def test_validate_recipe_json_rejects_out_of_range_value():
    raw = json.dumps({"vco_freq": 1.5, "vca_level": 0.8})
    with pytest.raises(RecipeParseError):
        validate_recipe_json(raw, CHANNELS)

def test_validate_recipe_json_rejects_missing_channel():
    raw = json.dumps({"vco_freq": 0.5})  # missing vca_level
    with pytest.raises(RecipeParseError):
        validate_recipe_json(raw, CHANNELS)

def test_build_recipe_model_works_for_different_channel_counts():
    # Confirms neither the schema builder nor the validator has a hidden
    # assumption about a fixed (e.g. 8-channel) set.
    three = {"a": "", "b": "", "c": ""}
    eight = {f"ch{i}": "" for i in range(8)}
    assert set(build_recipe_model(three).model_fields.keys()) == {"a", "b", "c"}
    assert len(build_recipe_model(eight).model_fields) == 8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_instruction_parser_schema.py -v`
Expected: FAIL (`instruction_parser` module doesn't exist yet).

- [ ] **Step 3: Implement**

`instruction_parser/base.py`:
```python
from abc import ABC, abstractmethod


class RecipeParseError(Exception):
    """Raised when an InstructionParser's underlying LLM response can't be
    turned into a valid CV recipe -- malformed JSON, a value outside
    [0, 1], or a missing/extra channel key. Carries a specific, actionable
    message naming what was wrong, not a generic parse failure."""


class InstructionParser(ABC):
    @abstractmethod
    def parse_recipe(self, instruction: str, channels: dict[str, str]) -> dict[str, float]:
        """Given a natural-language instruction and a mapping of channel
        name -> human-readable description (from the current patch's
        config), return a value in [0, 1] for every key in `channels`, no
        more, no fewer. Raises RecipeParseError if the underlying model's
        response can't be validated into exactly that shape."""
        ...
```

`instruction_parser/schema.py`:
```python
import json

import pydantic

from instruction_parser.base import RecipeParseError


def build_recipe_model(channels: dict[str, str]) -> type[pydantic.BaseModel]:
    fields = {
        name: (float, pydantic.Field(ge=0.0, le=1.0, description=description))
        for name, description in channels.items()
    }
    return pydantic.create_model(
        "CVRecipe", __config__=pydantic.ConfigDict(extra="forbid"), **fields
    )


def validate_recipe_json(raw_json: str, channels: dict[str, str]) -> dict[str, float]:
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise RecipeParseError(f"response was not valid JSON: {e}") from e

    model = build_recipe_model(channels)
    try:
        instance = model.model_validate(data)
    except pydantic.ValidationError as e:
        raise RecipeParseError(f"response did not match the expected recipe schema: {e}") from e

    return {name: getattr(instance, name) for name in channels}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_instruction_parser_schema.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add instruction_parser/ tests/test_instruction_parser_schema.py
git commit -m "Add InstructionParser interface and shared recipe schema/validation"
```

---

### Task 4: `AnthropicInstructionParser`

**Files:**
- Create: `instruction_parser/anthropic_parser.py`
- Test: `tests/test_anthropic_instruction_parser.py`

**Interfaces:**
- Consumes: `InstructionParser` (Task 3), `build_recipe_model`/`RecipeParseError` (Task 3).
- Produces: `AnthropicInstructionParser(model: str = "claude-opus-5")`, implementing `parse_recipe`.

Requires `pip install anthropic` (not yet installed on this machine).

- [ ] **Step 1: Install the `anthropic` package**

```bash
pip install anthropic
python3 -c "import anthropic; print(anthropic.__version__)"
```

Do this first — `instruction_parser/anthropic_parser.py` (Step 3) imports `anthropic` at module level, so the test file added in Step 2 below can't even be collected without it installed. Also add `anthropic` to `README.md`'s Requirements line alongside the existing prose dependency list, matching this project's existing convention of listing dependencies in prose rather than a manifest file.

- [ ] **Step 2: Write the failing tests (mocked client, no live API call)**

```python
# tests/test_anthropic_instruction_parser.py
from unittest.mock import MagicMock, patch

import pytest

from instruction_parser.anthropic_parser import AnthropicInstructionParser
from instruction_parser.base import RecipeParseError

CHANNELS = {"vco_freq": "pitch", "vca_level": "loudness"}


def _mock_parsed_response(**field_values):
    response = MagicMock()
    response.parsed_output = MagicMock(**field_values)
    # model_dump lets the parser extract a plain dict without depending on
    # the mock's own attribute set matching pydantic's real interface
    response.parsed_output.model_dump.return_value = field_values
    return response


def test_parse_recipe_returns_dict_from_parsed_output():
    parser = AnthropicInstructionParser()
    with patch("instruction_parser.anthropic_parser.anthropic.Anthropic") as MockClient:
        mock_client = MockClient.return_value
        mock_client.messages.parse.return_value = _mock_parsed_response(
            vco_freq=0.2, vca_level=0.8
        )
        result = parser.parse_recipe("make it deep and quiet", CHANNELS)
    assert result == {"vco_freq": 0.2, "vca_level": 0.8}


def test_parse_recipe_uses_configured_model():
    parser = AnthropicInstructionParser(model="claude-sonnet-5")
    with patch("instruction_parser.anthropic_parser.anthropic.Anthropic") as MockClient:
        mock_client = MockClient.return_value
        mock_client.messages.parse.return_value = _mock_parsed_response(
            vco_freq=0.5, vca_level=0.5
        )
        parser.parse_recipe("anything", CHANNELS)
    _, kwargs = mock_client.messages.parse.call_args
    assert kwargs["model"] == "claude-sonnet-5"


def test_parse_recipe_includes_channel_descriptions_in_system_prompt():
    parser = AnthropicInstructionParser()
    with patch("instruction_parser.anthropic_parser.anthropic.Anthropic") as MockClient:
        mock_client = MockClient.return_value
        mock_client.messages.parse.return_value = _mock_parsed_response(
            vco_freq=0.5, vca_level=0.5
        )
        parser.parse_recipe("anything", CHANNELS)
    _, kwargs = mock_client.messages.parse.call_args
    assert "pitch" in kwargs["system"]
    assert "loudness" in kwargs["system"]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_anthropic_instruction_parser.py -v`
Expected: FAIL (`instruction_parser.anthropic_parser` doesn't exist).

- [ ] **Step 4: Implement**

```python
# instruction_parser/anthropic_parser.py
import anthropic

from instruction_parser.base import InstructionParser, RecipeParseError
from instruction_parser.schema import build_recipe_model

SYSTEM_PREAMBLE = (
    "You control a modular synthesizer via a set of continuous CV "
    "(control voltage) channels, each in the range [0, 1]. Given an "
    "instruction describing a desired sound, choose a value for every "
    "channel listed below to best match the instruction. Channels and "
    "what each one controls:\n"
)


def _build_system_prompt(channels: dict[str, str]) -> str:
    lines = [SYSTEM_PREAMBLE]
    for name, description in channels.items():
        lines.append(f"- {name}: {description}")
    return "\n".join(lines)


class AnthropicInstructionParser(InstructionParser):
    def __init__(self, model: str = "claude-opus-5"):
        self._model = model

    def parse_recipe(self, instruction: str, channels: dict[str, str]) -> dict[str, float]:
        client = anthropic.Anthropic()
        recipe_model = build_recipe_model(channels)
        try:
            response = client.messages.parse(
                model=self._model,
                max_tokens=1024,
                system=_build_system_prompt(channels),
                messages=[{"role": "user", "content": instruction}],
                output_format=recipe_model,
            )
        except Exception as e:
            raise RecipeParseError(f"Anthropic API call failed: {e}") from e

        if response.parsed_output is None:
            # A refusal (stop_reason == "refusal") or any other non-normal
            # completion leaves parsed_output unset -- surface this as a
            # clear RecipeParseError instead of an opaque AttributeError
            # on the next line. Check response.stop_reason for the message.
            raise RecipeParseError(
                f"Anthropic response had no parsed output (stop_reason={response.stop_reason!r})"
            )

        parsed = response.parsed_output.model_dump()
        return {name: parsed[name] for name in channels}
```

(Check the `claude-api` skill's Python README/tool-use docs again while implementing this task — `client.messages.parse`'s exact keyword arguments and `response.parsed_output`'s exact shape are documented there; don't rely on this brief's transcription alone if something doesn't match. Also add a test alongside Task 4's Step 1 tests confirming a `None` `parsed_output` raises `RecipeParseError` rather than an `AttributeError` — mock `response.parsed_output = None` and `response.stop_reason = "refusal"`.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_anthropic_instruction_parser.py -v`
Expected: all PASS. No live API call is made anywhere in this test file — confirm by running with no `ANTHROPIC_API_KEY` set and no network access assumed.

- [ ] **Step 6: Commit**

```bash
git add instruction_parser/anthropic_parser.py tests/test_anthropic_instruction_parser.py README.md
git commit -m "Add AnthropicInstructionParser"
```

---

### Task 5: `LocalInstructionParser`

**Files:**
- Create: `instruction_parser/local_parser.py`
- Test: `tests/test_local_instruction_parser.py`

**Interfaces:**
- Consumes: `InstructionParser`, `validate_recipe_json`/`RecipeParseError` (Task 3).
- Produces: `LocalInstructionParser(base_url: str, model: str, api_key: str = "not-needed")`, implementing `parse_recipe`. Points the `openai` SDK's client at any OpenAI-compatible chat completions endpoint (vLLM, Ollama, llama.cpp server, LM Studio, ...) — `base_url` and `model` are required constructor arguments; no default server is assumed.

- [ ] **Step 1: Write the failing tests (mocked client, no live server)**

```python
# tests/test_local_instruction_parser.py
import json
from unittest.mock import MagicMock, patch

import pytest

from instruction_parser.base import RecipeParseError
from instruction_parser.local_parser import LocalInstructionParser

CHANNELS = {"vco_freq": "pitch", "vca_level": "loudness"}


def _mock_chat_response(content_str: str):
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content_str))]
    return response


def test_parse_recipe_returns_dict_from_valid_response():
    parser = LocalInstructionParser(base_url="http://localhost:8000/v1", model="local-model")
    with patch("instruction_parser.local_parser.openai.OpenAI") as MockClient:
        mock_client = MockClient.return_value
        mock_client.chat.completions.create.return_value = _mock_chat_response(
            json.dumps({"vco_freq": 0.3, "vca_level": 0.7})
        )
        result = parser.parse_recipe("make it deep and quiet", CHANNELS)
    assert result == {"vco_freq": 0.3, "vca_level": 0.7}


def test_parse_recipe_uses_configured_base_url_and_model():
    parser = LocalInstructionParser(base_url="http://myhost:1234/v1", model="my-local-model")
    with patch("instruction_parser.local_parser.openai.OpenAI") as MockClient:
        mock_client = MockClient.return_value
        mock_client.chat.completions.create.return_value = _mock_chat_response(
            json.dumps({"vco_freq": 0.5, "vca_level": 0.5})
        )
        parser.parse_recipe("anything", CHANNELS)
    _, init_kwargs = MockClient.call_args
    assert init_kwargs["base_url"] == "http://myhost:1234/v1"
    _, call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs["model"] == "my-local-model"


def test_parse_recipe_raises_on_malformed_local_model_output():
    # A real, expected failure mode for a local model that doesn't
    # strictly conform to the requested schema -- confirm it's caught and
    # re-raised as RecipeParseError, not left as a raw json/pydantic error.
    parser = LocalInstructionParser(base_url="http://localhost:8000/v1", model="local-model")
    with patch("instruction_parser.local_parser.openai.OpenAI") as MockClient:
        mock_client = MockClient.return_value
        mock_client.chat.completions.create.return_value = _mock_chat_response(
            "I think vco_freq should be low and vca_level should be high"  # not JSON at all
        )
        with pytest.raises(RecipeParseError):
            parser.parse_recipe("make it deep and quiet", CHANNELS)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_local_instruction_parser.py -v`
Expected: FAIL (`instruction_parser.local_parser` doesn't exist).

- [ ] **Step 3: Implement**

```python
# instruction_parser/local_parser.py
import openai

from instruction_parser.base import InstructionParser, RecipeParseError
from instruction_parser.anthropic_parser import _build_system_prompt
from instruction_parser.schema import validate_recipe_json


def _build_json_schema(channels: dict[str, str]) -> dict:
    return {
        "type": "object",
        "properties": {
            name: {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": description}
            for name, description in channels.items()
        },
        "required": list(channels.keys()),
        "additionalProperties": False,
    }


class LocalInstructionParser(InstructionParser):
    def __init__(self, base_url: str, model: str, api_key: str = "not-needed"):
        self._base_url = base_url
        self._model = model
        self._api_key = api_key

    def parse_recipe(self, instruction: str, channels: dict[str, str]) -> dict[str, float]:
        client = openai.OpenAI(base_url=self._base_url, api_key=self._api_key)
        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _build_system_prompt(channels)},
                    {"role": "user", "content": instruction},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "cv_recipe", "schema": _build_json_schema(channels)},
                },
            )
        except Exception as e:
            raise RecipeParseError(f"local model API call failed: {e}") from e

        content = response.choices[0].message.content
        # Local servers vary in how strictly they enforce the requested
        # schema (vLLM/Ollama/llama.cpp server all differ) -- never trust
        # the response is valid JSON just because it was requested;
        # validate_recipe_json is the same validation AnthropicInstructionParser
        # implicitly gets for free from Pydantic's own structured-output
        # enforcement, applied here explicitly since this path can't
        # assume that.
        return validate_recipe_json(content, channels)
```

(`_build_system_prompt` is reused from `anthropic_parser.py` — this is a deliberate shared helper, not a violation of "don't mix SDKs in one file": the function itself contains no Anthropic-specific code, it's plain string-building. If this feels awkward at implementation time, moving it to `instruction_parser/schema.py` or a new small `instruction_parser/prompts.py` instead of importing across the two provider modules is an equally valid choice — controller's call if raised as a question, otherwise use your own judgment and note which you picked in your report.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_local_instruction_parser.py -v`
Expected: all PASS. No live server is contacted anywhere in this test file.

- [ ] **Step 5: Commit**

```bash
git add instruction_parser/local_parser.py tests/test_local_instruction_parser.py
git commit -m "Add LocalInstructionParser (OpenAI-compatible: vLLM/Ollama/etc.)"
```

---

### Task 6: `instruction_to_preset.py` CLI + document the live-verification gap

**Files:**
- Create: `instruction_to_preset.py`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: `InstructionParser` implementations (Tasks 4-5), `read_aggregated_window` (Task 1), `VCVRackBackend.channel_descriptions()` (Task 2).
- Produces: a runnable CLI script. No new test file — this script's own logic is a thin composition of already-tested pieces (matching Task 5 of the Stage 0 plan's precedent for a capstone script); its correctness is demonstrated by live use once credentials/a local server exist, not by a new unit test suite.

- [ ] **Step 1: Implement the CLI**

```python
# instruction_to_preset.py
import argparse
import time

import numpy as np

from backends.vcv_rack import VCVRackBackend
from features import read_aggregated_window
from instruction_parser.anthropic_parser import AnthropicInstructionParser
from instruction_parser.local_parser import LocalInstructionParser


def main():
    parser = argparse.ArgumentParser(
        description="Parse a natural-language instruction into a CV recipe, "
        "apply it to the running VCV Rack patch, and report measured audio."
    )
    parser.add_argument("instruction", help="e.g. 'make a squelchy resonant acid bass'")
    parser.add_argument("--config", default="configs/sequencer_test.yaml")
    parser.add_argument("--llm", choices=["anthropic", "local"], default="anthropic")
    parser.add_argument("--model", default=None, help="model name/ID; provider-specific default if omitted")
    parser.add_argument("--base-url", default=None, help="required for --llm local")
    parser.add_argument("--settle-time-s", type=float, default=1.0)
    parser.add_argument("--aggregate-window-s", type=float, default=5.0)
    args = parser.parse_args()

    if args.llm == "anthropic":
        instruction_parser = AnthropicInstructionParser(model=args.model or "claude-opus-5")
    else:
        if not args.base_url:
            parser.error("--base-url is required when --llm local")
        instruction_parser = LocalInstructionParser(base_url=args.base_url, model=args.model)

    backend = VCVRackBackend(args.config)
    try:
        channels = backend.channel_descriptions()
        recipe = instruction_parser.parse_recipe(args.instruction, channels)

        for channel, value in recipe.items():
            backend.set_cv(channel, value)
        time.sleep(args.settle_time_s)

        measured = read_aggregated_window(backend, backend.sample_rate(), args.aggregate_window_s)

        print(f"instruction: {args.instruction!r}")
        print("recipe:")
        for channel, value in recipe.items():
            print(f"  {channel:<20} {value:.3f}")
        print(
            "measured [mean, std] x [loudness, brightness, pitch]:\n  "
            f"{np.array2string(measured, precision=4)}"
        )
    finally:
        backend.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-check the script's own wiring without live credentials**

Run `python3 instruction_to_preset.py --help` and confirm the argument parser builds and prints usage cleanly (catches any import-time error across the whole dependency chain — `instruction_parser`, `features`, `backends.vcv_rack` — without needing a live LLM call or a running VCV Rack instance). This is not a substitute for real verification; it only confirms the script is importable and its CLI surface is well-formed.

- [ ] **Step 3: Document the live-verification gap in `CLAUDE.md`**

Add a dated section (matching the file's existing style) stating plainly: Stage 1's code is written and unit-tested against mocked LLM responses only; `instruction_to_preset.py` has never been run against a real Anthropic API call or a real local model server, because this machine currently has neither Anthropic credentials nor a running local model server. This is a known, explicit gap, not an oversight — state what's needed to close it (an `ANTHROPIC_API_KEY`/`ant auth login` profile, or a running OpenAI-compatible local server plus its `--base-url`/`--model`) and that closing it is the natural next step whenever either becomes available.

- [ ] **Step 4: Run the full test suite**

Run: `python3 -m pytest -q`
Expected: all non-hardware tests pass (this plan adds none touching `ttnn`/hardware, so no `gozer`-leased run is needed for this stage).

- [ ] **Step 5: Commit**

```bash
git add instruction_to_preset.py CLAUDE.md
git commit -m "Add instruction_to_preset.py CLI; document Stage 1's live-verification gap"
```

## Verification

Full non-hardware test suite green (`python3 -m pytest -q`). No hardware-marked tests are added by this plan, so no `gozer`-leased run is required for Stage 1 itself. Live verification (a real LLM call against either provider) is explicitly out of scope for this plan's own completion bar — Task 6 documents that gap rather than closing it.
