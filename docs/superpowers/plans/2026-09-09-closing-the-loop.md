# Closing the Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `InstructionParser.parse_goal` (instruction → feature-space
goal) alongside the existing `parse_recipe`, and a capstone script that
hands that goal to Stage 3's CEM-planned control loop to steer toward
continuously.

**Architecture:** A second abstract method on the existing
`InstructionParser` interface, a fixed (non-per-patch) `GoalFeatures`
Pydantic schema, a new fixed system prompt, `parse_goal` implementations
on both existing provider classes, and one new capstone script
(`instruction_to_goal.py`) — all reusing Stage 1/3's existing
infrastructure unchanged.

**Tech Stack:** Python 3.12, `anthropic`, `openai`, `pydantic`, pytest,
mocked LLM clients (no live LLM access on this machine — a known,
documented gap, same as Stage 1's own).

**Spec:** `docs/superpowers/specs/2026-09-09-closing-the-loop-design.md`

## Global Constraints

- No edits to `parse_recipe`, `instruction_to_preset.py`,
  `trajectory_control_loop.py`, `cem_planner.py`,
  `tt_trajectory_inference.py`, `trajectory_model.py`,
  `trajectory_collection.py`, `features.py`, `backends/`, or
  `configs/*.yaml`.
- `parse_goal`'s schema (`GoalFeatures`) is fixed — one class, defined
  once, never rebuilt per call, never patch-specific.
- Goal values stay `[0, 1]` at every interface boundary.
- `GOAL_DIMS`'s order is exactly `features.py`'s documented interleaving:
  `["loud_mean", "loud_std", "bright_mean", "bright_std", "pitch_mean", "pitch_std"]`.
- Neither `parse_goal` implementation may mix the Anthropic SDK and the
  OpenAI SDK in the same file.
- New capstone file (`instruction_to_goal.py`) is flat at the repo root.
- Live verification against a real LLM is explicitly out of scope for
  every task in this plan (mocked tests only) — document the gap in
  `CLAUDE.md`, don't hide it.

---

### Task 1: `GoalFeatures` schema + `base.py`'s second abstract method

**Files:**
- Modify: `instruction_parser/base.py` (add abstract `parse_goal`)
- Modify: `instruction_parser/schema.py` (add `GOAL_DIMS`, `GoalFeatures`, `validate_goal_json`)
- Test: `tests/test_instruction_parser_schema.py` (add goal-schema tests)

**Interfaces:**
- Consumes: `pydantic`, `json`, the existing `RecipeParseError`.
- Produces: `GOAL_DIMS: list[str]`, `GoalFeatures` (pydantic model),
  `validate_goal_json(raw_json: str) -> dict[str, float]`. Tasks 3-5
  import `GOAL_DIMS`/`GoalFeatures`/`validate_goal_json` from
  `instruction_parser.schema`, and every concrete `InstructionParser`
  subclass (Task 3, Task 4) must implement the new abstract `parse_goal`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_instruction_parser_schema.py`:

```python
from instruction_parser.schema import GOAL_DIMS, GoalFeatures, validate_goal_json

GOAL_VALUES = {
    "loud_mean": 0.5, "loud_std": 0.1, "bright_mean": 0.3,
    "bright_std": 0.05, "pitch_mean": 0.7, "pitch_std": 0.2,
}


def test_goal_dims_matches_features_py_interleaved_order():
    # features.py's extract_features_aggregated documents its output
    # order as [mean(loudness), std(loudness), mean(brightness),
    # std(brightness), mean(pitch), std(pitch)] -- GOAL_DIMS must match
    # exactly, so np.array([goal[name] for name in GOAL_DIMS]) is
    # directly usable as run_trajectory_control_loop's goal_features.
    assert GOAL_DIMS == ["loud_mean", "loud_std", "bright_mean", "bright_std", "pitch_mean", "pitch_std"]


def test_goal_features_accepts_valid_values():
    instance = GoalFeatures(**GOAL_VALUES)
    for name, value in GOAL_VALUES.items():
        assert getattr(instance, name) == value


def test_goal_features_rejects_out_of_range():
    bad = dict(GOAL_VALUES)
    bad["pitch_mean"] = 1.5
    with pytest.raises(Exception):  # pydantic.ValidationError
        GoalFeatures(**bad)


def test_validate_goal_json_round_trips():
    raw = json.dumps(GOAL_VALUES)
    result = validate_goal_json(raw)
    assert result == GOAL_VALUES


def test_validate_goal_json_rejects_malformed_json():
    with pytest.raises(RecipeParseError):
        validate_goal_json("not json at all {{{")


def test_validate_goal_json_rejects_out_of_range_value():
    bad = dict(GOAL_VALUES)
    bad["loud_std"] = -0.1
    with pytest.raises(RecipeParseError):
        validate_goal_json(json.dumps(bad))


def test_validate_goal_json_rejects_missing_field():
    incomplete = dict(GOAL_VALUES)
    del incomplete["pitch_std"]
    with pytest.raises(RecipeParseError):
        validate_goal_json(json.dumps(incomplete))


def test_validate_goal_json_rejects_extra_field():
    extra = dict(GOAL_VALUES)
    extra["extra_dim"] = 0.1
    with pytest.raises(RecipeParseError):
        validate_goal_json(json.dumps(extra))
```

(The file already imports `json`, `pytest`, and `RecipeParseError` at the
top — reuse those imports, don't duplicate them.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_instruction_parser_schema.py -v`
Expected: FAIL with `ImportError: cannot import name 'GOAL_DIMS'`

- [ ] **Step 3: Write the implementation**

Add to `instruction_parser/schema.py` (after the existing
`validate_recipe_json`):

```python
GOAL_DIMS = ["loud_mean", "loud_std", "bright_mean", "bright_std", "pitch_mean", "pitch_std"]
# Order matches features.py's extract_features_aggregated documented
# interleaving exactly: [mean(loudness), std(loudness), mean(brightness),
# std(brightness), mean(pitch), std(pitch)] -- so
# np.array([goal_dict[name] for name in GOAL_DIMS]) is directly usable
# as run_trajectory_control_loop's goal_features with no reordering.


class GoalFeatures(pydantic.BaseModel):
    """A goal expressed in FEATURE space (what the sound should measure
    like), not CV space -- unlike CVRecipe, this schema is fixed and
    never rebuilt per call: loudness/brightness/pitch mean+std are
    measured the same way regardless of which patch is running."""
    model_config = pydantic.ConfigDict(extra="forbid")
    loud_mean: float = pydantic.Field(ge=0.0, le=1.0)
    loud_std: float = pydantic.Field(ge=0.0, le=1.0)
    bright_mean: float = pydantic.Field(ge=0.0, le=1.0)
    bright_std: float = pydantic.Field(ge=0.0, le=1.0)
    pitch_mean: float = pydantic.Field(ge=0.0, le=1.0)
    pitch_std: float = pydantic.Field(ge=0.0, le=1.0)


def validate_goal_json(raw_json: str) -> dict[str, float]:
    """Same validation shape as validate_recipe_json (malformed JSON,
    out-of-range value, missing/extra key all raise RecipeParseError with
    a specific message), against the fixed GoalFeatures schema instead of
    a per-call dynamic one."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise RecipeParseError(f"response was not valid JSON: {e}") from e

    try:
        instance = GoalFeatures.model_validate(data)
    except pydantic.ValidationError as e:
        raise RecipeParseError(f"response did not match the expected goal schema: {e}") from e

    return {name: getattr(instance, name) for name in GOAL_DIMS}
```

Add to `instruction_parser/base.py` (inside the existing
`InstructionParser` class, after `parse_recipe`):

```python
    @abstractmethod
    def parse_goal(self, instruction: str) -> dict[str, float]:
        """Given a natural-language instruction, return a target value in
        [0, 1] for each of schema.GOAL_DIMS (loud_mean, loud_std,
        bright_mean, bright_std, pitch_mean, pitch_std) -- a goal in
        FEATURE space (what the sound should measure like), not CV space.
        Unlike parse_recipe, this schema never varies by patch. Raises
        RecipeParseError on the same failure classes as parse_recipe
        (malformed response, out-of-range value, missing/extra key,
        refusal/empty completion)."""
        ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_instruction_parser_schema.py -v`
Expected: FAIL now with a DIFFERENT error — `AnthropicInstructionParser`
and `LocalInstructionParser` (in `instruction_parser/anthropic_parser.py`
and `instruction_parser/local_parser.py`) no longer satisfy the
`InstructionParser` ABC once it grows a second abstract method, so
`AnthropicInstructionParser()`/`LocalInstructionParser(...)` will raise
`TypeError: Can't instantiate abstract class ... without an implementation
for abstract method 'parse_goal'` wherever any EXISTING test constructs
one. This is expected and correct at this point in the plan — Task 1's
own new tests (which only touch `schema.py`, not the parser classes)
should already pass; Tasks 3 and 4 fix the now-broken existing parser
tests by adding the missing implementations. Confirm specifically that
`tests/test_instruction_parser_schema.py` itself is fully green before
moving on; a broken `test_anthropic_instruction_parser.py`/
`test_local_instruction_parser.py` at this exact point in the plan is
expected, not a regression to chase down here.

- [ ] **Step 5: Commit**

```bash
git add instruction_parser/base.py instruction_parser/schema.py tests/test_instruction_parser_schema.py
git commit -m "Add GoalFeatures schema and InstructionParser.parse_goal abstract method"
```

---

### Task 2: `build_goal_system_prompt`

**Files:**
- Modify: `instruction_parser/prompts.py`
- Test: `tests/test_prompts.py` (new file — no existing prompt-specific
  test file exists today; `build_system_prompt`'s own coverage lives
  inline inside the parser test files via the `system` prompt content
  assertions already shown above. This new file is the first dedicated
  prompt-content test, since `build_goal_system_prompt` has no per-call
  arguments to smuggle assertions through a parser test's mock the way
  channel descriptions currently do.)

**Interfaces:**
- Consumes: nothing (no imports beyond what's already in `prompts.py`).
- Produces: `build_goal_system_prompt() -> str`. Tasks 3 and 4 import and
  call this with no arguments.

- [ ] **Step 1: Write the failing test**

Create `tests/test_prompts.py`:

```python
from instruction_parser.prompts import build_goal_system_prompt
from instruction_parser.schema import GOAL_DIMS


def test_build_goal_system_prompt_mentions_all_goal_dims():
    prompt = build_goal_system_prompt()
    for name in GOAL_DIMS:
        assert name in prompt


def test_build_goal_system_prompt_takes_no_arguments():
    # Unlike build_system_prompt(channels), this prompt is patch-independent
    # -- calling it twice with no arguments must return the same string.
    assert build_goal_system_prompt() == build_goal_system_prompt()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_prompts.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_goal_system_prompt'`

- [ ] **Step 3: Write the implementation**

Add to `instruction_parser/prompts.py` (after the existing
`build_system_prompt`):

```python
GOAL_SYSTEM_PREAMBLE = (
    "You listen to a modular synthesizer and translate a description of "
    "a desired sound into a target for 3 measured audio qualities, each "
    "reported as a [mean, std] pair over a several-second listening "
    "window, every value in [0, 1]:\n"
    "- loudness (loud_mean, loud_std): overall level. High mean = loud, "
    "low mean = quiet/near-silent. High std = the loudness noticeably "
    "swells or drops within the window (e.g. a pumping or gated sound); "
    "low std = a steady, unchanging level.\n"
    "- brightness (bright_mean, bright_std): tonal brightness/harshness "
    "(spectral centroid). High mean = bright/harsh/open-filter, low mean "
    "= dark/muffled/closed-filter. High std = the brightness is actively "
    "sweeping or moving (e.g. a filter sweep, a wah-like motion); low "
    "std = a fixed tone color.\n"
    "- pitch (pitch_mean, pitch_std): perceived fundamental pitch, on a "
    "log scale from low (0) to high (1). High std = the pitch is "
    "actively moving (a sequence, an arpeggio, vibrato, a pitch sweep); "
    "low std = a held, steady pitch.\n"
    "Respond with a target value in [0, 1] for each of the 6 fields. "
    "Example: 'a steady, dark, low drone' -> low pitch_mean, low "
    "bright_mean, low std on everything. 'a bright, sweeping, squelchy "
    "sequence' -> high bright_mean AND high bright_std (the sweep), high "
    "pitch_std (the moving sequence)."
)


def build_goal_system_prompt() -> str:
    """Fixed system prompt for goal-parsing (unlike build_system_prompt,
    this never varies per patch -- the 6 feature dimensions it describes
    are the same regardless of which instrument is running, since they
    describe the SOUND, not any particular patch's CV channels)."""
    return GOAL_SYSTEM_PREAMBLE
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_prompts.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add instruction_parser/prompts.py tests/test_prompts.py
git commit -m "Add build_goal_system_prompt: fixed, patch-independent goal prompt"
```

---

### Task 3: `AnthropicInstructionParser.parse_goal`

**Files:**
- Modify: `instruction_parser/anthropic_parser.py`
- Modify: `tests/test_anthropic_instruction_parser.py` (add goal tests)

**Interfaces:**
- Consumes: `GoalFeatures` (Task 1, `instruction_parser.schema`),
  `build_goal_system_prompt` (Task 2, `instruction_parser.prompts`).
- Produces: `AnthropicInstructionParser.parse_goal(instruction: str) ->
  dict[str, float]` — completes this class's `InstructionParser`
  implementation (fixes the `TypeError` Task 1 introduced for this class).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_anthropic_instruction_parser.py`:

```python
GOAL_VALUES = {
    "loud_mean": 0.5, "loud_std": 0.1, "bright_mean": 0.3,
    "bright_std": 0.05, "pitch_mean": 0.7, "pitch_std": 0.2,
}


def test_parse_goal_returns_dict_from_parsed_output():
    parser = AnthropicInstructionParser()
    with patch("instruction_parser.anthropic_parser.anthropic.Anthropic") as MockClient:
        mock_client = MockClient.return_value
        mock_client.messages.parse.return_value = _mock_parsed_response(**GOAL_VALUES)
        result = parser.parse_goal("make it bright and sweeping")
    assert result == GOAL_VALUES


def test_parse_goal_uses_configured_model():
    parser = AnthropicInstructionParser(model="claude-sonnet-5")
    with patch("instruction_parser.anthropic_parser.anthropic.Anthropic") as MockClient:
        mock_client = MockClient.return_value
        mock_client.messages.parse.return_value = _mock_parsed_response(**GOAL_VALUES)
        parser.parse_goal("anything")
    _, kwargs = mock_client.messages.parse.call_args
    assert kwargs["model"] == "claude-sonnet-5"


def test_parse_goal_raises_recipe_parse_error_on_refusal():
    parser = AnthropicInstructionParser()
    with patch("instruction_parser.anthropic_parser.anthropic.Anthropic") as MockClient:
        mock_client = MockClient.return_value
        response = MagicMock()
        response.parsed_output = None
        response.stop_reason = "refusal"
        mock_client.messages.parse.return_value = response
        with pytest.raises(RecipeParseError, match="refusal"):
            parser.parse_goal("anything")
```

(Reuses this file's existing `_mock_parsed_response` helper, `CHANNELS`
constant left untouched, and existing imports — no new imports needed.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_anthropic_instruction_parser.py -v`
Expected: every test in this file FAILS with `TypeError: Can't
instantiate abstract class AnthropicInstructionParser without an
implementation for abstract method 'parse_goal'` (Task 1's expected
intermediate breakage for this file) — the three new tests above fail
for that same reason, not yet for a missing-method-specific reason.

- [ ] **Step 3: Write the implementation**

Add to `instruction_parser/anthropic_parser.py` (update the import line
to add `GoalFeatures`, and add the new method after `parse_recipe`):

```python
from instruction_parser.schema import GoalFeatures, build_recipe_model
```

```python
    def parse_goal(self, instruction: str) -> dict[str, float]:
        client = anthropic.Anthropic()
        try:
            response = client.messages.parse(
                model=self._model,
                max_tokens=1024,
                system=build_goal_system_prompt(),
                messages=[{"role": "user", "content": instruction}],
                output_format=GoalFeatures,
            )
        except Exception as e:
            raise RecipeParseError(f"Anthropic API call failed: {e}") from e

        if response.parsed_output is None:
            raise RecipeParseError(
                f"Anthropic response had no parsed output (stop_reason={response.stop_reason!r})"
            )

        # Deliberately no validate_goal_json call here, same reasoning as
        # parse_recipe: client.messages.parse's own output_format
        # enforcement already validates range/required-keys before
        # parsed_output is ever populated.
        return response.parsed_output.model_dump()
```

Also update the existing `from instruction_parser.prompts import
build_system_prompt` line to `from instruction_parser.prompts import
build_goal_system_prompt, build_system_prompt` (both prompts are now
used in this file).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_anthropic_instruction_parser.py -v`
Expected: all PASS (the whole file, both the pre-existing `parse_recipe`
tests and this task's new `parse_goal` ones).

- [ ] **Step 5: Commit**

```bash
git add instruction_parser/anthropic_parser.py tests/test_anthropic_instruction_parser.py
git commit -m "Add AnthropicInstructionParser.parse_goal"
```

---

### Task 4: `LocalInstructionParser.parse_goal`

**Files:**
- Modify: `instruction_parser/local_parser.py`
- Modify: `tests/test_local_instruction_parser.py` (add goal tests)

**Interfaces:**
- Consumes: `GoalFeatures`, `validate_goal_json` (Task 1,
  `instruction_parser.schema`), `build_goal_system_prompt` (Task 2,
  `instruction_parser.prompts`).
- Produces: `LocalInstructionParser.parse_goal(instruction: str) ->
  dict[str, float]` — completes this class's `InstructionParser`
  implementation (fixes the `TypeError` Task 1 introduced for this class).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_local_instruction_parser.py`:

```python
GOAL_VALUES = {
    "loud_mean": 0.5, "loud_std": 0.1, "bright_mean": 0.3,
    "bright_std": 0.05, "pitch_mean": 0.7, "pitch_std": 0.2,
}


def test_parse_goal_returns_dict_from_valid_response():
    parser = LocalInstructionParser(base_url="http://localhost:8000/v1", model="local-model")
    with patch("instruction_parser.local_parser.openai.OpenAI") as MockClient:
        mock_client = MockClient.return_value
        mock_client.chat.completions.create.return_value = _mock_chat_response(
            json.dumps(GOAL_VALUES)
        )
        result = parser.parse_goal("make it bright and sweeping")
    assert result == GOAL_VALUES


def test_parse_goal_raises_on_malformed_local_model_output():
    parser = LocalInstructionParser(base_url="http://localhost:8000/v1", model="local-model")
    with patch("instruction_parser.local_parser.openai.OpenAI") as MockClient:
        mock_client = MockClient.return_value
        mock_client.chat.completions.create.return_value = _mock_chat_response(
            "I think it should sound bright"  # not JSON at all
        )
        with pytest.raises(RecipeParseError):
            parser.parse_goal("make it bright and sweeping")


def test_parse_goal_raises_recipe_parse_error_on_empty_choices():
    parser = LocalInstructionParser(base_url="http://localhost:8000/v1", model="local-model")
    with patch("instruction_parser.local_parser.openai.OpenAI") as MockClient:
        mock_client = MockClient.return_value
        response = MagicMock()
        response.choices = []
        mock_client.chat.completions.create.return_value = response
        with pytest.raises(RecipeParseError):
            parser.parse_goal("make it bright and sweeping")


def test_parse_goal_sends_fixed_goal_json_schema():
    # Unlike parse_recipe's per-channel schema, this must be the SAME
    # fixed set of 6 fields regardless of anything about the caller --
    # there's no varying-channel-count case to sweep here.
    parser = LocalInstructionParser(base_url="http://localhost:8000/v1", model="local-model")
    with patch("instruction_parser.local_parser.openai.OpenAI") as MockClient:
        mock_client = MockClient.return_value
        mock_client.chat.completions.create.return_value = _mock_chat_response(
            json.dumps(GOAL_VALUES)
        )
        parser.parse_goal("anything")
    _, call_kwargs = mock_client.chat.completions.create.call_args
    schema = call_kwargs["response_format"]["json_schema"]["schema"]
    assert set(schema["properties"].keys()) == set(GOAL_VALUES.keys())
    assert set(schema["required"]) == set(GOAL_VALUES.keys())
    assert schema["additionalProperties"] is False


def test_parse_goal_includes_goal_dimension_explanation_in_system_message():
    parser = LocalInstructionParser(base_url="http://localhost:8000/v1", model="local-model")
    with patch("instruction_parser.local_parser.openai.OpenAI") as MockClient:
        mock_client = MockClient.return_value
        mock_client.chat.completions.create.return_value = _mock_chat_response(
            json.dumps(GOAL_VALUES)
        )
        parser.parse_goal("anything")
    _, call_kwargs = mock_client.chat.completions.create.call_args
    system_message = next(m["content"] for m in call_kwargs["messages"] if m["role"] == "system")
    assert "loudness" in system_message
    assert "brightness" in system_message
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_local_instruction_parser.py -v`
Expected: every test in this file FAILS with `TypeError: Can't
instantiate abstract class LocalInstructionParser without an
implementation for abstract method 'parse_goal'` (Task 1's expected
intermediate breakage for this file).

- [ ] **Step 3: Write the implementation**

Update the import lines at the top of `instruction_parser/local_parser.py`:

```python
from instruction_parser.prompts import build_goal_system_prompt, build_system_prompt
from instruction_parser.schema import GoalFeatures, validate_goal_json, validate_recipe_json
```

Add after the existing `parse_recipe` method:

```python
    def parse_goal(self, instruction: str) -> dict[str, float]:
        client = openai.OpenAI(base_url=self._base_url, api_key=self._api_key)
        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": build_goal_system_prompt()},
                    {"role": "user", "content": instruction},
                ],
                # No hand-rolled schema dict needed here (unlike
                # parse_recipe's _build_json_schema(channels)) -- GoalFeatures
                # is a fixed, already-fully-specified Pydantic model
                # (Field(ge=0,le=1) on every field, extra="forbid"), so its
                # own .model_json_schema() already produces the right
                # minimum/maximum/required/additionalProperties shape with
                # nothing to hand-maintain in a second place.
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "goal_features", "schema": GoalFeatures.model_json_schema()},
                },
                max_tokens=1024,
            )
        except Exception as e:
            raise RecipeParseError(f"local model API call failed: {e}") from e

        if not response.choices or response.choices[0].message.content is None:
            finish_reason = response.choices[0].finish_reason if response.choices else None
            raise RecipeParseError(
                f"local model returned no usable message content (finish_reason={finish_reason!r})"
            )

        return validate_goal_json(response.choices[0].message.content)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_local_instruction_parser.py -v`
Expected: all PASS (the whole file, both the pre-existing `parse_recipe`
tests and this task's new `parse_goal` ones).

- [ ] **Step 5: Run the full non-hardware suite to confirm no regressions**

Run: `python3 -m pytest -q`
Expected: all previously-passing tests pass, plus every new test from
Tasks 1-4 (deselected hardware count unchanged at 7).

- [ ] **Step 6: Commit**

```bash
git add instruction_parser/local_parser.py tests/test_local_instruction_parser.py
git commit -m "Add LocalInstructionParser.parse_goal"
```

---

### Task 5: `instruction_to_goal.py` capstone script, and documentation

**Files:**
- Create: `instruction_to_goal.py`
- Modify: `README.md` ("What's here" section — one new entry; Requirements
  line unchanged, no new dependency)
- Modify: `CLAUDE.md` (a dated section documenting this work and its
  carried-forward live-LLM gap, matching Stage 1 Task 6's own precedent)

**Interfaces:**
- Consumes: `AnthropicInstructionParser`/`LocalInstructionParser` (Tasks
  3-4), `GOAL_DIMS` (Task 1, `instruction_parser.schema`),
  `VCVRackBackend` (existing, unmodified), `TrajectoryTTInferenceEngine`
  (existing, unmodified), `run_trajectory_control_loop` (existing,
  unmodified).
- Produces: nothing consumed by later work — this is the stage's own
  capstone/completion check.

- [ ] **Step 1: Write `instruction_to_goal.py`**

```python
# instruction_to_goal.py
import argparse

import numpy as np

from backends.vcv_rack import VCVRackBackend
from instruction_parser.anthropic_parser import AnthropicInstructionParser
from instruction_parser.local_parser import LocalInstructionParser
from instruction_parser.schema import GOAL_DIMS
from trajectory_control_loop import run_trajectory_control_loop
from tt_trajectory_inference import TrajectoryTTInferenceEngine


def main():
    parser = argparse.ArgumentParser(
        description="Parse a natural-language instruction into a target sound "
        "(a goal in feature space), then steer the running VCV Rack patch "
        "toward it over time via the trajectory predictor + CEM planner."
    )
    parser.add_argument("instruction", help="e.g. 'make it bright and steady'")
    parser.add_argument("--config", default="configs/sequencer_test.yaml")
    parser.add_argument("--llm", choices=["anthropic", "local"], default="anthropic")
    parser.add_argument(
        "--model",
        default=None,
        help="model name/ID; defaults to claude-opus-5 for --llm anthropic, "
        "required (no default) for --llm local",
    )
    parser.add_argument("--base-url", default=None, help="required for --llm local")
    parser.add_argument(
        "--weights-path", default="data/sequencer_trajectory_model_weights.npz"
    )
    parser.add_argument("--max-iterations", type=int, default=100)
    args = parser.parse_args()

    if args.llm == "anthropic":
        instruction_parser = AnthropicInstructionParser(model=args.model or "claude-opus-5")
    else:
        if not args.base_url or not args.model:
            parser.error("--base-url and --model are required when --llm local")
        instruction_parser = LocalInstructionParser(base_url=args.base_url, model=args.model)

    goal_dict = instruction_parser.parse_goal(args.instruction)
    goal_features = np.array([goal_dict[name] for name in GOAL_DIMS])

    backend = VCVRackBackend(args.config)
    try:
        engine = TrajectoryTTInferenceEngine(
            weights_path=args.weights_path, expected_channels=backend.channels(),
        )
        try:
            print(f"instruction: {args.instruction!r}")
            print("goal:")
            for name, value in zip(GOAL_DIMS, goal_features):
                print(f"  {name:<12} {value:.3f}")

            history = run_trajectory_control_loop(
                backend, predict_fn=engine.predict_next_state, goal_features=goal_features,
                max_iterations=args.max_iterations,
            )
            print(f"final measured features: {history[-1]}, goal: {goal_features}")
        finally:
            engine.close()
    finally:
        backend.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the CLI**

Run: `python3 instruction_to_goal.py --help`
Expected: clean argument-parser help output, confirming the whole import
chain (`backends.vcv_rack`, `instruction_parser.anthropic_parser`,
`instruction_parser.local_parser`, `instruction_parser.schema`,
`trajectory_control_loop`, `tt_trajectory_inference`, and transitively
`cem_planner`) resolves with no stale import path, and that `argparse`
accepted every flag with no errors. This does not touch `ttnn`
(`tt_trajectory_inference.py` only opens a device inside
`TrajectoryTTInferenceEngine.__init__`, never at import time) — no
`gozer` lease is needed for this smoke test.

- [ ] **Step 3: Update README's "What's here" section**

Add one entry for `instruction_to_goal.py` immediately after the existing
`trajectory_control_loop.py` entry, matching the existing
one-paragraph-each style: what it does (parses an instruction into a
feature-space goal via a chosen `InstructionParser.parse_goal`, then
steers the running patch toward it via
`run_trajectory_control_loop`/`TrajectoryTTInferenceEngine`), and its
relationship to `instruction_to_preset.py` (same instruction-parsing
front end, but continuous goal-steering instead of a one-shot recipe
apply). No new Requirements-line entry — this task adds no new Python
package dependency.

- [ ] **Step 4: Full regression check**

```bash
python3 -m pytest -q
gozer run --chips 1 --who "claude:tt-cv-agency" --reason "final closing-the-loop suite check" -- python3 -m pytest -q -m hardware
```

Expected: the non-hardware count grows by this plan's new tests over the
120 baseline; hardware stays at 7 passed, 0 failures either way (nothing
in this plan touches `ttnn`).

- [ ] **Step 5: Document honestly in `CLAUDE.md`**

Add a new dated section (after the most recent Stage 3 follow-up section)
reporting: what was built (parse_goal on both providers, the fixed
GoalFeatures schema, the new capstone script), that all of it is
unit-tested against mocked LLM responses only, and — explicitly, the same
way Stage 1 Task 6's section already does — that this machine still has
neither an `ANTHROPIC_API_KEY`/`ant auth login` profile nor a running
local OpenAI-compatible server, so `instruction_to_goal.py` has never
been run end-to-end against a real model. Name exactly what closes this
gap (an API key or local-auth profile for `--llm anthropic`, or a running
server's `--base-url`/`--model` for `--llm local`) the same way Stage 1's
section does, so it reads as a remembered, explicit gap rather than an
oversight.

- [ ] **Step 6: Commit**

```bash
git add instruction_to_goal.py README.md CLAUDE.md
git commit -m "Add instruction_to_goal.py: closing the loop capstone, and its live-LLM gap"
```
