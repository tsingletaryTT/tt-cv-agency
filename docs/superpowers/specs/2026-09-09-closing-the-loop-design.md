# Closing the loop: instruction-to-goal steering — design

## Context

The roadmap's own stated closing move (`CLAUDE.md`, "A four-stage
roadmap"): "once Stage 3 exists, Stage 1 gets upgraded — an instruction
stops meaning 'set this static recipe' and starts meaning 'steer toward
this goal over time,' with Stage 3's predictor doing the steering." Stage
3 (trajectory-predicting control) is now done, including a follow-up fix
that made its live result beat Stage 0's baseline. This is that upgrade.

**What changes, concretely.** Today, `instruction_to_preset.py` parses an
instruction into a static per-channel CV recipe (`InstructionParser.
parse_recipe`) and applies it once. This adds a second capability,
`InstructionParser.parse_goal`, that parses an instruction into a target
in *feature* space — the same 6-dim `[mean, std] × [loudness, brightness,
pitch]` vector `run_trajectory_control_loop` already accepts as
`goal_features` — and a new capstone script that hands that goal to the
existing CEM-planned control loop to steer toward continuously, instead
of setting knobs once and stopping.

**Add, don't replace.** `parse_recipe` stays — it's still the right tool
for "just set this preset," and nothing about it needs to change.
`parse_goal` is a new, second method on the same `InstructionParser`
interface, sharing its existing client-construction, exception type, and
provider implementations (`AnthropicInstructionParser`,
`LocalInstructionParser`) rather than introducing a parallel hierarchy.

**A carried-forward gap, stated up front, not discovered later.** This
machine still has no `ANTHROPIC_API_KEY`/`ant auth login` profile and no
local OpenAI-compatible server running — the exact gap Stage 1 shipped
with and never closed. This work ships the same way: fully built and
tested against mocked LLM responses, never run against a real model. This
is a known, explicit limitation, not a blocker to building it now — Stage
1 didn't wait for credentials, and neither Stage 2 nor 3's own work has
changed that anything is available now.

Out of scope: any change to `parse_recipe`, `instruction_to_preset.py`,
`trajectory_control_loop.py`, `cem_planner.py`, `tt_trajectory_inference.py`,
`trajectory_model.py`, `trajectory_collection.py`, or anything upstream of
this stage's own two new capabilities (a goal-parsing method and a
capstone script that uses it). Also out of scope: any "continuous
re-target mid-listen" interaction (accepting a fresh instruction while a
control loop is already running) — this ships one instruction → one goal
→ steer-until-convergence-or-timeout → stop, matching every other
capstone script's shape in this project.

## Architecture

```
 instruction (text) ──▶ InstructionParser.parse_goal(instruction)
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
   AnthropicInstructionParser         LocalInstructionParser
   (messages.parse against a          (chat.completions.create
    FIXED GoalFeatures Pydantic        against GoalFeatures'
    class -- not built per-call,       own .model_json_schema(),
    unlike build_recipe_model)         same fixed-schema reasoning)
              │                               │
              └───────────────┬───────────────┘
                               ▼
                dict[str, float] over schema.GOAL_DIMS
                (loud_mean, loud_std, bright_mean,
                 bright_std, pitch_mean, pitch_std)
                               ▼
         instruction_to_goal.py: goal_features = np.array([...])
                               ▼
      run_trajectory_control_loop(backend, engine.predict_next_state,
                                   goal_features)  -- unchanged, Stage 3's
                               ▼
        steer toward the goal over time, report the measured history
```

### Why the goal schema is fixed, not dynamic

`parse_recipe`'s schema (`build_recipe_model`) is built fresh per call
from whatever `channels: dict[str, str]` the current patch has, because a
CV recipe is inherently patch-specific — different patches have different
channels. A *goal*, by contrast, is expressed in feature space: loudness,
brightness, and pitch are measured the same way regardless of which patch
is running (`features.py`'s `extract_features_aggregated` is already
patch-agnostic). So the goal schema is one fixed class, defined once, not
rebuilt per call — a genuine simplification, not a shortcut.

### `instruction_parser/base.py`: a second abstract method

```python
class InstructionParser(ABC):
    @abstractmethod
    def parse_recipe(self, instruction: str, channels: dict[str, str]) -> dict[str, float]:
        ...  # unchanged

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

### `instruction_parser/schema.py`: `GoalFeatures` + `validate_goal_json`

```python
GOAL_DIMS = ["loud_mean", "loud_std", "bright_mean", "bright_std", "pitch_mean", "pitch_std"]
# Order matches features.py's extract_features_aggregated documented
# interleaving exactly: [mean(loudness), std(loudness), mean(brightness),
# std(brightness), mean(pitch), std(pitch)] -- so
# np.array([goal_dict[name] for name in GOAL_DIMS]) is directly usable as
# run_trajectory_control_loop's goal_features with no reordering.

class GoalFeatures(pydantic.BaseModel):
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

### `instruction_parser/prompts.py`: `build_goal_system_prompt`

A fixed prompt (no `channels` argument, unlike `build_system_prompt`)
explaining what the 6 feature dimensions mean sonically, since "make it
squelchy" needs translating into "high `bright_std`" the same way
`build_system_prompt` currently helps translate it into "high
`vcf_resonance`." Must explain: loudness/brightness/pitch's `mean` (the
typical/average value over the listening window) vs. `std` (how much
that quality *moves* within the window — a filter sweep is high
`bright_std`, a held drone is low `pitch_std`), with 1-2 worked examples
so the model calibrates its output range the same way channel
descriptions currently calibrate `parse_recipe`'s.

### `instruction_parser/anthropic_parser.py`: `parse_goal`

Same shape as the existing `parse_recipe`: `client.messages.parse(...,
system=build_goal_system_prompt(), output_format=GoalFeatures)`, the same
`parsed_output is None` refusal guard, no re-validation (the SDK's own
`output_format` enforcement already validates, exactly as documented for
`parse_recipe`).

### `instruction_parser/local_parser.py`: `parse_goal`

Same shape as the existing `parse_recipe`: `client.chat.completions.
create(..., response_format={"type": "json_schema", "json_schema":
{"name": "goal_features", "schema": GoalFeatures.model_json_schema()}})`,
the same empty-choices/`None`-content guard, then `validate_goal_json`
on the raw response text. Uses `GoalFeatures.model_json_schema()`
directly rather than a hand-rolled schema dict (unlike `parse_recipe`'s
`_build_json_schema(channels)`) — since `GoalFeatures` is a fixed,
already-fully-specified Pydantic model (`Field(ge=0, le=1)` on every
field, `extra="forbid"`), pydantic's own JSON Schema export already
produces the right `minimum`/`maximum`/`required`/`additionalProperties:
false` shape with nothing to hand-maintain in a second place.

### `instruction_to_goal.py` (new capstone script)

CLI: an instruction string (positional), `--config` (default
`configs/sequencer_test.yaml`), `--llm {anthropic,local}`, `--model`,
`--base-url` (same required-together guard as `instruction_to_preset.py`:
`--llm local` requires both), `--weights-path` (default
`data/sequencer_trajectory_model_weights.npz`), `--max-iterations`
(default 100, forwarded to `run_trajectory_control_loop`). Steps: build
the chosen `InstructionParser`, call `parse_goal(instruction)`, convert
the returned dict to `goal_features` via `schema.GOAL_DIMS`'s order,
build a `VCVRackBackend` and a `TrajectoryTTInferenceEngine` (same
`expected_channels=backend.channels()` provenance check
`trajectory_control_loop.py`'s own `__main__` already uses), call
`run_trajectory_control_loop`, print the instruction, the parsed goal,
and the final measured feature vector vs. goal — same "report for a human
to judge, no automated grading" scope boundary `instruction_to_preset.py`
already established.

## Global constraints

- No edits to `parse_recipe`, `instruction_to_preset.py`,
  `trajectory_control_loop.py`, `cem_planner.py`,
  `tt_trajectory_inference.py`, `trajectory_model.py`,
  `trajectory_collection.py`, `features.py`, `backends/`, or
  `configs/*.yaml`.
- `parse_goal`'s schema is fixed (`GoalFeatures`, always the same 6
  fields) — never built per-call, never patch-specific.
- Goal values stay `[0, 1]` at every interface boundary, same as every
  other stage.
- `GOAL_DIMS`'s order is exactly `features.py`'s documented interleaving
  (`extract_features_aggregated`'s `[mean0, std0, mean1, std1, mean2,
  std2]`) — a mismatch here would be the same class of column-ordering
  bug this project's own CLAUDE.md already had to catch and fix once in
  Stage 2's capstone reporting.
- Follow the `claude-api` skill's guidance for any Anthropic-specific
  code, same as Stage 1.
- Neither `parse_goal` implementation may live in a file that mixes the
  Anthropic SDK and the OpenAI SDK — same file-separation discipline as
  `parse_recipe`.
- New capstone file (`instruction_to_goal.py`) is flat at the repo root,
  matching every other capstone script in this project.

## Verification plan

1. **`GoalFeatures`/`validate_goal_json` are unit-tested the same way
   `build_recipe_model`/`validate_recipe_json` already are**: a
   well-formed round-trip, malformed JSON, an out-of-range value, a
   missing field, and an extra field, each raising `RecipeParseError`
   with a specific message.
2. **`build_goal_system_prompt` is unit-tested for content, not just that
   it returns a string**: confirms all 6 `GOAL_DIMS` names appear in the
   prompt text, and that it takes no arguments (patch-independent, unlike
   `build_system_prompt`).
3. **Both `parse_goal` implementations are unit-tested against mocked
   clients**, mirroring `parse_recipe`'s existing test shape exactly: a
   well-formed mocked response round-trips to the expected dict; a
   refusal/empty-choices response raises `RecipeParseError`;
   `LocalInstructionParser`'s test additionally asserts the actual
   constructed request payload (system prompt, schema, model) the same
   way its `parse_recipe` tests already do.
4. **`instruction_to_goal.py` gets the same smoke-test bar
   `instruction_to_preset.py`'s Task 6 used**: `--help` runs clean,
   confirming the whole import chain resolves and the CLI's argument
   surface (including the `--llm local` requires-`--model`-and-`--base-url`
   guard) is well-formed. Live verification against a real LLM is
   explicitly out of scope, for the same reason it was for Stage 1 — no
   credentials or local server exist on this machine. Document this
   plainly in `CLAUDE.md`, the same way Stage 1's own gap was documented,
   rather than letting it look silently closed because the code merged
   and tests are green.

Full existing test suite (120 non-hardware + 7 hardware) must stay green;
this work adds only non-hardware tests (nothing here touches `ttnn`).
