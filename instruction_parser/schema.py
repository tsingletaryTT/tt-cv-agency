import json

import pydantic

from instruction_parser.base import RecipeParseError


def _strip_markdown_code_fence(text: str) -> str:
    """Strip a ```-delimited code fence (optionally ```json) wrapping the
    whole response, if present. A real, observed quirk of local models with
    no server-side schema/grammar enforcement (confirmed live against
    Qwen3-0.6B on tt-local-generator's prompt_server.py: even with an
    explicit "no markdown code fences" instruction in the system prompt,
    roughly 40% of sampled responses still wrapped an otherwise perfectly
    valid JSON body in ```json ... ``` -- not a JSON-validity problem, just
    a wrapper json.loads chokes on). AnthropicInstructionParser never hits
    this path (it gets real structured output from the SDK), so this is
    purely a local-server-compatibility concern, but it belongs here rather
    than duplicated in LocalInstructionParser since both validate_* entry
    points are the shared "make raw text into valid JSON" layer."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return text


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
        data = json.loads(_strip_markdown_code_fence(raw_json))
    except json.JSONDecodeError as e:
        raise RecipeParseError(f"response was not valid JSON: {e}") from e

    model = build_recipe_model(channels)
    try:
        instance = model.model_validate(data)
    except pydantic.ValidationError as e:
        raise RecipeParseError(f"response did not match the expected recipe schema: {e}") from e

    return {name: getattr(instance, name) for name in channels}


GOAL_DIMS = ("loud_mean", "loud_std", "bright_mean", "bright_std", "pitch_mean", "pitch_std")
# Order matches features.py's extract_features_aggregated documented
# interleaving exactly: [mean(loudness), std(loudness), mean(brightness),
# std(brightness), mean(pitch), std(pitch)] -- so
# np.array([goal_dict[name] for name in GOAL_DIMS]) is directly usable
# as run_trajectory_control_loop's goal_features with no reordering.
# A tuple, not a list: every consumer (this module, AnthropicInstructionParser
# .parse_goal via model_dump() ordering, instruction_to_goal.py) only ever
# reads this order, never mutates it -- a tuple makes that tamper-proof.


class GoalFeatures(pydantic.BaseModel):
    """A goal expressed in FEATURE space (what the sound should measure
    like), not CV space -- unlike CVRecipe, this schema is fixed and
    never rebuilt per call: loudness/brightness/pitch mean+std are
    measured the same way regardless of which patch is running."""
    model_config = pydantic.ConfigDict(extra="forbid")
    loud_mean: float = pydantic.Field(
        ge=0.0, le=1.0, description="Overall loudness level. High = loud, low = quiet/near-silent."
    )
    loud_std: float = pydantic.Field(
        ge=0.0,
        le=1.0,
        description=(
            "How much loudness varies within the listening window (e.g. a pumping or "
            "gated sound). In practice rarely exceeds ~0.3; low = a steady, unchanging level."
        ),
    )
    bright_mean: float = pydantic.Field(
        ge=0.0,
        le=1.0,
        description="Tonal brightness/harshness (spectral centroid). High = bright/harsh/open-filter, low = dark/muffled/closed-filter.",
    )
    bright_std: float = pydantic.Field(
        ge=0.0,
        le=1.0,
        description=(
            "How much brightness actively sweeps or moves within the window (e.g. a filter "
            "sweep). In practice rarely exceeds ~0.3; low = a fixed tone color."
        ),
    )
    pitch_mean: float = pydantic.Field(
        ge=0.0, le=1.0, description="Perceived fundamental pitch, on a log scale from low (0) to high (1)."
    )
    pitch_std: float = pydantic.Field(
        ge=0.0,
        le=1.0,
        description=(
            "How much the pitch actively moves within the window (a sequence, an arpeggio, "
            "vibrato, a pitch sweep). In practice rarely exceeds ~0.3; low = a held, steady pitch."
        ),
    )


def validate_goal_json(raw_json: str) -> dict[str, float]:
    """Same validation shape as validate_recipe_json (malformed JSON,
    out-of-range value, missing/extra key all raise RecipeParseError with
    a specific message), against the fixed GoalFeatures schema instead of
    a per-call dynamic one."""
    try:
        data = json.loads(_strip_markdown_code_fence(raw_json))
    except json.JSONDecodeError as e:
        raise RecipeParseError(f"response was not valid JSON: {e}") from e

    try:
        instance = GoalFeatures.model_validate(data)
    except pydantic.ValidationError as e:
        raise RecipeParseError(f"response did not match the expected goal schema: {e}") from e

    return {name: getattr(instance, name) for name in GOAL_DIMS}
