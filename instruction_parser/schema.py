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
