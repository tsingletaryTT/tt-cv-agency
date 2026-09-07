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
