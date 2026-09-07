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
