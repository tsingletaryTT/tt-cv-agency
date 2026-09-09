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

def test_validate_recipe_json_rejects_extra_channel():
    # The interface's "no more, no fewer" contract has two halves -- this
    # covers the half (extra/unexpected keys) that had no test despite the
    # underlying extra="forbid" enforcement already being correct (caught
    # in Task 3's review as a coverage gap, not a behavior defect).
    raw = json.dumps({"vco_freq": 0.5, "vca_level": 0.5, "extra_key": 0.1})
    with pytest.raises(RecipeParseError):
        validate_recipe_json(raw, CHANNELS)

def test_build_recipe_model_works_for_different_channel_counts():
    # Confirms neither the schema builder nor the validator has a hidden
    # assumption about a fixed (e.g. 8-channel) set.
    three = {"a": "", "b": "", "c": ""}
    eight = {f"ch{i}": "" for i in range(8)}
    assert set(build_recipe_model(three).model_fields.keys()) == {"a", "b", "c"}
    assert len(build_recipe_model(eight).model_fields) == 8


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
