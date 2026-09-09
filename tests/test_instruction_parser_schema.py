import json
import numpy as np
import pytest
from instruction_parser.base import RecipeParseError
from instruction_parser.schema import build_recipe_model, validate_recipe_json
from features import extract_features_aggregated

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


def test_goal_dims_matches_goal_features_field_order():
    # GoalFeatures.model_dump() (relied on by AnthropicInstructionParser
    # .parse_goal) iterates fields in declaration order -- GOAL_DIMS must
    # pin that same order, or a goal dict's values would silently land in
    # the wrong np.array slot when zipped against GOAL_DIMS elsewhere.
    assert list(GoalFeatures.model_fields.keys()) == list(GOAL_DIMS)


def test_goal_dims_matches_features_py_interleaved_order():
    # Real behavioral proof (not a second hardcoded copy of the same
    # literal) that GOAL_DIMS's order matches extract_features_aggregated's
    # actual interleaving: build blocks where ONLY loudness varies
    # block-to-block (same amplitude-modulation pattern already used in
    # tests/test_features.py's test_extract_features_aggregated_detects_modulation),
    # holding frequency (and therefore brightness/pitch) constant, and
    # confirm the resulting nonzero std lands at GOAL_DIMS's "loud_std"
    # index specifically, not at "bright_std" or "pitch_std".
    sample_rate = 48000
    t = np.arange(1024) / sample_rate
    modulated_blocks = [
        (0.1 + 0.4 * (i % 2)) * np.sin(2 * np.pi * 220 * t) for i in range(8)
    ]
    result = extract_features_aggregated(modulated_blocks, sample_rate)

    loud_std_value = result[GOAL_DIMS.index("loud_std")]
    bright_std_value = result[GOAL_DIMS.index("bright_std")]
    pitch_std_value = result[GOAL_DIMS.index("pitch_std")]

    assert loud_std_value > 0.01
    assert bright_std_value < loud_std_value
    assert pitch_std_value < loud_std_value


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
