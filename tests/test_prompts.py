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
