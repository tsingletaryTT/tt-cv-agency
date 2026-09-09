import anthropic

from instruction_parser.base import InstructionParser, RecipeParseError
from instruction_parser.prompts import build_goal_system_prompt, build_system_prompt
from instruction_parser.schema import GoalFeatures, build_recipe_model


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
                system=build_system_prompt(channels),
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
        # Deliberately no validate_recipe_json call here, unlike the local
        # path: client.messages.parse's own output_format enforcement already
        # validates range/required-keys before parsed_output is ever
        # populated, so re-validating would be redundant.
        return {name: parsed[name] for name in channels}

    def parse_goal(self, instruction: str) -> dict[str, float]:
        client = anthropic.Anthropic()
        try:
            response = client.messages.parse(
                model=self._model,
                # Higher than parse_recipe's 1024: on claude-opus-5, adaptive
                # thinking is on by default and its tokens count against
                # max_tokens, so a real call could exhaust the budget on
                # reasoning before ever emitting the 6-float JSON response.
                # 4096 gives plenty of headroom for that plus the response.
                max_tokens=4096,
                system=build_goal_system_prompt(),
                messages=[{"role": "user", "content": instruction}],
                output_format=GoalFeatures,
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

        # Deliberately no validate_goal_json call here, same reasoning as
        # parse_recipe: client.messages.parse's own output_format
        # enforcement already validates range/required-keys before
        # parsed_output is ever populated.
        return response.parsed_output.model_dump()
