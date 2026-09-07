import anthropic

from instruction_parser.base import InstructionParser, RecipeParseError
from instruction_parser.prompts import build_system_prompt
from instruction_parser.schema import build_recipe_model


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
