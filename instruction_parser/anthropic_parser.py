import anthropic

from instruction_parser.base import InstructionParser, RecipeParseError
from instruction_parser.schema import build_recipe_model

SYSTEM_PREAMBLE = (
    "You control a modular synthesizer via a set of continuous CV "
    "(control voltage) channels, each in the range [0, 1]. Given an "
    "instruction describing a desired sound, choose a value for every "
    "channel listed below to best match the instruction. Channels and "
    "what each one controls:\n"
)


def _build_system_prompt(channels: dict[str, str]) -> str:
    lines = [SYSTEM_PREAMBLE]
    for name, description in channels.items():
        lines.append(f"- {name}: {description}")
    return "\n".join(lines)


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
                system=_build_system_prompt(channels),
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
        return {name: parsed[name] for name in channels}
