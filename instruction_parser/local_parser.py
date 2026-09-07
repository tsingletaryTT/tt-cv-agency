import openai

from instruction_parser.base import InstructionParser, RecipeParseError
from instruction_parser.prompts import build_system_prompt
from instruction_parser.schema import validate_recipe_json


def _build_json_schema(channels: dict[str, str]) -> dict:
    return {
        "type": "object",
        "properties": {
            name: {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": description}
            for name, description in channels.items()
        },
        "required": list(channels.keys()),
        "additionalProperties": False,
    }


class LocalInstructionParser(InstructionParser):
    """InstructionParser backed by any OpenAI-compatible chat completions
    server (vLLM, Ollama, llama.cpp server, LM Studio, ...), reached via
    the `openai` SDK pointed at a configurable base_url. No default
    server is assumed -- base_url and model are required."""

    def __init__(self, base_url: str, model: str, api_key: str = "not-needed"):
        self._base_url = base_url
        self._model = model
        self._api_key = api_key

    def parse_recipe(self, instruction: str, channels: dict[str, str]) -> dict[str, float]:
        client = openai.OpenAI(base_url=self._base_url, api_key=self._api_key)
        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": build_system_prompt(channels)},
                    {"role": "user", "content": instruction},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "cv_recipe", "schema": _build_json_schema(channels)},
                },
            )
        except Exception as e:
            raise RecipeParseError(f"local model API call failed: {e}") from e

        content = response.choices[0].message.content
        # Local servers vary widely in how strictly they honor a requested
        # JSON schema (vLLM/Ollama/llama.cpp server/LM Studio all differ)
        # -- never assume the response is valid, schema-conforming JSON
        # just because it was requested. AnthropicInstructionParser gets
        # that guarantee for free from Pydantic-backed structured output;
        # here it must be checked explicitly, so delegate to the same
        # validate_recipe_json used there rather than reimplementing any
        # of its JSON-parsing/validation logic.
        return validate_recipe_json(content, channels)
