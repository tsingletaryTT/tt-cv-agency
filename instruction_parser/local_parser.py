import openai

from instruction_parser.base import InstructionParser, RecipeParseError
from instruction_parser.prompts import build_goal_system_prompt, build_system_prompt
from instruction_parser.schema import GoalFeatures, validate_goal_json, validate_recipe_json


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
                # "strict" is deliberately NOT set on the json_schema envelope
                # below -- support for it is uneven across OpenAI-compatible
                # servers (vLLM/Ollama/llama.cpp server/LM Studio), so schema
                # conformance is enforced downstream by validate_recipe_json
                # instead of relied upon here.
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "cv_recipe", "schema": _build_json_schema(channels)},
                },
                # Parity with AnthropicInstructionParser's max_tokens=1024 --
                # guards against a local server truncating the JSON response,
                # which would otherwise surface as an unhelpful generic "not
                # valid JSON" error instead of hinting at truncation.
                max_tokens=1024,
            )
        except Exception as e:
            raise RecipeParseError(f"local model API call failed: {e}") from e

        if not response.choices or response.choices[0].message.content is None:
            # A real, non-hypothetical failure mode for a local server: a
            # refusal, a tool_calls-only message, or an empty completion all
            # leave no usable text here (openai's own ChatCompletionMessage.
            # content is typed Optional[str]). Surface this as the same
            # RecipeParseError the InstructionParser interface promises,
            # instead of an IndexError/TypeError leaking out of this method.
            finish_reason = response.choices[0].finish_reason if response.choices else None
            raise RecipeParseError(
                f"local model returned no usable message content (finish_reason={finish_reason!r})"
            )

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

    def parse_goal(self, instruction: str) -> dict[str, float]:
        client = openai.OpenAI(base_url=self._base_url, api_key=self._api_key)
        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": build_goal_system_prompt()},
                    {"role": "user", "content": instruction},
                ],
                # No hand-rolled schema dict needed here (unlike
                # parse_recipe's _build_json_schema(channels)) -- GoalFeatures
                # is a fixed, already-fully-specified Pydantic model
                # (Field(ge=0,le=1) on every field, extra="forbid"), so its
                # own .model_json_schema() already produces the right
                # minimum/maximum/required/additionalProperties shape with
                # nothing to hand-maintain in a second place.
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "goal_features", "schema": GoalFeatures.model_json_schema()},
                },
                max_tokens=1024,
            )
        except Exception as e:
            raise RecipeParseError(f"local model API call failed: {e}") from e

        if not response.choices or response.choices[0].message.content is None:
            finish_reason = response.choices[0].finish_reason if response.choices else None
            raise RecipeParseError(
                f"local model returned no usable message content (finish_reason={finish_reason!r})"
            )

        return validate_goal_json(response.choices[0].message.content)
