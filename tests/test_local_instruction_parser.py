import json
from unittest.mock import MagicMock, patch

import pytest

from instruction_parser.base import RecipeParseError
from instruction_parser.local_parser import LocalInstructionParser

CHANNELS = {"vco_freq": "pitch", "vca_level": "loudness"}


def _mock_chat_response(content_str: str):
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content_str))]
    return response


def test_parse_recipe_returns_dict_from_valid_response():
    parser = LocalInstructionParser(base_url="http://localhost:8000/v1", model="local-model")
    with patch("instruction_parser.local_parser.openai.OpenAI") as MockClient:
        mock_client = MockClient.return_value
        mock_client.chat.completions.create.return_value = _mock_chat_response(
            json.dumps({"vco_freq": 0.3, "vca_level": 0.7})
        )
        result = parser.parse_recipe("make it deep and quiet", CHANNELS)
    assert result == {"vco_freq": 0.3, "vca_level": 0.7}


def test_parse_recipe_uses_configured_base_url_and_model():
    parser = LocalInstructionParser(base_url="http://myhost:1234/v1", model="my-local-model")
    with patch("instruction_parser.local_parser.openai.OpenAI") as MockClient:
        mock_client = MockClient.return_value
        mock_client.chat.completions.create.return_value = _mock_chat_response(
            json.dumps({"vco_freq": 0.5, "vca_level": 0.5})
        )
        parser.parse_recipe("anything", CHANNELS)
    _, init_kwargs = MockClient.call_args
    assert init_kwargs["base_url"] == "http://myhost:1234/v1"
    _, call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs["model"] == "my-local-model"


def test_parse_recipe_raises_on_malformed_local_model_output():
    # A real, expected failure mode for a local model that doesn't
    # strictly conform to the requested schema -- confirm it's caught and
    # re-raised as RecipeParseError, not left as a raw json/pydantic error.
    parser = LocalInstructionParser(base_url="http://localhost:8000/v1", model="local-model")
    with patch("instruction_parser.local_parser.openai.OpenAI") as MockClient:
        mock_client = MockClient.return_value
        mock_client.chat.completions.create.return_value = _mock_chat_response(
            "I think vco_freq should be low and vca_level should be high"  # not JSON at all
        )
        with pytest.raises(RecipeParseError):
            parser.parse_recipe("make it deep and quiet", CHANNELS)


def test_parse_recipe_raises_recipe_parse_error_on_empty_choices():
    # A real, non-hypothetical local-server failure mode: an empty
    # `choices` list (e.g. a refusal at the transport level). Must surface
    # as RecipeParseError, not a raw IndexError.
    parser = LocalInstructionParser(base_url="http://localhost:8000/v1", model="local-model")
    with patch("instruction_parser.local_parser.openai.OpenAI") as MockClient:
        mock_client = MockClient.return_value
        response = MagicMock()
        response.choices = []
        mock_client.chat.completions.create.return_value = response
        with pytest.raises(RecipeParseError):
            parser.parse_recipe("make it deep and quiet", CHANNELS)


def test_parse_recipe_raises_recipe_parse_error_on_none_content():
    # Another real, non-hypothetical case: a tool_calls-only message or an
    # empty completion leaves message.content as None (openai's own
    # ChatCompletionMessage.content is typed Optional[str]). Must surface
    # as RecipeParseError, not a raw TypeError from json.loads(None).
    parser = LocalInstructionParser(base_url="http://localhost:8000/v1", model="local-model")
    with patch("instruction_parser.local_parser.openai.OpenAI") as MockClient:
        mock_client = MockClient.return_value
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content=None), finish_reason="tool_calls")]
        mock_client.chat.completions.create.return_value = response
        with pytest.raises(RecipeParseError):
            parser.parse_recipe("make it deep and quiet", CHANNELS)


def test_parse_recipe_sends_correct_json_schema_for_varying_channel_counts():
    for channels in ({"a": "desc a", "b": "desc b"}, {f"ch{i}": f"description {i}" for i in range(8)}):
        parser = LocalInstructionParser(base_url="http://localhost:8000/v1", model="local-model")
        with patch("instruction_parser.local_parser.openai.OpenAI") as MockClient:
            mock_client = MockClient.return_value
            recipe = {name: 0.5 for name in channels}
            mock_client.chat.completions.create.return_value = _mock_chat_response(json.dumps(recipe))
            parser.parse_recipe("anything", channels)
        _, call_kwargs = mock_client.chat.completions.create.call_args
        schema = call_kwargs["response_format"]["json_schema"]["schema"]
        assert set(schema["properties"].keys()) == set(channels.keys())
        assert set(schema["required"]) == set(channels.keys())


def test_parse_recipe_includes_channel_descriptions_in_system_message():
    channels = {"vco_freq": "pitch control", "vca_level": "loudness control"}
    parser = LocalInstructionParser(base_url="http://localhost:8000/v1", model="local-model")
    with patch("instruction_parser.local_parser.openai.OpenAI") as MockClient:
        mock_client = MockClient.return_value
        mock_client.chat.completions.create.return_value = _mock_chat_response(
            json.dumps({"vco_freq": 0.5, "vca_level": 0.5})
        )
        parser.parse_recipe("anything", channels)
    _, call_kwargs = mock_client.chat.completions.create.call_args
    system_message = next(m["content"] for m in call_kwargs["messages"] if m["role"] == "system")
    assert "pitch control" in system_message
    assert "loudness control" in system_message
