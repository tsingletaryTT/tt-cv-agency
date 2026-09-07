from unittest.mock import MagicMock, patch

import pytest

from instruction_parser.anthropic_parser import AnthropicInstructionParser
from instruction_parser.base import RecipeParseError

CHANNELS = {"vco_freq": "pitch", "vca_level": "loudness"}


def _mock_parsed_response(**field_values):
    response = MagicMock()
    response.parsed_output = MagicMock(**field_values)
    # model_dump lets the parser extract a plain dict without depending on
    # the mock's own attribute set matching pydantic's real interface
    response.parsed_output.model_dump.return_value = field_values
    return response


def test_parse_recipe_returns_dict_from_parsed_output():
    parser = AnthropicInstructionParser()
    with patch("instruction_parser.anthropic_parser.anthropic.Anthropic") as MockClient:
        mock_client = MockClient.return_value
        mock_client.messages.parse.return_value = _mock_parsed_response(
            vco_freq=0.2, vca_level=0.8
        )
        result = parser.parse_recipe("make it deep and quiet", CHANNELS)
    assert result == {"vco_freq": 0.2, "vca_level": 0.8}


def test_parse_recipe_uses_configured_model():
    parser = AnthropicInstructionParser(model="claude-sonnet-5")
    with patch("instruction_parser.anthropic_parser.anthropic.Anthropic") as MockClient:
        mock_client = MockClient.return_value
        mock_client.messages.parse.return_value = _mock_parsed_response(
            vco_freq=0.5, vca_level=0.5
        )
        parser.parse_recipe("anything", CHANNELS)
    _, kwargs = mock_client.messages.parse.call_args
    assert kwargs["model"] == "claude-sonnet-5"


def test_parse_recipe_includes_channel_descriptions_in_system_prompt():
    parser = AnthropicInstructionParser()
    with patch("instruction_parser.anthropic_parser.anthropic.Anthropic") as MockClient:
        mock_client = MockClient.return_value
        mock_client.messages.parse.return_value = _mock_parsed_response(
            vco_freq=0.5, vca_level=0.5
        )
        parser.parse_recipe("anything", CHANNELS)
    _, kwargs = mock_client.messages.parse.call_args
    assert "pitch" in kwargs["system"]
    assert "loudness" in kwargs["system"]


def test_parse_recipe_raises_recipe_parse_error_on_refusal():
    # A refusal (or any other non-normal completion) leaves parsed_output
    # unset. This must surface as a clear RecipeParseError naming the
    # stop_reason, not an AttributeError from calling .model_dump() on None.
    parser = AnthropicInstructionParser()
    with patch("instruction_parser.anthropic_parser.anthropic.Anthropic") as MockClient:
        mock_client = MockClient.return_value
        response = MagicMock()
        response.parsed_output = None
        response.stop_reason = "refusal"
        mock_client.messages.parse.return_value = response
        with pytest.raises(RecipeParseError, match="refusal"):
            parser.parse_recipe("anything", CHANNELS)
