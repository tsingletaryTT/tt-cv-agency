from abc import ABC, abstractmethod


class RecipeParseError(Exception):
    """Raised when an InstructionParser's underlying LLM response can't be
    turned into a valid CV recipe -- malformed JSON, a value outside
    [0, 1], or a missing/extra channel key. Carries a specific, actionable
    message naming what was wrong, not a generic parse failure."""


class InstructionParser(ABC):
    @abstractmethod
    def parse_recipe(self, instruction: str, channels: dict[str, str]) -> dict[str, float]:
        """Given a natural-language instruction and a mapping of channel
        name -> human-readable description (from the current patch's
        config), return a value in [0, 1] for every key in `channels`, no
        more, no fewer. Raises RecipeParseError if the underlying model's
        response can't be validated into exactly that shape."""
        ...
