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

    @abstractmethod
    def parse_goal(self, instruction: str) -> dict[str, float]:
        """Given a natural-language instruction, return a target value in
        [0, 1] for each of schema.GOAL_DIMS (loud_mean, loud_std,
        bright_mean, bright_std, pitch_mean, pitch_std) -- a goal in
        FEATURE space (what the sound should measure like), not CV space.
        Unlike parse_recipe, this schema never varies by patch. Raises
        RecipeParseError on the same failure classes as parse_recipe
        (malformed response, out-of-range value, missing/extra key,
        refusal/empty completion)."""
        ...
