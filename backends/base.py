from abc import ABC, abstractmethod
import numpy as np


class CVBackend(ABC):
    @abstractmethod
    def set_cv(self, channel: str, value: float) -> None:
        ...

    @abstractmethod
    def read_audio_block(self) -> np.ndarray:
        ...

    @abstractmethod
    def channels(self) -> list[str]:
        ...

    @abstractmethod
    def last_known_cv(self, channel: str) -> float:
        ...


class FakeCVBackend(CVBackend):
    """In-memory CVBackend for tests. Returns a fixed audio block regardless
    of CV state -- callers that need CV-dependent audio should subclass and
    override read_audio_block."""

    def __init__(self, channel_names: list[str], audio_block: np.ndarray | None = None):
        self._channels = list(channel_names)
        self._last_cv = {name: 0.5 for name in self._channels}
        self._audio_block = audio_block if audio_block is not None else np.zeros(512, dtype=np.float32)
        self.set_cv_calls: list[tuple[str, float]] = []

    def channels(self) -> list[str]:
        return self._channels

    def set_cv(self, channel: str, value: float) -> None:
        if channel not in self._last_cv:
            raise KeyError(f"unknown channel: {channel}")
        self._last_cv[channel] = value
        self.set_cv_calls.append((channel, value))

    def last_known_cv(self, channel: str) -> float:
        return self._last_cv[channel]

    def read_audio_block(self) -> np.ndarray:
        return self._audio_block
