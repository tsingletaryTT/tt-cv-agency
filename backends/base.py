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

    @abstractmethod
    def sample_rate(self) -> int:
        """The backend's real audio sample rate, in Hz. Callers should pull
        this from the backend rather than independently hardcoding it, so a
        feature-extraction routine (features.py's estimate_pitch, etc.) is
        never fed a sample_rate that silently disagrees with what the
        backend actually captured its audio at."""
        ...

    @abstractmethod
    def block_size(self) -> int:
        """The backend's real audio block size, in samples. Callers should
        pull this from the backend rather than independently hardcoding it
        -- see sample_rate()'s docstring for why this matters."""
        ...


class FakeCVBackend(CVBackend):
    """In-memory CVBackend for tests. Returns a fixed audio block regardless
    of CV state -- callers that need CV-dependent audio should subclass and
    override read_audio_block."""

    def __init__(
        self,
        channel_names: list[str],
        audio_block: np.ndarray | None = None,
        sample_rate: int = 48000,
        block_size: int = 1024,
    ):
        self._channels = list(channel_names)
        self._last_cv = {name: 0.5 for name in self._channels}
        self._audio_block = audio_block if audio_block is not None else np.zeros(512, dtype=np.float32)
        self.set_cv_calls: list[tuple[str, float]] = []
        self._sample_rate = sample_rate
        self._block_size = block_size

    def channels(self) -> list[str]:
        return self._channels

    def sample_rate(self) -> int:
        return self._sample_rate

    def block_size(self) -> int:
        return self._block_size

    def set_cv(self, channel: str, value: float) -> None:
        if channel not in self._last_cv:
            raise KeyError(f"unknown channel: {channel}")
        self._last_cv[channel] = value
        self.set_cv_calls.append((channel, value))

    def last_known_cv(self, channel: str) -> float:
        return self._last_cv[channel]

    def read_audio_block(self) -> np.ndarray:
        return self._audio_block
