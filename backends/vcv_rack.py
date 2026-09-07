# backends/vcv_rack.py
import queue
import yaml
import mido
import sounddevice as sd
import numpy as np

from backends.base import CVBackend


class AudioBlockTimeoutError(RuntimeError):
    """Raised by VCVRackBackend.read_audio_block() when no audio block
    arrives from the capture stream within the configured timeout, instead
    of leaking the underlying queue.Empty or blocking forever. A stalled
    audio source (e.g. VCV Rack's audio engine wedged, the loopback device
    disconnected) must surface as a clear, specific failure here rather than
    hanging the caller indefinitely -- which matters especially because
    callers of this method may be holding a gozer chip lease at the time."""


def load_channel_config(config_path: str) -> tuple[str, dict[str, int], dict[str, str]]:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    channel_cc = {name: entry["cc"] for name, entry in raw["channels"].items()}
    channel_descriptions = {
        name: entry.get("description", "") for name, entry in raw["channels"].items()
    }
    return raw["midi_port_name"], channel_cc, channel_descriptions


class VCVRackBackend(CVBackend):
    def __init__(
        self,
        config_path: str,
        sample_rate: int = 48000,
        block_size: int = 1024,
        read_timeout_s: float = 5.0,
    ):
        midi_port_name, self._channel_cc, self._channel_descriptions = load_channel_config(config_path)
        self._channel_names = list(self._channel_cc.keys())
        self._last_cv = {name: 0.5 for name in self._channel_names}
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._read_timeout_s = read_timeout_s

        self._midi_port = mido.open_output(midi_port_name)

        # Bounded to a single slot: the audio callback must never block, and
        # callers only ever want the most recent block, not a deep backlog.
        # Each callback invocation drops whatever stale block is waiting (if
        # any) and replaces it with the freshly captured one.
        self._audio_queue: queue.Queue = queue.Queue(maxsize=1)

        def _callback(indata, frames, time_info, status):
            block = indata.copy()
            try:
                self._audio_queue.get_nowait()  # drop the stale block if one is waiting
            except queue.Empty:
                pass
            try:
                self._audio_queue.put_nowait(block)
            except queue.Full:
                pass  # extremely unlikely race with another consumer; drop this block rather than block the audio thread

        pulse_device = next(
            i for i, d in enumerate(sd.query_devices())
            if d["name"] == "pulse" and d["max_input_channels"] > 0
        )
        self._stream = sd.InputStream(
            device=pulse_device,
            channels=2,
            samplerate=sample_rate,
            blocksize=block_size,
            callback=_callback,
        )
        self._stream.start()

    def channels(self) -> list[str]:
        return self._channel_names

    def channel_descriptions(self) -> dict[str, str]:
        return self._channel_descriptions

    def sample_rate(self) -> int:
        return self._sample_rate

    def block_size(self) -> int:
        return self._block_size

    def set_cv(self, channel: str, value: float) -> None:
        if channel not in self._channel_cc:
            raise KeyError(f"unknown channel: {channel}")
        cc_value = max(0, min(127, round(value * 127)))
        self._midi_port.send(
            mido.Message("control_change", channel=0, control=self._channel_cc[channel], value=cc_value)
        )
        self._last_cv[channel] = value

    def last_known_cv(self, channel: str) -> float:
        return self._last_cv[channel]

    def read_audio_block(self) -> np.ndarray:
        try:
            block = self._audio_queue.get(timeout=self._read_timeout_s)
        except queue.Empty:
            raise AudioBlockTimeoutError(
                f"no audio block arrived within {self._read_timeout_s}s -- "
                "the audio capture stream may have stalled (check the "
                "PipeWire loopback routing and that VCV Rack's audio engine "
                "is still running)"
            ) from None
        return np.mean(block, axis=1)  # mono-mix stereo capture

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()
        self._midi_port.close()
