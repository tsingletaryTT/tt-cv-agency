# backends/vcv_rack.py
import queue
import yaml
import mido
import sounddevice as sd
import numpy as np

from backends.base import CVBackend


def load_channel_config(config_path: str) -> tuple[str, dict[str, int]]:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    channel_cc = {name: entry["cc"] for name, entry in raw["channels"].items()}
    return raw["midi_port_name"], channel_cc


class VCVRackBackend(CVBackend):
    def __init__(self, config_path: str, sample_rate: int = 48000, block_size: int = 1024):
        midi_port_name, self._channel_cc = load_channel_config(config_path)
        self._channel_names = list(self._channel_cc.keys())
        self._last_cv = {name: 0.5 for name in self._channel_names}

        self._midi_port = mido.open_output(midi_port_name)

        self._audio_queue: queue.Queue = queue.Queue()

        def _callback(indata, frames, time_info, status):
            self._audio_queue.put(indata.copy())

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
        block = self._audio_queue.get()
        return np.mean(block, axis=1)  # mono-mix stereo capture

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()
        self._midi_port.close()
