# TT-CV-Agency Inference Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put a small model on Tenstorrent hardware in the actual decision path of a closed-loop CV controller: read live VCV Rack audio, extract 3 features (loudness, brightness, pitch), ask a TTNN-run model what CV settings would produce a target feature vector, write those CV values back via MIDI-CAT, and repeat until the live audio converges on the goal.

**Architecture:** A `CVBackend` abstraction (today: `VCVRackBackend` over MIDI-CAT + a PipeWire loopback) sits below a feature-extraction layer, a small inverse-regression model (trained in PyTorch, run at inference time via `ttnn` on one `gozer`-leased TT chip), and a perceive-decide-act control loop that repeatedly measures real audio and steps CV settings toward the goal.

**Tech Stack:** Python 3.12, `mido` (MIDI I/O), `sounddevice`/PortAudio (audio capture), `numpy`/`scipy` (feature extraction), `torch` (training only, CPU), `ttnn` (inference, TT hardware), `pytest`, `PyYAML`. All of these are already installed in this environment — confirmed via `importlib.util.find_spec` and, for `ttnn`, an actual verified forward pass (see Task 6) — no dependency installation is needed in any task below.

**Spec:** `docs/superpowers/specs/2026-09-06-tt-cv-agency-inference-loop-design.md`

## Global Constraints

- Every touch of `ttnn` — including a bare `import ttnn` — must happen inside a `gozer run --chips 1 --who "claude:tt-cv-agency" --reason "<why>" -- <command>` invocation (or an active `gozer acquire`'d lease). Never import `ttnn` unleased, even to check something works, per this machine's hardware convention (`~/CLAUDE.md`).
- `CVBackend.set_cv(channel: str, value: float)` takes `value` in `[0.0, 1.0]`; the backend is responsible for converting to whatever units the underlying I/O needs (here: a MIDI CC integer `round(value * 127)`).
- Channel identity and MIDI CC numbers live only in `configs/bridge_test.yaml`, never hardcoded in `.py` files — this is what makes the code portable to a future hardware backend/patch.
- All new code lives under `~/code/vcv-cv-harness/`, following the file layout in the spec.
- MIDI output port name (verified, Task 2): `'Midi Through:Midi Through Port-0 14:0'`. Audio input: `sounddevice` device whose `name == 'pulse'` (PortAudio's Pulse host API only exposes this generic pseudo-device; it follows the system's default Pulse source, which must be `vcv_loop.monitor` — see Task 2 for how the backend verifies this itself rather than assuming it).

---

## Task 1: `CVBackend` interface + a fake backend for testing

**Files:**
- Create: `backends/__init__.py` (empty)
- Create: `backends/base.py`
- Test: `tests/test_base.py`

**Interfaces:**
- Produces: `class CVBackend` (abstract base, methods `set_cv(channel: str, value: float) -> None`, `read_audio_block() -> np.ndarray`, `channels() -> list[str]`, `last_known_cv(channel: str) -> float`), and a `FakeCVBackend` test double implementing it in-memory (records `set_cv` calls, returns a caller-supplied fixed/callable audio block from `read_audio_block`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_base.py
import numpy as np
import pytest
from backends.base import CVBackend, FakeCVBackend


def test_fake_backend_records_set_cv_and_tracks_last_known():
    backend = FakeCVBackend(channel_names=["a", "b"])
    assert backend.channels() == ["a", "b"]
    assert backend.last_known_cv("a") == 0.5  # documented default before any set_cv
    backend.set_cv("a", 0.9)
    assert backend.last_known_cv("a") == 0.9
    assert backend.last_known_cv("b") == 0.5


def test_fake_backend_rejects_unknown_channel():
    backend = FakeCVBackend(channel_names=["a"])
    with pytest.raises(KeyError):
        backend.set_cv("nonexistent", 0.5)


def test_fake_backend_read_audio_block_returns_configured_block():
    block = np.ones(512, dtype=np.float32) * 0.25
    backend = FakeCVBackend(channel_names=["a"], audio_block=block)
    result = backend.read_audio_block()
    assert np.array_equal(result, block)


def test_cv_backend_is_abstract():
    with pytest.raises(TypeError):
        CVBackend()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/code/vcv-cv-harness && python3 -m pytest tests/test_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backends'`

- [ ] **Step 3: Write minimal implementation**

```python
# backends/base.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/code/vcv-cv-harness && python3 -m pytest tests/test_base.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
cd ~/code/vcv-cv-harness
git add backends/__init__.py backends/base.py tests/test_base.py
git commit -m "Add CVBackend interface and FakeCVBackend test double"
```

---

## Task 2: Feature extraction

**Files:**
- Create: `features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: nothing from other tasks (pure functions over `np.ndarray` audio blocks).
- Produces: `rms(block: np.ndarray) -> float`, `spectral_centroid(block: np.ndarray, sample_rate: int) -> float`, `estimate_pitch(block: np.ndarray, sample_rate: int, fmin: float = 20.0, fmax: float = 4000.0) -> float`, `extract_features(block: np.ndarray, sample_rate: int) -> np.ndarray` (returns a length-3 array `[loudness, brightness, pitch]`, each normalized to roughly `[0, 1]`), and the normalization constants `LOUDNESS_REF_RMS = 0.4`, `BRIGHTNESS_REF_HZ = 12000.0` (half Nyquist at 48kHz — a workable ceiling for a sine-ish VCO's centroid), `PITCH_LOG_MIN_HZ = 20.0`, `PITCH_LOG_MAX_HZ = 4000.0` (log-scale, since pitch perception and VCO tracking are both roughly logarithmic).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features.py
import numpy as np
from features import rms, spectral_centroid, estimate_pitch, extract_features


def test_rms_of_silence_is_zero():
    assert rms(np.zeros(512, dtype=np.float32)) == 0.0


def test_rms_of_known_sine_matches_expected_amplitude():
    # A sine of amplitude A has RMS = A / sqrt(2).
    t = np.arange(4096) / 48000.0
    sine = 0.8 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    result = rms(sine)
    assert abs(result - 0.8 / np.sqrt(2)) < 0.01


def test_spectral_centroid_of_low_sine_is_near_its_frequency():
    t = np.arange(4096) / 48000.0
    sine = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    centroid = spectral_centroid(sine, sample_rate=48000)
    assert abs(centroid - 440.0) < 50.0


def test_spectral_centroid_of_higher_sine_is_higher():
    t = np.arange(4096) / 48000.0
    low = np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    high = np.sin(2 * np.pi * 2000.0 * t).astype(np.float32)
    assert spectral_centroid(high, 48000) > spectral_centroid(low, 48000)


def test_estimate_pitch_recovers_known_frequency():
    t = np.arange(4096) / 48000.0
    sine = np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    pitch = estimate_pitch(sine, sample_rate=48000)
    assert abs(pitch - 220.0) < 5.0


def test_extract_features_returns_length_3_array_in_unit_range():
    t = np.arange(4096) / 48000.0
    sine = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    feats = extract_features(sine, sample_rate=48000)
    assert feats.shape == (3,)
    assert np.all(feats >= 0.0) and np.all(feats <= 1.0)


def test_extract_features_of_silence_has_zero_loudness():
    feats = extract_features(np.zeros(4096, dtype=np.float32), sample_rate=48000)
    assert feats[0] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/code/vcv-cv-harness && python3 -m pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'features'`

- [ ] **Step 3: Write minimal implementation**

```python
# features.py
import numpy as np

LOUDNESS_REF_RMS = 0.4
BRIGHTNESS_REF_HZ = 12000.0
PITCH_LOG_MIN_HZ = 20.0
PITCH_LOG_MAX_HZ = 4000.0


def rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))


def spectral_centroid(block: np.ndarray, sample_rate: int) -> float:
    windowed = block.astype(np.float64) * np.hanning(len(block))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(block), d=1.0 / sample_rate)
    total = spectrum.sum()
    if total <= 0:
        return 0.0
    return float(np.sum(freqs * spectrum) / total)


def estimate_pitch(block: np.ndarray, sample_rate: int, fmin: float = 20.0, fmax: float = 4000.0) -> float:
    x = block.astype(np.float64)
    x = x - x.mean()
    if np.allclose(x, 0.0):
        return 0.0
    corr = np.correlate(x, x, mode="full")
    corr = corr[len(corr) // 2:]  # keep non-negative lags
    lag_min = int(sample_rate / fmax)
    lag_max = min(int(sample_rate / fmin), len(corr) - 1)
    if lag_max <= lag_min:
        return 0.0
    search = corr[lag_min:lag_max]
    if len(search) == 0 or search.max() <= 0:
        return 0.0
    best_lag = lag_min + int(np.argmax(search))
    return float(sample_rate / best_lag)


def extract_features(block: np.ndarray, sample_rate: int) -> np.ndarray:
    loudness = min(rms(block) / LOUDNESS_REF_RMS, 1.0)
    brightness = min(spectral_centroid(block, sample_rate) / BRIGHTNESS_REF_HZ, 1.0)
    pitch_hz = estimate_pitch(block, sample_rate, fmin=PITCH_LOG_MIN_HZ, fmax=PITCH_LOG_MAX_HZ)
    if pitch_hz <= 0:
        pitch_norm = 0.0
    else:
        pitch_hz = max(PITCH_LOG_MIN_HZ, min(pitch_hz, PITCH_LOG_MAX_HZ))
        pitch_norm = (np.log2(pitch_hz) - np.log2(PITCH_LOG_MIN_HZ)) / (
            np.log2(PITCH_LOG_MAX_HZ) - np.log2(PITCH_LOG_MIN_HZ)
        )
    return np.array([loudness, brightness, pitch_norm], dtype=np.float64)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/code/vcv-cv-harness && python3 -m pytest tests/test_features.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
cd ~/code/vcv-cv-harness
git add features.py tests/test_features.py
git commit -m "Add feature extraction: loudness, brightness, pitch"
```

---

## Task 3: `configs/bridge_test.yaml` + `VCVRackBackend`

**Files:**
- Create: `configs/bridge_test.yaml`
- Create: `backends/vcv_rack.py`
- Test: `tests/test_vcv_rack_backend.py`

**Interfaces:**
- Consumes: `CVBackend` (Task 1).
- Produces: `class VCVRackBackend(CVBackend)`, constructed as `VCVRackBackend(config_path: str, midi_port_name: str = 'Midi Through:Midi Through Port-0 14:0', sample_rate: int = 48000, block_size: int = 1024)`. Internally uses `mido.open_output(midi_port_name)` for `set_cv`, and a persistent `sounddevice.InputStream` (device whose name is `'pulse'`) with a callback pushing blocks into a `queue.Queue`, popped by `read_audio_block`.

This config records the mappings already created by hand in `bridge_test.vcv`'s MIDI-CAT (verified working end-to-end in Phase 1 — see `CLAUDE.md`'s "RESOLVED" section): CC1→VCO Frequency, CC2→VCO FM, CC3→VCA Level.

- [ ] **Step 1: Write the config file**

```yaml
# configs/bridge_test.yaml
midi_port_name: "Midi Through:Midi Through Port-0 14:0"
channels:
  vco_freq:
    cc: 1
  vco_fm:
    cc: 2
  vca_level:
    cc: 3
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_vcv_rack_backend.py
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from backends.vcv_rack import VCVRackBackend, load_channel_config


def test_load_channel_config_reads_cc_numbers(tmp_path):
    config_path = tmp_path / "test.yaml"
    config_path.write_text(
        "midi_port_name: 'Fake Port'\n"
        "channels:\n"
        "  chan_a:\n"
        "    cc: 5\n"
        "  chan_b:\n"
        "    cc: 9\n"
    )
    midi_port_name, channel_cc = load_channel_config(str(config_path))
    assert midi_port_name == "Fake Port"
    assert channel_cc == {"chan_a": 5, "chan_b": 9}


@patch("backends.vcv_rack.sd.InputStream")
@patch("backends.vcv_rack.mido.open_output")
def test_set_cv_sends_control_change_with_scaled_value(mock_open_output, mock_input_stream, tmp_path):
    config_path = tmp_path / "test.yaml"
    config_path.write_text(
        "midi_port_name: 'Fake Port'\nchannels:\n  vca_level:\n    cc: 3\n"
    )
    mock_port = MagicMock()
    mock_open_output.return_value = mock_port

    backend = VCVRackBackend(config_path=str(config_path))
    backend.set_cv("vca_level", 0.5)

    assert mock_port.send.call_count == 1
    sent_msg = mock_port.send.call_args[0][0]
    assert sent_msg.type == "control_change"
    assert sent_msg.control == 3
    assert sent_msg.value == round(0.5 * 127)


@patch("backends.vcv_rack.sd.InputStream")
@patch("backends.vcv_rack.mido.open_output")
def test_set_cv_rejects_unknown_channel(mock_open_output, mock_input_stream, tmp_path):
    config_path = tmp_path / "test.yaml"
    config_path.write_text("midi_port_name: 'Fake Port'\nchannels:\n  a:\n    cc: 1\n")
    backend = VCVRackBackend(config_path=str(config_path))
    with pytest.raises(KeyError):
        backend.set_cv("nonexistent", 0.5)


@patch("backends.vcv_rack.sd.InputStream")
@patch("backends.vcv_rack.mido.open_output")
def test_channels_returns_config_keys(mock_open_output, mock_input_stream, tmp_path):
    config_path = tmp_path / "test.yaml"
    config_path.write_text(
        "midi_port_name: 'Fake Port'\nchannels:\n  a:\n    cc: 1\n  b:\n    cc: 2\n"
    )
    backend = VCVRackBackend(config_path=str(config_path))
    assert backend.channels() == ["a", "b"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd ~/code/vcv-cv-harness && python3 -m pytest tests/test_vcv_rack_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backends.vcv_rack'`

- [ ] **Step 4: Write minimal implementation**

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd ~/code/vcv-cv-harness && python3 -m pytest tests/test_vcv_rack_backend.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Manually verify against the live patch**

This step needs the real environment (VCV Rack running `bridge_test.vcv`, PulseAudio/PipeWire default sink+source set to `vcv_loop`/`vcv_loop.monitor` per `CLAUDE.md`'s documented routing fix) — it's the first real (non-mocked) exercise of this class, so confirm it before building on top of it:

```bash
cd ~/code/vcv-cv-harness
pactl set-default-sink vcv_loop
pactl set-default-source vcv_loop.monitor
python3 -c "
from backends.vcv_rack import VCVRackBackend
import time
b = VCVRackBackend('configs/bridge_test.yaml')
b.set_cv('vca_level', 1.0)
time.sleep(0.5)
block = b.read_audio_block()
print('RMS at level=1.0:', (block.astype(float)**2).mean()**0.5)
b.set_cv('vca_level', 0.0)
time.sleep(0.5)
block = b.read_audio_block()
print('RMS at level=0.0:', (block.astype(float)**2).mean()**0.5)
b.close()
"
```

Expected: the first RMS is clearly larger than the second (matching the Phase 1 result: RMS ~9000+ at high int16 scale, or proportionally similar in this float32 capture — the exact number matters less than level=1.0 giving a much larger RMS than level=0.0).

- [ ] **Step 7: Commit**

```bash
cd ~/code/vcv-cv-harness
git add configs/bridge_test.yaml backends/vcv_rack.py tests/test_vcv_rack_backend.py
git commit -m "Add VCVRackBackend: MIDI-CAT CV out, PipeWire loopback audio in"
```

---

## Task 4: Data collection (random CV sweep)

**Files:**
- Create: `data_collection.py`
- Test: `tests/test_data_collection.py`

**Interfaces:**
- Consumes: `CVBackend` (Task 1, tested against `FakeCVBackend`), `extract_features` (Task 2).
- Produces: `collect_sweep_dataset(backend: CVBackend, n_samples: int, settle_time_s: float, sample_rate: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]` returning `(cv_array, feature_array)` each of shape `(n_samples, 3)` (channel order = `backend.channels()`), and a `if __name__ == "__main__"` entry point that runs it against a real `VCVRackBackend` and saves to `data/sweep_dataset.npz`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_collection.py
import numpy as np
from backends.base import FakeCVBackend
from data_collection import collect_sweep_dataset


def test_collect_sweep_dataset_shapes_and_ranges():
    backend = FakeCVBackend(
        channel_names=["a", "b", "c"],
        audio_block=(0.3 * np.sin(2 * np.pi * 440 * np.arange(2048) / 48000)).astype(np.float32),
    )
    rng = np.random.default_rng(seed=42)
    cv_array, feature_array = collect_sweep_dataset(
        backend, n_samples=10, settle_time_s=0.0, sample_rate=48000, rng=rng
    )
    assert cv_array.shape == (10, 3)
    assert feature_array.shape == (10, 3)
    assert np.all(cv_array >= 0.0) and np.all(cv_array <= 1.0)


def test_collect_sweep_dataset_actually_sets_cv_on_backend():
    backend = FakeCVBackend(channel_names=["a", "b", "c"])
    rng = np.random.default_rng(seed=1)
    collect_sweep_dataset(backend, n_samples=5, settle_time_s=0.0, sample_rate=48000, rng=rng)
    # 3 channels x 5 samples = 15 set_cv calls
    assert len(backend.set_cv_calls) == 15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/code/vcv-cv-harness && python3 -m pytest tests/test_data_collection.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data_collection'`

- [ ] **Step 3: Write minimal implementation**

```python
# data_collection.py
import time
import numpy as np

from backends.base import CVBackend
from features import extract_features


def collect_sweep_dataset(
    backend: CVBackend,
    n_samples: int,
    settle_time_s: float,
    sample_rate: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    channels = backend.channels()
    cv_array = np.zeros((n_samples, len(channels)))
    feature_array = np.zeros((n_samples, 3))

    for i in range(n_samples):
        cv_vec = rng.uniform(0.0, 1.0, size=len(channels))
        for ch, value in zip(channels, cv_vec):
            backend.set_cv(ch, float(value))
        time.sleep(settle_time_s)
        block = backend.read_audio_block()
        feature_array[i] = extract_features(block, sample_rate)
        cv_array[i] = cv_vec

    return cv_array, feature_array


if __name__ == "__main__":
    from backends.vcv_rack import VCVRackBackend

    backend = VCVRackBackend("configs/bridge_test.yaml")
    rng = np.random.default_rng(seed=0)
    # settle_time_s=0.5 -- confirmed generously above the real settle time
    # observed in Phase 1's round-trip verification (RMS transitions completed
    # within a few hundred ms); re-check with a stopwatch-style manual test
    # (Task 3 Step 6's pattern) if this dataset's model behaves oddly.
    cv_array, feature_array = collect_sweep_dataset(
        backend, n_samples=3000, settle_time_s=0.5, sample_rate=48000, rng=rng
    )
    import os
    os.makedirs("data", exist_ok=True)
    np.savez("data/sweep_dataset.npz", cv=cv_array, features=feature_array, channels=backend.channels())
    print(f"saved {len(cv_array)} samples to data/sweep_dataset.npz")
    backend.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/code/vcv-cv-harness && python3 -m pytest tests/test_data_collection.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the real collection against the live patch**

Requires VCV Rack running `bridge_test.vcv` with the audio routing set as in Task 3 Step 6.

```bash
cd ~/code/vcv-cv-harness
pactl set-default-sink vcv_loop
pactl set-default-source vcv_loop.monitor
python3 data_collection.py
```

Expected: prints `saved 3000 samples to data/sweep_dataset.npz` after roughly 25 minutes (3000 samples × ~0.5s settle). If that's too slow for a first pass, rerun with a smaller `n_samples` (e.g. 500) directly in a `python3 -c` one-liner using the same import pattern, to unblock Task 5 sooner, then re-collect the full 3000 later.

- [ ] **Step 6: Commit**

```bash
cd ~/code/vcv-cv-harness
git add data_collection.py tests/test_data_collection.py
git commit -m "Add random CV sweep data collection"
```

---

## Task 5: Model definition + training

**Files:**
- Create: `model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `data/sweep_dataset.npz` (Task 4's output, real data).
- Produces: `class InverseCVModel(torch.nn.Module)` (3→32→32→3, ReLU after the first two layers, forward signature `forward(self, target_features: torch.Tensor) -> torch.Tensor`), `train_model(cv_array: np.ndarray, feature_array: np.ndarray, epochs: int = 200, lr: float = 1e-3) -> InverseCVModel`, `save_weights(model: InverseCVModel, path: str) -> None` (writes `W1, b1, W2, b2, W3, b3` as a `.npz`, `W` matrices in `(in_features, out_features)` orientation — i.e. already transposed from PyTorch's native `Linear.weight` layout — so Task 6 can use them directly with `ttnn.linear`), and a `if __name__ == "__main__"` entry point loading `data/sweep_dataset.npz`, training, and saving to `data/model_weights.npz`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py
import numpy as np
import torch
from model import InverseCVModel, train_model, save_weights


def test_inverse_cv_model_forward_shape():
    model = InverseCVModel()
    out = model(torch.randn(1, 3))
    assert out.shape == (1, 3)


def test_train_model_reduces_loss_on_learnable_synthetic_data():
    # Synthetic ground truth: cv = features (identity-ish, easily learnable),
    # so a model that trains correctly should get close to it.
    rng = np.random.default_rng(0)
    features = rng.uniform(0, 1, size=(500, 3))
    cv = features.copy()

    model = train_model(cv, features, epochs=300, lr=1e-2)

    test_features = torch.tensor([[0.2, 0.5, 0.8]], dtype=torch.float32)
    prediction = model(test_features).detach().numpy()
    assert np.allclose(prediction, [[0.2, 0.5, 0.8]], atol=0.1)


def test_save_weights_writes_expected_arrays(tmp_path):
    model = InverseCVModel()
    out_path = tmp_path / "weights.npz"
    save_weights(model, str(out_path))

    loaded = np.load(out_path)
    assert set(loaded.keys()) == {"W1", "b1", "W2", "b2", "W3", "b3"}
    assert loaded["W1"].shape == (3, 32)
    assert loaded["b1"].shape == (32,)
    assert loaded["W2"].shape == (32, 32)
    assert loaded["b2"].shape == (32,)
    assert loaded["W3"].shape == (32, 3)
    assert loaded["b3"].shape == (3,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/code/vcv-cv-harness && python3 -m pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model'`

- [ ] **Step 3: Write minimal implementation**

```python
# model.py
import numpy as np
import torch
import torch.nn as nn


class InverseCVModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, target_features: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(target_features))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def train_model(cv_array: np.ndarray, feature_array: np.ndarray, epochs: int = 200, lr: float = 1e-3) -> InverseCVModel:
    model = InverseCVModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    x = torch.tensor(feature_array, dtype=torch.float32)
    y = torch.tensor(cv_array, dtype=torch.float32)

    for _ in range(epochs):
        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_fn(prediction, y)
        loss.backward()
        optimizer.step()

    return model


def save_weights(model: InverseCVModel, path: str) -> None:
    np.savez(
        path,
        W1=model.fc1.weight.detach().numpy().T,
        b1=model.fc1.bias.detach().numpy(),
        W2=model.fc2.weight.detach().numpy().T,
        b2=model.fc2.bias.detach().numpy(),
        W3=model.fc3.weight.detach().numpy().T,
        b3=model.fc3.bias.detach().numpy(),
    )


if __name__ == "__main__":
    data = np.load("data/sweep_dataset.npz")
    model = train_model(data["cv"], data["features"], epochs=500)
    save_weights(model, "data/model_weights.npz")
    print("saved trained weights to data/model_weights.npz")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/code/vcv-cv-harness && python3 -m pytest tests/test_model.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Train on the real collected data**

```bash
cd ~/code/vcv-cv-harness
python3 model.py
```

Expected: `saved trained weights to data/model_weights.npz`. If you want to see the loss curve, run this instead of the bare script and watch it decrease:

```bash
python3 -c "
import numpy as np
from model import train_model, save_weights
import torch, torch.nn as nn
data = np.load('data/sweep_dataset.npz')
model = train_model(data['cv'], data['features'], epochs=500)
x = torch.tensor(data['features'], dtype=torch.float32)
y = torch.tensor(data['cv'], dtype=torch.float32)
final_loss = nn.MSELoss()(model(x), y).item()
print('final training MSE:', final_loss)
save_weights(model, 'data/model_weights.npz')
"
```

- [ ] **Step 6: Commit**

```bash
cd ~/code/vcv-cv-harness
git add model.py tests/test_model.py
git commit -m "Add InverseCVModel (3->32->32->3) and training script"
```

---

## Task 6: TTNN inference

**Files:**
- Create: `tt_inference.py`
- Test: `tests/test_tt_inference.py`

**Interfaces:**
- Consumes: `data/model_weights.npz` (Task 5's output).
- Produces: `class TTInferenceEngine` with `__init__(self, weights_path: str, device_id: int = 0)` (opens the TT device via `ttnn.open_device`), `predict_cv(self, target_features: np.ndarray) -> np.ndarray` (shape `(3,)` in, `(3,)` out), and `close(self) -> None` (calls `ttnn.close_device`). **Every test and script in this task must run under a `gozer run --chips 1 ...` lease — never a bare `python3 -m pytest`** (see Global Constraints).

The forward pass (`linear → relu → linear → relu → linear`, verified working with `ttnn.linear(..., bias=...)` and `ttnn.relu(...)` matching a PyTorch reference within ~0.03 abs error at `bfloat16` precision) mirrors `InverseCVModel.forward` exactly, using the weights saved by Task 5's `save_weights` (already transposed to `(in_features, out_features)`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tt_inference.py
import numpy as np
import torch
from model import InverseCVModel, save_weights
from tt_inference import TTInferenceEngine


def test_tt_inference_matches_pytorch_reference(tmp_path):
    torch.manual_seed(0)
    model = InverseCVModel()
    weights_path = str(tmp_path / "weights.npz")
    save_weights(model, weights_path)

    target = np.array([0.3, 0.6, 0.9], dtype=np.float32)

    with torch.no_grad():
        reference = model(torch.tensor(target, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()

    engine = TTInferenceEngine(weights_path=weights_path)
    try:
        result = engine.predict_cv(target)
    finally:
        engine.close()

    assert result.shape == (3,)
    # bfloat16 on-device precision -- allow a generous but real tolerance,
    # verified in exploration to be within ~0.03 absolute for this model size.
    assert np.allclose(result, reference, atol=0.1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/code/vcv-cv-harness && gozer run --chips 1 --who "claude:tt-cv-agency" --reason "TDD: test_tt_inference should fail before implementation" -- python3 -m pytest tests/test_tt_inference.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tt_inference'`

- [ ] **Step 3: Write minimal implementation**

```python
# tt_inference.py
import numpy as np
import ttnn


class TTInferenceEngine:
    def __init__(self, weights_path: str, device_id: int = 0):
        weights = np.load(weights_path)
        self._device = ttnn.open_device(device_id=device_id)

        def to_device(array):
            import torch
            return ttnn.from_torch(
                torch.tensor(array, dtype=torch.float32),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self._device,
            )

        self._W1 = to_device(weights["W1"])
        self._b1 = to_device(weights["b1"].reshape(1, -1))
        self._W2 = to_device(weights["W2"])
        self._b2 = to_device(weights["b2"].reshape(1, -1))
        self._W3 = to_device(weights["W3"])
        self._b3 = to_device(weights["b3"].reshape(1, -1))

    def predict_cv(self, target_features: np.ndarray) -> np.ndarray:
        import torch
        x = ttnn.from_torch(
            torch.tensor(target_features, dtype=torch.float32).reshape(1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self._device,
        )
        h = ttnn.relu(ttnn.linear(x, self._W1, bias=self._b1))
        h = ttnn.relu(ttnn.linear(h, self._W2, bias=self._b2))
        out = ttnn.linear(h, self._W3, bias=self._b3)
        return ttnn.to_torch(out).float().numpy().reshape(-1)

    def close(self) -> None:
        ttnn.close_device(self._device)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/code/vcv-cv-harness && gozer run --chips 1 --who "claude:tt-cv-agency" --reason "TDD: verify test_tt_inference passes" -- python3 -m pytest tests/test_tt_inference.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
cd ~/code/vcv-cv-harness
git add tt_inference.py tests/test_tt_inference.py
git commit -m "Add TTNN inference engine for the trained inverse-CV model"
```

---

## Task 7: Control loop

**Files:**
- Create: `control_loop.py`
- Test: `tests/test_control_loop.py`

**Interfaces:**
- Consumes: `CVBackend` (Task 1), `extract_features` (Task 2), `TTInferenceEngine.predict_cv` (Task 6) — but the loop itself takes a `predict_fn: Callable[[np.ndarray], np.ndarray]` rather than the class directly, so it can be tested against a fake predictor without a chip lease.
- Produces: `run_control_loop(backend: CVBackend, predict_fn: Callable[[np.ndarray], np.ndarray], goal_features: np.ndarray, sample_rate: int, step_fraction: float = 0.3, control_interval_s: float = 0.1, max_iterations: int = 100, convergence_threshold: float = 0.05) -> list[np.ndarray]` (returns the list of measured feature vectors across iterations, so callers/tests can inspect convergence), and a `if __name__ == "__main__"` entry point wiring `VCVRackBackend` + `TTInferenceEngine` together under a `gozer` lease.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_control_loop.py
import numpy as np
from backends.base import FakeCVBackend
from control_loop import run_control_loop


class LinearFakeBackend(FakeCVBackend):
    """A fake whose audio block's features are a direct, known function of
    its current CV, so the control loop's convergence can be checked
    end-to-end without a real instrument or a chip."""

    def read_audio_block(self) -> np.ndarray:
        # Encode current CV directly as constant-amplitude/frequency content
        # sized so extract_features(...) recovers roughly the CV vector itself.
        level = self.last_known_cv("vca_level")
        t = np.arange(4096) / 48000.0
        return (level * 0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def fake_predict(target_features: np.ndarray) -> np.ndarray:
    # Correctly inverts LinearFakeBackend's known feature function so a
    # converged CV genuinely reproduces the target loudness feature:
    # feature = rms/REF_RMS = (level * 0.5 / sqrt(2)) / 0.4 = level * scale.
    scale = 0.5 / (0.4 * np.sqrt(2))
    target_level = np.clip(target_features[0] / scale, 0.0, 1.0)
    return np.array([0.5, 0.5, target_level])


def test_control_loop_converges_toward_loudness_goal():
    backend = LinearFakeBackend(channel_names=["vco_freq", "vco_fm", "vca_level"])
    goal = np.array([0.8, 0.5, 0.5])  # target loudness=0.8, others don't matter here

    history = run_control_loop(
        backend,
        predict_fn=fake_predict,
        goal_features=goal,
        sample_rate=48000,
        step_fraction=0.5,
        control_interval_s=0.0,
        max_iterations=20,
        convergence_threshold=0.05,
    )

    assert len(history) > 1
    first_error = abs(history[0][0] - goal[0])
    last_error = abs(history[-1][0] - goal[0])
    assert last_error < first_error
    assert last_error < 0.05


def test_control_loop_records_one_feature_vector_per_iteration():
    backend = LinearFakeBackend(channel_names=["vco_freq", "vco_fm", "vca_level"])
    goal = np.array([0.5, 0.5, 0.5])
    history = run_control_loop(
        backend, predict_fn=fake_predict, goal_features=goal, sample_rate=48000,
        control_interval_s=0.0, max_iterations=5, convergence_threshold=-1.0,  # never converge, run all 5
    )
    assert len(history) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/code/vcv-cv-harness && python3 -m pytest tests/test_control_loop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'control_loop'`

- [ ] **Step 3: Write minimal implementation**

```python
# control_loop.py
import time
from typing import Callable
import numpy as np

from backends.base import CVBackend
from features import extract_features


def run_control_loop(
    backend: CVBackend,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    goal_features: np.ndarray,
    sample_rate: int,
    step_fraction: float = 0.3,
    control_interval_s: float = 0.1,
    max_iterations: int = 100,
    convergence_threshold: float = 0.05,
) -> list[np.ndarray]:
    channels = backend.channels()
    history: list[np.ndarray] = []

    for _ in range(max_iterations):
        block = backend.read_audio_block()
        current_features = extract_features(block, sample_rate)
        history.append(current_features)

        if np.linalg.norm(current_features - goal_features) < convergence_threshold:
            time.sleep(control_interval_s)
            continue

        target_cv = predict_fn(goal_features)
        current_cv = np.array([backend.last_known_cv(ch) for ch in channels])
        next_cv = current_cv + step_fraction * (target_cv - current_cv)
        next_cv = np.clip(next_cv, 0.0, 1.0)

        for ch, value in zip(channels, next_cv):
            backend.set_cv(ch, float(value))

        time.sleep(control_interval_s)

    return history


if __name__ == "__main__":
    import sys

    from backends.vcv_rack import VCVRackBackend
    from tt_inference import TTInferenceEngine

    goal = np.array([float(x) for x in sys.argv[1:4]]) if len(sys.argv) >= 4 else np.array([0.7, 0.5, 0.5])

    backend = VCVRackBackend("configs/bridge_test.yaml")
    engine = TTInferenceEngine(weights_path="data/model_weights.npz")
    try:
        history = run_control_loop(
            backend, predict_fn=engine.predict_cv, goal_features=goal, sample_rate=48000,
        )
        print(f"final measured features: {history[-1]}, goal: {goal}")
    finally:
        engine.close()
        backend.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/code/vcv-cv-harness && python3 -m pytest tests/test_control_loop.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
cd ~/code/vcv-cv-harness
git add control_loop.py tests/test_control_loop.py
git commit -m "Add perceive-decide-act control loop"
```

---

## Task 8: End-to-end verification against the live patch

**Files:** none created — this task runs the full pipeline built in Tasks 1-7 against the real VCV Rack instance and records the result in `CLAUDE.md`.

**Interfaces:**
- Consumes: everything above.
- Produces: an entry in `CLAUDE.md` documenting the measured outcome (this is how the project's existing "verification status" section, which currently ends at Phase 1, gets its Phase 2 update).

- [ ] **Step 1: Confirm the live environment is ready**

VCV Rack running `bridge_test.vcv`, MIDI-CAT mappings intact (CC1/CC2/CC3 as in `configs/bridge_test.yaml`), PipeWire routed to the loopback:

```bash
pactl set-default-sink vcv_loop
pactl set-default-source vcv_loop.monitor
pw-link -l | grep -A2 "VCV Rack:output"
```

Expected: shows `VCV Rack:output_FL |-> vcv_loop:playback_FL` (and `_FR`) — confirms real routing, not just the UI's claimed device (this exact check caught a silent misroute during Phase 1; see `CLAUDE.md`).

- [ ] **Step 2: Run the control loop toward three different goals, under a chip lease**

```bash
cd ~/code/vcv-cv-harness
gozer run --chips 1 --who "claude:tt-cv-agency" --reason "end-to-end control loop verification" -- \
  python3 control_loop.py 0.2 0.5 0.2
gozer run --chips 1 --who "claude:tt-cv-agency" --reason "end-to-end control loop verification" -- \
  python3 control_loop.py 0.8 0.5 0.9
gozer run --chips 1 --who "claude:tt-cv-agency" --reason "end-to-end control loop verification" -- \
  python3 control_loop.py 0.5 0.5 0.5
```

Expected: each prints `final measured features: [...], goal: [...]` with the final measured vector visibly closer to the goal than a naive/random guess would be — record the actual printed numbers, don't just check for "no exception."

- [ ] **Step 3: Update `CLAUDE.md`**

Add a new section (after the existing "RESOLVED" section) titled `## Phase 2: hardware-in-the-loop control — result`, with:
- The three (goal, final-measured) pairs from Step 2, verbatim.
- Whichever of these turned out true, stated plainly: did it converge close to all three goals, only some, or none — and if not all, your best read on why (undertrained model on too little sweep data, `step_fraction`/`control_interval_s` too aggressive or too timid, a feature whose normalization constants from Task 2 don't match this patch's real range, etc.) — this is exactly the kind of "trust the subject, verify the instrument" note the rest of this file already keeps.

- [ ] **Step 4: Commit**

```bash
cd ~/code/vcv-cv-harness
git add CLAUDE.md
git commit -m "Document Phase 2 end-to-end control loop verification result"
```
