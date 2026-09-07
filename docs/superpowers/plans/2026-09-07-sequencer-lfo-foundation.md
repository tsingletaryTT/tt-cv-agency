# Stage 0: Sequencer + LFO Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a self-clocked sequencer + a slow filter-sweep LFO + a filter-envelope ADSR to the Minimoog patch, and replace single-block feature extraction with fixed-window mean/std aggregation, so the pipeline can measure and eventually train against a temporal (looping, evolving) instrument instead of only a static drone.

**Architecture:** Extend `patches/minimoog_test.vcv` (not modify it) into a new `patches/sequencer_test.vcv` by hand-authoring three new modules (`SEQ3`, `LFO`, `ADSR`) plus new cables and MIDI-CAT mappings into its patch JSON — same verified, no-UI-clicking technique used for every patch so far. Replace `features.extract_features` with a windowed aggregator, thread it through `data_collection.py`/`control_loop.py`, and make `InverseCVModel`'s dimensions parametric instead of hardcoded 3-in/3-out.

**Tech Stack:** Python 3.12, VCV Rack Free 2.6.6 (Fundamental + Stoermelder-P1 plugins, both already installed), `mido`, `sounddevice`, `numpy`, `torch`, `ttnn` (behind `gozer`), `pytest`.

**Spec:** `docs/superpowers/specs/2026-09-07-sequencer-lfo-foundation-design.md`

## Global Constraints

- Any `ttnn` use (module-level import or `ttnn.open_device`) goes through a `gozer run --chips 1 --who "claude:tt-cv-agency" --reason "..."` lease. Never a bare `import ttnn` — including inside a test file at module scope during pytest collection (this project was bitten by exactly that once; `tests/test_tt_inference.py`'s imports are inside function bodies for this reason — follow that pattern for any new hardware-marked test).
- CV values stay `[0, 1]` at the `CVBackend` interface. Physical range per channel is entirely the MIDI-CAT `min`/`max` remapping in the patch JSON, never in Python code.
- Channel identity and CC numbers live in per-patch YAML (`configs/sequencer_test.yaml`), never hardcoded in Python.
- Every new/changed VCV module's param/input/output IDs must be confirmed against the real installed plugin's C++ source (`gh api repos/VCVRack/Fundamental/contents/src/<Module>.cpp -H "Accept: application/vnd.github.raw"`) before being written into any patch JSON. This project has caught two real, otherwise-invisible bugs exactly this way (a mapping that was never actually saved to the patch file; an oscillator-tracking scheme that looked right in an audio sweep but wasn't). Do not guess a paramId/inputId/outputId from memory or by analogy to a different module.
- VCV Rack launch: `cd Rack2Free && LD_LIBRARY_PATH=. ./Rack ../patches/<file>.vcv`, backgrounded. A non-graceful kill leaves `~/.local/share/Rack2/log.txt` without a trailing `"END"`, which trips an invisible (XWayland-invisible) crash-recovery dialog on next launch — always `printf "END" >> ~/.local/share/Rack2/log.txt` after `kill -9`-ing a running Rack process, before relaunching. PipeWire loopback (`vcv_loop`/`vcv_loop.monitor`) and the ALSA MIDI loopback (`Midi Through:Midi Through Port-0 14:0`) are already configured as this machine's system defaults — don't recreate them, just confirm with `pactl get-default-sink`/`get-default-source` if audio verification looks wrong.
- Autosave (`~/.local/share/Rack2/autosave/patch.json`) is written periodically, not instantly — a raw-param read immediately after sending a CV change is a race, not a verification. Confirm the autosave file's own mtime is newer than your change (`stat -c '%y'`) before trusting what it shows, or better, verify via live audio measurement instead (the more reliable method used throughout this project).

---

### Task 1: Build and verify `patches/sequencer_test.vcv` + `configs/sequencer_test.yaml`

**Files:**
- Create: `patches/sequencer_test.vcv`
- Create: `configs/sequencer_test.yaml`
- Test: none (hardware-verified, not unit-testable — see Verification below)

**Interfaces:**
- Consumes: `patches/minimoog_test.vcv` (the existing 9-module patch: 3×VCO ids 1-3, Mixer id 4, VCF id 5, VCA id 6, AudioInterface2 id 7, MidiCat id 8, 8vert id 9) as the base to extend.
- Produces: a `.vcv` file with 12 modules (ids 1-12) and a config YAML naming 8 channels — `vco_freq`, `vcf_cutoff`, `vca_level` (unchanged CC 1/2/3) plus `seq_tempo`/`sweep_rate`/`sweep_depth`/`filter_env_amount`/`vcf_resonance` (new, CC 4-8). Every later task depends on this file and config existing with exactly these 8 channel names, in this order (matches `backend.channels()`, which later tasks assume is `["vco_freq", "vcf_cutoff", "vca_level", "seq_tempo", "sweep_rate", "sweep_depth", "filter_env_amount", "vcf_resonance"]`).

**Verified module facts to build from** (confirmed against `VCVRack/Fundamental` source this session — do not re-derive, but do not blindly trust either; spot-check at least `SEQ3`'s `ParamIds` enum yourself before using it, since it's the least obvious of the three):

`SEQ3` (`Fundamental`, slug `"SEQ3"`, installed version `2.6.4`) — `ParamIds` order: `TEMPO_PARAM`(0), `RUN_PARAM`(1), `RESET_PARAM`(2), `TRIG_PARAM`(3), `CV_PARAMS`+0..23 (4-27, laid out as row `j` step `i` at `4 + 8*j + i` for `j` in 0-2, `i` in 0-7 — row 0, ids 4-11, is the CV1/pitch row), `GATE_PARAMS`+0..7 (28-35, per-step momentary mute buttons, not the actual gate state), `TEMPO_CV_PARAM`(36), `STEPS_CV_PARAM`(37), `CLOCK_PARAM`(38). `InputIds`: `TEMPO_INPUT`(0), `CLOCK_INPUT`(1), `RESET_INPUT`(2), `STEPS_INPUT`(3), `RUN_INPUT`(4). `OutputIds`: `TRIG_OUTPUT`(0), `CV_OUTPUTS`+0..2 (1-3, row 0 → id 1), `STEP_OUTPUTS`+0..7 (4-11), `STEPS_OUTPUT`(12), `CLOCK_OUTPUT`(13), `RUN_OUTPUT`(14), `RESET_OUTPUT`(15). Confirmed in source: leaving `CLOCK_INPUT` unpatched makes the module self-clock from its own `TEMPO_PARAM` (`configParam(TEMPO_PARAM, -2.f, 4.f, 1.f, ...)`, exponential display) — no external clock module needed. Module-level (not param) state goes in `"data"`: `{"running": true, "gates": [1,1,1,1,1,1,1,1], "clockPassthrough": false}` (all 8 `gates` entries `1` = all steps active; `running: true` is required or the sequencer sits frozen on step 0 forever).

`LFO` (`Fundamental`, slug `"LFO"`, version `2.6.4`) — reuse the same enum already used for the Minimoog patch's context, but this instance's relevant field is `FREQ_PARAM`(2), range `-8..10`, exponential (`freq = 2^raw` Hz), default raw `1.0` (2 Hz — too fast for a "subtle sweep," will be overridden). `OutputIds`: `SIN_OUTPUT`(0), `TRI_OUTPUT`(1), `SAW_OUTPUT`(2), `SQR_OUTPUT`(3) — use `TRI_OUTPUT` for a smooth sweep, not `SIN_OUTPUT`/`SQR_OUTPUT` (arbitrary but deliberate: a triangle avoids the sine's very-low-second-harmonic-free-but-slightly-more-CPU shaping and the square's discontinuity — pick TRI unless you have a specific reason not to).

`ADSR` (`Fundamental`, slug `"ADSR"`, version `2.6.4`) — `ParamIds`: `ATTACK_PARAM`(0), `DECAY_PARAM`(1), `SUSTAIN_PARAM`(2), `RELEASE_PARAM`(3), CV attenuverters (4-7), `PUSH_PARAM`(8). `InputIds`: `ATTACK_INPUT`(0), `DECAY_INPUT`(1), `SUSTAIN_INPUT`(2), `RELEASE_INPUT`(3), `GATE_INPUT`(4), `RETRIG_INPUT`(5). `OutputIds`: `ENVELOPE_OUTPUT`(0). For a "pluck" character: short attack (raw param near its own minimum — confirm `MIN_TIME`/`MAX_TIME` in source, this module's own comments give `MIN_TIME = 1e-3f`, `MAX_TIME = 10.f`, exponential between them), short-to-moderate decay, low sustain (so the envelope actually falls away rather than holding at a high plateau), short release. Exact param values are your judgment call within "audibly plucks rather than holds" — verify by ear/measurement in this task's own Verification step, don't just trust the numbers.

`8vert` (`Fundamental`, already in the patch as module id 9) — `GAIN_PARAMS`+0..7 are independent per-row attenuverters, range `-1..1`, and each row's `IN_INPUTS+i` defaults to a constant `10.f` when left unpatched (confirmed in source — this is exactly how the Minimoog patch's row 0 already produces a constant transposition voltage from nothing but its own knob). Row 0 (`GAIN_PARAMS`+0, `IN_INPUTS`+0, `OUT_OUTPUTS`+0) is already used for `vco_freq`'s transposition — do not touch it. Use row 1 (`+1`) for `sweep_depth` (patched: LFO output → `IN_INPUTS`+1) and row 2 (`+2`) for `filter_env_amount` (patched: ADSR output → `IN_INPUTS`+2).

`VCF` (already module id 5) — **one existing default must change**: `FREQ_CV_PARAM` (paramId 3, the attenuverter on `VCF`'s own `FREQ_INPUT` jack) was left at its stock default `0.0` in `minimoog_test.vcv` because nothing used that jack. This task's whole sweep/envelope design routes through `FREQ_INPUT`, so `FREQ_CV_PARAM` must be raised to a real value (`1.0`, unity, is the reasonable default — leaves the summed sweep+envelope CV unattenuated) in the new patch's copy of module 5's params. If this is left at `0.0`, the sweep and envelope will visibly do nothing in Verification, which is the whole point of checking it there rather than assuming.

**Wiring** (new cables, in addition to every cable already in `minimoog_test.vcv`):
- `SEQ3` (id 10) `CV_OUTPUTS`+0 (output id 1) → `VCO1`/`VCO2`/`VCO3` `PITCH_INPUT` (input id 0 on modules 1, 2, 3) — three new cables, landing on the *same* jacks the existing `8vert` row-0 cables already land on (VCV sums multiple cables per input; this is deliberate, matching the spec).
- `SEQ3` `TRIG_OUTPUT` (output id 0) → `ADSR` (id 12) `GATE_INPUT` (input id 4).
- `LFO` (id 11) `TRI_OUTPUT` (output id 1) → `8vert` `IN_INPUTS`+1 (input id 1).
- `8vert` `OUT_OUTPUTS`+1 (output id 1) → `VCF` (id 5) `FREQ_INPUT` (input id 0).
- `ADSR` `ENVELOPE_OUTPUT` (output id 0) → `8vert` `IN_INPUTS`+2 (input id 2).
- `8vert` `OUT_OUTPUTS`+2 (output id 2) → `VCF` `FREQ_INPUT` (input id 0) — same jack as the sweep cable, summed.

**New MidiCat maps** (append to the existing 3 maps already in module 8's `data.maps`, same `min`/`max`-fraction convention used throughout):
- `cc: 4` → `moduleId: 10, paramId: 0` (`SEQ3.TEMPO_PARAM`), label `seq_tempo`, `min: 0.0, max: 1.0` (full range; the module's own `-2..4` raw range already gives a reasonable slow-to-fast spread).
- `cc: 5` → `moduleId: 11, paramId: 2` (`LFO.FREQ_PARAM`), label `sweep_rate`. Compute `min`/`max` fractions for a raw range of roughly `-6` to `-1` within the module's `-8..10` span (i.e. `min ≈ (-6-(-8))/18 ≈ 0.111`, `max ≈ (-1-(-8))/18 ≈ 0.389`) — this targets an audible period of roughly 1 to 64 seconds, "well under 1 Hz" as the spec calls for. Verify the actual achieved rate audibly/measurably in this task's Verification rather than trusting the arithmetic alone.
- `cc: 6` → `moduleId: 9, paramId: 1` (`8vert` row 1 gain), label `sweep_depth`. Use a narrow `min`/`max` band centered near the attenuverter's own zero (e.g. raw gain `-0.15` to `0.15` within the `-1..1` range → `min ≈ 0.425, max ≈ 0.575`) so the sweep stays subtle even at the channel's extremes.
- `cc: 7` → `moduleId: 9, paramId: 2` (`8vert` row 2 gain), label `filter_env_amount`. Wider band than `sweep_depth` (e.g. raw gain `-0.6` to `0.6` → `min ≈ 0.2, max ≈ 0.8`) — this is meant to be the dramatic "squelch" control.
- `cc: 8` → `moduleId: 5, paramId: 2` (`VCF.RES_PARAM`), label `vcf_resonance`, `min: 0.0, max: 1.0` — same full-range convention as `vcf_cutoff`/`vca_level`.

**Default sequence pattern** — `SEQ3` row 0 (`CV_PARAMS`, ids 4-11, `-10..10` V range, 1V/oct): an 8-step contour in semitones `[0, 0, 0, 12, 0, 7, 0, 3]` relative to whatever `vco_freq`'s transposition is holding, converted to volts (`semitones / 12`): `[0.0, 0.0, 0.0, 1.0, 0.0, 0.58333, 0.0, 0.25]`. This is a fixed default, not CV-controlled (per spec) — set these 8 param values directly, do not add MIDI-CAT maps for them.

- [ ] **Step 1: Write the patch-construction script**

Write a Python script (e.g. `/tmp/.../build_sequencer_patch.py`, not committed — only its *output* is) that:
1. Unpacks `patches/minimoog_test.vcv` (`tar --zstd -xf ... patch.json`) to get the existing 9-module/7-cable base as a Python dict.
2. Appends the 3 new modules (ids 10, 11, 12) with the params/data described above.
3. Mutates the existing module 5 (`VCF`) entry's `FREQ_CV_PARAM` (paramId 3) from `0.0` to `1.0`.
4. Appends the 6 new cables described above (with new unique cable ids, continuing from the existing patch's highest cable id).
5. Appends the 5 new MidiCat maps to the existing module 8's `data.maps` list.
6. Writes the result back out, repacks with `tar --zstd -cf patches/sequencer_test.vcv -C <dir> patch.json`.

- [ ] **Step 2: Write `configs/sequencer_test.yaml`**

```yaml
# configs/sequencer_test.yaml
# Channel map for patches/sequencer_test.vcv (adds a self-clocked SEQ3
# sequencer, a slow filter-sweep LFO, and an ADSR filter envelope on top
# of the Minimoog patch's 3xVCO -> Mixer -> VCF -> VCA signal path).
# See CLAUDE.md / docs/superpowers/specs/2026-09-07-sequencer-lfo-foundation-design.md
# for the design and docs/superpowers/plans/2026-09-07-sequencer-lfo-foundation.md
# Task 1 for exactly which module/param each channel drives.
midi_port_name: "Midi Through:Midi Through Port-0 14:0"
channels:
  vco_freq:
    cc: 1
  vcf_cutoff:
    cc: 2
  vca_level:
    cc: 3
  seq_tempo:
    cc: 4
  sweep_rate:
    cc: 5
  sweep_depth:
    cc: 6
  filter_env_amount:
    cc: 7
  vcf_resonance:
    cc: 8
```

- [ ] **Step 3: Launch and screenshot**

Kill any running Rack instance (`kill -9 <pid>`, then `printf "END" >> ~/.local/share/Rack2/log.txt`), launch `patches/sequencer_test.vcv` per the Global Constraints launch recipe, wait ~10s, screenshot the window (`import -window <id>`) and confirm visually: 12 modules present, `SEQ3`'s gate/CV cables visible, MIDI-CAT shows all 8 channel labels with `In: ALSA` resolved to the real device name (not "(No device)" — if it shows that, the `midiInput` field wasn't carried over correctly from the base patch; re-check Step 1).

- [ ] **Step 4: Verify each new channel against real audio, not assumption**

Using `mido` + `sounddevice` (the same pattern used for every prior channel in this project — see `CLAUDE.md` for worked examples), for each of the 5 new channels, sweep it across its CC range while holding the others at reasonable midpoints, and measure real audio:
- `seq_tempo`: confirm the audible loop period actually changes (e.g. measure time between amplitude/pitch transitions at two different `seq_tempo` settings — should differ).
- `sweep_rate`/`sweep_depth`/`filter_env_amount`: read audio across a window of several seconds (long enough to cover at least one full sweep/envelope cycle at the slowest setting being tested) and compute the *standard deviation* of `features.spectral_centroid` across blocks in that window. `sweep_depth` and `filter_env_amount` at their max should each produce visibly higher centroid-std than at their min; `sweep_rate` at its slowest should produce visibly *lower* centroid-std within a short window than at its fastest (slower movement per unit time).
- `vcf_resonance`: same CC-sweep-against-measured-audio pattern as `vcf_cutoff` in the Minimoog patch — expect a resonant peak's effect on the spectrum (a less flat, more concentrated spectral shape) as it increases, not necessarily a monotonic centroid change.
- Re-verify the original 3 channels (`vco_freq`, `vcf_cutoff`, `vca_level`) still behave as they did in the Minimoog patch — the new cables share input jacks with existing ones, so confirm nothing regressed.

Do not accept "no error thrown" or a single screenshot as verification — every channel needs a real measured-audio check, per this project's own established discipline (documented in `CLAUDE.md` for exactly this reason: a prior "it looked right in a sweep" check for the Minimoog patch's oscillator tracking turned out to be wrong).

- [ ] **Step 5: Commit**

```bash
git add patches/sequencer_test.vcv configs/sequencer_test.yaml
git commit -m "Add sequencer_test.vcv: SEQ3 + sweep LFO + filter ADSR on the Minimoog patch"
```

---

### Task 2: Windowed feature aggregation (`features.py`)

**Files:**
- Modify: `features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: nothing new — `rms`, `spectral_centroid`, `estimate_pitch` stay as-is (per-block, unchanged signatures). `extract_features(block, sample_rate) -> np.ndarray` (shape `(3,)`) also stays unchanged — later tasks still use it internally, one block at a time.
- Produces: a new function, `extract_features_aggregated(blocks: list[np.ndarray], sample_rate: int) -> np.ndarray` returning shape `(6,)`: `[mean(loudness), std(loudness), mean(brightness), std(brightness), mean(pitch_norm), std(pitch_norm)]` (mean/std computed across the per-block `extract_features` results, in that interleaved order — Task 4 depends on this exact ordering and length).

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from features import extract_features_aggregated

def test_extract_features_aggregated_shape():
    rng = np.random.default_rng(0)
    blocks = [rng.uniform(-1, 1, size=1024) for _ in range(5)]
    result = extract_features_aggregated(blocks, sample_rate=48000)
    assert result.shape == (6,)

def test_extract_features_aggregated_detects_modulation():
    # A block-to-block AMPLITUDE-MODULATED signal (loudness genuinely
    # changing across blocks) must show higher loudness-std than a
    # constant-amplitude signal with the same mean loudness.
    sample_rate = 48000
    t = np.arange(1024) / sample_rate
    static_blocks = [0.5 * np.sin(2 * np.pi * 220 * t) for _ in range(8)]
    modulated_blocks = [
        (0.1 + 0.4 * (i % 2)) * np.sin(2 * np.pi * 220 * t) for i in range(8)
    ]
    static_result = extract_features_aggregated(static_blocks, sample_rate)
    modulated_result = extract_features_aggregated(modulated_blocks, sample_rate)
    loudness_std_index = 1
    assert modulated_result[loudness_std_index] > static_result[loudness_std_index] * 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_features.py -k aggregated -v`
Expected: FAIL with "extract_features_aggregated not defined" (or similar `ImportError`/`AttributeError`).

- [ ] **Step 3: Implement `extract_features_aggregated`**

```python
def extract_features_aggregated(blocks: list[np.ndarray], sample_rate: int) -> np.ndarray:
    per_block = np.array([extract_features(block, sample_rate) for block in blocks])
    means = per_block.mean(axis=0)
    stds = per_block.std(axis=0)
    return np.array(
        [means[0], stds[0], means[1], stds[1], means[2], stds[2]], dtype=np.float64
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_features.py -v`
Expected: all PASS, including the two new tests and every pre-existing `features.py` test unchanged.

- [ ] **Step 5: Commit**

```bash
git add features.py tests/test_features.py
git commit -m "Add windowed mean/std feature aggregation for temporal patches"
```

---

### Task 3: Parametric `InverseCVModel` dimensions

**Files:**
- Modify: `model.py`
- Modify: `tt_inference.py`
- Test: `tests/test_model.py`, `tests/test_tt_inference.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `InverseCVModel(n_features: int = 3, n_channels: int = 3)` — defaults preserve the existing 3-in/3-out behavior so nothing currently using the old `InverseCVModel()` call breaks by accident, but every *new* call site (Task 5) passes explicit `n_features=6, n_channels=8`. `train_model(cv_array, feature_array, epochs=200, lr=1e-3)` currently hardcodes `InverseCVModel()` internally (checked in the current source — it ignores the actual data's shape entirely) — this is a real gap this task must close, not a maybe: change `train_model` to construct `InverseCVModel(n_features=feature_array.shape[1], n_channels=cv_array.shape[1])`, inferred from the training data itself rather than a caller-supplied argument, so a 6-feature/8-channel dataset (Task 5) and the existing 3/3 datasets both work through the same call unchanged. `save_weights` unchanged in its own signature but now records shapes that vary; `TTInferenceEngine` must size its staged TTNN tensors from the loaded `.npz`'s actual weight shapes (`W1.shape`, `W3.shape`) rather than an assumed `3`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_model.py additions
def test_inverse_cv_model_parametric_shapes():
    model = InverseCVModel(n_features=6, n_channels=8)
    x = torch.randn(4, 6)
    out = model(x)
    assert out.shape == (4, 8)

def test_inverse_cv_model_default_shapes_unchanged():
    model = InverseCVModel()
    x = torch.randn(4, 3)
    out = model(x)
    assert out.shape == (4, 3)
```

```python
# tests/test_tt_inference.py addition (hardware-marked, import ttnn inside the
# function body per the Global Constraints rule — do not import at module scope)
@pytest.mark.hardware
def test_tt_inference_handles_non_default_shape(tmp_path):
    import ttnn  # noqa: F401
    from tt_inference import TTInferenceEngine

    torch.manual_seed(0)
    model = InverseCVModel(n_features=6, n_channels=8)
    weights_path = str(tmp_path / "weights.npz")
    save_weights(model, weights_path)

    target = np.random.default_rng(0).uniform(0, 1, size=6).astype(np.float32)
    with torch.no_grad():
        reference = model(torch.tensor(target).unsqueeze(0)).squeeze(0).numpy()

    engine = TTInferenceEngine(weights_path=weights_path)
    try:
        result = engine.predict_cv(target)
    finally:
        engine.close()

    assert result.shape == (8,)
    assert np.allclose(result, reference, atol=0.1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_model.py -k parametric -v`
Expected: FAIL (constructor doesn't accept these kwargs yet, or shapes mismatch).

- [ ] **Step 3: Implement the parametric model**

```python
class InverseCVModel(nn.Module):
    def __init__(self, n_features: int = 3, n_channels: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, n_channels)

    def forward(self, target_features: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(target_features))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
```

Change `train_model` to infer dimensions from its own inputs instead of constructing a bare `InverseCVModel()`:

```python
def train_model(cv_array: np.ndarray, feature_array: np.ndarray, epochs: int = 200, lr: float = 1e-3) -> InverseCVModel:
    model = InverseCVModel(n_features=feature_array.shape[1], n_channels=cv_array.shape[1])
    ...  # unchanged below this line
```

Update `tt_inference.py`'s `TTInferenceEngine.__init__` weight-staging: wherever it currently assumes a fixed size (check for hardcoded `3` in shape assertions or tensor construction), derive it instead from the loaded `.npz`'s actual `W1`/`W3` array shapes. Keep the existing `ChannelMismatchError` check (comparing `channels` provenance length to `expected_channels`) — that logic is already shape-agnostic (`len(channels)`), just confirm it still fires correctly for an 8-channel weights file.

Add one more test alongside the two in Step 1, since this is a real behavior change: `test_train_model_infers_dimensions_from_data` — call `train_model` with a synthetic `feature_array` of shape `(N, 6)` and `cv_array` of shape `(N, 8)`, assert the returned model's `fc1.in_features == 6` and `fc3.out_features == 8`.

- [ ] **Step 4: Run tests to verify they pass**

Run non-hardware: `python3 -m pytest tests/test_model.py -v`
Run hardware (leased): `gozer run --chips 1 --who "claude:tt-cv-agency" --reason "verify parametric TTInferenceEngine shapes" -- python3 -m pytest tests/test_tt_inference.py -v -m hardware`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add model.py tt_inference.py tests/test_model.py tests/test_tt_inference.py
git commit -m "Make InverseCVModel dimensions parametric instead of hardcoded 3-in/3-out"
```

---

### Task 4: Wire aggregated features through `data_collection.py` and `control_loop.py`

**Files:**
- Modify: `data_collection.py`
- Modify: `control_loop.py`
- Test: `tests/test_data_collection.py`, `tests/test_control_loop.py`

**Interfaces:**
- Consumes: `extract_features_aggregated` (Task 2), `backend.channels()` (existing, already generic).
- Produces: `collect_sweep_dataset(..., aggregate_window_s: float = 5.0)` — reads multiple blocks spanning `aggregate_window_s` seconds per sample instead of one block, using `extract_features_aggregated`; `feature_array`'s column count becomes `6` instead of `3` (verify no code still assumes `3`). `run_control_loop` similarly reads across a window each iteration rather than one block, and its convergence check (`np.linalg.norm(current_features - goal_features)`) now operates on 6-dim vectors — `goal_features` callers must pass a 6-dim vector, not 3-dim (this is a breaking change to `run_control_loop`'s calling convention; Task 5 is the only caller that needs updating for it in this codebase).

- [ ] **Step 1: Write the failing tests**

Extend `tests/test_data_collection.py`'s existing `FakeCVBackend`-based tests: assert `collect_sweep_dataset` with a fake backend returns a `feature_array` with shape `(n_samples, 6)` instead of `(n_samples, 3)`, and that it calls `backend.read_audio_block()` multiple times per sample (not once) — e.g. by having the fake backend count calls and asserting the count is `> n_samples`.

Extend `tests/test_control_loop.py` similarly: `run_control_loop` with a fake backend and a 6-dim `goal_features` runs without error and its returned history entries are 6-dim.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_data_collection.py tests/test_control_loop.py -v`
Expected: FAIL against the current single-block, 3-dim implementation.

- [ ] **Step 3: Implement the windowed read + aggregation in both functions**

In `collect_sweep_dataset`: after `set_cv` + `settle_time_s` sleep, read `ceil(aggregate_window_s * sample_rate / block_size)` blocks in a loop (using `backend.block_size()` if available, matching the existing `sample_rate` pull-from-backend convention — check `CVBackend`'s interface for a `block_size()` method already added in an earlier phase) and pass the list to `extract_features_aggregated`.

In `run_control_loop`: same windowed-read replacing the single `backend.read_audio_block()` call each iteration, passed through `extract_features_aggregated`. Keep `step_fraction`/`control_interval_s`/`max_iterations`/`convergence_threshold` semantics unchanged — only the feature-extraction call and its dimensionality change.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_data_collection.py tests/test_control_loop.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add data_collection.py control_loop.py tests/test_data_collection.py tests/test_control_loop.py
git commit -m "Wire windowed feature aggregation through data collection and the control loop"
```

---

### Task 5: End-to-end sanity check against the new patch

**Files:**
- Modify: `data_collection.py` (`__main__` block — point at the new config/channel count)
- Modify: `control_loop.py` (`__main__` block — same)
- No new test file — this task's deliverable is a live verification, documented in `CLAUDE.md`, not new automated tests.

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: a small real dataset (`data/sequencer_sweep_dataset.npz`, gitignored, not committed — matches the existing `data/` gitignore rule), a retrained `data/sequencer_model_weights.npz` (also gitignored), and a documented live control-loop run.

- [ ] **Step 1: Launch `patches/sequencer_test.vcv`**

Per the Global Constraints launch recipe. Confirm via the same screenshot check as Task 1 Step 3 that MIDI-CAT resolved its device correctly (a fresh launch can occasionally need this reconfirmed).

- [ ] **Step 2: Collect a small sanity dataset**

```bash
python3 -c "
import numpy as np
from backends.vcv_rack import VCVRackBackend
from data_collection import collect_sweep_dataset

backend = VCVRackBackend('configs/sequencer_test.yaml')
try:
    rng = np.random.default_rng(seed=0)
    cv_array, feature_array = collect_sweep_dataset(
        backend, n_samples=300, settle_time_s=0.5, rng=rng, aggregate_window_s=5.0
    )
    np.savez('data/sequencer_sweep_dataset.npz', cv=cv_array, features=feature_array, channels=backend.channels())
    print(f'saved {len(cv_array)} samples')
finally:
    backend.close()
"
```

(300 samples × ~5.5s each ≈ 27 minutes — run in the background and poll, per this project's own established pattern for long collection runs, rather than blocking a foreground turn on it.)

- [ ] **Step 3: Retrain with the parametric model**

```bash
python3 -c "
import numpy as np
from model import InverseCVModel, train_model, train_val_split, evaluate_per_dimension, save_weights

data = np.load('data/sequencer_sweep_dataset.npz')
channels = data['channels'].tolist()
(cv_train, feat_train), (cv_val, feat_val) = train_val_split(data['cv'], data['features'], val_fraction=0.2, seed=0)
model = train_model(cv_train, feat_train, epochs=500)  # infers 6-in/8-out from the data itself, per Task 3
mse, r2 = evaluate_per_dimension(model, cv_val, feat_val, baseline_mean=cv_train.mean(axis=0))
for ch, m, r in zip(channels, mse, r2):
    print(f'{ch:<20} mse={m:.4f} r2={r:.4f}')
save_weights(model, 'data/sequencer_model_weights.npz', channels=channels)
"
```

- [ ] **Step 4: Live control-loop verification**

```bash
gozer run --chips 1 --who "claude:tt-cv-agency" --reason "Stage 0 end-to-end sanity check against sequencer_test.vcv" -- python3 -c "
import numpy as np
from backends.vcv_rack import VCVRackBackend
from control_loop import run_control_loop
from tt_inference import TTInferenceEngine

backend = VCVRackBackend('configs/sequencer_test.yaml')
try:
    engine = TTInferenceEngine(weights_path='data/sequencer_model_weights.npz', expected_channels=backend.channels())
    try:
        goal = np.array([0.5, 0.05, 0.5, 0.1, 0.5, 0.05])  # 6 dims: [mean, std] x [loudness, brightness, pitch]
        history = run_control_loop(backend, predict_fn=engine.predict_cv, goal_features=goal, aggregate_window_s=5.0)
        print(f'final: {history[-1]}, goal: {goal}')
    finally:
        engine.close()
finally:
    backend.close()
"
```

Report the actual result honestly, whatever it is — per this task's Verification bar (below), approximate convergence and "nothing throws" is success, not a tight numeric match.

- [ ] **Step 5: Document in `CLAUDE.md` and commit**

Write a dated section (matching this file's existing style) reporting: the collected dataset's per-channel MSE/R² (expect `vco_freq`/`vcf_cutoff`/`vca_level`/`vcf_resonance` to behave reasonably given they're direct-knob channels like before; genuinely note if `seq_tempo`/`sweep_rate`/`sweep_depth`/`filter_env_amount` show poor R² — with only 300 samples across an 8-dimensional space this is plausible and should be reported honestly, not hidden), and the live control-loop result. Explicitly state this is a foundation smoke test, not a claim that Stage 0's instrument is well-controlled yet — that's Stages 1-3's job.

```bash
git add CLAUDE.md
git commit -m "Document Stage 0 end-to-end sanity check against sequencer_test.vcv"
```

## Verification

Full test suite (existing + all new tests from Tasks 2-4) green, hardware-marked tests included (`python3 -m pytest -q` for non-hardware, `gozer run --chips 1 ... -- python3 -m pytest -q -m hardware` for hardware), before this stage is considered done. This plan's own Task 5 is itself the stage's end-to-end verification — no separate final check beyond it.
