# TT-CV-Agency: hardware-in-the-loop CV control — design

## Context

`vcv-cv-harness` proved (Phase 1) that an external process can read VCV
Rack's live audio and read/write its CV in real time, with zero ML, via
MIDI-CAT + a PipeWire loopback. This phase (Phase 2) puts a model on
Tenstorrent hardware in the actual decision path: perceive the current
audio state, decide a CV adjustment toward a goal, act by writing CV, repeat.

**Why this shape specifically**: the longer-term goal is to control a real
Eurorack system through Expert Sleepers CV/audio interfaces — a completely
different physical I/O layer (DC-coupled audio-interface channels instead of
MIDI-CAT-in-VCV). Everything above the I/O layer (feature extraction, the
model, the control loop) needs to be indifferent to which backend it's
talking to, so what gets learned and built against VCO/VCA/LFO/EG-type
modules here transfers directly once a hardware backend exists. This
constraint shapes the whole design: a backend abstraction is not a
nice-to-have, it's the point.

Out of scope for this phase: real Expert Sleepers hardware (VCV Rack stays
the only backend for now), multi-chip allocation (one TT chip is plenty for
a model this size), and richer descriptor sets (stick to 3 features — loudness,
brightness, pitch — matching the 3 CV channels already mapped in the test
patch).

## Architecture

```
 ┌─────────────┐     features      ┌──────────────┐    target CV   ┌─────────────┐
 │  Backend     │ ───────────────▶ │ Control loop │ ─────────────▶ │  Backend     │
 │ read_audio_  │                  │ (perceive/   │                │  set_cv()    │
 │  block()     │                  │  decide/act) │                │              │
 └─────────────┘                   └──────┬───────┘                └─────────────┘
                                           │ query(goal features)
                                           ▼
                                  ┌────────────────────┐
                                  │ TTNN inverse model  │
                                  │ (on TT chip)        │
                                  │ features → CV       │
                                  └────────────────────┘
```

### Backend interface

A small abstract interface, implemented once per instrument:

```python
class CVBackend:
    def set_cv(self, channel: str, value: float) -> None: ...
    def read_audio_block(self) -> np.ndarray: ...
    def channels(self) -> list[str]: ...
```

`VCVRackBackend` (this phase's only implementation) wraps:
- `set_cv` → sends a MIDI CC message via `mido` to the `Midi Through Port-0`
  loopback, using a per-patch config mapping channel name → CC number
  (`channel_1` → CC1, etc., matching the mappings already created in
  `bridge_test.vcv`'s MIDI-CAT).
- `read_audio_block` → reads from the `vcv_loop.monitor` PipeWire source
  (the same loopback proven in Phase 1), via a persistent `sounddevice`
  input stream rather than repeated short-lived `parec` calls (avoids the
  sink-idle/wake latency issues hit during Phase 1 verification).

Channel identity, MIDI CC numbers, and value ranges live in a small
per-patch YAML config (`configs/bridge_test.yaml`), not in code — this is
what makes "the same code, a different config" the story for a future
Expert Sleepers backend and a different patch/module set.

### Feature extraction (`features.py`)

Three features computed per audio block, matching the 3 controlled
channels:
- **Loudness** — RMS of the block.
- **Brightness** — spectral centroid of the block's magnitude spectrum.
- **Pitch** — autocorrelation-based fundamental frequency estimate (simple;
  good enough for a VCO's near-sinusoidal output, not aiming for robustness
  against noisy/inharmonic sources yet).

Each is normalized to roughly [0, 1] against a fixed reference range
(fixed a priori from the VCO/VCA's known ranges — e.g. loudness against
observed max RMS, pitch against the VCO's FREQ knob's Hz range — not
re-derived from the training data's min/max, so held-out goal values
outside the training sweep's observed range still make sense).

### Data collection (`data_collection.py`)

Sweeps random CV settings across all 3 channels via the backend (uniform
random in each channel's [0, 1] range), waits for the audio to settle after
each change (empirically-verified settle time from Phase 1 — on the order
of a few hundred ms, confirmed against real RMS transitions, not assumed),
records the resulting audio block, extracts features, and logs
`(cv_vec, feature_vec)` pairs to a flat file (`data/sweep_dataset.npz`).
Target: a few thousand samples — plenty for a 3-in/3-out regressor, cheap
to collect given each sample only costs a fraction of a second.

### Model (`model.py`)

A small feedforward regressor: `target_features (3) → hidden (e.g. 32) →
hidden (32) → target_cv (3)`, ReLU activations, trained with plain MSE
regression against the logged sweep data (the sweep gives us
`(cv, features)`; training pairs are formed as `features → cv`, i.e. the
inverse direction — the sweep's own recorded CV is the label for its own
resulting features). Trained in PyTorch on CPU (this model is tiny; no
hardware needed for training itself — only inference goes on the TT chip,
per the "hardware in the *decision* path" goal). Checkpoint saved as plain
weight arrays (not a pickled PyTorch object) so the TTNN side has a simple,
framework-independent thing to load.

### TTNN inference (`tt_inference.py`)

Loads the trained weight arrays and reimplements the same tiny forward
pass (two linear layers + ReLU) using `ttnn` ops, run under a `gozer`
chip lease (`gozer run --chips 1 --who "claude:tt-cv-agency" --reason
"inference for CV control loop" -- ...`, per this machine's hardware
convention). Given the model's small size, this is a direct, literal port
(matmul + bias-add + relu, twice) rather than anything requiring
TT-Lang-level custom kernels — exact `ttnn` call shapes to be confirmed
against the installed `ttnn` version during implementation, not
pre-guessed here. Exposes one function: `predict_cv(goal_features: np.ndarray)
-> np.ndarray`.

### Control loop (`control_loop.py`)

```
goal_features = <specified by caller>
loop:
    audio = backend.read_audio_block()
    current_features = extract_features(audio)
    if close_enough(current_features, goal_features): keep holding, re-check
    target_cv = tt_inference.predict_cv(goal_features)
    current_cv = backend.last_known_cv()   # tracked locally, since MIDI CC is write-only
    next_cv = current_cv + step_fraction * (target_cv - current_cv)
    for each channel: backend.set_cv(channel, next_cv[channel])
    sleep(control_interval)
```

`step_fraction` (e.g. 0.3) and `control_interval` (e.g. 100ms) are the
two tunable knobs for how aggressively/quickly it converges — real
measurement every iteration is what makes this closed-loop rather than a
single open-loop guess, and what would let it keep correcting for drift
if this were pointed at a less perfectly-repeatable instrument (i.e. real
hardware) later.

## Testing / verification

- Data collection: spot-check a handful of logged `(cv, features)` pairs
  against manually-triggered CV changes and directly-measured audio, the
  same way Phase 1's round-trip was verified (real RMS measurement, not
  assumed).
- Model: held-out validation split from the sweep dataset, report MSE;
  sanity-check a few predictions by hand (e.g. "goal = max loudness" should
  predict something close to `VCA level = 1.0`).
- TTNN port: compare TTNN forward-pass output against the PyTorch model's
  output on the same inputs (should match closely, small numerical
  differences aside) before wiring it into the live loop.
- End-to-end: run the control loop toward 2-3 different goal points, and
  confirm via live audio measurement (not just "no errors thrown") that
  the actual RMS/centroid/pitch measurements converge toward the
  requested goal over a handful of iterations.

## File layout (within `~/code/vcv-cv-harness/`)

```
backends/
  base.py           # CVBackend abstract interface
  vcv_rack.py        # VCVRackBackend
configs/
  bridge_test.yaml   # channel -> CC number mapping for the existing test patch
features.py
data_collection.py
model.py             # PyTorch model + training script
tt_inference.py       # TTNN forward pass, loads trained weights
control_loop.py
data/
  sweep_dataset.npz  # generated, not committed
  model_weights.npz  # generated, not committed
```
