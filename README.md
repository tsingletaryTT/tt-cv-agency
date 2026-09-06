# tt-cv-agency

A model, running on Tenstorrent hardware, in the decision loop of a synthesizer:
listen to the live audio, decide a CV adjustment toward a goal, write it back,
repeat. Built and proven against VCV Rack first; the CV backend is deliberately
abstracted so the same code should transfer to a real Eurorack system through
Expert Sleepers CV/audio interfaces later — the basics learned here with a
VCO/VCA/LFO/EG-style patch are meant to carry over to more exotic modules too.

**Status: work in progress, past first full pass.** Phase 1 (a plain, zero-ML
CV/audio bridge) is done and verified against live hardware. Phase 2 (a small
model trained and run via `ttnn` on a Tenstorrent chip, in the actual control
loop) has all 8 planned steps implemented, tested, and reviewed — feature
extraction, the `CVBackend`/`VCVRackBackend` pair, data collection, the
inverse-regression model, `TTInferenceEngine` (TTNN inference on real
hardware), `run_control_loop` (the closed loop itself), and a live
end-to-end verification against the running patch. A final whole-branch
review then found and fixed a real correctness bug in the pitch feature
(see "Results so far" below) and re-ran verification: the pipeline works
end-to-end on real hardware with no exceptions, one of three target
features (loudness) converges reliably, and the pitch fix produced a
measurable, confirmed improvement in the pitch channel's behavior — but
Phase 2 is not yet a full "hits every goal" result. `CLAUDE.md`'s final
sections have the detailed diagnosis.

## Why VCV Rack first

VCV Rack is free, fast to iterate in, and infinitely patchable — a good stand-in
for a modular synth before touching real hardware. Everything above the CV/audio
I/O layer (feature extraction, the model, the control loop) is written against a
`CVBackend` interface, not against VCV Rack directly, so a future backend that
talks to a real audio interface's DC-coupled CV outputs should be able to reuse
all of it.

## Results so far

**Phase 1 — the plain I/O bridge, no ML:** a Python process reads VCV Rack's live
audio (via a PipeWire loopback) and writes its CV (via MIDI-CAT) in real time.
Confirmed with a real MIDI CC sweep measured against actual captured audio, not
just visual inspection:

| CC value | measured RMS |
|---|---|
| 0 | 0.00 |
| 32 | 0.00 (near-silent at low VCA level, expected) |
| 64 | 3014.12 |
| 96 | 6029.75 |
| 127 | 9042.26 |

Evenly-spaced control values produce evenly-spaced audio response — a real
closed loop, not a coincidence.

**Phase 2:** a 3-feature state representation (loudness, brightness, pitch —
via RMS, spectral centroid, and autocorrelation) extracted from live audio; a
random CV sweep against the real running patch (500 samples); a small
inverse-regression model (3→32→32→3, ReLU) fit to that data; `TTInferenceEngine`
running that model's weights on a real Tenstorrent chip via `ttnn`; and
`run_control_loop` closing the loop (read audio → extract features → infer CV
→ write CV → repeat) against the live patch.

A final whole-branch review found and fixed a genuine bug in
`estimate_pitch`: at the real production `block_size` (1024 samples), its
peak search started inside the still-descending autocorrelation main lobe
for any fundamental below ~440 Hz, so it silently returned a constant
clipped-to-`fmax` value instead of the real pitch. Confirmed real-world
impact: 52% of the original 500-sample dataset's `pitch` column was that
constant artifact, and `corr(vco_freq, pitch) = -0.29` (the feature moved
the *wrong way* from the knob that controls it). After the fix and a fresh
500-sample collection: the clip-artifact rate dropped to 2.4%, and
`corr(vco_freq, pitch) = +0.797`.

Retrained on the corrected dataset with an 80/20 held-out validation split,
reporting per-channel MSE/R² (against a predict-the-mean baseline) instead
of one aggregate in-sample number:

| CV channel | held-out MSE | held-out R² |
|---|---|---|
| `vco_freq` | 0.0234 | 0.7085 |
| `vco_fm` | 0.0659 | 0.1001 |
| `vca_level` | 0.0168 | 0.8034 |

`vco_fm`'s low R² is not an unpatched-input artifact — its FM input is
confirmed patched to an LFO in the live patch — more likely the LFO's
rate/depth just doesn't move any of the three features much within a
single ~21ms audio block.

Re-running the same live 3-goal end-to-end verification with the fixed
pitch estimator and retrained model: loudness converges in 2 of 3 goals,
brightness still doesn't converge in any (unchanged, root cause still
open), and pitch — which previously moved in a non-monotonic,
goal-independent way — now tracks the goal monotonically and converges in
1 of 3. Full detail, numbers, and the honest read of what's still broken
are in `CLAUDE.md`'s "Phase 2 final-review fix pass" section.

## What's here

- `backends/base.py` — the `CVBackend` abstract interface (`set_cv`,
  `read_audio_block`, `channels`, `last_known_cv`) and `FakeCVBackend`, an
  in-memory test double.
- `backends/vcv_rack.py` — `VCVRackBackend`, the real implementation: MIDI-CAT
  CV output over a MIDI loopback port, audio capture from a PipeWire loopback
  sink via a persistent `sounddevice` stream.
- `configs/bridge_test.yaml` — channel → MIDI CC mapping for the test patch, so
  no channel identity or CC number is hardcoded in Python.
- `features.py` — loudness (RMS), brightness (spectral centroid), and pitch
  (autocorrelation) feature extraction, each normalized to roughly `[0, 1]`.
- `data_collection.py` — sweeps random CV settings against a `CVBackend` and
  logs the resulting `(cv, features)` pairs for training data.
- `model.py` — `InverseCVModel` (a tiny feedforward regressor mapping target
  features back to the CV settings that should produce them), its PyTorch
  training script (with an 80/20 train/val split and per-channel MSE/R²
  reporting), and `save_weights`/channel-order provenance.
- `tt_inference.py` — `TTInferenceEngine`: loads the trained weights and runs
  the model's forward pass on a real Tenstorrent chip via `ttnn` (always
  behind a `gozer` chip lease). Asserts the loaded weights' recorded CV
  channel order matches the backend's current one before running, so a
  channel-order mismatch fails loudly instead of silently driving the wrong
  physical CV channel.
- `control_loop.py` — `run_control_loop`: the perceive-decide-act loop itself
  (read audio → extract features → infer target CV → step toward it → write
  CV → repeat) against any `CVBackend`.
- `pyproject.toml` — pytest config; registers the `hardware` marker so tests
  that touch `ttnn` are skipped by default and only run explicitly, under a
  `gozer` lease.
- `roundtrip_test.py` — the very first Phase 1 proof script (CC sweep +
  measured RMS).
- `tests/` — the test suite for everything above; `FakeCVBackend` keeps most of
  it runnable without a live VCV Rack instance.
- `docs/superpowers/specs/`, `docs/superpowers/plans/` — the design spec and
  the task-by-task implementation plan this was built from.
- `CLAUDE.md` — a detailed build log: what was tried, what broke, and why —
  worth reading before touching the VCV Rack / PipeWire / MIDI-CAT side of this.

## Notable bring-up details

Getting a plain Python process to reliably read/write a live VCV Rack instance
turned out to be most of the actual work. Highlights (full detail in
`CLAUDE.md`):

- **VCV Rack patches are hand-authorable JSON.** A `.vcv` file is a
  zstd-compressed tar around `patch.json` — patches can be constructed
  programmatically rather than built by hand in the GUI.
- **A recurring "did you crash?" dialog, traced to one string.** Killing Rack
  non-gracefully leaves its log without a trailing `"END"` marker, which trips
  a crash-recovery prompt on next launch — one that renders as a native
  Wayland surface invisible to every X11-based automation tool. Appending
  `"END"` to the log file after a hard kill avoids it entirely.
- **A real PipeWire routing gotcha.** Selecting a named audio device in VCV's
  own UI doesn't reliably make the underlying stream connect to it — the
  system's *default* sink/source needs to be set to the loopback device for
  the routing to actually take effect, independent of what the UI claims.
- **Synthetic mouse clicks aren't full mouse clicks.** `xdotool`-driven clicks
  worked for menus, buttons, and screenshots, but not for VCV's "click any
  parameter to map it" gesture specifically — that one needed a real human
  click. Everything else about patch-building stayed scriptable.
- **A real, caught-before-it-mattered bug:** an early version of the audio
  capture buffered blocks in an unbounded queue, so it silently returned
  audio from *before* the most recent CV change rather than current audio.
  Caught by comparing live RMS measurements against expectation rather than
  trusting "no errors thrown" — fixed by bounding the queue to the single
  latest block.

## Requirements

Python 3.12, `torch`, `ttnn` (from a `tt-metal` checkout), `mido`,
`sounddevice`, `numpy`, `scipy`, `PyYAML`, `pytest`. VCV Rack (free edition) is
the current test instrument, with the [MIDI-CAT](https://github.com/stoermelder/vcvrack-packone)
module mapped to the patch's CV-controllable parameters over a MIDI loopback
port.

Any `ttnn` use goes through a chip lease (`gozer`, this machine's cooperative
Tenstorrent chip-leasing tool) — never a bare unleased `import ttnn`.
