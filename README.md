# tt-cv-agency

A model, running on Tenstorrent hardware, in the decision loop of a synthesizer:
listen to the live audio, decide a CV adjustment toward a goal, write it back,
repeat. Built and proven against VCV Rack first; the CV backend is deliberately
abstracted so the same code should transfer to a real Eurorack system through
Expert Sleepers CV/audio interfaces later — the basics learned here with a
VCO/VCA/LFO/EG-style patch are meant to carry over to more exotic modules too.

**Status: work in progress.** Phase 1 (a plain, zero-ML CV/audio bridge) is done
and verified against live hardware. Phase 2 (a small model trained and run via
`ttnn` on a Tenstorrent chip, in the actual control loop) is in progress —
5 of 8 planned steps are implemented, tested, and reviewed; the TTNN inference
step, the closed-loop wiring, and full live end-to-end verification are next.

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

**Phase 2 so far:** a 3-feature state representation (loudness, brightness,
pitch — via RMS, spectral centroid, and autocorrelation) extracted from live
audio; a random CV sweep against the real running patch (500 samples, all three
features showing real, non-stale variation — loudness alone spans 0.00–0.94,
std 0.27); a small inverse-regression model (3→32→32→3, ReLU, trained with
plain supervised regression) fit to that data, final training MSE 0.044.

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
  features back to the CV settings that should produce them) and its PyTorch
  training script.
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
