# Stage 0: sequencer + LFO foundation — design

## Context

This is Stage 0 of a four-stage roadmap agreed on 2026-09-07 (see
`CLAUDE.md`'s corresponding dated section for the full discussion): build
temporal capability into the instrument first (Stage 0, this spec), then
in order — an LLM-driven instruction-to-preset layer (Stage 1 / "Path A"),
an open-ended exploration/novelty-search agent (Stage 2 / "Path B"), and a
trajectory-predicting controller closer to the project's original
tt-vjepa2-inspired idea (Stage 3 / "Path C").

Every stage after this one needs an instrument that does something *over
time* — a held drone has no trajectory worth predicting, exploring, or
describing in an instruction ("play a looping sequence... sweeping subtly
on the filter"). Stage 0's entire job is to build that instrument and fix
the one thing about the existing pipeline that assumes audio is static:
`data_collection.py`/`control_loop.py` currently read a single ~21ms audio
block and treat it as "the settled state." Against a looping, evolving
pattern, one block just catches whatever random instant the loop happened
to be at — not a meaningful measurement of anything.

Out of scope for this stage: instruction parsing (Stage 1), exploration/
novelty search (Stage 2), trajectory prediction (Stage 3), a second ADSR
for an amplitude envelope (the patch stays legato — see "Deliberately
simple" below), and changing the already-calibrated `LOUDNESS_REF_RMS`/
`BRIGHTNESS_REF_HZ`/pitch-log-range constants. Also out of scope: any
large-scale data collection or exploration run against the new instrument
— Stage 0's own verification is a small sanity-check dataset, not a real
training run (that belongs to whichever of Stages 1-3 actually needs it).

## Architecture: the patch

New patch `patches/sequencer_test.vcv`, built the same hand-authored-JSON
way as `bridge_test_mapped.vcv` and `minimoog_test.vcv` — every module's
param/input/output enum confirmed against the real installed plugin
source before any JSON is written, not guessed (see CLAUDE.md's account of
the tracking bug the last patch caught this exact way). It extends
`minimoog_test.vcv`'s signal path (3×VCO → Mixer → VCF → VCA) rather than
replacing it — all three existing channels (`vco_freq`, `vcf_cutoff`,
`vca_level`) keep the same meaning.

New modules, all already installed (Fundamental only): `SEQ3` (8-step
sequencer, CV + gate), a second `LFO` instance used purely as a clock
source, a third `LFO` instance for the slow filter sweep, one `ADSR` for
the filter "pluck" envelope, and reusing the *existing* `8vert` module's
spare rows as attenuverters (see below) rather than adding new utility
modules for each one.

```
 clock LFO (square) ──▶ SEQ3 clock in
                          │ CV1 (pitch)           │ gate
                          ▼                        ▼
  8vert row1 (transposition) ──┐              ADSR (fixed env shape)
                                ├─▶ VCO1/2/3        │
  SEQ3 CV1 ──────────────────┘   PITCH_INPUT   8vert row3 (filter_env_amount)
                                  (summed)          │
                                                     ▼
  sweep LFO ──▶ 8vert row2 (sweep_depth) ──┐   VCF FREQ_INPUT (summed)
                                            ├─▶
  VCF FREQ_PARAM (vcf_cutoff, static base) ─┘
```

VCV sums multiple cables landing on the same input jack, so both
`PITCH_INPUT` and `VCF`'s cutoff CV input are shared, summed buses — no
new mixer module needed, the same trick the Minimoog patch's `8vert` row
already relied on.

One real interface constraint this surfaces: `VCF`'s cutoff CV input
(`FREQ_INPUT`) only affects the filter at all in proportion to its own
`FREQ_CV_PARAM` attenuverter knob — which the Minimoog patch left at its
default `0.0` (no existing cable used that jack, so it didn't matter
then). This stage's whole sweep/envelope design depends on `FREQ_CV_PARAM`
being turned up from that default — a fact about the interface, not just
a default value, so it's called out here rather than left for the build
to discover the hard way.

### New CV-controllable channels

| channel | drives | range goal |
|---|---|---|
| `seq_tempo` | clock LFO rate | roughly 1-8 Hz step rate (a 1-8 second loop for 8 steps) |
| `sweep_rate` | sweep LFO rate | slow — well under 1 Hz, "subtle" per the original ask |
| `sweep_depth` | 8vert row 2 gain (sweep LFO → VCF cutoff) | small — a wobble, not a full sweep across the whole cutoff range |
| `filter_env_amount` | 8vert row 3 gain (ADSR → VCF cutoff) | the main "squelch" control — wider range than `sweep_depth` |
| `vcf_resonance` | `VCF`'s own `RES_PARAM` | full 0-1, same convention as the other direct-knob channels |

Exact raw param ranges/MIDI-CAT `min`/`max` values are an implementation
detail resolved during the build (matching every other channel added so
far), not fixed here.

### Deliberately simple: no amplitude envelope yet

`vca_level` stays a continuously-held CV, not gated by `SEQ3`'s gate
output. This means notes are legato — the pitch steps through the
sequence and the filter wobbles/plucks, but nothing goes silent between
steps. A second `ADSR` gating `VCA` instead would give real note
articulation; deliberately deferred to keep this stage's scope to "prove
a temporal instrument and the aggregation fix work," not "build the most
authentic possible acid patch." Easy to add later once this foundation is
verified.

The sequence *pattern* itself (which 8 pitches) is a fixed, hand-picked
default baked into `SEQ3`'s own knobs, not a CV channel — turning
individual steps into continuous targets is a discrete, different kind of
control problem than everything else in this project and isn't part of
this stage.

## Feature aggregation (`features.py`, `data_collection.py`, `control_loop.py`)

The core change: replace "read one block, extract 3 scalars" with "read
blocks across a fixed aggregation window, extract mean and standard
deviation of each of the 3 base features across that window" — 6 numbers
instead of 3.

**Why a fixed window, not one computed from the current `seq_tempo`
value**: the aggregator could in principle read `backend.last_known_cv
("seq_tempo")` and size its listening window to match the current loop
period exactly. Rejected — it would make feature extraction aware of one
specific channel's patch-specific meaning ("this channel controls the
loop period"), breaking the backend-agnostic design this whole project is
built on (the same reason channel identity and CC numbers live in
per-patch YAML, not in `features.py`). Instead: a single fixed
`aggregate_window_s`, long enough to cover the slowest loop period the
`seq_tempo` channel's range can produce, with margin. Some CV settings
will be listened to for longer than one loop cycle; that's a fine
trade — an aggregation window a little longer than necessary is far
cheaper than a feature extractor that has to know what "tempo" means.

**Why mean *and* std, not just mean**: mean alone can't tell a static
drone from a wobbling, plucking pattern with the same average brightness
— exactly the distinction Stage 0 exists to make measurable. Std of
brightness across the window is a direct, cheap proxy for "how much is
the filter actually moving" — the same quantity `sweep_depth` and
`filter_env_amount` control. Not adding anything beyond mean/std (no
percentiles, no per-block time series) — two moments per feature is
enough to distinguish "moving" from "not moving" without turning the
feature vector into a large, low-signal blob.

This changes `extract_features`'s contract from returning 3 values to 6,
and every function currently passed a single `block` (`extract_features`,
and anything downstream that assumes a length-3 feature vector) needs an
aggregating counterpart or a changed signature — worked out in the plan,
not here.

### Model shape becomes parametric

`InverseCVModel` currently hardcodes `nn.Linear(3, 32)` in and
`nn.Linear(32, 3)` out. With features going from 3→6 and CV channels from
3→8 (`vco_freq`, `vcf_cutoff`, `vca_level`, `seq_tempo`, `sweep_rate`,
`sweep_depth`, `filter_env_amount`, `vcf_resonance`), both dimensions
change at once. `InverseCVModel` takes `n_features`/`n_channels`
constructor arguments instead of hardcoded `3`/`3`. This is a small,
mechanical change but a real one — every existing test that constructs an
`InverseCVModel` needs updating for the new signature, and
`tt_inference.py`'s weight-staging code needs to size its TTNN tensors
from the loaded weights' actual shape rather than an assumed `3`.

## Global constraints

- Any `ttnn` use stays behind a `gozer --chips 1` lease — no bare
  `import ttnn`, matching every prior stage.
- CV values stay `[0, 1]` at the `CVBackend` interface; the MIDI-CAT
  `min`/`max` remapping (or lack of it) is what turns that into the
  right physical range per channel, same convention as every channel so
  far.
- Channel identity and CC numbers stay in per-patch YAML
  (`configs/sequencer_test.yaml`), not in code.
- Every new/changed module's param/input/output IDs get confirmed against
  real plugin source before being written into patch JSON — this project
  has caught two real bugs (the MIDI-CAT-never-committed gap, the
  oscillator-tracking bug) exactly by insisting on this instead of
  trusting assumptions.

## Verification plan

1. **Patch loads and each new channel does something real**, checked the
   same way every channel so far has been: a live CC sweep against
   measured raw audio (RMS/centroid/pitch, or — new this time — the
   aggregated mean/std over a full window), not just "no error thrown."
   In particular: `seq_tempo` should change the audible loop period,
   `sweep_depth`/`filter_env_amount` should each visibly move brightness's
   *std* across the window while `sweep_rate` at its slowest should barely
   move it within one window (a real, checkable prediction, not a vague
   hope).
2. **Aggregated feature extraction has real unit tests**: at minimum, a
   synthetic test proving a modulated (time-varying) signal produces a
   detectably higher std than a static one at the same mean, and that a
   fixed-window aggregation over multiple blocks returns the expected
   6-vector shape.
3. **`InverseCVModel`'s new parametric shape has real tests**: construction
   with non-3/3 dimensions, and `tt_inference.py`'s TTNN path handling
   weights of that shape (behind a `gozer` lease).
4. **A small end-to-end sanity check, not a real training run**: collect a
   modest sample (order of a few hundred, not the 3000 used for the
   Minimoog dataset — this is a foundation smoke test, not training data
   for any of Stages 1-3), retrain, and run one live control-loop
   verification confirming the whole pipeline still functions against an
   8-channel, 6-feature, temporal instrument end to end. Convergence
   quality is not the bar here — "the loop runs, converges even
   approximately, and nothing throws" is; genuine training happens in
   whichever of Stages 1-3 needs it.

Full test suite (existing + new) must stay green, hardware-marked tests
included, before this stage is considered done.
