# vcv-cv-harness

## What this is

Proof-of-concept prototype for a research idea: a model that listens to a
synthesizer's audio+CV state and adjusts CV signals to steer it toward a goal
state (analogous to robot manipulation control). This repo is Phase 1 only —
proving a plain (no-ML) I/O bridge into/out of VCV Rack works, before any
learned model gets involved. Full research context and the multi-chip design
discussion live in the session's plan file, not duplicated here.

## Original prompt / how this started

User asked for research + prior art on using models (robotics/world models/
LLMs) to read CV+audio state and adjust CV to reach a goal, using VCV Rack as
a testbed before real Eurorack hardware. Research turned up close prior art
(tt-vjepa2's action-conditioned world model + CEM planner, ACIDS-IRCAM's
Neurorack/RAVE/Flow-Synthesizer line of work) and a working VCV↔external
bridge (OSC'elot). User then asked to actually build the first prototype:
prove a Python process can read VCV's audio and read/write its CV in real
time, with zero ML.

## Layout

- `Rack2Free/` — VCV Rack Free 2.6.6 Linux build (downloaded from
  vcvrack.com/downloads, not vendored via apt — no distro package exists).
- `patches/bridge_test.vcv` — the clean-slate test patch (VCO → VCA →
  AudioInterface2, plus an OSC'elot module), authored **directly as JSON**
  rather than built by hand in the GUI (see below). Kept unmodified for
  future OSC'elot/MIDI-CAT debugging — it deliberately has no CC mappings.
- `patches/bridge_test_mapped.vcv` — the one to actually launch for data
  collection / control-loop runs: the same patch plus a MIDI-CAT module
  with `vco_freq`/`vco_fm`/`vca_level` already mapped to CC 1/2/3. See
  "MIDI-CAT mappings were never actually committed" below for why this file
  exists separately and how it was built.
- `roundtrip_test.py` — the round-trip proof script (OSC control out,
  audio capture in).
- `quarantined-broken-plugins/` — a corrupted `Nozoid` plugin package that
  was already broken (truncated tar) from a prior VCV Rack session on this
  machine before this project touched anything; moved aside rather than
  deleted, since it wasn't ours to remove outright.
- `oscelot_src_Oscelot.cpp` — a local copy of OSC'elot's source, fetched for
  debugging the mapping-update issue below (see "Open issue").

## Key facts learned (worth not re-discovering next time)

### `.vcv` patch files are hand-authorable JSON
A `.vcv` file is a **zstd-compressed tar** containing `patch.json` (+ a
`modules/<id>/` dir for any binary blobs like wavetables, not needed for
simple patches) — `tar --zstd -cf out.vcv -C dir patch.json` round-trips
correctly. This means patches can be constructed programmatically rather
than built by hand in the GUI. Schema notes:
- `modules[].plugin` must be the plugin **slug**, not its display **brand**
  (e.g. OSC'elot's slug is `"OSCelot"`; brand is `"TheModularMind"` — using
  the brand produces a silent "modules not installed" dialog).
- `cables[]` reference `outputModuleId`/`outputId` → `inputModuleId`/
  `inputId`, where the `*Id` port indices match each module's C++
  `ParamIds`/`InputIds`/`OutputIds` enum declaration order (fetch the
  plugin's source, e.g. `VCVRack/Fundamental` on GitHub, to get these —
  don't guess).
- The **autosave** file (`~/.local/share/Rack2/autosave/patch.json`) is
  plain uncompressed JSON, not zstd-tar'd — easiest place to both inspect
  and inject live state.
- `Rack <patchfile>` as a CLI arg works (confirmed via `adapters/
  standalone.cpp`: parses via `getopt_long`, calls `APP->patch->launch
  (patchPath)`), but only if not blocked by the crash-recovery dialog below.

### The recurring "crashed last session" dialog, and its real fix
Killing Rack with `kill -9` (or any non-clean exit) leaves `~/.local/share/
Rack2/log.txt` without a trailing `"END"` marker. On next launch, Rack's
`logger::isTruncated()` (checks literally whether the log ends with `"END"`,
or the legacy `"Destroying logger\n"`) reports true, and it shows a
**zenity** dialog — "VCV Rack crashed during the last session... Clear your
patch and start over?" — before doing anything else, blocking `patch->launch`
whether windowed *or* `--headless`.
- **This dialog renders as a native Wayland surface** — invisible to
  `xdotool`/`wmctrl`/`xwininfo`/X11 tooling entirely, even though the main
  Rack window itself renders fine via XWayland and *is* controllable.
- **The actual fix**: `printf "END" >> ~/.local/share/Rack2/log.txt` before
  the next launch. Do this after every non-graceful kill.
- A *second*, more normal zenity dialog ("this patch includes modules that
  are not installed") uses the same invisible-Wayland-surface rendering —
  same class of problem, but that one's actually a real signal (fix the
  underlying `plugin` slug mismatch instead of dismissing it).

### Controlling the Rack window from the CLI (screenshot + input)
- `import -window <id>` (ImageMagick, via XWayland) reliably screenshots
  the window even without focus. `grim` does **not** work here — this KDE/
  KWin Wayland compositor doesn't implement `wlr-screencopy-unstable-v1`.
- `xdotool mousemove --window <id> x y` + `xdotool click --window <id> 1`
  works for real clicks, **but only once the window has been genuinely
  activated** — `xdotool windowactivate` itself fails here
  (`XGetWindowProperty[_NET_ACTIVE_WINDOW] failed`), so clicks silently go
  nowhere without it.
- **Working activation method**: KWin's D-Bus scripting API, which exposes
  the *legacy* (KWin 5-style) scripting object on this build —
  `workspace.clientList()` / `workspace.activeClient = c`, not the newer
  `workspace.windows` / `workspace.windowList()` (that's KWin 6 API and
  doesn't exist here). Load+run inline:
  ```
  qdbus org.kde.KWin /Scripting org.kde.kwin.Scripting.loadScript <file.js> <name>
  qdbus org.kde.KWin /Scripting org.kde.kwin.Scripting.start
  ```
  Match the target window by `.caption` substring.
- **Gotcha**: re-running the activation script *between every click* is
  actively harmful — it appears to reset in-progress/pending UI selections
  (twice reproduced: the audio driver selection reverted from PulseAudio
  back to ALSA after a KWin re-activation in between). Activate **once** per
  interaction sequence, then do all the clicks for that sequence without
  reactivating.

### Getting real audio in/out: PipeWire/Pulse device-targeting is unreliable
Selecting a *named* device in Rack's own Audio module UI (e.g.
"VCV_Loopback (1-2 in, 1-2 out)") does **not** reliably make the underlying
RtAudio/PortAudio-via-Pulse stream actually connect to that device — `wpctl
status` / `pw-link -l` showed VCV Rack's ports connected to the physical
HDMI output instead, silently, despite the UI correctly showing
"VCV_Loopback" selected. The fix that actually worked: set the **system
default** sink/source to the loopback device before launching/selecting:
```
pactl load-module module-null-sink sink_name=vcv_loop sink_properties=device.description=VCV_Loopback
pactl set-default-sink vcv_loop
pactl set-default-source vcv_loop.monitor
```
Both were needed (one for playback routing, one for `sounddevice`/PortAudio
capture — which also can't target a named Pulse source directly; its Pulse
host API only exposes a generic `"pulse"` pseudo-device that follows
whatever the system default is). **Remember to restore the original
defaults afterward** if this machine is used for other things — this is
someone's real desktop, not an isolated test box.

## Follow-up investigation (same session): GitHub issues + fresh mapping +
a different bridge module

Per the three leads chased down, in order:

1. **GitHub issues on the OSC'elot repo**: two close-sounding reports —
   [#7](https://github.com/The-Modular-Mind/oscelot/issues/7) "module is
   unresponsive to messages" (root cause there: user hadn't turned the
   receiver on — orange vs green LED confusion) and
   [#15](https://github.com/The-Modular-Mind/oscelot/issues/15) "will not
   start receiving" (root cause: the Audio interface module didn't have a
   device chosen — fixed by selecting one *and saving*). Neither is an exact
   match for our symptom (our receiver genuinely is on — confirmed via a
   real bound socket — and the very first bind message DOES apply), but the
   "select audio + save" detail was worth ruling out explicitly (see next).
2. **Full clean restart + fresh mapping**, incorporating the issue #15 fix:
   killed all instances, appended `END`, relaunched from the *original*
   unmodified `bridge_test.vcv` (fully clean slate, no accumulated
   mapping/session cruft), reselected PulseAudio + VCV_Loopback, explicitly
   saved with Ctrl+S (confirmed via `patch.cpp:116 save` log line), *then*
   created one brand-new OSC'elot mapping with a fresh controller id (`9`,
   never used before). **Identical result**: the completing bind value
   applies, no value after it does, confirmed via the live autosave JSON
   after 30+ seconds. This rules out session/state cruft as the cause.
3. **A different bridge module entirely**: installed stoermelder's
   [MIDI-CAT](https://github.com/stoermelder/vcvrack-packone) (`Stoermelder-P1`
   plugin, `MidiCat` module — same GitHub-release-asset install method as
   OSC'elot, no VCV account needed), wired to Rack's built-in ALSA
   `Midi Through Port-0` loopback (visible via `mido.get_output_names()` —
   Rack always creates this, no extra setup). Sent real Control Change
   messages via `mido`. **Result was actually a *different*, more basic
   failure**: MIDI-CAT never completes the learn step at all in this
   environment — clicking the target parameter (confirmed landing correctly
   via its own hover tooltip, e.g. "Channel 1 level: 50%") before *or* after
   sending the CC message, and even a synthetic click-drag-release instead
   of a plain click, all leave the row as `ccNN Unmapped` — the controller
   ID gets learned, but nothing ever attaches to a parameter. This is
   **not** the same bug as OSC'elot's (which *does* complete both learns
   successfully, twice) — it's a separate failure mode specific to how
   MIDI-CAT detects "which parameter was clicked," which apparently doesn't
   fire the same way under `xdotool`-synthesized clicks that OSC'elot's
   simpler click handling does.

### What this rules in vs. out

- Not caused by leftover state from the long debugging session (full clean
  restart reproduced it identically).
- Not an OSC'elot-only quirk in the trivial sense — a *different* module
  (MIDI-CAT) also fails to behave correctly under this exact automation
  setup, though via a different symptom. That's suspicious: it may point at
  something about **synthetic `xdotool` input specifically** not being
  fully equivalent to real mouse/keyboard input for VCV's "click any
  parameter to map it" pattern (used by both modules) — worth testing with
  a **real** mouse click (i.e., you clicking it live via NoMachine) as the
  next real discriminating test, since that's the one variable neither
  module swap nor a fresh mapping controlled for.

## RESOLVED: the bridge works — the blocker was synthetic-click automation, not a module bug

Taylor mapped a few parameters by hand via the live NoMachine session (real
mouse clicks, MIDI-CAT, CC1→VCO Frequency, CC2→VCO FM, CC3→VCA Level — visible
in the live `maps` array in MidiCat's own module `data`). Sending a real MIDI
CC sweep (0→32→64→96→127→64→0) to the CC3 mapping and measuring actual
captured audio RMS from `vcv_loop.monitor` (not the periodic autosave file,
which is stale between its ~15s writes and gave a false "stuck" reading the
first time) showed a **clean, monotonic, real-time response**:
```
CC3=  0  RMS=    0.00
CC3= 32  RMS=    0.00      (near-silent at very low VCA level, expected)
CC3= 64  RMS= 3014.12
CC3= 96  RMS= 6029.75
CC3=127  RMS= 9042.26
```
Evenly-spaced RMS steps tracking evenly-spaced CC values — this is the real
closed loop working, audio changing live in response to external control
messages, through a completely off-the-shelf module (MIDI-CAT) with zero ML.

**Root cause of every earlier failure**: `xdotool`-synthesized clicks are not
fully equivalent to real mouse clicks for VCV's "click any parameter to pick
it as a mapping target" gesture, specifically. Screenshots (`import -window`),
menu clicks (audio device selection), and toggle buttons all worked fine
synthetically — but this one interaction pattern silently didn't fully
register, which is why:
- OSC'elot's *very first* learn on this session (that one time) probably
  benefited from some accidental timing/motion that happened to satisfy it,
  but no message after the bind ever "counted" as a fresh update — plausible
  in hindsight if the underlying widget-click detection is where the flakiness
  lives, not the OSC message handling itself.
- MIDI-CAT's target-click *never* attached in any of ~5 automated attempts
  (both click orders, plain click, click-drag) — same class of problem, just
  with zero lucky hits instead of one.
- A **real** click, by an actual human, worked immediately and reliably, on
  the first try, for all three mappings.

**Practical implication for future automation of this patch-building process**:
CV/audio routing, module placement, cabling, and audio-device selection can
all be done headlessly (patch-JSON authoring + `xdotool` menu clicks). The one
step that currently requires a human at the real mouse is *creating a new
parameter mapping* in a MIDI-CAT/OSC'elot-style module. Once a mapping exists,
driving it programmatically via OSC/MIDI works perfectly — this only blocks
the *one-time setup* of a patch, not the runtime control loop itself.

**Latency**: not yet cleanly benchmarked (a couple of measurement attempts
were confounded by PipeWire's monitor-sink idle/wake behavior and by
mistakenly measuring against the stale autosave file) — worth a properly
isolated pass later, but not a blocker: the sweep above shows response well
within the same second as the sent message, which is already fast enough to
plan a real control loop around.

## Open issue: OSC'elot mapped parameter doesn't update after the initial bind
**Superseded by the finding above** — kept for the debugging trail, but the
likely real explanation is the synthetic-click issue, not an OSC'elot-specific
value-application bug.

Confirmed via source read (`processOscMessage` / `process()` in
`Oscelot.cpp`, `CONTROLLERMODE::DIRECT` case — the default mode a new
mapping is created with): logic looks correct on paper (compares
`getValueIn()` vs `getCurrentValue()`, applies via `oscParam[id].setValue()`
if they differ, sends feedback when the display value changes). In
practice:
- The very first `/fader 1 0.5` sent to complete the "click parameter, then
  send an OSC message" learn handshake **does** get applied (parameter
  became "VCA-2 > Channel 1 Level", mapped).
- Every subsequent `/fader 1 <value>` message (tried: discontinuous jumps,
  a slow ramp through the current value, waiting 30+ seconds in case of an
  extreme slew setting) leaves the parameter frozen at that first bound
  value (`0.5`, confirmed via the live autosave JSON and a raw socket
  listener on the SEND-feedback port showing zero packets — i.e. this isn't
  a read-back bug, the module genuinely isn't seeing these as
  parameter-changing events).
- Not yet tried: OSC'elot's per-mapping right-click context menu (didn't
  register via synthetic right-click — may need a real click, or a
  different xdotool click-button argument); whether `mapLen`/`processDivider`
  timing plays a role; whether a *second* independent mapping (not reusing
  the same learn slot) behaves differently; filing/checking upstream issues
  for this exact symptom.
- **This is the one thing standing between "the bridge exists" and "the
  bridge actually works for real-time control."** Worth a fresh, focused
  debugging pass before building anything on top of it.

## Verification status (against the approved plan's success criteria)

- ✅ CV round-trip *feedback* mechanism exists and is real (OSC'elot's
  learn-completion handshake, confirmed).
- ✅ Audio capture works end-to-end once the sink/source defaults are set
  correctly (`parec`/`sounddevice` both confirmed real, non-zero, correlated
  signal from VCV's actual audio engine).
- ❌ **Sustained closed-loop CV control does not yet work** — a control
  message beyond the very first one doesn't move the mapped parameter. This
  blocks the actual "sweep CV, observe audio change, measure latency" test
  the plan called for.

## Phase 2: hardware-in-the-loop control — result

**Superseded by "Phase 2 final-review fix pass" below.** The section
immediately following this note is kept as the historical record of the
first end-to-end run, but its "best-guess diagnosis" (leading hypothesis:
undertrained model) turned out to be backwards — a final whole-branch
review found and reproduced a genuine correctness bug in `estimate_pitch`
that fully explains the pitch-channel failure and likely contributed to
brightness's, independent of how much training data existed. See the final
section for the corrected diagnosis, the fix, and the re-run result.

End-to-end verification of Tasks 1-7 (feature extraction, CV backend, data
collection, `InverseCVModel` trained on 500 real sweep samples with final
MSE 0.044, `TTInferenceEngine`, `run_control_loop`) against the live,
already-running VCV Rack instance (`bridge_test.vcv`, MIDI-CAT CC1/CC2/CC3
mapped, PipeWire routed to `vcv_loop`/`vcv_loop.monitor` — routing verified
via `pw-link -l` immediately before the runs, same check that caught the
Phase 1 silent misroute). Each run: `gozer run --chips 1 -- python3
control_loop.py <goal>`, default `step_fraction=0.3`,
`control_interval_s=0.1`, `max_iterations=100`.

Three goals, verbatim final printed line from each run:

```
goal: [0.2 0.5 0.2]   final: [0.20073959 0.13780009 0.79264883]
goal: [0.8 0.5 0.9]   final: [0.7918775  0.0735466  0.66985748]
goal: [0.5 0.5 0.5]   final: [0.48163114 0.14985584 0.8148791 ]
```

(feature order is `[loudness, brightness, pitch_norm]`, per `features.py`.)

**Honest read: partial convergence, one feature out of three.**

- ✅ **Loudness (dim 0) converges well in all three runs** — 0.201 vs 0.2,
  0.792 vs 0.8, 0.482 vs 0.5. Error under 0.02 every time, well inside the
  0.05 `convergence_threshold`. The one feature the model actually learned
  a usable inverse mapping for.
- ❌ **Brightness (dim 1) does not converge in any run**, and the failure
  has a suspicious shape: the goal was 0.5 in *all three* runs, and the
  measured result landed at 0.138, 0.074, and 0.150 — clustered low
  regardless of anything else in the goal vector. That's not noise around
  the target, it's a near-constant output that never moves toward 0.5.
- ❌ **Pitch (dim 2) does not converge in any run**, with the same
  "stuck regardless of goal" shape but in the opposite direction: goals of
  0.2, 0.9, and 0.5 all landed in the same narrow high band — 0.793, 0.670,
  0.815. Notably, run 2 asked for the *highest* pitch goal (0.9) and
  produced the *lowest* of the three pitch results (0.670) — the ordering
  isn't even monotonic with the goal, which rules out "right direction, not
  enough gain" and points more at "this channel isn't being driven
  correctly at all."

**Best-guess diagnosis** (not confirmed, just the most likely candidates
given what's visible from the outside):

1. **Undertrained model, and specifically undertrained on the harder two
   dimensions.** The training set is 500 of the ~3000 samples the plan
   called for (Task 5). Loudness (driven near-linearly by VCA level) is the
   easiest inverse to learn from a small sample; brightness and pitch
   depend on VCO FM/frequency interacting less linearly with the measured
   spectrum, and 500 samples may simply not cover that relationship well
   enough for a 3→32→32→3 MLP to generalize — consistent with brightness
   and pitch both collapsing toward a fixed output almost independent of
   the requested goal, which is what an undertrained regressor does when it
   has learned "the average case" rather than the actual mapping.
2. **Normalization constants from Task 2 (`features.py`) may not match this
   patch's real range.** `BRIGHTNESS_REF_HZ = 12000.0` and
   `PITCH_LOG_MAX_HZ = 4000.0` were chosen before this exact VCO/VCA patch
   was swept on real hardware. If the patch's actual achievable spectral
   centroid or pitch range is much narrower than assumed (e.g. the VCO's FM
   input barely moves the audible pitch, or the VCA's harmonic content
   never gets bright enough to approach 12kHz centroid), most of the sweep
   data would land in a compressed corner of the normalized [0,1] range,
   which would produce exactly the "same output regardless of goal"
   symptom seen here for both stuck dimensions.
3. **`step_fraction`/`control_interval_s` are probably not the cause.**
   Loudness converges cleanly under the same step size and interval used
   for the other two channels in the same run, so the control loop's
   step/timing parameters are not obviously the bottleneck — the failure
   looks like it's in the model's inverse mapping (or the CV→feature
   physical relationship it was trained on), not the control loop's step
   dynamics.

**Net assessment**: the full pipeline runs end-to-end against real
hardware with no exceptions, and the loudness channel demonstrates the
closed loop genuinely works — but two of three target features do not
converge, and do so in a way (goal-independent, non-monotonic) that looks
more like an undertrained/miscalibrated inverse model than a tuning
problem. This is not a "phase 2 complete" result; it's a working harness
with a model that needs more (and possibly better-normalized) training
data before it can be trusted to hit brightness and pitch goals.

## Phase 2 final-review fix pass — corrected diagnosis and result

A final whole-branch review of the completed 8-task plan reproduced a
genuine correctness bug behind the pitch-channel failure above, fixed it,
re-collected the sweep dataset, retrained, and re-ran the same live 3-goal
verification. Honest result: **meaningfully better, still not fully
converged.**

### The real root cause: `estimate_pitch` was broken at production block size

`features.py`'s `estimate_pitch` searched for the autocorrelation peak
starting at `lag_min = int(sample_rate / fmax)`. At `VCVRackBackend`'s real
`block_size` (1024 samples, not the 4096 the original test used),
`lag_min` (12, for `fmax=4000`) fell inside the raw autocorrelation's
still-descending zero-lag main lobe for any fundamental below ~440 Hz — so
the function returned exactly `fmax` (4000 Hz) as a constant for any
low/mid fundamental. Confirmed in the original 500-sample dataset: **52% of
the `pitch` column was exactly the 4000 Hz clip artifact**, and
`corr(vco_freq, pitch) = -0.29` — negative, i.e. the one feature meant to
track the VCO frequency knob moved the wrong way from it. This — not an
undertrained model — is the real explanation for pitch never converging
above.

**Fix** (`features.py`): find the first lag where the autocorrelation
actually stops decreasing (the main lobe's real end) and start the peak
search there, using the old `lag_min` (from `fmax`) only as a floor.
Verified with a new regression test at the real production block size
(1024, across 55/110/220/330/440 Hz) — reproduced RED against the old
code (4/5 frequencies clipped to exactly 4000 Hz), GREEN after the fix.

**Also fixed while investigating** (secondary, lower-confidence
contributors flagged by the same review):
`spectral_centroid` didn't remove the DC component before computing the
centroid, unlike `estimate_pitch` — added `block = block - block.mean()`.
The `BRIGHTNESS_REF_HZ`/`PITCH_LOG_MAX_HZ` normalization-mismatch
hypothesis from the first run (item 2 above) was **not** independently
confirmed or refuted this pass — brightness still doesn't converge after
the pitch fix (see below), so a real range mismatch for brightness
specifically remains a live, unconfirmed hypothesis, not resolved by this
work.

### Re-collected dataset: pitch column sanity check

Re-ran the same 500-sample random CV sweep against the same live patch,
with the fixed `estimate_pitch`:

| check | old (broken) dataset | new (fixed) dataset |
|---|---|---|
| `pitch_norm` values == 1.0 (fmax-clip artifact) | 52% | **2.4%** (12/500) |
| `corr(vco_freq, pitch_norm)` | **-0.29** (wrong direction) | **+0.797** (strong, right direction) |
| `pitch_norm` std | 0.157 | 0.317 (real spread across the full range) |
| unique feature rows | 500/500 | 500/500 (still no stale/repeated blocks) |

The pitch feature now does what it was always supposed to do: track the
VCO frequency control, strongly and in the correct direction.

### Retrained model: per-output-dimension MSE/R² on a held-out validation split

`model.py` now does an 80/20 train/val split (400/100 samples) and reports
each CV channel's held-out MSE and R² against a predict-the-training-mean
baseline, instead of one aggregate in-sample MSE. Retrained on the new
dataset:

| CV channel | held-out MSE | held-out R² (vs. mean baseline) |
|---|---|---|
| `vco_freq` | 0.0234 | **0.7085** |
| `vco_fm` | 0.0659 | **0.1001** |
| `vca_level` | 0.0168 | **0.8034** |

`vco_freq` and `vca_level` both learned a genuinely useful inverse mapping
(R² 0.71 and 0.80). `vco_fm` still learned almost nothing (R² 0.10) — this
is the same weak dimension flagged in the first pass, and it is **not**
simply an unpatched-input artifact: the live patch's actual cable list was
checked directly, and VCO's FM input **is patched**, to an LFO module
(module id `4396770612866059`, connected to VCO's `FM_INPUT`). The
low R² is more likely because that LFO's specific rate/depth just doesn't
move loudness/brightness/pitch much within a single ~21ms audio-block
snapshot — a slow-moving modulation source can be genuinely connected and
still be nearly invisible to a single-block feature read. This is reported
as an open, honest finding, not something this pass attempted to fix by
altering the patch topology.

### New live 3-goal verification (same 3 goals, corrected pipeline)

Live environment reconfirmed immediately before running (`pw-link -l`,
`pactl get-default-sink`/`get-default-source`, `ps -p 1366446` — all
unchanged from the first run). Each run: `gozer run --chips 1 --who
"claude:tt-cv-agency" -- python3 control_loop.py <goal>`, same defaults as
before (`step_fraction=0.3`, `control_interval_s=0.1`, `max_iterations=100`).

Three goals, verbatim final printed line from each run:

```
goal: [0.2 0.5 0.2]   final: [0.2832652  0.05413392 0.60557207]
goal: [0.8 0.5 0.9]   final: [0.78318624 0.3143487  0.94570313]
goal: [0.5 0.5 0.5]   final: [0.5146741  0.12664438 0.74638462]
```

(feature order is `[loudness, brightness, pitch_norm]`, per `features.py`;
`convergence_threshold` is 0.05.)

**Honest read: meaningfully better on pitch, still not fully converged.**

- **Loudness (dim 0): converges in 2 of 3 runs** (errors 0.017, 0.015 —
  down from 3/3 last time). Run 1's loudness error (0.083) is just over
  the 0.05 threshold this time — a small regression on this one run, most
  likely just sampling noise from retraining on a different 500-sample
  draw, not a regression this pass caused on purpose. Still the
  best-behaved channel overall.
- **Brightness (dim 1): still does not converge in any run**, and still
  clusters low (0.054, 0.314, 0.127 vs. goals of 0.5 in all three) —
  unchanged in character from the first run. The pitch fix did not fix
  brightness; the `BRIGHTNESS_REF_HZ` normalization-mismatch hypothesis
  from the first pass remains the most likely open explanation, still
  unconfirmed.
- **Pitch (dim 2): converges in 1 of 3 runs now (up from 0 of 3), and —
  more importantly than the pass/fail count — its relationship with the
  goal is now monotonic and directionally correct**: goal 0.2 → 0.606,
  goal 0.5 → 0.746, goal 0.9 → 0.946. The highest goal now produces the
  highest result and the lowest goal the lowest result, and the run with
  the highest goal (0.9) converges (error 0.046). Compare to the *original*
  run, where goal 0.9 produced the *lowest* of the three pitch results
  (0.670) — a non-monotonic, "not driven correctly at all" shape. That
  specific failure mode is gone. What remains is a consistent high bias
  (results run above goal at low/mid goal values) rather than an
  incoherent one — consistent with a real, correctly-signed but
  imperfectly-calibrated inverse mapping (R² 0.71 on `vco_freq`, not 1.0),
  not a broken feature.

**Net assessment**: the pitch-estimator bug is fixed and confirmed fixed
by three independent lines of evidence — the dataset's clip-artifact rate
(52%→2.4%), its correlation sign (-0.29→+0.797), and the live control
loop's pitch channel going from non-monotonic/goal-independent to
monotonic/goal-tracking. This was a real, reproducible correctness bug,
not a training-data-volume problem, and fixing it changed the live
behavior in exactly the direction predicted. It is not, however, a full
fix for Phase 2: brightness still does not converge in any run (a
separate, still-unconfirmed normalization question), and pitch, while now
behaving coherently, converges in only 1 of 3 runs — the underlying
inverse model's R² for `vco_freq` (0.71) leaves real residual error even
with a correct feature signal. `vco_fm`'s weak R² (0.10) is now understood
to most likely be a modulation-rate/single-block-visibility limitation of
the actually-patched LFO, not a missing connection. **This is still not a
"phase 2 complete" result** — it is a confirmed, fixed correctness bug plus
an honestly-reported partial improvement, with brightness's root cause
still open for a future pass.

## Follow-up session (2026-09-07): MIDI-CAT mappings were never actually
committed, plus the brightness recalibration

Picking up the two open items from the previous pass ("recollect the full
sweep now that pitch is fixed" and "recalibrate `BRIGHTNESS_REF_HZ` against
real data") surfaced a bigger, more basic gap first.

**Discovery: `patches/bridge_test.vcv` never had a working CV mapping.**
Relaunching VCV Rack from the committed patch to start a fresh collection
run showed only `VCO → VCA → AudioInterface2 + OSCelot` — no MIDI-CAT
module, matching `git log -- patches/bridge_test.vcv` (exactly one commit,
the initial one). The MIDI-CAT module and its CC1/2/3 mappings that every
earlier session verified against were added to a *live, running* Rack
instance and never saved back into the committed `.vcv` file — they existed
only in that process's memory and were lost the moment it was cleanly
closed at the end of the previous session. A fresh clone of this repo could
launch the committed patch and get zero working CV control. Also lost along
with it: whatever LFO patching had been feeding `vco_freq`'s FM input in
that same live session (mentioned in the last pass's `vco_fm` discussion) —
the freshly-relaunched clean patch has nothing wired to the VCO's FM input
at all, so that specific prior finding no longer describes the current
patch state.

**Fix: hand-authored the MIDI-CAT module directly into patch.json, no UI
mapping-clicks required.** The same "VCV patches are hand-authorable JSON"
fact from Phase 1 turns out to extend past adding modules to actually
*mapping* them, which matters here because the "click a parameter to arm a
mapping" gesture is the one specific interaction already confirmed broken
for synthetic input in this environment. Recipe, fully scriptable:

1. Read `Fundamental`'s `VCO.cpp`/`VCA.cpp` source directly (`gh api
   repos/VCVRack/Fundamental/contents/src/VCO.cpp -H "Accept:
   application/vnd.github.raw"`) to get each `ParamIds` enum's declaration
   order — that order *is* the numeric `paramId` MidiCat needs. For this
   patch: VCO `FREQ_PARAM`=2, `FM_PARAM`=4; VCA `LEVEL1_PARAM`=0 (confirmed
   channel 1 is the one wired, via the patch's own `cables` list).
2. Read `stoermelder/vcvrack-packone`'s `MidiCat.cpp` `dataToJson`
   (`src/modules/midicat/MidiCat.cpp`) for the `maps` array schema — only
   `cc`, `ccMode`, `moduleId`, `paramId`, `label` are needed per mapping;
   everything else (`min`/`max`/`slew`/`curve`/etc.) defaults sanely when
   omitted, matching the plugin's own shipped `presets/MidiCat/cc01-32.txt`.
3. Append a `Stoermelder-P1`/`MidiCat` module object to `patch.json`'s
   `modules` array with those three maps, repack
   (`tar --zstd -cf out.vcv -C dir patch.json`), relaunch. Confirmed via
   screenshot: all three mappings (`vco_freq`/`vco_fm`/`vca_level`) showed
   up correctly labeled in the module's UI immediately — no click needed.
4. The MIDI **device** selection (`In: ALSA` / `(No device)`) is a
   *different* widget than the mapping-target click, but empirically it's
   equally unresponsive to synthetic `xdotool` clicks here (tried: plain
   click, focus-then-click, held mousedown with a screenshot mid-hold — no
   effect in any case). Rather than ask for a manual click, this also
   turned out to be hand-authorable: `midi::Port::toJson()`
   (`VCVRack/Rack`'s `src/midi.cpp`) writes `{"driver": <id>, "deviceName":
   <string>, "channel": <int>}`, matched back by *name* on load
   (`fromJson` searches the driver's device list for a matching
   `deviceName`), and RtMidi's ALSA backend builds that name as
   `"<client>:<port> <client#>:<port#>"` (`MidiInAlsa::getPortName` in
   `thestk/rtmidi`'s `RtMidi.cpp`) — the exact same string `mido` already
   reports (`"Midi Through:Midi Through Port-0 14:0"`). Writing
   `{"driver": 2, "deviceName": "Midi Through:Midi Through Port-0 14:0",
   "channel": -1}` directly (driver id `2` = `RtMidi::Api::LINUX_ALSA`,
   confirmed against `thestk/rtmidi`'s `RtMidi.h` enum and matching what a
   fresh MidiCat module already defaulted to) resolved correctly on load —
   confirmed both by the saved-back autosave JSON and by a screenshot
   showing the device name (not "(No device)") in the module's own display.
5. **Verified for real, not just "no error thrown"**: a live CC sweep on
   `vca_level` (CC3, with CC1/CC2 held at mid) via `mido`, measuring actual
   captured RMS via `sounddevice`, produced a clean monotonic response
   (cc=0→0.0, 32→0.091, 64→0.183, 96→0.275, 127→0.363) — matching the
   Phase-1-era numbers closely enough to confirm this is the same real
   signal path, not a coincidence.

Committed the working patch as `patches/bridge_test_mapped.vcv` (kept
`bridge_test.vcv` itself unmodified, still useful as a clean slate for any
future OSC'elot investigation). **Anyone reproducing this project from a
fresh clone should launch `bridge_test_mapped.vcv`, not `bridge_test.vcv`.**

**Brightness recalibration.** Before recollecting data, ran a 150-sample
random-CV recon pass (`VCVRackBackend` + `features.rms`/`spectral_centroid`
directly, no normalization) to measure this patch's actual achievable
ranges instead of continuing to guess:

| feature | median | p95 | max |
|---|---|---|---|
| RMS | 0.184 | 0.337 | 0.365 |
| spectral centroid (Hz) | 366 | 4167 | 6150 |

`LOUDNESS_REF_RMS=0.4` was already well-calibrated (max observed 0.365,
~10% headroom). `BRIGHTNESS_REF_HZ=12000.0` was not — real brightness
values could only ever reach ~0.51 normalized, permanently compressing the
usable range. Changed to `7000.0` (~14% headroom above the observed max,
matching loudness's margin) in `features.py`.

**Recollected the full dataset.** `data_collection.py --main--`'s
`n_samples=3000` default had never actually been used (every prior run used
500 samples via ad hoc overrides) — this pass finally used it, with both
fixes (pitch estimator, brightness constant) in place and the MIDI-CAT
mapping actually working from a cleanly relaunched patch.

**Retrained on 3000 samples (2400 train / 600 held-out), per-channel R²
against a predict-the-mean baseline:**

| CV channel | held-out MSE | held-out R² | previous R² (500 samples) |
|---|---|---|---|
| `vco_freq` | 0.0040 | 0.9540 | 0.7085 |
| `vco_fm` | 0.0819 | -0.0046 | 0.1001 |
| `vca_level` | 0.0008 | 0.9899 | 0.8034 |

`vco_freq` and `vca_level` both improved substantially with 6x the data.
`vco_fm`'s R² is now essentially zero (not just low) — and this time that's
fully explained, not just suspected: the freshly-rebuilt
`bridge_test_mapped.vcv` has **nothing patched into the VCO's FM input at
all** (confirmed via the patch's own `cables` list — empty for that port).
With no CV present at `FM_INPUT`, the `FM_PARAM` knob multiplies against a
constant zero and has no effect on any measured feature, so an R² of ~0 is
the *correct* result for this channel in this patch, not a modeling
failure. (The previous pass's "confirmed patched to an LFO" finding
described a different, never-persisted live session — see the discovery
above. It no longer describes this patch.)

**Live 3-goal verification, same procedure as before** (`gozer run --chips
1 --who "claude:tt-cv-agency" -- python3 control_loop.py <goal>`, defaults
unchanged):

```
goal: [0.2 0.5 0.2]   final: [0.25722776 0.01726999 0.32628938]
goal: [0.8 0.5 0.9]   final: [0.81281436 0.41661117 0.91326825]
goal: [0.5 0.5 0.5]   final: [0.55074932 0.0689186  0.57509306]
```

Per-goal absolute error, compared to the previous (500-sample) pass:

| goal | loudness err | (prev) | brightness err | (prev) | pitch err | (prev) |
|---|---|---|---|---|---|---|
| `[0.2,0.5,0.2]` | 0.057 | 0.083 | 0.483 | 0.446 | 0.126 | 0.406 |
| `[0.8,0.5,0.9]` | 0.013 | 0.017 | 0.083 | 0.186 | 0.013 | 0.246 |
| `[0.5,0.5,0.5]` | 0.051 | 0.015 | 0.431 | 0.373 | 0.075 | 0.046 |

**Honest read:** every dimension tightened noticeably in absolute terms —
pitch error dropped by roughly half to two-thirds on two of the three
goals, and brightness's best case (goal 2) nearly halved its error (0.186 →
0.083) — but the pass/fail count against the 0.05 convergence threshold
barely moved (loudness/pitch still converge on only 1 of 3 goals each,
brightness still 0 of 3), because two of the three goals now land just
*outside* the threshold rather than far outside it. `vco_freq` and
`vca_level`'s R² jump (to 0.95 and 0.99) shows up as tighter tracking
everywhere; brightness stays the weak dimension, and the recon data
suggests part of that may be a genuinely hard target, not just a modeling
gap — the real achievable spectral-centroid distribution is heavily
skewed (median 366 Hz, i.e. ~0.05 normalized, vs. a 6150 Hz observed max),
so a *mid-range* brightness goal like 0.5 may correspond to a narrow, less
densely sampled band of the real CV space rather than a typical setting.
`vco_fm` remains fully unobservable in the current patch (see above) —
patching a real modulation source into its FM input (e.g. an LFO) would be
a reasonable next step if that channel's control is ever needed for real,
but is out of scope for this pass.

## Minimoog-equivalent patch (2026-09-07): a proper subtractive-synthesis
foundation, replacing the ad hoc single-VCO test patch

User asked for a system "more or less the equivalent of a Buchla Easel
(West Coast) or a Minimoog (East Coast)" as a foundation for future
experiments. Checked the actual VCV Library and — better — the plugin
packs already installed on this machine (`ls
~/.local/share/Rack2/plugins-lin-x64/`): `Fundamental`, `Befaco`,
`AudibleInstruments`, `ESeries`, `Grayscale`, `AmalgamatedHarmonics`,
`Kilpatrick-Toolbox`, `Stoermelder-P1` — enough for either architecture
with zero new installs. Recommended and built the Minimoog (East Coast)
first: it reuses exactly the CV-target types already validated in this
project (continuous, monotonic — filter cutoff especially), where the
Easel's complex-oscillator/wavefolder/lopass-gate West Coast approach
would need genuinely new, untested modules and a real design compromise
(no true vactrol LPG is installed).

**Signal path**: 3× Fundamental `VCO` (saw, saw, sub-octave square) →
`Mixer` → `VCF` (24dB multimode, lowpass out) → `VCA` (slug `VCA`, the
same 2-channel module the original bridge_test patch used). `ADSR`/`LFO`
were deliberately left out of the CV-automated path: they're
gate/trigger-driven, and this project's whole control model is
"set a continuous [0,1] CV value and hold it" (`data_collection.py`,
`control_loop.py`) — an ADSR needs a `GATE_INPUT` transition to do
anything, which doesn't exist anywhere in this architecture yet. Wiring
one in would mean sampling audio at an arbitrary, inconsistent phase of
each envelope cycle. Both modules are available in Fundamental for a
future gate-triggered experiment; this patch just doesn't use them.

**New patch**: `patches/minimoog_test.vcv`, built the same
hand-authored-JSON way as `bridge_test_mapped.vcv` — every module's
`ParamIds`/`InputIds`/`OutputIds` enum order confirmed against
`VCVRack/Fundamental`'s actual source (`VCO.cpp`, `VCF.cpp`, `Mixer.cpp`,
`8vert.cpp`) before writing any JSON, not guessed. New config:
`configs/minimoog_test.yaml` (same `vco_freq`/`vcf_cutoff`/`vca_level`
channel shape as before — `vcf_cutoff` replaces the old patch's
`vco_fm`, which had nothing patched into its FM input and did nothing
audible; the filter's own cutoff is a real, always-effective brightness
control instead).

**A real mistake, caught by verification, not assumed away.** First
attempt tried to make VCO2 (+7 semitones) and VCO3 (-12 semitones) track
VCO1's pitch by mapping the *same* MIDI-CAT CC to three separate
`(moduleId, paramId)` slots at once, each with a shifted `min`/`max`
sub-range (band-shifting math to get a constant relative-semitone offset
at every CC value — mathematically sound on paper). Audio sweep
looked plausible at first (centroid/RMS changed smoothly with the CC).
But directly reading the raw `FREQ_PARAM` values back from a *fresh*
autosave (confirmed fresh via `stat`, not just "waited a bit") after
sending CC=0 and then CC=127 showed **the identical raw value both
times** — the multi-slot-per-CC mapping doesn't actually track reliably
(most likely a CC→slot dispatch collision inside MIDI-CAT when multiple
slots share one CC number; not fully root-caused in
`stoermelder/vcvrack-packone` source, and not worth fully root-causing
given a clean fix existed). This is exactly the project's own "trust the
subject, verify the instrument" principle: the *audio* evidence alone
would have been believed as confirmation, and would have been wrong.

**Fix**: added `Fundamental` `8vert` (an attenuverter module whose own
description says it "creates constant voltages" when its input is
unpatched — confirmed in source: `in[16] = {10.f}` is the default with
nothing patched in). One row's gain param, MIDI-CAT-mapped (a single,
ordinary one-CC-to-one-param slot, the already-proven pattern), produces
a shared CV that's cabled into all three VCOs' `PITCH_INPUT` (confirmed
via source that VCO pitch is `FREQ_PARAM/12 + PITCH_INPUT`, true 1V/oct
addition). Each VCO's own `FREQ_PARAM` knob stays fixed at its static
detune offset; the shared CV transposes all three together. Re-verified
with the same two methods that had disagreed the first time: a CC sweep
now shows clean, monotonic, ~doubling-per-quartile pitch (32.7 → 65.8 →
132.2 → 265.2 → 521.7 Hz across cc=0/32/64/96/127 — consistent with real
1V/oct exponential scaling), and this time a fresh raw-param read agrees.

**Verified all three channels for real** (CC sweep + measured
RMS/centroid/pitch, `cutoff` and `level` held at fixed midpoints/maxima
while sweeping each target channel):

| channel | sweep behavior |
|---|---|
| `vco_freq` | pitch 32.7→521.7 Hz, clean monotonic 1V/oct doubling |
| `vcf_cutoff` | rms 0.0003→0.201, centroid 113→2534 Hz, monotonic |
| `vca_level` | rms 0.0000→0.252, monotonic |

Not yet done: any data collection, retraining, or live control-loop
verification against this new patch — this session only built and
verified the instrument itself, per the user's ask for "a logical
default patch to use as the foundation for our next experiments." The
existing `data_collection.py`/`model.py`/`control_loop.py` code is
patch-agnostic (reads `configs/*.yaml` + whatever `.vcv` is currently
running) so pointing them at this new patch needs no code changes —
just launch `minimoog_test.vcv` and pass `configs/minimoog_test.yaml`
instead of the bridge_test files.
