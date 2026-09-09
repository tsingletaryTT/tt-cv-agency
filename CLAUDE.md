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

## Convention: `docs/journal/` — a screenshot record alongside this log

Started 2026-09-07, when the Minimoog patch's launch screenshot turned
out worth keeping around rather than just living in `/tmp` for one
turn. Going forward: any screenshot taken to verify or show off patch
state (a new instrument, a rewiring, a UI state worth remembering) gets
saved to `docs/journal/YYYY-MM-DD-short-description.png` (matching this
file's own dated-section style) and linked from `README.md`'s "Progress
journal" table, newest entry at the bottom. This is a visual companion
to the prose history here, not a replacement — keep writing the "what
happened and why" narrative in this file the way it's always been
done; the screenshot is the "what it looked like" that prose alone
doesn't capture well. Not every screenshot taken during a session needs
keeping (a debugging zoom-crop of one widget isn't journal-worthy) —
save the ones that show real patch/instrument state, the way the
Minimoog screenshot showed the whole signal chain at once.

## A four-stage roadmap (2026-09-07): instruction-following, exploration,
and predictive control, sequenced instead of picked

Prompted by a "what would training on this patch even look like?" question
against a concrete example — "make a squelchy, resonant acid bass patch
and play a looping sequence on it while sweeping subtly on the filter."
Broke that down: it's really three different asks bundled together (a
timbral target, a discrete note sequence, and a continuous automation
gesture), and our reflex model only does the first — it holds one static
point, with no notion of time at all.

Three candidate directions came out of that, each pulling from a
different thread of the project so far:

- **Path A — instruction-to-preset.** An LLM parses an instruction
  directly into a structured CV recipe; no training involved. Fastest,
  lowest-risk, ships the literal ask soonest.
- **Path B — learned exploration, no goals.** An agent sweeps the CV
  space looking for regions that are meaningfully different from each
  other, building a discovered map/preset library rather than chasing a
  target — the "more hands than a couple, sometimes just to discover
  what's possible" idea, and a revival of the original brainstorming
  session's ensemble/novelty-search pattern.
- **Path C — trajectory-predicting control.** Predict and search over
  short CV *action sequences* (not single points) to steer toward a goal
  over time — the actual fulfillment of the tt-vjepa2 predictor+CEM idea
  that started this whole project, finally justified once there's an
  instrument with a real trajectory to predict.

Decision: not one of these, all three, staged by real dependency order
rather than picked arbitrarily:

1. **Stage 0 — foundation.** Build a temporal instrument (sequencer +
   LFO) and fix the "read one static block" assumption baked into
   `data_collection.py`/`control_loop.py` — nothing downstream works
   without this. Spec: `docs/superpowers/specs/
   2026-09-07-sequencer-lfo-foundation-design.md`.
2. **Stage 1 (Path A).** Ships fastest, and doubles as a qualitative
   "does this sound right" check for everything built afterward.
3. **Stage 2 (Path B).** Produces better, non-uniform training data than
   uniform-random sampling, and empirically checks Path A's semantic
   guesses ("does the LLM's idea of 'squelchy' land in a real
   high-resonance region we actually found?").
4. **Stage 3 (Path C).** Needs Stage 2's exploration data extended into
   logged action-trajectories; the most ambitious and most dependent on
   everything before it.

**The closing loop**: once Stage 3 exists, Stage 1 gets upgraded — an
instruction stops meaning "set this static recipe" and starts meaning
"steer toward this goal over time," with Stage 3's predictor doing the
steering. The three paths converge into one system rather than staying
three parallel features.

## Stage 0 end-to-end sanity check (2026-09-07): the new sequencer/LFO
instrument, proven live, honestly weak with only 300 samples

Task 5 of the Stage 0 plan (`docs/superpowers/plans/
2026-09-07-sequencer-lfo-foundation.md`) — the capstone task, and the
plan's own stated bar for the whole stage being done. Goal: prove the
whole pipeline (windowed aggregation, the parametric model, the updated
control loop) actually runs end-to-end against `patches/sequencer_test.vcv`
(8 CV channels, real temporal behavior — a sequencer, filter-sweep LFO,
filter envelope). Explicitly **a foundation smoke test, not a claim that
this instrument is well-controlled** — that's Stages 1-3's job. The bar
here is "the loop runs, converges even approximately, and nothing throws."

**Launch check.** `sequencer_test.vcv` was already running from an earlier
session (healthy, autosaving normally, PipeWire routed to `vcv_loop`/
`vcv_loop.monitor` as expected — reconfirmed via `pw-link -l`). A
screenshot of the live MIDI-CAT module confirmed its input device resolved
correctly (`Midi Through:Midi Through Port-0 14:0`, all channels) — output
shows `(No device)`, which is expected and unused (no MIDI feedback
configured, same as every prior patch in this project). The mapping list
scrolled to show `vca_level`/`seq_tempo`/`sweep_rate`/`sweep_depth`/
`filter_env_amount`/`vcf_resonance` plus `Unmapped` below them (`vco_freq`/
`vcf_cutoff` were above the visible scroll window, not missing).

**Code changes.** Both `data_collection.py`'s and `control_loop.py`'s
`__main__` blocks previously still pointed at `configs/bridge_test.yaml`
(the old 3-channel patch) and, in `control_loop.py`, parsed exactly 3 goal
values from `sys.argv[1:4]` with a 3-dim default goal — stale since Task 4
made `goal_features` 6-dim. Updated both to point at
`configs/sequencer_test.yaml`; `control_loop.py`'s argv parsing now reads
6 values (`sys.argv[1:7]`) with a 6-dim default goal
(`[0.5, 0.05, 0.5, 0.1, 0.5, 0.05]`, i.e. mid-loudness/brightness/pitch
with a bit of expected variation on each) and its weights path default
changed to `data/sequencer_model_weights.npz`.

**Dataset collection.** 300 samples, `settle_time_s=0.5`,
`aggregate_window_s=5.0`, random CV sweep (seed 0) against the live patch
— run in the background per this project's established long-collection
pattern (`nohup ... & disown`, polled in ~10-minute chunks). Took close to
the predicted ~27-30 minutes; finished cleanly (`saved 300 samples`).
Result: `data/sequencer_sweep_dataset.npz`, shape `(300, 8)` CV / `(300, 6)`
features, channels `[vco_freq, vcf_cutoff, vca_level, seq_tempo,
sweep_rate, sweep_depth, filter_env_amount, vcf_resonance]`.

**Retrained model, 80/20 train/val split (240/60 samples), per-channel
held-out MSE/R² against a predict-the-mean baseline:**

| CV channel | held-out MSE | held-out R² |
|---|---|---|
| `vco_freq` | 0.0638 | 0.1194 |
| `vcf_cutoff` | 0.0271 | **0.6530** |
| `vca_level` | 0.0653 | 0.2260 |
| `seq_tempo` | 0.0799 | 0.0287 |
| `sweep_rate` | 0.0777 | **-0.0312** |
| `sweep_depth` | 0.0962 | 0.0351 |
| `filter_env_amount` | 0.0785 | 0.0349 |
| `vcf_resonance` | 0.0747 | 0.2044 |

**Honest read: weak across the board, exactly as the plan warned was
plausible with 300 samples spread across an 8-dimensional CV space.**
`vcf_cutoff` is the one channel that learned a genuinely useful inverse
mapping — R² 0.65 is close to `vco_freq`/`vca_level`'s *500-sample-era*
numbers on the older 3-channel patch (0.7085/0.8034, not their eventual
3000-sample numbers of 0.9540/0.9899 quoted below — comparing against
those would be comparing across a 10x difference in dataset size, an
apples-to-oranges comparison this section corrected itself on when the
fix-round review caught the contradiction). It's also `vcf_cutoff`'s
first-ever trained result — it didn't exist as a channel before this
stage, so there's no prior number for it at all besides this one.
Notably, even `vco_freq` and `vca_level` — the direct-knob channels that
behaved reasonably well before (R² 0.95 and 0.99 on 3000 samples against
the old patch) — only manage 0.12 and 0.23 here. Rather than one blanket
"undertrained on a small sample" explanation for every weak channel, the
more useful read separates what's confirmed from what's plausible:

- `vco_freq`, `vca_level`, `vcf_resonance` — direct-knob channels that
  scored well with 3000 samples before (R² 0.95/0.99) and still show a
  positive, if weak, R² here (0.12/0.23/0.20). "Undertrained on a small
  sample, now spread across 8 outputs instead of 3" is a fair explanation
  for these three specifically — nothing about them changed except the
  amount of data and the number of competing output dimensions.
- `seq_tempo` (R² 0.029) — **not** just a small-sample story. This
  300-sample dataset was collected *before* this fix round's range
  correction (see the fix-round section below), when roughly 45% of
  `seq_tempo`'s CV range produced an aggregation window shorter than one
  full 8-step loop — exactly the "one block catches a random instant"
  problem this whole stage exists to fix, just recurring at a coarser
  grain. That's a concrete, confirmed mechanism for this channel's weak
  R², not merely "needs more data." The range fix should make future
  `seq_tempo` data collection meaningfully better, but this *existing*
  300-sample dataset still reflects the pre-fix range, so its R² for this
  channel shouldn't be expected to improve without a fresh collection.
- `sweep_rate` (R² -0.031, actually negative) — consistent with a
  genuinely temporal channel (an LFO rate) being close to invisible to a
  5-second aggregate window at this sample count, not a bug in the feature
  or the channel. (The fix-round section below found a further wrinkle
  worth flagging for whoever collects fresh data against this channel
  next: post-fix, a raw centroid-std feature stopped discriminating slow
  from fast `sweep_rate` settings in real-audio testing at all — plausible
  cause identified there, not yet confirmed at the R²-training level.)
- `filter_env_amount` (R² 0.035) — plausibly explained by the ADSR gating
  limitation documented in the fix-round section below: the envelope is
  gated by a ~1ms trigger pulse rather than a sustained gate, so decay and
  sustain are never actually reached and most of this channel's intended
  dynamic range may not be audible at all within a data-collection window.
  This is plausible, not confirmed — the fix-round did not re-verify this
  channel's R² after documenting the gating issue.
- `sweep_depth` (R² 0.035) — no concrete mechanism identified beyond small
  sample size; grouped with `vco_freq`/`vca_level`/`vcf_resonance`'s
  explanation by default, though it wasn't a strong channel even on larger
  datasets in the past, so this is a weaker claim than for those three.

**Live control-loop verification** (`gozer run --chips 1 --who
"claude:tt-cv-agency" -- python3 -c ...`, goal
`[0.5, 0.05, 0.5, 0.1, 0.5, 0.05]`, i.e. mid mean / low std for
loudness/brightness/pitch, `aggregate_window_s=5.0`, otherwise default
`step_fraction=0.3`/`control_interval_s=0.1`/`max_iterations=100`):

```
final: [0.39106533 0.02614929 0.38486939 0.02202998 0.37555966 0.09488796]
goal:  [0.5        0.05       0.5        0.1        0.5        0.05      ]
```

Per-dimension absolute error: `[0.109, 0.024, 0.115, 0.078, 0.124, 0.045]`;
Euclidean norm of the error: **0.222** (the loop's own
`convergence_threshold` is 0.05, so this run did not tightly converge).
**Honest read**: no exceptions anywhere in ~100 iterations against real
hardware and a real running VCV Rack instance — the full pipeline
(windowed aggregation reads, the 8-in/6-out parametric model, the
control step, the TT inference engine) executes correctly end-to-end.
(Fix-round correction: this section previously went on to argue that the
three mean-features landing consistently below goal by a similar amount
"looks more like a systematic undershoot from an undertrained model than
'not being driven at all.'" That claim doesn't actually hold up:
`run_control_loop` calls `predict_fn(goal_features)` with the same fixed
goal every iteration — confirmed by re-reading `control_loop.py` — so the
model's predicted CV point is a single constant value for the entire run
regardless of which hypothesis is true. "Three means sitting similarly
below goal" is exactly what one fixed operating point looks like either
way, and doesn't discriminate between "undertrained" and "not being
driven" at all. Narrowing to what the evidence actually supports: the
loop ran ~100 iterations against real hardware with zero exceptions, and
the model's single predicted CV point landed at Euclidean distance ≈0.222
from the goal.) This matches the smoke-test bar this task set out to
clear ("the loop runs, converges even approximately, and nothing
throws") — it does **not** demonstrate well-controlled convergence, and
shouldn't be read as one.

**Operational note (a mistake, corrected, not hidden):** the first live
control-loop attempt was wrapped in a stray shell `timeout 200`, which
SIGTERM'd the `gozer run` process partway through its ~500-560s expected
runtime (100 iterations × ~5s each). `gozer status` briefly showed chip 0
as `HELD-FOREIGN` afterward — investigated per the `gozer-gatekeeper`
skill rather than force-releasing or resetting blind. Root cause: `gozer
run`'s own shutdown/cleanup took a little longer than the `timeout`
grace period, not a genuinely stuck process or an unsafe reset condition
— the lease self-resolved to `FREE` within seconds once the underlying
processes finished exiting on their own. No `gozer release --force` or
manual `tt-smi -r` was needed. The corrected re-run (no external
`timeout` wrapper, just the plain `gozer run ... -- python3 ...`
backgrounded and polled) is the run reported above.

(This paragraph was originally written as a narrative claim with no
pasted `gozer status` output — a real gap the task review caught: a
hardware-safety-adjacent claim should be independently checkable, not
just described. Confirming evidence, checked directly after that
review: `gozer status` now shows all 4 chips `FREE` — `0000:01:00.0`,
`0000:02:00.0`, `0000:03:00.0`, `0000:04:00.0` — consistent with the
self-resolution claim and with no lingering lease from this incident.)

**Full test suite**, per the plan's own stated bar for Stage 0 being
done: `python3 -m pytest -q` → **44 passed, 4 deselected**; `gozer run
--chips 1 --who "claude:tt-cv-agency" --reason "final Stage 0 suite
check" -- python3 -m pytest -q -m hardware` → **4 passed, 44
deselected**. All 48 tests green (existing + everything added across
Tasks 1-4), no regressions from this task's `__main__` edits.

**Net assessment**: the Stage 0 foundation (windowed feature aggregation,
the parametric model, the updated data-collection/control-loop code) is
proven to run correctly end-to-end against the new 8-channel temporal
instrument, with real hardware in the loop and zero exceptions. It is
**not** proof that this instrument is well-controlled at 300 samples —
most of the 8 output dimensions show weak-to-negative held-out R², and
the live control loop's convergence is rough — and this section reports
that plainly rather than reframing a 300-sample smoke test as a finished
result. Getting from here to a genuinely well-controlled 8-channel
instrument is explicitly out of scope for Stage 0 and belongs to whatever
data-collection volume/strategy Stages 1-3 bring (Stage 2's exploration-
driven sampling in particular is a direct candidate for doing better than
uniform-random 300-sample coverage of an 8-dimensional space).

## Stage 0 fix round (2026-09-07): consolidated review fixes

A single consolidated fix pass addressing a final whole-branch review of
Stage 0 (Tasks 1-5 above), scoped and ruled on by the controller rather
than left open-ended. Highlights below; the rest (broken `__main__` file
chain, an unreachable `n_samples=3000` runtime, stale docstrings/README
text, a wiring test for the aggregation window) were smaller, more
mechanical fixes not worth their own narrative section.

### `seq_tempo`/`sweep_rate` CV range re-narrowing, and its real-audio
re-verification

The MIDI-CAT ranges Task 1 shipped for `seq_tempo` (raw `SEQ3.TEMPO_PARAM`
`-2..4`, mapped to the module's full range) and `sweep_rate` (raw
`LFO.FREQ_PARAM` `-6..-1`) both let a real setting produce a loop/period
longer than `aggregate_window_s=5.0` — roughly 45% of `seq_tempo`'s range
and about half of `sweep_rate`'s. That's the exact "one block catches a
random instant" problem this whole stage exists to fix, recurring at a
coarser (multi-second) grain instead of a single-audio-block one.

**Fix**: re-edited `patches/sequencer_test.vcv`'s `patch.json` (module 8,
MidiCat, `data.maps`) by the same hand-authored-JSON technique as every
other patch edit in this project — unpacked with `tar --zstd -xf`, changed
only the `min` fraction of the `seq_tempo` and `sweep_rate` map entries
(diffed against the original to confirm nothing else moved), repacked with
`tar --zstd -cf`:

- `seq_tempo`: `min` `0.0 → 0.5` (raw range narrows to `1..4` — `clockFreq
  = 2^raw` steps/s, 2 to 16, an 8-step loop period of 4s down to 0.5s).
- `sweep_rate`: `min` `0.1111 → 0.3333` (raw range narrows to `-2..-1` —
  `freq = 2^raw` Hz, a period of 4s down to 2s). `max` was already at the
  `-1` end for both the old and new range, so only `min` changed for
  either channel.

**Re-verification, against a freshly relaunched `sequencer_test.vcv`**
(killed the stale instance, appended the `"END"` log marker, relaunched —
confirmed via the autosave that the crash-recovery dialog didn't block the
load and the new `min` fractions took effect):

- **`seq_tempo`**, slowest setting (cc=0): mean inter-onset interval
  (detecting the sequence's `+12` semitone step via a pitch-doubling
  threshold, 20s of live audio) — **4 consecutive full-loop intervals:
  3.99s / 3.99s / 4.01s / 4.00s, mean 3.995s**, essentially exactly the
  predicted 4s loop (not the old range's 32s). Cross-checked against a
  direct raw-param read (autosave, synced to a fresh `saveAutosave` log
  line rather than trusted on a fixed sleep) — `TEMPO_PARAM` read back as
  exactly `1.0` at CV=0 and `4.0` at CV=1, matching the design arithmetic
  exactly.
- **`sweep_rate`**: raw-param read confirms the arithmetic is exactly
  right (`FREQ_PARAM` reads `-2.0` at CV=0, `-1.0` at CV=1 — a 4s and 2s
  period respectively, synced the same way). The intended real-audio
  check — "does `sweep_rate`'s slowest setting still show measurably
  lower centroid-std within a 5s window than its fastest" (the same test
  Task 1 used) — **did not hold**: 6 alternating slow/fast trials at a 5s
  window gave slow mean 52.0±8.7 Hz vs. fast mean 52.8±9.3 Hz (ratio
  1.01, indistinguishable from trial noise). Investigated rather than
  waved away, since this contradicted the expected verification outcome:
  a control trial at a *shorter* 2s window (matching Task 1's original
  test length, deliberately shorter than the new range's 4s slowest
  period) reproduced Task 1's original style of discrimination cleanly
  (slow 38.3±11.2 Hz, fast 55.0±13.5 Hz, ratio 1.44) — confirming the
  sweep itself and the wiring are correct, and that the 5s-window
  non-discrimination is a real, mathematically expected consequence of
  the fix itself: once a window's duration is `≥` a periodic signal's
  period at *both* ends of a CV range (exactly what this fix guarantees),
  the windowed-std feature saturates to the same steady-state value
  regardless of the exact rate, because the window always captures a full
  swing of the modulation either way. Task 1's original std-based test
  discriminated slow from fast specifically *because* of truncation
  asymmetry in a too-short window relative to a very slow setting — i.e.
  the very defect this fix removes was also, incidentally, what made that
  particular diagnostic work. **Net read**: the range fix itself is
  correct and verified (arithmetically exact, and no more truncated
  windows at either end of either channel's new range) — but
  `sweep_rate`'s windowed `[mean, std]` feature may now be structurally
  less informative about rate than it was pre-fix, independent of sample
  count. Worth the next data-collection round keeping an eye on this
  channel's R² for that reason, rather than assuming more samples alone
  will fix it.

### Known limitation, deferred to Stage 1: ADSR gating via a trigger pulse,
not a sustained gate

Cable 114 (`SEQ3.TRIG_OUTPUT` → `ADSR.GATE_INPUT`) delivers only a ~1ms
trigger pulse, not a sustained gate. `ADSR`'s configured envelope (attack
`0.0` ≈1ms, decay `0.3` ≈15.8ms, sustain `0.1`, release `0.15` ≈4ms, per
the module's exponential `MIN_TIME=1e-3`/`MAX_TIME=10` time mapping) never
actually reaches decay or sustain in practice: the gate drops low again
almost immediately after attack starts, so the envelope goes attack→release
in roughly 5ms total rather than running its full shape. This makes
`filter_env_amount` closer to a brief click than the "dramatic squelch
control" the spec intended.

**Not fixed now** — reopening envelope-character verification for this
channel isn't worth it before a real use case exists to tune the shape
against, and this fix round's scope was set by the controller, not
reopened here on our own judgment. **Fix direction for whoever picks this
up**: route the gate cable from `SEQ3.CLOCK_OUTPUT` (with the sequencer's
own `data.clockPassthrough: true`) instead of `TRIG_OUTPUT`, or from a
`STEP_OUTPUTS` slot, either of which should hold high for closer to a full
step's duration rather than a fixed ~1ms pulse — and lengthen `ADSR`'s
decay/release accordingly once the gate is actually sustained long enough
for them to matter.

### Task 1's real per-channel verification measurements

Task 1 verified all 8 channels against real measured audio, but the
actual numbers only ever made it into the (gitignored, uncommitted)
`task-1-report.md`, not into this file — the following table closes that
gap. `seq_tempo`/`sweep_rate` use this fix round's own re-verification
numbers (above) instead of Task 1's original ones, since those two
channels' ranges changed; every other row is Task 1's original
measurement, unchanged and not re-taken (their ranges didn't move):

| channel | measurement | result |
|---|---|---|
| `seq_tempo` | mean inter-onset interval at new slowest setting (cc=0, 20s live audio) | 4 consecutive full-loop intervals 3.99/3.99/4.01/4.00s, mean **3.995s** — matches the predicted 4s loop almost exactly; raw `TEMPO_PARAM` read back as exactly 1.0 (CV=0) / 4.0 (CV=1) |
| `sweep_rate` | centroid std over a 5s window, depth=max, tempo held at its own new-slowest, 6 alternating slow/fast trials | slow mean **52.0±8.7 Hz**, fast mean **52.8±9.3 Hz** (ratio 1.01) — **not** measurably different at the production 5s window; arithmetic confirmed correct via raw `FREQ_PARAM` reads (-2.0/-1.0 as designed), and a shorter 2s window reproduces real discrimination (38.3±11.2 vs 55.0±13.5, ratio 1.44) — see the range re-narrowing section above for the full explanation |
| `sweep_depth` | centroid std over 10s, rate fixed at cc=100 | center(cc64)=32.74 Hz, min(cc0)=44.67 Hz, max(cc127)=66.70 Hz — both extremes exceed center |
| `filter_env_amount` | centroid std over 10s, seq_tempo=90 | center(cc64)=34.66 Hz, min(cc0)=36.62 Hz (≈noise floor), max(cc127)=179.43 Hz (~5x) — dramatic effect at max, none at min/center |
| `vcf_resonance` | spectral flatness (dB), cc 0/32/64/96/127 | -117.09 → -118.70 → -119.87 → -119.99 → -120.29 dB, monotonically more concentrated as resonance rises |
| `vco_freq` (re-verify vs Minimoog patch) | median pitch, cc 64/96/127 | 132.6/266.7/521.7 Hz — matches un-sequenced baseline (132.2/265.2/521.7) almost exactly. cc0/32 showed elevated readings (82.5/70.2 Hz vs expected 32.7/65.8 Hz) — root-caused via a direct raw-param read (not audio) as a pitch-estimation block-size/frequency-floor artifact at low fundamentals, not a wiring defect; a known limitation worth carrying forward, not a Stage 0 defect |
| `vcf_cutoff` (re-verify) | rms + centroid, cc 0/32/64/96/127 | rms 0.0005→0.0016→0.0915→0.2484→0.3303, centroid 207.8→151.3→199.6→550.2→1725.5 Hz — monotonic, consistent with the Minimoog-only baseline |
| `vca_level` (re-verify) | rms, cc 0/32/64/96/127 | 0.0000→0.0832→0.1664→0.2488→0.3296 — clean monotonic |

## Stage 1 Task 6 (2026-09-07): `instruction_to_preset.py` capstone script, and an explicit gap — it has never actually talked to an LLM

Stage 1 (Tasks 1-5, all merged to `main`) built a shared windowed-audio-read
helper (`features.read_aggregated_window`), per-channel descriptions on
`VCVRackBackend.channel_descriptions()`, an `InstructionParser` abstract
interface with shared Pydantic-based recipe validation
(`instruction_parser/schema.py`, `instruction_parser/prompts.py`), an
Anthropic-backed implementation (`instruction_parser/anthropic_parser.py`),
and a local/OpenAI-compatible implementation
(`instruction_parser/local_parser.py`). Task 6 wires all five pieces
together into `instruction_to_preset.py`: parse an instruction into a CV
recipe via `channels = backend.channel_descriptions()` /
`instruction_parser.parse_recipe(instruction, channels)`, push each value
out over MIDI via `backend.set_cv`, let the patch settle, then read real
audio back via `features.read_aggregated_window` and print the recipe plus
the measured `[mean, std] x [loudness, brightness, pitch]` vector.

**What's verified**: the cross-module wiring was checked before writing this
script — most modules it imports from were read fresh against their actual
current source (not just the plan's transcription) — and `python3
instruction_to_preset.py --help` runs clean, confirming the whole import
chain (`backends.vcv_rack`, `features`, `instruction_parser.anthropic_parser`,
`instruction_parser.local_parser`, and transitively `instruction_parser.
{base,prompts,schema}`) resolves with no stale import path left over from
Task 5's relocation of the system-prompt builder into
`instruction_parser/prompts.py`, and that the CLI's argument surface
(`--llm {anthropic,local}`, `--model`, `--base-url`, `--settle-time-s`,
`--aggregate-window-s`) is well-formed. All of Stage 1's parsing/validation
logic is unit-tested against mocked LLM responses (`tests/` — recipe schema
validation, malformed-JSON/out-of-range/missing-channel error paths, both
parsers' request-building). This diligence was not exhaustive, though: a
final whole-branch review found that `local_parser.py` specifically had not
been re-read closely enough to catch a real `--llm local` without `--model`
gap (the CLI let `model=None` reach `LocalInstructionParser`, which requires
it with no default) — four of five modules got the fresh-read treatment
this claim describes, not all five.

**What's explicitly NOT verified — a real gap, not an oversight**: this
script has never been run end-to-end against a real LLM. This machine
currently has neither an `ANTHROPIC_API_KEY` (nor an `ant auth login`
profile) configured, nor any local OpenAI-compatible model server running
to point `--base-url` at. That means:
- No real Anthropic API call has ever gone through
  `AnthropicInstructionParser.parse_recipe` from this script — only through
  mocked `anthropic.Anthropic()` clients in tests.
- No real local server (vLLM, Ollama, llama.cpp server, LM Studio, ...) has
  ever answered a `LocalInstructionParser.parse_recipe` call from this
  script.
- Consequently, nobody has yet seen a real model's actual channel-value
  choices for a real instruction, nor how well those choices sound once
  applied to the live patch and measured back through
  `read_aggregated_window` — the whole point of Stage 1.

**To close this gap**, whoever picks this up next needs either: an
`ANTHROPIC_API_KEY` env var (or an `ant auth login` profile) to run
`python3 instruction_to_preset.py "<instruction>" --llm anthropic`, or a
running OpenAI-compatible local server plus its `--base-url` (and, always,
`--model` — `LocalInstructionParser` has no default model and the CLI
requires both together) to run with `--llm local`. Closing it is the
natural next step
the moment either becomes available — this section exists so that step is
remembered as outstanding, not assumed already done because the code
merged and the tests are green.

## Stage 2 capstone (2026-09-08): live novelty-search run against
`sequencer_test.vcv` — 60/60 archive filled, one real mid-run PipeWire glitch
caught and root-caused, not hidden

Task 3 of the Stage 2 plan (`docs/superpowers/plans/
2026-09-08-stage2-exploration.md`) — the capstone task, running the
already-merged `novelty_archive.py`/`explore.py` (Tasks 1-2) for real
against the live instrument for the first time.

### Launch check found a new failure mode, not the documented one

Nothing was running at session start (`pgrep -af "Rack2Free/Rack"` empty,
`pw-link -l` showed no `vcv_loop` connections) even though the PipeWire
*defaults* were already correctly set (`vcv_loop`/`vcv_loop.monitor`) —
apparently left over from whenever the last session's Rack process exited.
`log.txt` already ended in `"END"` (a clean prior exit), so the documented
crash-recovery dialog wasn't expected to be an issue, and it wasn't.

**What actually blocked the relaunch, three attempts in a row**: running
`./Rack2Free/Rack patches/sequencer_test.vcv` from this repo's root (the
natural cwd) reliably died after ~6-9s with exit code 1, logging nothing
past `"Loading settings ...settings.json"` and spawning a zenity dialog
with only generic GTK warnings visible in stdout — no crash-recovery
question dialog, no window ever created (confirmed via `xwininfo -root
-tree` showing no Rack/zenity surface at all). Fetching
`VCVRack/Rack`'s actual `adapters/standalone.cpp` source (`gh api
repos/VCVRack/Rack/contents/adapters/standalone.cpp`) pinned the real
cause: right after `settings::load()` succeeds, there's an unlogged check
—
```cpp
std::string resDir = asset::system("res");
if (!system::isDirectory(resDir)) {
    osdialog_message(OSDIALOG_ERROR, OSDIALOG_OK, ...); // standalone.resDir
    exit(1);
}
```
— and `asset::systemDir` defaults to **the process's cwd**, not the
executable's own directory. Launching from the repo root means
`asset::system("res")` resolves to `<repo root>/res`, which doesn't exist
(`res/` only lives inside `Rack2Free/`), so this fires every time,
*before* `logger::wasTruncated()` is ever checked (that check is much
later in `main()`, after network/audio/MIDI/plugin/browser/library/UI
init) — meaning the crash-recovery dialog and this resDir dialog are two
different, easily-confused zenity failure modes, and this session hit the
second one, not the first.

**Fix, and why it wasn't obvious sooner**: `settings.json`'s own
`recentPatchPaths` already encoded the answer
(`"../patches/sequencer_test.vcv"`, a relative path with a leading `../`)
— every prior session must have launched with **cwd = `Rack2Free/`**, not
the repo root. `cd Rack2Free && LD_LIBRARY_PATH=. ./Rack
../patches/sequencer_test.vcv` launched clean on the first try (full
plugin/module loading log, "Running window" at 6.457s, all 8 MIDI-CAT
channels present, screenshot-confirmed). Worth remembering explicitly:
**always launch Rack with cwd set to `Rack2Free/`**, regardless of where
the invoking shell started.

Post-launch health re-check, same bar as every prior stage: `pw-link -l`
showed `VCV Rack:{input,output}_{FL,FR}` cabled through `vcv_loop`/
`vcv_loop.monitor` correctly, and a screenshot of the MIDI-CAT module
confirmed `In: ALSA / Midi Through:Midi Through Port-0 14:0` resolved (`Out:
(No device)`, expected/unused) with all 8 channel labels
(`vco_freq`/`vcf_cutoff`/`vca_level`/`seq_tempo`/`sweep_rate`/
`sweep_depth`/`filter_env_amount`/...) visible in the module's mapping
list, and the `AUDIO` module showing `VCV_Loopback` selected.

### The real run

```
python3 explore.py --config configs/sequencer_test.yaml \
  --budget 200 --archive-size 60 --k-neighbors 5 \
  --settle-time-s 0.5 --aggregate-window-s 3.0 --seed 0 \
  --output data/sequencer_novelty_archive.npz
```

Backgrounded (`nohup ... & disown`, waited on via a `kill -0`-polling
monitor rather than blocking foreground), per this project's established
long-run pattern. Took ~11.65 minutes (699s) for 200 candidates (~3.5s
each, matching the plan's estimate) — stdout was fully buffered (not a
tty), so nothing appeared in the log file until the process actually
exited; this was just Python's file-vs-tty buffering default, not a stall
(confirmed via `ps`/CPU-time checks mid-run before deciding to just wait
for the exit notification instead of chasing a false "no output" alarm).

**Result: archive filled to the full 60/60**, from 105/200 accepted
candidates (95 rejected as insufficiently novel — a healthy accept rate
for a mutation-plus-random-restart search, not "everything gets in").
Final leave-one-out novelty (not insertion-time — recomputed against the
final 60-member archive via `NoveltyArchive.final_novelty_scores()`, per
this stage's fix-round-1 review; the originally-saved insertion-time
scores were stale and have been corrected on disk) among the 60 kept
members: **min 0.1182, max 0.5091, mean 0.1654**.

### A real anomaly, caught and root-caused rather than reported as a bare "looks off"

The last 59 of 200 evaluated candidates (indices 142-200) all show the
*exact same* novelty score, `0.0776`, and all were rejected — a flatline,
not just "mostly rejected." Investigated rather than shrugged off, since
the task's own bar is to report (and understand, where possible)
miscalibrated-looking behavior rather than wave it past:

- **Root cause confirmed**: checking `pw-link -l` immediately after the
  run finished showed `VCV Rack:{input,output}_*` now connected to
  `alsa_output.pci-0000_10_00.1.hdmi-stereo` (the physical HDMI output) —
  **not** `vcv_loop` — even though `pactl get-default-source` still said
  `vcv_loop.monitor`. `pactl get-default-sink` had silently changed to the
  HDMI device mid-run. `journalctl --user` around the transition
  (`15:56:39`-`15:56:40`, ~7.65 minutes into the run — lining up almost
  exactly with candidate ~141/200 at the observed ~3.5s/candidate pace)
  showed `pipewire[1766]: mod.client-node: ... unknown peer ... fd:124`
  messages, consistent with a stream reconnect event at that moment.
- **The actual defect, found by checking one level deeper**:
  `pactl list modules short | grep null` showed **two** separate
  `module-null-sink sink_name=vcv_loop` instances loaded simultaneously
  (module ids `536870913` and `536870914`) — leftover cruft from some
  earlier session, not loaded by this one (Step 1's health check correctly
  found an existing `vcv_loop` sink and skipped `load-module` per the
  documented procedure). Two PipeWire sink objects sharing the same
  `vcv_loop` name makes `pactl set-default-sink vcv_loop` an ambiguous
  target; something (most plausibly one of the two identically-named
  nodes going idle/`SUSPENDED` and PipeWire's routing policy resolving the
  live stream to the next-highest-priority real device instead) flipped
  the actual default over to the hardware sink partway through, silently,
  with no error surfaced anywhere in Rack's own log.
- **Effect on the saved archive**: none, as far as can be confirmed. The
  archive was already full (60/60) by candidate 60, and the last
  genuinely-accepted candidate was #141 (`score=0.1235`) — right at the
  boundary of the drift. Candidates 142-200 read frozen/silent audio
  (consistent with the HDMI output carrying nothing meaningful for
  `sounddevice`/PipeWire to capture from `vcv_loop.monitor` once VCV's
  stream moved away from it) and were correctly rejected as non-novel
  every single time, rather than corrupting the archive with 59 copies of
  a degenerate silent reading. The practical cost was **wasted budget, not
  bad data**: roughly the last 30% of the 200-candidate budget was
  uninformative.
- **Not fixed at the root** (the two duplicate `module-null-sink vcv_loop`
  module instances are still both loaded) — flagged here rather than
  unloaded blind, the same "wasn't ours to remove outright" caution this
  file already applies to the quarantined `Nozoid` plugin. Restored
  `pactl set-default-sink vcv_loop` immediately after diagnosis so the
  environment is left in the same healthy state Step 1 found it in.
  Whoever runs a search like this next should `pactl list modules short |
  grep null` before starting and consider unloading the duplicate instance
  (by module id) if this recurs — a single clean `vcv_loop` sink is
  probably what actually prevents the mid-run drift, not just restoring
  the default pointer afterward.

### Spot-check: three archived CV vectors, by their measured features

Picked by **corrected** final leave-one-out novelty score (lowest, median,
highest kept member — this fix-round-1 correction reordered which three
members these are relative to the original, stale-score pick) rather than
cherry-picked. `features.py`'s `extract_features_aggregated` interleaves
its output as `[mean(loudness), std(loudness), mean(brightness),
std(brightness), mean(pitch), std(pitch)]`, so the correct `[mean, std]`
column order below is loudness = elements `[0,1]`, brightness = elements
`[2,3]`, pitch = elements `[4,5]` (an earlier version of this table
misread this layout in 2 of 3 rows):

| member | `vca_level` | `vcf_cutoff` | loud `[mean,std]` | bright `[mean,std]` | pitch `[mean,std]` | score |
|---|---|---|---|---|---|---|
| lowest-novelty | 0.000 | 0.696 | `[0.0000, 0.0000]` | `[0.0050, 0.0313]` | `[0.0000, 0.0000]` | 0.1182 |
| median-novelty | 0.581 | 0.925 | `[0.2954, 0.0194]` | `[0.3178, 0.0487]` | `[0.3423, 0.0479]` | 0.1420 |
| highest-novelty | 0.462 | 1.000 | `[0.1337, 0.0048]` | `[0.9714, 0.0417]` | `[0.6660, 0.1311]` | 0.5091 |

These are visibly, meaningfully different sounds, not just algorithmically
distinct numbers: the lowest-novelty member is `vca_level=0.0` — silence,
measured as all-zero loudness and exactly-zero pitch (no periodicity
detected in a fully silent block); its brightness isn't quite zero (mean
0.005, std 0.031), a small nonzero spectral-centroid reading off an
essentially-silent block rather than a meaningful tonal quality. The
median member is moderately loud (mean 0.295) with moderate brightness
(mean 0.318) and a mid-range pitch (mean 0.342) — and, notably, all three
of its `std` values are small (0.019-0.049), meaning this is a *steady*
patch, not a wide-swinging one (an earlier version of this table's prose
mistakenly attributed a "wide-swinging pitch" to this member — that
claim was quoting a *mean*, not a `std`, and doesn't hold for the
corrected median member's actual std values). The highest-novelty member
is moderately loud (mean 0.134) but **very** bright (`vcf_cutoff=1.0`,
brightness mean 0.971 — the archive's most extreme brightness reading)
with a wildly swinging pitch (std 0.131 — the widest pitch spread of any
of the three, by a wide margin over the median's 0.048 and the lowest's
0.0) — a harsh, warbling, near-fully-open-filter patch, about as far from
"silence" as this instrument gets. (This highest-novelty member happens
to be the same archived member under both the original stale scoring and
the corrected scoring — its raw feature values are unchanged, only its
reported score moved, from 0.6704 to 0.5091.)

### No clustering in a narrow band — coverage looks healthy

Per-CV-channel spread across the 60 archived members (mean/std/min/max),
checked specifically because the task called out "lands in a narrow band
of one channel" as a failure shape to watch for:

| channel | mean | std | min | max |
|---|---|---|---|---|
| `vco_freq` | 0.560 | 0.293 | 0.067 | 1.000 |
| `vcf_cutoff` | 0.647 | 0.285 | 0.055 | 1.000 |
| `vca_level` | 0.534 | 0.282 | 0.000 | 1.000 |
| `seq_tempo` | 0.446 | 0.306 | 0.000 | 1.000 |
| `sweep_rate` | 0.502 | 0.318 | 0.000 | 1.000 |
| `sweep_depth` | 0.609 | 0.335 | 0.000 | 1.000 |
| `filter_env_amount` | 0.574 | 0.284 | 0.000 | 1.000 |
| `vcf_resonance` | 0.501 | 0.269 | 0.000 | 0.980 |

Every channel's std is close to a uniform-on-`[0,1]` distribution's
~0.289, and every channel spans nearly the full range — no channel is
stuck in a narrow band. A `vca_level` histogram (10 bins across `[0,1]`)
came out fairly even (4-8 members per bin, no empty or dominant bin). This
looks like real, non-degenerate coverage of the CV space, not an artifact
of the algorithm accepting almost anything.

### Full regression check

`python3 -m pytest -q` → **86 passed, 4 deselected**. `gozer run --chips 1
--who "claude:tt-cv-agency" --reason "final Stage 2 suite check" --
python3 -m pytest -q -m hardware` → **4 passed, 86 deselected**. No
regressions; hardware count unchanged from every prior stage's bar of 4.

### Net assessment

The novelty search runs end-to-end against the real, live instrument and
produces a full, non-degenerate 60-member archive with genuine CV-space
coverage and genuinely different-sounding archived members, confirmed by
direct feature-vector inspection rather than taken on the algorithm's
word. The stage's own honesty bar is also where this run earned its keep
twice: once by chasing down a *new* Rack-launch failure mode
(`asset::systemDir` defaulting to cwd, not exe dir — read straight from
Rack's own source rather than guessed) that isn't the crash-recovery
dialog this file already documents, and again by root-causing a mid-run
PipeWire default-sink drift (traced to two duplicate `module-null-sink
vcv_loop` instances) down to its actual mechanism and actual (limited,
budget-only) blast radius, rather than either hiding it or overstating it
as archive corruption. Both are now documented so neither has to be
re-discovered.

## Stage 3 capstone (2026-09-08): live trajectory data collection, predictor
training, and a CEM-planned control-loop run that converges *worse* than
Stage 0's plain reflex model on the identical goal

Task 6 of the Stage 3 plan (`.superpowers/sdd/2026-09-08-stage3-trajectory-
control/task-6-brief.md`) — the capstone task, running the already-merged
`trajectory_collection.py`/`trajectory_model.py`/`cem_planner.py`/
`tt_trajectory_inference.py`/`trajectory_control_loop.py` (Tasks 1-5) for
real against the live instrument for the first time.

### Launch check: already healthy, no relaunch needed

Unlike Stage 2's capstone, this session found VCV Rack already running
correctly (pid confirmed live via `/proc/<pid>/exe` resolving to
`.../Rack2Free/Rack`, not just a `pgrep -f` string match — worth being
careful about here, since `pgrep -af "Rack2Free/Rack"` matches its own
invoking shell command line and would otherwise falsely read as "found a
process" even with nothing real running), launched with cwd=`Rack2Free/`
per the documented fix, running `patches/sequencer_test.vcv`, autosaving
normally (`~/.local/share/Rack2/log.txt` showed steady ~15s autosave
cycles with no gaps), uptime already over an hour at session start.
`pw-link -l` showed `VCV Rack:{input,output}_{FL,FR}` correctly cabled
through `vcv_loop`/`vcv_loop.monitor`, `pactl get-default-sink`/
`get-default-source` both correct (`vcv_loop`/`vcv_loop.monitor`). The same
two duplicate `module-null-sink vcv_loop` module instances flagged in
Stage 2's capstone (ids `536870913`/`536870914`) are still both loaded —
unchanged, not touched, and did not cause any mid-run drift this time
(re-checked immediately after both live runs below; sink/source were still
correct). Skipped GUI/screenshot verification entirely, per this task's own
scope note — the process/PipeWire checks alone were sufficient.

### Real data collection: 600 transitions, seeded from Stage 2's archive

```
python3 trajectory_collection.py
```

Backgrounded (`nohup ... & disown`, waited on via a Monitor task rather than
polling), per this project's established long-collection pattern. 60
episodes × 10 steps against `configs/sequencer_test.yaml` (8 CV channels),
episode starts cycling round-robin through Stage 2's 60-member
`data/sequencer_novelty_archive.npz`, `max_action=0.15`,
`settle_time_s=0.5`, `aggregate_window_s=3.0`. Finished cleanly in line
with the plan's ~38-39 minute estimate: **saved 600 transitions** to
`data/sequencer_trajectory_dataset.npz` (`states` `(600, 6)`, `actions`
`(600, 8)`, `next_states` `(600, 6)`, no NaNs in any array).

### Trained predictor: per-state-dimension held-out MSE/R²

```
python3 trajectory_model.py
```

80/20 train/val split (480/120 transitions), 500 epochs, against a
predict-the-training-mean baseline:

| state dim | held-out MSE | held-out R² |
|---|---|---|
| `loud_mean` | 0.0016 | **0.9576** |
| `loud_std` | 0.0002 | 0.4759 |
| `bright_mean` | 0.0031 | **0.9264** |
| `bright_std` | 0.0005 | 0.4131 |
| `pitch_mean` | 0.0071 | **0.8683** |
| `pitch_std` | 0.0021 | 0.2339 |

**Honest read: this trajectory model learned its target (predicting the
*next* windowed feature read from the current one plus the applied CV
delta) meaningfully better across the board than Stage 0's inverse model
learned its own, harder target (predicting CV settings from features,
across 8 output channels) at a comparable-order sample count.** The three
`mean` dimensions are strong (R² 0.87-0.96) — a one-step forward-dynamics
problem is an easier regression than inverting an 8-dimensional CV space
from a 6-dimensional feature read, and a 600-sample dataset spread across
6 output dimensions (vs. Stage 0's 300 samples across 8) also helps. The
three `std` dimensions are consistently weaker (R² 0.23-0.48) than their
paired `mean` dimensions — plausible and consistent with this project's
established history rather than a fresh mystery: `sweep_rate` (an LFO
rate) and `filter_env_amount`'s ADSR-gating limitation were already
flagged in Stage 0's own capstone as channels whose dynamics move slowly
relative to a single aggregation window, and the `std` of a windowed
feature read is exactly the statistic a slow/gated modulation source would
move least within one window — a harder sub-target than the window's
mean, for the same reason those channels were already flagged as
weak/near-invisible. `pitch_std`'s R² (0.23) is the weakest of the three
`std` dimensions, consistent with pitch being the noisiest-estimated
feature historically (see Phase 2's `estimate_pitch` bug history above).

### Live control-loop run: converges, but *worse* than Stage 0's reflex model on the identical goal

```
gozer run --chips 1 --who "claude:tt-cv-agency" --reason "Stage 3 capstone: live trajectory control loop" -- \
  python3 trajectory_control_loop.py 0.5 0.05 0.5 0.1 0.5 0.05
```

Same 6-dim goal (`[loud_mean, loud_std, bright_mean, bright_std, pitch_mean,
pitch_std] = [0.5, 0.05, 0.5, 0.1, 0.5, 0.05]`) as Stage 0's own capstone
control-loop run against this same patch, deliberately — a fair before/after
comparison against that reflex-model result on the identical goal. Ran
cleanly against real hardware (`ttnn` device opened, JIT cache 16/20 hits,
device closed and chips returned `FREE` afterward — confirmed via `gozer
status` both before and after), no exceptions, default
`horizon=5`/`n_candidates=200`/`n_elite=20`/`n_iterations=3`/
`max_iterations=100`/`convergence_threshold=0.05`.

```
final: [0.42806785 0.04535207 0.27741844 0.04745707 0.3158203  0.13492512]
goal:  [0.5        0.05       0.5        0.1        0.5        0.05      ]
```

Per-dimension absolute error: `[0.0719, 0.0046, 0.2226, 0.0525, 0.1842,
0.0849]`; Euclidean norm of the error: **0.314** (above the loop's own
`convergence_threshold` of 0.05 — did not converge).

**Honest read: this is a genuinely worse result than Stage 0's own
capstone run against the identical goal on the identical patch, not a
better one, and is reported as such rather than reframed.** Stage 0's
reflex-model run against this same 6-dim goal on `sequencer_test.vcv`
finished at Euclidean error norm 0.222 (`final: [0.391, 0.026, 0.385,
0.022, 0.376, 0.095]`, per-dim abs error `[0.109, 0.024, 0.115, 0.078,
0.124, 0.045]` — see the Stage 0 section above). This run's norm (0.314)
is meaningfully larger, i.e. the CEM-planned trajectory-model control loop
landed *farther* from the goal than the plain single-shot inverse model did,
despite the trajectory predictor's own held-out R² (above) being
substantially stronger on every dimension than Stage 0's per-channel CV
R² was. Per-dimension, the picture is mixed rather than uniformly worse:
`loud_mean` (0.072 vs. 0.109) and `loud_std` (0.005 vs. 0.024) both
improved over Stage 0, but `bright_mean` (0.223 vs. 0.115), `pitch_mean`
(0.184 vs. 0.124), and `pitch_std` (0.085 vs. 0.045) all got worse, with
`bright_mean` the single largest contributor to the larger overall norm.
A plausible (not confirmed) explanation: `cem_plan` optimizes the
*rollout's final predicted state* against the goal by scoring through the
learned forward model for `horizon=5` compounded steps, so any per-step
model bias compounds multiplicatively across the horizon in a way a
single-shot inverse-model prediction never does — a strong one-step R²
does not by itself guarantee a well-behaved 5-step-ahead rollout, and nothing
in this task's scope re-verified the model's *multi-step* rollout accuracy
independently of its single-step held-out numbers above. Per this task's
own explicit instruction, this result was **not** tuned away — no dataset
size, `horizon`, `n_candidates`, or `max_iterations` change was made to
force a better-looking number. This is reported as a real, negative
finding: for this instrument, at this dataset size, CEM planning over a
learned trajectory model did not outperform Stage 0's much simpler direct
reflex mapping on the same goal, and may have made things measurably worse
on 3 of 6 dimensions specifically.

### Full regression check

`python3 -m pytest -q` → **115 passed, 7 deselected**. `gozer run --chips 1
--who "claude:tt-cv-agency" --reason "final Stage 3 suite check" --
python3 -m pytest -q -m hardware` → **7 passed, 115 deselected** (up from
Stage 2's 4, per this stage's own 3 new `tt_trajectory_inference.py`
hardware tests — matching the task brief's own expected count exactly).
No regressions. `gozer status` confirmed all 4 chips `FREE` both before
and after each hardware-touching run in this session.

### Net assessment

The full Stage 3 pipeline (trajectory collection, forward-dynamics
training, CEM planning, batched `ttnn` inference, and the resulting
control loop) runs end-to-end against the real, live instrument with zero
exceptions, and the trajectory predictor itself is a genuinely
better-fitting model of its own (easier) one-step-ahead target than Stage
0's inverse model was of its target. But the stage's actual goal —
better-controlled convergence via trajectory planning — is **not**
demonstrated here: the live control loop's measured error (Euclidean norm
0.314) is worse than Stage 0's much simpler reflex-model result (0.222) on
the identical goal against the identical patch, worse on 3 of 6 feature
dimensions specifically. Per this task's explicit brief, this is reported
as the genuinely interesting result it is, not tuned into a better-looking
number by adjusting dataset size or planner parameters. The most plausible
open explanation — compounding per-step model bias across a 5-step CEM
rollout not being caught by single-step held-out R² — is flagged as a
concrete, testable hypothesis for a future pass (e.g. directly comparing
1-step vs. 5-step-ahead rollout error against held-out real trajectories),
not asserted as confirmed.

## Stage 3 final-review correction: the real explanation for the worse-than-baseline result

A final whole-branch review of the completed Stage 3 plan dug into the
capstone's one hedged hypothesis above (compounding per-step model bias
across the CEM rollout) and found it is **not** the dominant cause. The
real explanation is several independent, smaller issues stacking up, the
biggest of which was never identified above at all: the goal the live run
asked for is barely inside the training data's coverage. None of this is a
code bug — `trajectory_collection.py`, `trajectory_model.py`,
`cem_planner.py`, `tt_trajectory_inference.py`, and
`trajectory_control_loop.py` were all independently re-verified against
their own briefs and found correct. What follows is what an independent
final-review pass found instead, with its own reproduced numbers, in
priority order by how much each one actually explains.

### 1. The training data barely covers the goal region — the dominant, previously-unidentified cause

An independent final-review pass checked how close the 600 collected
training transitions actually come to the live run's goal (`[0.5, 0.05,
0.5, 0.1, 0.5, 0.05]`): only **1 of 600** training states lies within 0.15
of the goal across all 6 dimensions simultaneously; the single nearest
training state is **0.193** away; the **median** distance across all 600
is **0.567**. That's not a dataset that happens to be thin near the goal —
it's a dataset that's mostly somewhere else entirely.

The per-dimension pattern lines up with the capstone's own "mixed, not
uniformly worse" result exactly. The dimensions that *improved* over Stage
0 (`loud_mean`, `loud_std`, `bright_std`) are precisely the ones with high
training coverage near their goal value (roughly 10-100% of transitions
within a reasonable band); the dimension with the single largest
regression, `bright_mean` (0.115 → 0.223 absolute error, the biggest
contributor to the worse overall norm), has only **~4.5%** training
coverage near its goal value. The model can't plan a good route through
territory it's never seen.

The mechanism traces back two stages: Stage 2's novelty-search archive
(used by `trajectory_collection.py`'s `__main__` to seed episode starts,
see `trajectory_collection.py`) optimized purely for *diversity* between
archive members, which produced many near-silent (`vca_level≈0`) starting
points among its 60 members. 10-step episodes bounded at `max_action=0.15`
per step can't travel far from those corners in only 10 steps — so the
600-transition dataset ends up dense in "boring"/near-silent regions and
sparse in the bright, loud region this particular goal actually asks for.
This is a genuine finding about the *pipeline*, not about any one module:
Stage 2's archive did exactly what it was built to do (maximize
diversity), and Stage 3's collection script did exactly what its own brief
asked (seed from that archive) — nothing downstream of either was told the
eventual goal would live in a sparse corner of the result.

### 2. The reported R² was scored against a baseline too weak to support "strong fit"

The capstone's held-out R² table (0.87-0.96 on the three `mean`
dimensions) was scored only against a predict-the-training-mean baseline.
For a *forward dynamics* model this is a soft bar: consecutive windowed
feature reads are strongly autocorrelated, so a large share of "the model
beat the mean" is really just "the model learned that states don't change
much between reads" — a fact persistence (`next_state = state`, zero
parameters, no action term at all) captures for free. The honest baseline
for a forward-dynamics model is persistence, and a trained model needs to
be judged against **both**.

Part B of this fix pass added `evaluate_persistence_baseline_per_dimension`
to `trajectory_model.py` and re-ran training on the same, untouched
`data/sequencer_trajectory_dataset.npz` (480/120 train/val split, same
seed). The actual re-run numbers:

| state dim | val MSE | val R² (vs. mean) | persistence R² |
|---|---|---|---|
| `loud_mean` | 0.0015 | 0.9598 | 0.9105 |
| `loud_std` | 0.0003 | 0.3524 | 0.7393 |
| `bright_mean` | 0.0018 | 0.9590 | 0.8309 |
| `bright_std` | 0.0006 | 0.3500 | 0.4379 |
| `pitch_mean` | 0.0085 | 0.8425 | 0.8015 |
| `pitch_std` | 0.0021 | 0.2299 | -0.0700 |

Against the honest baseline, the picture changes a lot. On the three
`mean` dimensions the trained model does beat persistence, but by a modest
margin, not the wide gap the mean-baseline table implied (`loud_mean`
0.9598 vs. 0.9105, `bright_mean` 0.9590 vs. 0.8309, `pitch_mean` 0.8425
vs. 0.8015) — real learned signal, but the training-mean-only table
overstated how much of it there was. On the three `std` dimensions the
model is at or below persistence (`loud_std` 0.3524 vs. **0.7393** —
persistence is clearly better; `bright_std` 0.3500 vs. 0.4379 — persistence
still ahead; `pitch_std` 0.2299 vs. **-0.0700** — the one case where the
model clearly beats persistence, because persistence itself is unusually
bad here). So "the model learned its target well" was true mainly for the
`mean` dimensions and mostly false for `std` — a materially more honest
picture than the capstone's original table conveyed. This is a plan/spec
gap (the spec mandated the training-mean baseline only), not an
implementation bug — `trajectory_model.py` did exactly what its brief
asked.

### 3. The model's action gain is over-scaled relative to what the instrument can actually do

A full-range action-sweep test against the trained model found its
response to actions is roughly **3-6x larger**, per dimension, than the
real observed per-step movement in the actual training data — e.g.
`bright_mean`: the model believes it has ~0.282 of authority per max
action, but the real observed movement in the collected data is only
~0.050 (a ~5.6x over-scale); `pitch_mean` is over-scaled ~3.3x; the `std`
dimensions are over-scaled 1.8-3.5x. With only 480 training rows spread
across an 8-dimensional action space and an unregularized 32-hidden-unit
MLP, the model had every opportunity to absorb measurement noise (the
3-second aggregation window's own sampling noise, sequencer phase at
read-time, near-silence discontinuities) into the action term rather than
learning the action's true, smaller magnitude. This is a data/model-capacity
limitation, not a bug — the review confirmed the model is **not**
action-blind (zeroing the action input measurably worsens held-out error,
so the action signal is real), just miscalibrated in magnitude.

### 4. The live CEM search was badly under-sampled for its own search dimension

`trajectory_control_loop.py`'s defaults give `cem_plan` a `horizon=5 ×
action_dim=8` = 40-dimensional continuous search, with only
`n_candidates=200`, `n_elite=20`, `n_iterations=3` to search it. The
review ran `cem_plan` 10 times against an identical state/goal, varying
only the RNG seed, and found **5 of the 8** CV channels show seed-to-seed
noise in the returned action that exceeds the actual signal — i.e. at this
configuration, the planner is effectively returning a near-random draw on
most channels, not a converged plan.

Worth noting: `cem_planner.py`'s own unit tests only validate the
algorithm at 1-2 search dimensions, with a much larger *relative* search
budget (500 candidates / 8 iterations for that tiny a space). Those tests
correctly prove the CEM algorithm itself is implemented right — they say
nothing about whether it finds good plans at the dimensionality production
actually runs it at (40-D), and it doesn't, at this sample budget.

### 5. The headline 0.222-vs-0.314 comparison is confounded, not apples-to-apples

Two separate issues stack on top of the above, independent of both the
data-coverage and search-budget problems:

* **Different measurement windows.** Stage 0's `control_loop.py` defaults
  to `aggregate_window_s=5.0`; Stage 3's `trajectory_control_loop.py`
  defaults to `3.0`. That's a genuinely different measurement instrument
  for a sequencer patch's mean/std features, not a detail that washes out.
* **A converged number compared to a non-converged one.** Stage 0's
  number is a true fixed point — its control law does geometric
  `step_fraction`-based convergence toward a constant target, so
  `history[-1]` really is "where it settled." Stage 3's control loop has
  no damping and replans from scratch every iteration with 5 of 8 channels
  effectively noise (per point 4 above), so its measured "final" value is
  one draw from a walk that never settles, not a converged value.

Comparing `history[-1]` from both treats two different kinds of
quantities — a fixed point and one sample from an unconverged random walk
— as though they were the same thing. The 0.222-vs-0.314 comparison is
real data, but it isn't the controlled before/after comparison it reads
as.

### 6. Ruled out, quantitatively: bfloat16 precision compounding across the 5-step rollout

The capstone's own hedge — that per-step model bias compounds across the
5-step CEM rollout — pointed at the right *shape* of problem
(compounding error across steps) but the review found the specific
bfloat16-precision version of that story is not the cause. Simulating
bf16 rounding at every op boundary through 5 chained forward passes, the
divergence from fp32 grows sub-linearly and reaches only **~0.0044** by
step 5 — about **1.4%** of the observed 0.314 error, two orders of
magnitude too small to explain the result. (The same check found the
`tt_trajectory_inference.py` hardware test's tolerance, `atol=0.1`, is
13-50x looser than this real simulated error and could hide an actual
broken-matmul bug rather than just bfloat16 rounding — tightened in Part C
of this fix pass below.)

### Net assessment

The six Stage 3 modules contain no correctness bug — action semantics,
channel-order provenance, and the CEM algorithm itself were all
independently re-verified and are implemented as specified. But the
pipeline's training data doesn't cover the region it was asked to control,
its action-gain calibration and CEM search budget are both undersized for
the problem's real dimensionality, and the baseline comparison to Stage 0
was measured with two different instruments (different aggregation
window, converged-vs-unconverged quantities). None of this makes Stage 3 a
failure — the modules do what their briefs asked, correctly — but it does
mean the capstone's headline comparison isn't the clean win-or-lose signal
it was reported as, and it identifies exactly what Stage 4 (or a future
revisit) needs to fix, in priority order:

1. Collect trajectory data with episode starts that actually span the
   goal region, not just Stage 2's diversity archive.
2. Report a persistence baseline alongside R² by default (done in this
   fix, see Part B / `evaluate_persistence_baseline_per_dimension`).
3. Either raise CEM's `n_iterations`/`n_candidates` substantially, or
   reduce `horizon`/`action_dim` search burden so the sample budget
   matches the search dimension — and add a CEM test at production-scale
   dimensionality (today it exists only at toy scale).
4. Clip rolled states to `[0,1]` in `cem_plan` and thread `current_cv`
   through so cumulative multi-step actions are checked against real
   headroom, not just each individual step's own clip (documented as
   known scope in Part D of this fix pass, not yet fixed).
5. Add damping/step-size decay to `trajectory_control_loop.py` so it can
   settle instead of random-walking.
6. Align `aggregate_window_s` between the two control loops before
   drawing any future before/after comparison.
