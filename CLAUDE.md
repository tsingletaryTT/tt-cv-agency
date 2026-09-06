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
- `patches/bridge_test.vcv` — the test patch (VCO → VCA → AudioInterface2,
  plus an OSC'elot module), authored **directly as JSON** rather than built
  by hand in the GUI (see below).
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
