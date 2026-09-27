# Stage 3 tune-up: goal-spanning data, action-gain calibration, damping

## Context

Stage 3's final-review (documented in `CLAUDE.md`'s "Stage 3 final-review
correction" section) found the CEM-planned trajectory control loop
underperforming Stage 0's much simpler reflex model on an identical goal,
and root-caused it to several independent, smaller issues rather than one
bug:

1. The training data (seeded entirely from Stage 2's diversity-optimized
   novelty archive) barely covers the region most real goals ask for —
   only 1/600 training transitions landed within 0.15 of the standard
   test goal across all 6 dimensions. **Dominant, unfixed cause.**
2. Held-out R² was only ever reported against a predict-the-mean
   baseline, not the honest persistence baseline (fixed already, see
   `evaluate_persistence_baseline_per_dimension`).
3. The trained predictor's action term is over-scaled 3-6x relative to
   real observed per-step movement — confirmed via a one-off action-sweep
   test, never fixed or turned into a permanent check.
4. CEM's search budget vs. its real dimensionality (fixed already, see
   "Stage 3 follow-up (2026-09-09)").
5. `trajectory_control_loop.py` has no damping — it replans from scratch
   every iteration and applies the full CEM-planned action, closer to a
   random walk than a settling controller.
6. `trajectory_control_loop.py`'s `aggregate_window_s` (3.0) doesn't match
   `control_loop.py`'s (5.0) — any future before/after comparison is
   measured with two different instruments.

Items 2 and 4 are done. This tune-up closes 1, 3, 5, 6 in one pass, since
they touch overlapping files and a fresh data collection is the natural
point to also fix the window mismatch. This was additionally motivated by
a real live run this session (documented in `CLAUDE.md`'s "First real
live-LLM verification" section): a real LLM-parsed goal
(`[0.7, 0.15, 0.8, 0.15, 0.6, 0.15]`) scored a 0.593 Euclidean error,
worse than every prior baseline, and the per-dimension pattern (near-exact
on `pitch_mean`, worst on `bright_mean`) matches the goal-coverage gap
exactly — a real instruction hit the known hole, not a new one.

## Decision 1: Mixed archive/uniform seeding for trajectory collection

`trajectory_collection.py`'s `collect_trajectories` function itself is
unchanged — the fix is entirely in what `__main__` passes as
`seed_cv_vectors`. Currently: all 60 episode starts come from
`data/sequencer_novelty_archive.npz`'s 60 members (Stage 2's
diversity-optimized archive, which skews toward extremes including many
near-silent starting points).

New behavior: build a 60-vector seed array where the **first 30** entries
are the first 30 archive members (unchanged slice of existing behavior —
still gets diversity-seeded episodes) and the **last 30** are freshly
sampled uniform-random CV vectors (`rng.uniform(0.0, 1.0, size=(30,
n_channels))`, using a seed distinct from the main collection RNG so the
seeding draw is reproducible independent of the per-step action draws).
Total episode count (60) and episode length (10) stay the same, so this
is a same-size, same-shape dataset with a different seeding
*distribution* — not a bigger collection.

Rationale for a straight 50/50 split over anything fancier (e.g.
reachability-guided or goal-directed active sampling): the diagnosed
problem is specifically that archive-only seeding clusters starts in
extreme/boring corners; a uniform-random half directly targets the
"typical middle" a real goal is likely to want, without hand-tuning to
any one goal vector (which would risk re-creating the same
narrow-coverage problem for a different goal instead of fixing it
generally). This mirrors Stage 0's own original data-collection strategy
(before Stage 2's archive-seeding was introduced for Stage 3), so it's a
reuse of an already-proven pattern, not new machinery.

## Decision 2: Align `aggregate_window_s` to 5.0 (from 3.0)

Change the default in both `trajectory_collection.py`'s `collect_trajectories`
signature and its `__main__`, and `trajectory_control_loop.py`'s
`run_trajectory_control_loop` signature and its `__main__`, from `3.0` to
`5.0` — matching `control_loop.py`'s own default. Confirmed via a full
test-suite grep that no test relies on either function's default (every
call site passes its own explicit small value for fast tests), so this is
a zero-test-impact change.

Because the recollection in Decision 1 already needs a fresh
`trajectory_collection.py` run, this rides along in the same pass with no
new train/inference window mismatch: the freshly collected dataset, the
retrained model, and the live control loop's own read window will all be
5.0s, consistent with each other and with `control_loop.py`. This makes a
future baseline comparison against Stage 0 finally apples-to-apples on
this axis (goal-coverage and search-budget differences remain separate,
already-documented caveats).

Runtime cost, stated honestly up front: settle_time_s (0.5) +
aggregate_window_s (5.0) = 5.5s/read vs. the previous 3.5s — total
collection time rises from ~38-39 minutes to roughly **~60 minutes** for
the same 60 episodes x 10 steps.

## Decision 3: Action-gain regularization + a permanent gain-ratio gate

Two changes to `trajectory_model.py`:

**3a. `weight_decay` on the optimizer.** `train_predictor` gains a
`weight_decay: float = 0.0` parameter, passed straight through to
`torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)`.
`__main__` calls it with a nonzero value (`1e-3` — a standard starting
point for L2 regularization on a small MLP with normalized [0,1]
inputs/outputs; this is a first attempt, not a value tuned to a target
outcome). Existing callers/tests that don't pass `weight_decay` keep
today's behavior exactly (default `0.0` = no regularization), so this is
additive, not a behavior change for existing code.

**3b. A reusable, testable action-gain diagnostic.** New function:

```python
def estimate_action_gain_ratio_per_dimension(
    model: TrajectoryPredictor,
    states_val: np.ndarray, actions_val: np.ndarray, next_states_val: np.ndarray,
) -> np.ndarray:
    """Per-state-dimension ratio of the model's predicted step-to-step
    movement magnitude to the REAL observed movement magnitude, on held-out
    validation data. A ratio near 1.0 means the model's action response is
    calibrated to reality; a ratio of e.g. 5.6 (as found live for
    bright_mean in the original Stage 3 final review) means the model
    believes it has ~5.6x more authority per action than the instrument
    actually has -- a real risk with a small, unregularized MLP absorbing
    measurement noise into the action term rather than learning its true,
    smaller magnitude.

    Uses REAL (state, action, next_state) validation triples throughout
    (not a synthetic sweep from a fixed neutral state), so the ratio
    reflects the model's behavior in the same operating regime the CEM
    planner actually queries it in. Median (not mean) aggregation resists
    outliers from any single noisy transition; a small floor on the
    denominator avoids dividing by an near-zero real movement.
    """
    model.eval()
    with torch.no_grad():
        prediction = model(
            torch.tensor(states_val, dtype=torch.float32),
            torch.tensor(actions_val, dtype=torch.float32),
        ).numpy()
    predicted_delta = np.abs(prediction - states_val)
    real_delta = np.abs(next_states_val - states_val)
    return np.median(predicted_delta, axis=0) / np.maximum(np.median(real_delta, axis=0), 1e-4)
```

`__main__` prints this ratio per state dimension (alongside the existing
MSE/R² table) both so the retrained model's real calibration is visible,
and so a regression test can assert it stays within a sane band (e.g.
`0.3 <= ratio <= 3.0` per dimension) against a small synthetic model/data
fixture with a known, constructed gain — a permanent gate against this
exact failure mode recurring silently. This spec does **not** promise the
retrained model's real ratio will land inside any specific band; the
capstone step reports whatever the honest re-measured numbers are,
including if `weight_decay=1e-3` turns out insufficient on its own (a
plausible, acceptable outcome — Decision 1's better-covered data is
expected to help this too, not just regularization alone).

## Decision 4: Damping via `step_fraction` in the control loop

`run_trajectory_control_loop` gains a `step_fraction: float = 0.5`
parameter. Applied at the one point the CEM-planned action is turned into
a CV write:

```python
next_cv = np.clip(current_cv + step_fraction * action, 0.0, 1.0)
```

This mirrors `control_loop.py`'s own existing `step_fraction` convention
(same name, same purpose: apply a fraction of the suggested correction
per step). Since `cem_plan` already replans from a fresh `current_cv`
every iteration (receding-horizon usage), damping the applied action lets
per-iteration planner noise (documented in the original final review: 5
of 8 channels dominated by seed-to-seed sampling noise at the time) get
averaged down across iterations instead of being applied at full
magnitude on every single step — the mechanism a receding-horizon
controller needs to actually settle rather than random-walk around the
goal. `0.5` is a starting value (matching a natural "take half the
suggested step" default); this spec does not tune it further before
measuring the capstone result.

## Decision 5: Capstone — recollect, retrain, re-verify live

In order: run `trajectory_collection.py` (mixed seeding, 5.0s window) to
produce a fresh `data/sequencer_trajectory_dataset.npz`; run
`trajectory_model.py` (with `weight_decay=1e-3`) to retrain, reporting the
MSE/R²/persistence table (as today) plus the new action-gain ratio table,
compared explicitly against the original Stage 3 ratios documented in
`CLAUDE.md`; run `trajectory_control_loop.py` (with `step_fraction=0.5`)
live against `sequencer_test.vcv`, targeting the exact goal that scored
0.593 this session (`[0.7, 0.15, 0.8, 0.15, 0.6, 0.15]`, from "make it
bright and steady"), reporting the new Euclidean error and per-dimension
breakdown honestly against that number — whether it improves, doesn't, or
partially improves. No dataset size, `weight_decay`, `step_fraction`, or
other parameter gets re-tuned after seeing this result specifically to
force a better-looking number, per this project's established norm for
capstone comparisons.

## Global constraints

- No changes to `cem_planner.py`, `tt_trajectory_inference.py`,
  `features.py`, `backends/`, `instruction_parser/`, or any
  `configs/*.yaml` — this tune-up is scoped to the three files named
  above (`trajectory_collection.py`, `trajectory_model.py`,
  `trajectory_control_loop.py`) plus their tests.
- Every hardware-touching script keeps the established deferred-import
  convention (`tt_trajectory_inference`/`ttnn` imported inside
  `if __name__ == "__main__":` or `main()`, never at module top level) —
  already true in all three files today; no change should regress it.
- Every `gozer`-leased command follows the existing
  `--who "claude:tt-cv-agency" --reason "..."` convention.
- `weight_decay` and `step_fraction` are real, new, honestly-reported
  first attempts, not values tuned post-hoc to produce a target capstone
  number.

## Out of scope

- Any change to CEM's search budget/dimensionality (already fixed,
  2026-09-09).
- Any change to Stage 2's novelty archive itself (its diversity-first
  objective is correct for what it was built for; this tune-up adds a
  second, independent seeding source rather than changing the archive).
- Re-verifying `instruction_to_preset.py` against a real LLM (flagged as
  open in the live-LLM-verification section, not part of this tune-up).
- Any new CLI flags for `weight_decay`/`step_fraction` — both are
  function parameters with sensible defaults, consistent with how every
  existing tunable in these three files is already exposed.

## Verification plan

- Existing test suites for all three files (`test_trajectory_collection.py`,
  `test_trajectory_model.py` if it exists or the relevant test file,
  `test_trajectory_control_loop.py`) continue to pass unchanged where
  they test unchanged behavior; new tests cover the mixed-seeding split,
  the `weight_decay` plumbing, `estimate_action_gain_ratio_per_dimension`
  (both a synthetic-fixture unit test with a known constructed ratio, and
  a wiring test that `__main__` computes and prints it), and
  `step_fraction`'s effect on the applied CV delta.
- Full non-hardware suite (`pytest -q`) and hardware suite
  (`gozer run --chips 1 -- pytest -q -m hardware`) both green before the
  capstone step.
- The capstone step itself is the real verification: a live recollection,
  retrain, and control-loop run against real hardware, with honest
  before/after numbers reported in `CLAUDE.md` exactly as Decision 5
  describes, regardless of outcome.
