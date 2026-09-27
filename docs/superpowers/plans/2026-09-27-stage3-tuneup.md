# Stage 3 Tune-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the three remaining items from the Stage 3 final-review punch list (goal-coverage in trajectory data collection, action-gain calibration, control-loop damping), align the two control loops' measurement window, and re-verify live against the exact goal that scored worst this session.

**Architecture:** Three independent, same-shaped edits to the three existing Stage 3 files (`trajectory_collection.py`, `trajectory_model.py`, `trajectory_control_loop.py`) — a new pure helper function in each, wired into that file's own `__main__` — followed by a capstone task that recollects data, retrains, and runs a live control-loop verification using the new code.

**Tech Stack:** Python, numpy, PyTorch, pytest, existing `FakeCVBackend`/`VCVRackBackend`, `gozer` chip leasing.

**Spec:** `docs/superpowers/specs/2026-09-27-stage3-tuneup-design.md`

## Global Constraints

- No changes to `cem_planner.py`, `tt_trajectory_inference.py`, `features.py`, `backends/`, `instruction_parser/`, or any `configs/*.yaml`.
- Every hardware-touching script keeps `ttnn`-dependent imports (`tt_trajectory_inference`) deferred inside `if __name__ == "__main__":`/`main()`, never at module top level.
- Every `gozer`-leased command uses `--who "claude:tt-cv-agency" --reason "..."`.
- `weight_decay=1e-3` and `step_fraction=0.5` are first attempts, reported honestly in the capstone regardless of outcome — never retuned post-hoc to force a better-looking number.
- `data/` is gitignored — the capstone's regenerated dataset/weights files are never `git add`ed; only `CLAUDE.md` gets committed.

## Review Focus

- A missing or tiny (1-2 member) novelty archive fed into the new seeding helper — a person expects no crash and a correctly-shaped result, not a `ZeroDivisionError` or an off-by-one. (Task 1)
- `estimate_action_gain_ratio_per_dimension` scored against a validation set where some state dimension never changes at all (real movement is exactly zero) — a person expects a large finite ratio, not `inf`/`NaN`. (Task 2)
- `weight_decay=0.0` passed explicitly to `train_predictor` — a person expects byte-for-byte the same behavior as before this plan, not a silent change to already-shipped callers. (Task 2)
- `step_fraction=1.0` passed explicitly to `run_trajectory_control_loop` — a person migrating an existing caller expects this to exactly reproduce today's undamped behavior (applied delta == the planner's raw returned action), not some new scaling surprise. (Task 3)
- An odd-length archive (e.g. 61 members, not divisible by 2) fed into the new seeding helper — a person expects the returned array's length to still exactly match the input length, no silently dropped or duplicated seed vector. (Task 1)

---

### Task 1: Mixed archive/uniform seeding + window alignment in `trajectory_collection.py`

**Files:**
- Modify: `trajectory_collection.py` (add a new module-level function; update `__main__`; bump one default)
- Test: `tests/test_trajectory_collection.py`

**Interfaces:**
- Consumes: nothing new from other tasks.
- Produces: `build_mixed_seed_cv_vectors(archive_cv: np.ndarray, seed: int = 1) -> np.ndarray`, used by this file's own `__main__` only (no other task calls it).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_trajectory_collection.py`:

```python
from trajectory_collection import build_mixed_seed_cv_vectors


def test_build_mixed_seed_cv_vectors_keeps_first_half_archive_unchanged():
    archive_cv = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    result = build_mixed_seed_cv_vectors(archive_cv, seed=1)
    assert len(result) == len(archive_cv)
    # First half (n_total // 2 = 2 slots dropped from the END, so the
    # FIRST n_archive = n_total - n_uniform = 2 entries are kept verbatim).
    assert np.array_equal(result[:2], archive_cv[:2])


def test_build_mixed_seed_cv_vectors_second_half_is_uniform_random_in_range():
    archive_cv = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    result = build_mixed_seed_cv_vectors(archive_cv, seed=1)
    uniform_half = result[2:]
    assert uniform_half.shape == (2, 2)
    assert np.all(uniform_half >= 0.0) and np.all(uniform_half <= 1.0)
    # Not just coincidentally equal to the archive values it replaced.
    assert not np.array_equal(uniform_half, archive_cv[2:])


def test_build_mixed_seed_cv_vectors_is_reproducible_given_same_seed():
    archive_cv = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    result_a = build_mixed_seed_cv_vectors(archive_cv, seed=7)
    result_b = build_mixed_seed_cv_vectors(archive_cv, seed=7)
    assert np.array_equal(result_a, result_b)


def test_build_mixed_seed_cv_vectors_preserves_length_for_odd_input():
    archive_cv = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])  # 5 members, odd
    result = build_mixed_seed_cv_vectors(archive_cv, seed=1)
    assert len(result) == 5


def test_build_mixed_seed_cv_vectors_handles_tiny_archive_without_crashing():
    archive_cv = np.array([[0.5, 0.5]])  # single member
    result = build_mixed_seed_cv_vectors(archive_cv, seed=1)
    assert len(result) == 1
    assert result.shape == (1, 2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_trajectory_collection.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_mixed_seed_cv_vectors'`

- [ ] **Step 3: Implement `build_mixed_seed_cv_vectors` and wire it into `__main__`**

In `trajectory_collection.py`, add this function above the `if __name__ == "__main__":` block:

```python
def build_mixed_seed_cv_vectors(archive_cv: np.ndarray, seed: int = 1) -> np.ndarray:
    """Build episode-start seed vectors that mix Stage 2's diversity-archive
    members with fresh uniform-random CV vectors, addressing the Stage 3
    final-review finding that archive-only seeding clusters episode starts
    in extreme/near-silent corners and leaves the "typical" region most
    real goals ask for under-covered. The first `n_total - n_total // 2`
    entries of `archive_cv` are kept as-is (still get diversity-seeded
    episodes); the remaining `n_total // 2` slots are filled with freshly
    sampled uniform-random CV vectors instead, using `seed` (distinct from
    the main collection RNG) so the seeding draw is reproducible
    independent of per-step actions. Returns an array the same length as
    `archive_cv`, regardless of whether that length is even or odd, and
    handles a 1-member archive without crashing (n_uniform=0 in that case
    -- the single member is kept verbatim)."""
    n_total = len(archive_cv)
    n_uniform = n_total // 2
    n_archive = n_total - n_uniform
    rng = np.random.default_rng(seed)
    uniform_cv = rng.uniform(0.0, 1.0, size=(n_uniform, archive_cv.shape[1]))
    return np.concatenate([archive_cv[:n_archive], uniform_cv])
```

Change the `collect_trajectories` signature's default from `aggregate_window_s: float = 3.0` to `aggregate_window_s: float = 5.0` (matching `control_loop.py`'s own default, closing the window-mismatch gap documented in the spec).

Replace the `__main__` block's seeding section and its explicit `aggregate_window_s=3.0` with:

```python
if __name__ == "__main__":
    from backends.vcv_rack import VCVRackBackend

    # configs/sequencer_test.yaml -- the current instrument, 8 CV channels.
    backend = VCVRackBackend("configs/sequencer_test.yaml")
    try:
        rng = np.random.default_rng(seed=0)
        # Mixed seeding (Stage 3 tune-up, 2026-09-27): half the episode
        # starts still come from Stage 2's novelty archive (diversity),
        # half are freshly uniform-random (coverage of the "typical"
        # region most real goals ask for -- see build_mixed_seed_cv_vectors'
        # docstring and the design spec's Decision 1 for why archive-only
        # seeding under-covers this region). Falls back to uniform-random
        # starts entirely if the archive file isn't there, same as before.
        seed_cv_vectors = None
        try:
            archive = np.load("data/sequencer_novelty_archive.npz")
            seed_cv_vectors = build_mixed_seed_cv_vectors(archive["cv"], seed=1)
        except FileNotFoundError:
            pass

        # 60 episodes x 10 steps -- unchanged episode/step count from the
        # original Stage 3 collection. aggregate_window_s=5.0 (up from
        # 3.0, aligning with control_loop.py's own window -- see the
        # design spec's Decision 2) raises the per-read cost from ~3.5s to
        # ~5.5s (settle_time_s=0.5 + aggregate_window_s), so total runtime
        # rises from ~38-39 minutes to roughly ~60 minutes.
        states, actions, next_states = collect_trajectories(
            backend, n_episodes=60, episode_length=10, max_action=0.15,
            settle_time_s=0.5, rng=rng, aggregate_window_s=5.0,
            seed_cv_vectors=seed_cv_vectors,
        )
        import os
        os.makedirs("data", exist_ok=True)
        np.savez(
            "data/sequencer_trajectory_dataset.npz",
            states=states, actions=actions, next_states=next_states,
            channels=backend.channels(),
        )
        print(f"saved {len(states)} transitions to data/sequencer_trajectory_dataset.npz")
    finally:
        backend.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_trajectory_collection.py -v`
Expected: PASS (all tests, including the pre-existing 4)

- [ ] **Step 5: Commit**

```bash
git add trajectory_collection.py tests/test_trajectory_collection.py
git commit -m "Add mixed archive/uniform episode seeding, align aggregate_window_s to 5.0"
```

---

### Task 2: Action-gain regularization + a permanent gain-ratio gate in `trajectory_model.py`

**Files:**
- Modify: `trajectory_model.py` (add `weight_decay` param to `train_predictor`; add a new function; update `__main__`)
- Test: `tests/test_trajectory_model.py`

**Interfaces:**
- Consumes: nothing new from other tasks.
- Produces: `train_predictor(..., weight_decay: float = 0.0)` (existing signature, one new optional param, default preserves old behavior exactly); `estimate_action_gain_ratio_per_dimension(model: TrajectoryPredictor, states_val: np.ndarray, actions_val: np.ndarray, next_states_val: np.ndarray) -> np.ndarray` (shape `(state_dim,)`), used by this file's own `__main__` only.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_trajectory_model.py`:

```python
from trajectory_model import estimate_action_gain_ratio_per_dimension


def test_train_predictor_accepts_weight_decay_without_changing_default_behavior():
    # weight_decay=0.0 (the default) must be byte-for-byte identical to
    # today's un-regularized training -- same seed, same data, same epochs,
    # same everything else, just the new parameter passed explicitly.
    rng = np.random.default_rng(0)
    states = rng.uniform(0, 1, size=(50, 6))
    actions = rng.uniform(-0.1, 0.1, size=(50, 8))
    next_states = rng.uniform(0, 1, size=(50, 6))

    torch.manual_seed(0)
    model_a = train_predictor(states, actions, next_states, epochs=20)
    torch.manual_seed(0)
    model_b = train_predictor(states, actions, next_states, epochs=20, weight_decay=0.0)

    with torch.no_grad():
        out_a = model_a(torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32))
        out_b = model_b(torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32))
    assert torch.allclose(out_a, out_b)


def test_train_predictor_with_nonzero_weight_decay_produces_different_weights():
    rng = np.random.default_rng(0)
    states = rng.uniform(0, 1, size=(50, 6))
    actions = rng.uniform(-0.1, 0.1, size=(50, 8))
    next_states = rng.uniform(0, 1, size=(50, 6))

    torch.manual_seed(0)
    model_plain = train_predictor(states, actions, next_states, epochs=50)
    torch.manual_seed(0)
    model_regularized = train_predictor(states, actions, next_states, epochs=50, weight_decay=1e-2)

    assert not torch.allclose(model_plain.fc1.weight, model_regularized.fc1.weight)


def test_estimate_action_gain_ratio_matches_known_constructed_ratio():
    # A hand-built model whose predicted delta is EXACTLY 2x the real
    # observed delta on this fixture, for every state dimension --
    # confirms the ratio calculation itself, independent of any real
    # trained model's behavior.
    class DoubleGainModel(TrajectoryPredictor):
        def forward(self, state, action):
            real_delta = torch.tensor([0.1, 0.2], dtype=torch.float32)
            return state + 2.0 * real_delta.unsqueeze(0).repeat(state.shape[0], 1)

    model = DoubleGainModel(state_dim=2, action_dim=1)
    states_val = np.array([[0.3, 0.3], [0.5, 0.5]])
    actions_val = np.zeros((2, 1))
    next_states_val = states_val + np.array([0.1, 0.2])  # real delta

    ratio = estimate_action_gain_ratio_per_dimension(model, states_val, actions_val, next_states_val)
    assert np.allclose(ratio, [2.0, 2.0], atol=1e-4)


def test_estimate_action_gain_ratio_handles_zero_real_movement_without_crashing():
    # A state dimension that never changes in the validation data (real
    # delta median exactly 0.0) must not raise ZeroDivisionError/produce
    # inf or NaN -- the floor on the denominator must actually engage.
    class ConstantOffsetModel(TrajectoryPredictor):
        def forward(self, state, action):
            return state + torch.tensor([0.05, 0.0], dtype=torch.float32).unsqueeze(0).repeat(state.shape[0], 1)

    model = ConstantOffsetModel(state_dim=2, action_dim=1)
    states_val = np.array([[0.3, 0.5], [0.4, 0.5]])
    actions_val = np.zeros((2, 1))
    next_states_val = states_val.copy()  # dimension 1 never changes at all

    ratio = estimate_action_gain_ratio_per_dimension(model, states_val, actions_val, next_states_val)
    assert np.all(np.isfinite(ratio))
    assert ratio[0] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_trajectory_model.py -v`
Expected: FAIL — `train_predictor` rejects the unexpected `weight_decay` kwarg (`TypeError`), and `ImportError` for `estimate_action_gain_ratio_per_dimension`.

- [ ] **Step 3: Implement `weight_decay` and `estimate_action_gain_ratio_per_dimension`**

In `trajectory_model.py`, change `train_predictor`'s signature and optimizer construction:

```python
def train_predictor(
    states: np.ndarray, actions: np.ndarray, next_states: np.ndarray,
    epochs: int = 200, lr: float = 1e-3, hidden: int = 32, weight_decay: float = 0.0,
) -> TrajectoryPredictor:
    """Train a TrajectoryPredictor on a batch of (state, action, next_state) tuples.

    Performs full-batch gradient descent with the Adam optimizer and MSE loss.
    The state and action dimensions are inferred from the data shapes, allowing
    the same training function to work for any state/action dimensionality.
    `hidden` is exposed (default unchanged at 32, matching every existing
    caller's behavior) so a future dataset-size/capacity tradeoff can be
    explored without editing this function again.

    `weight_decay` (default 0.0, matching every existing caller's behavior
    exactly) adds L2 regularization via Adam's own weight_decay parameter --
    a real, previously-absent knob against the Stage 3 final-review finding
    that this model's action term can absorb measurement noise into an
    over-scaled gain when trained on a small, unregularized dataset. See
    estimate_action_gain_ratio_per_dimension below for how to check whether
    a given weight_decay value actually helped.
    """
    model = TrajectoryPredictor(state_dim=states.shape[1], action_dim=actions.shape[1], hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    s = torch.tensor(states, dtype=torch.float32)
    a = torch.tensor(actions, dtype=torch.float32)
    y = torch.tensor(next_states, dtype=torch.float32)

    for _ in range(epochs):
        optimizer.zero_grad()
        prediction = model(s, a)
        loss = loss_fn(prediction, y)
        loss.backward()
        optimizer.step()

    return model
```

Add this new function after `evaluate_persistence_baseline_per_dimension`:

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
    denominator avoids dividing by a near-zero real movement (a state
    dimension that happens to never change in the validation set would
    otherwise produce inf/NaN instead of a large-but-finite ratio).
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

Update `__main__` to use `weight_decay=1e-3` and print the gain-ratio table:

```python
if __name__ == "__main__":
    # data/sequencer_trajectory_dataset.npz -- written by
    # trajectory_collection.py's own __main__.
    data = np.load("data/sequencer_trajectory_dataset.npz")
    channels = data["channels"].tolist()

    (s_train, a_train, ns_train), (s_val, a_val, ns_val) = train_val_split_trajectories(
        data["states"], data["actions"], data["next_states"], val_fraction=0.2, seed=0
    )
    # weight_decay=1e-3 (Stage 3 tune-up, 2026-09-27): a first attempt at L2
    # regularization against the action-gain over-scaling the original
    # Stage 3 final review found (see estimate_action_gain_ratio_per_dimension
    # below and the design spec's Decision 3) -- not tuned to a target
    # outcome, reported honestly either way.
    model = train_predictor(s_train, a_train, ns_train, epochs=500, weight_decay=1e-3)

    train_mean = ns_train.mean(axis=0)
    mse_per_dim, r2_per_dim = evaluate_predictor_per_dimension(model, s_val, a_val, ns_val, baseline_mean=train_mean)
    persistence_mse, persistence_r2 = evaluate_persistence_baseline_per_dimension(
        s_val, ns_val, baseline_mean=train_mean
    )
    gain_ratio = estimate_action_gain_ratio_per_dimension(model, s_val, a_val, ns_val)

    state_dim_names = ["loud_mean", "loud_std", "bright_mean", "bright_std", "pitch_mean", "pitch_std"]
    print(f"train/val split: {len(s_train)} train / {len(s_val)} held-out validation transitions")
    print(f"{'state dim':<12} {'val MSE':>10} {'val R^2':>10} {'persist R^2':>12} {'gain ratio':>12}")
    for name, mse, r2, p_r2, gr in zip(state_dim_names, mse_per_dim, r2_per_dim, persistence_r2, gain_ratio):
        print(f"{name:<12} {mse:>10.4f} {r2:>10.4f} {p_r2:>12.4f} {gr:>12.4f}")

    save_predictor_weights(model, "data/sequencer_trajectory_model_weights.npz", action_channels=channels)
    print("saved trained predictor weights to data/sequencer_trajectory_model_weights.npz")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_trajectory_model.py -v`
Expected: PASS (all tests, including the pre-existing 11)

- [ ] **Step 5: Commit**

```bash
git add trajectory_model.py tests/test_trajectory_model.py
git commit -m "Add weight_decay regularization and a permanent action-gain-ratio gate"
```

---

### Task 3: Damping via `step_fraction` + window alignment in `trajectory_control_loop.py`

**Files:**
- Modify: `trajectory_control_loop.py` (add `step_fraction` param; apply it; bump one default; update `__main__`)
- Test: `tests/test_trajectory_control_loop.py`

**Interfaces:**
- Consumes: nothing new from other tasks.
- Produces: `run_trajectory_control_loop(..., step_fraction: float = 0.5)` (existing signature, one new optional param).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_trajectory_control_loop.py`:

```python
from unittest.mock import patch


def test_trajectory_control_loop_applies_step_fraction_to_planned_action():
    backend = FakeCVBackend(channel_names=["a", "b"])
    goal = np.array([0.5, 0.0, 0.5, 0.0, 0.5, 0.0])

    def dummy_predict_fn(states, actions):
        return states  # never actually invoked once cem_plan itself is mocked

    with patch("trajectory_control_loop.cem_plan", return_value=np.array([0.2, -0.1])):
        run_trajectory_control_loop(
            backend, predict_fn=dummy_predict_fn, goal_features=goal,
            horizon=1, n_candidates=5, n_elite=1, n_iterations=1,
            sample_rate=48000, control_interval_s=0.0, max_iterations=1,
            convergence_threshold=-1.0, aggregate_window_s=0.1, seed=0,
            step_fraction=0.5,
        )

    calls = dict(backend.set_cv_calls)
    # FakeCVBackend starts every channel at 0.5 (see backends/base.py);
    # cem_plan is mocked to always return [0.2, -0.1] regardless of state,
    # so step_fraction=0.5 must halve it before it's added:
    # 0.5 + 0.5*0.2 = 0.6, 0.5 + 0.5*(-0.1) = 0.45.
    assert np.isclose(calls["a"], 0.6)
    assert np.isclose(calls["b"], 0.45)


def test_trajectory_control_loop_step_fraction_1_0_reproduces_undamped_behavior():
    # A migrating caller that explicitly passes step_fraction=1.0 must get
    # exactly today's pre-tune-up behavior: the full planned action applied
    # unscaled.
    backend = FakeCVBackend(channel_names=["a", "b"])
    goal = np.array([0.5, 0.0, 0.5, 0.0, 0.5, 0.0])

    def dummy_predict_fn(states, actions):
        return states

    with patch("trajectory_control_loop.cem_plan", return_value=np.array([0.2, -0.1])):
        run_trajectory_control_loop(
            backend, predict_fn=dummy_predict_fn, goal_features=goal,
            horizon=1, n_candidates=5, n_elite=1, n_iterations=1,
            sample_rate=48000, control_interval_s=0.0, max_iterations=1,
            convergence_threshold=-1.0, aggregate_window_s=0.1, seed=0,
            step_fraction=1.0,
        )

    calls = dict(backend.set_cv_calls)
    assert np.isclose(calls["a"], 0.7)   # 0.5 + 1.0*0.2
    assert np.isclose(calls["b"], 0.4)   # 0.5 + 1.0*(-0.1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_trajectory_control_loop.py -v`
Expected: FAIL — `run_trajectory_control_loop` rejects the unexpected `step_fraction` kwarg (`TypeError`).

- [ ] **Step 3: Implement `step_fraction`**

In `trajectory_control_loop.py`, change the signature (add the new parameter after `convergence_threshold`, before `aggregate_window_s`, and bump `aggregate_window_s`'s own default):

```python
def run_trajectory_control_loop(
    backend: CVBackend,
    predict_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    goal_features: np.ndarray,
    horizon: int = 3,
    n_candidates: int = 600,
    n_elite: int = 60,
    n_iterations: int = 6,
    action_std_init: float = 0.1,
    max_action: float = 0.15,
    control_interval_s: float = 0.1,
    max_iterations: int = 100,
    convergence_threshold: float = 0.05,
    # step_fraction (Stage 3 tune-up, 2026-09-27): apply only this fraction
    # of cem_plan's returned first-step action, matching control_loop.py's
    # own existing step_fraction convention. cem_plan replans from a fresh
    # current_cv every iteration (receding-horizon usage), so damping the
    # applied action lets per-iteration planner noise (the original Stage 3
    # final review found 5 of 8 channels dominated by seed-to-seed sampling
    # noise at the pre-tune-up search budget) average down across
    # iterations instead of being applied at full magnitude every single
    # step -- the mechanism a receding-horizon controller needs to actually
    # settle rather than random-walk around the goal. step_fraction=1.0
    # reproduces the exact pre-tune-up undamped behavior.
    step_fraction: float = 0.5,
    aggregate_window_s: float = 5.0,
    sample_rate: int | None = None,
    seed: int = 0,
) -> list[np.ndarray]:
```

Change the CV-update line from:

```python
        next_cv = np.clip(current_cv + action, 0.0, 1.0)
```

to:

```python
        next_cv = np.clip(current_cv + step_fraction * action, 0.0, 1.0)
```

Update `__main__` (only the goal-array comment and the fact that `aggregate_window_s` now defaults to 5.0 needs no code change here, since `__main__` doesn't pass it explicitly today — leave `__main__` otherwise unchanged, it already calls `run_trajectory_control_loop` with only `backend`, `predict_fn`, `goal_features`, so the new defaults apply automatically).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_trajectory_control_loop.py -v`
Expected: PASS (all tests, including the pre-existing 3 — `test_trajectory_control_loop_converges_toward_loudness_goal` is expected to still pass with ample margin even with damping: the goal requires roughly 3 iterations' worth of movement at the damped step size, well inside its existing `max_iterations=15` budget. If it unexpectedly fails, increase that one test's `max_iterations` (e.g. to 30) to restore headroom for a now-damped loop — a legitimate adjustment to how many iterations a damped controller needs, not a change to what the test verifies.)

- [ ] **Step 5: Commit**

```bash
git add trajectory_control_loop.py tests/test_trajectory_control_loop.py
git commit -m "Add step_fraction damping, align aggregate_window_s to 5.0"
```

---

### Task 4: Capstone — recollect, retrain, live re-verify (executed directly, not dispatched to a subagent)

This task requires live VCV Rack process management, `gozer` chip leasing, and long real-time waits (~60 minute data collection, a live control-loop run) — matching every prior Stage 0-3 capstone in this project's history, all of which were run directly by the controlling session rather than a dispatched implementer subagent, since they need real-time hardware supervision and judgment calls a subagent can't make (checking `pw-link -l`/`gozer status` between steps, reacting to an unexpected mid-run failure). **Do not dispatch this task to a subagent-driven-development implementer** — the controller runs it directly after Tasks 1-3 are merged and reviewed clean.

**Files:**
- Modify: `CLAUDE.md` (append a new dated section, following this file's established format for every prior stage's live-verification writeup)
- Regenerates (gitignored, not committed): `data/sequencer_trajectory_dataset.npz`, `data/sequencer_trajectory_model_weights.npz`

**Interfaces:**
- Consumes: `trajectory_collection.py`'s updated `__main__` (Task 1), `trajectory_model.py`'s updated `__main__` (Task 2), `trajectory_control_loop.py`'s updated `run_trajectory_control_loop`/`__main__` (Task 3).
- Produces: nothing consumed by a later task — this is the plan's final step.

- [ ] **Step 1: Confirm the live environment is healthy**

Check VCV Rack is running against `patches/sequencer_test.vcv` (literal process name via `ps -eo pid,comm,args`, not a `pgrep -f` substring match — this project's own documented lesson about false negatives/positives in this sandboxed environment), `pw-link -l` shows `VCV Rack` routed through `vcv_loop`/`vcv_loop.monitor`, and `pactl get-default-sink`/`get-default-source` both correct. If not running: relaunch with `cd Rack2Free && LD_LIBRARY_PATH=. ./Rack ../patches/sequencer_test.vcv` (cwd MUST be `Rack2Free/`, per this project's documented `asset::systemDir` launch fix), after confirming `~/.local/share/Rack2/log.txt` ends in `END` (append it if not, to avoid the crash-recovery dialog).

- [ ] **Step 2: Recollect trajectory data**

```bash
gozer run --chips 1 --who "claude:tt-cv-agency" --reason "Stage 3 tune-up: goal-spanning, window-aligned trajectory recollection" -- \
  python3 trajectory_collection.py
```

Run in the background (matches this project's established long-collection pattern), expect ~60 minutes given the new `aggregate_window_s=5.0`. Confirm on completion: `saved 600 transitions to data/sequencer_trajectory_dataset.npz` printed, no exceptions.

- [ ] **Step 3: Retrain and record the gain-ratio table**

```bash
python3 trajectory_model.py
```

No hardware/gozer lease needed (pure PyTorch on CPU). Record the full printed table (MSE/R²/persistence R²/gain ratio per state dimension) for the write-up. Compare the gain-ratio column against the original Stage 3 final review's reported ratios (documented in `CLAUDE.md`'s "Stage 3 final-review correction" section, item 3) — report whatever the real comparison shows, whether the ratio improved, stayed flat, or didn't.

- [ ] **Step 4: Live re-verification against the goal that scored worst this session**

```bash
gozer run --chips 1 --who "claude:tt-cv-agency" --reason "Stage 3 tune-up: live re-verification" -- \
  python3 trajectory_control_loop.py 0.7 0.15 0.8 0.15 0.6 0.15
```

This is the exact goal `instruction_to_goal.py` parsed from "make it bright and steady" via Qwen3-0.6B this session (documented in `CLAUDE.md`'s "First real live-LLM verification" section), which scored a Euclidean error of 0.593 — the worst result in this project's history. Record the new `final measured features`/`goal` printout and compute the new per-dimension error and Euclidean norm.

- [ ] **Step 5: Confirm no regressions**

```bash
python3 -m pytest -q
gozer run --chips 1 --who "claude:tt-cv-agency" --reason "final Stage 3 tune-up suite check" -- python3 -m pytest -q -m hardware
```

Both must be green (same pass/deselect counts as before this plan, plus the new tests from Tasks 1-3).

- [ ] **Step 6: Write up the result in `CLAUDE.md`, honestly**

Append a new dated section (following the exact style of every prior stage's writeup in this file — a table of before/after numbers, an honest read that reports whatever actually happened rather than reframing it, and explicit callouts if any of the four fixes (mixed seeding, window alignment, weight_decay, step_fraction) didn't help as hoped). Do not retune any parameter after seeing this result to force a better number, per the Global Constraints above.

- [ ] **Step 7: Commit**

```bash
git add CLAUDE.md
git commit -m "Document Stage 3 tune-up capstone: recollection, retrain, live re-verification"
```
