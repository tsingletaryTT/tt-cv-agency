# Stage 3: trajectory-predicting control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an action-conditioned forward predictor and a CEM rollout
planner, so the control loop searches over short CV action *sequences*
instead of taking one single-shot step per iteration.

**Architecture:** Five new flat files at the repo root (matching this
project's existing convention): `trajectory_collection.py` (logs
`(state, action, next_state)` transitions, optionally seeded from Stage
2's discovered archive), `trajectory_model.py` (the
`TrajectoryPredictor` and its training/evaluation scaffold, mirroring
`model.py`'s conventions), `cem_planner.py` (pure numpy, hardware-free
rollout search), `tt_trajectory_inference.py` (a second, batched
`ttnn`-backed engine alongside the existing `TTInferenceEngine`), and
`trajectory_control_loop.py` (the same read-decide-act shape as
`control_loop.py`, with CEM search as the "decide" step).

**Tech Stack:** Python 3.12, numpy, PyTorch, `ttnn` (behind `gozer`),
pytest, `backends.base.FakeCVBackend` for hardware-free tests.

**Spec:** `docs/superpowers/specs/2026-09-08-stage3-trajectory-control-design.md`

## Global Constraints

- No edits to `features.py`, `backends/`, `configs/*.yaml`,
  `novelty_archive.py`, or `explore.py` — Stage 3 only *reads* Stage 2's
  saved archive file as an optional data source.
- CV values stay `[0, 1]` at every interface boundary; actions stay
  clipped to `[-max_action, max_action]` before being applied, and the
  *logged* action during data collection is always the real post-clip
  delta, never the raw sampled one.
- Channel identity never hardcoded — action dimension/order always come
  from `backend.channels()`, recorded with trained weights the same way
  `model.py`/`tt_inference.py` already do (`action_channels`, not
  `channels`, to distinguish this new provenance field from the existing
  reflex-model one).
- `cem_planner.py` has zero `ttnn`/hardware dependency — `predict_fn` is
  always a plain injected callable.
- `tt_trajectory_inference.py` is the only new file that touches `ttnn`;
  every test that imports/exercises it is `@pytest.mark.hardware` and
  imports `ttnn` inside the test function (never at module scope), and
  runs under a `gozer` lease — matching `tests/test_tt_inference.py`'s
  existing convention exactly.
- New files are flat, at the repo root, no new package.
- Work happens directly on `main`, no worktree — this project's
  established precedent from Stages 0-2.

---

### Task 1: `trajectory_collection.py`

**Files:**
- Create: `trajectory_collection.py`
- Test: `tests/test_trajectory_collection.py`

**Interfaces:**
- Consumes: `backends.base.CVBackend`/`FakeCVBackend` (tests) and
  `backends.vcv_rack.VCVRackBackend` (real use, in `__main__`);
  `features.read_aggregated_window`.
- Produces: `collect_trajectories(backend, n_episodes, episode_length,
  max_action, settle_time_s, rng, sample_rate=None,
  aggregate_window_s=3.0, seed_cv_vectors=None) -> tuple[np.ndarray,
  np.ndarray, np.ndarray]` (states, actions, next_states). Task 2 and the
  capstone (Task 6) consume this function's saved output shape
  (`states`/`actions`/`next_states`/`channels` in an `.npz`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trajectory_collection.py`:

```python
import numpy as np
from backends.base import FakeCVBackend
from trajectory_collection import collect_trajectories


def test_collect_trajectories_shapes():
    backend = FakeCVBackend(channel_names=["a", "b", "c"])
    rng = np.random.default_rng(0)
    states, actions, next_states = collect_trajectories(
        backend, n_episodes=3, episode_length=4, max_action=0.1,
        settle_time_s=0.0, rng=rng, sample_rate=48000, aggregate_window_s=0.05,
    )
    assert states.shape == (12, 6)
    assert actions.shape == (12, 3)
    assert next_states.shape == (12, 6)


def test_collect_trajectories_logs_action_matching_actual_backend_cv_delta():
    # Regardless of what the RNG draws (including draws that get clipped
    # at the [0,1] boundary), the logged action for each transition must
    # equal the CV delta actually applied to the backend -- not the raw,
    # possibly out-of-bounds sampled delta. Verified by reconstructing the
    # actually-applied CV vectors directly from FakeCVBackend's own
    # set_cv_calls log (ground truth of what really happened), independent
    # of collect_trajectories' own internal bookkeeping.
    backend = FakeCVBackend(channel_names=["a", "b"])
    # A large max_action (2.0, well beyond the [0,1] range) and a starting
    # point at a boundary (via seed_cv_vectors) makes clipping certain to
    # occur on most/all steps, regardless of RNG seed.
    seed_cv_vectors = np.array([[1.0, 0.0]])
    rng = np.random.default_rng(3)
    n_episodes, episode_length = 1, 5
    states, actions, next_states = collect_trajectories(
        backend, n_episodes=n_episodes, episode_length=episode_length, max_action=2.0,
        settle_time_s=0.0, rng=rng, sample_rate=48000, aggregate_window_s=0.05,
        seed_cv_vectors=seed_cv_vectors,
    )

    # Reconstruct the real applied CV vectors per step, per channel, from
    # the backend's own call log: the first 2 calls are the episode's
    # starting CV (channels "a","b"), then each subsequent pair of calls
    # is one step's post-clip next_cv.
    calls = backend.set_cv_calls
    channel_order = ["a", "b"]
    cv_sequence = []
    for i in range(0, len(calls), 2):
        pair = dict(calls[i:i + 2])
        cv_sequence.append(np.array([pair[ch] for ch in channel_order]))

    assert len(cv_sequence) == 1 + episode_length  # start + one per step
    for step in range(episode_length):
        expected_action = cv_sequence[step + 1] - cv_sequence[step]
        assert np.allclose(actions[step], expected_action)
        # And every applied CV must be within bounds -- proof the clip
        # actually happened where needed.
        assert np.all(cv_sequence[step + 1] >= 0.0) and np.all(cv_sequence[step + 1] <= 1.0)


def test_collect_trajectories_cycles_through_seed_cv_vectors_round_robin():
    backend = FakeCVBackend(channel_names=["a"])
    seed_cv_vectors = np.array([[0.1], [0.9]])
    rng = np.random.default_rng(0)
    n_episodes, episode_length = 5, 1
    collect_trajectories(
        backend, n_episodes=n_episodes, episode_length=episode_length, max_action=0.0,
        settle_time_s=0.0, rng=rng, sample_rate=48000, aggregate_window_s=0.05,
        seed_cv_vectors=seed_cv_vectors,
    )
    # max_action=0.0 -> every sampled delta is exactly 0.0 (a degenerate
    # Uniform(0,0) draw), so each episode's starting CV (the first set_cv
    # call of that episode) is directly observable and must cycle
    # 0.1, 0.9, 0.1, 0.9, 0.1. Single channel "a" -> 2 calls per episode
    # (1 start + 1 step), so starts are at even indices.
    calls = backend.set_cv_calls
    starts = [calls[i][1] for i in range(0, len(calls), 2)]
    assert np.allclose(starts, [0.1, 0.9, 0.1, 0.9, 0.1])


def test_collect_trajectories_uses_uniform_random_starts_without_seed_vectors():
    backend = FakeCVBackend(channel_names=["a", "b"])
    rng = np.random.default_rng(5)
    states, actions, next_states = collect_trajectories(
        backend, n_episodes=2, episode_length=1, max_action=0.1,
        settle_time_s=0.0, rng=rng, sample_rate=48000, aggregate_window_s=0.05,
        seed_cv_vectors=None,
    )
    # No crash, correct shapes, and CV values used for starts stayed in
    # [0, 1] (uniform-random fallback, not seed-vector-driven).
    calls = backend.set_cv_calls
    assert all(0.0 <= value <= 1.0 for _, value in calls)
    assert states.shape == (2, 6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_trajectory_collection.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trajectory_collection'`

- [ ] **Step 3: Write the implementation**

Create `trajectory_collection.py`:

```python
# trajectory_collection.py
import time
import numpy as np

from backends.base import CVBackend
from features import read_aggregated_window


def collect_trajectories(
    backend: CVBackend,
    n_episodes: int,
    episode_length: int,
    max_action: float,
    settle_time_s: float,
    rng: np.random.Generator,
    sample_rate: int | None = None,
    aggregate_window_s: float = 3.0,
    seed_cv_vectors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sample_rate is None:
        sample_rate = backend.sample_rate()
    channels = backend.channels()
    n_channels = len(channels)
    n_transitions = n_episodes * episode_length

    states = np.zeros((n_transitions, 6))
    actions = np.zeros((n_transitions, n_channels))
    next_states = np.zeros((n_transitions, 6))

    has_seeds = seed_cv_vectors is not None and len(seed_cv_vectors) > 0
    transition_idx = 0

    for episode in range(n_episodes):
        if has_seeds:
            start_cv = np.asarray(seed_cv_vectors[episode % len(seed_cv_vectors)], dtype=np.float64)
        else:
            start_cv = rng.uniform(0.0, 1.0, size=n_channels)

        current_cv = start_cv
        for ch, value in zip(channels, current_cv):
            backend.set_cv(ch, float(value))
        time.sleep(settle_time_s)
        current_state = read_aggregated_window(backend, sample_rate, aggregate_window_s)

        for _ in range(episode_length):
            raw_delta = rng.uniform(-max_action, max_action, size=n_channels)
            next_cv = np.clip(current_cv + raw_delta, 0.0, 1.0)
            applied_action = next_cv - current_cv

            for ch, value in zip(channels, next_cv):
                backend.set_cv(ch, float(value))
            time.sleep(settle_time_s)
            next_state = read_aggregated_window(backend, sample_rate, aggregate_window_s)

            states[transition_idx] = current_state
            actions[transition_idx] = applied_action
            next_states[transition_idx] = next_state
            transition_idx += 1

            current_cv = next_cv
            current_state = next_state

    return states, actions, next_states


if __name__ == "__main__":
    from backends.vcv_rack import VCVRackBackend

    # configs/sequencer_test.yaml -- the current instrument, 8 CV channels.
    backend = VCVRackBackend("configs/sequencer_test.yaml")
    try:
        rng = np.random.default_rng(seed=0)
        # Seed episode starts from Stage 2's discovered novelty archive when
        # it exists, so trajectory data is denser in the regions Stage 2
        # already found genuinely different from each other -- falls back
        # to uniform-random starts if the file isn't there.
        seed_cv_vectors = None
        try:
            archive = np.load("data/sequencer_novelty_archive.npz")
            seed_cv_vectors = archive["cv"]
        except FileNotFoundError:
            pass

        # 60 episodes x 10 steps -- one episode per Stage 2 archive member
        # (60) when the archive is present, 10 steps/episode for a short
        # but real trajectory. At settle_time_s=0.5 + aggregate_window_s=3.0
        # per read (~3.5s), total runtime is roughly
        # n_episodes * (1 + episode_length) * 3.5s =~ 38-39 minutes.
        states, actions, next_states = collect_trajectories(
            backend, n_episodes=60, episode_length=10, max_action=0.15,
            settle_time_s=0.5, rng=rng, aggregate_window_s=3.0,
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
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add trajectory_collection.py tests/test_trajectory_collection.py
git commit -m "Add trajectory_collection.py: log (state, action, next_state) rollouts"
```

---

### Task 2: `trajectory_model.py`

**Files:**
- Create: `trajectory_model.py`
- Test: `tests/test_trajectory_model.py`

**Interfaces:**
- Consumes: `numpy`, `torch`/`torch.nn` only.
- Produces: `TrajectoryPredictor(state_dim=6, action_dim=8, hidden=32)`
  (`forward(state, action) -> predicted_next_state`),
  `train_predictor(states, actions, next_states, epochs=200, lr=1e-3) ->
  TrajectoryPredictor`, `train_val_split_trajectories(states, actions,
  next_states, val_fraction=0.2, seed=0) -> tuple[tuple, tuple]`,
  `evaluate_predictor_per_dimension(model, states_val, actions_val,
  next_states_val, baseline_mean) -> tuple[np.ndarray, np.ndarray]`,
  `save_predictor_weights(model, path, action_channels=None) -> None`.
  Task 4's `TrajectoryTTInferenceEngine` loads weights saved by
  `save_predictor_weights` and expects the same `W1/b1/W2/b2/W3/b3` (+
  optional `action_channels`) keys.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trajectory_model.py`:

```python
import numpy as np
import torch
from trajectory_model import (
    TrajectoryPredictor, evaluate_predictor_per_dimension, save_predictor_weights,
    train_predictor, train_val_split_trajectories,
)


def test_trajectory_predictor_forward_shape():
    model = TrajectoryPredictor(state_dim=6, action_dim=8)
    state = torch.randn(1, 6)
    action = torch.randn(1, 8)
    out = model(state, action)
    assert out.shape == (1, 6)


def test_trajectory_predictor_batched_shape():
    model = TrajectoryPredictor(state_dim=6, action_dim=8)
    state = torch.randn(5, 6)
    action = torch.randn(5, 8)
    out = model(state, action)
    assert out.shape == (5, 6)


def test_train_predictor_infers_dimensions_from_data():
    rng = np.random.default_rng(0)
    states = rng.uniform(0, 1, size=(50, 6))
    actions = rng.uniform(-0.1, 0.1, size=(50, 8))
    next_states = rng.uniform(0, 1, size=(50, 6))

    model = train_predictor(states, actions, next_states, epochs=1)

    assert model.fc1.in_features == 14  # state_dim(6) + action_dim(8)
    assert model.fc3.out_features == 6


def test_train_predictor_reduces_loss_on_learnable_synthetic_data():
    # Synthetic ground truth: next_state = clip(state + action, 0, 1) --
    # additive, easily learnable, state_dim == action_dim == 4 here.
    rng = np.random.default_rng(0)
    states = rng.uniform(0, 1, size=(500, 4))
    actions = rng.uniform(-0.1, 0.1, size=(500, 4))
    next_states = np.clip(states + actions, 0.0, 1.0)

    model = train_predictor(states, actions, next_states, epochs=400, lr=1e-2)

    test_state = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)
    test_action = torch.tensor([[0.05, -0.05, 0.0, 0.02]], dtype=torch.float32)
    prediction = model(test_state, test_action).detach().numpy()
    expected = np.array([[0.55, 0.45, 0.5, 0.52]])
    assert np.allclose(prediction, expected, atol=0.1)


def test_train_val_split_trajectories_partitions_without_overlap():
    rng = np.random.default_rng(0)
    states = rng.uniform(0, 1, size=(100, 6))
    actions = rng.uniform(-0.1, 0.1, size=(100, 8))
    next_states = rng.uniform(0, 1, size=(100, 6))

    (s_train, a_train, ns_train), (s_val, a_val, ns_val) = train_val_split_trajectories(
        states, actions, next_states, val_fraction=0.2, seed=0
    )
    assert len(s_train) == 80 and len(s_val) == 20
    assert len(a_train) == 80 and len(a_val) == 20
    assert len(ns_train) == 80 and len(ns_val) == 20
    combined = np.concatenate([s_train, s_val], axis=0)
    assert np.allclose(np.sort(states, axis=0), np.sort(combined, axis=0))


def test_train_val_split_trajectories_is_reproducible_given_same_seed():
    rng = np.random.default_rng(1)
    states = rng.uniform(0, 1, size=(50, 6))
    actions = rng.uniform(-0.1, 0.1, size=(50, 8))
    next_states = rng.uniform(0, 1, size=(50, 6))

    (s_train_a, _, _), (s_val_a, _, _) = train_val_split_trajectories(
        states, actions, next_states, val_fraction=0.2, seed=7
    )
    (s_train_b, _, _), (s_val_b, _, _) = train_val_split_trajectories(
        states, actions, next_states, val_fraction=0.2, seed=7
    )

    assert np.array_equal(s_train_a, s_train_b)
    assert np.array_equal(s_val_a, s_val_b)


def test_evaluate_predictor_per_dimension_reports_zero_mse_and_perfect_r2_for_exact_model():
    class PerfectModel(TrajectoryPredictor):
        def forward(self, state, action):
            return state  # predicts next_state == state, matching the fixture below

    model = PerfectModel(state_dim=3, action_dim=2)
    states_val = np.array([[0.1, 0.5, 0.9], [0.2, 0.4, 0.8]])
    actions_val = np.zeros((2, 2))
    next_states_val = states_val.copy()
    baseline_mean = np.array([0.15, 0.45, 0.85])

    mse, r2 = evaluate_predictor_per_dimension(model, states_val, actions_val, next_states_val, baseline_mean=baseline_mean)

    assert np.allclose(mse, 0.0, atol=1e-6)
    assert np.allclose(r2, 1.0, atol=1e-6)


def test_evaluate_predictor_per_dimension_reports_zero_r2_for_mean_predicting_model():
    baseline_mean = np.array([0.3, 0.5, 0.7])

    class MeanModel(TrajectoryPredictor):
        def forward(self, state, action):
            return torch.tensor(baseline_mean, dtype=torch.float32).unsqueeze(0).repeat(state.shape[0], 1)

    model = MeanModel(state_dim=3, action_dim=2)
    rng = np.random.default_rng(0)
    states_val = rng.uniform(0, 1, size=(20, 3))
    actions_val = rng.uniform(-0.1, 0.1, size=(20, 2))
    next_states_val = rng.uniform(0, 1, size=(20, 3))

    _, r2 = evaluate_predictor_per_dimension(model, states_val, actions_val, next_states_val, baseline_mean=baseline_mean)

    assert np.allclose(r2, 0.0, atol=1e-6)


def test_save_predictor_weights_writes_expected_arrays(tmp_path):
    model = TrajectoryPredictor(state_dim=6, action_dim=8)
    out_path = tmp_path / "weights.npz"
    save_predictor_weights(model, str(out_path))

    loaded = np.load(out_path)
    assert set(loaded.keys()) == {"W1", "b1", "W2", "b2", "W3", "b3"}
    assert loaded["W1"].shape == (14, 32)
    assert loaded["b1"].shape == (32,)
    assert loaded["W2"].shape == (32, 32)
    assert loaded["b2"].shape == (32,)
    assert loaded["W3"].shape == (32, 6)
    assert loaded["b3"].shape == (6,)


def test_save_predictor_weights_records_action_channels_when_given(tmp_path):
    model = TrajectoryPredictor(state_dim=6, action_dim=3)
    out_path = tmp_path / "weights.npz"
    save_predictor_weights(model, str(out_path), action_channels=["vco_freq", "vco_fm", "vca_level"])

    loaded = np.load(out_path)
    assert "action_channels" in loaded.files
    assert loaded["action_channels"].tolist() == ["vco_freq", "vco_fm", "vca_level"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_trajectory_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trajectory_model'`

- [ ] **Step 3: Write the implementation**

Create `trajectory_model.py`:

```python
# trajectory_model.py
import numpy as np
import torch
import torch.nn as nn


class TrajectoryPredictor(nn.Module):
    def __init__(self, state_dim: int = 6, action_dim: int = 8, hidden: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, state_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def train_predictor(
    states: np.ndarray, actions: np.ndarray, next_states: np.ndarray,
    epochs: int = 200, lr: float = 1e-3,
) -> TrajectoryPredictor:
    model = TrajectoryPredictor(state_dim=states.shape[1], action_dim=actions.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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


def train_val_split_trajectories(
    states: np.ndarray, actions: np.ndarray, next_states: np.ndarray,
    val_fraction: float = 0.2, seed: int = 0,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    n = len(states)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n)
    n_val = int(round(n * val_fraction))
    val_idx = shuffled[:n_val]
    train_idx = shuffled[n_val:]
    return (
        (states[train_idx], actions[train_idx], next_states[train_idx]),
        (states[val_idx], actions[val_idx], next_states[val_idx]),
    )


def evaluate_predictor_per_dimension(
    model: TrajectoryPredictor,
    states_val: np.ndarray, actions_val: np.ndarray, next_states_val: np.ndarray,
    baseline_mean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        prediction = model(
            torch.tensor(states_val, dtype=torch.float32),
            torch.tensor(actions_val, dtype=torch.float32),
        ).numpy()

    mse_per_dim = np.mean((prediction - next_states_val) ** 2, axis=0)

    ss_res = np.sum((prediction - next_states_val) ** 2, axis=0)
    ss_tot = np.sum((next_states_val - baseline_mean) ** 2, axis=0)
    r2_per_dim = np.where(ss_tot > 0, 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, 1.0), np.nan)

    return mse_per_dim, r2_per_dim


def save_predictor_weights(model: TrajectoryPredictor, path: str, action_channels: list[str] | None = None) -> None:
    extra = {"action_channels": np.array(action_channels)} if action_channels is not None else {}
    np.savez(
        path,
        W1=model.fc1.weight.detach().numpy().T,
        b1=model.fc1.bias.detach().numpy(),
        W2=model.fc2.weight.detach().numpy().T,
        b2=model.fc2.bias.detach().numpy(),
        W3=model.fc3.weight.detach().numpy().T,
        b3=model.fc3.bias.detach().numpy(),
        **extra,
    )


if __name__ == "__main__":
    # data/sequencer_trajectory_dataset.npz -- written by
    # trajectory_collection.py's own __main__.
    data = np.load("data/sequencer_trajectory_dataset.npz")
    channels = data["channels"].tolist()

    (s_train, a_train, ns_train), (s_val, a_val, ns_val) = train_val_split_trajectories(
        data["states"], data["actions"], data["next_states"], val_fraction=0.2, seed=0
    )
    model = train_predictor(s_train, a_train, ns_train, epochs=500)

    train_mean = ns_train.mean(axis=0)
    mse_per_dim, r2_per_dim = evaluate_predictor_per_dimension(model, s_val, a_val, ns_val, baseline_mean=train_mean)

    state_dim_names = ["loud_mean", "loud_std", "bright_mean", "bright_std", "pitch_mean", "pitch_std"]
    print(f"train/val split: {len(s_train)} train / {len(s_val)} held-out validation transitions")
    print(f"{'state dim':<12} {'val MSE':>10} {'val R^2':>10}")
    for name, mse, r2 in zip(state_dim_names, mse_per_dim, r2_per_dim):
        print(f"{name:<12} {mse:>10.4f} {r2:>10.4f}")

    save_predictor_weights(model, "data/sequencer_trajectory_model_weights.npz", action_channels=channels)
    print("saved trained predictor weights to data/sequencer_trajectory_model_weights.npz")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_trajectory_model.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add trajectory_model.py tests/test_trajectory_model.py
git commit -m "Add trajectory_model.py: TrajectoryPredictor (state, action) -> next_state"
```

---

### Task 3: `cem_planner.py`

**Files:**
- Create: `cem_planner.py`
- Test: `tests/test_cem_planner.py`

**Interfaces:**
- Consumes: `numpy` only. `predict_fn` is an injected callable, batched:
  `predict_fn(states: np.ndarray, actions: np.ndarray) -> np.ndarray`
  where `states`/`actions` are shape `(batch, dim)`.
- Produces: `cem_plan(current_state, goal_state, predict_fn, action_dim,
  horizon, n_candidates, n_elite, n_iterations, action_std_init,
  max_action, rng) -> np.ndarray` (shape `(action_dim,)`, the first
  action of the refined CEM distribution). Task 5's
  `run_trajectory_control_loop` calls this every iteration with
  `predict_fn` bound to either `TrajectoryTTInferenceEngine.predict_next_state`
  (real use) or a test double.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cem_planner.py`:

```python
import numpy as np
from cem_planner import cem_plan


def additive_predict_fn(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    return states + actions


def test_cem_plan_converges_to_correct_action_single_step_toy_dynamics():
    rng = np.random.default_rng(0)
    current_state = np.array([0.0, 0.0])
    goal_state = np.array([1.0, -0.5])

    action = cem_plan(
        current_state, goal_state, predict_fn=additive_predict_fn,
        action_dim=2, horizon=1, n_candidates=500, n_elite=50,
        n_iterations=8, action_std_init=1.0, max_action=2.0, rng=rng,
    )
    assert action.shape == (2,)
    assert np.allclose(action, [1.0, -0.5], atol=0.15)


def test_cem_plan_multi_step_horizon_moves_meaningfully_toward_goal():
    # horizon=3, additive dynamics -- any split of the goal-reaching delta
    # across 3 steps scores equally well under this toy dynamics, so this
    # checks the first returned action moves state meaningfully toward the
    # goal (not away from it, not near-zero) -- the property an MPC-style
    # caller actually depends on at every real control step.
    rng = np.random.default_rng(1)
    current_state = np.array([0.0])
    goal_state = np.array([0.9])

    action = cem_plan(
        current_state, goal_state, predict_fn=additive_predict_fn,
        action_dim=1, horizon=3, n_candidates=500, n_elite=50,
        n_iterations=8, action_std_init=0.5, max_action=0.5, rng=rng,
    )
    assert action.shape == (1,)
    assert action[0] > 0.1  # meaningfully toward the goal, not stalled/backward


def test_cem_plan_returns_near_zero_action_when_already_at_goal():
    rng = np.random.default_rng(2)
    current_state = np.array([0.5, 0.5])
    goal_state = np.array([0.5, 0.5])

    action = cem_plan(
        current_state, goal_state, predict_fn=additive_predict_fn,
        action_dim=2, horizon=1, n_candidates=300, n_elite=30,
        n_iterations=5, action_std_init=0.3, max_action=1.0, rng=rng,
    )
    assert np.allclose(action, [0.0, 0.0], atol=0.1)


def test_cem_plan_respects_max_action_clipping():
    rng = np.random.default_rng(3)
    current_state = np.array([0.0])
    goal_state = np.array([100.0])  # unreachable in one bounded step

    action = cem_plan(
        current_state, goal_state, predict_fn=additive_predict_fn,
        action_dim=1, horizon=1, n_candidates=200, n_elite=20,
        n_iterations=5, action_std_init=1.0, max_action=0.3, rng=rng,
    )
    assert action[0] <= 0.3 + 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_cem_planner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cem_planner'`

- [ ] **Step 3: Write the implementation**

Create `cem_planner.py`:

```python
# cem_planner.py
from typing import Callable
import numpy as np


def cem_plan(
    current_state: np.ndarray,
    goal_state: np.ndarray,
    predict_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    action_dim: int,
    horizon: int,
    n_candidates: int,
    n_elite: int,
    n_iterations: int,
    action_std_init: float,
    max_action: float,
    rng: np.random.Generator,
) -> np.ndarray:
    mean = np.zeros((horizon, action_dim))
    std = np.full((horizon, action_dim), action_std_init)

    for _ in range(n_iterations):
        candidates = rng.normal(
            loc=mean[np.newaxis, :, :], scale=std[np.newaxis, :, :],
            size=(n_candidates, horizon, action_dim),
        )
        candidates = np.clip(candidates, -max_action, max_action)

        states = np.tile(current_state, (n_candidates, 1))
        for step in range(horizon):
            actions_step = candidates[:, step, :]
            states = predict_fn(states, actions_step)

        scores = -np.linalg.norm(states - goal_state, axis=1)

        elite_idx = np.argsort(scores)[-n_elite:]
        elite_candidates = candidates[elite_idx]

        mean = elite_candidates.mean(axis=0)
        std = elite_candidates.std(axis=0)
        # Floor std above exactly zero so a later iteration's sampling
        # doesn't collapse to a single repeated candidate with no further
        # exploration (an all-identical elite set drives std to 0.0).
        std = np.maximum(std, 1e-6)

    return mean[0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_cem_planner.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add cem_planner.py tests/test_cem_planner.py
git commit -m "Add cem_planner.py: CEM rollout search over action sequences"
```

---

### Task 4: `tt_trajectory_inference.py`

**Files:**
- Create: `tt_trajectory_inference.py`
- Test: `tests/test_tt_trajectory_inference.py`

**Interfaces:**
- Consumes: weights saved by Task 2's `save_predictor_weights`
  (`W1/b1/W2/b2/W3/b3` + optional `action_channels`); `ttnn`.
- Produces: `TrajectoryTTInferenceEngine(weights_path, device_id=0,
  expected_channels=None)` with `predict_next_state(states: np.ndarray,
  actions: np.ndarray) -> np.ndarray` (batched: shape `(batch, state_dim)`
  in, `(batch, state_dim)` out) and `close()`; `ActionChannelMismatchError`.
  Task 5's `__main__` and the capstone (Task 6) construct this engine and
  pass its `predict_next_state` as `cem_plan`'s `predict_fn`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tt_trajectory_inference.py`:

```python
import numpy as np
import pytest
import torch
from trajectory_model import TrajectoryPredictor, save_predictor_weights


@pytest.mark.hardware
def test_tt_trajectory_inference_matches_pytorch_reference(tmp_path):
    # Imported here (not at module scope) so merely collecting this test
    # file never touches ttnn / opens a device without a gozer lease.
    import ttnn  # noqa: F401
    from tt_trajectory_inference import TrajectoryTTInferenceEngine

    torch.manual_seed(0)
    model = TrajectoryPredictor(state_dim=6, action_dim=8)
    weights_path = str(tmp_path / "weights.npz")
    save_predictor_weights(model, weights_path)

    states = np.random.default_rng(0).uniform(0, 1, size=(4, 6)).astype(np.float32)
    actions = np.random.default_rng(1).uniform(-0.1, 0.1, size=(4, 8)).astype(np.float32)

    with torch.no_grad():
        reference = model(
            torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32)
        ).numpy()

    engine = TrajectoryTTInferenceEngine(weights_path=weights_path)
    try:
        result = engine.predict_next_state(states, actions)
    finally:
        engine.close()

    assert result.shape == (4, 6)
    # bfloat16 on-device precision -- same tolerance tt_inference.py's own
    # equivalent test already established as generously-but-really covering
    # this model size.
    assert np.allclose(result, reference, atol=0.1)


@pytest.mark.hardware
def test_tt_trajectory_inference_init_failure_does_not_leak_device(tmp_path):
    # Imported here (not at module scope) so merely collecting this test
    # file never touches ttnn / opens a device without a gozer lease.
    import ttnn
    from tt_trajectory_inference import TrajectoryTTInferenceEngine

    torch.manual_seed(0)
    model = TrajectoryPredictor(state_dim=6, action_dim=8)
    weights_path = str(tmp_path / "bad_weights.npz")
    save_predictor_weights(model, weights_path)

    # Corrupt the saved file by dropping a required key ("b3"), forcing
    # to_device(weights["b3"]) to raise a KeyError partway through
    # __init__'s weight-staging loop.
    good = np.load(weights_path)
    incomplete = {k: good[k] for k in good.files if k != "b3"}
    np.savez(weights_path, **incomplete)

    with pytest.raises(KeyError):
        TrajectoryTTInferenceEngine(weights_path=weights_path)

    # If the device from the failed construction were still open, this
    # would raise (device already active) instead of succeeding cleanly.
    probe_device = ttnn.open_device(device_id=0)
    ttnn.close_device(probe_device)


@pytest.mark.hardware
def test_tt_trajectory_inference_rejects_channel_order_mismatch(tmp_path):
    # Imported here (not at module scope) so merely collecting this test
    # file never touches ttnn / opens a device without a gozer lease.
    import ttnn  # noqa: F401
    from tt_trajectory_inference import ActionChannelMismatchError, TrajectoryTTInferenceEngine

    torch.manual_seed(0)
    model = TrajectoryPredictor(state_dim=6, action_dim=3)
    weights_path = str(tmp_path / "weights.npz")
    save_predictor_weights(model, weights_path, action_channels=["vco_freq", "vco_fm", "vca_level"])

    with pytest.raises(ActionChannelMismatchError):
        TrajectoryTTInferenceEngine(
            weights_path=weights_path, expected_channels=["vca_level", "vco_fm", "vco_freq"]
        )

    unlabeled_weights_path = str(tmp_path / "unlabeled_weights.npz")
    save_predictor_weights(model, unlabeled_weights_path)
    with pytest.raises(ActionChannelMismatchError):
        TrajectoryTTInferenceEngine(
            weights_path=unlabeled_weights_path, expected_channels=["vco_freq", "vco_fm", "vca_level"]
        )

    engine = TrajectoryTTInferenceEngine(
        weights_path=weights_path, expected_channels=["vco_freq", "vco_fm", "vca_level"]
    )
    engine.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `gozer run --chips 1 --who "claude:tt-cv-agency" --reason "Task 4 TDD red" -- python3 -m pytest tests/test_tt_trajectory_inference.py -v -m hardware`
Expected: FAIL with `ModuleNotFoundError: No module named 'tt_trajectory_inference'`

- [ ] **Step 3: Write the implementation**

Create `tt_trajectory_inference.py`:

```python
# tt_trajectory_inference.py
import numpy as np
import ttnn


class ActionChannelMismatchError(ValueError):
    """Raised when the CV channel order the loaded predictor weights were
    trained with doesn't match the backend's current channel order. The
    action input is per-channel and has no labels attached to it at
    inference time -- if the order silently disagreed with the backend's
    actual channels() ordering, actions would be interpreted against the
    wrong physical CV channel with no error at all, just a
    working-looking-but-wrong dynamics model."""


class TrajectoryTTInferenceEngine:
    def __init__(self, weights_path: str, device_id: int = 0, expected_channels: list[str] | None = None):
        weights = np.load(weights_path)

        if expected_channels is not None:
            trained_channels = weights["action_channels"].tolist() if "action_channels" in weights.files else None
            if trained_channels is None:
                raise ActionChannelMismatchError(
                    f"{weights_path} was saved without action-channel-order provenance "
                    "(trajectory_model.py's save_predictor_weights was called without "
                    f"`action_channels`), but the caller expects channel order {expected_channels!r} -- "
                    "retrain and save with `action_channels` so this can be verified."
                )
            if trained_channels != list(expected_channels):
                raise ActionChannelMismatchError(
                    f"{weights_path} was trained with action channel order "
                    f"{trained_channels!r}, but the backend's current channel "
                    f"order is {list(expected_channels)!r}. Applying actions in "
                    "that order would silently drive the wrong CV channel's dynamics."
                )

        self._device = ttnn.open_device(device_id=device_id)

        try:
            def to_device(array):
                import torch
                return ttnn.from_torch(
                    torch.tensor(array, dtype=torch.float32),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self._device,
                )

            self._W1 = to_device(weights["W1"])
            self._b1 = to_device(weights["b1"].reshape(1, -1))
            self._W2 = to_device(weights["W2"])
            self._b2 = to_device(weights["b2"].reshape(1, -1))
            self._W3 = to_device(weights["W3"])
            self._b3 = to_device(weights["b3"].reshape(1, -1))
        except Exception:
            # If any weight fails to stage (malformed/missing key in the
            # .npz), __init__ never returns an object -- so the caller
            # never gets a handle on which to call close(). Close the
            # device here ourselves before re-raising, so a construction
            # failure never leaks an open device on this shared box.
            ttnn.close_device(self._device)
            raise

    def predict_next_state(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        import torch
        x = np.concatenate([states, actions], axis=-1)
        x_device = ttnn.from_torch(
            torch.tensor(x, dtype=torch.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self._device,
        )
        h = ttnn.relu(ttnn.linear(x_device, self._W1, bias=self._b1))
        h = ttnn.relu(ttnn.linear(h, self._W2, bias=self._b2))
        out = ttnn.linear(h, self._W3, bias=self._b3)
        return ttnn.to_torch(out).float().numpy()

    def close(self) -> None:
        ttnn.close_device(self._device)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `gozer run --chips 1 --who "claude:tt-cv-agency" --reason "Task 4 TDD green" -- python3 -m pytest tests/test_tt_trajectory_inference.py -v -m hardware`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tt_trajectory_inference.py tests/test_tt_trajectory_inference.py
git commit -m "Add tt_trajectory_inference.py: batched ttnn forward pass for TrajectoryPredictor"
```

---

### Task 5: `trajectory_control_loop.py`

**Files:**
- Create: `trajectory_control_loop.py`
- Test: `tests/test_trajectory_control_loop.py`

**Interfaces:**
- Consumes: `backends.base.CVBackend`/`FakeCVBackend` (tests) and
  `backends.vcv_rack.VCVRackBackend` + `tt_trajectory_inference.TrajectoryTTInferenceEngine`
  (real use, in `__main__`); `cem_planner.cem_plan`;
  `features.read_aggregated_window`.
- Produces: `run_trajectory_control_loop(backend, predict_fn,
  goal_features, horizon=5, n_candidates=200, n_elite=20, n_iterations=3,
  action_std_init=0.1, max_action=0.15, control_interval_s=0.1,
  max_iterations=100, convergence_threshold=0.05, aggregate_window_s=3.0,
  sample_rate=None, seed=0) -> list[np.ndarray]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trajectory_control_loop.py`:

```python
import numpy as np
from backends.base import FakeCVBackend
from trajectory_control_loop import run_trajectory_control_loop


class LinearFakeBackend(FakeCVBackend):
    """Encodes current vca_level directly as the audio block's amplitude,
    mirroring tests/test_control_loop.py's own LinearFakeBackend exactly,
    so this loop's convergence is checkable end-to-end without a real
    instrument or a chip."""

    def read_audio_block(self) -> np.ndarray:
        level = self.last_known_cv("vca_level")
        t = np.arange(4096) / 48000.0
        return (level * 0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def additive_predict_fn(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    # A toy world model that's an EXACT match for LinearFakeBackend's real
    # loudness-vs-vca_level relationship: mean loudness = rms/LOUDNESS_REF_RMS
    # = (level * 0.5 / sqrt(2)) / 0.4, and level moves by exactly the applied
    # action (channel order below puts vca_level last), so
    # next_loudness = loudness + action[-1] * scale exactly, absent CV
    # clipping at the [0,1] boundary.
    scale = 0.5 / (0.4 * np.sqrt(2))
    next_states = states.copy()
    next_states[:, 0] = np.clip(states[:, 0] + actions[:, -1] * scale, 0.0, 1.0)
    return next_states


def test_trajectory_control_loop_converges_toward_loudness_goal():
    backend = LinearFakeBackend(channel_names=["vco_freq", "vco_fm", "vca_level"])
    goal = np.array([0.8, 0.0, 0.5, 0.0, 0.5, 0.0])

    history = run_trajectory_control_loop(
        backend, predict_fn=additive_predict_fn, goal_features=goal,
        horizon=2, n_candidates=100, n_elite=10, n_iterations=3,
        action_std_init=0.2, max_action=0.3,
        sample_rate=48000, control_interval_s=0.0, max_iterations=15,
        convergence_threshold=0.05, aggregate_window_s=0.1, seed=0,
    )

    assert len(history) > 1
    first_error = abs(history[0][0] - goal[0])
    last_error = abs(history[-1][0] - goal[0])
    assert last_error < first_error
    assert last_error < 0.1


def test_trajectory_control_loop_records_one_state_per_iteration():
    backend = LinearFakeBackend(channel_names=["vco_freq", "vco_fm", "vca_level"])
    goal = np.array([0.5, 0.0, 0.5, 0.0, 0.5, 0.0])
    history = run_trajectory_control_loop(
        backend, predict_fn=additive_predict_fn, goal_features=goal,
        horizon=1, n_candidates=50, n_elite=5, n_iterations=2,
        sample_rate=48000, control_interval_s=0.0, max_iterations=5,
        convergence_threshold=-1.0, aggregate_window_s=0.1, seed=0,
    )
    assert len(history) == 5
    for entry in history:
        assert entry.shape == (6,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_trajectory_control_loop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trajectory_control_loop'`

- [ ] **Step 3: Write the implementation**

Create `trajectory_control_loop.py`:

```python
# trajectory_control_loop.py
import time
from typing import Callable
import numpy as np

from backends.base import CVBackend
from cem_planner import cem_plan
from features import read_aggregated_window


def run_trajectory_control_loop(
    backend: CVBackend,
    predict_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    goal_features: np.ndarray,
    horizon: int = 5,
    n_candidates: int = 200,
    n_elite: int = 20,
    n_iterations: int = 3,
    action_std_init: float = 0.1,
    max_action: float = 0.15,
    control_interval_s: float = 0.1,
    max_iterations: int = 100,
    convergence_threshold: float = 0.05,
    aggregate_window_s: float = 3.0,
    sample_rate: int | None = None,
    seed: int = 0,
) -> list[np.ndarray]:
    if sample_rate is None:
        sample_rate = backend.sample_rate()
    channels = backend.channels()
    rng = np.random.default_rng(seed)
    history: list[np.ndarray] = []

    for _ in range(max_iterations):
        current_state = read_aggregated_window(backend, sample_rate, aggregate_window_s)
        history.append(current_state)

        if np.linalg.norm(current_state - goal_features) < convergence_threshold:
            time.sleep(control_interval_s)
            continue

        action = cem_plan(
            current_state, goal_features, predict_fn,
            action_dim=len(channels), horizon=horizon,
            n_candidates=n_candidates, n_elite=n_elite, n_iterations=n_iterations,
            action_std_init=action_std_init, max_action=max_action, rng=rng,
        )

        current_cv = np.array([backend.last_known_cv(ch) for ch in channels])
        next_cv = np.clip(current_cv + action, 0.0, 1.0)

        for ch, value in zip(channels, next_cv):
            backend.set_cv(ch, float(value))

        time.sleep(control_interval_s)

    return history


if __name__ == "__main__":
    import sys

    from backends.vcv_rack import VCVRackBackend
    from tt_trajectory_inference import TrajectoryTTInferenceEngine

    # 6-dim goal: [mean, std] x [loudness, brightness, pitch], same
    # convention as control_loop.py's own __main__.
    goal = (
        np.array([float(x) for x in sys.argv[1:7]])
        if len(sys.argv) >= 7
        else np.array([0.5, 0.05, 0.5, 0.1, 0.5, 0.05])
    )

    backend = VCVRackBackend("configs/sequencer_test.yaml")
    try:
        engine = TrajectoryTTInferenceEngine(
            weights_path="data/sequencer_trajectory_model_weights.npz",
            expected_channels=backend.channels(),
        )
        try:
            history = run_trajectory_control_loop(
                backend, predict_fn=engine.predict_next_state, goal_features=goal,
            )
            print(f"final measured features: {history[-1]}, goal: {goal}")
        finally:
            engine.close()
    finally:
        backend.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_trajectory_control_loop.py -v`
Expected: all PASS

- [ ] **Step 5: Run the full non-hardware suite to confirm no regressions**

Run: `python3 -m pytest -q`
Expected: all previously-passing tests still pass, plus this task's and
Tasks 1-3's new ones (deselected hardware count grows by Task 4's 3 new
hardware tests, to 7).

- [ ] **Step 6: Commit**

```bash
git add trajectory_control_loop.py tests/test_trajectory_control_loop.py
git commit -m "Add trajectory_control_loop.py: CEM-planned control loop"
```

---

### Task 6: Live capstone run and documentation

**Files:**
- Modify: `CLAUDE.md` (append a dated section reporting the real run)
- Modify: `README.md` ("What's here" section, new entries for
  `trajectory_collection.py`, `trajectory_model.py`, `cem_planner.py`,
  `tt_trajectory_inference.py`, `trajectory_control_loop.py`)
- Create: `data/sequencer_trajectory_dataset.npz`,
  `data/sequencer_trajectory_model_weights.npz` (real outputs of the run
  — check whether `data/*.npz` files are git-tracked in this repo, per
  Stage 2's Task 3's own finding that none are, and follow the same
  convention)

**Interfaces:**
- Consumes: all five files from Tasks 1-5; `data/sequencer_novelty_archive.npz`
  from Stage 2 (as an episode-start seed source, if present); the already-existing
  `patches/sequencer_test.vcv` / `configs/sequencer_test.yaml`.
- Produces: nothing new for later work — this is the stage's own
  capstone/completion check, matching the role every prior stage's final
  task has played.

- [ ] **Step 1: Confirm (don't assume) the live environment is healthy**

Same health check Stage 2's Task 3 already established (search `CLAUDE.md`
for "PipeWire", "Launch check", and the most recent Stage 2 capstone
section for the exact `cwd=Rack2Free/` launch fix a prior session found):

```bash
pgrep -af "Rack2Free/Rack"
pw-link -l | grep -i vcv_loop
pactl get-default-sink
pactl get-default-source
```

If already healthy, skip straight to Step 2. If not, relaunch per
`CLAUDE.md`'s documented recipe (cwd must be `Rack2Free/`, per the launch
bug that same section root-caused) and re-establish PipeWire routing.

- [ ] **Step 2: Real trajectory data collection**

```bash
python3 trajectory_collection.py
```

(uses `trajectory_collection.py`'s own `__main__` defaults: 60 episodes x
10 steps, seeded from `data/sequencer_novelty_archive.npz` if present).
At ~3.5s/read this is roughly 38-39 minutes — background it
(`nohup ... & disown`, poll in a few-minute chunks) per this project's
established long-collection pattern, not foregrounded and blocking.

- [ ] **Step 3: Train the predictor and report held-out per-dimension MSE/R²**

```bash
python3 trajectory_model.py
```

Report the printed per-state-dimension table honestly in a new dated
`CLAUDE.md` section — including if some dimensions learned poorly (Stage
0's own history already found some of this instrument's 8 channels are
weak/near-invisible to a windowed feature read, e.g. `sweep_rate`,
`vco_fm`/`filter_env_amount`'s gating limitation — a poorly-learned
trajectory dimension tied to one of those same channels would not be a
surprising or newly-introduced problem, and should be reported as
consistent with that known history, not as a fresh mystery).

- [ ] **Step 4: Live control-loop run against a real goal**

```bash
gozer run --chips 1 --who "claude:tt-cv-agency" --reason "Stage 3 capstone: live trajectory control loop" -- python3 trajectory_control_loop.py 0.5 0.05 0.5 0.1 0.5 0.05
```

(the same default 6-dim goal `control_loop.py`'s own `__main__` uses, for
a fair before/after comparison against the reflex model's own Stage 0
result against the identical goal). Report the final measured feature
vector vs. goal honestly, computing the same per-dimension absolute error
and Euclidean-norm-of-error this project's `CLAUDE.md` has reported for
every control-loop run so far (Phase 2, Stage 0) — including if
convergence is only partial, which every prior stage's own capstone has
also found.

- [ ] **Step 5: Update README's "What's here" section**

Add entries for `trajectory_collection.py`, `trajectory_model.py`,
`cem_planner.py`, `tt_trajectory_inference.py`, and
`trajectory_control_loop.py` immediately after the existing `explore.py`/
`novelty_archive.py` entries, matching the existing one-paragraph-each
style. No new Requirements-line entry needed — this stage adds no new
Python package dependency (only new local modules).

- [ ] **Step 6: Full regression check**

```bash
python3 -m pytest -q
gozer run --chips 1 --who "claude:tt-cv-agency" --reason "final Stage 3 suite check" -- python3 -m pytest -q -m hardware
```

Expected: the non-hardware count grows by this stage's new tests over
whatever baseline Task 5 left; the hardware count grows by Task 4's 3 new
tests over whatever baseline Task 5 left (was 4 before this stage; should
be 7 after).

- [ ] **Step 7: Commit**

```bash
git add trajectory_collection.py trajectory_model.py cem_planner.py \
  tt_trajectory_inference.py trajectory_control_loop.py \
  tests/test_trajectory_collection.py tests/test_trajectory_model.py \
  tests/test_cem_planner.py tests/test_tt_trajectory_inference.py \
  tests/test_trajectory_control_loop.py CLAUDE.md README.md
git add data/sequencer_trajectory_dataset.npz data/sequencer_trajectory_model_weights.npz  # only if data/*.npz files are already tracked in this repo -- check first, per Stage 2's own finding that none currently are
git commit -m "Stage 3 capstone: live trajectory data collection, predictor training, CEM control-loop run, docs"
```
