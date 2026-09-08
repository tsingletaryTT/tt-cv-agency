# Stage 3: trajectory-predicting control — design

## Context

Stage 3 of the roadmap (`CLAUDE.md`, "A four-stage roadmap", Path C): predict
and search over short CV *action sequences* -- not single points -- to steer
the instrument toward a goal over time. This is the actual fulfillment of the
tt-vjepa2 action-conditioned-predictor-plus-CEM idea that started this whole
project (see the session's original research plan file for the CEM/world-model
background), finally justified now that Stage 0 gave this project a real
temporal instrument and Stage 2 gave it a real, non-uniform-random dataset of
interesting states to build on. Explicitly the most ambitious and most
dependent stage of the four.

**Why this needs new architecture, not an extension.** `model.py`'s
`InverseCVModel` and `control_loop.py`'s `run_control_loop` already do
single-shot, single-step control: `predict_fn(goal_features) -> target_cv`,
then step partway toward that target every iteration, replanning from
scratch each time with no notion of a multi-step trajectory. Stage 3 needs a
model that predicts *forward in time* from a `(state, action)` pair, and a
planner that searches over *sequences* of actions before committing to the
first one -- neither exists yet.

**How Stage 2 feeds this stage**, per the roadmap's own stated dependency:
`data/sequencer_novelty_archive.npz` (Stage 2's discovered library of
diverse CV settings) becomes a source of *episode starting points* for
trajectory data collection, instead of pure uniform-random starts -- so
training data is denser in the regions Stage 2 already found genuinely
different from each other, rather than spread thin and uniform across an
8-dimensional space that Stage 2 already showed has plenty of "boring"
regions (e.g. `vca_level=0`, silence).

Out of scope for this stage: upgrading Stage 1's `InstructionParser` so an
instruction means "steer toward this over time" instead of "set this static
recipe" -- the roadmap names this as the *next* step once Stage 3 exists,
not part of Stage 3 itself. Also out of scope: changing `features.py`,
`backends/`, `configs/*.yaml`, `novelty_archive.py`, or `explore.py` --
Stage 3 only *reads* Stage 2's saved archive file, it doesn't modify how
that archive is produced.

## Architecture

```
 random episodes of small CV-delta actions (seeded from Stage 2's
 discovered archive when available, else uniform-random starts)
                              │
                              ▼
        trajectory_collection.py: log (state, action, next_state)
                              │
                              ▼
      trajectory_model.py: train TrajectoryPredictor(state, action)
                            -> predicted_next_state
                              │
                              ▼
        tt_trajectory_inference.py: TrajectoryTTInferenceEngine
        (batched ttnn forward pass, behind a gozer chip lease)
                              │
                              ▼
   cem_planner.py: cem_plan(current_state, goal_state, predict_fn, ...)
   -- sample candidate action SEQUENCES, roll each through the predictor
      (chained, batched), score by final-state distance to goal, refit
      the sampling distribution to the elite fraction, repeat, return
      the first action of the refined distribution
                              │
                              ▼
      trajectory_control_loop.py: read state -> cem_plan -> apply
      first action -> settle -> repeat (same read-decide-act shape as
      run_control_loop, but "decide" is now a full rollout search)
```

### Action space: CV deltas, not absolute settings

Every action is a per-channel delta (`dict[str, float]` or an aligned
`np.ndarray`, matching `channels = backend.channels()` order the same way
Stage 2's `generate_candidate` does), clipped to `[-max_action, max_action]`
before being added to the current CV vector, and the *result* is clipped to
`[0, 1]` per channel (same boundary discipline as every prior stage). The
action actually **logged** during data collection is the real post-clip
delta (`applied_cv - previous_cv`), not the raw sampled delta -- if a
sampled delta would have pushed a channel outside `[0, 1]`, the clip already
changed what really happened, and training on the un-clipped intent instead
of the real effect would teach the model a systematically wrong dynamics
model right at the space's boundaries, exactly where a real controller
needs it to be least wrong.

### `trajectory_collection.py`

```python
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
    """Returns (states, actions, next_states), each shape
    (n_episodes * episode_length, state_dim) or (..., action_dim) for
    actions. Each episode starts from a fresh CV setting -- drawn in
    round-robin order from seed_cv_vectors (Stage 2's discovered archive,
    shape (n_seeds, n_channels)) when given, cycling back to the start if
    n_episodes exceeds n_seeds; falls back to a uniform-random start when
    seed_cv_vectors is None or empty -- then takes episode_length random-
    action steps, logging (state, action, next_state) at each step. The
    action logged is the real post-clip delta actually applied, never the
    raw sampled one (see the design note above)."""
```

Every episode: apply the starting CV, settle, read the starting state via
the existing `read_aggregated_window(backend, sample_rate,
aggregate_window_s)`. Then `episode_length` times: sample a raw delta
`rng.uniform(-max_action, max_action, size=len(channels))`, compute
`next_cv = np.clip(current_cv + raw_delta, 0.0, 1.0)`, apply it via
`backend.set_cv` per channel, settle, read the next state the same way,
log `(current_state, next_cv - current_cv, next_state)`, then advance
(`current_cv, current_state = next_cv, next_state`) for the next step in
the episode.

### `trajectory_model.py`

Mirrors `model.py`'s existing shape and conventions exactly, generalized
from `(features -> cv)` to `(state, action -> next_state)`:

```python
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
        return self.fc3(x)  # predicted next_state (absolute, not a delta --
        # keeping this consistent with model.py's existing direct-prediction
        # convention rather than introducing residual/delta prediction as a
        # second new idea in the same stage)
```

`train_predictor(states, actions, next_states, epochs, lr) -> TrajectoryPredictor`,
a `train_val_split_trajectories` (three-array analogue of `model.py`'s
existing `train_val_split`), and `evaluate_predictor_per_dimension` (held-out
MSE/R² per *state* dimension against a predict-the-training-mean baseline,
mirroring `evaluate_per_dimension`'s exact reasoning: an aggregate error
hides exactly the kind of per-dimension weak spot Stage 0's own history
already found this instrument has, e.g. `vco_fm`/`sweep_rate`).

`save_predictor_weights(model, path, action_channels)` -- same
`np.savez`-of-plain-arrays shape `model.py.save_weights` already uses, but
also records `action_channels` (the CV channel order the action dimension
was trained against, from `backend.channels()`) for the same channel-order-
provenance reason `save_weights` already records `channels` -- the
predictor's action input is per-channel, and applying it with the wrong
channel order would be a silent-wrong-dynamics failure mode, not a loud one.

### `cem_planner.py`

Pure numpy, no `ttnn`/hardware dependency at all -- the model is injected as
a plain callable, so this module is fully unit-testable against a synthetic
toy predictor.

```python
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
    """predict_fn(states, actions) -> next_states is BATCHED: states and
    actions are each shape (batch, dim), returns shape (batch, state_dim) --
    this is what lets one rollout step evaluate every candidate sequence's
    transition in a single forward pass instead of one call per candidate.

    Returns the first action (shape (action_dim,)) of the CEM sampling
    distribution's final mean, after n_iterations of: sample n_candidates
    action sequences (shape (n_candidates, horizon, action_dim)) from the
    current per-(step, channel) Gaussian mean/std, clipped to
    [-max_action, max_action]; roll each sequence through predict_fn one
    horizon-step at a time (chained: each step's predicted next_state
    becomes the next step's input state, batched across all n_candidates
    at once); score each candidate by the negative Euclidean distance of
    its FINAL rolled-out state to goal_state; take the n_elite
    highest-scoring candidates; refit the per-(step, channel) mean/std to
    that elite set's actual sampled actions; repeat."""
```

### `tt_trajectory_inference.py`

A second `ttnn`-backed engine, alongside (not replacing) `tt_inference.py`'s
existing `TTInferenceEngine` -- different input shape (concatenated
state+action) and, critically, **batched**: CEM needs `n_candidates`
forward passes per horizon step per CEM iteration, so `predict_next_state`
takes `states: np.ndarray` shape `(batch, state_dim)` and `actions:
np.ndarray` shape `(batch, action_dim)` and returns `(batch, state_dim)` in
one device call, not one call per candidate. Same defensive pattern as the
existing engine's `ChannelMismatchError`: raise if the loaded weights'
recorded `action_channels` doesn't match the backend's current
`channels()`, for the same "silent wrong-channel dynamics" reason. Same
`open_device`/`close_device`/construction-failure-closes-device pattern
`TTInferenceEngine.__init__` already established.

### `trajectory_control_loop.py`

```python
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
    seed: int = 0,
) -> list[np.ndarray]:
    """Same read-state -> decide -> act -> repeat shape as
    control_loop.py's run_control_loop, but 'decide' is cem_plan (a full
    rollout search) instead of a single-shot predict_fn(goal_features)
    call. Returns the same kind of feature-vector history list."""
```

Every iteration: read `current_state` via `read_aggregated_window`; if
within `convergence_threshold` of `goal_features`, sleep and continue (same
early-exit as `run_control_loop`); else call `cem_plan(current_state,
goal_features, predict_fn, action_dim=len(channels), horizon, ...)` to get
the next action, compute `next_cv = np.clip(current_cv + action, 0.0,
1.0)`, apply via `backend.set_cv` per channel, sleep `control_interval_s`.

## Global constraints

- No edits to `features.py`, `backends/`, `configs/*.yaml`,
  `novelty_archive.py`, or `explore.py` -- Stage 3 only reads Stage 2's
  saved archive file as an optional data source.
- CV values stay `[0, 1]` at every interface boundary; actions stay
  clipped to `[-max_action, max_action]` before being applied.
- Channel identity never hardcoded -- action dimension and order always
  come from `backend.channels()`, recorded with the trained weights the
  same way `model.py`/`tt_inference.py` already do for the reflex model.
- `cem_planner.py` has zero `ttnn`/hardware dependency -- it takes
  `predict_fn` as a plain injected callable, so its own correctness is
  fully testable against a synthetic toy predictor with no `gozer` lease
  needed.
- `tt_trajectory_inference.py` is the only new file that touches `ttnn`;
  any test that actually imports/exercises it runs under a `gozer` lease
  and is marked `hardware`, matching this project's existing convention
  for `tt_inference.py`'s own tests.
- New files are flat, at the repo root, matching this project's existing
  convention (`model.py`, `tt_inference.py`, `control_loop.py`, etc. all
  live there) -- no new package.

## Verification plan

1. **`cem_planner.py` is tested against a synthetic toy predictor with no
   hardware at all**: a simple, hand-known dynamics function (e.g.
   `next_state = state + action`, so the "correct" plan to reach a goal
   from a known start is analytically obvious) confirms CEM's sampling
   distribution actually converges toward that correct action over its
   iterations, and that a goal already reached returns a near-zero-norm
   action.
2. **`trajectory_model.py` is tested the same way `model.py` already is**:
   synthetic `(state, action, next_state)` triples with a learnable
   ground-truth relationship, confirming training reduces held-out
   per-dimension MSE and that `save_predictor_weights`/reload round-trips
   both the weights and the `action_channels` provenance correctly.
3. **`trajectory_collection.py` is tested against `FakeCVBackend`**: confirms
   the right number of transitions are logged, that the logged action is
   the real post-clip delta (not the raw sampled one -- construct a case
   where a sampled delta would clip, and confirm the logged action
   reflects the clipped result), and that archive-seeded starts cycle
   through `seed_cv_vectors` in order (round-robin) when given.
4. **`tt_trajectory_inference.py` gets a hardware-marked test under a
   `gozer` lease**, mirroring `tests/test_tt_inference.py`'s existing
   pattern: a small toy-trained weight set's batched `ttnn` forward pass
   matches a plain numpy/torch forward pass closely, and the
   channel-order mismatch guard raises as expected.
5. **`trajectory_control_loop.py` is tested against `FakeCVBackend` and a
   fake (non-hardware) `predict_fn`**: confirms the loop's wiring (state
   read -> `cem_plan` call -> CV applied and clipped -> settle -> repeat),
   independent of whether the real `ttnn` engine or a synthetic stand-in
   supplies `predict_fn`.
6. **Live capstone, same bar as every prior stage**: collect real
   trajectory data against whichever patch is live (seeded from Stage 2's
   `data/sequencer_novelty_archive.npz` when present), train the
   predictor and report held-out per-dimension MSE/R² honestly, then run
   `run_trajectory_control_loop` against a real goal with the trained
   model on real Tenstorrent hardware (`gozer` lease) and report the
   actual convergence result -- including if it's only partial, the way
   every prior stage's own capstone has.

Full existing test suite (94 non-hardware + 4 hardware) must stay green;
this stage adds tests to both counts (the new `tt_trajectory_inference.py`
test is hardware-marked).
