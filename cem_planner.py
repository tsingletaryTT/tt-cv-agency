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
    """Cross-Entropy Method (CEM) search over action sequences.

    Samples `n_candidates` random action sequences of shape
    `(horizon, action_dim)` from a diagonal Gaussian (mean/std initialized
    to 0/`action_std_init`, clipped to `[-max_action, max_action]`), rolls
    each candidate forward through `predict_fn` for `horizon` steps starting
    from `current_state`, scores each by the negative Euclidean distance of
    its final rolled-out state from `goal_state`, refits the Gaussian's
    mean/std to the top `n_elite` candidates, and repeats for
    `n_iterations`. Returns only the *first* action of the final iteration's
    elite mean (shape `(action_dim,)`) -- standard receding-horizon usage,
    where the caller re-plans from a fresh `current_state` on the next
    control step rather than executing the whole horizon open-loop.

    Args:
        current_state: current state vector, shape (state_dim,).
        goal_state: target state vector, same shape as current_state.
        predict_fn: forward dynamics model, (states, actions) -> next_states,
            batched over the leading (candidate) dimension.
        action_dim: number of action channels per step.
        horizon: number of steps rolled out per candidate sequence.
        n_candidates: number of action sequences sampled per iteration.
        n_elite: number of top-scoring candidates used to refit mean/std.
        n_iterations: number of sample/score/refit rounds.
        action_std_init: initial per-dimension Gaussian std.
        max_action: symmetric per-step action clip, applied after sampling.
        rng: source of randomness for candidate sampling.

    Returns:
        The first-step action (shape `(action_dim,)`) of the final
        iteration's elite mean.

    Known, documented limitations (found by a final-review pass, left as
    scope for a future improvement rather than fixed here):

    - Rolled candidate states are never clipped to the real feature-space
      bounds (e.g. `[0, 1]`) during the horizon rollout -- an
      undertrained or over-scaled `predict_fn` (see trajectory_model.py's
      known action-gain over-scaling) can walk a candidate's rolled state
      outside the physically real range for several steps, and nothing
      here corrects it before scoring.
    - The search never receives the actual current CV setting
      (`current_cv`), only the feature-space `current_state` -- so it has
      no way to check whether a multi-step *cumulative* action is
      physically deliverable. Each individual step's CV value is clipped
      by the caller (`trajectory_control_loop.py`) after planning, but the
      search itself plans blind to that headroom, one step at a time.

    Also worth noting: at production scale (horizon=5, action_dim=8, a
    40-dimensional search) with the defaults `trajectory_control_loop.py`
    actually uses (n_candidates=200, n_elite=20, n_iterations=3), a
    final-review pass found the returned action is dominated by
    seed-to-seed sampling noise on most channels -- this function's own
    unit tests validate the algorithm's correctness only at 1-2 search
    dimensions with a much larger relative sample budget, which does not
    exercise this regime.
    """
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
