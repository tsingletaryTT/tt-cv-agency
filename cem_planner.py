# cem_planner.py
from typing import Callable
import numpy as np


def cem_plan(
    current_state: np.ndarray,
    current_cv: np.ndarray,
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
    from `current_state`/`current_cv`, scores each by the negative Euclidean
    distance of its final rolled-out state from `goal_state`, refits the
    Gaussian's mean/std to the top `n_elite` candidates, and repeats for
    `n_iterations`. Returns only the *first* action of the final iteration's
    elite mean (shape `(action_dim,)`) -- standard receding-horizon usage,
    where the caller re-plans from a fresh `current_state`/`current_cv` on
    the next control step rather than executing the whole horizon open-loop.

    At each horizon step, the sampled per-step action is clipped against a
    *running* per-candidate CV tally (`current_cv` + every prior step's real
    delta so far), exactly the way `trajectory_collection.py` and
    `trajectory_control_loop.py` apply and log a real action -- so the
    search scores and refits toward the delta that would actually be
    delivered, not a delta that a later boundary clip would silently shrink.
    This closes a real-review-found gap: without it, CEM could plan a
    cumulative move across the horizon (e.g. 5 steps at max_action=0.15,
    up to +-0.75) that the physical CV range can never actually supply.
    Rolled-forward states are also clipped to `[0, 1]` after every
    `predict_fn` call, so a `predict_fn` with an over-scaled action gain
    (see `trajectory_model.py`'s documented action-gain over-scaling) can't
    walk a candidate into a feature vector outside the real measurable
    range and have that impossible state silently drive its score.

    Args:
        current_state: current state vector, shape (state_dim,).
        current_cv: current CV setting, shape (action_dim,), same channel
            order as the actions this returns -- used only to compute each
            candidate's real, boundary-respecting per-step delta; never
            applied to the backend directly.
        goal_state: target state vector, same shape as current_state.
        predict_fn: forward dynamics model, (states, actions) -> next_states,
            batched over the leading (candidate) dimension.
        action_dim: number of action channels per step.
        horizon: number of steps rolled out per candidate sequence.
        n_candidates: number of action sequences sampled per iteration.
        n_elite: number of top-scoring candidates used to refit mean/std.
        n_iterations: number of sample/score/refit rounds.
        action_std_init: initial per-dimension Gaussian std.
        max_action: symmetric per-step action clip, applied before the
            cumulative-CV clip described above.
        rng: source of randomness for candidate sampling.

    Returns:
        The first-step action (shape `(action_dim,)`) of the final
        iteration's elite mean.

    Still worth noting: a final-review pass found that at
    `trajectory_control_loop.py`'s *original* defaults (horizon=5,
    action_dim=8 -- a 40-dimensional search -- with n_candidates=200,
    n_elite=20, n_iterations=3), the returned action was dominated by
    seed-to-seed sampling noise on most channels. `trajectory_control_loop.py`
    now uses a smaller horizon and a larger sample budget specifically to
    address this (see its own module docstring/comments), and
    `test_cem_plan_converges_at_production_scale_dimensionality` below
    exercises the algorithm at that scale directly -- but the general
    lesson (this algorithm's required sample budget grows with
    `horizon * action_dim`, and needs checking whenever either changes)
    remains true beyond whatever specific numbers are current.
    """
    mean = np.zeros((horizon, action_dim))
    std = np.full((horizon, action_dim), action_std_init)

    for _ in range(n_iterations):
        sampled = rng.normal(
            loc=mean[np.newaxis, :, :], scale=std[np.newaxis, :, :],
            size=(n_candidates, horizon, action_dim),
        )
        sampled = np.clip(sampled, -max_action, max_action)

        states = np.tile(current_state, (n_candidates, 1))
        cv = np.tile(current_cv, (n_candidates, 1))
        actual_actions = np.empty_like(sampled)
        for step in range(horizon):
            next_cv = np.clip(cv + sampled[:, step, :], 0.0, 1.0)
            actual_step = next_cv - cv
            actual_actions[:, step, :] = actual_step
            states = np.clip(predict_fn(states, actual_step), 0.0, 1.0)
            cv = next_cv

        scores = -np.linalg.norm(states - goal_state, axis=1)

        elite_idx = np.argsort(scores)[-n_elite:]
        elite_actions = actual_actions[elite_idx]

        mean = elite_actions.mean(axis=0)
        std = elite_actions.std(axis=0)
        # Floor std above exactly zero so a later iteration's sampling
        # doesn't collapse to a single repeated candidate with no further
        # exploration (an all-identical elite set drives std to 0.0).
        std = np.maximum(std, 1e-6)

    return mean[0]
