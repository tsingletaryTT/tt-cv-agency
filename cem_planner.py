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
