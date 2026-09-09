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
    # horizon=3 (not the original 5) x n_candidates/n_elite/n_iterations
    # raised well above the original 200/20/3: a final-review pass on the
    # live capstone found horizon=5 x action_dim=8 (a 40-dim search) at the
    # original budget left 5 of this patch's 8 CV channels dominated by
    # seed-to-seed sampling noise rather than real signal. Shrinking the
    # search dimension (horizon*action_dim: 40 -> 24) and growing the
    # budget together closes most of that gap without materially changing
    # per-control-step device-call count (horizon*n_iterations: 5*3=15
    # calls before, 3*6=18 after -- each call's batch just got bigger).
    horizon: int = 3,
    n_candidates: int = 600,
    n_elite: int = 60,
    n_iterations: int = 6,
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

        current_cv = np.array([backend.last_known_cv(ch) for ch in channels])
        action = cem_plan(
            current_state, current_cv, goal_features, predict_fn,
            action_dim=len(channels), horizon=horizon,
            n_candidates=n_candidates, n_elite=n_elite, n_iterations=n_iterations,
            action_std_init=action_std_init, max_action=max_action, rng=rng,
        )

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
