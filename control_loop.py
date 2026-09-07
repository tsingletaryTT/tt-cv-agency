import time
from typing import Callable
import numpy as np

from backends.base import CVBackend
from features import read_aggregated_window


def run_control_loop(
    backend: CVBackend,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    goal_features: np.ndarray,
    sample_rate: int | None = None,
    step_fraction: float = 0.3,
    control_interval_s: float = 0.1,
    max_iterations: int = 100,
    convergence_threshold: float = 0.05,
    aggregate_window_s: float = 5.0,
) -> list[np.ndarray]:
    # sample_rate defaults to the backend's own real value -- passing it
    # explicitly is kept only for backward compatibility / test convenience.
    # Real callers should let this pull from backend.sample_rate() so
    # feature extraction is never fed a sample_rate that silently disagrees
    # with what the backend actually captured its audio at.
    if sample_rate is None:
        sample_rate = backend.sample_rate()
    # Same windowed-read rationale as collect_sweep_dataset: one instantaneous
    # block can't represent a looping pattern, so each iteration reads a whole
    # window of blocks spanning aggregate_window_s seconds and reduces it to
    # the 6-dim [mean, std] x [loudness, brightness, pitch] aggregate.
    # goal_features must now be 6-dim to match (breaking change -- see Task 4
    # brief). The __main__ block below was updated in Task 5 to pass a 6-dim
    # goal and point at the new 8-channel sequencer_test patch.
    channels = backend.channels()
    history: list[np.ndarray] = []

    for _ in range(max_iterations):
        current_features = read_aggregated_window(backend, sample_rate, aggregate_window_s)
        history.append(current_features)

        if np.linalg.norm(current_features - goal_features) < convergence_threshold:
            time.sleep(control_interval_s)
            continue

        target_cv = predict_fn(goal_features)
        current_cv = np.array([backend.last_known_cv(ch) for ch in channels])
        next_cv = current_cv + step_fraction * (target_cv - current_cv)
        next_cv = np.clip(next_cv, 0.0, 1.0)

        for ch, value in zip(channels, next_cv):
            backend.set_cv(ch, float(value))

        time.sleep(control_interval_s)

    return history


if __name__ == "__main__":
    import sys

    from backends.vcv_rack import VCVRackBackend
    from tt_inference import TTInferenceEngine

    # 6-dim goal: [mean, std] x [loudness, brightness, pitch], per
    # extract_features_aggregated (features.py). Accept it as 6 positional
    # argv values (sys.argv[1:7]) instead of the old 3; fall back to a
    # sensible default goal (mid-loudness, low variation, mid-brightness,
    # a bit of variation, mid-pitch, low variation) if none are given.
    goal = (
        np.array([float(x) for x in sys.argv[1:7]])
        if len(sys.argv) >= 7
        else np.array([0.5, 0.05, 0.5, 0.1, 0.5, 0.05])
    )

    # configs/sequencer_test.yaml -- the current (Stage 0) instrument: 8 CV
    # channels driving a sequencer + filter-sweep LFO + filter envelope on
    # top of the Minimoog signal path (see CLAUDE.md's Stage 0 sections).
    backend = VCVRackBackend("configs/sequencer_test.yaml")
    try:
        # expected_channels ties the loaded weights' trained channel order to
        # this backend's actual current channel order -- raises a clear
        # ChannelMismatchError instead of silently driving the wrong CV
        # channel if they ever disagree (e.g. a config/model mismatch).
        engine = TTInferenceEngine(
            weights_path="data/sequencer_model_weights.npz", expected_channels=backend.channels(),
        )
        try:
            history = run_control_loop(
                backend, predict_fn=engine.predict_cv, goal_features=goal,
            )
            print(f"final measured features: {history[-1]}, goal: {goal}")
        finally:
            engine.close()
    finally:
        backend.close()
