import math
import time
from typing import Callable
import numpy as np

from backends.base import CVBackend
from features import extract_features_aggregated


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
    # brief). The __main__ block below still passes a 3-dim goal and is left
    # as-is here on purpose: updating it to point at the new 6-feature/
    # 8-channel patch is Task 5's job, not this task's.
    blocks_per_window = max(1, math.ceil(aggregate_window_s * sample_rate / backend.block_size()))
    channels = backend.channels()
    history: list[np.ndarray] = []

    for _ in range(max_iterations):
        blocks = [backend.read_audio_block() for _ in range(blocks_per_window)]
        current_features = extract_features_aggregated(blocks, sample_rate)
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

    goal = np.array([float(x) for x in sys.argv[1:4]]) if len(sys.argv) >= 4 else np.array([0.7, 0.5, 0.5])

    backend = VCVRackBackend("configs/bridge_test.yaml")
    try:
        # expected_channels ties the loaded weights' trained channel order to
        # this backend's actual current channel order -- raises a clear
        # ChannelMismatchError instead of silently driving the wrong CV
        # channel if they ever disagree (e.g. a config/model mismatch).
        engine = TTInferenceEngine(
            weights_path="data/model_weights.npz", expected_channels=backend.channels(),
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
