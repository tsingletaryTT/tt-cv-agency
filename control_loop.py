import time
from typing import Callable
import numpy as np

from backends.base import CVBackend
from features import extract_features


def run_control_loop(
    backend: CVBackend,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    goal_features: np.ndarray,
    sample_rate: int,
    step_fraction: float = 0.3,
    control_interval_s: float = 0.1,
    max_iterations: int = 100,
    convergence_threshold: float = 0.05,
) -> list[np.ndarray]:
    channels = backend.channels()
    history: list[np.ndarray] = []

    for _ in range(max_iterations):
        block = backend.read_audio_block()
        current_features = extract_features(block, sample_rate)
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
    engine = TTInferenceEngine(weights_path="data/model_weights.npz")
    try:
        history = run_control_loop(
            backend, predict_fn=engine.predict_cv, goal_features=goal, sample_rate=48000,
        )
        print(f"final measured features: {history[-1]}, goal: {goal}")
    finally:
        engine.close()
        backend.close()
