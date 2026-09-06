# data_collection.py
import time
import numpy as np

from backends.base import CVBackend
from features import extract_features


def collect_sweep_dataset(
    backend: CVBackend,
    n_samples: int,
    settle_time_s: float,
    sample_rate: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    channels = backend.channels()
    cv_array = np.zeros((n_samples, len(channels)))
    feature_array = np.zeros((n_samples, 3))

    for i in range(n_samples):
        cv_vec = rng.uniform(0.0, 1.0, size=len(channels))
        for ch, value in zip(channels, cv_vec):
            backend.set_cv(ch, float(value))
        time.sleep(settle_time_s)
        block = backend.read_audio_block()
        feature_array[i] = extract_features(block, sample_rate)
        cv_array[i] = cv_vec

    return cv_array, feature_array


if __name__ == "__main__":
    from backends.vcv_rack import VCVRackBackend

    backend = VCVRackBackend("configs/bridge_test.yaml")
    rng = np.random.default_rng(seed=0)
    # settle_time_s=0.5 -- confirmed generously above the real settle time
    # observed in Phase 1's round-trip verification (RMS transitions completed
    # within a few hundred ms); re-check with a stopwatch-style manual test
    # (Task 3 Step 6's pattern) if this dataset's model behaves oddly.
    cv_array, feature_array = collect_sweep_dataset(
        backend, n_samples=3000, settle_time_s=0.5, sample_rate=48000, rng=rng
    )
    import os
    os.makedirs("data", exist_ok=True)
    np.savez("data/sweep_dataset.npz", cv=cv_array, features=feature_array, channels=backend.channels())
    print(f"saved {len(cv_array)} samples to data/sweep_dataset.npz")
    backend.close()
