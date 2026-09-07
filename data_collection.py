# data_collection.py
import math
import time
import numpy as np

from backends.base import CVBackend
from features import extract_features_aggregated


def collect_sweep_dataset(
    backend: CVBackend,
    n_samples: int,
    settle_time_s: float,
    rng: np.random.Generator,
    sample_rate: int | None = None,
    aggregate_window_s: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    # sample_rate is pulled from the backend itself by default -- passing it
    # in independently is kept only for backward compatibility / test
    # convenience (e.g. a FakeCVBackend test that wants to assert behavior
    # at a sample_rate different from its own configured one). Real callers
    # should let this default to backend.sample_rate() so the value used for
    # feature extraction can never silently drift from what the backend
    # actually captured its audio at (this mismatch is exactly how the
    # estimate_pitch production-block-size bug stayed hidden).
    if sample_rate is None:
        sample_rate = backend.sample_rate()
    # A single instantaneous block can't represent a looping pattern (e.g.
    # the sequencer/filter-sweep patch from Task 1) -- read a whole window
    # of blocks spanning aggregate_window_s seconds and hand them all to
    # extract_features_aggregated, which reduces the window to [mean, std]
    # per feature (6-dim). block_size is pulled from the backend itself for
    # the same drift-proofing reason sample_rate is.
    blocks_per_window = max(1, math.ceil(aggregate_window_s * sample_rate / backend.block_size()))
    channels = backend.channels()
    cv_array = np.zeros((n_samples, len(channels)))
    feature_array = np.zeros((n_samples, 6))

    for i in range(n_samples):
        cv_vec = rng.uniform(0.0, 1.0, size=len(channels))
        for ch, value in zip(channels, cv_vec):
            backend.set_cv(ch, float(value))
        time.sleep(settle_time_s)
        blocks = [backend.read_audio_block() for _ in range(blocks_per_window)]
        feature_array[i] = extract_features_aggregated(blocks, sample_rate)
        cv_array[i] = cv_vec

    return cv_array, feature_array


if __name__ == "__main__":
    from backends.vcv_rack import VCVRackBackend

    backend = VCVRackBackend("configs/bridge_test.yaml")
    try:
        rng = np.random.default_rng(seed=0)
        # settle_time_s=0.5 -- confirmed generously above the real settle time
        # observed in Phase 1's round-trip verification (RMS transitions completed
        # within a few hundred ms); re-check with a stopwatch-style manual test
        # (Task 3 Step 6's pattern) if this dataset's model behaves oddly.
        cv_array, feature_array = collect_sweep_dataset(
            backend, n_samples=3000, settle_time_s=0.5, rng=rng
        )
        import os
        os.makedirs("data", exist_ok=True)
        np.savez("data/sweep_dataset.npz", cv=cv_array, features=feature_array, channels=backend.channels())
        print(f"saved {len(cv_array)} samples to data/sweep_dataset.npz")
    finally:
        # A crash partway through a several-minute collection run must not
        # leave the audio stream / MIDI port open -- close it even if the
        # loop above raised.
        backend.close()
