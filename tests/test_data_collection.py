import numpy as np
from backends.base import FakeCVBackend
from data_collection import collect_sweep_dataset


def test_collect_sweep_dataset_shapes_and_ranges():
    backend = FakeCVBackend(
        channel_names=["a", "b", "c"],
        audio_block=(0.3 * np.sin(2 * np.pi * 440 * np.arange(2048) / 48000)).astype(np.float32),
    )
    rng = np.random.default_rng(seed=42)
    cv_array, feature_array = collect_sweep_dataset(
        backend, n_samples=10, settle_time_s=0.0, sample_rate=48000, rng=rng
    )
    assert cv_array.shape == (10, 3)
    assert feature_array.shape == (10, 3)
    assert np.all(cv_array >= 0.0) and np.all(cv_array <= 1.0)


def test_collect_sweep_dataset_actually_sets_cv_on_backend():
    backend = FakeCVBackend(channel_names=["a", "b", "c"])
    rng = np.random.default_rng(seed=1)
    collect_sweep_dataset(backend, n_samples=5, settle_time_s=0.0, sample_rate=48000, rng=rng)
    # 3 channels x 5 samples = 15 set_cv calls
    assert len(backend.set_cv_calls) == 15
