import numpy as np
from backends.base import FakeCVBackend
from data_collection import collect_sweep_dataset


class CountingFakeBackend(FakeCVBackend):
    """A FakeCVBackend that counts read_audio_block() calls, so tests can
    assert the windowed-read behavior actually reads multiple blocks per
    sample rather than just one."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read_audio_block_calls = 0

    def read_audio_block(self) -> np.ndarray:
        self.read_audio_block_calls += 1
        return super().read_audio_block()


def test_collect_sweep_dataset_shapes_and_ranges():
    backend = FakeCVBackend(
        channel_names=["a", "b", "c"],
        audio_block=(0.3 * np.sin(2 * np.pi * 440 * np.arange(2048) / 48000)).astype(np.float32),
    )
    rng = np.random.default_rng(seed=42)
    # aggregate_window_s kept small here (rather than the 5.0 production
    # default) purely so this test's per-block feature extraction doesn't
    # dominate the test suite's runtime -- it still exercises multiple
    # blocks per sample (see the dedicated windowing test below).
    cv_array, feature_array = collect_sweep_dataset(
        backend, n_samples=10, settle_time_s=0.0, sample_rate=48000, rng=rng,
        aggregate_window_s=0.1,
    )
    assert cv_array.shape == (10, 3)
    assert feature_array.shape == (10, 6)
    assert np.all(cv_array >= 0.0) and np.all(cv_array <= 1.0)


def test_collect_sweep_dataset_actually_sets_cv_on_backend():
    backend = FakeCVBackend(channel_names=["a", "b", "c"])
    rng = np.random.default_rng(seed=1)
    collect_sweep_dataset(
        backend, n_samples=5, settle_time_s=0.0, sample_rate=48000, rng=rng,
        aggregate_window_s=0.1,
    )
    # 3 channels x 5 samples = 15 set_cv calls
    assert len(backend.set_cv_calls) == 15


def test_collect_sweep_dataset_reads_multiple_blocks_per_sample():
    backend = CountingFakeBackend(channel_names=["a", "b", "c"])
    rng = np.random.default_rng(seed=7)
    n_samples = 5
    collect_sweep_dataset(
        backend, n_samples=n_samples, settle_time_s=0.0, sample_rate=48000, rng=rng,
        aggregate_window_s=0.1,
    )
    assert backend.read_audio_block_calls > n_samples
