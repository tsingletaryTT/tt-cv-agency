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


class TimeVaryingFakeBackend(FakeCVBackend):
    """Unlike every other fake backend in this test file (and
    CountingFakeBackend above), which returns the exact same constant block
    on every read_audio_block() call, this one's output genuinely differs
    from call to call -- alternating between two distinct waveforms (both a
    different amplitude AND a different frequency, so loudness, brightness,
    and pitch all vary, not just one of the three).

    This matters because every existing test of collect_sweep_dataset uses
    a constant-block backend, which means all 3 "std" dimensions of
    extract_features_aggregated's 6-dim output are trivially 0.0 in every
    one of them -- a bug that collapsed the windowed block-reading loop
    (e.g. reading the same block N times instead of N genuinely distinct
    reads) would leave the whole existing test suite green. Only a
    time-varying backend like this one can catch that."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._call_count = 0

    def read_audio_block(self) -> np.ndarray:
        self._call_count += 1
        t = np.arange(2048) / 48000.0
        if self._call_count % 2 == 0:
            return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        else:
            return (0.1 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)


def test_collect_sweep_dataset_std_columns_nonzero_with_time_varying_audio():
    backend = TimeVaryingFakeBackend(channel_names=["a", "b", "c"])
    rng = np.random.default_rng(seed=3)
    # aggregate_window_s=0.5 -> tens of blocks per window at the fake's
    # default block_size(), so both waveforms in the alternating pattern
    # above are actually captured within each sample's window.
    cv_array, feature_array = collect_sweep_dataset(
        backend, n_samples=4, settle_time_s=0.0, sample_rate=48000, rng=rng,
        aggregate_window_s=0.5,
    )
    # Columns 1, 3, 5 are std(loudness), std(brightness), std(pitch_norm)
    # per extract_features_aggregated's documented ordering -- a collapsed
    # block-reading loop (reading the same block repeatedly instead of
    # genuinely distinct ones) would leave these near zero. Threshold is
    # 1e-6, not 0.0: identical float64 rows still produce a std on the
    # order of 1e-16-1e-18 from floating-point rounding, which satisfies
    # "> 0.0" even when the loop is genuinely collapsed -- confirmed by
    # reproducing the collapse bug directly against this fake backend
    # during this project's own review of this test. 1e-6 is far above
    # that noise floor and far below the real variation this fixture
    # produces (order 0.01-0.1+, given a 3x amplitude / 4x frequency
    # swing between the two alternating waveforms).
    assert np.all(feature_array[:, [1, 3, 5]] > 1e-6)
