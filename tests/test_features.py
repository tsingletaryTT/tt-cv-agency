import numpy as np
import pytest
from features import rms, spectral_centroid, estimate_pitch, extract_features, extract_features_aggregated, read_aggregated_window


def test_rms_of_silence_is_zero():
    assert rms(np.zeros(512, dtype=np.float32)) == 0.0


def test_rms_of_known_sine_matches_expected_amplitude():
    # A sine of amplitude A has RMS = A / sqrt(2).
    t = np.arange(4096) / 48000.0
    sine = 0.8 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    result = rms(sine)
    assert abs(result - 0.8 / np.sqrt(2)) < 0.01


def test_spectral_centroid_of_low_sine_is_near_its_frequency():
    t = np.arange(4096) / 48000.0
    sine = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    centroid = spectral_centroid(sine, sample_rate=48000)
    assert abs(centroid - 440.0) < 50.0


def test_spectral_centroid_of_higher_sine_is_higher():
    t = np.arange(4096) / 48000.0
    low = np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    high = np.sin(2 * np.pi * 2000.0 * t).astype(np.float32)
    assert spectral_centroid(high, 48000) > spectral_centroid(low, 48000)


def test_estimate_pitch_recovers_known_frequency():
    t = np.arange(4096) / 48000.0
    sine = np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    pitch = estimate_pitch(sine, sample_rate=48000)
    assert abs(pitch - 220.0) < 5.0


@pytest.mark.parametrize("freq", [55.0, 110.0, 220.0, 330.0, 440.0])
def test_estimate_pitch_recovers_known_frequency_at_production_block_size(freq):
    # VCVRackBackend's real block_size is 1024 samples (not the 4096 used
    # above) -- at sample_rate=48000, fmax=4000 (the default), lag_min =
    # int(48000/4000) = 12. The old implementation searched for the
    # autocorrelation peak starting at that fixed lag_min, but the raw
    # autocorrelation's descending zero-lag main lobe is still dominant at
    # lag 12 for any fundamental below ~440 Hz at this block size -- so the
    # old code returned exactly fmax (4000 Hz) as a constant for all of
    # these frequencies except 440. This regression guard exercises the
    # real production block size directly so that bug can never hide behind
    # a longer, easier test block again.
    t = np.arange(1024) / 48000.0
    sine = np.sin(2 * np.pi * freq * t).astype(np.float32)
    pitch = estimate_pitch(sine, sample_rate=48000)
    # A short block relative to a low fundamental's period gives coarser
    # resolution (unnormalized autocorrelation is biased toward shorter
    # lags at large lag, where fewer samples overlap) -- allow a tolerance
    # that scales with frequency rather than the fixed 5 Hz used for the
    # long-block case above.
    tolerance = max(5.0, freq * 0.15)
    assert abs(pitch - freq) < tolerance


def test_extract_features_returns_length_3_array_in_unit_range():
    t = np.arange(4096) / 48000.0
    sine = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    feats = extract_features(sine, sample_rate=48000)
    assert feats.shape == (3,)
    assert np.all(feats >= 0.0) and np.all(feats <= 1.0)


def test_extract_features_of_silence_has_zero_loudness():
    feats = extract_features(np.zeros(4096, dtype=np.float32), sample_rate=48000)
    assert feats[0] == 0.0


def test_extract_features_aggregated_shape():
    rng = np.random.default_rng(0)
    blocks = [rng.uniform(-1, 1, size=1024) for _ in range(5)]
    result = extract_features_aggregated(blocks, sample_rate=48000)
    assert result.shape == (6,)


def test_extract_features_aggregated_detects_modulation():
    # A block-to-block AMPLITUDE-MODULATED signal (loudness genuinely
    # changing across blocks) must show higher loudness-std than a
    # constant-amplitude signal with the same mean loudness.
    sample_rate = 48000
    t = np.arange(1024) / sample_rate
    static_blocks = [0.5 * np.sin(2 * np.pi * 220 * t) for _ in range(8)]
    modulated_blocks = [
        (0.1 + 0.4 * (i % 2)) * np.sin(2 * np.pi * 220 * t) for i in range(8)
    ]
    static_result = extract_features_aggregated(static_blocks, sample_rate)
    modulated_result = extract_features_aggregated(modulated_blocks, sample_rate)
    loudness_std_index = 1
    assert modulated_result[loudness_std_index] > static_result[loudness_std_index] * 2


def test_read_aggregated_window_reads_multiple_blocks():
    from backends.base import FakeCVBackend

    backend = FakeCVBackend(channel_names=["a"], sample_rate=48000, block_size=1024)
    result = read_aggregated_window(backend, sample_rate=48000, aggregate_window_s=0.1)
    assert result.shape == (6,)
    # 0.1s * 48000 / 1024 = 4.6875 -> ceil to 5 blocks
    assert len(backend.set_cv_calls) == 0  # this helper only reads, never writes CV
