import math
import numpy as np

from backends.base import CVBackend

LOUDNESS_REF_RMS = 0.4
# Recalibrated against this patch's real achievable spectral centroid range: a
# 150-sample random-CV recon pass measured raw (unnormalized) centroid values
# up to ~6150 Hz (median ~365 Hz), so the prior 12000.0 value left brightness
# permanently compressed into roughly the bottom half of [0, 1]. 7000.0 gives
# ~14% headroom above the observed max, matching how LOUDNESS_REF_RMS=0.4
# sits a similar margin above its own observed max RMS (~0.365).
BRIGHTNESS_REF_HZ = 7000.0
PITCH_LOG_MIN_HZ = 20.0
PITCH_LOG_MAX_HZ = 4000.0


def rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))


def spectral_centroid(block: np.ndarray, sample_rate: int) -> float:
    block = block.astype(np.float64)
    block = block - block.mean()  # remove DC, same as estimate_pitch does
    windowed = block * np.hanning(len(block))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(block), d=1.0 / sample_rate)
    total = spectrum.sum()
    if total <= 0:
        return 0.0
    return float(np.sum(freqs * spectrum) / total)


def estimate_pitch(block: np.ndarray, sample_rate: int, fmin: float = 20.0, fmax: float = 4000.0) -> float:
    x = block.astype(np.float64)
    x = x - x.mean()
    if np.allclose(x, 0.0):
        return 0.0
    corr = np.correlate(x, x, mode="full")
    corr = corr[len(corr) // 2:]  # keep non-negative lags
    lag_min_floor = int(sample_rate / fmax)
    lag_max = min(int(sample_rate / fmin), len(corr) - 1)
    if lag_max <= lag_min_floor:
        return 0.0

    # The raw (unnormalized) autocorrelation always starts at its global
    # maximum at lag 0 and descends through a "main lobe" before the first
    # period's peak appears. That descent is many lags wide relative to the
    # true period whenever the fundamental is low (or the block is short
    # relative to the period) -- e.g. at sample_rate=48000, fmax=4000
    # (lag_min_floor=12) and the real production block_size of 1024
    # samples, the main lobe is still descending well past lag 12 for any
    # fundamental below ~440 Hz. Searching for the peak starting at
    # lag_min_floor in that regime just finds a point partway down the
    # main lobe's descending slope, not the true periodicity peak -- which
    # is why the old implementation returned exactly fmax as a constant for
    # any low/mid fundamental at production block size.
    #
    # Fix: find where the main lobe actually finishes descending (the first
    # local minimum after lag 0 -- the first lag where the autocorrelation
    # stops decreasing), and only start the peak search there. lag_min_floor
    # (derived from fmax) still acts as a floor/sanity bound on top of that,
    # so the search never starts earlier than the fmax-implied minimum lag.
    main_lobe_end = 1
    while main_lobe_end < len(corr) - 1 and corr[main_lobe_end] < corr[main_lobe_end - 1]:
        main_lobe_end += 1
    search_start = max(lag_min_floor, main_lobe_end)
    if search_start >= lag_max:
        # The main lobe never finished descending before lag_max (e.g. a
        # fundamental so low its period doesn't fit the block) -- fall back
        # to the floor-based window rather than searching an empty range.
        search_start = lag_min_floor

    search = corr[search_start:lag_max]
    if len(search) == 0 or search.max() <= 0:
        return 0.0
    best_lag = search_start + int(np.argmax(search))
    return float(sample_rate / best_lag)


def extract_features(block: np.ndarray, sample_rate: int) -> np.ndarray:
    loudness = min(rms(block) / LOUDNESS_REF_RMS, 1.0)
    brightness = min(spectral_centroid(block, sample_rate) / BRIGHTNESS_REF_HZ, 1.0)
    pitch_hz = estimate_pitch(block, sample_rate, fmin=PITCH_LOG_MIN_HZ, fmax=PITCH_LOG_MAX_HZ)
    if pitch_hz <= 0:
        pitch_norm = 0.0
    else:
        pitch_hz = max(PITCH_LOG_MIN_HZ, min(pitch_hz, PITCH_LOG_MAX_HZ))
        pitch_norm = (np.log2(pitch_hz) - np.log2(PITCH_LOG_MIN_HZ)) / (
            np.log2(PITCH_LOG_MAX_HZ) - np.log2(PITCH_LOG_MIN_HZ)
        )
    return np.array([loudness, brightness, pitch_norm], dtype=np.float64)


def extract_features_aggregated(blocks: list[np.ndarray], sample_rate: int) -> np.ndarray:
    """Reduce a whole window of audio blocks to a fixed 6-dim feature vector,
    ordered exactly:

        [mean(loudness), std(loudness), mean(brightness), std(brightness),
         mean(pitch_norm), std(pitch_norm)]

    i.e. each of `extract_features`'s three per-block dimensions (loudness,
    brightness, pitch_norm, in that order) contributes a mean and a std,
    interleaved as [mean0, std0, mean1, std1, mean2, std2] rather than
    [mean0, mean1, mean2, std0, std1, std2]. `control_loop.py`,
    `data_collection.py`, and multiple tests all depend on this exact
    ordering -- changing it is a breaking change for every `goal_features`/
    `feature_array` consumer, not just this function's own return value.
    """
    per_block = np.array([extract_features(block, sample_rate) for block in blocks])
    means = per_block.mean(axis=0)
    stds = per_block.std(axis=0)
    return np.array(
        [means[0], stds[0], means[1], stds[1], means[2], stds[2]], dtype=np.float64
    )


def read_aggregated_window(backend: CVBackend, sample_rate: int, aggregate_window_s: float) -> np.ndarray:
    """Read a window of audio blocks spanning roughly aggregate_window_s
    seconds from `backend` and reduce it to the 6-dim aggregated feature
    vector via extract_features_aggregated. A single instantaneous block
    can't represent a looping/evolving pattern (a sequencer, an LFO sweep)
    -- this is the shared read used by both data_collection.py's sweep
    collection and control_loop.py's per-iteration read, so the two never
    drift out of sync on how a window is sized."""
    blocks_per_window = max(1, math.ceil(aggregate_window_s * sample_rate / backend.block_size()))
    blocks = [backend.read_audio_block() for _ in range(blocks_per_window)]
    return extract_features_aggregated(blocks, sample_rate)
