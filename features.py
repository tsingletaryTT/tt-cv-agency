import numpy as np

LOUDNESS_REF_RMS = 0.4
BRIGHTNESS_REF_HZ = 12000.0
PITCH_LOG_MIN_HZ = 20.0
PITCH_LOG_MAX_HZ = 4000.0


def rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))


def spectral_centroid(block: np.ndarray, sample_rate: int) -> float:
    windowed = block.astype(np.float64) * np.hanning(len(block))
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
    lag_min = int(sample_rate / fmax)
    lag_max = min(int(sample_rate / fmin), len(corr) - 1)
    if lag_max <= lag_min:
        return 0.0
    search = corr[lag_min:lag_max]
    if len(search) == 0 or search.max() <= 0:
        return 0.0
    best_lag = lag_min + int(np.argmax(search))
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
