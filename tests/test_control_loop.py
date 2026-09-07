import numpy as np
from backends.base import FakeCVBackend
from control_loop import run_control_loop


class LinearFakeBackend(FakeCVBackend):
    """A fake whose audio block's features are a direct, known function of
    its current CV, so the control loop's convergence can be checked
    end-to-end without a real instrument or a chip."""

    def read_audio_block(self) -> np.ndarray:
        # Encode current CV directly as constant-amplitude/frequency content
        # sized so extract_features(...) recovers roughly the CV vector itself.
        level = self.last_known_cv("vca_level")
        t = np.arange(4096) / 48000.0
        return (level * 0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def fake_predict(target_features: np.ndarray) -> np.ndarray:
    # Correctly inverts LinearFakeBackend's known feature function so a
    # converged CV genuinely reproduces the target loudness feature:
    # feature = rms/REF_RMS = (level * 0.5 / sqrt(2)) / 0.4 = level * scale.
    scale = 0.5 / (0.4 * np.sqrt(2))
    target_level = np.clip(target_features[0] / scale, 0.0, 1.0)
    return np.array([0.5, 0.5, target_level])


def test_control_loop_converges_toward_loudness_goal():
    backend = LinearFakeBackend(channel_names=["vco_freq", "vco_fm", "vca_level"])
    # 6-dim goal: [mean_loudness, std_loudness, mean_brightness, std_brightness,
    # mean_pitch, std_pitch]. Only index 0 (mean loudness) is exercised by
    # fake_predict / this assertion -- the rest are placeholders, same as the
    # pre-aggregation test's "others don't matter here" 3-dim goal.
    goal = np.array([0.8, 0.0, 0.5, 0.0, 0.5, 0.0])

    history = run_control_loop(
        backend,
        predict_fn=fake_predict,
        goal_features=goal,
        sample_rate=48000,
        step_fraction=0.5,
        control_interval_s=0.0,
        max_iterations=20,
        convergence_threshold=0.05,
        # Small window: LinearFakeBackend's CV doesn't change mid-window, so
        # a small window already demonstrates the windowed read without
        # paying for hundreds of autocorrelation passes per iteration.
        aggregate_window_s=0.1,
    )

    assert len(history) > 1
    first_error = abs(history[0][0] - goal[0])
    last_error = abs(history[-1][0] - goal[0])
    assert last_error < first_error
    assert last_error < 0.05


def test_control_loop_records_one_feature_vector_per_iteration():
    backend = LinearFakeBackend(channel_names=["vco_freq", "vco_fm", "vca_level"])
    goal = np.array([0.5, 0.0, 0.5, 0.0, 0.5, 0.0])
    history = run_control_loop(
        backend, predict_fn=fake_predict, goal_features=goal, sample_rate=48000,
        control_interval_s=0.0, max_iterations=5, convergence_threshold=-1.0,  # never converge, run all 5
        aggregate_window_s=0.1,
    )
    assert len(history) == 5


def test_control_loop_history_entries_are_six_dimensional():
    backend = LinearFakeBackend(channel_names=["vco_freq", "vco_fm", "vca_level"])
    goal = np.array([0.5, 0.0, 0.5, 0.0, 0.5, 0.0])
    history = run_control_loop(
        backend, predict_fn=fake_predict, goal_features=goal, sample_rate=48000,
        control_interval_s=0.0, max_iterations=3, convergence_threshold=-1.0,
        aggregate_window_s=0.1,
    )
    assert len(history) == 3
    for entry in history:
        assert entry.shape == (6,)
