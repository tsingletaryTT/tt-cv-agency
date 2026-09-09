import numpy as np
from backends.base import FakeCVBackend
from trajectory_control_loop import run_trajectory_control_loop


class LinearFakeBackend(FakeCVBackend):
    """Encodes current vca_level directly as the audio block's amplitude,
    mirroring tests/test_control_loop.py's own LinearFakeBackend exactly,
    so this loop's convergence is checkable end-to-end without a real
    instrument or a chip."""

    def read_audio_block(self) -> np.ndarray:
        level = self.last_known_cv("vca_level")
        t = np.arange(4096) / 48000.0
        return (level * 0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def additive_predict_fn(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    # A toy world model that's an EXACT match for LinearFakeBackend's real
    # loudness-vs-vca_level relationship: mean loudness = rms/LOUDNESS_REF_RMS
    # = (level * 0.5 / sqrt(2)) / 0.4, and level moves by exactly the applied
    # action (channel order below puts vca_level last), so
    # next_loudness = loudness + action[-1] * scale exactly, absent CV
    # clipping at the [0,1] boundary.
    scale = 0.5 / (0.4 * np.sqrt(2))
    next_states = states.copy()
    next_states[:, 0] = np.clip(states[:, 0] + actions[:, -1] * scale, 0.0, 1.0)
    return next_states


def test_trajectory_control_loop_converges_toward_loudness_goal():
    backend = LinearFakeBackend(channel_names=["vco_freq", "vco_fm", "vca_level"])
    goal = np.array([0.8, 0.0, 0.5, 0.0, 0.5, 0.0])

    history = run_trajectory_control_loop(
        backend, predict_fn=additive_predict_fn, goal_features=goal,
        horizon=2, n_candidates=100, n_elite=10, n_iterations=3,
        action_std_init=0.2, max_action=0.3,
        sample_rate=48000, control_interval_s=0.0, max_iterations=15,
        convergence_threshold=0.05, aggregate_window_s=0.1, seed=0,
    )

    assert len(history) > 1
    first_error = abs(history[0][0] - goal[0])
    last_error = abs(history[-1][0] - goal[0])
    assert last_error < first_error
    assert last_error < 0.1


def test_trajectory_control_loop_records_one_state_per_iteration():
    backend = LinearFakeBackend(channel_names=["vco_freq", "vco_fm", "vca_level"])
    goal = np.array([0.5, 0.0, 0.5, 0.0, 0.5, 0.0])
    history = run_trajectory_control_loop(
        backend, predict_fn=additive_predict_fn, goal_features=goal,
        horizon=1, n_candidates=50, n_elite=5, n_iterations=2,
        sample_rate=48000, control_interval_s=0.0, max_iterations=5,
        convergence_threshold=-1.0, aggregate_window_s=0.1, seed=0,
    )
    assert len(history) == 5
    for entry in history:
        assert entry.shape == (6,)
