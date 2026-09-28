from unittest.mock import patch

import numpy as np
from backends.base import FakeCVBackend
from features import LOUDNESS_REF_RMS
from trajectory_control_loop import run_trajectory_control_loop


class LinearFakeBackend(FakeCVBackend):
    """Encodes current vca_level directly as the audio block's amplitude --
    same idea as tests/test_control_loop.py's own LinearFakeBackend, but a
    smaller amplitude coefficient (0.3, not 0.5). At LOUDNESS_REF_RMS=0.175,
    coefficient 0.5 puts FakeCVBackend's default starting level (0.5) exactly
    at the loudness-clip ceiling (rms/REF > 1.0) -- harmless for
    run_control_loop's direct target-tracking (test_control_loop.py's test
    still passes), but fatal for CEM here: every candidate whose rollout
    doesn't escape the ceiling scores identically, so cem_plan's own search
    degenerates on a flat landscape at the very first iteration and doesn't
    reliably escape it within this test's candidate budget. 0.3 keeps the
    starting reading comfortably unclipped (loudness(0.5)=0.606) with a
    smooth, well-conditioned gradient from iteration 1, and the peak
    loudness(1.0)=1.212 (clipped) still leaves goal=0.8 reachable at
    level=0.660, nowhere near either boundary."""

    def read_audio_block(self) -> np.ndarray:
        level = self.last_known_cv("vca_level")
        t = np.arange(4096) / 48000.0
        return (level * 0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def additive_predict_fn(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    # A toy world model that's an EXACT match for LinearFakeBackend's real
    # loudness-vs-vca_level relationship: mean loudness = rms/LOUDNESS_REF_RMS
    # = (level * 0.3 / sqrt(2)) / LOUDNESS_REF_RMS, and level moves by exactly
    # the applied action (channel order below puts vca_level last), so
    # next_loudness = loudness + action[-1] * scale exactly, absent CV
    # clipping at the [0,1] boundary. Uses the real LOUDNESS_REF_RMS (not a
    # hardcoded copy of its value) so this fixture stays correct across any
    # future recalibration.
    scale = 0.3 / (LOUDNESS_REF_RMS * np.sqrt(2))
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


def test_trajectory_control_loop_does_not_truncate_history_when_converged_early():
    backend = LinearFakeBackend(channel_names=["vco_freq", "vco_fm", "vca_level"])
    goal = np.array([0.5, 0.0, 0.5, 0.0, 0.5, 0.0])
    history = run_trajectory_control_loop(
        backend, predict_fn=additive_predict_fn, goal_features=goal,
        horizon=1, n_candidates=10, n_elite=2, n_iterations=1,
        sample_rate=48000, control_interval_s=0.0, max_iterations=5,
        # convergence_threshold=10.0 is guaranteed satisfied on the very
        # first read (no real feature vector is ever 10.0 away from a
        # goal in [0,1]-normalized feature space) -- every iteration
        # takes the early-exit ("continue") branch without ever calling
        # cem_plan. This is exactly the path a continue-vs-break
        # regression would silently break: history would truncate to
        # length 1 instead of running the full max_iterations.
        convergence_threshold=10.0,
        aggregate_window_s=0.1, seed=0,
    )
    assert len(history) == 5


def test_trajectory_control_loop_applies_step_fraction_to_planned_action():
    backend = FakeCVBackend(channel_names=["a", "b"])
    goal = np.array([0.5, 0.0, 0.5, 0.0, 0.5, 0.0])

    def dummy_predict_fn(states, actions):
        return states  # never actually invoked once cem_plan itself is mocked

    with patch("trajectory_control_loop.cem_plan", return_value=np.array([0.2, -0.1])):
        run_trajectory_control_loop(
            backend, predict_fn=dummy_predict_fn, goal_features=goal,
            horizon=1, n_candidates=5, n_elite=1, n_iterations=1,
            sample_rate=48000, control_interval_s=0.0, max_iterations=1,
            convergence_threshold=-1.0, aggregate_window_s=0.1, seed=0,
            step_fraction=0.5,
        )

    calls = dict(backend.set_cv_calls)
    # FakeCVBackend starts every channel at 0.5 (see backends/base.py);
    # cem_plan is mocked to always return [0.2, -0.1] regardless of state,
    # so step_fraction=0.5 must halve it before it's added:
    # 0.5 + 0.5*0.2 = 0.6, 0.5 + 0.5*(-0.1) = 0.45.
    assert np.isclose(calls["a"], 0.6)
    assert np.isclose(calls["b"], 0.45)


def test_trajectory_control_loop_step_fraction_1_0_reproduces_undamped_behavior():
    # A migrating caller that explicitly passes step_fraction=1.0 must get
    # exactly today's pre-tune-up behavior: the full planned action applied
    # unscaled.
    backend = FakeCVBackend(channel_names=["a", "b"])
    goal = np.array([0.5, 0.0, 0.5, 0.0, 0.5, 0.0])

    def dummy_predict_fn(states, actions):
        return states

    with patch("trajectory_control_loop.cem_plan", return_value=np.array([0.2, -0.1])):
        run_trajectory_control_loop(
            backend, predict_fn=dummy_predict_fn, goal_features=goal,
            horizon=1, n_candidates=5, n_elite=1, n_iterations=1,
            sample_rate=48000, control_interval_s=0.0, max_iterations=1,
            convergence_threshold=-1.0, aggregate_window_s=0.1, seed=0,
            step_fraction=1.0,
        )

    calls = dict(backend.set_cv_calls)
    assert np.isclose(calls["a"], 0.7)   # 0.5 + 1.0*0.2
    assert np.isclose(calls["b"], 0.4)   # 0.5 + 1.0*(-0.1)
