import numpy as np
from backends.base import FakeCVBackend
from trajectory_collection import collect_trajectories


def test_collect_trajectories_shapes():
    backend = FakeCVBackend(channel_names=["a", "b", "c"])
    rng = np.random.default_rng(0)
    states, actions, next_states = collect_trajectories(
        backend, n_episodes=3, episode_length=4, max_action=0.1,
        settle_time_s=0.0, rng=rng, sample_rate=48000, aggregate_window_s=0.05,
    )
    assert states.shape == (12, 6)
    assert actions.shape == (12, 3)
    assert next_states.shape == (12, 6)


def test_collect_trajectories_logs_action_matching_actual_backend_cv_delta():
    # Regardless of what the RNG draws (including draws that get clipped
    # at the [0,1] boundary), the logged action for each transition must
    # equal the CV delta actually applied to the backend -- not the raw,
    # possibly out-of-bounds sampled delta. Verified by reconstructing the
    # actually-applied CV vectors directly from FakeCVBackend's own
    # set_cv_calls log (ground truth of what really happened), independent
    # of collect_trajectories' own internal bookkeeping.
    backend = FakeCVBackend(channel_names=["a", "b"])
    # A large max_action (2.0, well beyond the [0,1] range) and a starting
    # point at a boundary (via seed_cv_vectors) makes clipping certain to
    # occur on most/all steps, regardless of RNG seed.
    seed_cv_vectors = np.array([[1.0, 0.0]])
    rng = np.random.default_rng(3)
    n_episodes, episode_length = 1, 5
    states, actions, next_states = collect_trajectories(
        backend, n_episodes=n_episodes, episode_length=episode_length, max_action=2.0,
        settle_time_s=0.0, rng=rng, sample_rate=48000, aggregate_window_s=0.05,
        seed_cv_vectors=seed_cv_vectors,
    )

    # Reconstruct the real applied CV vectors per step, per channel, from
    # the backend's own call log: the first 2 calls are the episode's
    # starting CV (channels "a","b"), then each subsequent pair of calls
    # is one step's post-clip next_cv.
    calls = backend.set_cv_calls
    channel_order = ["a", "b"]
    cv_sequence = []
    for i in range(0, len(calls), 2):
        pair = dict(calls[i:i + 2])
        cv_sequence.append(np.array([pair[ch] for ch in channel_order]))

    assert len(cv_sequence) == 1 + episode_length  # start + one per step
    for step in range(episode_length):
        expected_action = cv_sequence[step + 1] - cv_sequence[step]
        assert np.allclose(actions[step], expected_action)
        # And every applied CV must be within bounds -- proof the clip
        # actually happened where needed.
        assert np.all(cv_sequence[step + 1] >= 0.0) and np.all(cv_sequence[step + 1] <= 1.0)


def test_collect_trajectories_cycles_through_seed_cv_vectors_round_robin():
    backend = FakeCVBackend(channel_names=["a"])
    seed_cv_vectors = np.array([[0.1], [0.9]])
    rng = np.random.default_rng(0)
    n_episodes, episode_length = 5, 1
    collect_trajectories(
        backend, n_episodes=n_episodes, episode_length=episode_length, max_action=0.0,
        settle_time_s=0.0, rng=rng, sample_rate=48000, aggregate_window_s=0.05,
        seed_cv_vectors=seed_cv_vectors,
    )
    # max_action=0.0 -> every sampled delta is exactly 0.0 (a degenerate
    # Uniform(0,0) draw), so each episode's starting CV (the first set_cv
    # call of that episode) is directly observable and must cycle
    # 0.1, 0.9, 0.1, 0.9, 0.1. Single channel "a" -> 2 calls per episode
    # (1 start + 1 step), so starts are at even indices.
    calls = backend.set_cv_calls
    starts = [calls[i][1] for i in range(0, len(calls), 2)]
    assert np.allclose(starts, [0.1, 0.9, 0.1, 0.9, 0.1])


def test_collect_trajectories_uses_uniform_random_starts_without_seed_vectors():
    backend = FakeCVBackend(channel_names=["a", "b"])
    rng = np.random.default_rng(5)
    states, actions, next_states = collect_trajectories(
        backend, n_episodes=2, episode_length=1, max_action=0.1,
        settle_time_s=0.0, rng=rng, sample_rate=48000, aggregate_window_s=0.05,
        seed_cv_vectors=None,
    )
    # No crash, correct shapes, and CV values used for starts stayed in
    # [0, 1] (uniform-random fallback, not seed-vector-driven).
    calls = backend.set_cv_calls
    assert all(0.0 <= value <= 1.0 for _, value in calls)
    assert states.shape == (2, 6)
