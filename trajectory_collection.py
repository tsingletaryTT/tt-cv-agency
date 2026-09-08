# trajectory_collection.py
import time
import numpy as np

from backends.base import CVBackend
from features import read_aggregated_window


def collect_trajectories(
    backend: CVBackend,
    n_episodes: int,
    episode_length: int,
    max_action: float,
    settle_time_s: float,
    rng: np.random.Generator,
    sample_rate: int | None = None,
    aggregate_window_s: float = 3.0,
    seed_cv_vectors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sample_rate is None:
        sample_rate = backend.sample_rate()
    channels = backend.channels()
    n_channels = len(channels)
    n_transitions = n_episodes * episode_length

    states = np.zeros((n_transitions, 6))
    actions = np.zeros((n_transitions, n_channels))
    next_states = np.zeros((n_transitions, 6))

    has_seeds = seed_cv_vectors is not None and len(seed_cv_vectors) > 0
    transition_idx = 0

    for episode in range(n_episodes):
        if has_seeds:
            start_cv = np.asarray(seed_cv_vectors[episode % len(seed_cv_vectors)], dtype=np.float64)
        else:
            start_cv = rng.uniform(0.0, 1.0, size=n_channels)

        current_cv = start_cv
        for ch, value in zip(channels, current_cv):
            backend.set_cv(ch, float(value))
        time.sleep(settle_time_s)
        current_state = read_aggregated_window(backend, sample_rate, aggregate_window_s)

        for _ in range(episode_length):
            raw_delta = rng.uniform(-max_action, max_action, size=n_channels)
            next_cv = np.clip(current_cv + raw_delta, 0.0, 1.0)
            applied_action = next_cv - current_cv

            for ch, value in zip(channels, next_cv):
                backend.set_cv(ch, float(value))
            time.sleep(settle_time_s)
            next_state = read_aggregated_window(backend, sample_rate, aggregate_window_s)

            states[transition_idx] = current_state
            actions[transition_idx] = applied_action
            next_states[transition_idx] = next_state
            transition_idx += 1

            current_cv = next_cv
            current_state = next_state

    return states, actions, next_states


if __name__ == "__main__":
    from backends.vcv_rack import VCVRackBackend

    # configs/sequencer_test.yaml -- the current instrument, 8 CV channels.
    backend = VCVRackBackend("configs/sequencer_test.yaml")
    try:
        rng = np.random.default_rng(seed=0)
        # Seed episode starts from Stage 2's discovered novelty archive when
        # it exists, so trajectory data is denser in the regions Stage 2
        # already found genuinely different from each other -- falls back
        # to uniform-random starts if the file isn't there.
        seed_cv_vectors = None
        try:
            archive = np.load("data/sequencer_novelty_archive.npz")
            seed_cv_vectors = archive["cv"]
        except FileNotFoundError:
            pass

        # 60 episodes x 10 steps -- one episode per Stage 2 archive member
        # (60) when the archive is present, 10 steps/episode for a short
        # but real trajectory. At settle_time_s=0.5 + aggregate_window_s=3.0
        # per read (~3.5s), total runtime is roughly
        # n_episodes * (1 + episode_length) * 3.5s =~ 38-39 minutes.
        states, actions, next_states = collect_trajectories(
            backend, n_episodes=60, episode_length=10, max_action=0.15,
            settle_time_s=0.5, rng=rng, aggregate_window_s=3.0,
            seed_cv_vectors=seed_cv_vectors,
        )
        import os
        os.makedirs("data", exist_ok=True)
        np.savez(
            "data/sequencer_trajectory_dataset.npz",
            states=states, actions=actions, next_states=next_states,
            channels=backend.channels(),
        )
        print(f"saved {len(states)} transitions to data/sequencer_trajectory_dataset.npz")
    finally:
        backend.close()
