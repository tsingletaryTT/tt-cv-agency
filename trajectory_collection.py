# trajectory_collection.py
import time
import numpy as np

from backends.base import CVBackend
from features import read_aggregated_window


def build_mixed_seed_cv_vectors(archive_cv: np.ndarray, seed: int = 1) -> np.ndarray:
    """Build episode-start seed vectors that mix Stage 2's diversity-archive
    members with fresh uniform-random CV vectors, addressing the Stage 3
    final-review finding that archive-only seeding clusters episode starts
    in extreme/near-silent corners and leaves the "typical" region most
    real goals ask for under-covered. The first `n_total - n_total // 2`
    entries of `archive_cv` are kept as-is (still get diversity-seeded
    episodes); the remaining `n_total // 2` slots are filled with freshly
    sampled uniform-random CV vectors instead, using `seed` (distinct from
    the main collection RNG) so the seeding draw is reproducible
    independent of per-step actions. Returns an array the same length as
    `archive_cv`, regardless of whether that length is even or odd, and
    handles a 1-member archive without crashing (n_uniform=0 in that case
    -- the single member is kept verbatim)."""
    n_total = len(archive_cv)
    n_uniform = n_total // 2
    n_archive = n_total - n_uniform
    rng = np.random.default_rng(seed)
    uniform_cv = rng.uniform(0.0, 1.0, size=(n_uniform, archive_cv.shape[1]))
    return np.concatenate([archive_cv[:n_archive], uniform_cv])


def collect_trajectories(
    backend: CVBackend,
    n_episodes: int,
    episode_length: int,
    max_action: float,
    settle_time_s: float,
    rng: np.random.Generator,
    sample_rate: int | None = None,
    aggregate_window_s: float = 5.0,
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
            # Log the post-clip delta (not the raw sampled draw) as the
            # action -- when raw_delta would push a channel past its [0,1]
            # CV boundary, the clip means what actually happened at that
            # boundary is smaller than what was intended. Training data must
            # reflect the real applied action, not the sampled-but-partially-
            # discarded one, or the model would learn a systematically
            # wrong action scale near the edges of CV range.
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
        # Mixed seeding (Stage 3 tune-up, 2026-09-27): half the episode
        # starts still come from Stage 2's novelty archive (diversity),
        # half are freshly uniform-random (coverage of the "typical"
        # region most real goals ask for -- see build_mixed_seed_cv_vectors'
        # docstring and the design spec's Decision 1 for why archive-only
        # seeding under-covers this region). Falls back to uniform-random
        # starts entirely if the archive file isn't there, same as before.
        seed_cv_vectors = None
        try:
            archive = np.load("data/sequencer_novelty_archive.npz")
            seed_cv_vectors = build_mixed_seed_cv_vectors(archive["cv"], seed=1)
        except FileNotFoundError:
            pass

        # 60 episodes x 10 steps -- unchanged episode/step count from the
        # original Stage 3 collection. aggregate_window_s=5.0 (up from
        # 3.0, aligning with control_loop.py's own window -- see the
        # design spec's Decision 2) raises the per-read cost from ~3.5s to
        # ~5.5s (settle_time_s=0.5 + aggregate_window_s), so total runtime
        # rises from ~38-39 minutes to roughly ~60 minutes.
        states, actions, next_states = collect_trajectories(
            backend, n_episodes=60, episode_length=10, max_action=0.15,
            settle_time_s=0.5, rng=rng, aggregate_window_s=5.0,
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
