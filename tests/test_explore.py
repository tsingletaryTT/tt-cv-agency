import numpy as np
import pytest

from backends.base import FakeCVBackend
from explore import generate_candidate, run_search, save_archive
from novelty_archive import NoveltyArchive


def test_generate_candidate_uses_uniform_random_when_archive_empty():
    archive = NoveltyArchive(max_size=5, k_neighbors=1, novelty_threshold=None)
    rng = np.random.default_rng(0)
    candidate = generate_candidate(["a", "b"], archive, rng, mutation_prob=1.0, mutation_sigma=0.1)
    assert set(candidate.keys()) == {"a", "b"}
    assert all(0.0 <= v <= 1.0 for v in candidate.values())


def test_generate_candidate_is_deterministic_under_seed():
    archive = NoveltyArchive(max_size=5, k_neighbors=1, novelty_threshold=None)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    c1 = generate_candidate(["a", "b"], archive, rng1, mutation_prob=1.0, mutation_sigma=0.1)
    c2 = generate_candidate(["a", "b"], archive, rng2, mutation_prob=1.0, mutation_sigma=0.1)
    assert c1 == c2


def test_generate_candidate_mutation_is_clipped_to_unit_range():
    archive = NoveltyArchive(max_size=5, k_neighbors=1, novelty_threshold=None)
    archive.add(np.array([1.0, 1.0]), np.array([0.0]), score=float("inf"))
    rng = np.random.default_rng(0)
    saw_extreme = False
    for _ in range(50):
        candidate = generate_candidate(["a", "b"], archive, rng, mutation_prob=1.0, mutation_sigma=1.0)
        assert all(0.0 <= v <= 1.0 for v in candidate.values())
        if any(v in (0.0, 1.0) for v in candidate.values()):
            saw_extreme = True
    assert saw_extreme


def test_generate_candidate_falls_back_to_random_when_archive_empty_even_with_mutation_prob_1():
    archive = NoveltyArchive(max_size=5, k_neighbors=1, novelty_threshold=None)
    rng = np.random.default_rng(0)
    # mutation_prob=1.0 would always try to mutate if the archive had
    # members; with an empty archive there is nothing to mutate, so this
    # must not raise and must fall back to the random path.
    candidate = generate_candidate(["a"], archive, rng, mutation_prob=1.0, mutation_sigma=0.1)
    assert 0.0 <= candidate["a"] <= 1.0


def test_run_search_evaluates_exactly_budget_candidates_and_sets_all_channels():
    backend = FakeCVBackend(
        channel_names=["a", "b"],
        audio_block=(0.3 * np.sin(2 * np.pi * 440 * np.arange(2048) / 48000)).astype(np.float32),
    )
    archive = NoveltyArchive(max_size=10, k_neighbors=1, novelty_threshold=None)
    rng = np.random.default_rng(0)
    run_search(
        backend, budget=5, archive=archive, rng=rng,
        mutation_prob=0.7, mutation_sigma=0.15,
        settle_time_s=0.0, aggregate_window_s=0.1,
    )
    assert {call[0] for call in backend.set_cv_calls} == {"a", "b"}
    assert len(backend.set_cv_calls) == 10  # 5 candidates x 2 channels


def test_save_archive_writes_channels_matching_backend(tmp_path):
    backend = FakeCVBackend(channel_names=["a", "b"])
    archive = NoveltyArchive(max_size=5, k_neighbors=1, novelty_threshold=None)
    archive.add(np.array([0.1, 0.2]), np.array([0.0] * 6), score=float("inf"))
    output_path = str(tmp_path / "archive.npz")
    save_archive(archive, backend, output_path)
    loaded = np.load(output_path, allow_pickle=True)
    assert list(loaded["channels"]) == ["a", "b"]
    assert loaded["cv"].shape == (1, 2)
    assert loaded["features"].shape == (1, 6)
    assert loaded["novelty_scores"].shape == (1,)


def test_save_archive_creates_output_directory(tmp_path):
    backend = FakeCVBackend(channel_names=["a"])
    archive = NoveltyArchive(max_size=5, k_neighbors=1, novelty_threshold=None)
    archive.add(np.array([0.5]), np.array([0.0]), score=float("inf"))
    output_path = str(tmp_path / "nested" / "dir" / "archive.npz")
    save_archive(archive, backend, output_path)
    assert (tmp_path / "nested" / "dir" / "archive.npz").exists()
