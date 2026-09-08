import numpy as np
import pytest

from novelty_archive import NoveltyArchive


def test_novelty_of_empty_archive_is_infinite():
    archive = NoveltyArchive(max_size=5, k_neighbors=2, novelty_threshold=None)
    score = archive.novelty(np.array([0.0, 0.0]))
    assert score == float("inf")


def test_novelty_k1_uses_nearest_neighbor_distance():
    archive = NoveltyArchive(max_size=5, k_neighbors=1, novelty_threshold=None)
    archive.add(np.array([0.0]), np.array([0.0, 0.0]), score=float("inf"))
    archive.add(np.array([1.0]), np.array([10.0, 0.0]), score=float("inf"))
    score = archive.novelty(np.array([1.0, 0.0]))
    assert score == pytest.approx(1.0)


def test_novelty_k2_averages_two_nearest_neighbors():
    archive = NoveltyArchive(max_size=5, k_neighbors=2, novelty_threshold=None)
    archive.add(np.array([0.0]), np.array([0.0, 0.0]), score=float("inf"))
    archive.add(np.array([1.0]), np.array([10.0, 0.0]), score=float("inf"))
    archive.add(np.array([2.0]), np.array([20.0, 0.0]), score=float("inf"))
    score = archive.novelty(np.array([1.0, 0.0]))
    assert score == pytest.approx(5.0)


def test_novelty_clamps_k_to_archive_size():
    archive = NoveltyArchive(max_size=5, k_neighbors=10, novelty_threshold=None)
    archive.add(np.array([0.0]), np.array([0.0, 0.0]), score=float("inf"))
    archive.add(np.array([1.0]), np.array([4.0, 0.0]), score=float("inf"))
    score = archive.novelty(np.array([2.0, 0.0]))
    assert score == pytest.approx(2.0)


def test_add_below_capacity_always_succeeds_without_threshold():
    archive = NoveltyArchive(max_size=3, k_neighbors=1, novelty_threshold=None)
    added = archive.add(np.array([0.0]), np.array([0.0, 0.0]), score=0.0)
    assert added is True
    assert len(archive) == 1


def test_add_rejects_when_below_threshold_even_with_room():
    archive = NoveltyArchive(max_size=3, k_neighbors=1, novelty_threshold=5.0)
    added = archive.add(np.array([0.0]), np.array([0.0, 0.0]), score=2.0)
    assert added is False
    assert len(archive) == 0


def test_add_evicts_least_novel_member_when_full():
    archive = NoveltyArchive(max_size=3, k_neighbors=1, novelty_threshold=None)
    archive.add(np.array([1.0]), np.array([0.0, 0.0]), score=float("inf"))    # M1
    archive.add(np.array([2.0]), np.array([1.0, 0.0]), score=float("inf"))    # M2
    archive.add(np.array([3.0]), np.array([100.0, 0.0]), score=float("inf"))  # M3
    assert len(archive) == 3

    # Leave-one-out novelty (k=1): M1 vs {M2,M3} -> min(1,100)=1;
    # M2 vs {M1,M3} -> min(1,99)=1; M3 vs {M1,M2} -> min(100,99)=99.
    # M1 and M2 tie at 1.0 -- the tie-break is "evict the first tied index"
    # (np.argmin's default behavior), so M1 (inserted first) is evicted.
    candidate_features = np.array([50.0, 0.0])
    candidate_score = archive.novelty(candidate_features)
    assert candidate_score == pytest.approx(49.0)  # nearest existing member is M2 (dist 49)

    added = archive.add(np.array([4.0]), candidate_features, score=candidate_score)
    assert added is True
    assert len(archive) == 3
    _, feature_array, _ = archive.to_arrays()
    assert not any(np.allclose(f, [0.0, 0.0]) for f in feature_array)     # M1 evicted
    assert any(np.allclose(f, [1.0, 0.0]) for f in feature_array)         # M2 survives
    assert any(np.allclose(f, [100.0, 0.0]) for f in feature_array)      # M3 survives
    assert any(np.allclose(f, [50.0, 0.0]) for f in feature_array)       # candidate added


def test_add_rejects_when_less_novel_than_archive_minimum():
    archive = NoveltyArchive(max_size=3, k_neighbors=1, novelty_threshold=None)
    archive.add(np.array([1.0]), np.array([0.0, 0.0]), score=float("inf"))
    archive.add(np.array([2.0]), np.array([1.0, 0.0]), score=float("inf"))
    archive.add(np.array([3.0]), np.array([100.0, 0.0]), score=float("inf"))

    candidate_features = np.array([0.5, 0.0])
    candidate_score = archive.novelty(candidate_features)
    assert candidate_score == pytest.approx(0.5)

    added = archive.add(np.array([4.0]), candidate_features, score=candidate_score)
    assert added is False
    assert len(archive) == 3
    _, feature_array, _ = archive.to_arrays()
    assert not any(np.allclose(f, [0.5, 0.0]) for f in feature_array)


def test_sample_member_cv_returns_an_archived_cv_vector():
    archive = NoveltyArchive(max_size=3, k_neighbors=1, novelty_threshold=None)
    archive.add(np.array([7.0]), np.array([0.0, 0.0]), score=float("inf"))
    rng = np.random.default_rng(0)
    sampled = archive.sample_member_cv(rng)
    assert np.allclose(sampled, [7.0])


def test_sample_member_cv_raises_on_empty_archive():
    archive = NoveltyArchive(max_size=3, k_neighbors=1, novelty_threshold=None)
    rng = np.random.default_rng(0)
    with pytest.raises(Exception):
        archive.sample_member_cv(rng)


def test_to_arrays_stacks_current_members():
    archive = NoveltyArchive(max_size=3, k_neighbors=1, novelty_threshold=None)
    archive.add(np.array([1.0, 2.0]), np.array([0.1, 0.2]), score=1.5)
    archive.add(np.array([3.0, 4.0]), np.array([0.3, 0.4]), score=2.5)
    cv_array, feature_array, scores = archive.to_arrays()
    assert cv_array.shape == (2, 2)
    assert feature_array.shape == (2, 2)
    assert scores.shape == (2,)
    assert np.allclose(cv_array[0], [1.0, 2.0])
    assert np.allclose(scores, [1.5, 2.5])
