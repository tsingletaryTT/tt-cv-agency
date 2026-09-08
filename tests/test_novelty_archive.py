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
    # score=0.5 (not 0.0): this test is only checking that capacity fill
    # happens unconditionally while below max_size and there's no
    # threshold set -- it isn't about whether zero-novelty scores are
    # allowed (add() unconditionally rejects score <= 0.0 as a degenerate
    # duplicate, tested separately below), so a nonzero placeholder score
    # keeps this test's original intent without contradicting that rule.
    added = archive.add(np.array([0.0]), np.array([0.0, 0.0]), score=0.5)
    assert added is True
    assert len(archive) == 1


def test_add_rejects_zero_novelty_duplicate_even_with_room():
    archive = NoveltyArchive(max_size=5, k_neighbors=1, novelty_threshold=None)
    first_features = np.array([1.0, 2.0])
    archive.add(np.array([0.1]), first_features, score=float("inf"))
    assert len(archive) == 1

    # An identical feature vector genuinely scores 0.0 novelty (its nearest
    # neighbor distance is exactly 0) -- add() must reject this even though
    # the archive (max_size=5) has plenty of room.
    duplicate_features = np.array([1.0, 2.0])
    duplicate_score = archive.novelty(duplicate_features)
    assert duplicate_score == pytest.approx(0.0)

    added = archive.add(np.array([0.2]), duplicate_features, score=duplicate_score)
    assert added is False
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


def test_final_novelty_scores_diverges_from_stale_to_arrays_scores_after_eviction():
    # Same M1/M2/M3-then-evict setup as
    # test_add_evicts_least_novel_member_when_full: M1=[0,0], M2=[1,0],
    # M3=[100,0], all inserted with score=inf (so to_arrays()'s stored
    # scores for the two survivors stay frozen at inf even though the
    # archive's composition -- and thus their real leave-one-out novelty --
    # changes once a 4th candidate replaces M1).
    archive = NoveltyArchive(max_size=3, k_neighbors=1, novelty_threshold=None)
    archive.add(np.array([1.0]), np.array([0.0, 0.0]), score=float("inf"))    # M1
    archive.add(np.array([2.0]), np.array([1.0, 0.0]), score=float("inf"))    # M2
    archive.add(np.array([3.0]), np.array([100.0, 0.0]), score=float("inf"))  # M3

    candidate_features = np.array([50.0, 0.0])
    candidate_score = archive.novelty(candidate_features)  # 49.0, evicts M1 (see above test)
    added = archive.add(np.array([4.0]), candidate_features, score=candidate_score)
    assert added is True

    _, feature_array, stale_scores = archive.to_arrays()
    final_scores = archive.final_novelty_scores()

    assert final_scores.shape == (len(archive),) == (3,)

    # Hand-verified (k=1, 1D feature values 1, 100, 50 after M1's eviction):
    # candidate (dist to M2=49, M3=50 -> nearest 49), M2 (dist to candidate
    # =49, M3=99 -> nearest 49), M3 (dist to candidate=50, M2=99 -> nearest
    # 50). Order follows to_arrays(): candidate replaced M1's slot (index
    # 0), M2 at index 1, M3 at index 2.
    for f, expected in zip(feature_array, [50.0, 1.0, 100.0]):
        assert f[0] == pytest.approx(expected)
    assert final_scores == pytest.approx([49.0, 49.0, 50.0])

    # The stored (stale) scores for M2 and M3 are still their insertion-time
    # value of +inf, which no longer reflects their real current novelty --
    # this is exactly the divergence final_novelty_scores() corrects.
    assert stale_scores[1] == float("inf")
    assert stale_scores[2] == float("inf")
    assert final_scores[1] != stale_scores[1]
    assert final_scores[2] != stale_scores[2]


def test_init_rejects_non_positive_max_size():
    with pytest.raises(ValueError):
        NoveltyArchive(max_size=0, k_neighbors=1, novelty_threshold=None)
    with pytest.raises(ValueError):
        NoveltyArchive(max_size=-1, k_neighbors=1, novelty_threshold=None)


def test_init_rejects_non_positive_k_neighbors():
    with pytest.raises(ValueError):
        NoveltyArchive(max_size=5, k_neighbors=0, novelty_threshold=None)
    with pytest.raises(ValueError):
        NoveltyArchive(max_size=5, k_neighbors=-1, novelty_threshold=None)
