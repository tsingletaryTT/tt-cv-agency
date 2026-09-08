import numpy as np
import pytest

from backends.base import FakeCVBackend
from explore import generate_candidate, run_search, run_search_and_save, save_archive
from novelty_archive import NoveltyArchive


class TimeVaryingFakeBackend(FakeCVBackend):
    """A FakeCVBackend whose read_audio_block() genuinely differs from call
    to call, alternating between two distinct waveforms (different
    amplitude AND frequency, so loudness, brightness, and pitch all vary).
    Modeled on tests/test_data_collection.py's TimeVaryingFakeBackend of
    the same name/purpose -- used here to exercise run_search's acceptance
    path against audio that actually changes, not just a fixed block."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._call_count = 0

    def read_audio_block(self) -> np.ndarray:
        self._call_count += 1
        t = np.arange(2048) / 48000.0
        if self._call_count % 2 == 0:
            return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        else:
            return (0.1 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)


class RepeatingFakeBackend(FakeCVBackend):
    """A FakeCVBackend that always returns the exact same fixed audio block
    on every read_audio_block() call -- every candidate's feature vector is
    therefore bit-identical, and only the very first one (scored +inf
    against an empty archive) should ever be accepted. This is actually
    identical to the base FakeCVBackend's default behavior (a fixed
    constant block); this subclass exists mainly to make that intent
    explicit at the call site."""


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


def test_save_archive_writes_final_novelty_scores_not_stale_insertion_scores(tmp_path):
    # Same M1/M2/M3-then-evict setup as
    # test_final_novelty_scores_diverges_from_stale_to_arrays_scores_after_eviction
    # in tests/test_novelty_archive.py: after the eviction, to_arrays()'s
    # stored scores for the two survivors are stale (frozen at their
    # insertion-time +inf), while final_novelty_scores() reflects their
    # real current standing. save_archive() must persist the latter.
    archive = NoveltyArchive(max_size=3, k_neighbors=1, novelty_threshold=None)
    archive.add(np.array([1.0]), np.array([0.0, 0.0]), score=float("inf"))    # M1
    archive.add(np.array([2.0]), np.array([1.0, 0.0]), score=float("inf"))    # M2
    archive.add(np.array([3.0]), np.array([100.0, 0.0]), score=float("inf"))  # M3
    candidate_features = np.array([50.0, 0.0])
    candidate_score = archive.novelty(candidate_features)
    archive.add(np.array([4.0]), candidate_features, score=candidate_score)   # evicts M1

    backend = FakeCVBackend(channel_names=["a"])
    output_path = str(tmp_path / "archive.npz")
    save_archive(archive, backend, output_path)

    expected_final_scores = archive.final_novelty_scores()
    loaded = np.load(output_path, allow_pickle=True)
    assert np.allclose(loaded["novelty_scores"], expected_final_scores)
    # And confirm those are NOT the stale to_arrays() scores (which still
    # include +inf for the two survivors that were never re-scored).
    assert not np.any(np.isinf(loaded["novelty_scores"]))


def test_run_search_rejects_repeated_identical_readings():
    # RepeatingFakeBackend's read_audio_block() always returns the exact
    # same fixed block -- identical to the base FakeCVBackend's default
    # behavior, used directly here since it's the simplest way to exercise
    # this. Every candidate after the first therefore has a feature vector
    # bit-identical to what's already archived, scoring exactly 0.0
    # novelty -- only the very first candidate (scored +inf against an
    # empty archive) should ever be accepted.
    backend = RepeatingFakeBackend(channel_names=["a", "b"])
    archive = NoveltyArchive(max_size=10, k_neighbors=1, novelty_threshold=None)
    rng = np.random.default_rng(0)
    run_search(
        backend, budget=5, archive=archive, rng=rng,
        mutation_prob=0.7, mutation_sigma=0.15,
        settle_time_s=0.0, aggregate_window_s=0.1,
    )
    assert len(archive) == 1


def test_run_search_accepts_new_members_when_audio_genuinely_varies():
    # Unlike every other run_search test here (all of which use a
    # constant-block backend and therefore only ever exercise the
    # rejection path after the first candidate), TimeVaryingFakeBackend's
    # audio genuinely differs from call to call -- so this test exercises
    # run_search's *acceptance* path for a second, genuinely-different
    # candidate too, not just its very first (+inf-scored) one.
    backend = TimeVaryingFakeBackend(channel_names=["a", "b"])
    archive = NoveltyArchive(max_size=10, k_neighbors=1, novelty_threshold=None)
    rng = np.random.default_rng(0)
    run_search(
        backend, budget=8, archive=archive, rng=rng,
        mutation_prob=0.7, mutation_sigma=0.15,
        settle_time_s=0.0, aggregate_window_s=0.1,
    )
    # Hand-confirmed (see this task's fix-round-1 report): with
    # aggregate_window_s=0.1 at this backend's block_size, each candidate's
    # window spans an odd number of blocks, so the two alternating
    # waveforms land at a different phase in successive windows, producing
    # exactly one additional genuinely-novel (and thus accepted) feature
    # vector before settling into a repeating 2-cycle that every later
    # candidate matches exactly (and is correctly rejected as non-novel).
    assert len(archive) == 2


class _RaisesAfterNCallsBackend(FakeCVBackend):
    """A FakeCVBackend whose read_audio_block() raises partway through a
    run -- simulating a hardware fault (e.g. the class of failure the
    Stage 2 capstone's mid-run PipeWire drift demonstrated is possible)
    partway through a budget-length run_search loop."""

    def __init__(self, *args, fail_after_calls: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._call_count = 0
        self._fail_after_calls = fail_after_calls

    def read_audio_block(self) -> np.ndarray:
        self._call_count += 1
        if self._call_count > self._fail_after_calls:
            raise RuntimeError("simulated hardware fault mid-run")
        return super().read_audio_block()


def test_run_search_and_save_persists_partial_archive_on_mid_run_failure(tmp_path):
    # blocks_per_window at aggregate_window_s=0.1 here is 5 (see
    # test_run_search_accepts_new_members_when_audio_genuinely_varies'
    # comment), so failing after 6 read_audio_block() calls lets exactly
    # the first candidate's window complete (5 calls) before the second
    # candidate's window fails partway through -- guaranteeing the archive
    # has accumulated exactly 1 member (the first, scored +inf) by the
    # time the failure hits.
    backend = _RaisesAfterNCallsBackend(channel_names=["a", "b"], fail_after_calls=6)
    archive = NoveltyArchive(max_size=10, k_neighbors=1, novelty_threshold=None)
    rng = np.random.default_rng(0)
    output_path = str(tmp_path / "archive.npz")

    with pytest.raises(RuntimeError, match="simulated hardware fault"):
        run_search_and_save(
            backend, budget=5, archive=archive, rng=rng,
            mutation_prob=0.7, mutation_sigma=0.15,
            settle_time_s=0.0, aggregate_window_s=0.1,
            output_path=output_path,
        )

    # The exception propagated (confirmed above), but whatever the archive
    # accumulated before the failure must still have been saved.
    assert len(archive) == 1
    loaded = np.load(output_path, allow_pickle=True)
    assert loaded["cv"].shape[0] == 1
    assert loaded["features"].shape[0] == 1
    assert loaded["novelty_scores"].shape[0] == 1
