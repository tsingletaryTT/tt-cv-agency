# Stage 2: exploration / novelty search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a bounded-archive novelty-search agent that sweeps the running
VCV Rack instrument's CV space with no goal, keeping a diverse library of
meaningfully-different-sounding settings.

**Architecture:** Two new flat files at the repo root (matching this
project's existing convention — no new package): `novelty_archive.py`
(the `NoveltyArchive` class: k-NN novelty scoring, capacity + eviction)
and `explore.py` (candidate generation + the CLI search loop, reusing
`VCVRackBackend` and `features.read_aggregated_window` unchanged).

**Tech Stack:** Python 3.12, numpy, pytest, `backends.base.FakeCVBackend`
for hardware-free tests, `backends.vcv_rack.VCVRackBackend` for the real
capstone run.

**Spec:** `docs/superpowers/specs/2026-09-08-stage2-exploration-design.md`

## Global Constraints

- No new `ttnn`/hardware dependency and no `gozer` lease anywhere in this
  plan except the final full-suite hardware-marked regression check
  (which is inherited, pre-existing, not new to this stage).
- CV values stay `[0, 1]` at every interface boundary — `generate_candidate`'s
  mutation path must clip.
- Channel identity is never hardcoded — `NoveltyArchive` and
  `generate_candidate` operate on whatever `channels: list[str]` the
  backend reports.
- No edits to `features.py`, `backends/`, or any `configs/*.yaml` — this
  plan only adds `novelty_archive.py` and `explore.py`.
- Work happens directly on `main`, no worktree — matching this project's
  established precedent from Stages 0 and 1 (recorded explicitly rather
  than re-asked each time).

---

### Task 1: `NoveltyArchive`

**Files:**
- Create: `novelty_archive.py`
- Test: `tests/test_novelty_archive.py`

**Interfaces:**
- Consumes: `numpy` only.
- Produces: `NoveltyArchive(max_size: int, k_neighbors: int, novelty_threshold: float | None)`
  with methods `novelty(feature_vector) -> float`, `add(cv_vector, feature_vector, score) -> bool`,
  `sample_member_cv(rng) -> np.ndarray`, `__len__() -> int`,
  `to_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]` (cv_array, feature_array, scores).
  Task 2 imports `NoveltyArchive` from this module.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_novelty_archive.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_novelty_archive.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'novelty_archive'`

- [ ] **Step 3: Write the implementation**

Create `novelty_archive.py`:

```python
# novelty_archive.py
import numpy as np


class NoveltyArchive:
    """Bounded archive of (cv_vector, feature_vector) pairs for novelty
    search: a candidate is kept only if its feature vector is novel enough
    relative to what's already archived. See
    docs/superpowers/specs/2026-09-08-stage2-exploration-design.md."""

    def __init__(self, max_size: int, k_neighbors: int, novelty_threshold: float | None):
        self._max_size = max_size
        self._k_neighbors = k_neighbors
        self._novelty_threshold = novelty_threshold
        self._cv_vectors: list[np.ndarray] = []
        self._feature_vectors: list[np.ndarray] = []
        self._scores: list[float] = []

    def __len__(self) -> int:
        return len(self._feature_vectors)

    def _distances_to(self, feature_vector: np.ndarray, exclude_index: int | None = None) -> np.ndarray:
        others = [f for i, f in enumerate(self._feature_vectors) if i != exclude_index]
        if not others:
            return np.array([])
        return np.linalg.norm(np.array(others) - feature_vector, axis=1)

    def novelty(self, feature_vector: np.ndarray) -> float:
        distances = self._distances_to(feature_vector)
        if distances.size == 0:
            return float("inf")
        k = min(self._k_neighbors, distances.size)
        nearest = np.partition(distances, k - 1)[:k]
        return float(nearest.mean())

    def _leave_one_out_novelty(self, index: int) -> float:
        distances = self._distances_to(self._feature_vectors[index], exclude_index=index)
        if distances.size == 0:
            return float("inf")
        k = min(self._k_neighbors, distances.size)
        nearest = np.partition(distances, k - 1)[:k]
        return float(nearest.mean())

    def add(self, cv_vector: np.ndarray, feature_vector: np.ndarray, score: float) -> bool:
        if self._novelty_threshold is not None and score <= self._novelty_threshold:
            return False

        if len(self) < self._max_size:
            self._cv_vectors.append(cv_vector)
            self._feature_vectors.append(feature_vector)
            self._scores.append(score)
            return True

        leave_one_out = np.array([self._leave_one_out_novelty(i) for i in range(len(self))])
        min_index = int(np.argmin(leave_one_out))
        if score > leave_one_out[min_index]:
            self._cv_vectors[min_index] = cv_vector
            self._feature_vectors[min_index] = feature_vector
            self._scores[min_index] = score
            return True
        return False

    def sample_member_cv(self, rng: np.random.Generator) -> np.ndarray:
        if not self._cv_vectors:
            raise ValueError("cannot sample from an empty archive")
        index = rng.integers(0, len(self._cv_vectors))
        return self._cv_vectors[index]

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.array(self._cv_vectors),
            np.array(self._feature_vectors),
            np.array(self._scores),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_novelty_archive.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add novelty_archive.py tests/test_novelty_archive.py
git commit -m "Add NoveltyArchive: bounded k-NN novelty search archive"
```

---

### Task 2: `explore.py` — candidate generation and the CLI search loop

**Files:**
- Create: `explore.py`
- Test: `tests/test_explore.py`

**Interfaces:**
- Consumes: `NoveltyArchive` from Task 1 (`novelty_archive.py`);
  `backends.base.FakeCVBackend` (tests) and `backends.vcv_rack.VCVRackBackend`
  (real CLI use); `features.read_aggregated_window(backend, sample_rate, aggregate_window_s) -> np.ndarray`.
- Produces: `generate_candidate(channels, archive, rng, mutation_prob, mutation_sigma) -> dict[str, float]`,
  `run_search(backend, budget, archive, rng, mutation_prob, mutation_sigma, settle_time_s, aggregate_window_s) -> None`,
  `save_archive(archive, backend, output_path) -> None`, and the `explore.py` CLI
  (`--config`, `--budget`, `--archive-size`, `--k-neighbors`,
  `--novelty-threshold`, `--mutation-prob`, `--mutation-sigma`,
  `--settle-time-s`, `--aggregate-window-s`, `--seed`, `--output`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_explore.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_explore.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'explore'`

- [ ] **Step 3: Write the implementation**

Create `explore.py`:

```python
# explore.py
import argparse
import os
import time

import numpy as np

from backends.vcv_rack import VCVRackBackend
from features import read_aggregated_window
from novelty_archive import NoveltyArchive


def generate_candidate(
    channels: list[str],
    archive: NoveltyArchive,
    rng: np.random.Generator,
    mutation_prob: float,
    mutation_sigma: float,
) -> dict[str, float]:
    if len(archive) > 0 and rng.random() < mutation_prob:
        base = archive.sample_member_cv(rng)
        mutated = base + rng.normal(0.0, mutation_sigma, size=len(channels))
        values = np.clip(mutated, 0.0, 1.0)
    else:
        values = rng.uniform(0.0, 1.0, size=len(channels))
    return {ch: float(v) for ch, v in zip(channels, values)}


def run_search(
    backend,
    budget: int,
    archive: NoveltyArchive,
    rng: np.random.Generator,
    mutation_prob: float,
    mutation_sigma: float,
    settle_time_s: float,
    aggregate_window_s: float,
) -> None:
    channels = backend.channels()
    sample_rate = backend.sample_rate()
    accepted = 0
    for i in range(budget):
        candidate = generate_candidate(channels, archive, rng, mutation_prob, mutation_sigma)
        for ch in channels:
            backend.set_cv(ch, candidate[ch])
        time.sleep(settle_time_s)
        feature_vector = read_aggregated_window(backend, sample_rate, aggregate_window_s)
        cv_vector = np.array([candidate[ch] for ch in channels])
        score = archive.novelty(feature_vector)
        added = archive.add(cv_vector, feature_vector, score)
        if added:
            accepted += 1
        print(
            f"[{i + 1}/{budget}] {'accepted' if added else 'rejected'} "
            f"score={score:.4f} archive_size={len(archive)}"
        )
    print(f"done: accepted {accepted}/{budget} candidates, final archive size {len(archive)}")


def save_archive(archive: NoveltyArchive, backend, output_path: str) -> None:
    cv_array, feature_array, scores = archive.to_arrays()
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.savez(
        output_path,
        cv=cv_array,
        features=feature_array,
        novelty_scores=scores,
        channels=backend.channels(),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Novelty search over the running VCV Rack patch's CV space: "
        "no goal, just discover and archive meaningfully different-sounding settings."
    )
    parser.add_argument("--config", default="configs/sequencer_test.yaml")
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--archive-size", type=int, default=60)
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--novelty-threshold", type=float, default=None)
    parser.add_argument("--mutation-prob", type=float, default=0.7)
    parser.add_argument("--mutation-sigma", type=float, default=0.15)
    parser.add_argument("--settle-time-s", type=float, default=0.5)
    parser.add_argument("--aggregate-window-s", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="data/sequencer_novelty_archive.npz")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    archive = NoveltyArchive(
        max_size=args.archive_size,
        k_neighbors=args.k_neighbors,
        novelty_threshold=args.novelty_threshold,
    )

    backend = VCVRackBackend(args.config)
    try:
        run_search(
            backend, args.budget, archive, rng,
            args.mutation_prob, args.mutation_sigma,
            args.settle_time_s, args.aggregate_window_s,
        )
        save_archive(archive, backend, args.output)
        _, _, scores = archive.to_arrays()
        if len(scores) > 0:
            print(
                f"saved {len(scores)} archive members to {args.output} "
                f"(novelty score min={scores.min():.4f} max={scores.max():.4f} mean={scores.mean():.4f})"
            )
        else:
            print(f"saved empty archive to {args.output}")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_explore.py -v`
Expected: all PASS

- [ ] **Step 5: Run the full non-hardware suite to confirm no regressions**

Run: `python3 -m pytest -q`
Expected: all previously-passing tests still pass, plus this task's new ones (deselected hardware count unchanged at 4)

- [ ] **Step 6: Commit**

```bash
git add explore.py tests/test_explore.py
git commit -m "Add explore.py: novelty-search CLI over the running VCV Rack patch"
```

---

### Task 3: Live capstone run and documentation

**Files:**
- Modify: `CLAUDE.md` (append a dated section reporting the real run)
- Modify: `README.md` ("What's here" section, new entries for
  `novelty_archive.py` and `explore.py`)
- Create: `data/sequencer_novelty_archive.npz` (the real output of the run —
  not committed to git if `data/` is already gitignored; check `git status`
  after the run and follow whatever the existing `data/*.npz` files in this
  repo already do — if earlier dataset files like
  `data/sequencer_sweep_dataset.npz` are tracked in git, commit this one the
  same way for consistency; if they're gitignored, leave it untracked)

**Interfaces:**
- Consumes: `explore.py`'s CLI from Task 2, the already-existing
  `patches/sequencer_test.vcv` / `configs/sequencer_test.yaml` from Stage 0.
- Produces: nothing new for later tasks — this is the stage's own
  capstone/completion check, matching the role Stage 0's Task 5 and Stage
  1's Task 6 played for their stages.

- [ ] **Step 1: Confirm (don't assume) the live environment is healthy**

Before touching anything, check what's actually running — this project's
`CLAUDE.md` documents exactly how (search it for "Launch check",
"PipeWire/Pulse device-targeting", and "crash-recovery dialog" before
doing anything by hand):

```bash
pgrep -af "Rack2Free/Rack"
pw-link -l | grep -i vcv_loop
pactl get-default-sink
pactl get-default-source
```

If a `Rack2Free/Rack` process is already running AND `pw-link -l` shows
routing through `vcv_loop`/`vcv_loop.monitor` AND the default sink/source
are already `vcv_loop`/`vcv_loop.monitor`: the environment is healthy,
skip straight to Step 2.

If not: relaunch `patches/sequencer_test.vcv` (append the `"END"` marker
to `~/.local/share/Rack2/log.txt` first if the last exit wasn't clean, per
`CLAUDE.md`'s "recurring crashed last session dialog" section) and
re-establish the PipeWire routing (`pactl load-module module-null-sink
sink_name=vcv_loop sink_properties=device.description=VCV_Loopback`,
`pactl set-default-sink vcv_loop`, `pactl set-default-source
vcv_loop.monitor` — skip `load-module` if the sink already exists).
Confirm the MIDI-CAT module's input device resolved
(`Midi Through:Midi Through Port-0 14:0`) via a screenshot, the same check
Stage 0's own launch check used.

- [ ] **Step 2: Run the real capstone search**

```bash
python3 explore.py --config configs/sequencer_test.yaml \
  --budget 200 --archive-size 60 --k-neighbors 5 \
  --settle-time-s 0.5 --aggregate-window-s 3.0 --seed 0 \
  --output data/sequencer_novelty_archive.npz
```

At ~3.5s/candidate this is roughly 12 minutes — run it the same way this
project already runs anything of that length (background with `nohup ...
& disown`, poll in a few-minute chunks per `CLAUDE.md`'s established
long-collection pattern), not foregrounded and blocking.

- [ ] **Step 3: Report honestly, including if something looks off**

Load the saved archive and report, in a new dated `CLAUDE.md` section:
final archive size (did it fill to 60, or stop short — and if short, is
that because `--budget 200` wasn't enough candidates, or because
candidates kept getting rejected?), the min/max/mean novelty score among
kept members, and a concrete spot-check: pick 2-3 archived CV vectors,
print their measured `[mean, std] x loudness, brightness, pitch]` feature
vectors side by side, and confirm by eye that they really do look
meaningfully different from each other (not just accepted by the
algorithm on paper). If the archive clusters unexpectedly (e.g. most
members land in a narrow band of one channel), or if score behavior looks
miscalibrated against this real instrument (e.g. almost everything gets
accepted, or almost everything gets rejected after the first few), report
that plainly — this stage's own bar for "done" is an honest report of what
a real run produced, not a claim of a well-organized discovered library.

- [ ] **Step 4: Update README's "What's here" section**

Add entries for `novelty_archive.py` and `explore.py` immediately after
the existing `instruction_to_preset.py` entry, in the same style as the
other entries in that section (one paragraph each, what it does, how it
relates to the existing pipeline). No new entries needed in the
Requirements line — this stage adds no new dependency.

- [ ] **Step 5: Full regression check**

```bash
python3 -m pytest -q
gozer run --chips 1 --who "claude:tt-cv-agency" --reason "final Stage 2 suite check" -- python3 -m pytest -q -m hardware
```

Expected: the non-hardware count grows by this stage's new tests over the
68 baseline, hardware stays at 4 passed, 0 failures either way.

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md README.md
git add data/sequencer_novelty_archive.npz  # only if data/*.npz files are already tracked in this repo -- check first
git commit -m "Stage 2 capstone: live novelty-search run against sequencer_test.vcv, docs"
```
