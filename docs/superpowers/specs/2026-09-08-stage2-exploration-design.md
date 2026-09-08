# Stage 2: exploration / novelty search — design

## Context

Stage 2 of the roadmap (`CLAUDE.md`, "A four-stage roadmap", Path B): an
agent sweeps the instrument's CV space with no goal at all, looking for
settings that are meaningfully different from each other, and keeps a
bounded library of the most diverse ones it's found. This replaces
uniform-random sampling (what `data_collection.py` does today) with
something that actively seeks coverage of the interesting regions of an
8-dimensional space, and gives Stage 1 something concrete to check its
semantic guesses against later ("does the LLM's idea of 'squelchy' land
in a real high-resonance region this stage actually found?").

Out of scope for this stage: chasing any goal or target (that's every
other stage's job), training a model on the discovered archive (Stage
3's/a future stage's job — this stage only produces the archive),
touching the sequencer/LFO patch, feature extraction, or the CV backend
(Stage 0's already-shipped foundation), and any LLM/text involvement
(Stage 1's territory, already shipped).

## Algorithm: bounded-archive novelty search

Not MAP-Elites: MAP-Elites needs a predefined behavior-space grid plus a
per-cell fitness/quality function, and this stage has no target or
quality metric to optimize — "no goals" is explicit in the roadmap.
Plain novelty search fits directly instead: maintain a capped archive of
`(cv_vector, feature_vector)` pairs; repeatedly propose a candidate CV
vector, measure it for real, score how novel its feature vector is
against the archive's current contents, and keep it only if it's novel
enough to earn a spot.

```
                    ┌─────────────────────────────┐
                    │ generate_candidate            │
                    │  - mutate a random archive     │
                    │    member (Gaussian, clipped     │
                    │    to [0,1]), OR                 │
                    │  - draw fresh uniform-random     │
                    │    (keeps search from getting     │
                    │    stuck near one region)          │
                    └──────────────┬────────────────┘
                                   ▼
                    apply via backend.set_cv() per channel
                                   ▼
                        sleep settle_time_s
                                   ▼
                read_aggregated_window (real audio measurement,
                same shared helper Stage 0/1 already use)
                                   ▼
                    NoveltyArchive.novelty(feature_vector)
                    = mean distance to its k nearest
                      neighbors among the archive's
                      current feature vectors
                    (archive empty -> +inf, maximally novel)
                                   ▼
                    NoveltyArchive.add(cv, features, score)
                    - below novelty_threshold (if set)? reject
                    - archive has room? add
                    - else: compare against the archive's
                      current least-novel member (recomputed
                      leave-one-out); replace it if the
                      candidate beats it, else reject
                                   ▼
                    repeat for `budget` iterations, then
                    save the final archive to disk
```

### `NoveltyArchive` (new file: `novelty_archive.py`)

```python
class NoveltyArchive:
    def __init__(self, max_size: int, k_neighbors: int, novelty_threshold: float | None):
        """max_size: cap on archive entries. k_neighbors: how many nearest
        neighbors' distances to average for a novelty score (clamped to
        min(k_neighbors, current archive size) whenever the archive holds
        fewer than k_neighbors members). novelty_threshold: if not None, a
        candidate whose novelty score does not exceed this value is
        rejected outright, even if the archive has room -- if None, no
        threshold filtering, diversity is maintained purely by capacity
        and eviction."""

    def novelty(self, feature_vector: np.ndarray) -> float:
        """Mean Euclidean distance from feature_vector to its k nearest
        neighbors among the feature vectors currently in the archive.
        Returns float('inf') when the archive is empty (nothing to compare
        against -- anything is maximally novel)."""

    def add(self, cv_vector: np.ndarray, feature_vector: np.ndarray, score: float) -> bool:
        """Attempt to add (cv_vector, feature_vector) with precomputed
        novelty `score` (the caller already has it from calling novelty()
        before deciding to measure -- add() does not recompute it for the
        candidate). Returns whether it was actually added.

        - If novelty_threshold is not None and score <= novelty_threshold:
          reject (return False), regardless of capacity.
        - Else if len(archive) < max_size: append, return True.
        - Else: recompute leave-one-out novelty for every existing member
          (each member's novelty against the rest of the archive, i.e.
          itself excluded) using the same k formula, find the minimum;
          if score > that minimum, evict that member and append the
          candidate, return True; else reject, return False.
        """

    def sample_member_cv(self, rng: np.random.Generator) -> np.ndarray:
        """Return a uniformly random archive member's cv_vector, for
        mutation-based candidate generation. Raises if the archive is
        empty -- callers must fall back to pure-random generation in that
        case (see generate_candidate below)."""

    def __len__(self) -> int: ...

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (cv_array, feature_array, novelty_scores) stacked in
        insertion order, for saving to disk."""
```

Internally: a plain Python list of `(cv_vector, feature_vector, score)`
tuples (or three parallel lists) is enough at these sizes (tens to low
hundreds of members) — `novelty()`/eviction are O(archive_size ×
feature_dim) per call, trivial at this scale. No need for a k-d tree or
similar; don't add one.

### Candidate generation (in `explore.py`)

```python
def generate_candidate(
    channels: list[str],
    archive: NoveltyArchive,
    rng: np.random.Generator,
    mutation_prob: float,
    mutation_sigma: float,
) -> dict[str, float]:
    """With probability mutation_prob (only if the archive is non-empty),
    mutate a random archive member: add independent Gaussian noise
    (std=mutation_sigma) to each channel's value, clip to [0, 1].
    Otherwise (or if the archive is empty), draw every channel
    independently from Uniform(0, 1). Returns a dict keyed by channel
    name, matching backend.channels() order/identity, so the caller can
    apply it via backend.set_cv(channel, value) per key."""
```

### `explore.py` (new file, CLI)

Same shape as `instruction_to_preset.py`/`control_loop.py`'s `__main__`:
build a `VCVRackBackend`, run the loop, save results, always `close()` in
a `finally`.

```
usage: explore.py [-h] [--config CONFIG] [--budget BUDGET]
                   [--archive-size ARCHIVE_SIZE] [--k-neighbors K_NEIGHBORS]
                   [--novelty-threshold NOVELTY_THRESHOLD]
                   [--mutation-prob MUTATION_PROB]
                   [--mutation-sigma MUTATION_SIGMA]
                   [--settle-time-s SETTLE_TIME_S]
                   [--aggregate-window-s AGGREGATE_WINDOW_S] [--seed SEED]
                   [--output OUTPUT]
```

Defaults: `--config configs/sequencer_test.yaml` (the current 8-channel
instrument), `--budget 200` (candidates evaluated -- each one is a real
hardware measurement, so this directly sets runtime: at
`settle_time_s=0.5` + `aggregate_window_s=3.0`, ~3.5s/candidate ->
~12 minutes for 200, matching this project's established
background-collection pattern for anything longer), `--archive-size 60`,
`--k-neighbors 5`, `--novelty-threshold None` (no threshold filtering by
default -- rely on capacity + eviction; pass a float to also reject
outright-boring candidates), `--mutation-prob 0.7`, `--mutation-sigma
0.15`, `--settle-time-s 0.5`, `--aggregate-window-s 3.0` (shorter than
Stage 0's 5.0s default -- an explicit, cheaper choice for this stage's
much larger candidate count; still spans most of `sequencer_test.vcv`'s
fastest loop periods per the Stage 0 fix-round's re-narrowed `seq_tempo`/
`sweep_rate` ranges), `--seed 0`, `--output
data/sequencer_novelty_archive.npz`.

Loop body per iteration: `generate_candidate` -> apply each channel via
`backend.set_cv` -> `time.sleep(settle_time_s)` -> `read_aggregated_window
(backend, backend.sample_rate(), aggregate_window_s)` -> `archive.novelty
(feature_vector)` -> `archive.add(cv_vector, feature_vector, score)` ->
print one progress line per iteration (iteration number, whether added,
current archive size, score) so a long background run's progress is
checkable the same way `data_collection.py`'s long runs already are.

At the end: `archive.to_arrays()` -> `np.savez(output, cv=cv_array,
features=feature_array, novelty_scores=scores, channels=backend.channels
())` -- same `channels=` provenance convention `data_collection.py`
already uses, so a later consumer can verify channel-order agreement the
same way `tt_inference.py` already does for the trained model's weights.
Print a final summary: archive size, min/max/mean novelty score among
kept members, and how many of `budget` candidates were accepted overall.

## Global constraints

- No new `ttnn`/hardware dependency and no `gozer` lease anywhere in this
  stage -- this is pure CPU controller logic against the existing
  `CVBackend` abstraction, same as Stage 0/1's non-hardware code.
- CV values stay `[0, 1]` at every interface boundary, same as every
  prior stage -- `generate_candidate`'s mutation path must clip.
- Channel identity lives in the patch's config/backend, never hardcoded
  -- `NoveltyArchive` and `generate_candidate` operate on whatever
  `channels: list[str]` the backend reports, never a fixed count or name.
- Reuse `features.read_aggregated_window` and `VCVRackBackend` unchanged
  -- no edits to `features.py`, `backends/`, or any `configs/*.yaml` in
  this stage.
- New code is two flat files (`novelty_archive.py`, `explore.py`) at the
  repo root, matching this project's existing convention (`features.py`,
  `data_collection.py`, `control_loop.py`, `model.py` all live at the
  root) -- no new package. Unlike Stage 1's `instruction_parser/`, there
  is no swappable-implementation abstraction here that would justify one.

## Verification plan

1. **`NoveltyArchive` is fully unit-testable against synthetic feature
   vectors, no hardware needed**: novelty scoring (a hand-computed k-NN
   distance case, k=1 and k>1), the empty-archive `+inf` case, filling
   below `max_size` always adds, threshold rejection when set, eviction
   replacing the correct (least-novel) member once full, eviction
   correctly rejecting a candidate that is less novel than every current
   member, `k_neighbors` clamping when the archive holds fewer members
   than `k_neighbors`.
2. **`generate_candidate` is deterministic under a seeded RNG and always
   produces valid `[0, 1]` values**: a seeded `np.random.default_rng`
   reproduces the same sequence of candidates across two runs; a mutation
   pushed outside `[0, 1]` by construction (e.g. a base value near an
   edge plus noise) comes back clipped; an empty archive always falls
   back to pure-random regardless of `mutation_prob`.
3. **The `explore.py` loop is tested against `FakeCVBackend`** (from
   `backends/base.py`, already used by `tests/test_data_collection.py`)
   with a controllable/varying fake audio block per call, confirming:
   the loop calls `set_cv` for every channel per candidate, respects
   `budget` (exactly that many candidates evaluated), and the saved
   output's `channels=` array matches `backend.channels()`.
4. **Live capstone run, actually achievable this time** (unlike Stage
   1's LLM calls, this needs no external credentials -- only the already
   -proven `VCVRackBackend` against a running VCV Rack instance): run
   `explore.py` against whichever patch is live (`sequencer_test.vcv` per
   the current default config), and report the real result honestly --
   final archive size, the spread of novelty scores among kept members,
   and a spot-check that a few archived CV vectors really do sound
   different from each other (measured feature vectors far apart, not
   just accepted by the algorithm). This is the stage's own bar for
   "done," the same role Stage 0's Task 5 and Stage 1's capstone script
   played for their stages -- report what actually happens, including if
   the archive ends up less diverse than hoped (e.g. clustering in one
   region of a dimension the instrument doesn't actually vary much) or
   if a novelty-search parameter choice above turns out miscalibrated
   against this real instrument.

Full existing test suite (68 non-hardware + 4 hardware) must stay green;
this stage adds tests, doesn't touch anything hardware-marked.
