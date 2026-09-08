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
