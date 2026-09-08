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
