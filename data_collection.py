# data_collection.py
import time
import numpy as np

from backends.base import CVBackend
from features import read_aggregated_window


def collect_sweep_dataset(
    backend: CVBackend,
    n_samples: int,
    settle_time_s: float,
    rng: np.random.Generator,
    sample_rate: int | None = None,
    aggregate_window_s: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    # sample_rate is pulled from the backend itself by default -- passing it
    # in independently is kept only for backward compatibility / test
    # convenience (e.g. a FakeCVBackend test that wants to assert behavior
    # at a sample_rate different from its own configured one). Real callers
    # should let this default to backend.sample_rate() so the value used for
    # feature extraction can never silently drift from what the backend
    # actually captured its audio at (this mismatch is exactly how the
    # estimate_pitch production-block-size bug stayed hidden).
    if sample_rate is None:
        sample_rate = backend.sample_rate()
    # A single instantaneous block can't represent a looping pattern (e.g.
    # the sequencer/filter-sweep patch from Task 1) -- read a whole window
    # of blocks spanning aggregate_window_s seconds and hand them all to
    # extract_features_aggregated, which reduces the window to [mean, std]
    # per feature (6-dim). block_size is pulled from the backend itself for
    # the same drift-proofing reason sample_rate is.
    channels = backend.channels()
    cv_array = np.zeros((n_samples, len(channels)))
    feature_array = np.zeros((n_samples, 6))

    for i in range(n_samples):
        cv_vec = rng.uniform(0.0, 1.0, size=len(channels))
        for ch, value in zip(channels, cv_vec):
            backend.set_cv(ch, float(value))
        time.sleep(settle_time_s)
        feature_array[i] = read_aggregated_window(backend, sample_rate, aggregate_window_s)
        cv_array[i] = cv_vec

    return cv_array, feature_array


if __name__ == "__main__":
    from backends.vcv_rack import VCVRackBackend

    # configs/sequencer_test.yaml -- the current (Stage 0) instrument: 8
    # CV channels driving a sequencer + filter-sweep LFO + filter envelope
    # on top of the Minimoog signal path (see CLAUDE.md's Stage 0 sections).
    # aggregate_window_s=5.0 (the collect_sweep_dataset default) matters here
    # too -- it must be long enough to span at least one loop of the
    # sequencer/LFO for the windowed [mean, std] features to mean anything.
    backend = VCVRackBackend("configs/sequencer_test.yaml")
    try:
        rng = np.random.default_rng(seed=0)
        # settle_time_s=0.5 -- confirmed generously above the real settle time
        # observed in Phase 1's round-trip verification (RMS transitions completed
        # within a few hundred ms); re-check with a stopwatch-style manual test
        # (Task 3 Step 6's pattern) if this dataset's model behaves oddly.
        #
        # n_samples=300 -- what Task 5 actually collected and verified
        # end-to-end against this patch. Before the windowed aggregate_window_s
        # read was added, a much larger n_samples (e.g. 3000) took ~25 minutes
        # at settle_time_s=0.5 alone; now every sample also reads a full
        # aggregate_window_s=5.0 window, so total runtime is roughly
        # n_samples * (settle_time_s + aggregate_window_s) seconds -- 3000
        # samples would now take close to 4.6 hours. Bump this deliberately,
        # with that arithmetic in mind, not by habit.
        cv_array, feature_array = collect_sweep_dataset(
            backend, n_samples=300, settle_time_s=0.5, rng=rng
        )
        import os
        os.makedirs("data", exist_ok=True)
        # sequencer_sweep_dataset.npz (not the old sweep_dataset.npz name --
        # that was the retired 3-channel dataset's filename, and model.py's
        # __main__ now reads this same name so the two scripts form a real
        # chain again).
        np.savez("data/sequencer_sweep_dataset.npz", cv=cv_array, features=feature_array, channels=backend.channels())
        print(f"saved {len(cv_array)} samples to data/sequencer_sweep_dataset.npz")
    finally:
        # A crash partway through a several-minute collection run must not
        # leave the audio stream / MIDI port open -- close it even if the
        # loop above raised.
        backend.close()
