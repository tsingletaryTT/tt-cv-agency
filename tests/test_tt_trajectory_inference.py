import numpy as np
import pytest
import torch
from trajectory_model import TrajectoryPredictor, save_predictor_weights


@pytest.mark.hardware
def test_tt_trajectory_inference_matches_pytorch_reference(tmp_path):
    # Imported here (not at module scope) so merely collecting this test
    # file never touches ttnn / opens a device without a gozer lease.
    import ttnn  # noqa: F401
    from tt_trajectory_inference import TrajectoryTTInferenceEngine

    torch.manual_seed(0)
    model = TrajectoryPredictor(state_dim=6, action_dim=8)
    weights_path = str(tmp_path / "weights.npz")
    save_predictor_weights(model, weights_path)

    states = np.random.default_rng(0).uniform(0, 1, size=(4, 6)).astype(np.float32)
    actions = np.random.default_rng(1).uniform(-0.1, 0.1, size=(4, 8)).astype(np.float32)

    with torch.no_grad():
        reference = model(
            torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32)
        ).numpy()

    engine = TrajectoryTTInferenceEngine(weights_path=weights_path)
    try:
        result = engine.predict_next_state(states, actions)
    finally:
        engine.close()

    assert result.shape == (4, 6)
    # bfloat16 on-device precision -- same tolerance tt_inference.py's own
    # equivalent test already established as generously-but-really covering
    # this model size.
    assert np.allclose(result, reference, atol=0.1)


@pytest.mark.hardware
def test_tt_trajectory_inference_init_failure_does_not_leak_device(tmp_path):
    # Imported here (not at module scope) so merely collecting this test
    # file never touches ttnn / opens a device without a gozer lease.
    import ttnn
    from tt_trajectory_inference import TrajectoryTTInferenceEngine

    torch.manual_seed(0)
    model = TrajectoryPredictor(state_dim=6, action_dim=8)
    weights_path = str(tmp_path / "bad_weights.npz")
    save_predictor_weights(model, weights_path)

    # Corrupt the saved file by dropping a required key ("b3"), forcing
    # to_device(weights["b3"]) to raise a KeyError partway through
    # __init__'s weight-staging loop.
    good = np.load(weights_path)
    incomplete = {k: good[k] for k in good.files if k != "b3"}
    np.savez(weights_path, **incomplete)

    with pytest.raises(KeyError):
        TrajectoryTTInferenceEngine(weights_path=weights_path)

    # If the device from the failed construction were still open, this
    # would raise (device already active) instead of succeeding cleanly.
    probe_device = ttnn.open_device(device_id=0)
    ttnn.close_device(probe_device)


@pytest.mark.hardware
def test_tt_trajectory_inference_rejects_channel_order_mismatch(tmp_path):
    # Imported here (not at module scope) so merely collecting this test
    # file never touches ttnn / opens a device without a gozer lease.
    import ttnn  # noqa: F401
    from tt_trajectory_inference import ActionChannelMismatchError, TrajectoryTTInferenceEngine

    torch.manual_seed(0)
    model = TrajectoryPredictor(state_dim=6, action_dim=3)
    weights_path = str(tmp_path / "weights.npz")
    save_predictor_weights(model, weights_path, action_channels=["vco_freq", "vco_fm", "vca_level"])

    with pytest.raises(ActionChannelMismatchError):
        TrajectoryTTInferenceEngine(
            weights_path=weights_path, expected_channels=["vca_level", "vco_fm", "vco_freq"]
        )

    unlabeled_weights_path = str(tmp_path / "unlabeled_weights.npz")
    save_predictor_weights(model, unlabeled_weights_path)
    with pytest.raises(ActionChannelMismatchError):
        TrajectoryTTInferenceEngine(
            weights_path=unlabeled_weights_path, expected_channels=["vco_freq", "vco_fm", "vca_level"]
        )

    engine = TrajectoryTTInferenceEngine(
        weights_path=weights_path, expected_channels=["vco_freq", "vco_fm", "vca_level"]
    )
    engine.close()
