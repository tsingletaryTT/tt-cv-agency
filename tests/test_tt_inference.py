import numpy as np
import pytest
import torch
from model import InverseCVModel, save_weights


@pytest.mark.hardware
def test_tt_inference_matches_pytorch_reference(tmp_path):
    # Imported here (not at module scope) so merely collecting this test
    # file never touches ttnn / opens a device without a gozer lease.
    import ttnn  # noqa: F401
    from tt_inference import TTInferenceEngine

    torch.manual_seed(0)
    model = InverseCVModel()
    weights_path = str(tmp_path / "weights.npz")
    save_weights(model, weights_path)

    target = np.array([0.3, 0.6, 0.9], dtype=np.float32)

    with torch.no_grad():
        reference = model(torch.tensor(target, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()

    engine = TTInferenceEngine(weights_path=weights_path)
    try:
        result = engine.predict_cv(target)
    finally:
        engine.close()

    assert result.shape == (3,)
    # bfloat16 on-device precision -- allow a generous but real tolerance,
    # verified in exploration to be within ~0.03 absolute for this model size.
    assert np.allclose(result, reference, atol=0.1)


@pytest.mark.hardware
def test_tt_inference_init_failure_does_not_leak_device(tmp_path):
    """A malformed weights file should raise a clear exception out of
    __init__ AND must not leak the opened device handle.

    Since Python discards a partially-constructed object when __init__
    raises, the caller never gets a TTInferenceEngine to call .close() on.
    __init__ must therefore close the device itself before re-raising. We
    verify this by immediately opening a fresh device on the same
    device_id afterward: if the failed engine had leaked its device, this
    second open would fail (device already in use) instead of succeeding.
    """
    # Imported here (not at module scope) so merely collecting this test
    # file never touches ttnn / opens a device without a gozer lease.
    import ttnn
    from tt_inference import TTInferenceEngine

    torch.manual_seed(0)
    model = InverseCVModel()
    weights_path = str(tmp_path / "bad_weights.npz")
    save_weights(model, weights_path)

    # Corrupt the saved file by dropping a required key ("b3"), forcing
    # to_device(weights["b3"]) to raise a KeyError partway through
    # __init__'s weight-staging loop.
    good = np.load(weights_path)
    incomplete = {k: good[k] for k in good.files if k != "b3"}
    np.savez(weights_path, **incomplete)

    with pytest.raises(KeyError):
        TTInferenceEngine(weights_path=weights_path)

    # If the device from the failed construction were still open, this
    # would raise (device already active) instead of succeeding cleanly.
    probe_device = ttnn.open_device(device_id=0)
    ttnn.close_device(probe_device)


@pytest.mark.hardware
def test_tt_inference_rejects_channel_order_mismatch(tmp_path):
    # Imported here (not at module scope) so merely collecting this test
    # file never touches ttnn / opens a device without a gozer lease.
    import ttnn  # noqa: F401
    from tt_inference import ChannelMismatchError, TTInferenceEngine

    torch.manual_seed(0)
    model = InverseCVModel()
    weights_path = str(tmp_path / "weights.npz")
    save_weights(model, weights_path, channels=["vco_freq", "vco_fm", "vca_level"])

    # A mismatched expected_channels order must be rejected before a device
    # is even opened -- the channel check happens ahead of ttnn.open_device.
    with pytest.raises(ChannelMismatchError):
        TTInferenceEngine(weights_path=weights_path, expected_channels=["vca_level", "vco_fm", "vco_freq"])

    # A weights file with no channel provenance at all is also rejected
    # when the caller supplies expected_channels, rather than silently
    # skipping the check.
    unlabeled_weights_path = str(tmp_path / "unlabeled_weights.npz")
    save_weights(model, unlabeled_weights_path)
    with pytest.raises(ChannelMismatchError):
        TTInferenceEngine(weights_path=unlabeled_weights_path, expected_channels=["vco_freq", "vco_fm", "vca_level"])

    # A matching order constructs cleanly and must still be closed.
    engine = TTInferenceEngine(weights_path=weights_path, expected_channels=["vco_freq", "vco_fm", "vca_level"])
    engine.close()


@pytest.mark.hardware
def test_tt_inference_handles_non_default_shape(tmp_path):
    # Imported here (not at module scope) so merely collecting this test
    # file never touches ttnn / opens a device without a gozer lease.
    import ttnn  # noqa: F401
    from tt_inference import TTInferenceEngine

    torch.manual_seed(0)
    model = InverseCVModel(n_features=6, n_channels=8)
    weights_path = str(tmp_path / "weights.npz")
    save_weights(model, weights_path)

    target = np.random.default_rng(0).uniform(0, 1, size=6).astype(np.float32)
    with torch.no_grad():
        reference = model(torch.tensor(target).unsqueeze(0)).squeeze(0).numpy()

    engine = TTInferenceEngine(weights_path=weights_path)
    try:
        result = engine.predict_cv(target)
    finally:
        engine.close()

    assert result.shape == (8,)
    assert np.allclose(result, reference, atol=0.1)
