import numpy as np
import pytest
import torch
import ttnn
from model import InverseCVModel, save_weights
from tt_inference import TTInferenceEngine


def test_tt_inference_matches_pytorch_reference(tmp_path):
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
