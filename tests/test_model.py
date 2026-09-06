import numpy as np
import torch
from model import InverseCVModel, train_model, save_weights


def test_inverse_cv_model_forward_shape():
    model = InverseCVModel()
    out = model(torch.randn(1, 3))
    assert out.shape == (1, 3)


def test_train_model_reduces_loss_on_learnable_synthetic_data():
    # Synthetic ground truth: cv = features (identity-ish, easily learnable),
    # so a model that trains correctly should get close to it.
    rng = np.random.default_rng(0)
    features = rng.uniform(0, 1, size=(500, 3))
    cv = features.copy()

    model = train_model(cv, features, epochs=300, lr=1e-2)

    test_features = torch.tensor([[0.2, 0.5, 0.8]], dtype=torch.float32)
    prediction = model(test_features).detach().numpy()
    assert np.allclose(prediction, [[0.2, 0.5, 0.8]], atol=0.1)


def test_save_weights_writes_expected_arrays(tmp_path):
    model = InverseCVModel()
    out_path = tmp_path / "weights.npz"
    save_weights(model, str(out_path))

    loaded = np.load(out_path)
    assert set(loaded.keys()) == {"W1", "b1", "W2", "b2", "W3", "b3"}
    assert loaded["W1"].shape == (3, 32)
    assert loaded["b1"].shape == (32,)
    assert loaded["W2"].shape == (32, 32)
    assert loaded["b2"].shape == (32,)
    assert loaded["W3"].shape == (32, 3)
    assert loaded["b3"].shape == (3,)
