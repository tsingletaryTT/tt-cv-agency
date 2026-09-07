import numpy as np
import torch
from model import InverseCVModel, evaluate_per_dimension, save_weights, train_model, train_val_split


def test_inverse_cv_model_forward_shape():
    model = InverseCVModel()
    out = model(torch.randn(1, 3))
    assert out.shape == (1, 3)


def test_inverse_cv_model_parametric_shapes():
    model = InverseCVModel(n_features=6, n_channels=8)
    x = torch.randn(4, 6)
    out = model(x)
    assert out.shape == (4, 8)


def test_inverse_cv_model_default_shapes_unchanged():
    model = InverseCVModel()
    x = torch.randn(4, 3)
    out = model(x)
    assert out.shape == (4, 3)


def test_train_model_infers_dimensions_from_data():
    rng = np.random.default_rng(0)
    feature_array = rng.uniform(0, 1, size=(50, 6))
    cv_array = rng.uniform(0, 1, size=(50, 8))

    model = train_model(cv_array, feature_array, epochs=1)

    assert model.fc1.in_features == 6
    assert model.fc3.out_features == 8


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


def test_save_weights_records_channel_order_when_given(tmp_path):
    model = InverseCVModel()
    out_path = tmp_path / "weights.npz"
    save_weights(model, str(out_path), channels=["vco_freq", "vco_fm", "vca_level"])

    loaded = np.load(out_path)
    assert "channels" in loaded.files
    assert loaded["channels"].tolist() == ["vco_freq", "vco_fm", "vca_level"]


def test_train_val_split_partitions_without_overlap_and_respects_fraction():
    rng = np.random.default_rng(0)
    cv = rng.uniform(0, 1, size=(100, 3))
    features = rng.uniform(0, 1, size=(100, 3))

    (cv_train, feat_train), (cv_val, feat_val) = train_val_split(cv, features, val_fraction=0.2, seed=0)

    assert len(cv_train) == 80
    assert len(cv_val) == 20
    assert len(feat_train) == 80
    assert len(feat_val) == 20
    # Every validation row must be a real row from the original array (not
    # zeros/garbage), and train+val together must cover all 100 rows
    # (no sample dropped, no sample duplicated across the split).
    combined = np.concatenate([cv_train, cv_val], axis=0)
    original_sorted = np.sort(cv, axis=0)
    combined_sorted = np.sort(combined, axis=0)
    assert np.allclose(original_sorted, combined_sorted)


def test_train_val_split_is_reproducible_given_same_seed():
    rng = np.random.default_rng(1)
    cv = rng.uniform(0, 1, size=(50, 3))
    features = rng.uniform(0, 1, size=(50, 3))

    (cv_train_a, _), (cv_val_a, _) = train_val_split(cv, features, val_fraction=0.2, seed=7)
    (cv_train_b, _), (cv_val_b, _) = train_val_split(cv, features, val_fraction=0.2, seed=7)

    assert np.array_equal(cv_train_a, cv_train_b)
    assert np.array_equal(cv_val_a, cv_val_b)


def test_evaluate_per_dimension_reports_zero_mse_and_perfect_r2_for_exact_model():
    # A model wired to predict cv_val exactly should report ~0 MSE and R^2
    # of ~1.0 on every dimension, regardless of the baseline.
    class PerfectModel(InverseCVModel):
        def forward(self, target_features):
            return target_features

    model = PerfectModel()
    cv_val = np.array([[0.1, 0.5, 0.9], [0.2, 0.4, 0.8]])
    feat_val = cv_val.copy()
    baseline_mean = np.array([0.15, 0.45, 0.85])

    mse, r2 = evaluate_per_dimension(model, cv_val, feat_val, baseline_mean=baseline_mean)

    assert np.allclose(mse, 0.0, atol=1e-6)
    assert np.allclose(r2, 1.0, atol=1e-6)


def test_evaluate_per_dimension_reports_zero_r2_for_mean_predicting_model():
    # A model that always predicts the training mean should score R^2 ~ 0
    # against a baseline_mean equal to that same constant -- it's exactly
    # as good as the baseline it's being compared to, no better, no worse.
    baseline_mean = np.array([0.3, 0.5, 0.7])

    class MeanModel(InverseCVModel):
        def forward(self, target_features):
            return torch.tensor(baseline_mean, dtype=torch.float32).unsqueeze(0).repeat(target_features.shape[0], 1)

    model = MeanModel()
    rng = np.random.default_rng(0)
    cv_val = rng.uniform(0, 1, size=(20, 3))
    feat_val = rng.uniform(0, 1, size=(20, 3))

    _, r2 = evaluate_per_dimension(model, cv_val, feat_val, baseline_mean=baseline_mean)

    assert np.allclose(r2, 0.0, atol=1e-6)
