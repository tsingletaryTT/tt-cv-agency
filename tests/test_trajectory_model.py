import numpy as np
import torch
from trajectory_model import (
    TrajectoryPredictor, evaluate_predictor_per_dimension, save_predictor_weights,
    train_predictor, train_val_split_trajectories,
)


def test_trajectory_predictor_forward_shape():
    model = TrajectoryPredictor(state_dim=6, action_dim=8)
    state = torch.randn(1, 6)
    action = torch.randn(1, 8)
    out = model(state, action)
    assert out.shape == (1, 6)


def test_trajectory_predictor_batched_shape():
    model = TrajectoryPredictor(state_dim=6, action_dim=8)
    state = torch.randn(5, 6)
    action = torch.randn(5, 8)
    out = model(state, action)
    assert out.shape == (5, 6)


def test_train_predictor_infers_dimensions_from_data():
    rng = np.random.default_rng(0)
    states = rng.uniform(0, 1, size=(50, 6))
    actions = rng.uniform(-0.1, 0.1, size=(50, 8))
    next_states = rng.uniform(0, 1, size=(50, 6))

    model = train_predictor(states, actions, next_states, epochs=1)

    assert model.fc1.in_features == 14  # state_dim(6) + action_dim(8)
    assert model.fc3.out_features == 6


def test_train_predictor_reduces_loss_on_learnable_synthetic_data():
    # Synthetic ground truth: next_state = clip(state + action, 0, 1) --
    # additive, easily learnable, state_dim == action_dim == 4 here.
    rng = np.random.default_rng(0)
    states = rng.uniform(0, 1, size=(500, 4))
    actions = rng.uniform(-0.1, 0.1, size=(500, 4))
    next_states = np.clip(states + actions, 0.0, 1.0)

    model = train_predictor(states, actions, next_states, epochs=400, lr=1e-2)

    test_state = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)
    test_action = torch.tensor([[0.05, -0.05, 0.0, 0.02]], dtype=torch.float32)
    prediction = model(test_state, test_action).detach().numpy()
    expected = np.array([[0.55, 0.45, 0.5, 0.52]])
    assert np.allclose(prediction, expected, atol=0.1)


def test_train_val_split_trajectories_partitions_without_overlap():
    rng = np.random.default_rng(0)
    states = rng.uniform(0, 1, size=(100, 6))
    actions = rng.uniform(-0.1, 0.1, size=(100, 8))
    next_states = rng.uniform(0, 1, size=(100, 6))

    (s_train, a_train, ns_train), (s_val, a_val, ns_val) = train_val_split_trajectories(
        states, actions, next_states, val_fraction=0.2, seed=0
    )
    assert len(s_train) == 80 and len(s_val) == 20
    assert len(a_train) == 80 and len(a_val) == 20
    assert len(ns_train) == 80 and len(ns_val) == 20
    combined = np.concatenate([s_train, s_val], axis=0)
    assert np.allclose(np.sort(states, axis=0), np.sort(combined, axis=0))


def test_train_val_split_trajectories_is_reproducible_given_same_seed():
    rng = np.random.default_rng(1)
    states = rng.uniform(0, 1, size=(50, 6))
    actions = rng.uniform(-0.1, 0.1, size=(50, 8))
    next_states = rng.uniform(0, 1, size=(50, 6))

    (s_train_a, _, _), (s_val_a, _, _) = train_val_split_trajectories(
        states, actions, next_states, val_fraction=0.2, seed=7
    )
    (s_train_b, _, _), (s_val_b, _, _) = train_val_split_trajectories(
        states, actions, next_states, val_fraction=0.2, seed=7
    )

    assert np.array_equal(s_train_a, s_train_b)
    assert np.array_equal(s_val_a, s_val_b)


def test_evaluate_predictor_per_dimension_reports_zero_mse_and_perfect_r2_for_exact_model():
    class PerfectModel(TrajectoryPredictor):
        def forward(self, state, action):
            return state  # predicts next_state == state, matching the fixture below

    model = PerfectModel(state_dim=3, action_dim=2)
    states_val = np.array([[0.1, 0.5, 0.9], [0.2, 0.4, 0.8]])
    actions_val = np.zeros((2, 2))
    next_states_val = states_val.copy()
    baseline_mean = np.array([0.15, 0.45, 0.85])

    mse, r2 = evaluate_predictor_per_dimension(model, states_val, actions_val, next_states_val, baseline_mean=baseline_mean)

    assert np.allclose(mse, 0.0, atol=1e-6)
    assert np.allclose(r2, 1.0, atol=1e-6)


def test_evaluate_predictor_per_dimension_reports_zero_r2_for_mean_predicting_model():
    baseline_mean = np.array([0.3, 0.5, 0.7])

    class MeanModel(TrajectoryPredictor):
        def forward(self, state, action):
            return torch.tensor(baseline_mean, dtype=torch.float32).unsqueeze(0).repeat(state.shape[0], 1)

    model = MeanModel(state_dim=3, action_dim=2)
    rng = np.random.default_rng(0)
    states_val = rng.uniform(0, 1, size=(20, 3))
    actions_val = rng.uniform(-0.1, 0.1, size=(20, 2))
    next_states_val = rng.uniform(0, 1, size=(20, 3))

    _, r2 = evaluate_predictor_per_dimension(model, states_val, actions_val, next_states_val, baseline_mean=baseline_mean)

    assert np.allclose(r2, 0.0, atol=1e-6)


def test_save_predictor_weights_writes_expected_arrays(tmp_path):
    model = TrajectoryPredictor(state_dim=6, action_dim=8)
    out_path = tmp_path / "weights.npz"
    save_predictor_weights(model, str(out_path))

    loaded = np.load(out_path)
    assert set(loaded.keys()) == {"W1", "b1", "W2", "b2", "W3", "b3"}
    assert loaded["W1"].shape == (14, 32)
    assert loaded["b1"].shape == (32,)
    assert loaded["W2"].shape == (32, 32)
    assert loaded["b2"].shape == (32,)
    assert loaded["W3"].shape == (32, 6)
    assert loaded["b3"].shape == (6,)


def test_save_predictor_weights_records_action_channels_when_given(tmp_path):
    model = TrajectoryPredictor(state_dim=6, action_dim=3)
    out_path = tmp_path / "weights.npz"
    save_predictor_weights(model, str(out_path), action_channels=["vco_freq", "vco_fm", "vca_level"])

    loaded = np.load(out_path)
    assert "action_channels" in loaded.files
    assert loaded["action_channels"].tolist() == ["vco_freq", "vco_fm", "vca_level"]
