# trajectory_model.py
import numpy as np
import torch
import torch.nn as nn


class TrajectoryPredictor(nn.Module):
    """A 3-layer MLP that predicts the next state given current state and action.

    The model concatenates state and action into a single input vector, then
    passes through two hidden ReLU layers before outputting the predicted next
    state. This architecture matches the structure of InverseCVModel in model.py
    (which maps features -> CV outputs), adapted here to map (state, action) ->
    next_state for trajectory prediction. The concatenation is the natural way
    to combine heterogeneous inputs (state dimensions + action dimensions) into
    a single learned representation.
    """
    def __init__(self, state_dim: int = 6, action_dim: int = 8, hidden: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, state_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def train_predictor(
    states: np.ndarray, actions: np.ndarray, next_states: np.ndarray,
    epochs: int = 200, lr: float = 1e-3, hidden: int = 32,
) -> TrajectoryPredictor:
    """Train a TrajectoryPredictor on a batch of (state, action, next_state) tuples.

    Performs full-batch gradient descent with the Adam optimizer and MSE loss.
    The state and action dimensions are inferred from the data shapes, allowing
    the same training function to work for any state/action dimensionality.
    `hidden` is exposed (default unchanged at 32, matching every existing
    caller's behavior) so a future dataset-size/capacity tradeoff can be
    explored without editing this function again.
    """
    model = TrajectoryPredictor(state_dim=states.shape[1], action_dim=actions.shape[1], hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    s = torch.tensor(states, dtype=torch.float32)
    a = torch.tensor(actions, dtype=torch.float32)
    y = torch.tensor(next_states, dtype=torch.float32)

    for _ in range(epochs):
        optimizer.zero_grad()
        prediction = model(s, a)
        loss = loss_fn(prediction, y)
        loss.backward()
        optimizer.step()

    return model


def train_val_split_trajectories(
    states: np.ndarray, actions: np.ndarray, next_states: np.ndarray,
    val_fraction: float = 0.2, seed: int = 0,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Split (state, action, next_state) trajectory tuples into a train set and a
    held-out validation set, shuffled by `seed` so the split is reproducible.
    Returns ((states_train, actions_train, next_states_train), (states_val, actions_val, next_states_val)).

    A held-out validation split is essential because in-sample training error can hide
    that the model learned very little on certain state dimensions. Per-dimension
    validation metrics surface which dimensions the model has mastered (high R^2) and
    which it has not (low R^2), revealing whether the model is suitable for
    trajectory-following control on all dimensions. An aggregate in-sample MSE alone
    cannot expose these per-dimension differences.
    """
    n = len(states)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n)
    n_val = int(round(n * val_fraction))
    val_idx = shuffled[:n_val]
    train_idx = shuffled[n_val:]
    return (
        (states[train_idx], actions[train_idx], next_states[train_idx]),
        (states[val_idx], actions[val_idx], next_states[val_idx]),
    )


def evaluate_predictor_per_dimension(
    model: TrajectoryPredictor,
    states_val: np.ndarray, actions_val: np.ndarray, next_states_val: np.ndarray,
    baseline_mean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-state-dimension MSE and R^2 (against a predict-the-mean baseline) on a
    held-out (states_val, actions_val, next_states_val) set. `baseline_mean`
    should be the *training* set's per-dimension next_state mean (not the
    validation set's own mean) so the baseline isn't computed from the same
    data it's being judged against -- an R^2 of 0.0 means "no better than always
    predicting the training-set average next-state value for this dimension,"
    which is the honest bar a trajectory-following model needs to clear per
    dimension, not just on average across the whole state space.
    """
    model.eval()
    with torch.no_grad():
        prediction = model(
            torch.tensor(states_val, dtype=torch.float32),
            torch.tensor(actions_val, dtype=torch.float32),
        ).numpy()

    mse_per_dim = np.mean((prediction - next_states_val) ** 2, axis=0)

    ss_res = np.sum((prediction - next_states_val) ** 2, axis=0)
    ss_tot = np.sum((next_states_val - baseline_mean) ** 2, axis=0)
    # Guard against a degenerate all-identical validation column (ss_tot=0)
    # rather than dividing by zero -- shouldn't happen with real trajectory data,
    # but a NaN/inf R^2 would be a confusing way to find that out.
    r2_per_dim = np.where(ss_tot > 0, 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, 1.0), np.nan)

    return mse_per_dim, r2_per_dim


def evaluate_persistence_baseline_per_dimension(
    states_val: np.ndarray, next_states_val: np.ndarray, baseline_mean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Held-out per-dimension MSE/R^2 for the trivial 'next_state == state'
    baseline (zero parameters, no action used at all). This matters
    specifically for a forward dynamics model: states are strongly
    autocorrelated step to step, so scoring only against a
    predict-the-training-mean baseline (evaluate_predictor_per_dimension's
    baseline) can make a model look like it's learned a lot when it's
    really only capturing that states don't change much between reads --
    a persistence baseline can score a high R^2 too, and a trained model
    should be judged against BOTH. See baseline_mean's docstring on
    evaluate_predictor_per_dimension for why it must come from the
    training set, not the validation set."""
    mse_per_dim = np.mean((states_val - next_states_val) ** 2, axis=0)
    ss_res = np.sum((states_val - next_states_val) ** 2, axis=0)
    ss_tot = np.sum((next_states_val - baseline_mean) ** 2, axis=0)
    r2_per_dim = np.where(ss_tot > 0, 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, 1.0), np.nan)
    return mse_per_dim, r2_per_dim


def save_predictor_weights(model: TrajectoryPredictor, path: str, action_channels: list[str] | None = None) -> None:
    """Save the model's weights and optional action channel metadata to an npz file.

    `action_channels` records the action channel order (e.g. the current patch's
    ["vco_freq", "vco_fm", "vca_level"] or whatever action set was used during
    training) that this model's action input was trained against. It's optional
    (so existing callers/tests that only care about the weight arrays keep
    working unchanged), but any real training run should pass it: without it,
    nothing stops the model's action input vector being fed actions in the wrong
    order at inference time, a silent-wrong-action failure mode that would look
    like a working-but-wrong trajectory model rather than an error. Task 4's
    TrajectoryTTInferenceEngine loads these weights and expects the same
    W1/b1/W2/b2/W3/b3 (+ optional action_channels) keys.
    """
    extra = {"action_channels": np.array(action_channels)} if action_channels is not None else {}
    np.savez(
        path,
        W1=model.fc1.weight.detach().numpy().T,
        b1=model.fc1.bias.detach().numpy(),
        W2=model.fc2.weight.detach().numpy().T,
        b2=model.fc2.bias.detach().numpy(),
        W3=model.fc3.weight.detach().numpy().T,
        b3=model.fc3.bias.detach().numpy(),
        **extra,
    )


if __name__ == "__main__":
    # data/sequencer_trajectory_dataset.npz -- written by
    # trajectory_collection.py's own __main__.
    data = np.load("data/sequencer_trajectory_dataset.npz")
    channels = data["channels"].tolist()

    (s_train, a_train, ns_train), (s_val, a_val, ns_val) = train_val_split_trajectories(
        data["states"], data["actions"], data["next_states"], val_fraction=0.2, seed=0
    )
    model = train_predictor(s_train, a_train, ns_train, epochs=500)

    train_mean = ns_train.mean(axis=0)
    mse_per_dim, r2_per_dim = evaluate_predictor_per_dimension(model, s_val, a_val, ns_val, baseline_mean=train_mean)
    persistence_mse, persistence_r2 = evaluate_persistence_baseline_per_dimension(
        s_val, ns_val, baseline_mean=train_mean
    )

    state_dim_names = ["loud_mean", "loud_std", "bright_mean", "bright_std", "pitch_mean", "pitch_std"]
    print(f"train/val split: {len(s_train)} train / {len(s_val)} held-out validation transitions")
    print(f"{'state dim':<12} {'val MSE':>10} {'val R^2':>10} {'persist R^2':>12}")
    for name, mse, r2, p_r2 in zip(state_dim_names, mse_per_dim, r2_per_dim, persistence_r2):
        print(f"{name:<12} {mse:>10.4f} {r2:>10.4f} {p_r2:>12.4f}")

    save_predictor_weights(model, "data/sequencer_trajectory_model_weights.npz", action_channels=channels)
    print("saved trained predictor weights to data/sequencer_trajectory_model_weights.npz")
