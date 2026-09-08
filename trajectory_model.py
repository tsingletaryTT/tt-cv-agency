# trajectory_model.py
import numpy as np
import torch
import torch.nn as nn


class TrajectoryPredictor(nn.Module):
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
    epochs: int = 200, lr: float = 1e-3,
) -> TrajectoryPredictor:
    model = TrajectoryPredictor(state_dim=states.shape[1], action_dim=actions.shape[1])
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
    model.eval()
    with torch.no_grad():
        prediction = model(
            torch.tensor(states_val, dtype=torch.float32),
            torch.tensor(actions_val, dtype=torch.float32),
        ).numpy()

    mse_per_dim = np.mean((prediction - next_states_val) ** 2, axis=0)

    ss_res = np.sum((prediction - next_states_val) ** 2, axis=0)
    ss_tot = np.sum((next_states_val - baseline_mean) ** 2, axis=0)
    r2_per_dim = np.where(ss_tot > 0, 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, 1.0), np.nan)

    return mse_per_dim, r2_per_dim


def save_predictor_weights(model: TrajectoryPredictor, path: str, action_channels: list[str] | None = None) -> None:
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

    state_dim_names = ["loud_mean", "loud_std", "bright_mean", "bright_std", "pitch_mean", "pitch_std"]
    print(f"train/val split: {len(s_train)} train / {len(s_val)} held-out validation transitions")
    print(f"{'state dim':<12} {'val MSE':>10} {'val R^2':>10}")
    for name, mse, r2 in zip(state_dim_names, mse_per_dim, r2_per_dim):
        print(f"{name:<12} {mse:>10.4f} {r2:>10.4f}")

    save_predictor_weights(model, "data/sequencer_trajectory_model_weights.npz", action_channels=channels)
    print("saved trained predictor weights to data/sequencer_trajectory_model_weights.npz")
