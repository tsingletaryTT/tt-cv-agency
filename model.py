import numpy as np
import torch
import torch.nn as nn


class InverseCVModel(nn.Module):
    def __init__(self, n_features: int = 3, n_channels: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, n_channels)

    def forward(self, target_features: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(target_features))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def train_model(cv_array: np.ndarray, feature_array: np.ndarray, epochs: int = 200, lr: float = 1e-3) -> InverseCVModel:
    model = InverseCVModel(n_features=feature_array.shape[1], n_channels=cv_array.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    x = torch.tensor(feature_array, dtype=torch.float32)
    y = torch.tensor(cv_array, dtype=torch.float32)

    for _ in range(epochs):
        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_fn(prediction, y)
        loss.backward()
        optimizer.step()

    return model


def train_val_split(
    cv_array: np.ndarray, feature_array: np.ndarray, val_fraction: float = 0.2, seed: int = 0
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Split (cv, features) sample pairs into a train set and a held-out
    validation set, shuffled by `seed` so the split is reproducible.
    Returns ((cv_train, feature_train), (cv_val, feature_val)).

    The plan's spec called for a held-out validation split reporting MSE;
    the original training script only ever reported one aggregate MSE
    measured on the same data it trained on, which hid that one output
    dimension (vco_fm) had learned almost nothing (R^2 ~ 0.07) while
    another (vca_level) had learned the mapping well (R^2 ~ 0.89) -- an
    aggregate in-sample MSE cannot surface either fact.
    """
    n = len(cv_array)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n)
    n_val = int(round(n * val_fraction))
    val_idx = shuffled[:n_val]
    train_idx = shuffled[n_val:]
    return (cv_array[train_idx], feature_array[train_idx]), (cv_array[val_idx], feature_array[val_idx])


def evaluate_per_dimension(
    model: InverseCVModel, cv_val: np.ndarray, feature_val: np.ndarray, baseline_mean: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-output-dimension MSE and R^2 (against a predict-the-mean
    baseline) on a held-out (cv_val, feature_val) set. `baseline_mean`
    should be the *training* set's per-dimension CV mean (not the
    validation set's own mean) so the baseline isn't computed from the same
    data it's being judged against -- an R^2 of 0.0 means "no better than
    always predicting the training-set average CV setting for this
    channel," which is the honest bar a control-loop-worthy model needs to
    clear per channel, not just on average across all three.
    """
    model.eval()
    with torch.no_grad():
        prediction = model(torch.tensor(feature_val, dtype=torch.float32)).numpy()

    mse_per_dim = np.mean((prediction - cv_val) ** 2, axis=0)

    ss_res = np.sum((prediction - cv_val) ** 2, axis=0)
    ss_tot = np.sum((cv_val - baseline_mean) ** 2, axis=0)
    # Guard against a degenerate all-identical validation column (ss_tot=0)
    # rather than dividing by zero -- shouldn't happen with real sweep data,
    # but a NaN/inf R^2 would be a confusing way to find that out.
    r2_per_dim = np.where(ss_tot > 0, 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, 1.0), np.nan)

    return mse_per_dim, r2_per_dim


def save_weights(model: InverseCVModel, path: str, channels: list[str] | None = None) -> None:
    # `channels` records the CV channel order (from backend.channels()) this
    # model's outputs were trained against -- e.g. ["vco_freq", "vco_fm",
    # "vca_level"]. It's optional (so existing callers/tests that only care
    # about the weight arrays keep working unchanged), but any real training
    # run should pass it: without it, nothing stops the model's output
    # vector being applied to CV channels in the wrong order at inference
    # time, a silent-wrong-channel failure mode that would look like a
    # working-but-wrong control loop rather than an error.
    extra = {"channels": np.array(channels)} if channels is not None else {}
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
    data = np.load("data/sweep_dataset.npz")
    channels = data["channels"].tolist()

    (cv_train, feat_train), (cv_val, feat_val) = train_val_split(
        data["cv"], data["features"], val_fraction=0.2, seed=0
    )
    model = train_model(cv_train, feat_train, epochs=500)

    train_mean = cv_train.mean(axis=0)
    mse_per_dim, r2_per_dim = evaluate_per_dimension(model, cv_val, feat_val, baseline_mean=train_mean)

    print(f"train/val split: {len(cv_train)} train / {len(cv_val)} held-out validation samples")
    print(f"{'channel':<12} {'val MSE':>10} {'val R^2':>10}")
    for ch, mse, r2 in zip(channels, mse_per_dim, r2_per_dim):
        print(f"{ch:<12} {mse:>10.4f} {r2:>10.4f}")

    save_weights(model, "data/model_weights.npz", channels=channels)
    print("saved trained weights to data/model_weights.npz")
