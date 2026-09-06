import numpy as np
import torch
import torch.nn as nn


class InverseCVModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, target_features: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(target_features))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def train_model(cv_array: np.ndarray, feature_array: np.ndarray, epochs: int = 200, lr: float = 1e-3) -> InverseCVModel:
    model = InverseCVModel()
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


def save_weights(model: InverseCVModel, path: str) -> None:
    np.savez(
        path,
        W1=model.fc1.weight.detach().numpy().T,
        b1=model.fc1.bias.detach().numpy(),
        W2=model.fc2.weight.detach().numpy().T,
        b2=model.fc2.bias.detach().numpy(),
        W3=model.fc3.weight.detach().numpy().T,
        b3=model.fc3.bias.detach().numpy(),
    )


if __name__ == "__main__":
    data = np.load("data/sweep_dataset.npz")
    model = train_model(data["cv"], data["features"], epochs=500)
    save_weights(model, "data/model_weights.npz")
    print("saved trained weights to data/model_weights.npz")
