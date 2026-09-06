import numpy as np
import ttnn


class TTInferenceEngine:
    def __init__(self, weights_path: str, device_id: int = 0):
        weights = np.load(weights_path)
        self._device = ttnn.open_device(device_id=device_id)

        def to_device(array):
            import torch
            return ttnn.from_torch(
                torch.tensor(array, dtype=torch.float32),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self._device,
            )

        self._W1 = to_device(weights["W1"])
        self._b1 = to_device(weights["b1"].reshape(1, -1))
        self._W2 = to_device(weights["W2"])
        self._b2 = to_device(weights["b2"].reshape(1, -1))
        self._W3 = to_device(weights["W3"])
        self._b3 = to_device(weights["b3"].reshape(1, -1))

    def predict_cv(self, target_features: np.ndarray) -> np.ndarray:
        import torch
        x = ttnn.from_torch(
            torch.tensor(target_features, dtype=torch.float32).reshape(1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self._device,
        )
        h = ttnn.relu(ttnn.linear(x, self._W1, bias=self._b1))
        h = ttnn.relu(ttnn.linear(h, self._W2, bias=self._b2))
        out = ttnn.linear(h, self._W3, bias=self._b3)
        return ttnn.to_torch(out).float().numpy().reshape(-1)

    def close(self) -> None:
        ttnn.close_device(self._device)
