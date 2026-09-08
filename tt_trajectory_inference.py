# tt_trajectory_inference.py
import numpy as np
import ttnn


class ActionChannelMismatchError(ValueError):
    """Raised when the CV channel order the loaded predictor weights were
    trained with doesn't match the backend's current channel order. The
    action input is per-channel and has no labels attached to it at
    inference time -- if the order silently disagreed with the backend's
    actual channels() ordering, actions would be interpreted against the
    wrong physical CV channel with no error at all, just a
    working-looking-but-wrong dynamics model."""


class TrajectoryTTInferenceEngine:
    def __init__(self, weights_path: str, device_id: int = 0, expected_channels: list[str] | None = None):
        weights = np.load(weights_path)

        if expected_channels is not None:
            trained_channels = weights["action_channels"].tolist() if "action_channels" in weights.files else None
            if trained_channels is None:
                raise ActionChannelMismatchError(
                    f"{weights_path} was saved without action-channel-order provenance "
                    "(trajectory_model.py's save_predictor_weights was called without "
                    f"`action_channels`), but the caller expects channel order {expected_channels!r} -- "
                    "retrain and save with `action_channels` so this can be verified."
                )
            if trained_channels != list(expected_channels):
                raise ActionChannelMismatchError(
                    f"{weights_path} was trained with action channel order "
                    f"{trained_channels!r}, but the backend's current channel "
                    f"order is {list(expected_channels)!r}. Applying actions in "
                    "that order would silently drive the wrong CV channel's dynamics."
                )

        self._device = ttnn.open_device(device_id=device_id)

        try:
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
        except Exception:
            # If any weight fails to stage (malformed/missing key in the
            # .npz), __init__ never returns an object -- so the caller
            # never gets a handle on which to call close(). Close the
            # device here ourselves before re-raising, so a construction
            # failure never leaks an open device on this shared box.
            ttnn.close_device(self._device)
            raise

    def predict_next_state(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        import torch
        x = np.concatenate([states, actions], axis=-1)
        x_device = ttnn.from_torch(
            torch.tensor(x, dtype=torch.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self._device,
        )
        h = ttnn.relu(ttnn.linear(x_device, self._W1, bias=self._b1))
        h = ttnn.relu(ttnn.linear(h, self._W2, bias=self._b2))
        out = ttnn.linear(h, self._W3, bias=self._b3)
        return ttnn.to_torch(out).float().numpy()

    def close(self) -> None:
        ttnn.close_device(self._device)
