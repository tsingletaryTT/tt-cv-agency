import numpy as np
import ttnn


class ChannelMismatchError(ValueError):
    """Raised when the CV channel order the loaded weights were trained
    with doesn't match the backend's current channel order. The model's
    output vector is a plain array with no channel labels attached to it at
    inference time -- if the order silently disagreed with the backend's
    actual channels() ordering, predict_cv()'s output would get applied to
    the wrong physical CV channel with no error at all, just a
    working-looking-but-wrong control loop."""


class TTInferenceEngine:
    def __init__(self, weights_path: str, device_id: int = 0, expected_channels: list[str] | None = None):
        weights = np.load(weights_path)

        if expected_channels is not None:
            trained_channels = weights["channels"].tolist() if "channels" in weights.files else None
            if trained_channels is None:
                raise ChannelMismatchError(
                    f"{weights_path} was saved without channel-order provenance "
                    "(model.py's save_weights was called without `channels`), "
                    f"but the caller expects channel order {expected_channels!r} -- "
                    "retrain and save with `channels` so this can be verified."
                )
            if trained_channels != list(expected_channels):
                raise ChannelMismatchError(
                    f"{weights_path} was trained with channel order "
                    f"{trained_channels!r}, but the backend's current channel "
                    f"order is {list(expected_channels)!r}. Applying this "
                    "model's output to the backend's channels in that order "
                    "would silently drive the wrong CV channel."
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
