import numpy as np
import torch
from model import InverseCVModel, save_weights
from tt_inference import TTInferenceEngine


def test_tt_inference_matches_pytorch_reference(tmp_path):
    torch.manual_seed(0)
    model = InverseCVModel()
    weights_path = str(tmp_path / "weights.npz")
    save_weights(model, weights_path)

    target = np.array([0.3, 0.6, 0.9], dtype=np.float32)

    with torch.no_grad():
        reference = model(torch.tensor(target, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()

    engine = TTInferenceEngine(weights_path=weights_path)
    try:
        result = engine.predict_cv(target)
    finally:
        engine.close()

    assert result.shape == (3,)
    # bfloat16 on-device precision -- allow a generous but real tolerance,
    # verified in exploration to be within ~0.03 absolute for this model size.
    assert np.allclose(result, reference, atol=0.1)
