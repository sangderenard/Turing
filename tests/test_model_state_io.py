import numpy as np
from pathlib import Path

from src.common.tensors import AbstractTensor
from src.common.tensors.numpy_backend import NumPyTensorOperations
from src.common.tensors.abstract_nn.core import Model, Linear
from src.common.tensors.abstract_nn.activations import Identity

def _to_numpy(t):
    np_backend = AbstractTensor.get_tensor(cls=NumPyTensorOperations)
    return t.to_backend(np_backend).numpy()


def test_model_state_roundtrip(tmp_path: Path):
    like = AbstractTensor.get_tensor(cls=NumPyTensorOperations)
    layer = Linear(2, 2, like=like)
    model = Model([layer], activations=[Identity()])

    inp = AbstractTensor.get_tensor(np.array([[1.0, 2.0]], dtype=np.float32))
    out_initial = model.forward(inp)

    state = model.state_dict()

    # Zero the parameters and ensure output changes
    for p in model.parameters():
        p.data[...] = 0
    out_zero = model.forward(inp)
    assert not np.allclose(_to_numpy(out_initial), _to_numpy(out_zero))

    # Restore from the in-memory state
    model.load_state_dict(state)
    out_restored = model.forward(inp)
    assert np.allclose(_to_numpy(out_initial), _to_numpy(out_restored))

    # Persist to disk and load again
    file_path = tmp_path / "model_state.json"
    model.save_state(file_path)

    for p in model.parameters():
        p.data[...] = 0
    model.load_state(file_path)
    out_file_restored = model.forward(inp)
    assert np.allclose(_to_numpy(out_initial), _to_numpy(out_file_restored))
