import numpy as np

from src.common.dt_system.engine_api import DtCompatibleEngine, ComputeShaderSpec
from src.common.dt_system.dt_scaler import Metrics


class DummyEngine(DtCompatibleEngine):
    """Minimal engine exposing a compute shader spec."""

    def __init__(self):
        super().__init__()
        self.buf_in = np.zeros(4, dtype=np.float32)
        self.buf_out = np.zeros(4, dtype=np.float32)

    def step(self, dt: float, state, state_table):
        return True, Metrics(), state

    def get_state(self, state=None):
        return state

    def get_compute_shaders(self):
        src = "layout(local_size_x = 1) in; void main() { }"
        return [
            ComputeShaderSpec(
                name="pass0",
                source=src,
                buffers={"in": self.buf_in, "out": self.buf_out},
                next=["pass1"],
            )
        ]


def test_engine_provides_compute_spec():
    eng = DummyEngine()
    specs = eng.get_compute_shaders()
    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == "pass0"
    assert spec.next == ["pass1"]
    assert spec.buffers["in"].shape == (4,)
    assert spec.buffers["in"] is not None
    # Ensure buffers are passed through without conversion
    assert spec.buffers["in"] is eng.buf_in
