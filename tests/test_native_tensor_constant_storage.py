"""A shaped scalar Const denotes filled tensor storage, not a pointer literal."""

import numpy as np
import pytest

from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.transmogrifier.ssa import BasicBlock, Function, Instr, IRModule, SSAValue


@pytest.mark.parametrize('payload', [0.0, -2.5])
def test_native_scalar_payload_fills_shaped_constant(tmp_path, payload):
    value = SSAValue(0, 'float64', (2, 3))
    root = Function('root', [], {'entry': BasicBlock('entry', [
        Instr('Const', [], value, attributes={'value': payload}),
        Instr('Ret', [value], None),
    ])})
    artifact = emit_ssa_module_to_c(IRModule({'root': root}), 'root')
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'native', optimization='O0')
    execution = artifact.prepare_execution({0: np.full((2, 3), np.nan)})
    execution.run()
    np.testing.assert_array_equal(execution.buffers[0], np.full((2, 3), payload))
