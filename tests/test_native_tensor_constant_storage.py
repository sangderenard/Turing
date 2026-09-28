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


def test_full_value_trace_prints_every_shaped_result_lane(tmp_path, monkeypatch):
    value = SSAValue(0, 'float64', (2, 3))
    root = Function('root', [], {'entry': BasicBlock('entry', [
        Instr('Const', [], value, attributes={'value': -2.5}),
        Instr('Ret', [value], None),
    ])})
    artifact = emit_ssa_module_to_c(
        IRModule({'root': root}), 'root', trace=True,
        trace_full_values=True,
    )
    assert artifact.complete, artifact.shortfalls
    assert 'turing_trace_arr_full(' in artifact.source
    artifact.compile(tmp_path / 'native-trace', optimization='O0')
    trace_path = tmp_path / 'values.log'
    monkeypatch.setenv('TURING_TRACE_FILE', str(trace_path))
    execution = artifact.prepare_execution({0: np.full((2, 3), np.nan)})
    execution.run()

    trace = trace_path.read_text(encoding='utf-8')
    assert '%t0 = Const' in trace
    assert '[0]=-2.5' in trace
    assert '[5]=-2.5' in trace
