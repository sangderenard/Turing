"""An element read of ``array(typecode)`` has the typecode's layout.

``a = array("Q"); a.append(n); a[0]``: the typecode states the array's one
column (``Q`` is uint64), whatever Python ints the appends carry.  The local
allocation posted no element type, so ``a[0]`` reached its region with no
settled dtype and lowering refused.  The allocation now carries the
typecode's column dtype (``sequence_column_dtypes``) and the element read
takes its class from it, on a local array and on a record field read in a
method alike.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"

LOCAL = '''
from array import array


def root(n: int) -> int:
    values = array("{code}")
    values.append(n)
    return values[0]
'''

FIELD = '''
from array import array


class Store:
    def __init__(self):
        self.count = array("{code}")

    def put(self, n: int) -> int:
        self.count.append(n)
        return self.count[0]


def root(n: int) -> int:
    store = Store()
    return store.put(n)
'''


def _lowered(source, name):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name=name,
        extraction_contract=str(CONTRACT), progress=lambda _message: None,
    )
    root = module.functions[exports[0]]
    outputs = [
        value
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret" and not instruction.attributes.get("fault_exit")
        for value in instruction.args
    ]
    return module, exports[0], dict(root.metadata["parameter_names"]), outputs


def _run(module, export, parameters, outputs, lane, tmp_path, n):
    feeds = {parameters["n"]: np.array([n], dtype=np.int64)}
    if lane == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c

        artifact = emit_ssa_to_c(module, export)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / "native")
        execution = artifact.prepare_execution(feeds).run()
    else:
        from src.compiler.ssa_llvm_backend import (
            compile_artifact, emit_ssa_function_to_llvm,
            prepare_artifact_execution,
        )

        emitted = emit_ssa_function_to_llvm(module, export)
        assert not emitted.shortfalls, emitted.shortfalls
        artifact = compile_artifact(emitted, directory=tmp_path / "llvm")
        execution = prepare_artifact_execution(artifact, feeds).run()
    return int(execution.buffers[int(outputs[0].id)].reshape(-1)[0])


@pytest.mark.parametrize("code,dtype", [("B", "uint8"), ("I", "uint32"), ("Q", "uint64")])
def test_local_array_element_read_has_typecode_layout(tmp_path, code, dtype):
    module, export, parameters, outputs = _lowered(
        LOCAL.format(code=code), f"array_local_read_{code}",
    )
    assert len(outputs) == 1 and outputs[0].dtype == dtype
    for lane in ("c", "llvm"):
        assert _run(module, export, parameters, outputs, lane, tmp_path / lane, 5) == 5


@pytest.mark.parametrize("code,dtype", [("I", "uint32"), ("Q", "uint64")])
def test_array_field_element_read_in_method(tmp_path, code, dtype):
    module, export, parameters, outputs = _lowered(
        FIELD.format(code=code), f"array_field_read_{code}",
    )
    assert len(outputs) == 1
    for lane in ("c", "llvm"):
        assert _run(module, export, parameters, outputs, lane, tmp_path / lane, 7) == 7
