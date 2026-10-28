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


# The token-orchard store shape: typed array fields grown by one method and
# read by another (a uint64 begin offset plus an int index, a uint32 count
# compared with a list length).
STORE = '''
from array import array


class Store:
    def __init__(self):
        self.epoch = 0
        self.domain = array("B")
        self.begin = array("Q")
        self.count = array("I")
        self.atoms = array("I")

    def add(self, domain: int, values: list[int], b: int, e: int) -> int:
        row = len(self.domain)
        self.epoch += 1
        self.domain.append(domain)
        self.begin.append(len(self.atoms))
        self.count.append(e - b)
        i = b
        while i < e:
            self.atoms.append(values[i])
            i += 1
        return row

    def matches(self, row: int, values: list[int]) -> bool:
        count = self.count[row]
        if count != len(values):
            return False
        source = self.begin[row]
        i = 0
        while i < count:
            if self.atoms[source + i] != values[i]:
                return False
            i += 1
        return True


def root(n: int) -> int:
    s = Store()
    i = 0
    total = 0
    while i < n:
        row = s.add(1, [65 + i, 128578, 7], 0, 3)
        if s.matches(row, [65 + i, 128578, 7]):
            total += 1
        total += row
        i += 1
    return total + s.epoch
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


def test_array_store_fields_read_across_methods(tmp_path):
    module, export, parameters, outputs = _lowered(STORE, "array_store_reads")
    assert len(outputs) == 1
    for lane in ("c", "llvm"):
        for n, expected in ((0, 0), (1, 2), (17, 170)):
            assert _run(
                module, export, parameters, outputs, lane, tmp_path / f"{lane}{n}", n,
            ) == expected
