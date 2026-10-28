"""``-> int`` is a class, not a width: a call result carries the callee's.

``def probe(self, row) -> int: return self.count[row]`` over ``array("I")``
returns uint32.  The caller's call result was stamped int64 from the
class-only annotation receipt before the callee settled its return, so
``1 + s.probe(0)`` refused with "physical concordance disagreement
'uint32' versus 'int64'" (string core ``token_domain``).  The call result
now carries the width the callee's own return record settles, and the
arithmetic proceeds through the existing exact mixed-width rules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"

METHOD = '''
from array import array


class Store:
    def __init__(self):
        self.count = array("{code}")

    def add(self, n: int) -> int:
        row = len(self.count)
        self.count.append(n)
        return row

    def probe(self, row: int) -> int:
        return self.count[row]


def root(n: int) -> int:
    s = Store()
    s.add(n)
    return {expression}
'''

FUNCTION = '''
from array import array


def head(n: int) -> int:
    a = array("I")
    a.append(n)
    return a[0]


def root(n: int) -> int:
    return 1 + head(n)
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
    return module, exports[0], root, dict(root.metadata["parameter_names"]), outputs


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


def _call_result_dtypes(root, callee_fragment):
    return {
        str(instruction.res.dtype)
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call" and instruction.res is not None
        and callee_fragment in str(instruction.attributes.get("callee"))
    }


@pytest.mark.parametrize("code,dtype,expression,cases", [
    ("I", "uint32", "1 + s.probe(0)", ((0, 1), (5, 6))),
    ("B", "uint8", "s.probe(0) * 1000 - n", ((3, 2997), (200, 199800))),
    ("Q", "uint64", "s.probe(0) - 3", ((5, 2), (1, -2))),
])
def test_method_int_return_carries_callee_width(tmp_path, code, dtype, expression, cases):
    module, export, root, parameters, outputs = _lowered(
        METHOD.format(code=code, expression=expression), f"method_ret_{code}",
    )
    assert _call_result_dtypes(root, "probe") == {dtype}
    for lane in ("c", "llvm"):
        for n, expected in cases:
            assert _run(
                module, export, parameters, outputs, lane, tmp_path / f"{lane}{n}", n,
            ) == expected


def test_function_int_return_carries_callee_width(tmp_path):
    module, export, root, parameters, outputs = _lowered(FUNCTION, "function_ret_I")
    for lane in ("c", "llvm"):
        for n, expected in ((0, 1), (5, 6)):
            assert _run(
                module, export, parameters, outputs, lane, tmp_path / f"{lane}{n}", n,
            ) == expected
