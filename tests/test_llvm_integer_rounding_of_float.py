"""``round``/``math.floor``/``math.ceil``/``math.trunc`` of a float, declared
int, in the LLVM single-function lane.

The declared i64 was the compute type there: the double operand was
truncated by ``fptosi`` and the rounding on the int was ``add x, 0``, so
round(2.7) gave 2 and math.floor(-1.5) gave -1.  The rounding now runs in
the double column (``llvm.roundeven/floor/ceil/trunc``) and only its
integral result converts (corpus c920, c921).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"

VALUES = (2.7, 0.6, -1.5, 2.5, -2.5, -2.7, 0.5, -0.5, 1.5, -0.6, 4503599627370497.0)

SOURCE = '''
import math


def root(x: float):
    return OP(x * 1.0)
'''


@pytest.mark.parametrize("spelling", ["round", "math.floor", "math.ceil", "math.trunc", "int"])
def test_integer_rounding_of_float_matches_python(tmp_path, spelling):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    module, _outputs, _exports = lower_ast_source_to_ssa(
        SOURCE.replace("OP", spelling), "root", name="llvm_int_rounding",
        extraction_contract=str(CONTRACT), progress=lambda _message: None,
    )
    region = next(name for name in module.functions if "planned_region" in name)
    function = module.functions[region]
    emitted = emit_ssa_function_to_llvm(module, region)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tmp_path)
    (argument,) = function.args
    result = next(
        instruction.args[0]
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret"
    )
    python = eval(spelling, {"math": math})
    for x in VALUES:
        execution = prepare_artifact_execution(
            artifact, {int(argument.id): np.array([x])}
        ).run()
        got = int(np.asarray(execution.buffers[int(result.id)]).reshape(-1)[0])
        assert got == python(x), (spelling, x, got)
