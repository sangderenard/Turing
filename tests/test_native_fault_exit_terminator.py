"""A checked index's fault exit is a ``Trap``, not a second ``Ret``.

``count[1]`` on a grown list is a checked read.  Its out-of-range edge used
to leave through a zero-valued ``Ret`` marked ``fault_exit``; the LLVM lane
named the output buffer after the FIRST ``Ret`` and the result landed under
the fault exit's zero constant, while the real return id had no buffer.  The
fault exit now publishes nothing: one ``Ret`` names the outputs, and the
out-of-range read still raises ``IndexError`` on both lanes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"

SOURCE = '''
def root(n: int) -> int:
    count = []
    count.append(n)
    count.append(n + 1)
    return count[1] + len(count)
'''

OUT_OF_RANGE = '''
def root(n: int) -> int:
    count = []
    count.append(n)
    return count[n]
'''


def _lowered(source):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="fault_exit_terminator",
        extraction_contract=str(CONTRACT), progress=lambda _message: None,
    )
    return module, exports[0]


def _returns(module, export):
    return [
        instruction
        for block in module.functions[export].blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret"
    ]


def _run(tmp_path, lane, source, n):
    module, export = _lowered(source)
    (ret,) = _returns(module, export)
    (output,) = (int(value.id) for value in ret.args)
    parameters = dict(module.functions[export].metadata["parameter_names"])
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
        artifact = compile_artifact(emitted, directory=tmp_path)
        execution = prepare_artifact_execution(artifact, feeds).run()
    return int(execution.buffers[output].reshape(-1)[0])


def test_grown_list_read_has_one_ret_and_a_trap_fault_exit():
    module, export = _lowered(SOURCE)
    returns = _returns(module, export)
    assert len(returns) == 1 and len(returns[0].args) == 1, returns
    assert not returns[0].attributes.get("fault_exit")
    traps = [
        instruction
        for block in module.functions[export].blocks.values()
        for instruction in block.instrs
        if instruction.op == "Trap"
    ]
    assert traps and all(trap.attributes.get("fault_exit") for trap in traps)
    assert all(not trap.args for trap in traps)


@pytest.mark.parametrize("lane", ["c", "llvm"])
@pytest.mark.parametrize("n", [0, 5])
def test_result_lands_under_the_one_return(tmp_path, lane, n):
    assert _run(tmp_path, lane, SOURCE, n) == (n + 1) + 2


@pytest.mark.parametrize("lane", ["c", "llvm"])
def test_out_of_range_read_still_raises(tmp_path, lane):
    from src.compiler.native_faults import NativeIndexError

    assert _run(tmp_path, lane, OUT_OF_RANGE, 0) == 0
    with pytest.raises(NativeIndexError):
        _run(tmp_path, lane, OUT_OF_RANGE, 1)
