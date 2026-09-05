"""One non-deployment program compiled through four native backend lanes."""

from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

from src.common.tensors.accelerator_backends.c_backend import CTensor
from src.common.tensors.accelerator_backends.c_primitive_program import (
    prepare_fused_program,
)
from src.common.tensors.fused_ir import ordered_feed_ids
from src.common.tensors.numpy_backend import NumPyTensorOperations  # noqa: F401
from src.compiler.backend_sources import normalized_program
from src.compiler.fortran_fidelity import verify_fortran_module
from src.compiler.fused_program_python_backend import compile_single_region_python
from src.compiler.fused_program_wasm_backend import emit_wasm_module
from src.compiler.native_sorting_process_learner import capture_sorting_process
from src.compiler.precompile_to_ssa import lower_fused_program_to_ssa
from src.compiler.ssa_fortran_backend import (
    emit_module as emit_fortran,
    fortran_compiler,
)
from src.compiler.ssa_llvm_backend import (
    compile_artifact,
    emit_ssa_function_to_llvm,
    prepare_artifact_execution,
)
from src.compiler.wasm_fidelity import node_runtime, verify_wasm_module
from src.transmogrifier.ssa import IRModule


EXAMPLE = Path(__file__).parents[1] / "examples" / "learnable_sort.py"


def _captured_case():
    captured = capture_sorting_process(EXAMPLE, batch_size=4, seed=11)
    program = normalized_program(captured.cycle.forward_program)
    feed_ids = ordered_feed_ids(program)
    feeds = {
        value_id: np.asarray(
            captured.cycle.feed_values[value_id].tolist(), dtype=np.float64,
        )
        for value_id in feed_ids
    }
    reference = compile_single_region_python(
        program,
        {value_id: f"feed{index}" for index, value_id in enumerate(feed_ids)},
        dialect="numpy",
        function_name="learnable_sort_reference",
    ).callable
    raw = reference(*(feeds[value_id] for value_id in feed_ids))
    expected = tuple(raw) if len(program.outputs) > 1 else (raw,)
    function, shortfalls = lower_fused_program_to_ssa(
        program, function_name="learnable_sort_forward",
    )
    assert not shortfalls
    module = IRModule({function.name: function})
    outputs = next(
        list(instruction.args)
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret"
    )
    return program, feeds, expected, function, module, outputs


def test_learnable_sort_emits_complete_c_llvm_fortran_and_wasm_programs():
    program, feeds, expected, function, module, outputs = _captured_case()
    assert len(program.steps) == 136
    assert len(program.outputs) == 8

    prepared = prepare_fused_program(program, {
        value_id: CTensor.from_list(value.reshape(-1).tolist(), value.shape)
        for value_id, value in feeds.items()
    })
    prepared.execute()
    for name, wanted in zip(program.outputs, expected):
        np.testing.assert_allclose(prepared.outputs[name].tolist(), wanted)

    fortran = emit_fortran(
        module,
        name="learnable_sort_fortran",
        outputs={function.name: outputs},
    )
    llvm = emit_ssa_function_to_llvm(module, function.name)
    wasm = emit_wasm_module(program, name=function.name)
    assert fortran.complete, [item.format() for item in fortran.shortfalls]
    assert llvm.complete, [item.reason for item in llvm.shortfalls]
    assert wasm.complete, wasm.shortfall_report()
    assert wasm.binary


@pytest.mark.skipif(
    fortran_compiler() is None
    or node_runtime() is None
    or find_spec("ziglang") is None,
    reason="full backend fidelity needs Fortran, Node, and ziglang toolchains",
)
def test_learnable_sort_compiled_artifacts_execute_with_numpy_parity(tmp_path):
    program, feeds, expected, function, module, outputs = _captured_case()
    case = (("captured", feeds),)

    fortran = emit_fortran(
        module,
        name="learnable_sort_fortran",
        outputs={function.name: outputs},
    )
    assert verify_fortran_module(
        fortran,
        program,
        feeds,
        tmp_path / "fortran",
        entrypoint=function.name,
        cases=case,
    )["passed"]

    llvm = compile_artifact(
        emit_ssa_function_to_llvm(module, function.name),
        directory=tmp_path / "llvm",
    )
    execution = prepare_artifact_execution(llvm, feeds).run()
    for value_id, wanted in zip(program.outputs.values(), expected):
        np.testing.assert_allclose(execution.buffers[value_id], wanted)

    wasm = emit_wasm_module(program, name=function.name)
    assert verify_wasm_module(
        wasm,
        program,
        feeds,
        tmp_path / "wasm",
        entrypoint=function.name,
        cases=case,
    )["passed"]
