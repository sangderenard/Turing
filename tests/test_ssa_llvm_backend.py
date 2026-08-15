"""The LLVM lane, end to end, on a full real pythonic program.

The input is ``examples/xor_project/train_xor.py`` -- the ordinary
abstract_nn training program (Model/Linear/Adam classes, the actual loop),
exactly as a user wrote it. It travels the compiler's own entries:
source -> dual IR (``compile_ast_aot``, precompile-only, whole-program) ->
repository SSA (``lower_precompile_and_control_to_ssa`` with the C
computational-core reference) -> LLVM (the likeness-table emitter) -> native
artifact (Zig's clang, ahead of time). Every stage's shortfall census must be
empty or name itself; nothing is bypassed and nothing synthetic stands in
for the program.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from src.common.tensors.accelerator_backends.aot_compile import compile_ast_aot
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.precompile_to_ssa import lower_precompile_and_control_to_ssa
from src.compiler.ssa_llvm_backend import (
    compile_artifact,
    emit_ssa_function_to_llvm,
    prepare_artifact_execution,
    with_native_sgd_loop,
)
from src.transmogrifier.ssa import BasicBlock, Function, IRModule, Instr, SSAValue


def _native_scalar_buffers(artifact, values):
    buffers = {}
    for value_id, shape in zip(artifact.buffer_order, artifact.buffer_shapes):
        value = values.get(value_id, 0)
        if value_id in values and isinstance(value, bool):
            buffers[value_id] = ctypes.c_bool(value)
        else:
            buffers[value_id] = ctypes.c_int32(int(value))
    pointers = (ctypes.c_void_p * len(artifact.buffer_order))(*(
        ctypes.cast(ctypes.pointer(buffers[value_id]), ctypes.c_void_p)
        for value_id in artifact.buffer_order
    ))
    extents = (ctypes.c_int32 * len(artifact.extent_order))()
    return buffers, pointers, extents


def test_multiblock_conditional_phi_executes_natively(tmp_path):
    predicate = SSAValue(0, "bool")
    true_value = SSAValue(1, "int")
    false_value = SSAValue(2, "int")
    selected = SSAValue(3, "int")
    function = Function("select_value", [predicate, true_value, false_value], {
        "entry": BasicBlock("entry", [Instr(
            "CondBr", [predicate], None,
            attributes={"true_target": "if_true", "false_target": "if_false"},
        )], ["if_true", "if_false"]),
        "if_true": BasicBlock("if_true", [
            Instr("Br", [], None, attributes={"target": "merge"}),
        ], ["merge"]),
        "if_false": BasicBlock("if_false", [
            Instr("Br", [], None, attributes={"target": "merge"}),
        ], ["merge"]),
        "merge": BasicBlock("merge", [
            Instr(
                "Phi", [true_value, false_value], selected,
                attributes={"incoming_blocks": ("if_true", "if_false")},
            ),
            Instr("Ret", [selected], None),
        ]),
    })
    artifact = emit_ssa_function_to_llvm(
        IRModule({function.name: function}), function.name,
    )
    assert artifact.shortfalls == ()
    native = compile_artifact(artifact, directory=tmp_path / "native_branch")
    buffers, pointers, extents = _native_scalar_buffers(
        native, {0: True, 1: 17, 2: -4},
    )
    entry = native.entry()
    entry(pointers, extents)
    assert buffers[3].value == 17
    buffers[0].value = False
    entry(pointers, extents)
    assert buffers[3].value == -4


def test_descending_loop_phi_executes_natively(tmp_path):
    trip_count = SSAValue(0, "int")
    zero = SSAValue(1, "int")
    negative_one = SSAValue(2, "int")
    current = SSAValue(3, "int")
    updated = SSAValue(4, "int")
    active = SSAValue(5, "bool")
    function = Function("descending_loop", [trip_count], {
        "entry": BasicBlock("entry", [
            Instr("Const", [], zero, attributes={"value": 0}),
            Instr("Const", [], negative_one, attributes={"value": -1}),
            Instr("Br", [], None, attributes={"target": "loop_header"}),
        ], ["loop_header"]),
        "loop_header": BasicBlock("loop_header", [
            Instr(
                "Phi", [trip_count, updated], current,
                attributes={"incoming_blocks": ("entry", "loop_latch")},
            ),
            Instr("Gt", [current, zero], active),
            Instr(
                "CondBr", [active], None,
                attributes={"true_target": "loop_body", "false_target": "loop_exit"},
            ),
        ], ["loop_body", "loop_exit"]),
        "loop_body": BasicBlock("loop_body", [
            Instr("Br", [], None, attributes={"target": "loop_latch"}),
        ], ["loop_latch"]),
        "loop_latch": BasicBlock("loop_latch", [
            Instr("Add", [current, negative_one], updated),
            Instr("Br", [], None, attributes={"target": "loop_header"}),
        ], ["loop_header"]),
        "loop_exit": BasicBlock("loop_exit", [Instr("Ret", [current], None)]),
    })
    artifact = emit_ssa_function_to_llvm(
        IRModule({function.name: function}), function.name,
    )
    assert artifact.shortfalls == ()
    native = compile_artifact(artifact, directory=tmp_path / "native_loop")
    buffers, pointers, extents = _native_scalar_buffers(native, {0: 7})
    native.entry()(pointers, extents)
    assert buffers[3].value == 0


def test_native_sgd_wrapper_repeats_motion_and_updates_parameter(tmp_path):
    parameter = SSAValue(0, "float64")
    target = SSAValue(1, "float64")
    gradient = SSAValue(2, "float64")
    loss = SSAValue(3, "float64")
    function = Function("scalar_training_motion", [parameter, target], {
        "entry": BasicBlock("entry", [
            Instr("Sub", [parameter, target], gradient),
            Instr("Mul", [gradient, gradient], loss),
            Instr("Ret", [loss, gradient], None),
        ]),
    })
    motion = emit_ssa_function_to_llvm(
        IRModule({function.name: function}), function.name,
    )
    assert motion.shortfalls == ()
    loop = with_native_sgd_loop(
        motion,
        parameter_gradient_pairs=((parameter.id, gradient.id),),
    )
    native = compile_artifact(loop, directory=tmp_path / "native_sgd_loop")
    execution = prepare_artifact_execution(native, {
        parameter.id: 4.0,
        target.id: 1.0,
        native.training_steps_value_id: 6,
        native.learning_rate_value_id: 0.25,
    })
    execution.run()

    assert float(execution.buffers[parameter.id]) == pytest.approx(
        1.0 + 3.0 * (0.75 ** 6)
    )
    assert float(execution.buffers[gradient.id]) == pytest.approx(
        3.0 * (0.75 ** 5)
    )
    assert float(execution.buffers[loss.id]) == pytest.approx(
        (3.0 * (0.75 ** 5)) ** 2
    )
    settled = float(execution.buffers[parameter.id])
    execution.buffers[native.training_steps_value_id][...] = 0
    execution.run()
    assert float(execution.buffers[parameter.id]) == settled


def test_executor_measures_dynamic_shape_vector_from_feed():
    tensor = SSAValue(10, "float64", ("rows", "cols"))
    shape = SSAValue(11, "int32", (2,))
    function = Function("dynamic_shape", [tensor], {
        "entry": BasicBlock("entry", [
            Instr(
                "extent", [tensor], shape,
                attributes={"extent_kind": "shape"},
            ),
            Instr("Ret", [], None),
        ]),
    })
    artifact = emit_ssa_function_to_llvm(
        IRModule({function.name: function}), function.name,
    )
    assert artifact.shortfalls == ()
    assert artifact.extent_order == (
        (tensor.id, "shape", 0),
        (tensor.id, "shape", 1),
    )

    execution = prepare_artifact_execution(
        artifact,
        {tensor.id: np.arange(6, dtype=np.float64).reshape(2, 3)},
    )

    assert list(execution.extents) == [2, 3]
    assert execution.buffers[tensor.id].shape == (2, 3)

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples" / "xor_project" / "train_xor.py"
)


@pytest.fixture(scope="module")
def ssa_module():
    source = EXAMPLE.read_text(encoding="utf-8")
    compilation = compile_ast_aot(
        source, "train", {},
        precompile_only=True, bake_mode="whole_program",
        mutable_parameters=("steps", "lr"),
    )
    result = lower_precompile_and_control_to_ssa(
        compilation.compiled_shell_program,
        compilation.shell_control_program,
        region_programs=dict(compilation.region_programs),
        hierarchy_plan=compilation.hierarchy_plan,
        identity_table=compilation.identity_table,
        function_outputs=tuple(compilation.function_outputs),
        function_parameters=tuple(compilation.function_parameters),
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
    )
    return result


def _region_functions(module) -> list[str]:
    return sorted(
        name for name in module.functions
        if name.startswith("numerical_region_")
    )


def test_real_program_lowers_to_ssa(ssa_module):
    for shortfall in ssa_module.shortfalls:
        print("lowering shortfall:", shortfall)
    assert ssa_module.shortfalls == ()
    assert _region_functions(ssa_module.module)


def test_real_program_regions_emit_llvm(ssa_module):
    for region in _region_functions(ssa_module.module):
        artifact = emit_ssa_function_to_llvm(
            ssa_module.module, region, entry_name=region,
        )
        for shortfall in artifact.shortfalls:
            print(f"{region} emission shortfall:", shortfall)
        assert artifact.shortfalls == (), region
        # Every local in the emitted entry is defined exactly once.
        seen: set[str] = set()
        in_entry = False
        for line in artifact.llvm_ir.splitlines():
            if line.startswith(f"define void @{region}"):
                in_entry = True
                continue
            if in_entry and line.startswith("}"):
                break
            if in_entry and " = " in line:
                name = line.strip().split(" = ")[0]
                assert name not in seen, f"{region}: duplicate local {name}"
                seen.add(name)


def test_real_program_regions_compile_to_native_artifacts(ssa_module, tmp_path):
    for region in _region_functions(ssa_module.module):
        artifact = emit_ssa_function_to_llvm(
            ssa_module.module, region, entry_name=region,
        )
        if not artifact.complete:
            pytest.fail(f"{region} has emission shortfalls")
        compiled = compile_artifact(artifact, directory=tmp_path / region)
        assert compiled.library_path is not None and compiled.library_path.is_file()
        assert compiled.entry() is not None
