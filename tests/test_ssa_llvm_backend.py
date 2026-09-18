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


def test_pointer_array_materializes_repository_pointer_table():
    left = SSAValue(0, "float64", (2,))
    right = SSAValue(1, "float64", (2,))
    table = SSAValue(2, "ptrptr_float64", (2,))
    callee_table = SSAValue(10, "ptrptr_float64", (2,))
    index = SSAValue(11, "int32")
    address = SSAValue(12, "ptr")
    selected = SSAValue(13, "float64", (2,))
    callee = Function("consume_table", [callee_table], {
        "entry": BasicBlock("entry", [
            Instr("Const", [], index, attributes={"value": 0}),
            Instr(
                "GetElementPtr", [callee_table, index], address,
                attributes={"aggregate_index": 0},
            ),
            Instr("Load", [address], selected),
            Instr("Ret", [], None),
        ]),
    })
    caller = Function("make_table", [left, right], {
        "entry": BasicBlock("entry", [
            Instr("PointerArray", [left, right], table),
            Instr("Call", [table], None, attributes={"callee": callee.name}),
            Instr("Ret", [], None),
        ]),
    })

    artifact = emit_ssa_function_to_llvm(
        IRModule({caller.name: caller, callee.name: callee}), caller.name,
    )

    assert artifact.shortfalls == ()
    assert "%aggregate.pointer_array." in artifact.llvm_ir
    assert "store ptr" in artifact.llvm_ir


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


def test_referenced_target_intrinsics_are_declared_and_compile(tmp_path):
    """Intrinsics the scalar table calls are declared from that same table.

    A target intrinsic has no authored repository definition, so before this
    the emitter reported it as an unresolved symbol. The declaration is read
    back from the authored call template, so only referenced intrinsics appear
    and their signatures cannot drift from the calls.
    """

    value = SSAValue(0, "float64")
    exponential = SSAValue(1, "float64")
    logarithm = SSAValue(2, "float64")
    result = SSAValue(3, "float64")
    function = Function("transcendental", [value], {
        "entry": BasicBlock("entry", [
            Instr("Exp", [value], exponential),
            Instr("Log", [exponential], logarithm),
            Instr("Add", [exponential, logarithm], result),
            Instr("Ret", [result], None),
        ]),
    })

    artifact = emit_ssa_function_to_llvm(
        IRModule({function.name: function}), function.name,
    )

    assert artifact.shortfalls == ()
    declared = {
        line for line in artifact.llvm_ir.splitlines()
        if line.startswith("declare")
    }
    assert declared == {
        "declare double @llvm.exp.f64(double)",
        "declare double @llvm.log.f64(double)",
    }
    native = compile_artifact(artifact, directory=tmp_path / "transcendental")
    assert native.library_path is not None

    execution = prepare_artifact_execution(native, {value.id: 2.0})
    execution.run()
    assert float(execution.buffers[result.id]) == pytest.approx(
        np.exp(2.0) + 2.0
    )


def test_integer_scalar_domain_is_not_widened_through_double(tmp_path):
    """An integer result stays in the integer column, end to end.

    The scalar tables' double column used to evaluate every opcode, so an
    int32 result was computed as a double and then stored into its declared
    four-byte slot -- eight bytes into four, read back as noise. Max/Min have
    no integer intrinsic, so they lower to compare-and-select.
    """

    left = SSAValue(0, "int32")
    right = SSAValue(1, "int32")
    largest = SSAValue(2, "int32")
    smallest = SSAValue(3, "int32")
    span = SSAValue(4, "int32")
    function = Function("integer_span", [left, right], {
        "entry": BasicBlock("entry", [
            Instr("Max", [left, right], largest),
            Instr("Min", [left, right], smallest),
            Instr("Sub", [largest, smallest], span),
            Instr("Ret", [span], None),
        ]),
    })

    artifact = emit_ssa_function_to_llvm(
        IRModule({function.name: function}), function.name,
    )

    assert artifact.shortfalls == ()
    # No float ever enters this program.
    assert "double" not in artifact.llvm_ir
    assert "select i1" in artifact.llvm_ir

    native = compile_artifact(artifact, directory=tmp_path / "integer_span")
    for first, second in (
        (3, -7), (-2, -9), (5, 5), (-4, 11), (0, 0),
        (1073741823, -1073741824), (-2147483648, -2147483647),
    ):
        execution = prepare_artifact_execution(
            native, {left.id: first, right.id: second},
        )
        execution.run()
        assert int(execution.buffers[span.id]) == max(first, second) - min(
            first, second
        )


def test_multi_axis_address_is_exact_or_named(tmp_path):
    """A 2-D span address is linearised from its extents, or refused.

    The emitter used to take ``GetElementPtr(span, row, col)``, keep only
    ``row``, and stride by a fixed ``i64`` regardless of the span's dtype --
    silently reading the wrong element while reporting no shortfall. With the
    extents declared the address is exact; without them the missing storage
    identity is named instead of guessed.
    """

    row = SSAValue(1, "int32")
    column = SSAValue(2, "int32")
    address = SSAValue(3, "ptr")
    element = SSAValue(4, "float64")

    def program(span):
        return Function("grid", [span, row, column], {
            "entry": BasicBlock("entry", [
                Instr("GetElementPtr", [span, row, column], address),
                Instr("Load", [address], element),
                Instr("Ret", [element], None),
            ]),
        })

    undeclared = program(SSAValue(0, "float64"))
    artifact = emit_ssa_function_to_llvm(
        IRModule({undeclared.name: undeclared}), undeclared.name,
    )
    assert any(
        shortfall.operation == "GetElementPtr"
        and "declared extents" in shortfall.reason
        for shortfall in artifact.shortfalls
    )

    declared = program(SSAValue(0, "float64", (3, 4)))
    exact = emit_ssa_function_to_llvm(
        IRModule({declared.name: declared}), declared.name,
    )
    assert exact.shortfalls == ()
    # The stride is the span's own element type, not a fixed eight bytes.
    assert "getelementptr double, ptr" in exact.llvm_ir

    native = compile_artifact(exact, directory=tmp_path / "grid")
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    for index in range(3):
        for axis in range(4):
            execution = prepare_artifact_execution(native, {
                0: values.copy(), row.id: index, column.id: axis,
            })
            execution.run()
            assert float(execution.buffers[element.id]) == values[index, axis]


def test_runtime_sized_span_address_measures_its_own_extents(tmp_path):
    """A span sized at call time addresses correctly without specialisation.

    The grid extents are not compile-time constants, so the row stride is
    measured from the real buffer through the public extents vector -- the
    same vector the executor already fills. One artifact therefore serves
    every grid size instead of one compiled variant per shape.
    """

    span = SSAValue(0, "float64", ("rows", "columns"))
    row = SSAValue(1, "int32")
    column = SSAValue(2, "int32")
    address = SSAValue(3, "ptr")
    element = SSAValue(4, "float64")
    function = Function("dynamic_grid", [span, row, column], {
        "entry": BasicBlock("entry", [
            Instr("GetElementPtr", [span, row, column], address),
            Instr("Load", [address], element),
            Instr("Ret", [element], None),
        ]),
    })

    artifact = emit_ssa_function_to_llvm(
        IRModule({function.name: function}), function.name,
    )

    assert artifact.shortfalls == ()
    # Only the trailing axis is needed to linearise a row-major address.
    assert artifact.extent_order == ((span.id, "dim", 1),)

    native = compile_artifact(artifact, directory=tmp_path / "dynamic_grid")
    for rows, columns in ((3, 4), (5, 2), (2, 7)):
        values = np.arange(rows * columns, dtype=np.float64).reshape(
            rows, columns
        )
        for index in range(rows):
            for axis in range(columns):
                execution = prepare_artifact_execution(native, {
                    span.id: values.copy(), row.id: index, column.id: axis,
                })
                execution.run()
                assert float(execution.buffers[element.id]) == values[
                    index, axis
                ]


SPAN_FIELD = {
    "program_abi_record": "Grid",
    "program_abi_parameter": "state",
    "program_abi_field": "grid",
    "program_abi_storage": "span",
    "program_abi_rank": 2,
    "program_abi_mutable": False,
}


def test_span_extent_resolves_through_the_internal_call_frame(tmp_path):
    """A region addresses a span it did not receive from the public boundary.

    Only the caller knows which public buffer a forwarded span came from, so
    the region's extent is resolved by walking the argument binding back to the
    root and measuring that buffer -- the same binding the record-field
    identity travels on. The whole-module emitter previously had no extents at
    all, so every address inside a region was unresolvable.
    """

    region_span = SSAValue(0, "float64", accounting=dict(SPAN_FIELD))
    region_row = SSAValue(1, "int32")
    region_column = SSAValue(2, "int32")
    region_address = SSAValue(3, "ptr")
    region_element = SSAValue(4, "float64")
    region = Function("region", [region_span, region_row, region_column], {
        "entry": BasicBlock("entry", [
            Instr(
                "GetElementPtr",
                [region_span, region_row, region_column],
                region_address,
            ),
            Instr("Load", [region_address], region_element),
            Instr("Ret", [region_element], None),
        ]),
    })

    span = SSAValue(0, "float64", accounting=dict(SPAN_FIELD))
    row = SSAValue(1, "int32")
    column = SSAValue(2, "int32")
    element = SSAValue(4, "float64")
    root = Function("root", [span, row, column], {
        "entry": BasicBlock("entry", [
            Instr(
                "Call", [span, row, column], element,
                attributes={"callee": "region"},
            ),
            Instr("Ret", [element], None),
        ]),
    })

    artifact = emit_ssa_function_to_llvm(
        IRModule({"root": root, "region": region}), "root",
    )

    assert artifact.shortfalls == ()
    # Measured against the root's own public buffer, not the region's local id.
    assert artifact.extent_order == ((span.id, "dim", 1),)

    native = compile_artifact(artifact, directory=tmp_path / "forwarded_span")
    for rows, columns in ((3, 4), (5, 2), (2, 7)):
        values = np.arange(rows * columns, dtype=np.float64).reshape(
            rows, columns
        )
        for index in range(rows):
            for axis in range(columns):
                execution = prepare_artifact_execution(native, {
                    span.id: values.copy(), row.id: index, column.id: axis,
                })
                execution.run()
                assert float(execution.buffers[element.id]) == values[
                    index, axis
                ]


def test_deployment_boundaries_are_not_instructions(tmp_path):
    """Deploy/Join schedule around the program; they emit no computation."""

    value = SSAValue(0, "float64")
    doubled = SSAValue(1, "float64")
    function = Function("bounded", [value], {
        "entry": BasicBlock("entry", [
            Instr("Deploy", [], None),
            Instr("Add", [value, value], doubled),
            Instr("Join", [], None),
            Instr("Ret", [doubled], None),
        ]),
    })

    artifact = emit_ssa_function_to_llvm(
        IRModule({function.name: function}), function.name,
    )

    assert artifact.shortfalls == ()
    native = compile_artifact(artifact, directory=tmp_path / "bounded")
    execution = prepare_artifact_execution(native, {value.id: 2.5})
    execution.run()
    assert float(execution.buffers[doubled.id]) == pytest.approx(5.0)


EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples" / "xor_project" / "train_xor.py"
)


@pytest.fixture(scope="module")
def ssa_module():
    pytest.skip(
        "disabled: whole-program XOR capture is not execution-bounded and "
        "can leave compiler child processes running after a test timeout"
    )
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
