from __future__ import annotations

import ctypes
import math
import os
import random

import numpy as np
import pytest

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    LLVM_SSA_MODULE,
    TRANSLATIONS,
    c_backend_repository_ssa_reference,
)
from src.common.tensors.accelerator_backends.aot_compile import compile_ast_aot
from src.common.tensors.accelerator_backends.llvm_repository_ssa import (
    import_llvm_to_repository_ssa,
)
from src.common.tensors.accelerator_backends.ssa_backend import (
    SSA_TENSOR_FORTRAN_SOURCE_ONLY,
    SSATensorOperations,
    SSATensorProgram,
    emit_ssa_tensor_backend_runtime,
    replace_abstract_tensor_content_with_ssa,
)
from src.common.tensors.numpy_backend import NumPyTensorOperations
from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.abstract_nn.activations import Tanh
from src.common.tensors.abstract_nn.core import Linear, Model
from src.compiler.precompile_to_ssa import (
    find_ssa_cycles,
    lower_control_sections_to_ssa,
)
from src.compiler.ssa_fortran_backend import compile_module, emit_module, fortran_compiler
from src.compiler.tensor_ssa_lowering import lower_tensor_calls_to_repository_ssa
from src.compiler.ssa_features import (
    RANDOM_SSA_MODULE,
    XOROSHIRO128SS_FILL,
    link_required_ssa_features,
)
from src.transmogrifier.ssa import BasicBlock, Function, IRModule, Instr, SSAValue
from src.transmogrifier.ssa_registry import Handler


def test_real_llvm_tensor_algorithms_import_to_fundamental_repository_ssa():
    result = import_llvm_to_repository_ssa(LLVM_SSA_MODULE)

    assert result.complete, result.shortfall_report()
    legal = {handler.value for handler in Handler}
    assert all(
        instruction.op in legal
        for function in result.module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
    )
    assert {
        translation.llvm_symbol
        for translation in TRANSLATIONS
    } <= set(result.module.functions)


def test_c_backend_is_one_complete_repository_ssa_code_reference():
    reference = c_backend_repository_ssa_reference()

    assert reference.source_identity == "c_backend_llvm_ssa.LLVM_SSA_MODULE"
    assert reference.operation("neg").entrypoints == ("unary_double",)
    assert len(reference.primitive_entrypoints) == 25
    assert {
        "binary_double",
        "binary_scalar_double",
        "binary_value",
    } <= set(reference.dependency_closure("binary_double", "binary_scalar_double"))
    assert all(
        "tensor_operation" not in instruction.attributes
        for function in reference.module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
    )


def test_ssa_abstract_tensor_basis_is_emittable_as_a_runtime_module():
    reference = c_backend_repository_ssa_reference()
    emitted = emit_ssa_tensor_backend_runtime(reference=reference)

    assert emitted.complete, [item.format() for item in emitted.shortfalls]
    assert {subroutine.name for subroutine in emitted.subroutines} == (
        set(reference.primitive_entrypoints) - SSA_TENSOR_FORTRAN_SOURCE_ONLY
    )


def test_tensor_instruction_materializes_real_repository_ssa_kernel_operands():
    source = SSAValue(1000, dtype="float64", shape=(4,))
    result = SSAValue(1001, dtype="float64", shape=(4,))
    caller = Function(
        "program",
        [source],
        {
            "entry": BasicBlock("entry", [
                Instr(
                    Handler.Call.value,
                    [source],
                    result,
                    attributes={
                        "tensor_operation": "neg",
                        "callee": "unary_double",
                    },
                ),
                Instr(Handler.Ret.value, [result], None),
            ])
        },
    )
    module = IRModule({"program": caller})

    shortfalls = lower_tensor_calls_to_repository_ssa(
        module,
        c_backend_repository_ssa_reference(),
    )

    assert shortfalls == ()
    call = caller.blocks["entry"].instrs[2]
    assert call.op == Handler.Call.value
    assert call.attributes == {
        "callee": "unary_double",
        "ssa_output_argument": 1,
    }
    assert call.args[:2] == [source, result]
    assert [argument.dtype for argument in call.args[2:]] == ["int32", "int32"]
    assert "unary_double" in module.functions
    assert not any(
        instruction.attributes.get("tensor_operation")
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
    )


def test_whole_module_tensor_recipes_expand_views_transpose_reduction_cast_and_cbrt():
    source = SSAValue(2000, dtype="float64", shape=(2, 3))
    dim0 = SSAValue(2001, dtype="int32")
    dim1 = SSAValue(2002, dtype="int32")
    transposed = SSAValue(2003, dtype="float64", shape=(3, 2))
    viewed = SSAValue(2004, dtype="float64", shape=(6,))
    axis = SSAValue(2005, dtype="int32")
    reduced = SSAValue(2006, dtype="float64", shape=())
    cast = SSAValue(2007, dtype="int64", shape=())
    cube_root = SSAValue(2008, dtype="float64", shape=())
    caller = Function("whole_recipes", [source], {"entry": BasicBlock("entry", [
        Instr(Handler.Const.value, [], dim0, attributes={"constant": 0}),
        Instr(Handler.Const.value, [], dim1, attributes={"constant": 1}),
        Instr(Handler.Call.value, [source, dim0, dim1], transposed,
              attributes={"tensor": "transpose"}),
        Instr(Handler.Call.value, [transposed], viewed,
              attributes={"tensor_operation": "reshape"}),
        Instr(Handler.Const.value, [], axis, attributes={"constant": 0}),
        Instr(Handler.Call.value, [viewed, axis], reduced,
              attributes={"tensor_operation": "sum", "axis": 0}),
        Instr(Handler.Call.value, [reduced], cast,
              attributes={"tensor_operation": "long"}),
        Instr(Handler.Call.value, [cast], cube_root,
              attributes={"tensor_operation": "cbrt"}),
        Instr(Handler.Ret.value, [cube_root], None),
    ])})
    module = IRModule({caller.name: caller})

    shortfalls = lower_tensor_calls_to_repository_ssa(
        module, c_backend_repository_ssa_reference()
    )

    assert shortfalls == ()
    calls = [
        instruction for instruction in caller.blocks["entry"].instrs
        if instruction.op == Handler.Call.value
    ]
    callees = [instruction.attributes["callee"] for instruction in calls]
    assert "transpose_double" in callees
    assert "reduce_dim_double" in callees
    assert "cast_double_to_int_values" in callees
    assert callees[-4:] == [
        "unary_double", "sign_double", "binary_scalar_double", "binary_double"
    ]
    assert all("tensor_operation" not in instruction.attributes for instruction in calls)
    # Reshape is a metadata alias: reduction consumes the transposed arena under
    # the view's flat shape, and no runtime reshape call exists.
    reduction = next(item for item in calls if item.attributes["callee"] == "reduce_dim_double")
    assert reduction.args[0].id == transposed.id
    assert reduction.args[0].shape == viewed.shape
    emitted = emit_module(module, name="whole_tensor_recipes")
    assert emitted.complete, [item.format() for item in emitted.shortfalls]


def test_unknown_tensor_extent_is_not_silently_compiled_as_one_element():
    source = SSAValue(2100)
    result = SSAValue(2101)
    caller = Function("unknown_extent", [source], {"entry": BasicBlock("entry", [
        Instr(Handler.Call.value, [source], result,
              attributes={"tensor_operation": "exp"}),
    ])})
    module = IRModule({caller.name: caller})

    shortfalls = lower_tensor_calls_to_repository_ssa(
        module, c_backend_repository_ssa_reference()
    )

    assert len(shortfalls) == 1
    assert "extent/shape" in shortfalls[0].reason
    assert caller.blocks["entry"].instrs[0].attributes["tensor_operation"] == "exp"
    table = module.tensor_tables["unknown_extent"]
    assert table.by_id(source.id).metadata_state == "unresolved"
    assert table.by_id(result.id).metadata_state == "unresolved"
    emitted = emit_module(module, name="unknown_tensor_metadata")
    assert not emitted.complete
    assert any(
        shortfall.op == "ssa_tensor_table"
        for shortfall in emitted.shortfalls
    )


def test_abstract_tensor_backend_dispenses_advanced_composition_source_closure():
    program = SSATensorProgram("clamped")
    source = SSATensorOperations.input(program, (4,))

    result = source.clamp(min=0.0, max=1.0)
    module = result.compilable_ssa()

    caller = module.functions["clamped"]
    calls = [
        instruction
        for instruction in caller.blocks["entry"].instrs
        if instruction.op == Handler.Call.value
    ]
    assert [call.attributes["callee"] for call in calls] == [
        "binary_scalar_double",
        "binary_scalar_double",
    ]
    assert {"binary_scalar_double", "binary_value"} <= set(module.functions)
    table = module.tensor_tables["clamped"]
    assert table.by_id(source.data.tensor_id).storage == "input"
    assert table.by_id(result.data.tensor_id).storage == "output"
    assert table.by_id(result.data.tensor_id).static_element_count == 4
    assert not any(
        instruction.attributes.get("tensor_operation")
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
    )
    emitted = emit_module(module, name="clamped_ssa")
    assert emitted.complete, [item.format() for item in emitted.shortfalls]
    assert emitted.api.metadata["tensor_table_schema"] == (
        "turing.repository-ssa-tensor-table.v1"
    )
    assert len(emitted.api.metadata["tensor_tables"]["clamped"]) == 3


def test_ssa_tensor_table_separates_logical_view_from_shared_data_arena():
    program = SSATensorProgram("view_table")
    source = program.input_tensor((2, 3))
    result = source.reshape(3, 2)

    module = result.compilable_ssa()
    table = module.tensor_tables["view_table"]
    source_record = table.by_id(source.data.tensor_id)
    view_record = table.by_id(result.data.tensor_id)

    assert source_record.shape == (2, 3)
    assert source_record.strides == (3, 1)
    assert view_record.shape == (3, 2)
    assert view_record.strides == (2, 1)
    assert view_record.alias_of == source_record.tensor_id
    assert view_record.data_value_id == source_record.data_value_id
    assert view_record.storage == "output"


def test_incoming_abstract_tensor_is_replaced_by_ssa_and_keeps_compound_surface():
    incoming = NumPyTensorOperations()
    incoming.data = np.asarray([-2.0, -0.5, 0.5, 3.0], dtype=np.float64)
    program = SSATensorProgram("replaced_abstract_tensor")

    source = SSATensorOperations.replace_abstract_tensor(
        program, incoming, input_name="source"
    )
    incoming.data[:] = 99.0
    result = ((source.exp() + 1.0).log()).clamp(
        min=-2.0, max=2.0
    ).reshape(4).sum()
    module = result.compilable_ssa()

    assert source.data.shape == (4,)
    assert source.data.descriptor.storage == "input"
    assert source.data.descriptor.arena_id == source.data.tensor_id
    assert source.data.descriptor.allocation_owner == source.data.tensor_id
    assert source.data.descriptor.owns_allocation
    assert program.function.metadata["tensor_input_names"] == (
        (source.data.tensor_id, "source"),
    )
    assert not any(
        instruction.attributes.get("tensor_operation")
        or instruction.attributes.get("tensor")
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
    )
    emitted = emit_module(module, name="replaced_abstract_tensor_ssa")
    assert emitted.complete, [item.format() for item in emitted.shortfalls]


def test_snapshot_replacement_detaches_values_and_recursive_replacement_erases_leaves():
    first = NumPyTensorOperations()
    first.data = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    second = NumPyTensorOperations()
    second.data = np.asarray([5, 6], dtype=np.int64)
    program = SSATensorProgram("detached_tensor_snapshot")

    replaced = replace_abstract_tensor_content_with_ssa(
        program,
        {"first": first, "nested": (second, "kept")},
        snapshot_content=True,
        path="feeds",
    )
    first.data[:] = -1.0
    second.data[:] = -1

    assert isinstance(replaced["first"], SSATensorOperations)
    assert isinstance(replaced["nested"][0], SSATensorOperations)
    assert replaced["nested"][1] == "kept"
    consts = [
        instruction.attributes.get("values")
        for instruction in program.block.instrs
        if instruction.op == Handler.Const.value
    ]
    assert consts == [(1.0, 2.0, 3.0, 4.0), (5, 6)]
    assert all(
        descriptor.storage == "constant"
        and not descriptor.writable
        for descriptor in program.tensor_table.tensors.values()
    )


def test_process_graph_preserves_replaced_tensor_record_through_complete_ssa():
    incoming = NumPyTensorOperations()
    incoming.data = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    program = SSATensorProgram("process_graph_feed")
    feed = SSATensorOperations.replace_abstract_tensor(
        program, incoming, input_name="x"
    )

    compilation = compile_ast_aot(
        "def tensor_entry(x):\n"
        "    return ((x.exp() + 1.0).log()).sum()\n",
        "tensor_entry",
        {"x": feed},
        precompile_only=True,
        bake_mode="whole_program",
    )
    module, shortfalls, outputs = lower_control_sections_to_ssa(
        compilation.shell_control_program,
        hierarchy_plan=compilation.hierarchy_plan,
        control_name="tensor_entry",
        identity_table=compilation.identity_table,
        function_outputs=compilation.function_outputs,
        function_parameters=compilation.function_parameters,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
    )

    assert shortfalls == ()
    region_name = "tensor_entry__planned_region_0"
    records = tuple(module.tensor_tables[region_name].tensors.values())
    assert records[0].shape == (4,)
    assert records[0].storage == "input"
    assert [record.shape for record in records] == [
        (4,), (4,), (4,), (4,), ()
    ]
    assert all(record.metadata_state == "static" for record in records)
    assert not any(
        instruction.attributes.get("tensor_operation")
        or instruction.attributes.get("tensor")
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
    )

    emitted = emit_module(
        module,
        name="process_graph_tensor_ssa",
        outputs=outputs,
        extra_roots=tuple(module.functions),
    )
    assert emitted.complete, [item.format() for item in emitted.shortfalls]


@pytest.mark.parametrize(
    ("name", "build"),
    (
        ("clamp", lambda x: x.clamp(-1.0, 1.0)),
        ("clip", lambda x: x.clip(a_min=-1.0, a_max=1.0)),
        ("sec", lambda x: x.sec()),
        ("csc", lambda x: x.csc()),
        ("cot", lambda x: x.cot()),
        ("sech", lambda x: x.sech()),
        ("csch", lambda x: x.csch()),
        ("coth", lambda x: x.coth()),
        ("sinc", lambda x: x.sinc()),
        ("deg2rad", lambda x: x.deg2rad()),
        ("rad2deg", lambda x: x.rad2deg()),
        ("nan_to_num", lambda x: AbstractTensor.nan_to_num(x)),
        ("mean", lambda x: x.mean()),
        ("softmax", lambda x: x.softmax(0)),
        ("log_softmax", lambda x: x.log_softmax(0)),
    ),
)
def test_extended_abstract_tensor_catalog_dissolves_into_ordinary_ssa(name, build):
    program = SSATensorProgram(f"catalog_{name}")
    source = SSATensorOperations.input(program, (4,))

    result = build(source)
    module = result.compilable_ssa()

    assert not any(
        instruction.attributes.get("tensor_operation")
        or instruction.attributes.get("tensor")
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
    )
    emitted = emit_module(module, name=f"catalog_{name}")
    assert emitted.complete, [item.format() for item in emitted.shortfalls]


def test_abstract_nn_parameters_and_forward_join_one_ssa_program():
    previous_random_state = random.getstate()
    try:
        random.seed(1729)
        program = SSATensorProgram("abstract_nn_xor_mlp")
        source = SSATensorOperations.input(program, (4, 2))
        model = Model(
            layers=[
                Linear(2, 8, like=source, bias=True, init="xavier"),
                Linear(8, 1, like=source, bias=True, init="xavier"),
            ],
            activations=[Tanh(), None],
        )

        result = model.forward(source)
        parameters = tuple(model.parameters())
        emitted = emit_module(
            result.compilable_ssa(), name="abstract_nn_xor_mlp_ssa"
        )
    finally:
        random.setstate(previous_random_state)

    assert result.shape == (4, 1)
    assert [parameter.shape for parameter in parameters] == [
        (2, 8), (1, 8), (8, 1), (1, 1)
    ]
    assert all(
        isinstance(parameter.data, type(source.data))
        and parameter.data.program is program
        for parameter in parameters
    )
    assert emitted.complete, [item.format() for item in emitted.shortfalls]


@pytest.mark.skipif(fortran_compiler() is None, reason="no Fortran compiler installed")
def test_process_graph_two_layer_tensor_network_compiles_and_executes(tmp_path):
    shapes = {
        "x": (4, 2),
        "w1": (2, 8),
        "b1": (1, 8),
        "w2": (8, 1),
        "b2": (1, 1),
    }
    program = SSATensorProgram("process_graph_tensor_network")
    feeds = {
        name: SSATensorOperations.input(program, shape)
        for name, shape in shapes.items()
    }
    compilation = compile_ast_aot(
        "def network(x, w1, b1, w2, b2):\n"
        "    return (x @ w1 + b1).tanh() @ w2 + b2\n",
        "network",
        feeds,
        precompile_only=True,
        bake_mode="whole_program",
    )
    module, shortfalls, outputs = lower_control_sections_to_ssa(
        compilation.shell_control_program,
        hierarchy_plan=compilation.hierarchy_plan,
        control_name="network",
        identity_table=compilation.identity_table,
        function_outputs=compilation.function_outputs,
        function_parameters=compilation.function_parameters,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
    )
    emitted = emit_module(
        module,
        name="process_graph_tensor_network_ssa",
        outputs=outputs,
        extra_roots=tuple(module.functions),
    )
    library = compile_module(emitted, directory=tmp_path, standalone=False)

    x = np.asarray(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    w1 = np.arange(16, dtype=np.float64).reshape(2, 8) / 20.0 - 0.4
    b1 = np.linspace(-0.1, 0.1, 8, dtype=np.float64).reshape(1, 8)
    w2 = np.linspace(-0.3, 0.3, 8, dtype=np.float64).reshape(8, 1)
    b2 = np.asarray([[0.05]], dtype=np.float64)
    expected = np.tanh(x @ w1 + b1) @ w2 + b2
    output = np.empty((4, 1), dtype=np.float64)

    compiler_directory = os.path.dirname(str(fortran_compiler()))
    context = (
        os.add_dll_directory(compiler_directory)
        if os.name == "nt"
        else None
    )
    try:
        dll = ctypes.CDLL(str(library))
        entry = dll.network
        pointer = ctypes.POINTER(ctypes.c_double)
        entry.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
        ]
        entry(
            1,
            2,
            4,
            8,
            *(array.ctypes.data_as(pointer) for array in (
                x, w1, b1, w2, b2, output
            )),
        )
    finally:
        if context is not None:
            context.close()

    assert shortfalls == ()
    assert emitted.complete, [item.format() for item in emitted.shortfalls]
    assert {
        "network",
        "network__planned_region_0",
        "broadcast_double",
        "matmul_double",
    } <= set(module.functions)
    assert output == pytest.approx(expected, rel=1e-13, abs=1e-13)


@pytest.mark.skipif(fortran_compiler() is None, reason="no Fortran compiler installed")
def test_abstract_tensor_ssa_backend_source_compiles_and_executes(tmp_path):
    program = SSATensorProgram("advanced_tensor")
    incoming = NumPyTensorOperations()
    incoming.data = np.asarray(
        [[-2.0, -0.5], [0.5, 3.0]], dtype=np.float64
    )
    source = SSATensorOperations.replace_abstract_tensor(
        program, incoming, input_name="source"
    )
    # Discovery content is not retained by the SSA replacement. Runtime data
    # reaches the compiled input arena through the generated ABI below.
    incoming.data[:] = 123.0
    result = ((source.exp() + 1.0).log()).clamp(
        min=-2.0, max=2.0
    ).reshape(4).sum()
    emitted = emit_module(
        result.compilable_ssa(), name="advanced_tensor_ssa"
    )
    library = compile_module(emitted, directory=tmp_path, standalone=False)

    compiler_directory = os.path.dirname(str(fortran_compiler()))
    context = (
        os.add_dll_directory(compiler_directory)
        if os.name == "nt"
        else None
    )
    try:
        dll = ctypes.CDLL(str(library))
        entry = dll.advanced_tensor
        entry.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        values = (ctypes.c_double * 4)(-2.0, -0.5, 0.5, 3.0)
        output = ctypes.c_double()
        entry(2, values, ctypes.byref(output))
    finally:
        if context is not None:
            context.close()

    expected = sum(
        max(-2.0, min(2.0, math.log(math.exp(value) + 1.0)))
        for value in values
    )
    assert output.value == pytest.approx(expected, rel=1e-13, abs=1e-13)


@pytest.mark.skipif(fortran_compiler() is None, reason="no Fortran compiler installed")
def test_extended_softmax_composite_compiles_and_executes(tmp_path):
    program = SSATensorProgram("softmax_composite")
    source = SSATensorOperations.input(program, (4,))
    result = source.softmax(0)
    emitted = emit_module(
        result.compilable_ssa(), name="softmax_composite_ssa"
    )
    library = compile_module(emitted, directory=tmp_path, standalone=False)

    compiler_directory = os.path.dirname(str(fortran_compiler()))
    context = (
        os.add_dll_directory(compiler_directory)
        if os.name == "nt"
        else None
    )
    try:
        dll = ctypes.CDLL(str(library))
        entry = dll.softmax_composite
        entry.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        values = (ctypes.c_double * 4)(-2.0, -0.5, 0.5, 3.0)
        output = (ctypes.c_double * 4)()
        entry(1, 4, values, output)
    finally:
        if context is not None:
            context.close()

    scale = max(values)
    exponentials = [math.exp(value - scale) for value in values]
    denominator = sum(exponentials)
    assert list(output) == pytest.approx(
        [value / denominator for value in exponentials],
        rel=1e-13,
        abs=1e-13,
    )


def test_llvm_tensor_loops_retain_phi_cfg_cycles():
    result = import_llvm_to_repository_ssa(LLVM_SSA_MODULE)
    fill = result.module.functions["fill_double"]
    cycles = find_ssa_cycles(fill)

    assert len(cycles) == 1
    assert cycles[0].represented_by_phi


def test_llvm_switch_is_legalized_to_existing_compare_and_branch_ops():
    result = import_llvm_to_repository_ssa(LLVM_SSA_MODULE)
    binary = result.module.functions["binary_value"]
    instructions = [
        instruction
        for block in binary.blocks.values()
        for instruction in block.instrs
    ]

    assert "Switch" not in {instruction.op for instruction in instructions}
    assert any(
        instruction.op == Handler.Eq.value
        and instruction.attributes.get("llvm_opcode") == "switch"
        for instruction in instructions
    )
    assert any(
        instruction.op == Handler.CondBr.value
        and instruction.attributes.get("llvm_opcode") == "switch"
        for instruction in instructions
    )


def test_random_feature_imports_as_real_bitwise_repository_ssa():
    imported = import_llvm_to_repository_ssa(RANDOM_SSA_MODULE)

    assert imported.complete, imported.shortfall_report()
    function = imported.module.functions[XOROSHIRO128SS_FILL]
    operations = {
        instruction.op
        for block in function.blocks.values()
        for instruction in block.instrs
    }
    assert {
        Handler.Xor.value,
        Handler.Or.value,
        Handler.Shl.value,
        Handler.Shr.value,
    } <= operations


def test_random_feature_is_linked_only_when_called():
    output = SSAValue(0, dtype="float64", shape=(4,))
    caller = Function(
        "program",
        [],
        {
            "entry": BasicBlock(
                "entry",
                [
                    Instr(
                        Handler.Call.value,
                        [],
                        output,
                        attributes={"callee": XOROSHIRO128SS_FILL},
                    )
                ],
            )
        },
    )

    assert XOROSHIRO128SS_FILL in link_required_ssa_features({"program": caller})
    assert XOROSHIRO128SS_FILL not in link_required_ssa_features({})
