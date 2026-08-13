import ctypes
import os
from pathlib import Path

import pytest

from src.compiler.ir_sequence_tables import (
    SSASequenceShortfallCode,
    lower_sequence_add,
    lower_sequence_fill,
    lower_sequence_aggregate_constants,
    lower_sequence_append,
    lower_sequence_append_slice,
    lower_sequence_pack_bits,
    lower_sequence_prepend,
    lower_sequence_prepend_packed_bytes,
    lower_sequence_extend,
    lower_sequence_insert,
    lower_table_delete,
    lower_table_lookup,
    lower_table_store,
)
from src.compiler.control_source import (
    ControlProgram,
    ControlSequenceMutation,
    LoopBlock,
    SequenceBlock,
)
from src.compiler.precompile_to_ssa import lower_control_sections_to_ssa
from src.compiler.ssa_fortran_backend import FortranEmissionError, emit_module
from src.compiler.ssa_fortran_backend import fortran_compiler
from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
from src.transmogrifier.ssa import (
    BasicBlock,
    Function,
    IRModule,
    Instr,
    SSASequenceCapacityPolicy,
    SSASequenceDescriptor,
    SSAChildTablePoolDescriptor,
    SSASequenceTable,
    SSAValue,
    SSARecordDescriptor,
    SSARecordFieldDescriptor,
    SSARecordFieldStorage,
    SSARecordTable,
)


def _ops(function):
    return [
        instruction.op
        for block in function.blocks.values()
        for instruction in block.instrs
    ]


def _sequence(sequence_id, base, *, key_columns=(), live=False, dynamic=False):
    return SSASequenceDescriptor(
        sequence_id=sequence_id,
        column_value_ids=(base,),
        length_address_id=base + 1,
        capacity_value_id=base + 2,
        column_dtypes=("int64",),
        key_columns=key_columns,
        live_flags_value_id=base + 3 if live else None,
        capacity_policy=(
            SSASequenceCapacityPolicy.DYNAMIC
            if dynamic
            else SSASequenceCapacityPolicy.FIXED
        ),
    )


def test_list_descriptor_allows_duplicates_and_lowers_direct_memory_insert():
    descriptor = _sequence(1, 10)
    lowering = lower_sequence_append(descriptor)

    assert descriptor.allows_duplicates
    assert lowering.complete
    function = lowering.functions[0]
    assert "unique_scan_header" not in function.blocks
    assert {"GetElementPtr", "Store", "Load", "Lt", "CondBr"} <= set(_ops(function))
    assert function.metadata["allows_duplicates"] is True
    assert function.metadata["ssa_sequence_operation"] == "append"


def test_set_descriptor_retains_linear_unique_key_and_live_row_policy():
    descriptor = _sequence(2, 20, key_columns=(0,), live=True)
    lowering = lower_sequence_add(descriptor)

    assert not descriptor.allows_duplicates
    assert descriptor.retains_deleted_rows
    assert lowering.complete
    function = lowering.functions[0]
    assert "unique_scan_header" in function.blocks
    assert "duplicate" in function.blocks
    assert {"Phi", "Eq", "LAnd", "GetElementPtr", "Load"} <= set(_ops(function))
    assert function.metadata["key_columns"] == (0,)
    assert function.metadata["ssa_sequence_operation"] == "add"


def test_table_delete_lookup_and_store_share_live_flag_contract():
    descriptor = SSASequenceDescriptor(
        sequence_id=8,
        column_value_ids=(80, 81),
        length_address_id=82,
        capacity_value_id=83,
        status_address_id=84,
        column_dtypes=("int64", "int64"),
        key_columns=(0,),
        live_flags_value_id=85,
    )

    lookup = lower_table_lookup(descriptor).functions[0]
    store = lower_table_store(descriptor).functions[0]
    delete = lower_table_delete(descriptor).functions[0]

    assert {"LAnd", "Load"} <= set(_ops(lookup))
    assert {"LAnd", "Load", "Store"} <= set(_ops(store))
    assert {"LAnd", "Load", "Store"} <= set(_ops(delete))
    assert delete.metadata["status_values"] == {"missing": 0, "deleted": 4}
    assert delete.metadata["ssa_sequence_operation"] == "table_delete"
    for function in (lookup, store, delete):
        assert descriptor.live_flags_value_id in function.metadata[
            "sequence_array_argument_ids"
        ]


def test_multicolumn_table_key_compares_and_passes_every_component():
    descriptor = SSASequenceDescriptor(
        sequence_id=9,
        column_value_ids=(90, 91, 92, 93),
        length_address_id=94,
        capacity_value_id=95,
        status_address_id=96,
        column_dtypes=("int64", "int64", "int64", "int64"),
        key_columns=(0, 1, 2),
        live_flags_value_id=97,
    )

    lookup = lower_table_lookup(descriptor).functions[0]
    store = lower_table_store(descriptor).functions[0]
    delete = lower_table_delete(descriptor).functions[0]

    # storage + status + three authored key components (+ value for store)
    assert len(lookup.args) == 11
    assert len(delete.args) == 11
    assert len(store.args) == 12
    for function in (lookup, store, delete):
        assert _ops(function).count("Eq") == 3
        assert _ops(function).count("LAnd") >= 3
        assert function.metadata["key_columns"] == (0, 1, 2)


def test_runtime_sequence_replication_is_fixed_arena_fill_cfg():
    descriptor = SSASequenceDescriptor(
        sequence_id=12,
        column_value_ids=(120,),
        length_address_id=121,
        capacity_value_id=122,
        status_address_id=123,
        column_dtypes=("int64",),
    )

    function = lower_sequence_fill(descriptor).functions[0]

    assert function.metadata["ssa_sequence_operation"] == "fill"
    assert {"Phi", "Le", "Lt", "GetElementPtr", "Store", "CondBr"} <= set(
        _ops(function)
    )
    assert "fill_header" in function.blocks
    assert "capacity_exhausted" in function.blocks


def test_sequence_append_slice_normalizes_bounds_and_copies_between_arenas():
    destination = _sequence(13, 130)
    source = SSASequenceDescriptor(
        sequence_id=14,
        column_value_ids=(140,),
        length_address_id=141,
        capacity_value_id=142,
        column_dtypes=("int64",),
        writable=False,
    )

    lowering = lower_sequence_append_slice(destination, source)

    assert lowering.complete
    function = lowering.functions[0]
    assert function.metadata["ssa_sequence_operation"] == "append_slice"
    assert function.metadata["slice_step"] == 1
    assert {"Phi", "Lt", "Gt", "Le", "Load", "Store"} <= set(
        _ops(function)
    )
    assert "lower_negative" in function.blocks
    assert "upper_negative" in function.blocks
    assert "capacity_exhausted" in function.blocks


def test_sequence_pack_bits_is_nested_iterative_memory_ssa():
    destination = _sequence(15, 150)
    source = SSASequenceDescriptor(
        sequence_id=16,
        column_value_ids=(160,),
        length_address_id=161,
        capacity_value_id=162,
        column_dtypes=("int64",),
        writable=False,
    )

    lowering = lower_sequence_pack_bits(destination, source)

    assert lowering.complete
    function = lowering.functions[0]
    assert function.metadata["ssa_sequence_operation"] == "pack_bits"
    assert {"Phi", "Shl", "Or", "FloorDiv", "Load", "Store"} <= set(
        _ops(function)
    )
    assert "word_header" in function.blocks
    assert "bit_header" in function.blocks


def test_sequence_prepend_shifts_resident_rows_backwards():
    descriptor = _sequence(17, 170)

    lowering = lower_sequence_prepend(descriptor)

    assert lowering.complete
    function = lowering.functions[0]
    assert function.metadata["ssa_sequence_operation"] == "prepend"
    assert {"Phi", "Gt", "Sub", "Load", "Store"} <= set(_ops(function))
    assert "shift_header" in function.blocks
    assert "store_prefix" in function.blocks


def test_prepend_packed_bytes_shifts_then_packs_little_endian_words():
    destination = _sequence(18, 180)
    source = SSASequenceDescriptor(
        sequence_id=19,
        column_value_ids=(190,),
        length_address_id=191,
        capacity_value_id=192,
        column_dtypes=("int64",),
        writable=False,
    )

    lowering = lower_sequence_prepend_packed_bytes(destination, source)

    assert lowering.complete
    function = lowering.functions[0]
    assert function.metadata["ssa_sequence_operation"] == "prepend_packed_bytes"
    assert {"FloorDiv", "Shl", "Or", "Load", "Store"} <= set(_ops(function))
    assert "shift_header" in function.blocks
    assert "byte_header" in function.blocks


def test_nested_table_pool_is_structural_and_validated_against_function_values():
    pool = SSAChildTablePoolDescriptor(
        handle_column=1,
        column_value_ids=(90, 91),
        length_value_id=92,
        capacity_value_id=93,
        row_stride_value_id=94,
        status_value_id=95,
        live_flags_value_id=96,
        column_dtypes=("int64", "int64"),
    )
    descriptor = SSASequenceDescriptor(
        sequence_id=80,
        column_value_ids=(80, 81),
        length_address_id=82,
        capacity_value_id=83,
        status_address_id=84,
        column_dtypes=("int64", "int64"),
        key_columns=(0,),
        child_table_pool=pool,
    )
    function = Function(
        "nested_pool_probe",
        [SSAValue(value_id) for value_id in range(80, 96)],
        {"entry": BasicBlock("entry", [Instr("Ret", [], None)])},
    )
    module = IRModule(
        {function.name: function},
        sequence_tables={function.name: SSASequenceTable({80: descriptor})},
    )

    with pytest.raises(FortranEmissionError, match="child-table pool.*96"):
        emit_module(module, name="nested_pool_probe", extra_roots=(function.name,))
    assert descriptor.to_mapping()["child_table_pool"] == pool.to_mapping()


def test_extend_uses_destination_insert_policy_not_source_policy():
    destination = _sequence(3, 30, key_columns=(0,))
    source = _sequence(4, 40)
    lowering = lower_sequence_extend(destination, source)

    assert lowering.complete
    insert, extend = lowering.functions
    assert "unique_scan_header" in insert.blocks
    assert extend.metadata["destination_key_columns"] == (0,)
    call = next(
        instruction
        for block in extend.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
    )
    assert call.attributes["callee"] == insert.name
    assert {"Phi", "GetElementPtr", "Load", "Call"} <= set(_ops(extend))


def test_dynamic_growth_is_an_explicit_typed_shortfall():
    lowering = lower_sequence_insert(_sequence(5, 50, dynamic=True))

    assert not lowering.complete
    assert lowering.functions == ()
    assert lowering.shortfalls[0].code is (
        SSASequenceShortfallCode.DYNAMIC_GROWTH_UNAVAILABLE
    )


def _loop_with_mutation(mutation):
    return ControlProgram(LoopBlock(
        "i", "0", "4", "1", SequenceBlock(()),
        sequence_mutations=(mutation,),
    ))


def test_retained_loop_append_becomes_internal_sequence_ssa_call_and_table():
    program = _loop_with_mutation(ControlSequenceMutation(
        sequence_value_id=10,
        operator="append",
        argument_value_ids=(11,),
        effect_node_id=12,
        policy="duplicates",
    ))
    module, shortfalls, _outputs = lower_control_sections_to_ssa(
        program,
        identity_table={"items": (10,), "value": (11,)},
    )

    assert shortfalls == ()
    assert "ssa_sequence_10_append" in module.functions
    assert module.sequence_tables["planned_control"].by_id(10).allows_duplicates
    control_calls = [
        instruction
        for block in module.functions["planned_control"].blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("ssa_sequence_operation") == "append"
    ]
    assert len(control_calls) == 1
    descriptor = module.sequence_tables["planned_control"].by_id(10)
    assert descriptor.status_address_id is not None
    assert not any(
        instruction.attributes.get("callee") == "turing_sequence_capacity_error"
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
    )
    assert any(
        instruction.op == "Store"
        and instruction.attributes.get("binding") == "ssa_sequence_status"
        for block in module.functions["planned_control"].blocks.values()
        for instruction in block.instrs
    )
    helper_ops = set(_ops(module.functions["ssa_sequence_10_append"]))
    assert {"GetElementPtr", "Load", "Store", "Lt", "CondBr"} <= helper_ops
    emitted = emit_module(
        module,
        name="sequence_append_probe",
        extra_roots=tuple(module.functions),
    )
    assert emitted.complete, [item.format() for item in emitted.shortfalls]
    descriptor_record = emitted.api.metadata["sequence_tables"][
        "planned_control"
    ][0]
    assert descriptor_record["sequence_id"] == 10
    assert descriptor_record["source_names"] == ["items"]
    entry = emitted.api.entry_point("planned_control")
    parameters = {parameter.name: parameter for parameter in entry.parameters}
    assert parameters[f"t{descriptor.column_value_ids[0]}"].passing == "reference"
    assert parameters[f"t{descriptor.length_address_id}"].passing == "reference"
    assert parameters[f"t{descriptor.status_address_id}"].passing == "reference"
    assert parameters[f"t{descriptor.capacity_value_id}"].passing == "value"


@pytest.mark.skipif(
    fortran_compiler() is None,
    reason="no Fortran compiler installed",
)
def test_compiled_retained_loop_mutates_caller_sequence_record(tmp_path):
    program = _loop_with_mutation(ControlSequenceMutation(
        sequence_value_id=10,
        operator="append",
        argument_value_ids=(11,),
        effect_node_id=12,
        policy="duplicates",
    ))
    module, shortfalls, _ = lower_control_sections_to_ssa(
        program,
        identity_table={"items": (10,), "value": (11,)},
    )
    assert shortfalls == ()
    emitted = emit_module(
        module,
        name="compiled_sequence_record",
        extra_roots=tuple(module.functions),
    )
    artifact = compile_fortran_module_c_shell(
        emitted,
        {},
        tmp_path,
        entrypoint="planned_control",
        name="compiled_sequence_record",
        library=True,
    )
    descriptor = module.sequence_tables["planned_control"].by_id(10)
    assert descriptor is not None and descriptor.status_address_id is not None
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(Path(fortran_compiler()).resolve().parent))
    library = ctypes.CDLL(str(artifact.executable_path))
    function = library.planned_control
    function.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_double,
    ]
    function.restype = None
    arena = (ctypes.c_double * 4)()
    length = ctypes.c_int32(0)
    status = ctypes.c_int32(-1)
    assert artifact.entrypoint == "planned_control"
    assert [parameter.name for parameter in emitted.api.entry_point(
        "planned_control"
    ).parameters][:1] == ["extent_1"]
    function(1, arena, ctypes.byref(length), 4, ctypes.byref(status), 7.5)
    assert length.value == 4
    assert status.value == 1
    assert list(arena) == [7.5, 7.5, 7.5, 7.5]


def test_retained_loop_set_add_keeps_unique_scan_and_extend_inherits_it():
    add_program = _loop_with_mutation(ControlSequenceMutation(
        sequence_value_id=20,
        operator="add",
        argument_value_ids=(21,),
        effect_node_id=22,
        policy="unique",
    ))
    add_module, add_shortfalls, _ = lower_control_sections_to_ssa(
        add_program,
        identity_table={"seen": (20,), "value": (21,)},
    )
    assert add_shortfalls == ()
    add_helper = add_module.functions["ssa_sequence_20_add"]
    assert "unique_scan_header" in add_helper.blocks

    extend_program = _loop_with_mutation(ControlSequenceMutation(
        sequence_value_id=30,
        operator="extend",
        argument_value_ids=(31,),
        effect_node_id=32,
        policy="unique",
        argument_kind="sequence",
    ))
    extend_module, extend_shortfalls, _ = lower_control_sections_to_ssa(
        extend_program,
        identity_table={"seen": (30,), "incoming": (31,)},
    )
    assert extend_shortfalls == ()
    insert = extend_module.functions["ssa_sequence_30_insert"]
    assert "unique_scan_header" in insert.blocks
    assert extend_module.sequence_tables["planned_control"].by_id(30).key_columns == (0,)


def test_retained_generator_extend_is_typed_shortfall_not_opaque_or_fallback():
    program = _loop_with_mutation(ControlSequenceMutation(
        sequence_value_id=40,
        operator="extend",
        argument_value_ids=(41,),
        effect_node_id=42,
        policy="duplicates",
        argument_kind="generator",
    ))
    _module, shortfalls, _ = lower_control_sections_to_ssa(
        program,
        identity_table={"items": (40,), "generated": (41,)},
    )

    assert len(shortfalls) == 1
    assert shortfalls[0].domain == "ssa-sequence"
    assert "iterator contract" in shortfalls[0].reason


def test_filtered_eager_comprehension_requires_compact_materialization():
    program = _loop_with_mutation(ControlSequenceMutation(
        sequence_value_id=50,
        operator="extend",
        argument_value_ids=(51,),
        effect_node_id=52,
        policy="duplicates",
        argument_kind="filtered_sequence",
    ))
    _module, shortfalls, _ = lower_control_sections_to_ssa(
        program,
        identity_table={"items": (50,), "comprehension": (51,)},
    )

    assert len(shortfalls) == 1
    assert "predicated compact materialization" in shortfalls[0].reason


def test_empty_aggregate_const_becomes_explicit_sequence_arena_argument():
    arena = SSAValue(60)
    function = Function(
        "empty_sequence",
        [],
        {"entry": BasicBlock("entry", [
            Instr("Const", [], arena, attributes={"value": []}),
            Instr("Ret", [], None),
        ])},
    )
    descriptor = _sequence(60, 60)

    lower_sequence_aggregate_constants(
        {function.name: function},
        {function.name: SSASequenceTable({60: descriptor})},
    )

    assert [instruction.op for instruction in function.blocks["entry"].instrs] == ["Ret"]
    assert [argument.id for argument in function.args] == [60]
    assert function.metadata["sequence_aggregate_inputs"] == (60,)
    module = IRModule(
        {function.name: function},
        sequence_tables={
            function.name: SSASequenceTable({60: descriptor})
        },
    )
    emitted = emit_module(
        module,
        name="empty_sequence_probe",
        extra_roots=(function.name,),
    )
    assert emitted.complete, [item.format() for item in emitted.shortfalls]
    assert emitted.api.metadata["sequence_table_schema"] == (
        "turing.repository-ssa-sequence-table.v1"
    )
    assert emitted.api.metadata["sequence_tables"] == {
        function.name: [{**descriptor.to_mapping(), "source_names": []}]
    }


def test_non_sequence_empty_literal_is_not_removed():
    value = SSAValue(70)
    function = Function(
        "ordinary_literal",
        [],
        {"entry": BasicBlock("entry", [
            Instr("Const", [], value, attributes={"value": []}),
        ])},
    )

    lower_sequence_aggregate_constants({function.name: function}, {})

    assert function.blocks["entry"].instrs[0].attributes["value"] == []


def test_record_table_groups_sequence_storage_without_object_handle():
    function = Function(
        "graph_record_probe",
        [SSAValue(80), SSAValue(81, dtype="int", shape=(1,)), SSAValue(82, dtype="int")],
        {"entry": BasicBlock("entry", [Instr("Ret", [], None)])},
    )
    sequence = SSASequenceDescriptor(
        sequence_id=80,
        column_value_ids=(80,),
        length_address_id=81,
        capacity_value_id=82,
        column_dtypes=("int64",),
    )
    record = SSARecordDescriptor(
        90,
        "networkx.DiGraph",
        fields=(SSARecordFieldDescriptor(
            "_node",
            SSARecordFieldStorage.SEQUENCE,
            value_ids=(80, 81, 82),
            sequence_id=80,
        ),),
    )
    module = IRModule(
        {function.name: function},
        sequence_tables={function.name: SSASequenceTable({80: sequence})},
        record_tables={function.name: SSARecordTable({90: record})},
    )
    emitted = emit_module(module, name="graph_record_probe", extra_roots=(function.name,))
    assert emitted.complete
    assert emitted.api.metadata["record_table_schema"] == (
        "turing.repository-ssa-record-table.v1"
    )
    assert emitted.api.metadata["record_tables"][function.name] == [
        record.to_mapping()
    ]
    # A record field is a correlation to the actual sequence arenas, never a
    # runtime object/handler token.
    assert record.fields[0].value_ids == (80, 81, 82)


def test_record_table_rejects_storage_references_absent_from_function():
    function = Function(
        "broken_record_probe",
        [],
        {"entry": BasicBlock("entry", [Instr("Ret", [], None)])},
    )
    record = SSARecordDescriptor(
        90,
        "networkx.DiGraph",
        fields=(SSARecordFieldDescriptor(
            "_node",
            SSARecordFieldStorage.SEQUENCE,
            value_ids=(80,),
            sequence_id=80,
        ),),
    )
    module = IRModule(
        {function.name: function},
        record_tables={function.name: SSARecordTable({90: record})},
    )
    with pytest.raises(FortranEmissionError, match="absent SSA values.*80"):
        emit_module(module, name="broken_record_probe", extra_roots=(function.name,))


def test_retained_field_sequence_populates_typed_record_from_lowered_storage():
    program = _loop_with_mutation(ControlSequenceMutation(
        sequence_value_id=10,
        operator="append",
        argument_value_ids=(11,),
        effect_node_id=12,
        policy="duplicates",
    ))
    module, shortfalls, _ = lower_control_sections_to_ssa(
        program,
        identity_table={"self": (5,), "self.items": (10,), "value": (11,)},
        self_value_id=5,
        field_ops=(("read", 10, 0),),
        field_count=1,
        field_names=("items",),
        record_identity="Accumulator",
    )

    assert shortfalls == ()
    record = module.record_tables["planned_control"].records[5]
    (field,) = record.fields
    descriptor = module.sequence_tables["planned_control"].by_id(10)
    assert record.identity == "Accumulator"
    assert field.name == "items"
    assert field.storage is SSARecordFieldStorage.SEQUENCE
    assert field.sequence_id == descriptor.sequence_id
    assert set(field.value_ids) == {
        *descriptor.column_value_ids,
        descriptor.length_address_id,
        descriptor.capacity_value_id,
        descriptor.status_address_id,
    }
    emitted = emit_module(
        module,
        name="retained_field_record",
        extra_roots=tuple(module.functions),
    )
    assert emitted.complete
    assert emitted.api.metadata["record_tables"]["planned_control"] == [
        record.to_mapping()
    ]
