"""Sequence helper GEP bases retain the declared storage formal identity."""

from src.compiler.ir_sequence_tables import lower_sequence_extend, lower_sequence_replace
from src.transmogrifier.ssa import SSASequenceCapacityPolicy, SSASequenceDescriptor


def _sequence(column: int, *, owned: bool) -> SSASequenceDescriptor:
    return SSASequenceDescriptor(
        sequence_id=column,
        column_value_ids=(column,),
        length_address_id=column + 100,
        capacity_value_id=column + 200,
        status_address_id=column + 300,
        column_dtypes=("int64",),
        capacity_policy=(SSASequenceCapacityPolicy.DYNAMIC if owned
                         else SSASequenceCapacityPolicy.FIXED),
        heap_owner_value_id=column + 400 if owned else None,
    )


def _assert_indexed_storage_is_formal(function, ids: set[int]) -> None:
    formals = {int(value.id): value for value in function.args}
    seen = set()
    for block in function.blocks.values():
        for instruction in block.instrs:
            if instruction.op != "GetElementPtr":
                continue
            base = instruction.args[0]
            if int(base.id) not in ids:
                continue
            seen.add(int(base.id))
            assert base is formals[int(base.id)]
            assert base.dtype == formals[int(base.id)].dtype
    assert seen == ids


def test_dynamic_replace_uses_ptr_destination_and_fixed_source_formals():
    # A bytearray replacement grows a ptr arena and reads an int64 row arena.
    destination, source = _sequence(15, owned=True), _sequence(25, owned=False)
    lowering = lower_sequence_replace(
        destination, source, function_name="owned_bytes_replace",
    )
    assert lowering.complete, lowering.shortfalls
    function = lowering.entry
    _assert_indexed_storage_is_formal(function, {15, 25, 115, 125})
    formals = {int(value.id): value for value in function.args}
    assert formals[15].dtype == "ptr"
    assert formals[25].dtype == "int64"


def test_extend_uses_exact_source_formal_for_each_indexed_row():
    # The real tensor_octets_to_bytes extend reads a fixed source row while
    # its destination can grow through the incumbent nested insert helper.
    destination, source = _sequence(50, owned=True), _sequence(47, owned=False)
    lowering = lower_sequence_extend(
        destination, source, function_name="owned_bytes_extend",
    )
    assert lowering.complete, lowering.shortfalls
    function = next(function for function in lowering.functions
                    if function.metadata.get("ssa_sequence_operation") == "extend")
    _assert_indexed_storage_is_formal(function, {47, 147})
    formals = {int(value.id): value for value in function.args}
    assert formals[47].dtype == "int64"

