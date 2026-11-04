"""Backend-neutral storage capacity recovered from repository SSA.

An SSA id is a logical identity, while a compiled frame needs a physical byte
capacity.  Linked programs often retain several shaped *views* of one storage
id; choosing the empty canonical formal (or one arbitrary view) allocates a
single element and corrupts every native backend alike.  This module derives
one conservative allocation requirement from the evidence already carried by
SSA occurrences and the canonical ``SSATensorTable``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Mapping
import re

from ..transmogrifier.dtype_layout import DEFAULT_DTYPE
from .ssa_aggregate_abi import _constant_integer


@dataclass(frozen=True, slots=True)
class SSAStorageRequirement:
    value_id: int
    dtype: str
    shape: tuple[int, ...]
    element_count: int | None
    views: tuple[tuple[int, ...], ...]
    dynamic: bool


@dataclass(frozen=True, slots=True)
class OwnedTensorDataBinding:
    tensor_id: int
    data_value_id: int
    owner_value_id: int
    count_value_id: int
    element_dtype: str
    cell: Any


def owned_tensor_data_bindings(module: Any, function_name: str) -> tuple[OwnedTensorDataBinding, ...]:
    """The audited owned pointer carriers of one exact native frame."""

    from .concordance_declarations import TENSOR_OWNED_HEAP_BINDING
    from .identity_concordance import CorrelationTable, identity_book
    from .ssa_record_return_state import function_scope_of

    book = identity_book(module)
    scope = function_scope_of(module.functions[function_name])
    page = book.page(TENSOR_OWNED_HEAP_BINDING)
    rows = tuple(row for row in page.scope_rows(scope)
                 if row[1] == function_name)
    if not rows:
        return ()
    if CorrelationTable._tensor_owned_heap_binding_findings(module):
        return ()
    bindings = []
    for row in rows:
        cell = book.latest_ref(TENSOR_OWNED_HEAP_BINDING, row)
        data, owner, count, dtype = page.cell(cell.row, cell.column)
        bindings.append(OwnedTensorDataBinding(
            int(row[2]), int(data), int(owner), int(count), str(dtype), cell,
        ))
    return tuple(bindings)


@dataclass(frozen=True, slots=True)
class PublicSequenceCapacitySource:
    """Caller-owned output capacity bounded by an exact input sequence."""

    output_sequence_id: int
    input_sequence_id: int
    output_data_id: int
    output_length_id: int
    output_capacity_id: int
    input_data_id: int
    input_length_id: int
    input_capacity_id: int
    bound_cell: Any
    output_descriptor_cell: Any
    input_descriptor_cell: Any


@dataclass(frozen=True, slots=True)
class PublicTensorCapacitySource:
    """Exact output/input and computed metadata ports of a bounded tensor."""

    output_data_id: int
    input_data_id: int
    output_shape_id: int
    output_count_id: int
    bound_cell: Any
    metadata_binding_cell: Any


def public_tensor_capacity_sources(module: Any, function_name: str) -> tuple[PublicTensorCapacitySource, ...]:
    """Read source capacity through the exact computed-result call binding."""

    from .concordance_declarations import (
        TENSOR_METADATA_RESULT_BINDING, TENSOR_OUTPUT_CAPACITY_BOUND,
    )
    from .ssa_record_return_state import function_scope_of

    function = (getattr(module, "functions", {}) or {}).get(function_name)
    book = (getattr(module, "metadata", {}) or {}).get("identity_book")
    if function is None or book is None:
        return ()
    scope = function_scope_of(function)
    inputs = {int(value.id) for value in function.args}
    returns = {
        int(value.id): value
        for block in function.blocks.values()
        for instruction in block.instrs if instruction.op == "Ret"
        for value in instruction.args
    }
    bounds = book.page(TENSOR_OUTPUT_CAPACITY_BOUND)
    bindings = book.page(TENSOR_METADATA_RESULT_BINDING)
    result = []
    for row in bindings.rows():
        if row[0] != scope or row[3] not in returns:
            continue
        bound_row = (scope, row[3])
        bound_ref = book.latest_ref(TENSOR_OUTPUT_CAPACITY_BOUND, bound_row)
        binding_ref = book.latest_ref(TENSOR_METADATA_RESULT_BINDING, row)
        if bound_ref is None or binding_ref is None:
            continue
        bound = bounds.cell(bound_ref.row, bound_ref.column)
        binding = bindings.cell(binding_ref.row, binding_ref.column)
        input_id = int(bound.input_cell.row[-1])
        if input_id not in inputs or binding[0] != row[3]:
            continue
        if tuple(binding[1]) != (
            int(bound.shape_cell.row[-1]), int(bound.rank_cell.row[-1]),
            int(bound.count_cell.row[-1]),
        ):
            continue
        output_shape_id, _rank_id, output_count_id = binding[2]
        output = returns[row[3]]
        accounting = output.accounting or {}
        if (accounting.get("tensor_shape_value_id") != output_shape_id
                or accounting.get("tensor_element_count_value_id") != output_count_id):
            continue
        result.append(PublicTensorCapacitySource(
            int(row[3]), input_id, int(output_shape_id),
            int(output_count_id), bound_ref, binding_ref,
        ))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class PublicSequenceInputPorts:
    """Exact native ports of one caller supplied bytes sequence."""

    data_id: int
    length_id: int
    capacity_id: int
    descriptor_cell: Any
    contract_cell: Any
    source_identity_cell: Any
    source_class_cell: Any


@dataclass(frozen=True, slots=True)
class PublicOwnedSequencePorts:
    """Native ports of one exact source-returned owned sequence."""

    data_id: int
    length_id: int
    capacity_id: int
    owner_id: int
    element_dtype: str
    public_cell: Any
    source_cell: Any
    descriptor_cell: Any
    owner_cell: Any
    return_slot_cell: Any


def public_owned_sequence_returns(
    module: Any, function_name: str,
) -> tuple[PublicOwnedSequencePorts, ...]:
    """Read the completed source return, descriptor, and heap owner chain."""

    from .concordance_declarations import (
        PUBLIC_OWNED_SEQUENCE_RETURN, SOURCE_OWNED_SEQUENCE_RETURN,
        PublicOwnedSequenceReturn, SourceOwnedSequenceReturn,
    )
    from .identity_concordance import CorrelationTable
    from ..transmogrifier.ssa import SSASequenceCapacityPolicy

    function = module.functions.get(str(function_name))
    table = (getattr(module, "sequence_tables", {}) or {}).get(str(function_name))
    book = None if table is None else table.book
    if function is None or book is None:
        return ()
    if (
        CorrelationTable._source_owned_sequence_return_findings(module)
        or CorrelationTable._public_owned_sequence_return_findings(module)
    ):
        return ()
    arguments = {int(value.id): value for value in function.args}
    returned = {
        int(value.id)
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret"
        and not instruction.attributes.get("fault_exit")
        for value in instruction.args
    }
    result = []
    page = book.page(PUBLIC_OWNED_SEQUENCE_RETURN)
    for row in page.rows():
        if not isinstance(row, tuple) or row[0] != str(function_name):
            continue
        cell = book.latest_ref(PUBLIC_OWNED_SEQUENCE_RETURN, row)
        fact = page.cell(cell.row, cell.column)
        if not isinstance(fact, PublicOwnedSequenceReturn):
            continue
        source = book.page(SOURCE_OWNED_SEQUENCE_RETURN).cell(
            fact.source_cell.row, fact.source_cell.column,
        )
        descriptor = table.sequences.get(int(row[1]))
        if (
            not isinstance(source, SourceOwnedSequenceReturn)
            or descriptor is None
            or descriptor.capacity_policy is not SSASequenceCapacityPolicy.DYNAMIC
            or descriptor.heap_owner_value_id is None
            or len(descriptor.column_value_ids) != 1
            or len(descriptor.column_dtypes) != 1
            or int(descriptor.column_value_ids[0]) != int(row[1])
            or int(row[1]) not in returned
        ):
            continue
        data_id = int(row[1])
        length_id = int(descriptor.length_address_id)
        capacity_id = int(descriptor.capacity_value_id)
        owner_id = int(descriptor.heap_owner_value_id)
        if (
            any(value_id not in arguments for value_id in (
                data_id, length_id, capacity_id, owner_id,
            ))
            or not (arguments[owner_id].accounting or {}).get("heap_owner")
            or (arguments[data_id].accounting or {}).get(
                "heap_owner_value_id"
            ) != owner_id
        ):
            continue
        result.append(PublicOwnedSequencePorts(
            data_id, length_id, capacity_id, owner_id,
            str(descriptor.column_dtypes[0]), cell, fact.source_cell,
            fact.descriptor_cell, fact.owner_cell, source.return_slot_cell,
        ))
    return tuple(result)


def _sequence_descriptor_identity(table: Any, sequence_id: int):
    """Follow one descriptor through its contract to one ingestion origin."""

    from .concordance_declarations import (
        CANONICAL_VALUE, SEQUENCE_CONTRACT_CONCORDANCE,
        SEQUENCE_DESCRIPTOR,
    )
    from .concorded_literals import origin_cell

    book = table.book
    descriptor = table.sequences.get(int(sequence_id))
    cell = book.latest_ref(
        SEQUENCE_DESCRIPTOR, (table.owner, int(sequence_id)),
    )
    if descriptor is None or cell is None:
        return None
    contracts = tuple(
        parent for parent, _stage in book.edges_into(cell)
        if parent.page == SEQUENCE_CONTRACT_CONCORDANCE
    )
    if len(contracts) != 1:
        return None
    visited = set()
    pending = [contracts[0]]
    identities = set()
    while pending:
        contract = pending.pop()
        if contract in visited:
            continue
        visited.add(contract)
        for parent, _stage in book.edges_into(contract):
            if parent.page == CANONICAL_VALUE:
                try:
                    identities.add(origin_cell(book, parent))
                except ValueError:
                    return None
            elif parent.page == SEQUENCE_CONTRACT_CONCORDANCE:
                pending.append(parent)
    if len(identities) != 1:
        return None
    return descriptor, cell, contracts[0], next(iter(identities))


def public_sequence_input_ports(
    module: Any, function_name: str,
) -> tuple[PublicSequenceInputPorts, ...]:
    """Read bytes input ports from one descriptor and its authored class.

    The public formal IDs select the resident descriptor. Its canonical
    contract and the authored class must reach the same ingestion value.
    Returned slices and their capacity forwarding are irrelevant here.
    """

    from .concordance_declarations import SOURCE_VALUE_CLASS

    function = module.functions.get(str(function_name))
    table = (getattr(module, "sequence_tables", {}) or {}).get(str(function_name))
    book = None if table is None else table.book
    if function is None or book is None:
        return ()
    arguments = {int(value.id): value for value in function.args}
    result = []
    for sequence_id in table.sequences:
        identity = _sequence_descriptor_identity(table, int(sequence_id))
        if identity is None:
            continue
        descriptor, descriptor_cell, contract_cell, source_cell = identity
        columns = tuple(int(value) for value in descriptor.column_value_ids)
        if len(columns) != 1 or columns[0] != int(sequence_id):
            continue
        data_id = columns[0]
        length_id = int(descriptor.length_address_id)
        capacity_id = int(descriptor.capacity_value_id)
        if any(value_id not in arguments for value_id in (
            data_id, length_id, capacity_id,
        )):
            continue
        data = arguments[data_id]
        accounting = dict(data.accounting or {})
        if (
            not accounting.get("program_abi_parameter")
            or accounting.get("program_abi_storage") != "span"
            or accounting.get("program_abi_rank") != 1
        ):
            continue
        # Only the authored class birth is a source-class authority. A
        # specialized peer with the same numeric ID cannot identify this
        # formal unless its exact ingestion cause is the descriptor origin.
        classes = []
        for row in book.page(SOURCE_VALUE_CLASS).rows():
            class_cell = book.latest_ref(SOURCE_VALUE_CLASS, row)
            if class_cell is None:
                continue
            class_fact = book.page(SOURCE_VALUE_CLASS).cell(
                class_cell.row, class_cell.column,
            )
            if not isinstance(class_fact, tuple) or class_fact[:1] != (
                "builtins.bytes",
            ):
                continue
            causes = {parent for parent, _stage in book.edges_into(class_cell)}
            if source_cell in causes:
                classes.append(class_cell)
        if len(classes) != 1:
            continue
        result.append(PublicSequenceInputPorts(
            data_id, length_id, capacity_id, descriptor_cell,
            contract_cell, source_cell, classes[0],
        ))
    return tuple(result)


def public_sequence_capacity_sources(
    module: Any, function_name: str,
) -> tuple[PublicSequenceCapacitySource, ...]:
    """Read bytes-slice capacity from the source bound and both SSA views.

    `sequence_capacity_sources` is only a forwarding index. The authority is
    the source bound joined by unique canonical origins to the two sequence
    contracts, and the actual Ret/output and public Input SSA roles.
    """

    from .concordance_declarations import (
        SOURCE_SEQUENCE_CAPACITY_BOUND, SourceSequenceCapacityBound,
    )
    from .concorded_literals import origin_cell
    from .identity_concordance import CorrelationTable

    function = module.functions.get(str(function_name))
    table = (getattr(module, "sequence_tables", {}) or {}).get(str(function_name))
    book = None if table is None else table.book
    if function is None or table is None or book is None:
        return ()
    if CorrelationTable._source_sequence_capacity_bound_findings(module):
        return ()
    returned = {
        int(value.id)
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret"
        and not instruction.attributes.get("fault_exit")
        for value in instruction.args
    }
    public_inputs = {
        int(value.id)
        for value in function.args
        if (value.accounting or {}).get("program_abi_parameter")
    }

    bound_page = book.page(SOURCE_SEQUENCE_CAPACITY_BOUND)
    relations = []
    for output_id, input_id in function.metadata.get(
        "sequence_capacity_sources", ()
    ):
        output_id, input_id = int(output_id), int(input_id)
        if output_id not in returned or input_id not in public_inputs:
            continue
        output = _sequence_descriptor_identity(table, output_id)
        source = _sequence_descriptor_identity(table, input_id)
        if output is None or source is None:
            continue
        output_descriptor, output_cell, _output_contract, output_origin = output
        source_descriptor, source_cell, _source_contract, source_origin = source
        candidates = []
        for row in bound_page.rows():
            bound_cell = book.latest_ref(SOURCE_SEQUENCE_CAPACITY_BOUND, row)
            fact = (
                None if bound_cell is None else
                bound_page.cell(bound_cell.row, bound_cell.column)
            )
            if not isinstance(fact, SourceSequenceCapacityBound):
                continue
            try:
                matches = (
                    origin_cell(book, fact.destination_cell) == output_origin
                    and origin_cell(book, fact.source_cell) == source_origin
                )
            except ValueError:
                continue
            if matches:
                candidates.append(bound_cell)
        if len(candidates) != 1:
            continue
        if (len(output_descriptor.column_value_ids) != 1
                or len(source_descriptor.column_value_ids) != 1
                or int(output_descriptor.column_value_ids[0]) != output_id
                or int(source_descriptor.column_value_ids[0]) != input_id):
            continue
        relations.append(PublicSequenceCapacitySource(
            output_id, input_id,
            output_id, int(output_descriptor.length_address_id),
            int(output_descriptor.capacity_value_id),
            input_id, int(source_descriptor.length_address_id),
            int(source_descriptor.capacity_value_id),
            candidates[0], output_cell, source_cell,
        ))
    return tuple(relations)


def is_compiler_owned_storage(value: Any) -> bool:
    """Whether a root-frame value is workspace rather than authored ABI."""

    accounting = dict(getattr(value, "accounting", None) or {})
    if (
        accounting.get("program_abi_parameter")
        or accounting.get("caller_provided_workspace")
    ):
        return False
    return any(accounting.get(key) not in {None, ""} for key in (
        "linked_call_frame_storage",
        "returned_record_storage",
        "compiler_frame_storage",
    ))


def is_structural_abi_value(function: Any, value: Any) -> bool:
    """Whether a root formal is a descriptor rather than physical storage:
    a declared record parameter, an aggregate, or a keyed ProgramABI slot."""

    metadata = getattr(function, "metadata", None) or {}
    records = dict(metadata.get("parameter_record_abi") or {})
    record_parameter_ids = {
        int(value_id)
        for parameter_name, value_id in metadata.get("parameter_names", ())
        if str(parameter_name) in records
    }
    accounting = dict(getattr(value, "accounting", None) or {})
    return (
        int(value.id) in record_parameter_ids
        or str(getattr(value, "dtype", "") or "").casefold() == "ssa.aggregate"
        or accounting.get("program_abi_storage") == "keyed"
    )


def _static_shape(value: Any) -> tuple[int, ...]:
    shape = tuple(getattr(value, "shape", ()) or ())
    if not shape or any(not isinstance(item, int) or item < 0 for item in shape):
        return ()
    return tuple(map(int, shape))


def module_storage_requirements(
    module: Any,
) -> Mapping[str, Mapping[int, SSAStorageRequirement]]:
    """Recover storage requirements and propagate them across call bindings."""

    functions = getattr(module, "functions", module) or {}
    values: dict[str, dict[int, list[Any]]] = {
        str(name): {} for name in functions
    }
    shapes: dict[str, dict[int, set[tuple[int, ...]]]] = {
        str(name): {} for name in functions
    }
    dynamic_ids: dict[str, set[int]] = {str(name): set() for name in functions}

    def observe(function_name: str, value: Any) -> None:
        if value is None or not hasattr(value, "id"):
            return
        value_id = int(value.id)
        values[function_name].setdefault(value_id, []).append(value)
        shape = _static_shape(value)
        if shape:
            shapes[function_name].setdefault(value_id, set()).add(shape)

    for function_name, function in functions.items():
        function_name = str(function_name)
        integer_constants: dict[int, int] = {}
        address_indices: dict[int, list[tuple[int, ...] | None]] = {}
        for value in getattr(function, "args", ()):
            observe(function_name, value)
        for block in getattr(function, "blocks", {}).values():
            for instruction in getattr(block, "instrs", ()):
                for value in getattr(instruction, "args", ()):
                    observe(function_name, value)
                observe(function_name, getattr(instruction, "res", None))
                if (
                    str(instruction.op).casefold() in {"const", "constant"}
                    and instruction.res is not None
                ):
                    attributes = getattr(instruction, "attributes", {}) or {}
                    constant = _constant_integer(instruction)
                    if constant is not None:
                        integer_constants[int(instruction.res.id)] = constant
                    else:
                        literal = attributes.get("llvm_literal")
                        match = (
                            re.fullmatch(
                                r"i(?:1|8|16|32|64)\s+([-+]?\d+)",
                                literal.strip(),
                            )
                            if isinstance(literal, str) else None
                        )
                        if match is not None:
                            integer_constants[int(instruction.res.id)] = int(
                                match.group(1)
                            )
                if (
                    str(instruction.op).casefold() == "getelementptr"
                    and len(getattr(instruction, "args", ())) > 1
                ):
                    base_id = int(instruction.args[0].id)
                    indices = tuple(
                        integer_constants.get(int(index.id))
                        for index in instruction.args[1:]
                    )
                    address_indices.setdefault(base_id, []).append(
                        tuple(map(int, indices))
                        if all(index is not None for index in indices)
                        else None
                    )
        # When every address operation over one otherwise-rankless base uses
        # compile-time indices, those accesses prove the minimum physical span
        # the function can touch.  This is storage evidence, not a guessed
        # semantic tensor shape, and it propagates to the owning caller below.
        for value_id, accesses in address_indices.items():
            if not accesses or any(access is None for access in accesses):
                continue
            ranks = {len(access) for access in accesses if access is not None}
            if len(ranks) != 1 or next(iter(ranks)) < 1:
                continue
            rank = next(iter(ranks))
            shape = tuple(
                max(access[axis] for access in accesses if access is not None) + 1
                for axis in range(rank)
            )
            if all(extent > 0 for extent in shape):
                shapes[function_name].setdefault(value_id, set()).add(shape)

    tensor_tables = dict(getattr(module, "tensor_tables", {}) or {})
    # Dynamic topk keeps its logical extent in SSA metadata.  Its separate
    # concorded capacity is the input tensor's static physical span: Min(k,
    # axis_extent) can never require more storage on that axis.
    from .concordance_declarations import TENSOR_OUTPUT_CAPACITY_BOUND
    from .ssa_record_return_state import function_scope_of
    capacity_book = (getattr(module, "metadata", {}) or {}).get("identity_book")
    if capacity_book is not None:
        capacity_page = capacity_book.page(TENSOR_OUTPUT_CAPACITY_BOUND)
        for row in capacity_page.rows():
            bound = capacity_page.latest(row)
            if bound is None or not bound.capacity_shape:
                continue
            if int(row[1]) != int(bound.output_cell.row[-1]):
                continue
            for owner, function in functions.items():
                if function_scope_of(function) != row[0]:
                    continue
                if int(row[1]) not in values[str(owner)]:
                    continue
                shapes[str(owner)].setdefault(int(row[1]), set()).add(
                    tuple(map(int, bound.capacity_shape))
                )
    for function_name, table in (getattr(module, "sequence_tables", {}) or {}).items():
        function = functions.get(function_name)
        if function is None:
            continue
        # Program: ``sum(i * k for i in range(6))``.  Wrong record: the
        # control builder publishes each static producer loop's extent as
        # ``sequence_static_capacity_bounds`` (the generator arena holds at
        # most 6 rows), but only ``static_mapping_capacity_bounds`` was read,
        # so the arena had no element count and emission refused it as
        # dynamic growth.  Rule: both records state an upper bound on the
        # rows of one resident sequence; each bounds its column storage.
        static_bounds = (
            *dict(function.metadata.get("static_mapping_capacity_bounds", {})).items(),
            *tuple(function.metadata.get("sequence_static_capacity_bounds", ())),
        )
        for sequence_id, count in static_bounds:
            descriptor = table.sequences.get(int(sequence_id))
            if descriptor is None:
                continue
            row_shapes = descriptor.column_shapes or tuple(
                () for _column in descriptor.column_value_ids
            )
            for column_id, row_shape in zip(
                descriptor.column_value_ids, row_shapes, strict=True,
            ):
                shapes[function_name].setdefault(int(column_id), set()).add(
                    (max(1, int(count)), *tuple(row_shape))
                )
            # Program: ``d = {1: n, 2: 3}; del d[1]; len(d)``.  Wrong
            # record: the live-flag arena (one flag per row) had no bound,
            # so it was allocated one element and the second row's flag was
            # written past it.  Rule: live flags are bounded like a column.
            if descriptor.live_flags_value_id is not None:
                shapes[function_name].setdefault(
                    int(descriptor.live_flags_value_id), set()
                ).add((max(1, int(count)),))
        # Program: ``g = [[1, 2], [3, n]]; g[i][1]`` (corpus c732).  A list
        # of lists keeps one fixed-stride slice per row in its child-table
        # pool: ``rows`` slices of ``stride`` elements.  The pool's columns
        # and live flags hold ``rows * stride`` elements; its length and
        # status arenas one cell per row.
        for outer_id, rows, stride in tuple(
            function.metadata.get("child_pool_static_extents", ())
        ):
            descriptor = table.sequences.get(int(outer_id))
            pool = None if descriptor is None else descriptor.child_table_pool
            if pool is None:
                continue
            for column_id in pool.column_value_ids:
                shapes[function_name].setdefault(int(column_id), set()).add(
                    (int(rows) * int(stride),)
                )
            if pool.live_flags_value_id is not None:
                shapes[function_name].setdefault(
                    int(pool.live_flags_value_id), set()
                ).add((int(rows) * int(stride),))
            for member_id in (pool.length_value_id, pool.status_value_id):
                if member_id is not None:
                    shapes[function_name].setdefault(
                        int(member_id), set()
                    ).add((int(rows),))
    for function_name, table in tensor_tables.items():
        function_name = str(function_name)
        if function_name not in functions:
            continue
        for descriptor in getattr(table, "tensors", {}).values():
            value_id = int(descriptor.data_value_id)
            if descriptor.metadata_state == "static" and tuple(descriptor.shape):
                shapes[function_name].setdefault(value_id, set()).add(
                    tuple(map(int, descriptor.shape))
                )
            elif descriptor.metadata_state != "static":
                dynamic_ids[function_name].add(value_id)

    for function_name, function in functions.items():
        function_name = str(function_name)
        for item in (getattr(function, "metadata", {}) or {}).get(
            "storage_formals", ()
        ):
            shape = tuple(map(int, item.get("shape") or ()))
            if shape:
                shapes[function_name].setdefault(
                    int(item["value_id"]), set()
                ).add(shape)

    # Formal/actual bindings describe the same physical span. Aggregate
    # projections similarly bind a callee output slot to caller storage.
    from .ssa_aggregate_abi import analyze_aggregate_abi
    aggregate_calls = {
        id(record.call): record
        for record in analyze_aggregate_abi(module, tuple(functions)).calls
    }

    def join_dynamic(left_function, left_id, right_function, right_id):
        left = dynamic_ids[left_function]
        right = dynamic_ids[right_function]
        if int(left_id) not in left and int(right_id) not in right:
            return False
        changed = int(left_id) not in left or int(right_id) not in right
        left.add(int(left_id))
        right.add(int(right_id))
        return changed

    changed = True
    while changed:
        changed = False
        for caller_name, caller in functions.items():
            caller_name = str(caller_name)
            sequence_table = (getattr(module, "sequence_tables", {}) or {}).get(caller_name)
            for sequence_id, source_id in caller.metadata.get("sequence_capacity_sources", ()):
                descriptor = None if sequence_table is None else sequence_table.sequences.get(int(sequence_id))
                if descriptor is None:
                    continue
                # A retained comprehension emits at most one row per input
                # element. Filtering/deduplication can only reduce that count.
                outer_bounds = {
                    max(1, shape[0])
                    for shape in shapes[caller_name].get(int(source_id), ())
                    if shape
                }
                row_shapes = descriptor.column_shapes or tuple(
                    () for _column in descriptor.column_value_ids
                )
                for column_id, row_shape in zip(
                    descriptor.column_value_ids, row_shapes, strict=True,
                ):
                    bounds = {
                        (outer, *tuple(row_shape)) for outer in outer_bounds
                    }
                    column_shapes = shapes[caller_name].setdefault(int(column_id), set())
                    if not bounds.issubset(column_shapes):
                        column_shapes.update(bounds)
                        changed = True
                if descriptor.live_flags_value_id is not None:
                    bounds = {(outer,) for outer in outer_bounds}
                    flag_shapes = shapes[caller_name].setdefault(
                        int(descriptor.live_flags_value_id), set()
                    )
                    if not bounds.issubset(flag_shapes):
                        flag_shapes.update(bounds)
                        changed = True
            # Program: ``count_true([True, False, True])`` where the callee
            # loops ``for b in bs: if b: ...`` (c197; lane E).  Wrong
            # record: the callee's list formal keeps live flags, its column
            # was bounded to three rows, but the flags were bounded only by
            # the static and comprehension bounds above, so the flag formal
            # and the caller actual minted for it had no element count and
            # the Fortran entry asked for an extent no feed can supply.
            # Rule: a live-flag arena holds one flag per row of its
            # sequence, so each bound on the first column's rows bounds it.
            for descriptor in (
                () if sequence_table is None
                else sequence_table.sequences.values()
            ):
                if (
                    descriptor.live_flags_value_id is None
                    or not descriptor.column_value_ids
                ):
                    continue
                row_rank = len(tuple(
                    (descriptor.column_shapes or ((),))[0]
                ))
                bounds = {
                    (shape[0],)
                    for shape in shapes[caller_name].get(
                        int(descriptor.column_value_ids[0]), ()
                    )
                    if len(shape) == row_rank + 1
                }
                flag_shapes = shapes[caller_name].setdefault(
                    int(descriptor.live_flags_value_id), set()
                )
                if not bounds.issubset(flag_shapes):
                    flag_shapes.update(bounds)
                    changed = True
            for block in getattr(caller, "blocks", {}).values():
                instructions = tuple(getattr(block, "instrs", ()))
                for call in instructions:
                    if call.op not in {"Call", "call"}:
                        continue
                    callee_name = str(call.attributes.get("callee") or "")
                    callee = functions.get(callee_name)
                    if callee is None:
                        continue
                    for actual, formal in zip(call.args, callee.args):
                        changed |= join_dynamic(
                            caller_name, actual.id, callee_name, formal.id,
                        )
                        actual_shapes = shapes[caller_name].setdefault(
                            int(actual.id), set()
                        )
                        formal_shapes = shapes[callee_name].setdefault(
                            int(formal.id), set()
                        )
                        union = actual_shapes | formal_shapes
                        if union != actual_shapes:
                            actual_shapes.update(union)
                            changed = True
                        if union != formal_shapes:
                            formal_shapes.update(union)
                            changed = True
                    aggregate = aggregate_calls.get(id(call))
                    named_outputs = tuple(
                        value_id
                        for _name, value_id in (
                            getattr(callee, "metadata", {}) or {}
                        ).get("named_outputs", ())
                    )
                    if call.res is None or (not named_outputs and aggregate is None):
                        continue
                    projected: dict[int, int] = {}
                    addresses: dict[int, int] = {}
                    for candidate in instructions:
                        if (
                            candidate.op in {"GetElementPtr", "getelementptr"}
                            and candidate.res is not None
                            and candidate.args
                            and int(candidate.args[0].id) == int(call.res.id)
                            and candidate.attributes.get("aggregate_index") is not None
                        ):
                            addresses[int(candidate.res.id)] = int(
                                candidate.attributes["aggregate_index"]
                            )
                        elif (
                            candidate.op in {"Load", "load"}
                            and candidate.res is not None
                            and candidate.args
                            and int(candidate.args[0].id) in addresses
                        ):
                            projected[addresses[int(candidate.args[0].id)]] = int(
                                candidate.res.id
                            )
                    # arange -> cast -> sum returns dynamic spans through
                    # ordinary region aggregate projections. Those regions
                    # have no named_outputs table; reading only that table
                    # lost the callee's dynamic storage state and described
                    # root values 6/8 as scalar one-cell allocations. The
                    # common aggregate ABI is the exact output binding.
                    output_bindings = (
                        tuple((projection.output_id, projection.value.id)
                              for projection in aggregate.projections)
                        if aggregate is not None else tuple(
                            (value_id, projected.get(index))
                            for index, value_id in enumerate(named_outputs)
                        )
                    )
                    for callee_value_id, caller_value_id in output_bindings:
                        if caller_value_id is None:
                            continue
                        changed |= join_dynamic(
                            caller_name, caller_value_id,
                            callee_name, callee_value_id,
                        )
                        caller_shapes = shapes[caller_name].setdefault(
                            caller_value_id, set()
                        )
                        callee_shapes = shapes[callee_name].setdefault(
                            int(callee_value_id), set()
                        )
                        union = caller_shapes | callee_shapes
                        if union != caller_shapes:
                            caller_shapes.update(union)
                            changed = True
                        if union != callee_shapes:
                            callee_shapes.update(union)
                            changed = True

    result: dict[str, dict[int, SSAStorageRequirement]] = {}
    for function_name, function_values in values.items():
        requirements: dict[int, SSAStorageRequirement] = {}
        table = tensor_tables.get(function_name)
        for value_id, occurrences in function_values.items():
            ordered_views = tuple(sorted(
                shapes[function_name].get(value_id, ()),
                key=lambda item: (prod(item), item),
            ))
            shape = ordered_views[-1] if ordered_views else ()
            # A value with an exact static descriptor in ITS OWN function
            # has exactly that storage.  The cross-call view union above is
            # evidence for shapeless values only; letting its largest member
            # override an exact descriptor gave a (8,4,2,3) temporary the
            # storage of an unrelated (3,800,1600) leaf, and every multi-axis
            # address into it then had no extent origin.
            # ``SSATensorTable.by_id`` is keyed by TENSOR id; the value-keyed
            # reader is ``by_data_value``.  Reading ``by_id(value_id)`` here
            # handed ``catalog_clamp``'s count Const (value 2, shape ``()``)
            # the descriptor of tensor 2 (data value 6, shape ``(4,)``), and
            # the Fortran lane then refused the ``binary_scalar_double`` call
            # whose ``n`` operand it saw as a four-element array.  A value's
            # own storage is the allocation-owning static descriptor over it;
            # a view shares the owner's data value with its own shape and
            # ``owns_allocation=False``, so it is not the storage.
            own_storage = (
                set()
                if table is None
                else {
                    tuple(map(int, own.shape))
                    for own in table.by_data_value(int(value_id))
                    if own.metadata_state == "static"
                    and tuple(own.shape)
                    and own.owns_allocation
                }
            )
            if len(own_storage) > 1:
                raise ValueError(
                    f"{function_name}: value {value_id} has "
                    f"{len(own_storage)} allocation-owning static tensor "
                    f"descriptors with different shapes "
                    f"{sorted(own_storage)}"
                )
            if own_storage:
                shape = next(iter(own_storage))
                if shape not in ordered_views:
                    ordered_views = (*ordered_views, shape)
            dtype = next((
                str(value.dtype)
                for value in occurrences
                if getattr(value, "dtype", None)
            ), DEFAULT_DTYPE)
            requirements[value_id] = SSAStorageRequirement(
                value_id=value_id,
                dtype=dtype,
                shape=shape,
                element_count=(prod(shape) if shape else None),
                views=ordered_views,
                dynamic=(
                    not shape and value_id in dynamic_ids[function_name]
                ),
            )
        result[function_name] = requirements
    return result


def function_storage_requirements(
    module: Any, function_name: str,
) -> Mapping[int, SSAStorageRequirement]:
    """Return the propagated requirements for one function."""

    return module_storage_requirements(module)[str(function_name)]


__all__ = [
    "OwnedTensorDataBinding", "owned_tensor_data_bindings",
    "PublicSequenceCapacitySource", "PublicSequenceInputPorts",
    "PublicOwnedSequencePorts", "SSAStorageRequirement",
    "function_storage_requirements", "public_sequence_capacity_sources",
    "public_sequence_input_ports", "public_owned_sequence_returns",
    "is_compiler_owned_storage", "module_storage_requirements",
]
