"""Record field lowering: planned-region record projection captures, record receipts, field reads, optional record presence lowering (relocated from fortran_c_shell)"""

from __future__ import annotations
import ast
from typing import (
    Any,
    Iterable,
    Mapping,
)
from types import SimpleNamespace
from fnmatch import fnmatchcase

from .function_alias_concordance import _concordant_function_aliases


def _lower_planned_region_record_projection_captures(
    functions: Mapping[str, Any],
    record_tables: Mapping[str, Any],
) -> int:
    """Pass proven physical record fields into numerical regions explicitly.

    A planned numerical region cannot interpret a conceptual Python record
    handle.  Late record linking can nevertheless leave an initial-state
    ``getattr`` in such a region after the owner has already resolved that
    projection to one physical scalar or span field.  Move that exact resident
    value across the region boundary and replace the projection with a cast.

    The proof is deliberately closed: the region names one owner, every
    projection result reaches a finite owner alias chain, the call's receiver
    names an owner record, and that record's named scalar field contains the
    alias incumbent.  Cycles, ambiguity, and incompatible candidates make no
    change.  Equal evidence reuses the first capture already installed for
    the resident field.
    """

    from ..transmogrifier.ssa import Instr, SSARecordFieldStorage, SSAValue

    def function_values(function: Any) -> dict[int, Any]:
        values = {int(value.id): value for value in function.args}
        values.update(
            (int(instruction.res.id), instruction.res)
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        )
        return values

    def alias_incumbent(function: Any, value_id: int) -> int | None:
        aliases = _concordant_function_aliases(function)
        current = int(value_id)
        seen: set[int] = set()
        followed = False
        while current in aliases:
            if current in seen:
                return None
            seen.add(current)
            current = aliases[current]
            followed = True
        return current if followed else None

    def record_incumbent(function: Any, value_id: int) -> int | None:
        aliases = _concordant_function_aliases(
            function, include_output_identities=True,
        )
        current = int(value_id)
        seen: set[int] = set()
        while current in aliases:
            if current in seen:
                return None
            seen.add(current)
            current = aliases[current]
        return current

    def calls_into(callee_name: str) -> list[tuple[Any, Any]]:
        return [
            (caller, instruction)
            for caller in functions.values()
            for block in caller.blocks.values()
            for instruction in block.instrs
            if (
                str(instruction.op).casefold() == "call"
                and str(instruction.attributes.get("callee") or "")
                == str(callee_name)
            )
        ]

    lowered = 0
    for region_name, region in functions.items():
        integral = dict(
            (region.metadata or {}).get("source_region_integral") or {}
        )
        owner_name = str(integral.get("owner") or "")
        owner = functions.get(owner_name)
        owner_records = record_tables.get(owner_name)
        caller_calls = calls_into(str(region_name))
        if owner is None or owner_records is None or not caller_calls:
            continue
        if any(caller is not owner for caller, _call in caller_calls):
            continue

        owner_values = function_values(owner)
        region_values = function_values(region)
        formal_positions = {
            int(formal.id): index for index, formal in enumerate(region.args)
        }
        candidates: list[tuple[Any, int, str, int, Any]] = []
        for block in region.blocks.values():
            for instruction in block.instrs:
                if (
                    str(instruction.op).casefold() != "getattr"
                    or instruction.res is None
                    or len(instruction.args) != 1
                ):
                    continue
                receiver_id = int(instruction.args[0].id)
                receiver_position = formal_positions.get(receiver_id)
                attribute = str(
                    instruction.attributes.get("attribute")
                    or instruction.attributes.get("name")
                    or ""
                )
                resident_id = alias_incumbent(owner, int(instruction.res.id))
                resident = (
                    None if resident_id is None
                    else owner_values.get(resident_id)
                )
                if (
                    receiver_position is None
                    or resident is None
                ):
                    continue

                proven = True
                for _caller, call in caller_calls:
                    if len(call.args) != len(region.args):
                        proven = False
                        break
                    declared = call.attributes.get("callee_input_ids")
                    if declared is not None:
                        declared = tuple(map(int, declared))
                        if (
                            len(declared) != len(call.args)
                            or receiver_id not in declared
                        ):
                            proven = False
                            break
                        actual = call.args[declared.index(receiver_id)]
                    else:
                        actual = call.args[receiver_position]
                    actual_record_id = int(actual.id)
                    same_receiver = (
                        record_incumbent(owner, actual_record_id)
                        == record_incumbent(owner, receiver_id)
                    )
                    descriptor = owner_records.records.get(actual_record_id)
                    if descriptor is None and same_receiver:
                        # An exact returned-record identity call has no native
                        # scalar result: the call mutates the receiver fields
                        # in place. Region planning still names the authored
                        # result handle while its scheduled call operand names
                        # the receiver handle. The owner's identity ledgers are
                        # the proof that these are the same record; use the
                        # descriptor already registered for either spelling.
                        descriptor = (
                            owner_records.records.get(receiver_id)
                            or owner_records.records.get(
                                int(record_incumbent(owner, receiver_id))
                            )
                        )
                    if not attribute and descriptor is not None:
                        # Region lowering can consume the source-only getattr
                        # annotation after assigning the projection its SSA
                        # result. The record table and the owner's exact result
                        # alias survive. Recover the name only when that
                        # resident belongs to one scalar field of this exact
                        # receiver descriptor.
                        resident_fields = tuple(
                            field.name for field in descriptor.fields
                            if (
                                field.storage in {
                                    SSARecordFieldStorage.SCALAR,
                                    SSARecordFieldStorage.SPAN,
                                }
                                and resident_id in map(int, field.value_ids)
                            )
                        )
                        if len(resident_fields) == 1:
                            attribute = str(resident_fields[0])
                    matching = tuple(
                        field for field in (() if descriptor is None else descriptor.fields)
                        if (
                            field.name == attribute
                            and field.storage in {
                                SSARecordFieldStorage.SCALAR,
                                SSARecordFieldStorage.SPAN,
                            }
                            and resident_id in map(int, field.value_ids)
                        )
                    )
                    if descriptor is not None and len(matching) != 1:
                        proven = False
                        break
                    if descriptor is None and not same_receiver:
                        proven = False
                        break
                if proven:
                    candidates.append((
                        instruction, receiver_id, attribute,
                        int(resident_id), resident,
                    ))

        captures: dict[int, Any] = {}
        receipts: list[tuple[int, str, int, int]] = []
        for instruction, receiver_id, attribute, resident_id, resident in candidates:
            capture = captures.get(resident_id)
            if capture is None:
                occupied = region_values.get(resident_id)
                if occupied is not None:
                    # A pre-existing definition has priority.  It is usable
                    # only when it is already the same formal capture.
                    if occupied not in region.args:
                        continue
                    capture = occupied
                else:
                    capture = SSAValue(
                        resident_id,
                        dtype=resident.dtype,
                        shape=tuple(resident.shape or ()),
                        device=resident.device,
                        accounting={
                            **dict(resident.accounting or {}),
                            "source_record_field_capture": attribute,
                            "source_record_receiver_id": receiver_id,
                            "source_value_id": resident_id,
                            "capture_owner": owner_name,
                        },
                    )
                    region.args.append(capture)
                    region_values[resident_id] = capture
                    for _caller, call in caller_calls:
                        old_arity = len(call.args)
                        call.args.append(resident)
                        for key, value in (
                            ("feed_ids", resident_id),
                            ("feed_shapes", tuple(resident.shape or ())),
                            ("feed_dtypes", resident.dtype),
                            ("input_ids", resident_id),
                            ("callee_input_ids", resident_id),
                        ):
                            sequence = call.attributes.get(key)
                            if sequence is not None and len(tuple(sequence)) == old_arity:
                                call.attributes[key] = (*tuple(sequence), value)
                    capture_ids = list(map(
                        int, integral.get("capture_value_ids") or ()
                    ))
                    if resident_id not in capture_ids:
                        capture_ids.append(resident_id)
                    integral["capture_value_ids"] = tuple(capture_ids)
                captures.setdefault(resident_id, capture)

            instruction.op = "Cast"
            instruction.args = [capture]
            instruction.arg_roles = []
            instruction.attributes = {
                "target_dtype": str(
                    instruction.res.dtype or capture.dtype or "float64"
                ),
                "resolved_record_field_projection": True,
                "record_receiver_id": receiver_id,
                "record_field": attribute,
                "resident_value_id": resident_id,
                "tie_policy": "incumbent",
            }
            receipts.append((
                int(instruction.res.id), attribute, receiver_id, resident_id,
            ))
            lowered += 1

        if receipts:
            replaced_receivers = {receipt[2] for receipt in receipts}
            still_consumed = {
                int(argument.id)
                for block in region.blocks.values()
                for instruction in block.instrs
                for argument in instruction.args
            }
            integral["capture_value_ids"] = tuple(
                value_id
                for value_id in integral.get("capture_value_ids", ())
                if (
                    int(value_id) not in replaced_receivers
                    or int(value_id) in still_consumed
                )
            )
            region_metadata = dict(region.metadata or {})
            region_metadata["source_region_integral"] = integral
            existing = tuple(
                region_metadata.get("lowered_record_projection_captures", ())
            )
            region_metadata["lowered_record_projection_captures"] = tuple(
                dict.fromkeys((*existing, *receipts))
            )
            region.metadata = region_metadata
    return lowered


def _record_receipts_for_function(
    program_abi: Mapping[str, Any],
    function_name: str,
    parameters: Iterable[str],
    *,
    method_owner: str | None = None,
) -> dict[str, Mapping[str, Any]]:
    """Select explicit record bindings plus an exact method receiver schema.

    The receipt form is used after the extraction contract object has been
    reduced to serializable graph metadata.  Keep its selection semantics in
    step with :meth:`ProgramABIContract.records_for_function`: an unannotated
    ``self`` is safely identifiable from ``method_owner`` only when exactly
    one declared record has that class identity.
    """

    parameter_names = set(map(str, parameters))
    records = dict(program_abi.get("records") or {})
    selected: dict[str, Mapping[str, Any]] = {}
    for binding in tuple(program_abi.get("bindings") or ()):
        parameter = str(binding.get("parameter") or "")
        record_name = str(binding.get("record") or "")
        if (
            parameter in parameter_names
            and fnmatchcase(str(function_name), str(binding.get("function") or ""))
            and record_name in records
        ):
            selected[parameter] = records[record_name]
    if (
        method_owner is not None
        and "self" in parameter_names
        and "self" not in selected
    ):
        owner = str(method_owner)
        candidates = tuple(
            record
            for name, record in records.items()
            if (
                str(name) == owner
                or str(record.get("identity") or "") == owner
                or str(record.get("identity") or "").rsplit(".", 1)[-1]
                == owner
            )
        )
        if len(candidates) == 1:
            selected["self"] = candidates[0]
    return selected


def _record_field_read(graph: Any, node_id: Any, data: Mapping[str, Any]):
    """Return ``(result, owner, field)`` for either spelling of a field read.

    Python's ``obj.field`` is already a GetAttr graph node.  The equivalent
    ``getattr(obj, "field", default)`` remains an intrinsic Call until SSA
    lowering, but it names the same ProgramABI storage and presence bit.
    Recognize both from exact graph roles and the literal field name.
    """

    operation = str(data.get("type") or data.get("op") or "").casefold()
    parents = tuple(data.get("parents") or ())
    if operation == "getattr":
        field = (data.get("attributes") or {}).get("attribute")
        owner = next((
            parent for parent, role in parents
            if str(role) in {"value", "object", "base", "receiver"}
        ), None)
    elif (
        operation in {"call", "plancall"}
        and str((data.get("attributes") or {}).get(
            "static_python_reference"
        ) or "") in {"getattr", "builtins.getattr"}
    ):
        owner = next((
            parent for parent, role in parents if str(role) == "arg:0"
        ), None)
        field_node = next((
            parent for parent, role in parents if str(role) == "arg:1"
        ), None)
        field_data = graph.nodes.get(field_node, {})
        field_attributes = field_data.get("attributes") or {}
        field = field_attributes.get(
            "value", field_data.get("constant")
        )
    else:
        return None
    if owner is None or not isinstance(field, str):
        return None
    return (
        int(data.get("value_id", node_id)),
        int(graph.nodes[owner].get("value_id", owner)),
        field,
    )


def _exact_structural_value_origin(graph_obj: Any, value_id: int) -> int:
    """Follow only proved identity-return forwarding to its source value.

    ``_fold_callsite_structural_values`` records an exact callee-return fact on
    the call result.  A field read through ``alias = identity(record)`` is
    therefore still a projection of the caller-owned record, not a new
    anonymous storage location.  Resolve by value identity, and stop if a
    value has no unique graph node or if a malformed cycle is encountered.
    """

    current = int(value_id)
    visited: set[int] = set()
    while current not in visited:
        visited.add(current)
        candidates = [
            int(node_id)
            for node_id, data in graph_obj.nodes(data=True)
            if int(data.get("value_id", node_id)) == current
        ]
        if len(candidates) != 1:
            break
        forwarded = (
            graph_obj.nodes[candidates[0]].get("attributes") or {}
        ).get("structural_identity_actual_value_id")
        if forwarded is None:
            break
        current = int(forwarded)
    return current


def _lower_optional_record_presence_graph(graph_obj: Any) -> int:
    """Give each optional record field one explicit presence identity.

    Source ``is None``/``is not None`` tests are identity tests, not numeric
    comparisons against the payload.  Before region planning, replace every
    such test with a read of one canonical Boolean input for the field (or its
    logical negation).  All occurrences in a function therefore observe the
    same caller-owned cell, including after a mutable field write.
    """

    graph = graph_obj
    if (graph.graph.get("optional_record_presence_lowered") is not None):
        return int(graph.graph.get("optional_record_presence_lowered") or 0)
    # A discovery graph can be visited before its ProgramABI receipt is
    # reattached after topology reduction.  That visit is not a completed
    # lowering: leave the marker unset so the canonical pre-planning graph
    # can still acquire and lower its optional fields later.
    if not graph.graph.get("parameter_record_abi"):
        return 0
    identities = graph.graph.get("identity_table") or {}
    receipts: list[dict[str, Any]] = []

    def value_id(node_id: Any) -> int:
        return int(graph.nodes[node_id].get("value_id", node_id))

    def is_none_node(node_id: Any) -> bool:
        if node_id not in graph:
            return False
        data = graph.nodes[node_id]
        attributes = data.get("attributes") or {}
        return (
            str(data.get("type") or data.get("op") or "").casefold()
            in {"constant", "const", "none", "nonevalue"}
            and attributes.get("value", data.get("constant")) is None
        )

    from ..common.tensors.topological_reducer import _set_operands

    view = SimpleNamespace(G=graph)

    def detach_inputs(node_id: Any) -> None:
        data = graph.nodes[node_id]
        for parent, role in tuple(data.get("parents") or ()):
            if parent in graph:
                graph.nodes[parent]["children"] = [
                    item for item in graph.nodes[parent].get("children") or ()
                    if not (int(item[0]) == int(node_id) and item[1] == role)
                ]
                if graph.has_edge(parent, node_id):
                    graph.remove_edge(parent, node_id)
        _set_operands(view, node_id, [], cause="optional_presence_detach")

    def replace_uses(old_id: Any, new_id: Any) -> None:
        for child in tuple(graph.successors(old_id)):
            edge = dict(graph.get_edge_data(old_id, child) or {})
            role = edge.get("role")
            child_data = graph.nodes[child]
            _set_operands(view, child, [
                (new_id if parent == old_id else parent, held_role)
                for parent, held_role in child_data.get("parents") or ()
            ], cause="optional_presence_replace", same={old_id: new_id})
            # _set_operands is the one writer of parents, children and the
            # networkx edge (concordance step 9); the hand-written half that
            # followed here re-added the edge and then removed one that
            # _set_operands had already removed, which raised.
        graph.nodes[old_id]["children"] = []

    from .process_graph_value_ids import next_process_value_id

    for parameter, record in dict(
        graph.graph.get("parameter_record_abi") or {}
    ).items():
        owner_ids = set(map(int, identities.get(str(parameter), ())))
        if not owner_ids:
            continue
        for field_name, field in dict(record.get("fields") or {}).items():
            if not bool(field.get("optional", False)):
                continue
            comparisons: list[tuple[Any, bool]] = []
            for getter_id, getter in tuple(graph.nodes(data=True)):
                field_read = _record_field_read(
                    graph, getter_id, getter
                )
                if field_read is None:
                    continue
                _result_id, owner_id, read_field = field_read
                owner_origin = _exact_structural_value_origin(
                    graph, int(owner_id)
                )
                if (
                    read_field != str(field_name)
                    or owner_origin not in owner_ids
                ):
                    continue
                for compare_id in tuple(graph.successors(getter_id)):
                    compare = graph.nodes[compare_id]
                    expression = compare.get("expr_obj")
                    if not (
                        isinstance(expression, ast.Compare)
                        and len(expression.ops) == 1
                        and isinstance(expression.ops[0], (ast.Is, ast.IsNot))
                        and any(
                            is_none_node(parent)
                            for parent, _role in compare.get("parents") or ()
                            if parent != getter_id
                        )
                    ):
                        continue
                    comparisons.append((
                        compare_id, isinstance(expression.ops[0], ast.IsNot),
                    ))
            comparisons = list(dict.fromkeys(comparisons))
            positive = next((
                node_id for node_id, present_when in comparisons if present_when
            ), None)
            if positive is None:
                positive = next_process_value_id(graph)
                graph.add_node(
                    positive,
                    label=f"{parameter}.{field_name}.__present",
                    type="Input",
                    op="input",
                    expr_obj=None,
                    parents=[],
                    children=[],
                    attributes={},
                    value_id=positive,
                    tensor={"shape": (), "dtype": "bool"},
                )
            else:
                detach_inputs(positive)
            positive_data = graph.nodes[positive]
            _set_operands(view, positive, [], cause="optional_presence_input")
            positive_data.update({
                "label": f"{parameter}.{field_name}.__present",
                "type": "Input",
                "op": "input",
                "expr_obj": None,
                "constant": None,
                "tensor": {"shape": (), "dtype": "bool"},
                "attributes": {
                    **dict(positive_data.get("attributes") or {}),
                    "binding_kind": "program_abi_optional_presence",
                    "program_abi_parameter": str(parameter),
                    "program_abi_field": str(field_name),
                    "program_abi_optional_presence": True,
                    "program_abi_optional_present_when": True,
                },
            })
            for compare_id, present_when in comparisons:
                if compare_id == positive:
                    continue
                if present_when:
                    replace_uses(compare_id, positive)
                    detach_inputs(compare_id)
                    graph.remove_node(compare_id)
                    continue
                detach_inputs(compare_id)
                compare = graph.nodes[compare_id]
                # ``x is None`` becomes ``not presence``: the test no longer
                # reads ``x``; its operand position is retired and the
                # presence operand recorded (``identity_transition``).
                _set_operands(
                    view, compare_id, [(positive, "operand")],
                    cause="optional_presence_negation",
                )
                compare.update({
                    "label": "logical_not",
                    "type": "logical_not",
                    "op": "logical_not",
                    "expr_obj": None,
                    "constant": None,
                    "tensor": {"shape": (), "dtype": "bool"},
                    "attributes": {
                        **dict(compare.get("attributes") or {}),
                        "optional_presence_negation": True,
                        "program_abi_parameter": str(parameter),
                        "program_abi_field": str(field_name),
                    },
                })
                graph.add_edge(positive, compare_id, role="operand")
                children = positive_data.setdefault("children", [])
                if (compare_id, "operand") not in children:
                    children.append((compare_id, "operand"))
            receipts.append({
                "parameter": str(parameter),
                "field": str(field_name),
                "presence_value_id": value_id(positive),
                "test_count": len(comparisons),
                "mutable": bool(field.get("mutable", False)),
            })
    graph.graph["optional_record_presence_receipts"] = tuple(receipts)
    graph.graph["optional_record_presence_lowered"] = len(receipts)
    return len(receipts)


def _lower_optional_record_presence_catalogue(graph: Any) -> int:
    """Lower optional-field identity tests throughout one function table.

    Control planning consumes the post-reduction function catalogue, so this
    pass must run on those graphs before deployment strategy is selected.
    The per-graph marker makes repeated catalogue visits finite and stable.
    """

    lowered = 0
    seen: set[int] = set()
    for candidate in (
        graph,
        *(getattr(entry, "graph", None) for entry in graph.function_table),
    ):
        graph_obj = getattr(candidate, "G", None)
        if graph_obj is None or id(graph_obj) in seen:
            continue
        seen.add(id(graph_obj))
        lowered += _lower_optional_record_presence_graph(graph_obj)
    return lowered
