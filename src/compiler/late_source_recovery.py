"""Late source recovery: recover authored literals, slice offsets, unary and pure expressions after lowering; source-graph spelling index (relocated from fortran_c_shell)"""

from __future__ import annotations
import ast
import copy
from typing import (
    Any,
    Mapping,
)


_UNCOPYABLE_LITERAL_TYPES: set[str] = set()


def _copy_literal_payload(payload: Any) -> Any:
    """Deep-copy a literal payload captured off a graph node, if it can be.

    Ingested stdlib source (``re._compiler`` and friends) puts objects behind
    ``Constant`` nodes whose payloads are not always deep-copyable. Sharing
    the original reference is the honest fallback: the payload already had
    that identity everywhere upstream, so aliasing here adds no hazard a copy
    would have removed.
    """

    try:
        return copy.deepcopy(payload)
    except Exception:
        type_name = type(payload).__qualname__
        if type_name not in _UNCOPYABLE_LITERAL_TYPES:
            _UNCOPYABLE_LITERAL_TYPES.add(type_name)
            import warnings

            warnings.warn(
                f"literal payload of type {type_name} is not deep-copyable; "
                "sharing the original reference",
                RuntimeWarning,
                stacklevel=3,
            )
        return payload


def _exact_source_literal_values(graph_obj: Any) -> dict[int, Any]:
    """Snapshot exact-node authored literals before deployment mutates graphs."""

    literals: dict[int, Any] = {}
    for node_id, data in graph_obj.nodes(data=True):
        value_id = int(data.get("value_id", node_id))
        if int(node_id) != value_id:
            continue
        expression = data.get("expr_obj")
        operation = str(data.get("op") or data.get("type") or "").casefold()
        if isinstance(expression, ast.Constant):
            literal = expression.value
        elif operation in {"const", "constant"}:
            literal = (data.get("attributes") or {}).get(
                "value", data.get("constant")
            )
        else:
            continue
        if isinstance(literal, (str, bool, int, float, type(None))):
            literals[value_id] = literal
    return literals


def _recover_late_source_literals(
    function: Any,
    graph_obj: Any | None = None,
    *,
    literal_values: Mapping[int, Any] | None = None,
) -> tuple[tuple[int, Any], ...]:
    """Define exact authored literals exposed only by late call expansion.

    Sequence/record call linking can add a use after the ordinary control
    constant pass has run.  A literal at that seam is still an internal source
    producer, never a caller input.  Materialize only exact AST constants;
    computed expressions remain rejected until their real lowering is linked.
    """

    if not function.blocks:
        return ()
    produced = {
        int(instruction.res.id)
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.res is not None
    }
    parameters = {
        int(value_id)
        for _name, value_id in (function.metadata or {}).get("parameter_names", ())
    }
    authored_constants = dict(literal_values or {})
    authored_constants.update(dict(
        (function.metadata or {}).get("authored_constant_values", ())
    ))
    source_candidates: dict[int, list[tuple[int, Mapping[str, Any]]]] = {}
    if graph_obj is not None:
        for node_id, data in graph_obj.nodes(data=True):
            value_id = int(data.get("value_id", node_id))
            source_candidates.setdefault(value_id, []).append((int(node_id), data))
    # A formal claimed on the book as a member of one of this function's
    # declared records (the ``record_member`` page under a table owned by
    # this function) is a caller input.  Its id equal to a graph Constant's
    # value id is not provenance: the record member row is.
    declared_members: set[int] = set()
    from .identity_concordance import current_identity_book

    member_page = current_identity_book().pages.get("record_member")
    if member_page is not None:
        for row in member_page.rows():
            owner = row[0] if isinstance(row, tuple) and len(row) == 2 else None
            if (
                isinstance(owner, tuple) and owner
                and owner[0] == function.name
                and isinstance(row[1], int)
                and member_page.latest(row)
            ):
                declared_members.add(int(row[1]))
    recovered = []
    instructions = []
    retained_args = []
    from .string_table import string_token
    from ..transmogrifier.ssa import Instr

    for formal in function.args:
        value_id = int(formal.id)
        if value_id in declared_members:
            retained_args.append(formal)
            continue
        candidates = source_candidates.get(value_id, ())
        exact = tuple(data for node_id, data in candidates if node_id == value_id)
        # Structural folding may leave aliases that share a value_id.  The
        # graph node whose own identity equals the SSA value is the authoritative
        # authored occurrence; insertion order of an alias is not provenance.
        data = exact[0] if len(exact) == 1 else (
            candidates[0][1] if len(candidates) == 1 else {}
        )
        expression = data.get("expr_obj")
        operation = str(data.get("op") or data.get("type") or "").casefold()
        has_authored_ledger = value_id in authored_constants
        is_literal = has_authored_ledger or isinstance(expression, ast.Constant) or operation in {
            "const", "constant",
        }
        if value_id in produced or value_id in parameters or not is_literal:
            retained_args.append(formal)
            continue
        literal = authored_constants[value_id] if has_authored_ledger else (
            expression.value
            if isinstance(expression, ast.Constant)
            else (data.get("attributes") or {}).get("value", data.get("constant"))
        )
        if isinstance(literal, str):
            formal.dtype = "int64"
            instruction = Instr(
                "string_token", [], formal,
                attributes={"token": string_token(literal), "text": literal},
            )
        elif literal is None:
            formal.dtype = "none"
            instruction = Instr("NoneValue", [], formal)
        elif isinstance(literal, (bool, int, float)):
            formal.dtype = (
                "bool" if isinstance(literal, bool)
                else "int64" if isinstance(literal, int)
                else "float64"
            )
            instruction = Instr("Const", [], formal, attributes={"value": literal})
        else:
            retained_args.append(formal)
            continue
        formal.accounting = {
            **dict(formal.accounting or {}), "authored_constant": True,
        }
        instructions.append(instruction)
        recovered.append((value_id, literal))
    if instructions:
        next(iter(function.blocks.values())).instrs[0:0] = instructions
        function.args = retained_args
    return tuple(recovered)


def _recover_late_source_slice_offsets(
    function: Any,
    graph_obj: Any,
) -> tuple[tuple[int, int, int | None, int], ...]:
    """Materialize static index slices as their integer base offsets.

    Numerical region addressing represents a shaped source subscript as
    ``GetElementPtr(base, selectors...)`` followed by a shaped ``Load``.  The
    result shape retains each slice extent; the address selector itself is the
    slice's lower bound.  A source ``ast.Slice`` must therefore become an
    integer constant rather than an anonymous runtime formal.

    Only exact, non-negative, unit-stride slices used exclusively in an index
    role are accepted.  Dynamic bounds, negative normalization, non-unit
    strides, aliases, and slices used as ordinary Python values remain
    unresolved.  The complete static bounds stay on the instruction as the
    provenance receipt even though address arithmetic consumes only ``lower``.
    """

    from ..transmogrifier.ssa import Instr

    if not function.blocks:
        return ()
    produced = {
        int(instruction.res.id)
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.res is not None
    }
    parameters = {
        int(value_id)
        for _name, value_id in (function.metadata or {}).get(
            "parameter_names", ()
        )
    }
    recovered: list[tuple[int, int, int | None, int]] = []
    instructions = []
    retained_args = []

    def static_integer(expression: ast.AST | None) -> int | None:
        if expression is None:
            return None
        if not (
            isinstance(expression, ast.Constant)
            and isinstance(expression.value, int)
            and not isinstance(expression.value, bool)
        ):
            raise ValueError
        return int(expression.value)

    for formal in function.args:
        value_id = int(formal.id)
        data = graph_obj.nodes.get(value_id, {})
        expression = data.get("expr_obj")
        operation = str(data.get("op") or data.get("type") or "").casefold()
        children = tuple(data.get("children") or ())
        if (
            value_id in produced
            or value_id in parameters
            or operation != "slice"
            or not isinstance(expression, ast.Slice)
            or not children
            or any(str(role) != "index" for _child, role in children)
        ):
            retained_args.append(formal)
            continue
        try:
            lower = static_integer(expression.lower)
            upper = static_integer(expression.upper)
            step = static_integer(expression.step)
        except ValueError:
            retained_args.append(formal)
            continue
        lower = 0 if lower is None else lower
        step = 1 if step is None else step
        if lower < 0 or step != 1 or upper is not None and upper < lower:
            retained_args.append(formal)
            continue
        formal.dtype = "int64"
        formal.shape = ()
        formal.accounting = {
            **dict(formal.accounting or {}),
            "authored_static_slice": True,
            "source_value_id": value_id,
        }
        instructions.append(Instr(
            "Const", [], formal,
            attributes={
                "value": lower,
                "static_slice_bounds": (lower, upper, step),
                "source_slice_value_id": value_id,
            },
        ))
        recovered.append((value_id, lower, upper, step))
    if instructions:
        next(iter(function.blocks.values())).instrs[0:0] = instructions
        function.args = retained_args
    return tuple(recovered)


def _phi_initial_spellings(instruction: Any) -> frozenset[int]:
    """The ids a Phi names its initial by: the resident ``initial_value_id``
    (read from the ``phi_initial_binding`` row) and the graph's spelling of the
    same version, ``initial_spelled_value_id``.  A source operand is a graph id."""

    attributes = instruction.attributes or {}
    return frozenset(
        int(attributes[key])
        for key in ("initial_value_id", "initial_spelled_value_id")
        if attributes.get(key) is not None
    )


def _recover_late_source_unary_operations(
    function: Function,
    graph: Any,
) -> tuple[tuple[int, int, str], ...]:
    """Define pure source unary operations externalized during frame linking.

    Structural/control planning can retain a tensor operation only as an
    operand of a later control expression.  Call-frame linking then sees that
    operand before a repository-SSA instruction has claimed it and creates a
    provisional function formal.  That formal is not an ABI input: the source
    graph still names both the reduction and its tensor operand.

    Recover only dominance-proof cases here. A reduction may consume an
    entry-produced tensor. A scalar ``item()`` inside a retained loop must
    instead consume the loop-carried Phi whose ``initial_value_id`` names the
    source operand. Reuse the placeholder object held by consumers, insert its
    producer after the exact operand producer, and remove it from the function
    signature. More involved branch-local operations are deliberately refused.
    """

    from ..transmogrifier.ssa import Instr

    if not function.blocks:
        return ()
    entry = next(iter(function.blocks.values()))
    produced = {
        int(instruction.res.id): (block, index, instruction.res)
        for block in function.blocks.values()
        for index, instruction in enumerate(block.instrs)
        if instruction.res is not None
    }
    block_names = {id(block): name for name, block in function.blocks.items()}
    predecessors = {name: set() for name in function.blocks}
    for name, block in function.blocks.items():
        for successor in block.successors:
            if successor in predecessors:
                predecessors[successor].add(name)
    entry_name = block_names[id(entry)]
    dominators = {
        name: ({name} if name == entry_name else set(function.blocks))
        for name in function.blocks
    }
    changed = True
    while changed:
        changed = False
        for name in function.blocks:
            if name == entry_name:
                continue
            incoming = predecessors[name]
            updated = {name} | (
                set.intersection(*(dominators[parent] for parent in incoming))
                if incoming else set()
            )
            if updated != dominators[name]:
                dominators[name] = updated
                changed = True
    recovered: list[tuple[int, int, str]] = []
    for placeholder in tuple(function.args):
        value_id = int(placeholder.id)
        data = graph.nodes.get(value_id, {})
        operation = str(
            data.get("op") or data.get("type") or ""
        ).casefold()
        if operation not in {"all", "any", "item"} or value_id in produced:
            continue
        operands = tuple(
            int(parent)
            for parent, role in data.get("parents") or ()
            if str(role) in {"operand", "arg:0", "value"}
        )
        if len(operands) != 1:
            continue
        operand_id = int(operands[0])
        source = produced.get(operand_id)
        if operation == "item":
            carried = tuple(
                (block, index, instruction.res)
                for block in function.blocks.values()
                for index, instruction in enumerate(block.instrs)
                if (
                    instruction.op == "Phi"
                    and instruction.res is not None
                    and operand_id in _phi_initial_spellings(instruction)
                    and (instruction.attributes or {}).get("binding")
                    == "loop_carried"
                )
            )
            if len(carried) == 1:
                source = carried[0]
            elif carried:
                continue
        if source is None:
            continue
        producer_block, producer_index, operand = source
        producer_name = block_names[id(producer_block)]
        consumer_names = {
            name
            for name, block in function.blocks.items()
            for instruction in block.instrs
            if any(int(argument.id) == value_id for argument in instruction.args)
        }
        if (
            not consumer_names
            or any(producer_name not in dominators[name]
                   for name in consumer_names)
            or operation in {"all", "any"} and producer_block is not entry
            or operation == "item" and tuple(operand.shape or ())
        ):
            continue
        function.args.remove(placeholder)
        placeholder.dtype = (
            "bool" if operation in {"all", "any"} else operand.dtype
        )
        placeholder.shape = ()
        placeholder.accounting = {
            **dict(placeholder.accounting or {}),
            "recovered_structural_source": operation,
            "source_value_id": value_id,
            "source_operand_id": operand_id,
        }
        opcode = operation if operation in {"all", "any"} else "Cast"
        instruction = Instr(
            opcode,
            [operand],
            placeholder,
            attributes={
                "tensor_operation": operation,
                "structural_operation": (
                    "tensor_reduction"
                    if operation in {"all", "any"} else "scalar_item"
                ),
                **({"target_dtype": operand.dtype}
                   if operation == "item" else {}),
                "late_call_feed_recovery": True,
            },
        )
        insertion_index = producer_index + 1
        while (
            insertion_index < len(producer_block.instrs)
            and producer_block.instrs[insertion_index].op == "Phi"
        ):
            insertion_index += 1
        producer_block.instrs.insert(insertion_index, instruction)
        # Subsequent recoveries must use positions in the expanded block.
        for resident_id, (owner, offset, resident) in tuple(produced.items()):
            if owner is producer_block and offset >= insertion_index:
                produced[resident_id] = (owner, offset + 1, resident)
        produced[value_id] = (
            producer_block, insertion_index, placeholder,
        )
        recovered.append((value_id, operand_id, operation))
    return tuple(recovered)


def _recover_late_source_pure_expressions(
    function: Function,
    graph: Any,
) -> tuple[tuple[int, int, str], ...]:
    """Materialize exact scalar source expressions lost at frame linking.

    Linking may expose a complete, pure source expression as a provisional
    formal even though none of its interior nodes reached repository SSA. A
    backend must not accept that internal value as a new ABI input. Rebuild a
    deliberately small, backend-neutral scalar expression closure when all of
    its runtime leaves have one proven placement which dominates every use.

    Constants come from their exact source nodes. ``item()`` over a loop input
    binds to the unique loop-carried Phi naming that input, so reconstruction
    observes the current iteration rather than the entry value. Expressions
    with non-dominating branch-dependent leaves, multiple possible carried
    versions, shaped intermediates, or unsupported/effectful operations are
    left for the provenance gate to reject.
    """

    from ..transmogrifier.ssa import Instr, SSAValue

    if not function.blocks:
        return ()
    blocks = function.blocks
    entry_name = next(iter(blocks))
    predecessors = {name: set() for name in blocks}
    for name, block in blocks.items():
        for successor in block.successors:
            if successor in predecessors:
                predecessors[successor].add(name)
    dominators = {
        name: ({name} if name == entry_name else set(blocks))
        for name in blocks
    }
    changed = True
    while changed:
        changed = False
        for name in blocks:
            if name == entry_name:
                continue
            incoming = predecessors[name]
            updated = {name} | (
                set.intersection(*(dominators[parent] for parent in incoming))
                if incoming else set()
            )
            if updated != dominators[name]:
                dominators[name] = updated
                changed = True

    values: dict[int, SSAValue] = {}
    conflicting_value_ids: set[int] = set()

    def remember(value: SSAValue | None) -> None:
        if value is None:
            return
        value_id = int(value.id)
        resident = values.setdefault(value_id, value)
        if resident is not value:
            conflicting_value_ids.add(value_id)

    for argument in function.args:
        remember(argument)
    produced: dict[int, tuple[str, int, SSAValue]] = {}
    for block_name, block in blocks.items():
        for index, instruction in enumerate(block.instrs):
            for argument in instruction.args:
                remember(argument)
            remember(instruction.res)
            if instruction.res is not None:
                produced[int(instruction.res.id)] = (
                    block_name, index, instruction.res,
                )

    binary_operations = {
        "add": "Add",
        "sub": "Sub",
        "mul": "Mul",
        "mult": "Mul",
        "div": "Div",
        "truediv": "Div",
    }

    def source_operation(value_id: int) -> tuple[dict[str, Any], str]:
        data = graph.nodes.get(int(value_id), {})
        operation = str(data.get("op") or data.get("type") or "").casefold()
        return data, operation

    def exact_parents(
        data: Mapping[str, Any], roles: tuple[str, ...],
    ) -> tuple[int, ...]:
        by_role: dict[str, list[int]] = {role: [] for role in roles}
        for parent, role in data.get("parents") or ():
            role = str(role)
            if role in by_role:
                by_role[role].append(int(parent))
        if any(len(by_role[role]) != 1 for role in roles):
            return ()
        return tuple(by_role[role][0] for role in roles)

    def literal_value(data: Mapping[str, Any]) -> Any:
        expression = data.get("expr_obj")
        if isinstance(expression, ast.Constant):
            return expression.value
        attributes = data.get("attributes") or {}
        return attributes.get("value", data.get("constant"))

    recovered: list[tuple[int, int, str]] = []
    for placeholder in tuple(function.args):
        root_id = int(placeholder.id)
        _root_data, root_operation = source_operation(root_id)
        if root_operation not in binary_operations or root_id in produced:
            continue

        planned: list[tuple[str, list[SSAValue], SSAValue, dict[str, Any]]] = []
        runtime_leaves: list[tuple[str, int, SSAValue]] = []
        source_roots: set[int] = set()
        visiting: set[int] = set()
        built: dict[int, SSAValue] = {}
        failed = [False]

        def result_value(value_id: int, dtype: str) -> SSAValue:
            existing = values.get(int(value_id))
            if existing is not None:
                if int(value_id) in conflicting_value_ids:
                    failed[0] = True
                return existing
            created = SSAValue(int(value_id), dtype, shape=())
            values[int(value_id)] = created
            return created

        def build(value_id: int) -> SSAValue | None:
            value_id = int(value_id)
            if value_id in built:
                return built[value_id]
            if value_id in visiting:
                failed[0] = True
                return None
            data, operation = source_operation(value_id)
            attributes = data.get("attributes") or {}
            # A sequence arena may have a scalar storage handle. That does
            # not make authored list replication numeric multiplication.
            if attributes.get("aggregate_kind") in {"list", "tuple", "dict", "set"}:
                failed[0] = True
                return None
            existing_producer = produced.get(value_id)
            if existing_producer is not None:
                block_name, index, value = existing_producer
                if tuple(value.shape or ()):
                    failed[0] = True
                    return None
                runtime_leaves.append((block_name, index, value))
                source_roots.add(value_id)
                built[value_id] = value
                return value
            visiting.add(value_id)
            try:
                if operation in {"const", "constant"}:
                    literal = literal_value(data)
                    if not isinstance(literal, (bool, int, float)):
                        failed[0] = True
                        return None
                    dtype = (
                        "bool" if isinstance(literal, bool)
                        else "int64" if isinstance(literal, int)
                        else "float64"
                    )
                    result = result_value(value_id, dtype)
                    result.dtype = dtype
                    result.shape = ()
                    planned.append((
                        "Const", [], result,
                        {"value": literal, "late_source_expression": True},
                    ))
                    built[value_id] = result
                    return result
                if operation == "item":
                    parents = exact_parents(data, ("operand",))
                    if not parents:
                        failed[0] = True
                        return None
                    initial_id = int(parents[0])
                    carried = tuple(
                        (block_name, index, instruction.res)
                        for block_name, block in blocks.items()
                        for index, instruction in enumerate(block.instrs)
                        if (
                            instruction.op == "Phi"
                            and instruction.res is not None
                            and initial_id in _phi_initial_spellings(instruction)
                            and (instruction.attributes or {}).get("binding")
                            == "loop_carried"
                        )
                    )
                    if len(carried) == 1:
                        block_name, index, operand = carried[0]
                    elif carried:
                        failed[0] = True
                        return None
                    else:
                        source = produced.get(initial_id)
                        if source is None:
                            failed[0] = True
                            return None
                        block_name, index, operand = source
                    if tuple(operand.shape or ()):
                        failed[0] = True
                        return None
                    runtime_leaves.append((block_name, index, operand))
                    source_roots.add(initial_id)
                    result = result_value(
                        value_id, str(operand.dtype or "float64"),
                    )
                    result.dtype = operand.dtype
                    result.shape = ()
                    planned.append((
                        "Cast", [operand], result,
                        {
                            "tensor_operation": "item",
                            "structural_operation": "scalar_item",
                            "target_dtype": operand.dtype,
                            "late_source_expression": True,
                            "source_operand_id": initial_id,
                        },
                    ))
                    built[value_id] = result
                    return result
                opcode = binary_operations.get(operation)
                if opcode is None:
                    failed[0] = True
                    return None
                parents = exact_parents(data, ("lhs", "rhs"))
                if not parents:
                    failed[0] = True
                    return None
                left = build(parents[0])
                right = build(parents[1])
                if left is None or right is None:
                    failed[0] = True
                    return None
                dtypes = {str(left.dtype or ""), str(right.dtype or "")}
                dtype = (
                    "float64"
                    if opcode == "Div" or any("float" in item for item in dtypes)
                    else "int64"
                )
                result = result_value(value_id, dtype)
                result.dtype = dtype
                result.shape = ()
                planned.append((
                    opcode, [left, right], result,
                    {
                        "structural_operation": "pure_scalar_expression",
                        "source_operation": operation,
                        "late_source_expression": True,
                    },
                ))
                built[value_id] = result
                return result
            finally:
                visiting.discard(value_id)

        result = build(root_id)
        if failed[0] or result is not placeholder or not planned or not runtime_leaves:
            continue
        consumer_positions = tuple(
            (block_name, index)
            for block_name, block in blocks.items()
            for index, instruction in enumerate(block.instrs)
            if any(int(argument.id) == root_id for argument in instruction.args)
        )
        if not consumer_positions:
            continue
        candidate_names = tuple(
            name for name in blocks
            if (
                all(producer_name in dominators[name]
                    for producer_name, _index, _value in runtime_leaves)
                and all(name in dominators[consumer_name]
                        for consumer_name, _index in consumer_positions)
            )
        )
        if not candidate_names:
            continue
        placement = max(candidate_names, key=lambda name: len(dominators[name]))
        insertion_index = 0
        for producer_name, producer_index, _value in runtime_leaves:
            if producer_name == placement:
                insertion_index = max(insertion_index, producer_index + 1)
        while (
            insertion_index < len(blocks[placement].instrs)
            and blocks[placement].instrs[insertion_index].op == "Phi"
        ):
            insertion_index += 1
        local_consumers = tuple(
            index for block_name, index in consumer_positions
            if block_name == placement
        )
        if local_consumers and insertion_index > min(local_consumers):
            continue
        instructions = [
            Instr(opcode, arguments, result_, attributes=attributes)
            for opcode, arguments, result_, attributes in planned
        ]
        blocks[placement].instrs[insertion_index:insertion_index] = instructions
        for resident_id, (owner, offset, resident) in tuple(produced.items()):
            if owner == placement and offset >= insertion_index:
                produced[resident_id] = (owner, offset + len(instructions), resident)
        function.args.remove(placeholder)
        placeholder.accounting = {
            **dict(placeholder.accounting or {}),
            "recovered_structural_source": root_operation,
            "source_value_id": root_id,
            "source_root_ids": tuple(sorted(source_roots)),
        }
        for offset, instruction in enumerate(instructions):
            produced[int(instruction.res.id)] = (
                placement, insertion_index + offset, instruction.res,
            )
        primary_root = min(source_roots) if source_roots else root_id
        recovered.append((root_id, primary_root, root_operation))
    return tuple(recovered)


def _index_source_graph_spellings(
    indexed: dict[str, Any],
    function_graph: Any,
    *symbols: Any,
) -> None:
    """Index one graph by repository, local, and qualified source spellings."""

    for symbol in (
        *symbols,
        function_graph.graph.get("function_name"),
        function_graph.graph.get("qualified_name"),
    ):
        if symbol:
            indexed.setdefault(str(symbol), function_graph)


def _resolved_source_graphs_by_symbol(resolved_graph: Any) -> dict[str, Any]:
    """Index final resolved function graphs by every authored symbol spelling.

    The whole-program wrapper owns the resolved project graph, while the
    deployment helper has its own shell-to-symbol table. Final provenance
    reconciliation must use the former rather than reaching into a helper's
    local table. Function-table entries preserve both the local and qualified
    authored names, so either repository-SSA spelling resolves to the same
    exact post-reduction graph.
    """

    indexed: dict[str, Any] = {}
    for entry in getattr(resolved_graph, "function_table", ()):
        function_graph = getattr(getattr(entry, "graph", None), "G", None)
        if function_graph is None:
            continue
        _index_source_graph_spellings(
            indexed,
            function_graph,
            getattr(entry, "name", None),
            getattr(entry, "qualified_name", None),
        )
    return indexed


def _source_graph_for_lowered_function(
    source_graphs: Mapping[str, Any],
    function_symbol: str,
    function: Any,
) -> Any | None:
    """Resolve a specialized SSA function back to its authored graph."""

    source_symbol = str(
        (getattr(function, "metadata", None) or {}).get(
            "source_qualified_name"
        )
        or function_symbol
    )
    return (
        source_graphs.get(str(function_symbol))
        or source_graphs.get(source_symbol)
        or source_graphs.get(source_symbol.rsplit(".", 1)[-1])
    )


def _resolved_source_literals_by_symbol(
    resolved_graph: Any,
) -> dict[str, dict[int, Any]]:
    """Freeze literal provenance while the resolved function graphs exist."""

    return {
        symbol: _exact_source_literal_values(function_graph)
        for symbol, function_graph in _resolved_source_graphs_by_symbol(
            resolved_graph
        ).items()
    }


def _record_module_closure_formals(
    module: Any,
    source_graphs: Mapping[str, Any] | None = None,
) -> tuple[tuple[str, str, int], ...]:
    """Name the formals a function reads but never declares or produces.

    A nested function consumes bindings from its enclosing scope:
    ``_wrench_force`` reads ``wrench_k`` and ``plane_normal_q``. Those are
    real formals the caller must supply, but they are not authored
    parameters, so the signature looked as though it had grown values no
    caller could name. The source graph marks each one as an ``Input`` with
    an external/closure binding, which is the exact discriminator: an
    escaped local temporary is not such an input and keeps announcing
    itself.
    """

    source_graphs = dict(source_graphs or {})
    if not source_graphs:
        return ()
    recorded: list[tuple[str, str, int]] = []
    for function_symbol, function in module.functions.items():
        graph = _source_graph_for_lowered_function(
            source_graphs, str(function_symbol), function,
        )
        if graph is None:
            continue
        metadata = function.metadata or {}
        named = {
            int(value_id)
            for _name, value_id in metadata.get("parameter_names", ())
        }
        authored = set(map(str, metadata.get("authored_parameters", ()) or ()))
        captures: dict[int, str] = {}
        for node_id, data in graph.nodes(data=True):
            attributes = data.get("attributes") or {}
            if str(data.get("type") or "") != "Input":
                continue
            if str(attributes.get("binding_kind") or "") not in {
                "closure", "external",
            }:
                continue
            name = str(attributes.get("binding_name") or "")
            if not name or name in authored:
                continue
            value_id = int(data.get("value_id", node_id))
            if value_id in named:
                continue
            captures.setdefault(value_id, name)
        entries = tuple(
            {
                "name": captures[int(argument.id)],
                "value_id": int(argument.id),
                "priority": "exact_source_capture_identity",
                "tie_policy": "incumbent",
            }
            for argument in function.args
            if int(argument.id) in captures
        )
        if not entries:
            continue
        function.metadata["closure_formals"] = entries
        recorded.extend(
            (str(function_symbol), entry["name"], int(entry["value_id"]))
            for entry in entries
        )
    return tuple(recorded)


def _recover_module_late_source_literals(
    module: Any,
    resolved_graph: Any | None = None,
    *,
    source_literals_by_symbol: Mapping[str, Mapping[int, Any]] | None = None,
) -> tuple[tuple[str, int, Any], ...]:
    """Reconcile final specialized signatures with exact authored constants."""

    source_graphs = (
        {} if resolved_graph is None
        else _resolved_source_graphs_by_symbol(resolved_graph)
    )
    literal_catalogue = dict(source_literals_by_symbol or {})
    recovered: list[tuple[str, int, Any]] = []
    for function_symbol, function in module.functions.items():
        source_symbol = str(
            (function.metadata or {}).get("source_qualified_name")
            or function_symbol
        )
        function_graph = _source_graph_for_lowered_function(
            source_graphs, str(function_symbol), function,
        )
        literal_values = (
            literal_catalogue.get(source_symbol)
            or literal_catalogue.get(source_symbol.rsplit(".", 1)[-1])
        )
        if function_graph is None and literal_values is None:
            continue
        rows = _recover_late_source_literals(
            function, function_graph, literal_values=literal_values,
        )
        if not rows:
            continue
        function.metadata["recovered_late_source_literals"] = tuple(
            dict.fromkeys((
                *function.metadata.get("recovered_late_source_literals", ()),
                *(value_id for value_id, _literal in rows),
            ))
        )
        recovered.extend(
            (str(function_symbol), int(value_id), literal)
            for value_id, literal in rows
        )
    module.metadata["final_recovered_source_literals"] = tuple(
        (function_symbol, value_id)
        for function_symbol, value_id, _literal in recovered
    )
    return tuple(recovered)
