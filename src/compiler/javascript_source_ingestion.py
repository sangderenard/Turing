"""Lower one JavaScript function to a repository SSA module.

This is the JavaScript counterpart of :func:`lower_glsl_source_to_ssa`: the
result is an ordinary :class:`IRModule` with one function whose arguments are
the JavaScript parameters and whose ``Ret`` carries the returned value, so the
native backends (``emit_ssa_to_c`` and friends) compile it like any other
repository module.

It does not own JavaScript semantics.  The one semantic reader is
``javascript_process_graph``'s interpreter: it decides every operation,
truthiness program, realm identity and refusal.  This module reads that
interpreter's region node for node -- each node already names its existing
``Handler`` and its operands by role -- and writes the same operations as
``Instr`` records.  It is not the ProcessGraph scheduler route
(``process_graph_to_ssa_instrs``); nothing is scheduled or re-derived.

The function ABI is declared, never defaulted.  JavaScript has no static
types, so every parameter needs a JSDoc ``@param {number}`` or
``@param {boolean}`` tag (or an explicit ``parameter_kinds`` entry).  A
parameter without one is a named shortfall.  A JavaScript Number is an IEEE
double, so number values are ``float64`` everywhere, literals included.

The module carries its identity book (``module.metadata["identity_book"]``),
the one the interpreter posted its ingestion cells and name bindings to.
Every SSA value is its graph node (the value id IS the node id) and has an
``ssa_value`` row DERIVED from that node's ingestion cell; every block has
an ``ssa_block`` row.  So the backends post their emission rows onto the
same book instead of recording ``no_book_at_emission`` gaps.  Inside an open
compile the lowering posts to that compile's book; otherwise it opens its
own.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from ..transmogrifier.ssa import BasicBlock, Function, IRModule, Instr, SSAValue
from ..transmogrifier.ssa_registry import Handler
from .javascript_process_graph import (
    PORTABLE_NUMERIC_ASSUMPTIONS,
    _function_candidates,
    ingest_javascript_program,
)


#: JavaScript value kind -> repository dtype.  Only proved kinds appear.
JAVASCRIPT_KIND_DTYPES: Mapping[str, str] = {
    "number": "float64",
    "boolean": "bool",
}

_ROLE_ORDER = {
    "left": 0, "right": 1, "operand": 0, "value": 0,
    "condition": 0, "if_true": 1, "if_false": 2,
}

_JSDOC = re.compile(r"/\*\*(?P<body>(?:(?!\*/).)*)\*/\s*$", re.DOTALL)
_JSDOC_PARAM = re.compile(
    r"@param\s*\{\s*(?P<type>[A-Za-z]+)\s*\}\s*(?P<name>[A-Za-z_$][\w$]*)"
)


@dataclass(frozen=True, order=True)
class JavaScriptSourceShortfall:
    code: str
    message: str

    def format(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass(frozen=True)
class JavaScriptSSALoweringResult:
    module: IRModule
    function_name: str
    parameter_names: tuple[str, ...]
    parameter_kinds: Mapping[str, str]
    result_kind: str
    shortfalls: tuple[JavaScriptSourceShortfall, ...]

    @property
    def complete(self) -> bool:
        return not self.shortfalls

    def shortfall_report(self) -> str:
        if not self.shortfalls:
            return "JavaScript source to SSA: complete"
        return "JavaScript source to SSA shortfalls:\n" + "\n".join(
            f"- {item.format()}" for item in self.shortfalls
        )


def javascript_jsdoc_parameter_kinds(
    source: str, declaration: Mapping[str, Any],
) -> dict[str, str]:
    """Read ``@param {type} name`` from the JSDoc block right before a function.

    Only the block that ends immediately before the declaration counts; a tag
    elsewhere in the file says nothing about this function.  The type
    spelling is returned lowercased (``Number`` and ``number`` agree).
    """

    start = declaration.get("start")
    if not isinstance(start, int):
        return {}
    match = _JSDOC.search(source[:start])
    if match is None:
        return {}
    return {
        item.group("name"): item.group("type").lower()
        for item in _JSDOC_PARAM.finditer(match.group("body"))
    }


#: Interpreter attributes that stay on the instruction.
_CARRIED_ATTRIBUTES = ("javascript_identity", "ecmascript_approximated")


def _carried(attributes: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: attributes[key] for key in _CARRIED_ATTRIBUTES if key in attributes
    }


def _ordered_parents(data: Mapping[str, Any]) -> list[int]:
    parents = list(data.get("parents") or ())

    def rank(item: tuple[int, str]) -> tuple[int, int]:
        role = str(item[1])
        if role.startswith("arg") and role[3:].isdigit():
            return (0, int(role[3:]))
        return (0, _ROLE_ORDER.get(role, 99))

    return [int(parent) for parent, _role in sorted(parents, key=rank)]


def lower_javascript_source_to_ssa(
    source: str,
    *,
    function_name: str | None = None,
    parameter_kinds: Mapping[str, str] | None = None,
    owner: str = "<javascript>",
    program: Mapping[str, Any] | None = None,
) -> JavaScriptSSALoweringResult:
    """Lower one JavaScript function (by name, or the unique one) to SSA."""

    from .identity_concordance import (
        _ACTIVE_IDENTITY_BOOK, begin_identity_book, end_identity_book,
    )

    book = _ACTIVE_IDENTITY_BOOK.get()
    token = None
    if book is None or getattr(book, "detached", False):
        book, token = begin_identity_book()
    try:
        lowered = _lower_javascript_function(
            source, function_name=function_name,
            parameter_kinds=parameter_kinds, owner=owner, program=program,
        )
    finally:
        if token is not None:
            end_identity_book(token)
    lowered.module.metadata["identity_book"] = book
    return lowered


def _block_kind(label: str) -> Any:
    from .concordance_declarations import SSABlockKind

    if label == "entry":
        return SSABlockKind.ENTRY
    return SSABlockKind(
        "loop_" + label.rsplit("_", 1)[1]
    )


def _post_ssa_identities(
    graph: Any,
    function: Function,
    nodes_of: Mapping[int, int],
    block_nodes: Mapping[str, int],
) -> None:
    """``ssa_value`` and ``ssa_block`` rows for the lowered ``function``.

    A value's row is DERIVED from the ingestion cell of the graph node it
    is (``nodes_of``: value id -> node id; the ids are equal except for a
    parameter the interpreter made no node for); a block's from the node
    that owns it: the region's root for ``entry``, a loop's condition for
    its header, body and exit.  The signature is on the book too: one
    ``function_parameter`` row per parameter and one ``function_output``
    row for the returned value, each DERIVED from that value's row.
    """

    from ..common.tensors.topological_reducer import node_identity_cell
    from .concordance_declarations import (
        FUNCTION_OUTPUT, FUNCTION_PARAMETER, JAVASCRIPT_SSA_LOWERING,
        SSA_BLOCK, SSA_VALUE, SSAValueFact, SSAValueOrigin,
    )
    from .identity_concordance import Derived, Mode, current_identity_book
    from .ssa_record_return_state import function_scope_of

    book = current_identity_book()
    scope = function_scope_of(function)
    values = list(function.args) + [
        instruction.res
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.res is not None
    ]
    cells = {}
    for value in values:
        node_id = nodes_of[int(value.id)]
        cells[int(value.id)] = book.post(
            SSA_VALUE, (scope, int(value.id)),
            SSAValueFact(
                value.dtype, tuple(value.shape or ()),
                SSAValueOrigin.ADOPTED_GRAPH_ID
                if node_id == int(value.id) else SSAValueOrigin.MINTED,
            ),
            stage=JAVASCRIPT_SSA_LOWERING,
            provenance=Derived((node_identity_cell(graph, node_id),)),
            mode=Mode.CONCORD,
        )
    for name, value_id in function.metadata.get("parameter_names", ()):
        book.post(
            FUNCTION_PARAMETER, (scope, str(name)), int(value_id),
            stage=JAVASCRIPT_SSA_LOWERING,
            provenance=Derived((cells[int(value_id)],)), mode=Mode.CONCORD,
        )
    returned = [
        instruction.args
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == Handler.Ret.value
    ]
    for slot, value in enumerate(returned[0] if returned else ()):
        book.post(
            FUNCTION_OUTPUT, (scope, slot), (None, int(value.id)),
            stage=JAVASCRIPT_SSA_LOWERING,
            provenance=Derived((cells[int(value.id)],)), mode=Mode.CONCORD,
        )
    for label in function.blocks:
        book.post(
            SSA_BLOCK, (scope, str(function.name), str(label)),
            _block_kind(str(label)), stage=JAVASCRIPT_SSA_LOWERING,
            provenance=Derived((
                node_identity_cell(graph, block_nodes[str(label)]),
            )),
            mode=Mode.CONCORD,
        )


def _lower_javascript_function(
    source: str,
    *,
    function_name: str | None,
    parameter_kinds: Mapping[str, str] | None,
    owner: str,
    program: Mapping[str, Any] | None,
) -> JavaScriptSSALoweringResult:
    from ..transmogrifier.graph.graph_express2 import ProcessGraph
    from .dream_document import _run_javascript_ast_parser

    if program is None:
        program = _run_javascript_ast_parser(source)
    candidates = [
        item for item in _function_candidates(program)
        if function_name is None or item[0] == function_name
    ]
    if len(candidates) != 1:
        wanted = function_name or "a unique JavaScript function"
        raise ValueError(
            f"expected {wanted!r}; found {[name for name, _, _ in candidates]!r}"
        )
    name, function, declaration = candidates[0]
    shortfalls: list[JavaScriptSourceShortfall] = []

    declared = javascript_jsdoc_parameter_kinds(source, declaration)
    declared.update(dict(parameter_kinds or {}))
    parameter_names = tuple(
        str(parameter.get("name", ""))
        for parameter in function.get("params", ())
        if parameter.get("type") == "Identifier"
    )
    kinds: dict[str, str] = {}
    for parameter_name in parameter_names:
        kind = declared.get(parameter_name)
        if kind not in JAVASCRIPT_KIND_DTYPES:
            shortfalls.append(JavaScriptSourceShortfall(
                "undeclared-parameter-kind",
                f"parameter {parameter_name!r} of {name!r} needs a JSDoc "
                f"@param {{number}} or {{boolean}}; found {kind!r}",
            ))
            continue
        kinds[parameter_name] = kind

    graph = ProcessGraph(materialize_memory=False)
    lowering = ingest_javascript_program(
        graph, program, owner=owner, entry=name,
        assumptions=PORTABLE_NUMERIC_ASSUMPTIONS,
        parameter_kinds=kinds,
    )
    shortfalls.extend(
        JavaScriptSourceShortfall(item.code, item.message)
        for item in lowering.shortfalls
    )

    nodes = graph.G.nodes
    values: dict[int, SSAValue] = {}
    arguments: list[SSAValue] = []
    # A value's id is its node's id, so the book's node cells name it.
    nodes_of: dict[int, int] = {}
    unbound_id = max(nodes, default=-1) + 1
    root = int(lowering.root)

    def fresh(dtype: str | None, node_id: int | None) -> SSAValue:
        nonlocal unbound_id
        if node_id is None:
            # A parameter the interpreter made no node for stands on the
            # region's root.
            value_id, node_id = unbound_id, root
            unbound_id += 1
        else:
            value_id = node_id
        nodes_of[value_id] = node_id
        return SSAValue(value_id, dtype=dtype)

    def kind_of(node_id: int) -> str:
        return str(
            (nodes[node_id].get("attributes") or {}).get(
                "javascript_value_kind", "unknown"
            )
        )

    def dtype_of(node_id: int) -> str | None:
        kind = kind_of(node_id)
        dtype = JAVASCRIPT_KIND_DTYPES.get(kind)
        if dtype is None and nodes[node_id].get("type") not in {
            "Untranslated", Handler.Ret.name,
        }:
            shortfalls.append(JavaScriptSourceShortfall(
                "unproved-value-kind",
                f"node {node_id} ({nodes[node_id].get('type')}) has no "
                f"proved JavaScript value kind (found {kind!r})",
            ))
        return dtype

    # Parameters first, in authored order, so the ABI is the signature.
    parameter_nodes = {
        str((data.get("attributes") or {}).get("binding_name")): node_id
        for node_id, data in nodes(data=True)
        if data.get("type") == "Input"
        and (data.get("attributes") or {}).get("binding_kind") == "parameter"
    }
    for parameter_name in parameter_names:
        node_id = parameter_nodes.get(parameter_name)
        value = fresh(
            JAVASCRIPT_KIND_DTYPES.get(kinds.get(parameter_name, "")), node_id,
        )
        arguments.append(value)
        if node_id is not None:
            values[node_id] = value

    # Blocks.  The interpreter's region is acyclic except for its loops,
    # which it records in ``javascript_loops`` with the id ranges their
    # header (Phi nodes, then the condition) and body occupy; node ids are
    # creation order, so an id range is a program region.  Each loop's
    # back edge (a Phi's ``next`` value) lives in that table, not on the
    # graph.
    loops = sorted(
        (dict(item) for item in graph.G.graph.get("javascript_loops", ())),
        key=lambda item: int(item["header_start"]),
    )
    phi_loop = {
        int(phi): loop for loop in loops for phi in loop["phis"].values()
    }

    returned_nodes = _ordered_parents(nodes[root])
    # Every loop is emitted, used or not: a loop that never terminates in
    # JavaScript must not terminate here either.
    needed: set[int] = set()
    pending = list(returned_nodes)
    for loop in loops:
        pending.append(int(loop["condition"]))
        pending.extend(int(item) for item in loop["phis"].values())
        pending.extend(int(item) for item in loop["next"].values())
    while pending:
        node_id = pending.pop()
        if node_id in needed:
            continue
        needed.add(node_id)
        pending.extend(_ordered_parents(nodes[node_id]))
        if node_id in phi_loop:
            loop = phi_loop[node_id]
            carried_name = next(
                key for key, value in loop["phis"].items()
                if int(value) == node_id
            )
            pending.append(int(loop["next"][carried_name]))
            pending.append(int(loop["condition"]))

    blocks: dict[str, BasicBlock] = {}
    block_nodes: dict[str, int] = {"entry": root}
    current = BasicBlock("entry", [])
    blocks["entry"] = current
    phi_instructions: dict[int, Instr] = {}
    failed = False

    def emit(node_id: int) -> None:
        nonlocal failed
        if node_id in values or node_id not in needed:
            return
        data = nodes[node_id]
        node_type = str(data.get("type"))
        attributes = dict(data.get("attributes") or {})
        if node_type == "Phi":
            init = values.get(_ordered_parents(data)[0])
            if init is None:
                failed = True
                return
            value = fresh(dtype_of(node_id), node_id)
            instruction = Instr(
                Handler.Phi.value, [init], value,
                attributes={"binding_name": attributes.get("binding_name")},
            )
            current.instrs.append(instruction)
            phi_instructions[node_id] = instruction
            values[node_id] = value
            return
        operands = [values.get(parent) for parent in _ordered_parents(data)]
        if node_type == "Untranslated" or any(item is None for item in operands):
            failed = True
            return
        if node_type == "Input":
            shortfalls.append(JavaScriptSourceShortfall(
                "free-javascript-name",
                f"{attributes.get('binding_name')!r} is not a parameter or "
                "const binding of this function",
            ))
            failed = True
            return
        dtype = dtype_of(node_id)
        if node_type == "Constant":
            constant = data.get("constant")
            if kind_of(node_id) == "number":
                # Acorn hands integral literals over as Python ints; the
                # JavaScript value is the double.
                constant = float(constant)
            value = fresh(dtype, node_id)
            current.instrs.append(Instr(
                Handler.Const.value, [], value,
                attributes={"value": constant},
            ))
        elif node_type == "Call":
            # A realm identity names an existing repository operation in
            # ``tensor_operation``; it is written as that operation, the way
            # the Python frontend writes math.sin as ``sin``.
            operation = str(attributes.get("tensor_operation") or "")
            if not operation:
                shortfalls.append(JavaScriptSourceShortfall(
                    "unlowered-call",
                    f"call node {node_id} names no repository operation",
                ))
                failed = True
                return
            value = fresh(dtype, node_id)
            current.instrs.append(Instr(
                operation, list(operands), value,
                attributes=_carried(attributes),
            ))
        else:
            try:
                handler = Handler[node_type]
            except KeyError:
                shortfalls.append(JavaScriptSourceShortfall(
                    "unlowered-node",
                    f"node type {node_type!r} has no SSA instruction",
                ))
                failed = True
                return
            value = fresh(dtype, node_id)
            current.instrs.append(Instr(
                handler.value, list(operands), value,
                attributes=_carried(attributes),
            ))
        values[node_id] = value

    def branch(target: BasicBlock) -> None:
        current.instrs.append(Instr(
            Handler.Br.value, [], None, attributes={"target": target.name},
        ))
        current.successors.append(target.name)

    def region(low: int, high: int) -> None:
        nonlocal current
        position = low
        inner = [
            loop for loop in loops
            if low <= int(loop["header_start"]) and int(loop["body_end"]) <= high
        ]
        top = []
        for loop in inner:
            if not any(
                other is not loop
                and int(other["header_start"]) <= int(loop["header_start"])
                and int(loop["body_end"]) <= int(other["body_end"])
                for other in inner
            ):
                top.append(loop)
        for loop in top:
            for node_id in range(position, int(loop["header_start"])):
                emit(node_id)
            label = f"loop{int(loop['id'])}"
            header = BasicBlock(f"{label}_header", [])
            body = BasicBlock(f"{label}_body", [])
            exit_block = BasicBlock(f"{label}_exit", [])
            for block in (header, body, exit_block):
                block_nodes[block.name] = int(loop["condition"])
            preheader = current
            branch(header)
            blocks[header.name] = header
            current = header
            for node_id in range(
                int(loop["header_start"]), int(loop["condition_end"])
            ):
                emit(node_id)
            condition = values.get(int(loop["condition"]))
            if condition is None:
                shortfalls.append(JavaScriptSourceShortfall(
                    "unlowered-loop",
                    f"loop {loop['id']} condition has no value",
                ))
                return
            current.instrs.append(Instr(
                Handler.CondBr.value, [condition], None,
                attributes={
                    "true_target": body.name, "false_target": exit_block.name,
                },
            ))
            current.successors.extend((body.name, exit_block.name))
            blocks[body.name] = body
            current = body
            region(int(loop["condition_end"]), int(loop["body_end"]))
            latch = current
            for carried_name, phi in loop["phis"].items():
                instruction = phi_instructions.get(int(phi))
                carried = values.get(int(loop["next"][carried_name]))
                if instruction is None or carried is None:
                    shortfalls.append(JavaScriptSourceShortfall(
                        "unlowered-loop",
                        f"loop {loop['id']} carries {carried_name!r} without a value",
                    ))
                    continue
                instruction.args.append(carried)
                instruction.attributes["incoming_blocks"] = (
                    preheader.name, latch.name,
                )
            branch(header)
            blocks[exit_block.name] = exit_block
            current = exit_block
            position = int(loop["body_end"])
        for node_id in range(position, high):
            emit(node_id)

    region(0, root)
    result_kind = "unknown"
    returned: list[SSAValue] = []
    if returned_nodes:
        result_kind = kind_of(returned_nodes[0])
        result = values.get(returned_nodes[0])
        if result is not None:
            returned.append(result)
    current.instrs.append(Instr(Handler.Ret.value, returned, None))
    function_ir = Function(name, arguments, blocks)
    function_ir.metadata["parameter_names"] = tuple(
        (parameter_name, value.id)
        for parameter_name, value in zip(parameter_names, arguments)
    )
    _post_ssa_identities(graph, function_ir, nodes_of, block_nodes)
    return JavaScriptSSALoweringResult(
        IRModule({name: function_ir}),
        name,
        parameter_names,
        dict(kinds),
        result_kind,
        tuple(dict.fromkeys(shortfalls)),
    )


__all__ = [
    "JAVASCRIPT_KIND_DTYPES",
    "JavaScriptSSALoweringResult",
    "JavaScriptSourceShortfall",
    "javascript_jsdoc_parameter_kinds",
    "lower_javascript_source_to_ssa",
]
