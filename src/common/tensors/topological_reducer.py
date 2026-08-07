"""Topological reduction stage for AbstractTensor ProcessGraphs."""

from __future__ import annotations

import ast
import builtins
import copy
from dataclasses import MISSING, dataclass, fields, is_dataclass
import importlib
import logging
import types
from typing import Any, Callable

import networkx as nx

from ...transmogrifier.function_table import (
    ExternalFunctionTable,
    FunctionTable,
)
from ...transmogrifier.ssa_registry import ast_ssa_name_map, c_ssa_name_map


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _StaticPythonReference:
    """Ephemeral resolved Python object; never emitted as a graph value."""

    value: Any
    path: str


# The export table: which graph node types are runnable definitions
# (functions/methods), and how to read each one's own name off it. Python's
# three cases are the built-in defaults; a foreign language's ingestion
# (oop_language_translations.py's install_c_role_schemas, for C's
# pycparser.c_ast.FuncDef, say) extends this the same way role_schemas
# itself is extended -- registering its own node-type name and a callable
# to read that type's own name field, rather than this reducer growing an
# isinstance branch per language. This is the frontend-side half of
# FunctionTable already being "shared function references for ProcessGraph
# and SSA compilation" (function_table.py's own docstring) -- the table was
# already meant to be language-neutral; only *populating* it was Python-AST-
# specific until this registry existed.
_RUNNABLE_DEFINITION_NAME_EXTRACTORS: dict[str, Callable[[Any], str]] = {
    "FunctionDef": lambda node: str(node.name),
    "AsyncFunctionDef": lambda node: str(node.name),
    "Lambda": lambda node: (
        f"<lambda:{getattr(node, 'lineno', 0)}:"
        f"{getattr(node, 'col_offset', 0)}>"
    ),
}


def register_runnable_definition_type(
    type_name: str, name_extractor: Callable[[Any], str],
) -> None:
    """Register a foreign language's function/method-definition node type.

    ``type_name`` is ``type(node).__name__`` for that language's own
    function-definition node class (``"FuncDef"`` for ``pycparser``, say).
    ``name_extractor`` reads that node's own name (``pycparser.c_ast.FuncDef``
    keeps it at ``node.decl.name``, not ``node.name``, so this cannot be one
    generic ``.name`` access across languages). Idempotent: re-registering
    the same type name overwrites, it does not duplicate or error.
    """

    _RUNNABLE_DEFINITION_NAME_EXTRACTORS[str(type_name)] = name_extractor


def is_runnable_definition(node: Any) -> bool:
    """Is ``node`` a registered function/method-definition node, in any
    registered language -- the export-table membership test."""

    return type(node).__name__ in _RUNNABLE_DEFINITION_NAME_EXTRACTORS


def runnable_definition_name(node: Any) -> str:
    """The name of a registered function/method-definition node."""

    return _RUNNABLE_DEFINITION_NAME_EXTRACTORS[type(node).__name__](node)


# Same export-table pattern, for the two other node shapes call/return
# ownership walking needs to recognize: a call, and a return-shaped node.
# "Return" is deliberately one shared key for every registered language
# rather than a per-language key -- pycparser's own Return node is *also*
# literally named "Return" (not a naming collision to route around, an
# actual shared vocabulary word both languages use for the same construct).
#
# The returned value itself is read from the graph's own dependency
# structure (a node's "value"/"expr" parent-edge role -- role_schemas'
# Return: {"up": {"value": 1}} for Python, {"up": {"expr": 1}} for C), never
# from the raw source node's own attributes. This is deliberate, not
# incidental: everywhere else this session resolved a value this way
# (ShellMemoryReference.base_node_id, _resolve_reference_node in
# glsl_deployment_strategy.py) it went through the graph's own parents/role
# edges, not getattr on expr_obj -- the graph is the abstract, language-
# neutral representation; the raw node is not, and reaching back into it
# here would just reintroduce a second, Python-shaped assumption
# (node.value vs node.expr) into code meant to be language-neutral.
_CALL_SHAPED_TYPE_NAMES: set[str] = {"Call"}
_RETURN_VALUE_ROLES: set[str] = {"value", "expr"}


def register_call_shaped_type(type_name: str) -> None:
    """Register a foreign language's call-expression node type name."""

    _CALL_SHAPED_TYPE_NAMES.add(str(type_name))


def register_return_value_role(role: str) -> None:
    """Register a foreign language's own parent-edge role name for "this is
    the value a return-shaped node returns" (role_schemas' own "up" key for
    that language's Return-equivalent node)."""

    _RETURN_VALUE_ROLES.add(str(role))


def source_child_nodes(node: Any) -> tuple[Any, ...]:
    """Every direct child of a source node, in any registered language.

    The one traversal primitive a language must supply. Python spells it
    ``ast.iter_child_nodes``; pycparser spells it ``node.children()``
    (yielding ``(role_name, child)`` pairs). Neither understands the other:
    ``ast.NodeVisitor.generic_visit`` requires ``node._fields``, which
    pycparser nodes do not have, which is why walking a C body with an
    ``ast`` visitor raises ``AttributeError: 'Decl' object has no attribute
    '_fields'`` rather than simply finding nothing.

    Dispatch is on the protocol the node actually implements, not on a
    registered type name, because that is what genuinely varies -- and it
    means a third frontend whose nodes are ``ast``-shaped or
    ``children()``-shaped needs no registration here at all.
    """

    if isinstance(node, ast.AST):
        return tuple(ast.iter_child_nodes(node))
    children = getattr(node, "children", None)
    if callable(children):
        return tuple(child for _role, child in children())
    return ()


def source_body_statements(definition: Any) -> tuple[Any, ...]:
    """The ordered statements making up a function definition's body.

    Python keeps them directly on ``FunctionDef.body``; C wraps them in a
    ``Compound`` whose ``block_items`` holds the list (and which may be
    ``None`` for an empty body).
    """

    body = getattr(definition, "body", None)
    if body is None:
        return ()
    if isinstance(body, list):
        return tuple(body)
    block_items = getattr(body, "block_items", None)
    if block_items is not None:
        return tuple(block_items)
    # A lambda-style single-expression body is itself the only statement.
    return (body,)


def source_walk(node: Any) -> tuple[Any, ...]:
    """``ast.walk`` for any registered language, via ``source_child_nodes``.

    Callers filter the result with ``isinstance`` against the constructs
    they care about. That stays correct for a foreign language rather than
    merely not crashing: C genuinely has no ``ast.ExceptHandler`` and no
    Python-style ``for``, so a scan for those legitimately finds nothing,
    which is the right answer and not a silent gap.
    """

    collected: list[Any] = []
    stack = [node]
    while stack:
        current = stack.pop()
        collected.append(current)
        stack.extend(source_child_nodes(current))
    return tuple(collected)


def function_parameter_names(definition: Any) -> tuple[str, ...]:
    """A function definition's declared parameter names, in order.

    Genuinely per-language grammar rather than shared vocabulary, so this
    is one of the few places a language really must be taught its own
    shape: Python hangs them off ``FunctionDef.args`` as ``arg`` nodes
    across three lists (positional-only, ordinary, keyword-only), while C
    reaches ``FuncDef -> decl -> type -> args.params`` to a list of
    ``Decl`` nodes whose ``.name`` is a plain string.
    """

    arguments = getattr(definition, "args", None)
    if arguments is not None and hasattr(arguments, "posonlyargs"):
        return tuple(
            argument.arg
            for argument in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
            )
        )
    declaration = getattr(definition, "decl", None)
    if declaration is not None:
        parameter_list = getattr(getattr(declaration, "type", None), "args", None)
        return tuple(
            str(parameter.name)
            for parameter in (getattr(parameter_list, "params", None) or ())
            if getattr(parameter, "name", None) is not None
        )
    return ()


def _record_owned_calls_and_returns(
    graph: Any,
    owner_reference: Any,
    function_node_id: int,
    call_owners: dict[int, Any],
    function_return_values: dict[int, list[int]],
) -> None:
    """Walk ``graph.G`` (not the raw source tree) to find every call/return
    belonging to the function at ``function_node_id``, stopping at any
    nested function/method-definition boundary -- the graph-based,
    language-neutral replacement for a source-tree ``ast.NodeVisitor`` walk,
    which cannot walk a foreign language's own node types at all (a
    ``pycparser`` node has no ``_fields`` attribute Python's
    ``ast.NodeVisitor.generic_visit`` requires). Bounded, not
    ``nx.ancestors``' full transitive closure: a nested definition's own
    calls/returns belong to *it*, not this enclosing function, the same
    rule the original visitor's ``visit_FunctionDef`` returning ``None``
    (not recursing) enforced.
    """

    visited: set[int] = set()
    stack = list(graph.G.predecessors(function_node_id))
    while stack:
        node_id = stack.pop()
        if node_id in visited or node_id not in graph.G:
            continue
        visited.add(node_id)
        data = graph.G.nodes[node_id]
        expression = data.get("expr_obj")
        if node_id != function_node_id and is_runnable_definition(expression):
            # A nested definition owns its own calls/returns; do not
            # descend into it from here.
            continue
        node_type = str(data.get("type"))
        if node_type in _CALL_SHAPED_TYPE_NAMES:
            call_owners[node_id] = owner_reference
        if node_type == "Return":
            # The returned value's own node id, read off the Return node's
            # graph-native parent edges (role "value" for Python, "expr" for
            # C) -- never off the raw source node's attributes. Already a
            # graph node id (an "up" edge's producer), not a raw object,
            # so it needs no id()/wrapping before use.
            value_node_id = next(
                (
                    parent_id
                    for parent_id, role in (data.get("parents") or ())
                    if str(role) in _RETURN_VALUE_ROLES
                ),
                None,
            )
            if value_node_id is not None:
                function_return_values.setdefault(
                    function_node_id, [],
                ).append(value_node_id)
        stack.extend(graph.G.predecessors(node_id))


# Source-language node type name -> the canonical SSA vocabulary spelling it
# universalizes to. The point of routing through the registry rather than
# hardcoding strings here is that a second language's own spelling for the
# same construct lands on the *same* canonical type: C's ``FuncCall`` and
# Python's ``Call`` both become "Call", so downstream passes see one
# vocabulary instead of one per frontend. Extend by registering the foreign
# spelling in ssa_registry.py (c_ssa_equivalents, say) and naming its node
# type here -- not by adding a branch downstream.
_AST_PROCESS_GRAPH_ALIASES = {
    "Name": ast_ssa_name_map["name"].value,
    "Assign": ast_ssa_name_map["assign"].value,
    "Call": ast_ssa_name_map["call"].value,
    # C (pycparser c_ast), joining the same canonical spellings above.
    # StructRef is C's field read (``self->value``), the same reading
    # Python's ``attribute:load`` has -- both are Load.
    "ID": c_ssa_name_map["id"].value,
    "Assignment": c_ssa_name_map["assignment"].value,
    "FuncCall": c_ssa_name_map["funccall"].value,
    "StructRef": c_ssa_name_map["structref"].value,
}

# C node types that exist only to spell out a *type*, carrying no runtime
# value of their own. Python's grammar has no equivalent (it is untyped at
# the syntax level), which is why this set has no Python counterpart --
# these are stripped for the same reason ast.Nonlocal/ast.Global are:
# compile-time syntax, not operations. ``Decl`` is deliberately NOT here --
# it carries both the declared name and its initializer, so it is a real
# binding event, not type machinery.
_C_COMPILE_TIME_SYNTAX = frozenset({
    "TypeDecl",
    "FuncDecl",
    "PtrDecl",
    "ArrayDecl",
    "IdentifierType",
    "ParamList",
    "Typedef",
    "Struct",
    "Union",
    "Enum",
})

_BITOPS_TO_EXECUTABLE = {
    "Xor": "bitxor",
    "Neg": "neg",
    "Eq": "equal",
    "Ne": "not_equal",
    "Lt": "less",
    "Le": "less_equal",
    "Gt": "greater",
    "Ge": "greater_equal",
    "MatMul": "matmul",
    "LAnd": "logical_and",
    "LOr": "logical_or",
    "LNot": "logical_not",
}


def _qualified_handler(prefix: str, operator: ast.AST) -> str:
    spelling = f"{prefix}:{type(operator).__name__.lower()}"
    handler = ast_ssa_name_map.get(spelling)
    if handler is None:
        raise KeyError(f"no existing operator alias for {spelling!r}")
    return _BITOPS_TO_EXECUTABLE.get(handler.value, handler.value)


def _c_qualified_handler(prefix: str, operator: str) -> str:
    """``_qualified_handler`` for C, whose operators are strings not classes.

    Python spells an operator as a child *node class* (``ast.Add``, hence
    ``binop:add``); pycparser spells it as a plain string attribute on the
    parent (``BinaryOp.op == '+'``, hence ``binaryop:+``). Two surface
    spellings, one canonical Handler -- which is the whole point of the
    registry, and why this returns the identical vocabulary
    ``_qualified_handler`` does rather than a parallel C-flavored one.
    """

    spelling = f"{prefix}:{str(operator).lower()}"
    handler = c_ssa_name_map.get(spelling)
    if handler is None:
        raise KeyError(f"no existing C operator alias for {spelling!r}")
    return _BITOPS_TO_EXECUTABLE.get(handler.value, handler.value)


def _c_constant_value(expression: Any) -> Any:
    """A ``pycparser`` Constant's real Python value.

    pycparser keeps every literal as the *source text* plus a type name
    (``Constant(type='int', value='10')`` -- the value is the string "10",
    not the integer 10). Downstream compiler stages treat ``data["constant"]``
    as a real value, so the conversion has to happen here, at the one place
    C literals are canonicalized, rather than being re-derived (or silently
    left as a string) at each consumer.
    """

    raw = str(getattr(expression, "value", ""))
    kind = str(getattr(expression, "type", ""))
    if "char" in kind:
        return raw.strip("'")
    if "string" in kind:
        return raw.strip('"')
    if "float" in kind or "double" in kind:
        return float(raw.rstrip("fFlL"))
    if "int" in kind:
        # C integer literals carry base prefixes and width/sign suffixes that
        # Python's int() will not accept directly (0x1FUL, 10u, 07).
        text = raw.rstrip("uUlL")
        return int(text, 0) if text else 0
    return raw


# The type given to a node standing in for something ingestion could not
# translate. Mangled data is mangled data: it needs a real, named, *non*
# operation rather than either a silent hole or a hard stop.
#
# This type name MUST NEVER be added to any operator table -- not
# ``operator_defs.operator_signatures``' executable entries, not
# ``graph_deep_compiler``'s ``op_table``, not ``ssa_webgpu_backend``'s
# ``_BINARY``/``_UNARY``, not ``fused_ir``'s ``ELEMENTWISE_*``. Its safety is
# structural, not a matter of anyone remembering to check: because no table
# maps it, every backend that meets it already reports it through that
# backend's own existing "no instruction for this" path. Nothing can quietly
# treat it as zero, which is the one line this repository does not cross --
# never silently produce a plausible wrong number.
#
# It is deliberately *one* generic type carrying free-text provenance, not a
# taxonomy of failure kinds. A closed enum of defect categories would need a
# central owner to admit every new construct a contributor or agent adds --
# exactly the bottleneck ``role_schemas``,
# ``_RUNNABLE_DEFINITION_NAME_EXTRACTORS`` and the ssa_registry spelling
# tables exist to remove. ``operator_signatures`` falls back to 'Default'
# for an unrecognized type, so a graph carrying these still finalizes and
# stays walkable; the translation simply reports how far it got.
UNTRANSLATED_NODE_TYPE = "Untranslated"


# How completely the affected construct survived translation. This is an
# ordered *scale*, not a taxonomy -- which is why it is a closed set where
# ``reason`` is deliberately free text. A scale has rungs that mean
# something relative to each other and can be compared and sorted; a
# taxonomy of failure kinds would need a central owner to admit every new
# construct a contributor or agent introduces, which is the bottleneck this
# module keeps removing. Add a rung only if it is genuinely between or
# beyond these, never to describe a new *kind* of failure.
#
# Nothing here is binary. A translation is not "worked" or "failed": it
# reached some depth and stopped, and how far it reached is the useful
# fact. ``complete`` on the backend artifacts is the cheapest summary of
# this scale, not a replacement for it.
TRANSLATION_ABSENT = "absent"      # nothing of the construct translated
TRANSLATION_PARTIAL = "partial"    # some constituents translated, some not
TRANSLATION_DEGRADED = "degraded"  # fully translated, reduced fidelity
TRANSLATION_UNKNOWN = "unknown"    # not determinable at this pass

TRANSLATION_GRADES = (
    TRANSLATION_ABSENT,
    TRANSLATION_PARTIAL,
    TRANSLATION_DEGRADED,
    TRANSLATION_UNKNOWN,
)


@dataclass(frozen=True)
class GraphTranslationShortfall:
    """One construct graph ingestion could not fully translate.

    Same contract the backend shortfall records already use
    (``WasmShortfall``, ``WGSLShortfall``): structured fields for the parts
    worth querying across many programs, and one free-text ``reason`` so
    reporting something new never requires extending a type. Recorded on
    ``graph.G.graph["translation_shortfalls"]``, alongside the existing
    ``state_machine_control_shortfalls``.

    ``grade`` carries how far the translation got, and ``observed``/
    ``expected`` the countable evidence behind that grade (operands
    resolved out of operands required, say). A reader deciding whether a
    specimen is solid enough to reason about needs the degree, not just the
    fact -- and a caller aggregating across many programs needs the counts,
    not a prose summary of them.
    """

    pass_name: str
    node_id: int
    operation: str
    role: str
    source_span: str
    reason: str
    grade: str = TRANSLATION_UNKNOWN
    observed: int = 0
    expected: int = 0

    @property
    def coverage(self) -> float | None:
        """Fraction of the construct that translated, when countable."""

        if self.expected <= 0:
            return None
        return self.observed / self.expected

    def format(self) -> str:
        where = f" at {self.source_span}" if self.source_span else ""
        role = f" ({self.role})" if self.role else ""
        ratio = (
            f" [{self.observed}/{self.expected}]" if self.expected else ""
        )
        return (
            f"{self.pass_name}: {self.operation}{role}{where} "
            f"<{self.grade}{ratio}>: {self.reason}"
        )


# Cap on a folded sequence, so `[0] * 10**9` is refused rather than
# materialized. A refused fold is not a failure: it falls through to the
# ordinary path and reports itself there like anything else.
_MAX_FOLDED_SEQUENCE_ELEMENTS = 1 << 20


def _static_sequence_literal(expression: Any) -> Any:
    """Fold literal sequence replication (``[0] * 256``) to its value.

    Returns the folded list, or ``None`` when this is not that construct.

    ``[0] * 256`` is **Python list replication, not arithmetic** -- but the
    AST spells it ``BinOp(op=Mult)``, so canonicalization would otherwise
    rewrite it into a ``Mul`` dataflow operation that was never arithmetic.
    Its operands are not in the graph either (a literal list inside a
    comprehension is never descended into), so the ``Mul`` ends up with two
    absent operands.

    This is compile-time table allocation -- the shape a decoder's lookup
    tables are declared in -- and its value is knowable now, so it becomes
    constant data. Only the both-operands-literal case folds; anything
    depending on a runtime value is left alone.
    """

    if not isinstance(expression, ast.BinOp) or not isinstance(
        expression.op, ast.Mult
    ):
        return None
    for sequence_node, count_node in (
        (expression.left, expression.right),
        (expression.right, expression.left),
    ):
        if not isinstance(sequence_node, (ast.List, ast.Tuple)):
            continue
        try:
            sequence = ast.literal_eval(sequence_node)
            count = ast.literal_eval(count_node)
        except (ValueError, TypeError, SyntaxError, MemoryError):
            return None
        if isinstance(count, bool) or not isinstance(count, int):
            return None
        if count < 0:
            return None
        if len(sequence) * count > _MAX_FOLDED_SEQUENCE_ELEMENTS:
            return None
        return list(sequence) * count
    return None


def _source_span(expression: Any) -> str:
    line = getattr(expression, "lineno", None)
    if line is None:
        return ""
    column = getattr(expression, "col_offset", None)
    return f"line {line}" + (f":{column}" if column is not None else "")


def record_translation_shortfall(
    graph: Any,
    *,
    pass_name: str,
    node_id: int,
    operation: str,
    reason: str,
    role: str = "",
    source_span: str = "",
    grade: str = TRANSLATION_UNKNOWN,
    observed: int = 0,
    expected: int = 0,
) -> None:
    """Accumulate one graph-level shortfall instead of raising.

    Ingestion reports the way the backends already do -- a partial
    translation is a real, inspectable result, not a failure. Raising here
    would surface exactly one defect per run of a multi-minute build, when
    the same run could have named every one of them.
    """

    existing = graph.G.graph.get("translation_shortfalls") or ()
    graph.G.graph["translation_shortfalls"] = (
        *existing,
        GraphTranslationShortfall(
            pass_name=str(pass_name),
            node_id=int(node_id),
            operation=str(operation),
            role=str(role),
            source_span=str(source_span),
            reason=str(reason),
            grade=str(grade),
            observed=int(observed),
            expected=int(expected),
        ),
    )


def _replace_inputs(
    graph: Any,
    node_id: int,
    inputs: tuple[tuple[int, str], ...],
) -> None:
    """Replace one wrapper's incoming topology with executable operands.

    An operand that is not in the graph is replaced by an explicit
    ``UNTRANSLATED_NODE_TYPE`` node rather than passed to ``add_edge``.
    NetworkX *creates* an absent endpoint rather than raising, which turned
    a missing operand into a node with no ``type``, no ``expr_obj`` and no
    ``label`` -- invisible here and fatal thousands of nodes later, as a
    bare ``KeyError('type')`` naming nothing. The placeholder keeps the
    consuming operation's arity intact (an operation silently short an
    operand is a *wrong* program, not a smaller one) while making it
    honestly inexecutable.
    """

    for predecessor in tuple(graph.G.predecessors(node_id)):
        graph.G.remove_edge(predecessor, node_id)
        graph.G.nodes[predecessor]["children"] = [
            (child_id, role)
            for child_id, role in graph.G.nodes[predecessor].get(
                "children", ()
            )
            if child_id != node_id
        ]
    consumer = graph.G.nodes[node_id] if node_id in graph.G else {}
    expected = len(inputs)
    present = sum(1 for operand, _role in inputs if operand in graph.G)
    # The grade describes the *construct*, not the individual operand: an
    # operation with one of two operands resolved is partially translated,
    # even though each absent operand is individually absent.
    grade = TRANSLATION_ABSENT if present == 0 else TRANSLATION_PARTIAL
    resolved: list[tuple[int, str]] = []
    for predecessor, role in inputs:
        if predecessor not in graph.G:
            predecessor = _untranslated_operand(
                graph,
                consumer_id=node_id,
                consumer=consumer,
                role=role,
                absent_id=predecessor,
                grade=grade,
                observed=present,
                expected=expected,
            )
        resolved.append((predecessor, role))
    graph.G.nodes[node_id]["parents"] = list(resolved)
    for predecessor, role in resolved:
        graph.G.add_edge(predecessor, node_id, role=role)
        children = graph.G.nodes[predecessor].setdefault("children", [])
        if node_id not in {child_id for child_id, _role in children}:
            children.append((node_id, role))


def _untranslated_operand(
    graph: Any,
    *,
    consumer_id: int,
    consumer: Any,
    role: str,
    absent_id: int,
    grade: str = TRANSLATION_UNKNOWN,
    observed: int = 0,
    expected: int = 0,
) -> int:
    """Materialize one explicit stand-in for an operand that never arrived."""

    expression = consumer.get("expr_obj") if hasattr(consumer, "get") else None
    operation = str(consumer.get("type") or type(expression).__name__)
    span = _source_span(expression)
    label = f"untranslated[{operation}.{role}]"
    placeholder_id = id(label)
    while placeholder_id in graph.G:
        # ``id`` of a fresh string is unique among live objects, but this
        # label is not retained, so a later identical label could reuse the
        # address. Step off any collision rather than silently aliasing two
        # unrelated stand-ins onto one node.
        placeholder_id += 1
    graph.G.add_node(
        placeholder_id,
        label=label,
        type=UNTRANSLATED_NODE_TYPE,
        op=UNTRANSLATED_NODE_TYPE.lower(),
        expr_obj=None,
        extra_args={},
        domain_node=None,
        store_id=None,
        parents=[],
        children=[],
        attributes={
            "untranslated": True,
            "consumer_node": int(consumer_id),
            "consumer_operation": operation,
            "operand_role": str(role),
            "absent_node_id": int(absent_id),
            "source_span": span,
            "translation_grade": str(grade),
            "translated_operands": int(observed),
            "expected_operands": int(expected),
        },
    )
    record_translation_shortfall(
        graph,
        pass_name="process-graph-operands",
        node_id=consumer_id,
        operation=operation,
        role=role,
        source_span=span,
        grade=grade,
        observed=observed,
        expected=expected,
        reason=(
            "operand was never ingested as a graph node; a non-operation "
            "stands in its place so the translation reports rather than "
            "computes"
        ),
    )
    return placeholder_id


def _remove_node(graph: Any, node_id: int) -> None:
    """Remove one reduced-away node and its cached adjacency metadata."""

    if node_id not in graph.G:
        return
    for predecessor in tuple(graph.G.predecessors(node_id)):
        graph.G.nodes[predecessor]["children"] = [
            (child_id, role)
            for child_id, role in graph.G.nodes[predecessor].get(
                "children", ()
            )
            if child_id != node_id
        ]
    for successor in tuple(graph.G.successors(node_id)):
        graph.G.nodes[successor]["parents"] = [
            (parent_id, role)
            for parent_id, role in graph.G.nodes[successor].get(
                "parents", ()
            )
            if parent_id != node_id
        ]
    graph.roots = [root for root in graph.roots if root != node_id]
    graph.G.remove_node(node_id)


def _redirect_value(
    graph: Any,
    old_id: int,
    producer_id: int,
) -> None:
    """Fan every use of one lexical occurrence out from its value producer."""

    if old_id == producer_id or old_id not in graph.G:
        return
    for successor in tuple(graph.G.successors(old_id)):
        successor_data = graph.G.nodes[successor]
        replacement = []
        for parent_id, role in successor_data.get("parents", ()):
            replacement.append(
                (producer_id if parent_id == old_id else parent_id, role)
            )
        successor_data["parents"] = replacement
        graph.G.add_edge(producer_id, successor, role=graph.G.edges[
            old_id,
            successor,
        ].get("role"))
        children = graph.G.nodes[producer_id].setdefault("children", [])
        for _parent_id, role in replacement:
            if _parent_id == producer_id and (
                successor,
                role,
            ) not in {
                (child_id, child_role)
                for child_id, child_role in children
            }:
                children.append((successor, role))
    graph.roots = [
        producer_id if root == old_id else root for root in graph.roots
    ]
    _remove_node(graph, old_id)


def _normalize_lexical_values(
    function_graph: Any,
    statement: ast.FunctionDef | ast.AsyncFunctionDef,
    static_bindings: dict[str, Any],
    function_table: FunctionTable,
    lexical_function_bindings: dict[str, Any] | None = None,
) -> None:
    """Resolve unique lexical occurrences into a monotonic value DAG.

    AST ingestion intentionally gives every ``Name`` occurrence its own node.
    At topological reduction, loads are consolidated against the definition
    visible at that program point.  Consumers then fan out directly from the
    defining value; source-language ``Load`` and ``Store`` wrappers disappear.
    """

    graph = function_graph
    environment: dict[str, int] = {}
    static_environment: dict[str, _StaticPythonReference] = {}
    deleted_names: set[str] = set()
    identity_bindings: dict[str, list[int]] = {}
    loop_target_bindings_by_ast: dict[int, int] = {}
    static_reference_nodes: dict[tuple[int, str], int] = {}
    first_class_function_nodes: dict[int, int] = {}
    static_constant_nodes: dict[str, int] = {}
    static_attribute_values: dict[tuple[int, str], int] = {}
    parameter_names = set(function_parameter_names(statement))
    exception_local_names = {
        target.id
        for handler in (
            node
            for node in source_walk(statement)
            if isinstance(node, ast.ExceptHandler)
        )
        for body_node in handler.body
        for assignment in source_walk(body_node)
        if isinstance(assignment, (ast.Assign, ast.AnnAssign, ast.AugAssign))
        for target in (
            (*assignment.targets,)
            if isinstance(assignment, ast.Assign)
            else (assignment.target,)
        )
        if isinstance(target, ast.Name)
    }
    scalar_loop_target_ast_ids: set[int] = set()

    def target_name_nodes(target: ast.AST) -> tuple[ast.Name, ...]:
        if isinstance(target, ast.Name):
            return (target,)
        if isinstance(target, (ast.Tuple, ast.List)):
            return tuple(
                name
                for element in target.elts
                for name in target_name_nodes(element)
            )
        return ()

    for loop in source_walk(statement):
        if not (
            isinstance(loop, ast.For)
            and isinstance(loop.iter, ast.Call)
            and isinstance(loop.iter.func, ast.Name)
        ):
            continue
        if loop.iter.func.id == "range":
            scalar_loop_target_ast_ids.update(
                id(name) for name in target_name_nodes(loop.target)
            )
        elif loop.iter.func.id == "enumerate":
            if isinstance(loop.target, (ast.Tuple, ast.List)) and loop.target.elts:
                scalar_loop_target_ast_ids.update(
                    id(name)
                    for name in target_name_nodes(loop.target.elts[0])
                )
    scalar_loop_binding_ids: set[int] = set()
    for body_statement in source_body_statements(statement):
        if not isinstance(body_statement, ast.Return):
            continue
        returned = body_statement.value
        expressions = (
            tuple(returned.elts)
            if isinstance(returned, (ast.Tuple, ast.List))
            else (returned,)
        )
        graph.G.graph["function_outputs"] = tuple(
            expression.id
            if isinstance(expression, ast.Name)
            else f"result_{index}"
            for index, expression in enumerate(expressions)
        )
        break
    temporary_id = max(graph.G.nodes, default=-1) + 1

    def new_node(
        node_type: str,
        label: str,
        *,
        attributes: dict[str, Any] | None = None,
        parents: tuple[tuple[int, str], ...] = (),
    ) -> int:
        nonlocal temporary_id
        node_id = temporary_id
        temporary_id += 1
        graph.G.add_node(
            node_id,
            label=label,
            type=node_type,
            op=node_type.lower(),
            expr_obj=None,
            extra_args={},
            domain_node=None,
            store_id=None,
            parents=list(parents),
            children=[],
            attributes=dict(attributes or {}),
        )
        if node_type in {"Const", "Constant"}:
            graph.G.nodes[node_id]["constant"] = (
                attributes or {}
            ).get("value")
        for parent_id, role in parents:
            if parent_id not in graph.G:
                continue
            graph.G.add_edge(parent_id, node_id, role=role)
            graph.G.nodes[parent_id].setdefault("children", []).append(
                (node_id, role)
            )
        return node_id

    def input_value(name: str, *, binding_kind: str) -> int:
        value = environment.get(name)
        if value is not None:
            return value
        value = new_node(
            "Input",
            name,
            attributes={
                "binding_name": name,
                "binding_kind": binding_kind,
            },
        )
        environment[name] = value
        identity_bindings.setdefault(name, []).append(value)
        return value

    def static_constant(name: str, value: Any) -> int:
        existing = static_constant_nodes.get(name)
        if existing is not None:
            return existing
        node_id = new_node(
            "Constant",
            name,
            attributes={"value": value, "binding_name": name},
        )
        static_constant_nodes[name] = node_id
        return node_id

    def is_static_literal(value: Any) -> bool:
        if value is None or isinstance(
            value,
            (bool, bytes, complex, float, int, str),
        ):
            return True
        if isinstance(value, (tuple, list)):
            return all(is_static_literal(item) for item in value)
        if isinstance(value, dict):
            return all(
                is_static_literal(key) and is_static_literal(item)
                for key, item in value.items()
            )
        return False

    def static_reference_node(reference: _StaticPythonReference) -> int:
        """Materialize one compiler reference without exposing its Python value."""

        key = (id(reference.value), reference.path)
        existing = static_reference_nodes.get(key)
        if existing is not None:
            return existing
        target = reference.value
        target_name = str(getattr(target, "__name__", ""))
        class_descriptor = graph.G.graph.get("class_table", {}).get(
            target_name
        )
        if class_descriptor is not None and is_dataclass(target):
            # Imported dataclasses enter the function table as structural
            # class references, but their defining ClassDef belongs to a
            # different source unit.  Recover the constructor schema here so
            # omitted literal defaults (for example SuperstepPlan.eps) remain
            # compiler facts instead of disappearing into Python reflection
            # at execution time.
            dataclass_fields = tuple(fields(target))
            class_descriptor["fields"] = tuple(
                field.name for field in dataclass_fields
            )
            defaults = dict(class_descriptor.get("field_defaults") or {})
            for field in dataclass_fields:
                if field.default is not MISSING and is_static_literal(
                    field.default
                ):
                    defaults[field.name] = copy.deepcopy(field.default)
            class_descriptor["field_defaults"] = defaults
        function_reference = (
            function_table.reference(target_name) if target_name else None
        )
        if getattr(target, "__self__", None) is not None:
            # A bound runtime/builtin method is identified by both callable
            # and receiver.  Linking it to an unrelated source method by the
            # final spelling alone (for example ContextVar.get versus a
            # user class's get) discards the receiver and is never valid.
            function_reference = None
        attributes = {
            "static_python_reference": reference.path,
            "reference_kind": (
                "function_subgraph"
                if function_reference is not None
                else (
                    "class_subgraphs"
                    if class_descriptor is not None
                    else "static_symbol"
                )
            ),
        }
        if function_reference is not None:
            attributes["function_ref"] = function_reference.address
        if class_descriptor is not None:
            attributes["class_ref"] = target_name
        node_id = new_node(
            "StaticReference",
            reference.path,
            attributes=attributes,
        )
        static_reference_nodes[key] = node_id
        return node_id

    def first_class_function_node(name: str, reference: Any) -> int:
        """Represent a source function used as data by its table address."""

        address = int(reference.address)
        existing = first_class_function_nodes.get(address)
        if existing is not None:
            return existing
        node_id = new_node(
            "StaticReference",
            name,
            attributes={
                "function_ref": address,
                "first_class_function_ref": address,
                "reference_kind": "function_subgraph",
            },
        )
        first_class_function_nodes[address] = node_id
        return node_id

    def bind_loop_target(target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            target_identity = id(target)
            value = loop_target_bindings_by_ast.get(target_identity)
            if value is None:
                value = new_node(
                    "Input",
                    target.id,
                    attributes={
                        "binding_name": target.id,
                        "binding_kind": "loop",
                    },
                )
                loop_target_bindings_by_ast[target_identity] = value
                identity_bindings.setdefault(target.id, []).append(value)
            if target_identity in scalar_loop_target_ast_ids:
                scalar_loop_binding_ids.add(value)
            environment[target.id] = value
            _remove_node(graph, id(target))
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                bind_loop_target(element)

    def loop_target_names(target: ast.AST) -> tuple[str, ...]:
        if isinstance(target, ast.Name):
            return (target.id,)
        if isinstance(target, (ast.Tuple, ast.List)):
            return tuple(
                name
                for element in target.elts
                for name in loop_target_names(element)
            )
        return ()

    def resolve_expression(expression: ast.AST | None) -> int | None:
        if expression is None:
            return None
        if isinstance(expression, ast.NamedExpr):
            value = resolve_expression(expression.value)
            bind_target(expression.target, value)
            node_id = id(expression)
            if isinstance(value, int):
                _redirect_value(graph, node_id, value)
            else:
                _remove_node(graph, node_id)
            return value
        if isinstance(
            expression,
            (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp),
        ):
            # Comprehensions and generator expressions have owned their own
            # scope since Python 3.0: a `for x in ...` target inside one is
            # never visible outside it, even when an outer variable of the
            # same name already exists or gets bound later. (The one
            # exception, a `:=` walrus target, deliberately leaks to the
            # enclosing scope per PEP 572 -- that goes through bind_target,
            # not bind_loop_target, and is untouched here.) Without this,
            # a comprehension's `for` target silently overwrote (or was
            # later mistaken for) an unrelated same-named binding elsewhere
            # in the function -- for example `tuple(f(x) for x in seq)`
            # followed later by an ordinary `for x in other:` picking up
            # the comprehension's already-evaporated node as its own
            # "before this loop" value, and crashing much later with
            # "missing ProcessGraph input" once that node was removed.
            comprehension_target_names = {
                name
                for generator in expression.generators
                for name in loop_target_names(generator.target)
            }
            shadowed_bindings = {
                name: environment[name]
                for name in comprehension_target_names
                if name in environment
            }
            for generator in expression.generators:
                resolve_expression(generator)
            if isinstance(expression, ast.DictComp):
                resolve_expression(expression.key)
                resolve_expression(expression.value)
            else:
                resolve_expression(expression.elt)
            node_id = id(expression)
            if isinstance(expression, (ast.ListComp, ast.SetComp)):
                value_id = resolve_expression(expression.elt)
                for generator in expression.generators:
                    generator_id = id(generator)
                    if (
                        generator_id in graph.G
                        and isinstance(value_id, int)
                        and value_id in graph.G
                        and node_id in graph.G
                    ):
                        attributes = graph.G.nodes[
                            generator_id
                        ].setdefault("attributes", {})
                        outputs = list(
                            attributes.get("loop_iteration_outputs", ())
                        )
                        outputs.append({
                            "value_id": value_id,
                            "result_value_id": node_id,
                            "materializer_node_id": node_id,
                        })
                        attributes["loop_iteration_outputs"] = tuple(outputs)
            for name in comprehension_target_names:
                if name in shadowed_bindings:
                    environment[name] = shadowed_bindings[name]
                else:
                    environment.pop(name, None)
            return node_id if node_id in graph.G else None
        if isinstance(expression, ast.comprehension):
            resolve_expression(expression.iter)
            bind_loop_target(expression.target)
            node_id = id(expression)
            if node_id in graph.G:
                graph.G.nodes[node_id].setdefault("attributes", {})[
                    "loop_target_bindings"
                ] = {
                    name: environment[name]
                    for name in loop_target_names(expression.target)
                }
            for condition in expression.ifs:
                resolve_expression(condition)
            return node_id if node_id in graph.G else None
        if isinstance(expression, ast.Name):
            node_id = id(expression)
            if isinstance(expression.ctx, ast.Load):
                static_reference = static_environment.get(expression.id)
                if static_reference is not None:
                    _remove_node(graph, node_id)
                    return static_reference
                producer_id = environment.get(expression.id)
                static_value = static_bindings.get(expression.id)
                function_reference = (
                    (lexical_function_bindings or {}).get(expression.id)
                    or function_table.reference(expression.id)
                )
                if (
                    producer_id is None
                    and expression.id not in parameter_names
                    and function_reference is not None
                ):
                    _remove_node(graph, node_id)
                    return first_class_function_node(
                        expression.id,
                        function_reference,
                    )
                if (
                    producer_id is None
                    and expression.id not in parameter_names
                    and static_value is not None
                    and isinstance(
                        static_value,
                        (
                            types.ModuleType,
                            type,
                            types.FunctionType,
                            types.BuiltinFunctionType,
                            types.MethodType,
                        ),
                    )
                ):
                    _remove_node(graph, node_id)
                    return _StaticPythonReference(
                        static_value,
                        expression.id,
                    )
                if (
                    producer_id is None
                    and expression.id not in parameter_names
                    and expression.id in static_bindings
                    and is_static_literal(static_value)
                ):
                    constant_id = static_constant(
                        expression.id,
                        static_value,
                    )
                    _redirect_value(graph, node_id, constant_id)
                    return constant_id
                if producer_id is None:
                    producer_id = input_value(
                        expression.id,
                        binding_kind=(
                            "parameter"
                            if expression.id in parameter_names
                            else "exception"
                            if expression.id in exception_local_names
                            else "external"
                        ),
                    )
                _redirect_value(graph, node_id, producer_id)
                return producer_id
            return environment.get(expression.id)
        if isinstance(expression, ast.Attribute):
            if (
                isinstance(expression.value, ast.Name)
                and expression.value.id not in environment
                and expression.value.id not in parameter_names
                and expression.value.id in static_bindings
            ):
                _remove_node(graph, id(expression.value))
                receiver = _StaticPythonReference(
                    static_bindings[expression.value.id],
                    expression.value.id,
                )
            else:
                receiver = resolve_expression(expression.value)
            if isinstance(receiver, _StaticPythonReference):
                attribute_key = (id(receiver.value), expression.attr)
                assigned_value = static_attribute_values.get(attribute_key)
                if assigned_value is not None:
                    _redirect_value(graph, id(expression), assigned_value)
                    return assigned_value
                try:
                    value = getattr(receiver.value, expression.attr)
                except AttributeError:
                    pass
                else:
                    node_id = id(expression)
                    if is_static_literal(value):
                        constant_id = static_constant(
                            f"{receiver.path}.{expression.attr}",
                            value,
                        )
                        _redirect_value(graph, node_id, constant_id)
                        return constant_id
                    _remove_node(graph, node_id)
                    return _StaticPythonReference(
                        value,
                        f"{receiver.path}.{expression.attr}",
                    )
            elif isinstance(receiver, int) and id(expression) in graph.G:
                _replace_inputs(
                    graph,
                    id(expression),
                    ((receiver, "value"),),
                )

        if isinstance(expression, ast.Call):
            node_id = id(expression)
            if (
                isinstance(expression.func, ast.Name)
                and expression.func.id
                in (graph.G.graph.get("class_definitions") or ())
                and node_id in graph.G
            ):
                # A source-local class construction, already known from
                # ``map_ir`` at ingestion -- no python_bindings/static
                # reference resolution needed or wanted for this: the
                # class isn't external, it's right here in the source.
                graph.G.nodes[node_id].setdefault(
                    "attributes", {},
                )["class_ref"] = expression.func.id
            callee = resolve_expression(expression.func)
            if isinstance(callee, int) and callee in graph.G:
                callee_reference = (
                    graph.G.nodes[callee].get("attributes") or {}
                ).get("function_ref")
                if callee_reference is not None and node_id in graph.G:
                    graph.G.nodes[node_id].setdefault(
                        "attributes",
                        {},
                    )["callee_ref"] = callee_reference
            if isinstance(callee, _StaticPythonReference) and node_id in graph.G:
                reference_node_id = static_reference_node(callee)
                attributes = graph.G.nodes[node_id].setdefault(
                    "attributes",
                    {},
                )
                attributes["static_python_reference"] = callee.path
                reference_attributes = graph.G.nodes[
                    reference_node_id
                ].get("attributes") or {}
                if "function_ref" in reference_attributes:
                    attributes["callee_ref"] = reference_attributes[
                        "function_ref"
                    ]
                if "class_ref" in reference_attributes:
                    attributes["class_ref"] = reference_attributes[
                        "class_ref"
                    ]
                if str(graph.G.nodes[node_id].get("type")) == "Call":
                    if not graph.G.has_edge(reference_node_id, node_id):
                        graph.G.add_edge(
                            reference_node_id,
                            node_id,
                            role="callee",
                        )
                        graph.G.nodes[reference_node_id].setdefault(
                            "children",
                            [],
                        ).append((node_id, "callee"))
                    parents = graph.G.nodes[node_id].setdefault(
                        "parents",
                        [],
                    )
                    if (reference_node_id, "callee") not in parents:
                        parents.append((reference_node_id, "callee"))
                else:
                    # A canonical numerical operation already identifies its
                    # implementation by node type.  Preserve the wrapper as
                    # compiler metadata, but never feed it to the operation as
                    # tensor data.
                    attributes["operator_reference_node"] = reference_node_id
                static_arguments = {}
                for index, argument in enumerate(expression.args):
                    resolved = resolve_expression(argument)
                    if isinstance(resolved, _StaticPythonReference):
                        static_arguments[f"arg:{index}"] = resolved.path
                        attributes.setdefault(
                            "static_call_values", {}
                        )[f"arg:{index}"] = resolved.value
                        graph.G.graph.setdefault(
                            "static_python_values", {}
                        )[resolved.path] = resolved.value
                    elif (
                        isinstance(resolved, int)
                        and resolved in graph.G
                        and (graph.G.nodes[resolved].get("attributes") or {}).get(
                            "first_class_function_ref"
                        ) is not None
                    ):
                        role = f"arg:{index}"
                        if not graph.G.has_edge(resolved, node_id):
                            graph.G.add_edge(resolved, node_id, role=role)
                        children = graph.G.nodes[resolved].setdefault(
                            "children", []
                        )
                        if (node_id, role) not in children:
                            children.append((node_id, role))
                        parents = graph.G.nodes[node_id].setdefault(
                            "parents", []
                        )
                        if (resolved, role) not in parents:
                            parents.append((resolved, role))
                for keyword in expression.keywords:
                    if keyword.arg is None:
                        continue
                    resolved = resolve_expression(keyword.value)
                    if isinstance(resolved, _StaticPythonReference):
                        static_arguments[f"kw:{keyword.arg}"] = resolved.path
                        attributes.setdefault(
                            "static_call_values", {}
                        )[f"kw:{keyword.arg}"] = resolved.value
                        graph.G.graph.setdefault(
                            "static_python_values", {}
                        )[resolved.path] = resolved.value
                    elif (
                        isinstance(resolved, int)
                        and resolved in graph.G
                        and (graph.G.nodes[resolved].get("attributes") or {}).get(
                            "first_class_function_ref"
                        ) is not None
                    ):
                        role = f"kw:{keyword.arg}"
                        if not graph.G.has_edge(resolved, node_id):
                            graph.G.add_edge(resolved, node_id, role=role)
                        children = graph.G.nodes[resolved].setdefault(
                            "children", []
                        )
                        if (node_id, role) not in children:
                            children.append((node_id, role))
                        parents = graph.G.nodes[node_id].setdefault(
                            "parents", []
                        )
                        if (resolved, role) not in parents:
                            parents.append((resolved, role))
                if static_arguments:
                    attributes["static_call_arguments"] = static_arguments

            # A source-defined function may itself be a positional/keyword
            # value (callbacks, policies, hooks).  Its original Name node was
            # intentionally removed, so reconnect the compiler-owned static
            # reference to the call with the argument's real role.
            if node_id in graph.G:
                attributes = graph.G.nodes[node_id].setdefault(
                    "attributes", {}
                )
                static_arguments = dict(
                    attributes.get("static_call_arguments") or {}
                )
                for role, argument in (
                    *tuple(
                        (f"arg:{index}", argument)
                        for index, argument in enumerate(expression.args)
                    ),
                    *tuple(
                        (f"kw:{keyword.arg}", keyword.value)
                        for keyword in expression.keywords
                        if keyword.arg is not None
                    ),
                ):
                    if (
                        isinstance(argument, ast.Name)
                        and isinstance(
                            getattr(builtins, argument.id, None), type
                        )
                    ):
                        static_arguments[role] = argument.id
                        attributes.setdefault(
                            "static_call_values", {}
                        )[role] = getattr(builtins, argument.id)
                        graph.G.graph.setdefault(
                            "static_python_values", {}
                        )[argument.id] = getattr(builtins, argument.id)
                if static_arguments:
                    attributes["static_call_arguments"] = static_arguments
                for role, argument in (
                    *tuple(
                        (f"arg:{index}", argument)
                        for index, argument in enumerate(expression.args)
                    ),
                    *tuple(
                        (f"kw:{keyword.arg}", keyword.value)
                        for keyword in expression.keywords
                        if keyword.arg is not None
                    ),
                ):
                    resolved = resolve_expression(argument)
                    if not (
                        isinstance(resolved, int)
                        and resolved in graph.G
                        and (graph.G.nodes[resolved].get("attributes") or {}).get(
                            "first_class_function_ref"
                        ) is not None
                    ):
                        continue
                    if not graph.G.has_edge(resolved, node_id):
                        graph.G.add_edge(resolved, node_id, role=role)
                    children = graph.G.nodes[resolved].setdefault(
                        "children", []
                    )
                    if (node_id, role) not in children:
                        children.append((node_id, role))
                    parents = graph.G.nodes[node_id].setdefault(
                        "parents", []
                    )
                    if (resolved, role) not in parents:
                        parents.append((resolved, role))

        # A named callee already represented by a function-table reference is
        # not a runtime value.  Its arguments still are.
        children = tuple(source_child_nodes(expression))
        if isinstance(expression, ast.Call):
            call_data = (
                graph.G.nodes[id(expression)]
                if id(expression) in graph.G
                else {}
            )
            call_attributes = call_data.get("attributes") or {}
            if (
                isinstance(expression.func, ast.Name)
                and (
                    "callee_ref" in call_attributes
                    or "external_callee_ref" in call_attributes
                )
            ):
                children = tuple(
                    child
                    for child in children
                    if child is not expression.func
                )
            elif isinstance(callee, _StaticPythonReference):
                children = tuple(
                    child
                    for child in children
                    if child is not expression.func
                )
            elif (
                isinstance(callee, int)
                and callee in graph.G
                and (
                    graph.G.nodes[callee].get("attributes") or {}
                ).get("function_ref") is not None
            ):
                children = tuple(
                    child
                    for child in children
                    if child is not expression.func
                )

        # Resolve children first so the existing executable node receives
        # producer IDs rather than lexical occurrence IDs.
        for child in children:
            if isinstance(child, ast.expr_context):
                continue
            resolve_expression(child)
        node_id = id(expression)
        if (
            isinstance(expression, (ast.Tuple, ast.List, ast.Set, ast.Dict))
            and node_id in graph.G
        ):
            aggregate = graph.G.nodes[node_id]
            aggregate_attributes = aggregate.setdefault("attributes", {})
            aggregate_attributes["producer_kind"] = "aggregate"
            aggregate_attributes["aggregate_leaf_value_ids"] = tuple(
                int(parent)
                for parent, role in aggregate.get("parents") or ()
                if str(role) in {
                    "elts",
                    "elt",
                    "element",
                    "item",
                    "keys",
                    "values",
                    "key",
                    "value",
                }
            )
        if (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
            and expression.func.id in {"tuple", "list"}
            and len(expression.args) == 1
            and node_id in graph.G
        ):
            materializer = graph.G.nodes[node_id]
            materializer_attributes = materializer.setdefault(
                "attributes",
                {},
            )
            materializer_attributes["producer_kind"] = (
                "loop_materialization"
                if isinstance(expression.args[0], ast.GeneratorExp)
                else "aggregate_materialization"
            )
            materializer_attributes["materialization_axis"] = 0
            materializer_attributes["materialized_source_value_ids"] = tuple(
                int(parent)
                for parent, role in materializer.get("parents") or ()
                if str(role).startswith("arg:")
            )
        if (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
            and expression.func.id in {"tuple", "list"}
            and len(expression.args) == 1
            and isinstance(expression.args[0], ast.GeneratorExp)
            and node_id in graph.G
        ):
            # This re-resolves generator_expression.elt, which the
            # (ListComp, SetComp, DictComp, GeneratorExp) branch above
            # already resolved once, inside its own scoped push/pop of the
            # generator's `for` target names. resolve_expression() is not
            # memoized for Name lookups -- it re-reads `environment` on
            # every call -- so this second, unscoped pass through the same
            # `elt` subtree re-looks-up the target name(s) (e.g. `address`
            # in `tuple(f(address) for address in xs)`) after that first
            # pass already popped them, silently creating a fresh
            # "external input" binding for what should be, and already
            # was, a properly loop-scoped local. That stray rebinding then
            # leaked into whatever unrelated code came later in the
            # function and happened to reuse the same name. Same
            # push/pop discipline as above, applied to this second walk.
            generator_expression = expression.args[0]
            comprehension_target_names = {
                name
                for generator in generator_expression.generators
                for name in loop_target_names(generator.target)
            }
            shadowed_bindings = {
                name: environment[name]
                for name in comprehension_target_names
                if name in environment
            }
            for generator in generator_expression.generators:
                bind_loop_target(generator.target)
            value_id = resolve_expression(generator_expression.elt)
            for generator in generator_expression.generators:
                generator_id = id(generator)
                if (
                    generator_id not in graph.G
                    or not isinstance(value_id, int)
                    or value_id not in graph.G
                ):
                    continue
                attributes = graph.G.nodes[generator_id].setdefault(
                    "attributes",
                    {},
                )
                outputs = list(
                    attributes.get("loop_iteration_outputs", ())
                )
                outputs.append({
                    "value_id": value_id,
                    "result_value_id": node_id,
                    "materializer_node_id": node_id,
                })
                attributes["loop_iteration_outputs"] = tuple(outputs)
            for name in comprehension_target_names:
                if name in shadowed_bindings:
                    environment[name] = shadowed_bindings[name]
                else:
                    environment.pop(name, None)
        return node_id if node_id in graph.G else None

    def bind_target(
        target: ast.AST,
        value: int | _StaticPythonReference | None,
    ) -> None:
        if value is None:
            return
        if isinstance(target, ast.Name):
            deleted_names.discard(target.id)
            if isinstance(value, _StaticPythonReference):
                environment.pop(target.id, None)
                static_environment[target.id] = value
                _remove_node(graph, id(target))
                return
            static_environment.pop(target.id, None)
            environment[target.id] = value
            identity_bindings.setdefault(target.id, []).append(value)
            _remove_node(graph, id(target))
            return
        if isinstance(target, ast.Attribute):
            if isinstance(value, _StaticPythonReference):
                raise TypeError(
                    "a static Python reference cannot be stored as a runtime "
                    f"object attribute in {statement.name}: "
                    f"target={ast.dump(target, include_attributes=False)}, "
                    f"value={value.path}"
                )
            receiver = resolve_expression(target.value)
            static_receiver = (
                receiver if isinstance(receiver, _StaticPythonReference) else None
            )
            if static_receiver is not None:
                receiver = static_reference_node(static_receiver)
            if not isinstance(receiver, int) or not isinstance(value, int):
                raise TypeError(
                    "attribute assignment requires resolved object and value "
                    f"nodes in {statement.name}: "
                    f"target={ast.dump(target, include_attributes=False)}, "
                    f"receiver={receiver!r}, value={value!r}"
                )
            node_id = id(target)
            if node_id not in graph.G:
                # Resolving a compile-time receiver's literal attribute read
                # redirects and removes the source Attribute wrapper.  The
                # subsequent write is still a distinct runtime effect.
                node_id = new_node(
                    "SetAttr",
                    f"setattr[{target.attr}]",
                    attributes={"attribute": target.attr},
                )
            # In ``object.field += value`` the Attribute node is the read
            # feeding the AugAssign result.  Reusing that same node as the
            # write would create ``Attribute -> AugAssign -> SetAttr`` while
            # SetAttr is still the Attribute node: an artificial dataflow
            # cycle.  Keep the read and write as distinct program events.
            if node_id == value or nx.has_path(graph.G, node_id, value):
                node_id = new_node(
                    "SetAttr",
                    f"setattr[{target.attr}]",
                    attributes={"attribute": target.attr},
                )
            node_data = graph.G.nodes[node_id]
            node_data["type"] = "SetAttr"
            node_data["op"] = "setattr"
            node_data.setdefault("attributes", {})["attribute"] = target.attr
            _replace_inputs(
                graph,
                node_id,
                (
                    (receiver, "object"),
                    (value, "value"),
                ),
            )
            if static_receiver is not None:
                static_attribute_values[
                    (id(static_receiver.value), target.attr)
                ] = value
            # A plain-named receiver (``counter.value = ...``) gets its own
            # identity binding the same way a bare ``ast.Name`` target does
            # above -- the field write is already a real, correctly wired
            # SetAttr node; it was just never named, so it never reached
            # ``identity_table`` for anything downstream to select by name.
            if isinstance(target.value, ast.Name):
                identity_bindings.setdefault(
                    f"{target.value.id}.{target.attr}", []
                ).append(node_id)
            return
        if isinstance(target, ast.Subscript):
            if isinstance(value, _StaticPythonReference):
                raise TypeError(
                    "a static Python reference cannot be assigned through "
                    "a runtime tensor index"
                )
            base = resolve_expression(target.value)
            index_expressions = (
                tuple(target.slice.elts)
                if isinstance(target.slice, ast.Tuple)
                else (target.slice,)
            )
            indices = tuple(
                resolve_expression(index) for index in index_expressions
            )
            if not isinstance(base, int) or not isinstance(value, int) or not all(
                isinstance(index, int) for index in indices
            ):
                raise TypeError(
                    "indexed assignment requires resolved tensor, index, "
                    f"and value nodes in {statement.name}"
                )
            node_id = id(target)
            if (
                node_id not in graph.G
                or node_id == base
                or node_id == value
                or nx.has_path(graph.G, node_id, value)
            ):
                node_id = new_node("IndexedStore", "indexed_store")
            node_data = graph.G.nodes[node_id]
            node_data["type"] = "IndexedStore"
            node_data["op"] = "IndexedStore"
            node_data.setdefault("attributes", {})[
                "source_type"
            ] = "SubscriptStore"
            _replace_inputs(
                graph,
                node_id,
                (
                    (base, "base"),
                    *((index, "index") for index in indices),
                    (value, "value"),
                ),
            )
            if isinstance(target.value, ast.Name):
                bind_target(target.value, node_id)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            if isinstance(value, _StaticPythonReference):
                raise TypeError(
                    "a static Python reference cannot be destructured as "
                    "a runtime graph value"
                )
            for index, element in enumerate(target.elts):
                index_id = new_node(
                    "Constant",
                    str(index),
                    attributes={"value": index},
                )
                projected_id = new_node(
                    "Indexed",
                    f"unpack[{index}]",
                    parents=(
                        (value, "base"),
                        (index_id, "index"),
                    ),
                )
                bind_target(element, projected_id)

    def delete_target(target: ast.AST) -> int | None:
        """Lower one Python deletion target to its explicit state effect."""

        if isinstance(target, ast.Name):
            environment.pop(target.id, None)
            static_environment.pop(target.id, None)
            deleted_names.add(target.id)
            _remove_node(graph, id(target))
            return None
        if isinstance(target, ast.Attribute):
            receiver = resolve_expression(target.value)
            if isinstance(receiver, _StaticPythonReference):
                receiver = static_reference_node(receiver)
            if not isinstance(receiver, int):
                raise TypeError(
                    "attribute deletion requires a resolved object node in "
                    f"{statement.name}: "
                    f"target={ast.dump(target, include_attributes=False)}, "
                    f"receiver={receiver!r}"
                )
            node_id = id(target)
            if node_id not in graph.G:
                node_id = new_node(
                    "DelAttr",
                    f"delattr[{target.attr}]",
                    attributes={"attribute": target.attr},
                )
            node_data = graph.G.nodes[node_id]
            node_data["type"] = "DelAttr"
            node_data["op"] = "delattr"
            node_data.setdefault("attributes", {})["attribute"] = target.attr
            _replace_inputs(graph, node_id, ((receiver, "object"),))
            return node_id
        if isinstance(target, ast.Subscript):
            base = resolve_expression(target.value)
            indices = (
                tuple(target.slice.elts)
                if isinstance(target.slice, ast.Tuple)
                else (target.slice,)
            )
            resolved_indices = tuple(resolve_expression(index) for index in indices)
            if not isinstance(base, int) or not all(
                isinstance(index, int) for index in resolved_indices
            ):
                raise TypeError(
                    "indexed deletion requires resolved container and index "
                    f"nodes in {statement.name}: "
                    f"target={ast.dump(target, include_attributes=False)}"
                )
            node_id = id(target)
            if node_id not in graph.G:
                node_id = new_node("DelItem", "delitem")
            node_data = graph.G.nodes[node_id]
            node_data["type"] = "DelItem"
            node_data["op"] = "delitem"
            node_data.setdefault("attributes", {})["source_type"] = "Subscript"
            _replace_inputs(
                graph,
                node_id,
                (
                    (base, "base"),
                    *((index, "index") for index in resolved_indices),
                ),
            )
            return node_id
        if isinstance(target, (ast.Tuple, ast.List)):
            effects = [delete_target(element) for element in target.elts]
            return next((effect for effect in reversed(effects) if effect is not None), None)
        raise TypeError(
            "unsupported Python deletion target: "
            f"{ast.dump(target, include_attributes=False)}"
        )

    def reduce_statement(body_statement: ast.stmt) -> int | None:
        if isinstance(body_statement, (ast.Nonlocal, ast.Global)):
            # Scope declarations affect name binding while parsing; they are
            # not runtime operations and must never reach an execution
            # backend as operator nodes.  The reducer's lexical environments
            # and explicit Input/SetAttr effects carry the resulting values.
            _remove_node(graph, id(body_statement))
            return None
        if isinstance(body_statement, ast.Delete):
            effects = [delete_target(target) for target in body_statement.targets]
            _remove_node(graph, id(body_statement))
            return next(
                (effect for effect in reversed(effects) if effect is not None),
                None,
            )
        if isinstance(body_statement, (ast.Assign, ast.AnnAssign)):
            value = resolve_expression(body_statement.value)
            targets = (
                tuple(body_statement.targets)
                if isinstance(body_statement, ast.Assign)
                else (body_statement.target,)
            )
            for target in targets:
                bind_target(target, value)
            _remove_node(graph, id(body_statement))
            return value
        if isinstance(body_statement, ast.AugAssign):
            arena_mask = None
            if isinstance(body_statement.target, ast.Name):
                current = environment.get(body_statement.target.id)
                if current is None:
                    current = input_value(
                        body_statement.target.id,
                        binding_kind=(
                            "parameter"
                            if body_statement.target.id in parameter_names
                            else "exception"
                            if body_statement.target.id in exception_local_names
                            else "external"
                        ),
                    )
            elif (
                isinstance(body_statement.target, ast.Subscript)
                and not all(
                    isinstance(item, ast.Slice)
                    or (
                        isinstance(item, ast.Constant)
                        and (
                            isinstance(item.value, int)
                            or item.value is Ellipsis
                        )
                    )
                    or (
                        isinstance(item, ast.Name)
                        and environment.get(item.id)
                        in scalar_loop_binding_ids
                    )
                    for item in (
                        tuple(body_statement.target.slice.elts)
                        if isinstance(body_statement.target.slice, ast.Tuple)
                        else (body_statement.target.slice,)
                    )
                )
            ):
                # Keep aligned boolean-mask updates arena-shaped.  Evaluating
                # ``field[mask] += rhs[mask]`` literally creates two compact,
                # data-dependent vectors and then requires a scatter.  The
                # equivalent functional form
                # ``field = where(mask, field + rhs, field)`` retains the
                # preallocated grid shape through every compiler stage.
                current = resolve_expression(body_statement.target.value)
                arena_mask = resolve_expression(
                    body_statement.target.slice
                )
                target_slice = ast.dump(
                    body_statement.target.slice,
                    include_attributes=False,
                )
                for candidate in source_walk(body_statement.value):
                    if not isinstance(candidate, ast.Subscript):
                        continue
                    if ast.dump(
                        candidate.slice,
                        include_attributes=False,
                    ) != target_slice:
                        continue
                    aligned = resolve_expression(candidate.value)
                    if isinstance(aligned, int):
                        _redirect_value(graph, id(candidate), aligned)
            else:
                current = resolve_expression(body_statement.target)
            if current is not None:
                _redirect_value(
                    graph,
                    id(body_statement.target),
                    current,
                )
            resolve_expression(body_statement.value)
            node_id = id(body_statement)
            if current is None or node_id not in graph.G:
                return None
            if isinstance(arena_mask, int):
                where_id = new_node(
                    "where",
                    "where",
                    attributes={
                        "source_type": "MaskedAugAssign",
                        "arena_shaped": True,
                    },
                    parents=(
                        (arena_mask, "condition"),
                        (node_id, "true"),
                        (current, "false"),
                    ),
                )
                bind_target(body_statement.target.value, where_id)
                return where_id
            bind_target(body_statement.target, node_id)
            return node_id
        if isinstance(body_statement, ast.Return):
            returned = body_statement.value
            expressions = (
                tuple(returned.elts)
                if isinstance(returned, (ast.Tuple, ast.List))
                else (returned,)
            )
            output_names = tuple(
                graph.G.graph.get("function_outputs", ())
            )
            resolved = []
            for index, expression in enumerate(expressions):
                value = resolve_expression(expression)
                if value is None:
                    continue
                resolved.append(value)
                if index < len(output_names):
                    identity_bindings.setdefault(
                        str(output_names[index]), []
                    ).append(value)
            if len(expressions) == 1:
                value = resolved[0] if resolved else None
                if value is not None:
                    graph.G.graph.setdefault(
                        "return_value_nodes",
                        {},
                    )[id(body_statement)] = value
                return value
            # Preserve the structural tuple/list node for callers that consume
            # it as one Python-shaped value while the output identities above
            # expose each semantic result directly to compiled call binding.
            value = resolve_expression(returned)
            if value is not None:
                graph.G.graph.setdefault(
                    "return_value_nodes",
                    {},
                )[id(body_statement)] = value
            return value
        if isinstance(body_statement, (ast.With, ast.AsyncWith)):
            static_contexts = []
            for item in body_statement.items:
                context_value = resolve_expression(item.context_expr)
                if context_value is not None and context_value in graph.G:
                    context_data = graph.G.nodes[context_value]
                    reference = (
                        context_data.get("attributes") or {}
                    ).get("static_python_reference")
                    if reference == "autograd.no_grad":
                        static_contexts.append(
                            {
                                "reference": reference,
                                "effect": "disable_backward_recording",
                                "lineno": getattr(
                                    item.context_expr,
                                    "lineno",
                                    None,
                                ),
                                "end_lineno": getattr(
                                    item.context_expr,
                                    "end_lineno",
                                    None,
                                ),
                            }
                        )
                        _remove_node(graph, context_value)
                resolve_expression(item.optional_vars)
            result = None
            for nested in body_statement.body:
                result = reduce_statement(nested)
            if static_contexts:
                recorded = list(
                    graph.G.graph.get("static_contexts", ())
                )
                recorded.extend(static_contexts)
                graph.G.graph["static_contexts"] = tuple(recorded)
                for item in body_statement.items:
                    _remove_node(graph, id(item))
                _remove_node(graph, id(body_statement))
            return result
        if isinstance(body_statement, ast.If):
            test_value = resolve_expression(body_statement.test)
            # Control-flow value merging remains a planner responsibility.
            # Reduce lexical occurrences within each arm without pretending
            # that either arm executed unconditionally.
            before = dict(environment)
            body_environment = dict(before)
            environment.clear()
            environment.update(body_environment)
            body_result = None
            for nested in body_statement.body:
                body_result = reduce_statement(nested)
            body_environment = dict(environment)
            environment.clear()
            environment.update(before)
            else_result = None
            for nested in body_statement.orelse:
                else_result = reduce_statement(nested)
            else_environment = dict(environment)
            environment.clear()
            environment.update(before)
            for name in set(before) | set(body_environment) | set(
                else_environment
            ):
                body_value = body_environment.get(name, before.get(name))
                else_value = else_environment.get(name, before.get(name))
                if body_value == else_value:
                    if body_value is not None:
                        environment[name] = body_value
                    continue
                if (
                    isinstance(test_value, int)
                    and isinstance(body_value, int)
                    and isinstance(else_value, int)
                ):
                    environment[name] = new_node(
                        "Phi",
                        name,
                        attributes={"binding_name": name},
                        parents=(
                            (test_value, "test"),
                            (body_value, "body"),
                            (else_value, "orelse"),
                        ),
                    )
            def terminal_branch(statements: list[ast.stmt]) -> bool:
                if not statements:
                    return False
                terminal = statements[-1]
                return isinstance(terminal, (ast.Return, ast.Raise)) or (
                    isinstance(terminal, ast.If)
                    and id(terminal) in graph.G
                    and bool(
                        (graph.G.nodes[id(terminal)].get("attributes") or {}).get(
                            "terminal_return_merge"
                        )
                    )
                )

            if (
                isinstance(test_value, int)
                and isinstance(body_result, int)
                and isinstance(else_result, int)
                and terminal_branch(body_statement.body)
                and terminal_branch(body_statement.orelse)
                and id(body_statement) in graph.G
            ):
                _replace_inputs(
                    graph,
                    id(body_statement),
                    (
                        (test_value, "test"),
                        (body_result, "body"),
                        (else_result, "orelse"),
                    ),
                )
                graph.G.nodes[id(body_statement)].setdefault(
                    "attributes", {}
                ).update({
                    "terminal_return_merge": True,
                    "terminal_return_values": (
                        int(body_result),
                        int(else_result),
                    ),
                })
            return id(body_statement)
        if isinstance(body_statement, ast.Try):
            before = dict(environment)
            for nested in body_statement.body:
                reduce_statement(nested)
            body_environment = dict(environment)

            handler_environments = []
            for handler in body_statement.handlers:
                environment.clear()
                environment.update(before)
                if handler.name:
                    input_value(
                        handler.name,
                        binding_kind="exception",
                    )
                for nested in handler.body:
                    reduce_statement(nested)
                handler_environments.append(dict(environment))

            continuing_handler_environments = [
                candidate
                for handler, candidate in zip(
                    body_statement.handlers,
                    handler_environments,
                )
                if not (
                    handler.body
                    and isinstance(
                        handler.body[-1],
                        (ast.Raise, ast.Return),
                    )
                )
            ]
            # Only control-flow paths that can reach the statement following
            # the try participate in its lexical-value merge.  A handler
            # ending in raise/return has no continuation edge and therefore
            # cannot turn values assigned by the successful body into
            # invented external inputs.
            environments = [
                body_environment,
                *continuing_handler_environments,
            ]
            environment.clear()
            if environments:
                common_names = set.intersection(
                    *(set(candidate) for candidate in environments)
                )
                for name in common_names:
                    values = {
                        candidate[name] for candidate in environments
                    }
                    if len(values) == 1:
                        environment[name] = values.pop()
            for nested in body_statement.orelse:
                reduce_statement(nested)
            for nested in body_statement.finalbody:
                reduce_statement(nested)
            return id(body_statement)
        if isinstance(body_statement, (ast.For, ast.While)):
            before_loop = dict(environment)
            # A discarded bound-method result is an effectful body statement
            # unless a later lowering proves otherwise.  Record the actual
            # statement calls here; loop composition must not rediscover
            # state transitions from method-name lists or output reachability.
            state_effect_calls = tuple(
                expression_statement.value
                for nested_statement in body_statement.body
                for expression_statement in source_walk(nested_statement)
                if (
                    isinstance(expression_statement, ast.Expr)
                    and isinstance(expression_statement.value, ast.Call)
                    and isinstance(
                        expression_statement.value.func,
                        ast.Attribute,
                    )
                    and isinstance(
                        expression_statement.value.func.value,
                        ast.Name,
                    )
                )
            )
            if isinstance(body_statement, ast.For):
                resolve_expression(body_statement.iter)
                bind_loop_target(body_statement.target)
            else:
                resolve_expression(body_statement.test)
            for nested in body_statement.body:
                reduce_statement(nested)
            for nested in body_statement.orelse:
                reduce_statement(nested)
            loop_id = id(body_statement)
            if loop_id in graph.G:
                body_member_ids = {
                    id(member)
                    for nested in body_statement.body
                    for member in source_walk(nested)
                }
                direct_loop_target_names = (
                    set(loop_target_names(body_statement.target))
                    if isinstance(body_statement, ast.For)
                    else set()
                )
                current_loop_bindings = {
                    name: value_id
                    for name, value_id in environment.items()
                    if (
                        name in direct_loop_target_names
                        and
                        (
                            graph.G.nodes[value_id].get("attributes") or {}
                        ).get("binding_kind") == "loop"
                        and before_loop.get(name) != value_id
                    )
                }
                # AST ingestion initially wires name occurrences before
                # lexical rebinding is known.  A reused spelling (for example
                # a comprehension's ``width`` followed by a ``for width``)
                # must not leave the second loop body attached to the first
                # binding.  Rewrite only nodes lexically owned by this body.
                for member_id in body_member_ids:
                    if member_id not in graph.G:
                        continue
                    parents = list(
                        graph.G.nodes[member_id].get("parents") or ()
                    )
                    replacements = {
                        old: current_loop_bindings[name]
                        for name in current_loop_bindings
                        for old in identity_bindings.get(name, ())
                        if old != current_loop_bindings[name]
                    }
                    rewritten = [
                        (replacements.get(parent, parent), role)
                        for parent, role in parents
                    ]
                    if rewritten != parents:
                        _replace_inputs(
                            graph,
                            member_id,
                            tuple(rewritten),
                        )
                loop_attributes = graph.G.nodes[loop_id].setdefault(
                    "attributes",
                    {},
                )
                loop_target_bindings = current_loop_bindings
                loop_carried_bindings = {
                    name: (before_loop[name], environment[name])
                    for name in before_loop.keys() & environment.keys()
                    if (
                        before_loop[name] != environment[name]
                        and name not in loop_target_bindings
                    )
                }
                loop_attributes["loop_carried_bindings"] = (
                    loop_carried_bindings
                )
                loop_attributes["loop_target_bindings"] = (
                    loop_target_bindings
                )
                loop_attributes["loop_target_initials"] = {
                    name: before_loop[name]
                    for name in loop_target_bindings
                    if name in before_loop
                }
                # This pass resolves source/value identities only.  It records
                # the body value selected by the lexical continuation, but it
                # must not manufacture a loop latch, exit, collection owner,
                # or backend schedule.  The post-canonical loop reducer will
                # either thread these values through straight-line unrolled
                # SSA or create retained-loop result ports.
                for name, (_initial, updated) in (
                    loop_carried_bindings.items()
                ):
                    environment[name] = updated
                    identity_bindings.setdefault(name, []).append(updated)
                # A discarded bound-method call is retained here only as a
                # source effect fact.  Its Python return value is not the
                # mutated state, and no synthetic state transition belongs in
                # the critical value graph before loop realization is known.
                state_effects = []
                for call in state_effect_calls:
                    name = call.func.value.id
                    initial = before_loop.get(name)
                    call_id = id(call)
                    if (
                        initial is None
                        or initial not in graph.G
                        or call_id not in graph.G
                        or name in loop_carried_bindings
                    ):
                        continue
                    call_parents = tuple(
                        graph.G.nodes[call_id].get("parents") or ()
                    )
                    argument_ids = tuple(
                        int(parent)
                        for parent, role in call_parents
                        if str(role).startswith("arg")
                    )
                    environment[name] = initial
                    state_effects.append({
                        "state_name": name,
                        "operator": call.func.attr,
                        "effect_mode": (
                            "indexed_publication"
                            if (
                                call.func.attr == "append"
                                and len(argument_ids) == 1
                            )
                            else "opaque"
                        ),
                        "state_input_id": int(initial),
                        "effect_node_id": int(call_id),
                        "argument_value_ids": argument_ids,
                    })
                if state_effects:
                    loop_attributes["loop_state_effects"] = tuple(
                        state_effects
                    )
            return id(body_statement)
        if isinstance(body_statement, ast.Expr):
            return resolve_expression(body_statement.value)
        for child in source_child_nodes(body_statement):
            if isinstance(child, ast.expr):
                resolve_expression(child)
        return id(body_statement) if id(body_statement) in graph.G else None

    returned_values = []
    for body_statement in statement.body:
        value = reduce_statement(body_statement)
        if isinstance(body_statement, ast.Return) and value is not None:
            returned_values.append(value)
    if returned_values:
        graph.roots = list(dict.fromkeys(returned_values))

    # A source-linked unbound method can arrive with its receiver absent from
    # the owned subgraph even though the Attribute load itself is owned.
    # Restore the ordinary edge before lexical Name cleanup; an Attribute
    # without its receiver is not a valid structural operation. The receiver
    # can be a parameter (``self`` bound directly at the call boundary) or an
    # ordinary local (``machine = load_pe(...)``) -- both are recorded under
    # the same name in ``environment`` by ``bind_target``, so there is no
    # reason to restore only the parameter case and leave a local's receiver
    # permanently unresolved. A chained access (``machine.runner.tick``) puts
    # an Attribute (``machine.runner``), not a Name, in the receiver position
    # of the outer Attribute (``.tick``); that inner Attribute is itself just
    # another node this same loop repairs on its own matching iteration, so
    # linking straight to its node id (when it is still a live node) is
    # enough -- the two repairs do not need to happen in any particular
    # order, since neither removes a node, only reattaches an edge.
    for node_id, data in list(graph.G.nodes(data=True)):
        expression = data.get("expr_obj")
        if not (
            isinstance(expression, ast.Attribute)
            and not data.get("parents")
            and isinstance(expression.value, (ast.Name, ast.Attribute))
        ):
            continue
        if isinstance(expression.value, ast.Name):
            receiver = environment.get(expression.value.id)
            if receiver is None:
                receiver = input_value(
                    expression.value.id,
                    binding_kind=(
                        "parameter"
                        if expression.value.id in parameter_names
                        else "local"
                    ),
                )
        else:
            inner_id = id(expression.value)
            if inner_id not in graph.G:
                continue
            receiver = inner_id
        _replace_inputs(graph, node_id, ((receiver, "value"),))

    # Any surviving lexical occurrence is either unused syntax or an unresolved
    # source label.  It is not executable work in the reduced value graph --
    # but only when nothing still structurally depends on it. A Name node
    # that still has a real successor (an Attribute load whose receiver is
    # this exact node, say) is not unused syntax; removing it anyway does
    # not erase that dependency, it just strips the successor's own record
    # of it, leaving that successor orphaned for whatever later pass expects
    # to find its receiver -- the exact "receiver Name absent" case the
    # repair above this loop exists to patch after the fact. Checking
    # liveness first means there is nothing left to repair.
    for node_id, data in list(graph.G.nodes(data=True)):
        expression = data.get("expr_obj")
        if isinstance(expression, ast.Name):
            if graph.G.out_degree(node_id) > 0:
                continue
            _remove_node(graph, node_id)
            continue
        if (
            data.get("type") == "Input"
            and graph.G.out_degree(node_id) == 0
            and node_id not in graph.roots
            and (data.get("attributes") or {}).get("binding_kind")
            == "external"
        ):
            # A named Call target initially looks like a lexical external.
            # Once the Call owns a function-table reference, that temporary
            # value has no consumers and must not leak into the public input
            # contract.
            _remove_node(graph, node_id)

    # Stable topological relabeling turns opaque Python object identities into
    # compact monotonic value IDs without changing the faithfully captured AST.
    invalid_node_ids = [
        node_id for node_id in graph.G if type(node_id) is not int
    ]
    invalid_parent_ids = [
        parent_id
        for _node_id, data in graph.G.nodes(data=True)
        for parent_id, _role in data.get("parents", ())
        if type(parent_id) is not int
    ]
    invalid_roots = [
        root for root in graph.roots if type(root) is not int
    ]
    assert not invalid_node_ids, (
        "compile-time references must be represented by integer-keyed "
        f"StaticReference nodes, not graph keys: {invalid_node_ids!r}"
    )
    assert not invalid_parent_ids, (
        "compile-time references must not appear as runtime parent IDs: "
        f"{invalid_parent_ids!r}"
    )
    assert not invalid_roots, (
        "compile-time references must not appear as graph roots: "
        f"{invalid_roots!r}"
    )
    source_position = {
        node_id: (
            getattr(data.get("expr_obj"), "lineno", -1),
            getattr(data.get("expr_obj"), "col_offset", -1),
            str(data.get("type", "")),
            int(node_id),
        )
        for node_id, data in graph.G.nodes(data=True)
    }
    ordered = list(
        nx.lexicographical_topological_sort(
            graph.G,
            key=lambda node_id: source_position[node_id],
        )
    )
    mapping = {
        node_id: value_id for value_id, node_id in enumerate(ordered)
    }
    relabeled = nx.relabel_nodes(graph.G, mapping, copy=True)
    ordered_graph = nx.DiGraph()
    ordered_graph.graph.update(relabeled.graph)
    ordered_graph.graph["canonical_value_ids"] = True
    map_ir = dict(ordered_graph.graph.get("map_ir") or {})
    map_ir["schema_node_ids"] = tuple(
        mapping[node_id]
        for node_id in map_ir.get("schema_node_ids", ())
        if node_id in mapping
    )
    map_ir["schema_roots"] = tuple(
        mapping[node_id]
        for node_id in map_ir.get("schema_roots", ())
        if node_id in mapping
    )
    ordered_graph.graph["map_ir"] = map_ir
    for value_id in range(len(mapping)):
        ordered_graph.add_node(value_id, **relabeled.nodes[value_id])
    ordered_graph.add_edges_from(relabeled.edges(data=True))
    graph.G = ordered_graph
    graph.roots = [
        mapping[root] for root in graph.roots if root in mapping
    ]
    graph.levels = {
        mapping[node_id]: level
        for node_id, level in graph.levels.items()
        if node_id in mapping
    }
    graph.G.graph["identity_table"] = {
        name: tuple(
            mapping[value_id]
            for value_id in value_ids
            if value_id in mapping
        )
        for name, value_ids in identity_bindings.items()
    }
    for value_id, data in graph.G.nodes(data=True):
        data["value_id"] = value_id
        attributes = data.get("attributes") or {}
        if "loop_carried_bindings" in attributes:
            attributes["loop_carried_bindings"] = {
                name: (mapping[initial], mapping[updated])
                for name, (initial, updated) in attributes[
                    "loop_carried_bindings"
                ].items()
                if initial in mapping and updated in mapping
            }
        if "loop_target_bindings" in attributes:
            attributes["loop_target_bindings"] = {
                name: mapping[target]
                for name, target in attributes[
                    "loop_target_bindings"
                ].items()
                if target in mapping
            }
        if "loop_target_initials" in attributes:
            attributes["loop_target_initials"] = {
                name: mapping[initial]
                for name, initial in attributes[
                    "loop_target_initials"
                ].items()
                if initial in mapping
            }
        if "terminal_return_values" in attributes:
            attributes["terminal_return_values"] = tuple(
                mapping[value_id]
                for value_id in attributes["terminal_return_values"]
                if value_id in mapping
            )
        if "loop_state_effects" in attributes:
            attributes["loop_state_effects"] = tuple(
                {
                    **effect,
                    "state_input_id": mapping[effect["state_input_id"]],
                    "effect_node_id": mapping[effect["effect_node_id"]],
                    "argument_value_ids": tuple(
                        mapping[value_id]
                        for value_id in effect["argument_value_ids"]
                        if value_id in mapping
                    ),
                }
                for effect in attributes["loop_state_effects"]
                if all(
                    effect[key] in mapping
                    for key in (
                        "state_input_id",
                        "effect_node_id",
                    )
                )
            )
        if "loop_iteration_outputs" in attributes:
            attributes["loop_iteration_outputs"] = tuple(
                {
                    key: mapping[output[key]]
                    for key in (
                        "value_id",
                        "result_value_id",
                        "materializer_node_id",
                    )
                }
                for output in attributes["loop_iteration_outputs"]
                if all(
                    output[key] in mapping
                    for key in (
                        "value_id",
                        "result_value_id",
                        "materializer_node_id",
                    )
                )
            )
        for key in (
            "aggregate_leaf_value_ids",
            "materialized_source_value_ids",
        ):
            if key in attributes:
                attributes[key] = tuple(
                    mapping[value_id]
                    for value_id in attributes[key]
                    if value_id in mapping
                )
        data["parents"] = [
            (mapping[parent_id], role)
            for parent_id, role in data.get("parents", ())
            if parent_id in mapping
        ]
        data["children"] = [
            (mapping[child_id], role)
            for child_id, role in data.get("children", ())
            if child_id in mapping
        ]

    terminal_merges = [
        int(node_id)
        for node_id, data in graph.G.nodes(data=True)
        if (data.get("attributes") or {}).get("terminal_return_merge")
    ]
    for merge_id in terminal_merges:
        branch_values = set(
            (graph.G.nodes[merge_id].get("attributes") or {}).get(
                "terminal_return_values", ()
            )
        )
        if branch_values and branch_values.issubset(graph.roots):
            graph.roots = [
                root for root in graph.roots if root not in branch_values
            ]
            graph.roots.append(merge_id)


def reduce_abstract_tensor_topology(graph: Any) -> Any:
    """Apply existing ProcessGraph names to the three structural AST nodes."""

    function_table = getattr(graph, "function_table", None)
    if function_table is None:
        function_table = FunctionTable()
        graph.function_table = function_table
    external_function_table = getattr(
        graph,
        "external_function_table",
        None,
    )
    if external_function_table is None:
        external_function_table = ExternalFunctionTable()
        graph.external_function_table = external_function_table

    function_nodes: dict[int, Any] = {}
    function_return_values: dict[int, list[int]] = {}
    call_owners: dict[int, Any] = {}
    method_owners: dict[int, str] = {}
    class_definitions: dict[str, ast.ClassDef] = {}
    lexical_parent_by_function: dict[int, int] = {}
    function_definitions = {
        int(node_id): node_data.get("expr_obj")
        for node_id, node_data in graph.G.nodes(data=True)
        if is_runnable_definition(node_data.get("expr_obj"))
    }

    class _DirectNestedFunctionVisitor(ast.NodeVisitor):
        def __init__(self, owner_id: int):
            self.owner_id = int(owner_id)

        def visit_FunctionDef(self, node):
            lexical_parent_by_function[id(node)] = self.owner_id

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Lambda(self, node):
            lexical_parent_by_function[id(node)] = self.owner_id

        def visit_ClassDef(self, node):
            return None

    for owner_id, definition in function_definitions.items():
        if not isinstance(
            definition, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        visitor = _DirectNestedFunctionVisitor(owner_id)
        for body_member in definition.body:
            visitor.visit(body_member)
    for _node_id, node_data in graph.G.nodes(data=True):
        class_definition = node_data.get("expr_obj")
        if isinstance(
            class_definition,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            source_identity = getattr(
                class_definition,
                "_python_source_identity",
                None,
            )
            qualified = (
                str(source_identity[1])
                if (
                    isinstance(source_identity, tuple)
                    and len(source_identity) == 2
                )
                else ""
            )
            parts = tuple(
                part for part in qualified.split(".") if part != "<locals>"
            )
            if "<locals>" not in qualified and len(parts) >= 2:
                method_owners[id(class_definition)] = parts[-2]
        if not isinstance(class_definition, ast.ClassDef):
            continue
        class_definitions[class_definition.name] = class_definition
        for member in class_definition.body:
            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_owners[id(member)] = class_definition.name
    for node_id, data in graph.G.nodes(data=True):
        statement = data.get("expr_obj")
        if not is_runnable_definition(statement):
            continue
        function_name = runnable_definition_name(statement)
        owner_name = method_owners.get(node_id)

        def lexical_qualified_name(definition_id: int) -> str:
            definition = function_definitions[definition_id]
            identity = getattr(definition, "_python_source_identity", None)
            if isinstance(identity, tuple) and len(identity) == 2:
                module_name, python_qualified = map(str, identity)
                return ".".join(
                    part
                    for part in (module_name, python_qualified)
                    if part
                )
            parent_id = lexical_parent_by_function.get(definition_id)
            if parent_id is not None:
                return (
                    f"{lexical_qualified_name(parent_id)}.<locals>."
                    f"{getattr(definition, 'name', function_name)}"
                )
            method_owner = method_owners.get(definition_id)
            return (
                f"{method_owner}.{getattr(definition, 'name', function_name)}"
                if method_owner is not None
                else str(getattr(definition, "name", function_name))
            )

        qualified_name = lexical_qualified_name(node_id)
        reference = function_table.declare(
            function_name,
            qualified_name=qualified_name,
            metadata={
                "source_type": type(statement).__name__,
                "source_node": node_id,
                **(
                    {
                        "process_graph_boundary": getattr(
                            statement,
                            "_process_graph_boundary",
                        )
                    }
                    if hasattr(statement, "_process_graph_boundary")
                    else {}
                ),
            },
        )
        boundary_callable = getattr(
            statement,
            "_process_graph_boundary_callable",
            None,
        )
        if boundary_callable is not None:
            function_table.resolve_callable(reference, boundary_callable)
        data.setdefault("attributes", {})[
            "function_ref"
        ] = reference.address
        function_nodes[node_id] = reference
        if isinstance(statement, ast.Lambda):
            # A lambda's body is one expression, not a list of statements,
            # and is itself the implicit return value -- no explicit
            # Return-shaped node exists to walk to. Python-only construct;
            # nothing here needs generalizing for another language.
            function_return_values[node_id] = [id(statement.body)]
        else:
            _record_owned_calls_and_returns(
                graph, reference, node_id, call_owners, function_return_values,
            )

    def class_field_defaults(definition: ast.ClassDef) -> dict[str, Any]:
        """Retain literal class-field defaults as structural compiler facts."""

        def is_literal(value: Any) -> bool:
            if value is None or isinstance(
                value,
                (bool, bytes, complex, float, int, str),
            ):
                return True
            if isinstance(value, (tuple, list)):
                return all(is_literal(item) for item in value)
            if isinstance(value, dict):
                return all(
                    is_literal(key) and is_literal(item)
                    for key, item in value.items()
                )
            return False

        defaults: dict[str, Any] = {}
        for member in definition.body:
            if not (
                isinstance(member, ast.AnnAssign)
                and isinstance(member.target, ast.Name)
                and member.value is not None
            ):
                continue
            try:
                value = ast.literal_eval(member.value)
            except (ValueError, TypeError, SyntaxError):
                continue
            if is_literal(value):
                defaults[member.target.id] = value
        return defaults

    graph.G.graph["class_table"] = {
        class_name: {
            "methods": {
                member.name: function_nodes[id(member)].address
                for member in definition.body
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                and id(member) in function_nodes
            },
            "fields": tuple(
                dict.fromkeys((
                    *(
                        member.target.id
                        for member in definition.body
                        if isinstance(member, ast.AnnAssign)
                        and isinstance(member.target, ast.Name)
                    ),
                    *(
                        target.attr
                        for member in definition.body
                        if isinstance(
                            member,
                            (ast.FunctionDef, ast.AsyncFunctionDef),
                        )
                        for target in source_walk(member)
                        if isinstance(target, ast.Attribute)
                        and isinstance(target.ctx, ast.Store)
                        and isinstance(target.value, ast.Name)
                        and target.value.id in {"self", "cls"}
                    ),
                ))
            ),
            # An omitted dataclass/class field is not an unknown shell input.
            # Its source-level literal default is part of the class schema,
            # just like the field order.  Preserve only literal data here:
            # arbitrary Python objects and executable default factories remain
            # unresolved and must be represented by ordinary graph structure.
            "field_defaults": class_field_defaults(definition),
        }
        for class_name, definition in class_definitions.items()
    }
    for function_node_id, owner_name in method_owners.items():
        reference = function_nodes.get(function_node_id)
        if reference is None:
            continue
        descriptor = graph.G.graph["class_table"].setdefault(
            owner_name,
            {"methods": {}, "fields": (), "field_defaults": {}},
        )
        function_definition = graph.G.nodes[function_node_id].get(
            "expr_obj"
        )
        if isinstance(
            function_definition,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            descriptor["methods"][function_definition.name] = int(
                reference.address
            )
    method_owner_by_reference = {
        int(reference.address): method_owners.get(function_node_id)
        for function_node_id, reference in function_nodes.items()
        if method_owners.get(function_node_id) is not None
    }

    contextual_requirements = list(
        graph.G.graph.get("contextual_requirements", ())
    )
    import_bindings: dict[
        str,
        tuple[str, str, dict[str, Any]],
    ] = {}
    for node_id, data in graph.G.nodes(data=True):
        statement = data.get("expr_obj")
        if not isinstance(statement, (ast.Import, ast.ImportFrom)):
            continue
        requirement = {
            "kind": (
                "import_from"
                if isinstance(statement, ast.ImportFrom)
                else "import"
            ),
            "module": (
                statement.module
                if isinstance(statement, ast.ImportFrom)
                else None
            ),
            "level": (
                int(statement.level)
                if isinstance(statement, ast.ImportFrom)
                else 0
            ),
            "names": tuple(
                (imported.name, imported.asname)
                for imported in statement.names
            ),
        }
        if requirement not in contextual_requirements:
            contextual_requirements.append(requirement)
        data.setdefault("attributes", {})[
            "contextual_requirement"
        ] = requirement
        for imported in statement.names:
            local_name = imported.asname or imported.name.split(".")[0]
            if isinstance(statement, ast.ImportFrom):
                module = "." * int(statement.level) + (
                    statement.module or ""
                )
                qualified_name = (
                    f"{module}.{imported.name}"
                    if module
                    else imported.name
                )
            else:
                qualified_name = imported.name
            import_bindings[local_name] = (
                qualified_name,
                imported.name,
                requirement,
            )
            imported_id = id(imported)
            if imported_id in graph.G:
                graph.G.nodes[imported_id].setdefault("attributes", {})[
                    "contextual_requirement"
                ] = requirement
        logger.info(
            "retaining ProcessGraph import as a contextual requirement: %s",
            requirement,
        )
    graph.G.graph["contextual_requirements"] = tuple(
        contextual_requirements
    )
    static_bindings = dict(getattr(graph, "python_bindings", {}) or {})
    # Parent-source expansion is optional, but literal module constants are
    # required static bindings in either ingestion mode.  Recover only
    # assignments outside function/class bodies; locals remain SSA values.
    scoped_member_ids = {
        id(member)
        for _owner_id, owner_data in graph.G.nodes(data=True)
        for owner in (owner_data.get("expr_obj"),)
        if isinstance(
            owner,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
        )
        for member in source_walk(owner)
        if member is not owner
    }
    for _node_id, node_data in graph.G.nodes(data=True):
        expression = node_data.get("expr_obj")
        if id(expression) in scoped_member_ids:
            continue
        if isinstance(expression, ast.Assign):
            targets = expression.targets
            value_node = expression.value
        elif isinstance(expression, ast.AnnAssign):
            targets = (expression.target,)
            value_node = expression.value
        else:
            continue
        if value_node is None:
            continue
        try:
            literal = ast.literal_eval(value_node)
        except (TypeError, ValueError, SyntaxError):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                static_bindings[target.id] = literal
    python_package = getattr(graph, "python_package", None)
    for local_name, (
        qualified_name,
        imported_name,
        requirement,
    ) in import_bindings.items():
        try:
            if requirement["kind"] == "import_from":
                module_name = (
                    "." * int(requirement.get("level", 0))
                    + str(requirement.get("module") or "")
                )
                module = importlib.import_module(
                    module_name,
                    package=python_package,
                )
                static_bindings[local_name] = getattr(
                    module,
                    imported_name,
                )
            elif requirement["kind"] == "import":
                static_bindings[local_name] = importlib.import_module(
                    qualified_name,
                    package=python_package,
                )
        except (ImportError, AttributeError, TypeError, ValueError):
            # The contextual requirement remains available to deployment even
            # when it cannot be resolved in the compiler's current process.
            continue
    for builtin_name, value in vars(builtins).items():
        static_bindings.setdefault(builtin_name, value)
    graph.python_bindings = static_bindings
    function_parameters_by_address = {}
    for function_node_id, reference in function_nodes.items():
        definition = graph.G.nodes[function_node_id].get("expr_obj")
        if not isinstance(
            definition,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda),
        ):
            continue
        function_parameters_by_address[reference.address] = {
            argument.arg
            for argument in (
                *definition.args.posonlyargs,
                *definition.args.args,
                *definition.args.kwonlyargs,
                *(
                    (definition.args.vararg,)
                    if definition.args.vararg is not None
                    else ()
                ),
                *(
                    (definition.args.kwarg,)
                    if definition.args.kwarg is not None
                    else ()
                ),
            )
        }
    lexical_functions_by_owner: dict[int, dict[str, Any]] = {}
    for child_id, parent_id in lexical_parent_by_function.items():
        child_reference = function_nodes.get(child_id)
        parent_reference = function_nodes.get(parent_id)
        child_definition = function_definitions.get(child_id)
        if (
            child_reference is None
            or parent_reference is None
            or not isinstance(
                child_definition,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            )
        ):
            continue
        lexical_functions_by_owner.setdefault(
            int(parent_reference.address), {}
        )[child_definition.name] = child_reference

    for _node_id, data in graph.G.nodes(data=True):
        source_type = data.get("type")
        process_type = _AST_PROCESS_GRAPH_ALIASES.get(source_type)
        if process_type is None:
            continue
        data["type"] = process_type
        data["op"] = process_type
        data.setdefault("attributes", {})["source_type"] = source_type

    for node_id, data in list(graph.G.nodes(data=True)):
        expression = data.get("expr_obj")
        # --- C (pycparser c_ast) canonicalization -------------------------
        # Dispatched on the type *name*, never isinstance, so this module
        # never imports pycparser (it is a frontend-optional dependency; a
        # pure-Python compile must not require it to be installed).
        #
        # This is the same universalization the Python branches below
        # perform, and it belongs here for the same reason: once a syntactic
        # element is canonicalized into the shared operator vocabulary
        # (Add/Load/Store/Call/Constant), no downstream pass needs to know
        # which language it came from. Skipping it is precisely what let raw
        # C node types survive all the way to the lexical-value pass.
        c_type_name = type(expression).__name__
        if expression is not None and c_type_name in _C_COMPILE_TIME_SYNTAX:
            # Pure type machinery, with no runtime value of its own: the
            # C counterpart of Python's Nonlocal/Global handling below.
            _remove_node(graph, node_id)
            continue
        if c_type_name == "Decl" and not isinstance(expression, ast.AST):
            # `int total = expr;` -- a declaration is not itself an
            # operation. Its initializer is the value the declared name
            # denotes, so the Decl collapses onto that value exactly the way
            # Python's own Name-store nodes collapse onto their producer.
            # A declaration with no initializer (`int total;`) names no
            # value at all yet and carries no runtime effect of its own.
            initializer = getattr(expression, "init", None)
            if initializer is not None and id(initializer) in graph.G:
                _redirect_value(graph, node_id, id(initializer))
            else:
                _remove_node(graph, node_id)
            continue
        if c_type_name == "Constant" and not isinstance(expression, ast.AST):
            value = _c_constant_value(expression)
            data["type"] = "Constant"
            data["op"] = "const"
            data["constant"] = value
            data.setdefault("attributes", {})["value"] = value
            continue
        if c_type_name == "BinaryOp":
            operation = _c_qualified_handler("binaryop", expression.op)
            data["type"] = operation
            data["op"] = operation
            data.setdefault("attributes", {})["source_type"] = "BinaryOp"
            # Absent operands are handled uniformly by _replace_inputs, which
            # substitutes an explicit non-operation and records a shortfall.
            # Filtering them out here instead would drop the operand
            # silently, leaving an operation short an argument -- a wrong
            # program rather than an honestly incomplete one.
            _replace_inputs(
                graph,
                node_id,
                (
                    (id(expression.left), "lhs"),
                    (id(expression.right), "rhs"),
                ),
            )
            continue
        if c_type_name == "UnaryOp" and not isinstance(expression, ast.AST):
            if str(expression.op) in {"+", "&", "*"}:
                # Unary plus is a no-op conversion, the same reading Python's
                # ast.UAdd gets below.
                #
                # Address-of and dereference are identity *in this value
                # graph specifically*: a node here denotes a value, and the
                # graph has no separate address space for `&x` to point
                # into that `x` does not already name. The cpp shell emits
                # `&obj` for every method receiver, and `obj`'s own node is
                # already exactly the receiver the callee needs. These stay
                # registered as GetElementPtr/Load in ssa_registry (their
                # true meanings, which a real memory model would need) --
                # collapsing them is a property of this representation, not
                # a claim that C's & and * are no-ops. Revisit when pointer
                # arithmetic or aliasing enters the shell's scope; today it
                # is excluded by CPP_LIKE_SHELL_FOR_C_INTENT.md.
                _redirect_value(graph, node_id, id(expression.expr))
                continue
            operation = _c_qualified_handler("unaryop", expression.op)
            data["type"] = operation
            data["op"] = operation
            data.setdefault("attributes", {})["source_type"] = "UnaryOp"
            _replace_inputs(
                graph,
                node_id,
                ((id(expression.expr), "operand"),),
            )
            continue
        if isinstance(expression, (ast.Nonlocal, ast.Global)):
            # Root whole-graph deployment regions are formed from this graph,
            # not only from the normalized per-function copies below.  Scope
            # declarations are compile-time syntax in both representations.
            _remove_node(graph, node_id)
            continue
        if isinstance(expression, ast.Delete):
            # The target nodes carry the executable deletion effects.  The
            # statement wrapper is ordering syntax, not an operator.
            _remove_node(graph, node_id)
            continue
        if (
            isinstance(expression, ast.Name)
            and isinstance(expression.ctx, ast.Del)
        ):
            # Deleting a lexical binding has no object-level runtime effect.
            _remove_node(graph, node_id)
            continue
        if (
            isinstance(expression, ast.Attribute)
            and isinstance(expression.ctx, ast.Del)
        ):
            data["type"] = "DelAttr"
            data["op"] = "delattr"
            data.setdefault("attributes", {})["attribute"] = expression.attr
            _replace_inputs(
                graph,
                node_id,
                ((id(expression.value), "object"),),
            )
            continue
        if isinstance(expression, ast.Constant):
            data["type"] = "Constant"
            data["op"] = "const"
            data["constant"] = expression.value
            data.setdefault("attributes", {})["value"] = expression.value
        elif isinstance(expression, ast.BinOp):
            folded = _static_sequence_literal(expression)
            if folded is not None:
                # Sequence replication is table allocation, not arithmetic.
                data["type"] = "Constant"
                data["op"] = "const"
                data["constant"] = folded
                attributes = data.setdefault("attributes", {})
                attributes["value"] = folded
                attributes["source_type"] = "BinOp"
                attributes["constant_folded"] = "sequence-replication"
                continue
            operation = _qualified_handler("binop", expression.op)
            data["type"] = operation
            data["op"] = operation
            data.setdefault("attributes", {})["source_type"] = "BinOp"
            _replace_inputs(
                graph,
                node_id,
                (
                    (id(expression.left), "lhs"),
                    (id(expression.right), "rhs"),
                ),
            )
        elif isinstance(expression, ast.AugAssign):
            operation = _qualified_handler("binop", expression.op)
            data["type"] = operation
            data["op"] = operation
            data.setdefault("attributes", {})["source_type"] = "AugAssign"
            _replace_inputs(
                graph,
                node_id,
                (
                    (id(expression.target), "lhs"),
                    (id(expression.value), "rhs"),
                ),
            )
        elif isinstance(expression, ast.UnaryOp):
            if isinstance(expression.op, ast.UAdd):
                _redirect_value(
                    graph,
                    node_id,
                    id(expression.operand),
                )
                continue
            operation = _qualified_handler("unaryop", expression.op)
            data["type"] = operation
            data["op"] = operation
            data.setdefault("attributes", {})["source_type"] = "UnaryOp"
            _replace_inputs(
                graph,
                node_id,
                ((id(expression.operand), "operand"),),
            )
        elif isinstance(expression, ast.BoolOp):
            data.setdefault("attributes", {})["source_type"] = "BoolOp"
            for value in expression.values[1:]:
                value_id = id(value)
                if value_id in graph.G:
                    graph.G.nodes[value_id].setdefault(
                        "attributes",
                        {},
                    )["coordinator_short_circuit"] = True
            _replace_inputs(
                graph,
                node_id,
                tuple(
                    (id(value), f"value:{index}")
                    for index, value in enumerate(expression.values)
                ),
            )
        elif isinstance(expression, ast.Compare) and len(expression.ops) == 1:
            operation = _qualified_handler("compare", expression.ops[0])
            data["type"] = operation
            data["op"] = operation
            data.setdefault("attributes", {})["source_type"] = "Compare"
            _replace_inputs(
                graph,
                node_id,
                (
                    (id(expression.left), "lhs"),
                    (id(expression.comparators[0]), "rhs"),
                ),
            )
        elif isinstance(expression, ast.Subscript):
            # Indexing is one operation over the parent tensor and the complete
            # index tuple.  In particular, Ellipsis is not an independent
            # scalar operation: AbstractTensor.__getitem__ and each specialized
            # backend resolve it against the parent tensor's rank.
            index_expressions = (
                tuple(expression.slice.elts)
                if isinstance(expression.slice, ast.Tuple)
                else (expression.slice,)
            )
            deleting = isinstance(expression.ctx, ast.Del)
            data["type"] = "DelItem" if deleting else "Indexed"
            data["op"] = "delitem" if deleting else "Indexed"
            data.setdefault("attributes", {})["source_type"] = "Subscript"
            _replace_inputs(
                graph,
                node_id,
                (
                    (id(expression.value), "base"),
                    *(
                        (id(index_expression), "index")
                        for index_expression in index_expressions
                    ),
                ),
            )

            # The AST Tuple only grouped the index components.  Once those
            # components feed Indexed directly, retaining the wrapper would
            # incorrectly schedule a second tuple-producing computation.
            slice_id = id(expression.slice)
            if (
                isinstance(expression.slice, ast.Tuple)
                and slice_id in graph.G
                and graph.G.out_degree(slice_id) == 0
            ):
                for predecessor in tuple(graph.G.predecessors(slice_id)):
                    graph.G.nodes[predecessor]["children"] = [
                        (child_id, role)
                        for child_id, role in graph.G.nodes[
                            predecessor
                        ].get("children", ())
                        if child_id != slice_id
                    ]
                graph.G.remove_node(slice_id)
        elif (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Attribute)
        ):
            operation = expression.func.attr
            data["type"] = operation
            data["op"] = operation
            data.setdefault("attributes", {})["source_type"] = "Call"
            receiver_name = (
                expression.func.value.id
                if isinstance(expression.func.value, ast.Name)
                else None
            )
            call_owner = call_owners.get(node_id)
            lexical_class = (
                method_owner_by_reference.get(int(call_owner.address))
                if call_owner is not None
                else None
            )
            if receiver_name in {"self", "cls"} and lexical_class is not None:
                method_reference = (
                    graph.G.graph.get("class_table", {})
                    .get(lexical_class, {})
                    .get("methods", {})
                    .get(expression.func.attr)
                )
                if method_reference is not None:
                    data.setdefault("attributes", {})["method_ref"] = int(
                        method_reference
                    )
            _replace_inputs(
                graph,
                node_id,
                (
                    (id(expression.func.value), "operand"),
                    *(
                        (id(argument), f"arg{index}")
                        for index, argument in enumerate(expression.args)
                    ),
                    *(
                        (
                            id(keyword.value),
                            (
                                f"kw:{keyword.arg}"
                                if keyword.arg is not None
                                else "kwargs"
                            ),
                        )
                        for keyword in expression.keywords
                    ),
                ),
            )
        elif isinstance(expression, ast.Call):
            call_inputs: list[tuple[int, str]] = [
                (
                    id(argument),
                    f"arg:{index}",
                )
                for index, argument in enumerate(expression.args)
            ]
            call_inputs.extend(
                (
                    id(keyword.value),
                    (
                        f"kw:{keyword.arg}"
                        if keyword.arg is not None
                        else "kwargs"
                    ),
                )
                for keyword in expression.keywords
            )
            if isinstance(expression.func, ast.Name):
                callee_name = expression.func.id
                owner = call_owners.get(node_id)
                callee_is_parameter = (
                    owner is not None
                    and callee_name in function_parameters_by_address.get(
                        owner.address,
                        (),
                    )
                )
                if callee_is_parameter:
                    attributes = data.setdefault("attributes", {})
                    attributes.pop("callee_ref", None)
                    attributes.pop("external_callee_ref", None)
                    call_inputs.insert(
                        0,
                        (id(expression.func), "callee"),
                    )
                    reference = None
                else:
                    reference = function_table.reference(callee_name)
                if (
                    not callee_is_parameter
                    and reference is None
                    and callee_name in import_bindings
                ):
                    (
                        qualified_name,
                        imported_name,
                        requirement,
                    ) = import_bindings[
                        callee_name
                    ]
                    external_reference = external_function_table.declare(
                        callee_name,
                        qualified_name=qualified_name,
                        external=True,
                        metadata={
                            "contextual_requirement": requirement,
                            "imported_name": imported_name,
                        },
                    )
                    data.setdefault("attributes", {})[
                        "external_callee_ref"
                    ] = external_reference.address
                elif (
                    not callee_is_parameter
                    and reference is not None
                ):
                    data.setdefault("attributes", {})[
                        "callee_ref"
                    ] = reference.address
                elif not callee_is_parameter:
                    call_inputs.insert(
                        0,
                        (id(expression.func), "callee"),
                    )
            else:
                call_inputs.insert(0, (id(expression.func), "callee"))
            _replace_inputs(graph, node_id, tuple(call_inputs))

            # Keyword nodes carry only source spelling once their value and
            # name have been transferred to the Call edge role.
            for keyword in expression.keywords:
                keyword_id = id(keyword)
                if (
                    keyword_id not in graph.G
                    or graph.G.out_degree(keyword_id) != 0
                ):
                    continue
                for predecessor in tuple(graph.G.predecessors(keyword_id)):
                    graph.G.nodes[predecessor]["children"] = [
                        (child_id, role)
                        for child_id, role in graph.G.nodes[
                            predecessor
                        ].get("children", ())
                        if child_id != keyword_id
                    ]
                graph.G.remove_node(keyword_id)

    for node_id, data in list(graph.G.nodes(data=True)):
        statement = data.get("expr_obj")
        if not isinstance(statement, ast.For):
            continue
        attributes = data.setdefault("attributes", {})
        attributes["source_type"] = "For"
        attributes["target"] = (
            statement.target.id
            if isinstance(statement.target, ast.Name)
            else ast.dump(statement.target, include_attributes=False)
        )
        loop_inputs: list[tuple[int, str]] = []
        iterator = statement.iter
        if (
            isinstance(iterator, ast.Call)
            and isinstance(iterator.func, ast.Name)
            and iterator.func.id == "range"
        ):
            attributes["iterator_kind"] = "arithmetic_sequence"
            range_arguments = tuple(iterator.args)
            if not 1 <= len(range_arguments) <= 3:
                raise ValueError("range loop requires one to three arguments")
            roles = (
                ("stop",)
                if len(range_arguments) == 1
                else ("start", "stop")
                if len(range_arguments) == 2
                else ("start", "stop", "step")
            )
            if len(range_arguments) == 1:
                attributes["start"] = 0
                attributes["step"] = 1
            elif len(range_arguments) == 2:
                attributes["step"] = 1
            loop_inputs.extend(
                (id(argument), role)
                for argument, role in zip(range_arguments, roles)
            )
        else:
            attributes["iterator_kind"] = "iterable"
            loop_inputs.append((id(iterator), "iterable"))
        loop_inputs.extend(
            (id(body_statement), "body")
            for body_statement in statement.body
        )
        loop_inputs.extend(
            (id(else_statement), "else")
            for else_statement in statement.orelse
        )
        _replace_inputs(graph, node_id, tuple(loop_inputs))

        if isinstance(iterator, ast.Call) and id(iterator) in graph.G:
            iterator_id = id(iterator)
            if graph.G.out_degree(iterator_id) == 0:
                for predecessor in tuple(graph.G.predecessors(iterator_id)):
                    graph.G.nodes[predecessor]["children"] = [
                        (child_id, role)
                        for child_id, role in graph.G.nodes[
                            predecessor
                        ].get("children", ())
                        if child_id != iterator_id
                    ]
                graph.G.remove_node(iterator_id)

    for node_id, data in list(graph.G.nodes(data=True)):
        if data.get("type") != "Expr":
            continue
        predecessors = tuple(graph.G.predecessors(node_id))
        successors = tuple(graph.G.successors(node_id))
        if len(predecessors) > 1:
            raise ValueError("Expr wrapper must contain at most one value")
        predecessor = predecessors[0] if predecessors else None
        for successor in successors:
            successor_data = graph.G.nodes[successor]
            replacement_parents = []
            for parent_id, role in successor_data.get("parents", ()):
                if parent_id != node_id:
                    replacement_parents.append((parent_id, role))
                elif predecessor is not None:
                    replacement_parents.append((predecessor, role))
            successor_data["parents"] = replacement_parents
            if predecessor is not None:
                graph.G.add_edge(predecessor, successor)
                predecessor_children = graph.G.nodes[predecessor].setdefault(
                    "children", []
                )
                if successor not in {
                    child_id for child_id, _role in predecessor_children
                }:
                    predecessor_children.append((successor, "output"))
        if predecessor is not None:
            graph.G.nodes[predecessor]["children"] = [
                (child_id, role)
                for child_id, role in graph.G.nodes[predecessor].get(
                    "children", ()
                )
                if child_id != node_id
            ]
        graph.roots = [
            predecessor if root_id == node_id and predecessor is not None
            else root_id
            for root_id in graph.roots
            if root_id != node_id or predecessor is not None
        ]
        graph.G.remove_node(node_id)

    for node_id, data in list(graph.G.nodes(data=True)):
        if data.get("type") != "Return":
            continue
        predecessors = tuple(graph.G.predecessors(node_id))
        if len(predecessors) > 1:
            raise ValueError("Return wrapper must contain at most one value")
        returned = predecessors[0] if predecessors else None
        for successor in tuple(graph.G.successors(node_id)):
            graph.G.nodes[successor]["parents"] = [
                (parent_id, role)
                for parent_id, role in graph.G.nodes[successor].get(
                    "parents", ()
                )
                if parent_id != node_id
            ]
        if returned is not None:
            graph.G.nodes[returned]["children"] = [
                (child_id, role)
                for child_id, role in graph.G.nodes[returned].get(
                    "children", ()
                )
                if child_id != node_id
            ]
            if returned not in graph.roots:
                graph.roots.append(returned)
        graph.roots = [root for root in graph.roots if root != node_id]
        graph.G.remove_node(node_id)

    call_graph = nx.DiGraph()
    call_graph.add_nodes_from(
        reference.address for reference in function_nodes.values()
    )
    referenced_calls: list[tuple[int, Any, Any]] = []
    for call_id, owner in call_owners.items():
        if call_id not in graph.G:
            continue
        callee_address = (
            graph.G.nodes[call_id].get("attributes") or {}
        ).get("callee_ref")
        if callee_address is None:
            continue
        referenced_calls.append((call_id, owner, callee_address))
        try:
            callee_entry = function_table.entry(callee_address)
        except KeyError:
            continue
        if callee_entry.graph is None and callee_address not in {
            reference.address for reference in function_nodes.values()
        }:
            continue
        call_graph.add_edge(owner.address, callee_address)

    recursive_edges: set[tuple[int, int]] = set()
    for component in nx.strongly_connected_components(call_graph):
        if len(component) > 1:
            recursive_edges.update(
                (source, target)
                for source, target in call_graph.edges(component)
                if target in component
            )
        else:
            address = next(iter(component))
            if call_graph.has_edge(address, address):
                recursive_edges.add((address, address))
    for call_id, owner, callee_address in referenced_calls:
        if (owner.address, callee_address) not in recursive_edges:
            continue
        graph.G.nodes[call_id].setdefault("attributes", {})[
            "recursive_backedge"
        ] = True
        function_table.entry(owner).recursive = True

    for node_id, reference in function_nodes.items():
        if node_id not in graph.G:
            continue
        return_values = [
            value
            for value in function_return_values.get(node_id, ())
            if value in graph.G
        ]
        terminal_merges = [
            int(member)
            for member in graph.G
            if (
                (graph.G.nodes[member].get("attributes") or {}).get(
                    "terminal_return_merge"
                )
                and set(
                    (graph.G.nodes[member].get("attributes") or {}).get(
                        "terminal_return_values", ()
                    )
                ).issubset(return_values)
            )
        ]
        if terminal_merges:
            merged_values = {
                int(value)
                for member in terminal_merges
                for value in (
                    graph.G.nodes[member].get("attributes") or {}
                ).get("terminal_return_values", ())
            }
            return_values = [
                value for value in return_values if value not in merged_values
            ]
            return_values.extend(terminal_merges)
        statement = graph.G.nodes[node_id].get("expr_obj")
        definition_static_bindings = dict(
            getattr(statement, "_python_bindings", static_bindings)
        )
        for builtin_name, builtin_value in vars(builtins).items():
            definition_static_bindings.setdefault(
                builtin_name,
                builtin_value,
            )
        # Function ownership is already exact in the saved Python AST.  Use
        # that ownership directly: graph ancestry from only the return value
        # silently discarded assignments, calls, loops, and side effects.
        #
        # This deliberately still walks the raw ``ast`` tree, not
        # ``graph.G`` -- unlike ``_record_owned_calls_and_returns`` above,
        # which runs right after ingestion while the graph still mirrors the
        # source tree 1:1. By the time this loop runs, earlier reduction
        # passes have already restructured ``graph.G`` (edges no longer
        # necessarily match the original AST parent/child shape), so a
        # graph-native walk here would silently miss nodes a raw-tree walk
        # still finds (confirmed empirically: a graph-based version of this
        # walk lost the Return node itself for a plain `return (...)`
        # function body). Generalizing *this* walk to a foreign language
        # needs its own graph-native traversal designed against the
        # post-reduction graph shape, not a drop-in swap -- tracked
        # separately, not attempted here.
        owned_members: set[int] = set()

        def record_owned_member(member: Any) -> None:
            if isinstance(member, ast.ClassDef):
                # A class's body belongs to its own methods' entries.
                return
            if is_runnable_definition(member):
                # A nested definition's body belongs to another
                # function-table entry, not to this enclosing shell. A
                # lambda is still itself a value the enclosing scope
                # references, so it is owned without being descended into.
                if isinstance(member, ast.Lambda) and id(member) in graph.G:
                    owned_members.add(id(member))
                return
            if id(member) in graph.G and not isinstance(
                member,
                (
                    ast.arguments,
                    ast.arg,
                    ast.expr_context,
                    ast.operator,
                    ast.unaryop,
                    ast.boolop,
                    ast.cmpop,
                    ast.keyword,
                    ast.alias,
                    ast.Import,
                    ast.ImportFrom,
                ),
            ):
                owned_members.add(id(member))
            for child in source_child_nodes(member):
                record_owned_member(child)

        if isinstance(statement, ast.Lambda):
            # A lambda's body is one expression, and is the body itself
            # rather than a list of statements -- descend into it directly
            # instead of treating it as a nested-definition boundary.
            for child in source_child_nodes(statement.body):
                record_owned_member(child)
            if id(statement.body) in graph.G:
                owned_members.add(id(statement.body))
        else:
            for body_member in source_body_statements(statement):
                record_owned_member(body_member)
        included = owned_members
        included = {
            member
            for member in included
            if (
                member in return_values
                or any(
                    neighbor in included
                    for neighbor in graph.G.predecessors(member)
                )
                or any(
                    neighbor in included
                    for neighbor in graph.G.successors(member)
                )
            )
        }
        function_graph = copy.copy(graph)
        function_graph.G = graph.G.subgraph(included).copy()
        # The definition may carry literal module constants and source-local
        # imports beyond the root graph's generic binding set.  Preserve that
        # exact static environment on the function graph consumed by compiled
        # shells; otherwise those globals reappear as missing runtime inputs.
        function_graph.python_bindings = definition_static_bindings
        function_graph.levels = {
            member: level
            for member, level in graph.levels.items()
            if member in included
        }
        function_graph.roots = return_values or [node_id]
        function_graph.function_table = function_table
        positional_parameters = ()
        keyword_only_parameters = ()
        variadic_parameters = ()
        if isinstance(
            statement,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda),
        ):
            positional_parameters = tuple(
                argument.arg
                for argument in (
                    *statement.args.posonlyargs,
                    *statement.args.args,
                )
            )
            keyword_only_parameters = tuple(
                argument.arg for argument in statement.args.kwonlyargs
            )
            variadic_parameters = tuple(
                argument.arg
                for argument in (statement.args.vararg, statement.args.kwarg)
                if argument is not None
            )
        parameter_names = (
            *positional_parameters,
            *keyword_only_parameters,
            *variadic_parameters,
        )
        parameter_defaults = {}
        if isinstance(
            statement,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda),
        ):
            if statement.args.vararg is not None:
                parameter_defaults[statement.args.vararg.arg] = ()
            if statement.args.kwarg is not None:
                parameter_defaults[statement.args.kwarg.arg] = {}
        scalar_parameter_names = set()
        if isinstance(
            statement,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda),
        ):
            for argument in (
                *statement.args.posonlyargs,
                *statement.args.args,
                *statement.args.kwonlyargs,
            ):
                if (
                    isinstance(argument.annotation, ast.Name)
                    and argument.annotation.id
                    in {"bool", "bytes", "complex", "float", "int", "str"}
                ):
                    scalar_parameter_names.add(argument.arg)
            positional_defaults = zip(
                positional_parameters[
                    len(positional_parameters) - len(statement.args.defaults):
                ],
                statement.args.defaults,
            )
            keyword_defaults = zip(
                keyword_only_parameters,
                statement.args.kw_defaults,
            )
            for name, default_expression in (
                *positional_defaults,
                *keyword_defaults,
            ):
                if default_expression is None:
                    continue
                try:
                    value = ast.literal_eval(default_expression)
                except (ValueError, TypeError):
                    if isinstance(default_expression, ast.Name):
                        value = static_bindings.get(
                            default_expression.id,
                            default_expression,
                        )
                    else:
                        continue
                parameter_defaults[name] = value
                if isinstance(
                    value,
                    (bool, bytes, complex, float, int, str),
                ):
                    scalar_parameter_names.add(name)
        function_graph.G.graph.update(
            function_ref=reference.address,
            function_name=function_table.entry(reference).name,
            method_owner=method_owners.get(node_id),
            method_binding=(
                "class"
                if isinstance(
                    statement,
                    (ast.FunctionDef, ast.AsyncFunctionDef),
                )
                and any(
                    isinstance(decorator, ast.Name)
                    and decorator.id == "classmethod"
                    for decorator in statement.decorator_list
                )
                else (
                    "static"
                    if isinstance(
                        statement,
                        (ast.FunctionDef, ast.AsyncFunctionDef),
                    )
                    and any(
                        isinstance(decorator, ast.Name)
                        and decorator.id == "staticmethod"
                        for decorator in statement.decorator_list
                    )
                    else (
                        "instance"
                        if method_owners.get(node_id) is not None
                        else None
                    )
                )
            ),
            function_parameters=parameter_names,
            positional_parameters=positional_parameters,
            keyword_only_parameters=keyword_only_parameters,
            parameter_defaults=parameter_defaults,
            scalar_parameters=tuple(sorted(scalar_parameter_names)),
            function_body=(
                tuple(statement.body)
                if isinstance(
                    statement,
                    (ast.FunctionDef, ast.AsyncFunctionDef),
                )
                else ()
            ),
        )
        for member in function_graph.G:
            member_data = function_graph.G.nodes[member]
            expression = member_data.get("expr_obj")
            if (
                isinstance(expression, ast.Name)
                and isinstance(expression.ctx, ast.Load)
                and expression.id in parameter_names
            ):
                member_data["type"] = "Input"
                member_data["op"] = "input"
                member_data["label"] = expression.id
                member_data["parents"] = []
            if (
                member_data.get("type") == "Input"
                and (
                    member_data.get("attributes") or {}
                ).get("binding_name") in scalar_parameter_names
            ):
                member_data.setdefault("attributes", {})[
                    "value_kind"
                ] = "scalar"
            member_data["parents"] = [
                (parent, role)
                for parent, role in member_data.get("parents", ())
                if parent in included
            ]
            member_data["children"] = [
                (child, role)
                for child, role in member_data.get("children", ())
                if child in included
            ]
        normalization_statement = statement
        if isinstance(statement, ast.Lambda):
            normalization_statement = ast.copy_location(
                ast.FunctionDef(
                    name=function_table.entry(reference).name,
                    args=statement.args,
                    body=[
                        ast.copy_location(
                            ast.Return(value=statement.body),
                            statement.body,
                        )
                    ],
                    decorator_list=[],
                    returns=None,
                    type_comment=None,
                ),
                statement,
            )
        _normalize_lexical_values(
            function_graph,
            normalization_statement,
            definition_static_bindings,
            function_table,
            lexical_functions_by_owner.get(int(reference.address), {}),
        )
        generator_yields = tuple(
            node_id
            for node_id, node_data in function_graph.G.nodes(data=True)
            if isinstance(
                node_data.get("expr_obj"),
                (ast.Yield, ast.YieldFrom),
            )
        )
        if generator_yields:
            function_graph.G.graph["generator_stream"] = {
                "yield_nodes": generator_yields,
                "flow_control": "downstream_capacity",
                "execution_owner": "planner_shell",
            }
        for _member, member_data in function_graph.G.nodes(data=True):
            if (
                member_data.get("type") == "Input"
                and (
                    member_data.get("attributes") or {}
                ).get("binding_name") in scalar_parameter_names
            ):
                member_data.setdefault("attributes", {})[
                    "value_kind"
                ] = "scalar"
        if hasattr(statement, "_python_bindings"):
            delattr(statement, "_python_bindings")
        function_table.resolve_graph(reference, function_graph)
    # External call references and static Python bindings are two views of the
    # same compile-time environment.  Join them once after every call has been
    # declared so compiled shells can invoke imported constructors/functions
    # without a second lookup mechanism.
    for entry in external_function_table:
        target = static_bindings.get(entry.name)
        if callable(target):
            external_function_table.resolve_callable(entry.reference, target)
    return graph


__all__ = ["reduce_abstract_tensor_topology"]
