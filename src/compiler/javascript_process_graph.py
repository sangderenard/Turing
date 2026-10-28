"""Conservative JavaScript-to-ProcessGraph interpretation.

Acorn/ESTree spellings are consumed here and replaced with the repository's
existing ProcessGraph operations.  This module does not own tokenization:
graphs produced here use the normal node schema, ingestion cells and operand
writer, so Turing's ordinary canonical relabel produces the structural token
chains and ``ssa_identity_tokens`` later in lowering.

The accepted program region is intentionally small and exact: one acyclic
function, parameters, ``const`` declarations, pure expressions, returns and
an early-return ``if``.  Everything else remains an explicit shortfall.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

from ..common.tensors.topological_reducer import _set_operands, node_identity_cell
from ..transmogrifier.graph.javascript_identity_programs import (
    begin_javascript_ingestion,
    post_javascript_ingestion_identity,
    resolve_javascript_identity,
)
from ..transmogrifier.solver_types import DomainNode
from ..transmogrifier.ssa_registry import Handler
from .concordance_declarations import (
    BindingFact,
    INGEST_EDGE,
    INGESTION,
    NAME_BINDING,
)
from .identity_concordance import Derived, Mode, current_identity_book
from .javascript_source_tables import (
    JAVASCRIPT_BINARY_TO_SSA,
    JAVASCRIPT_DIRECT_CALLS,
    JAVASCRIPT_SURFACE_IDENTITIES,
    JAVASCRIPT_UNARY_TO_SSA,
    JAVASCRIPT_UNSUPPORTED_BINARY,
    JAVASCRIPT_UNSUPPORTED_CALLS,
    JAVASCRIPT_UNSUPPORTED_UNARY,
    SSA_TO_JAVASCRIPT_BINARY,
    SSA_TO_JAVASCRIPT_UNARY,
)


_REVERSE_DIRECT_CALLS: Mapping[str, str] = {
    name: spelling for spelling, name in JAVASCRIPT_DIRECT_CALLS.items()
}


@dataclass(frozen=True, order=True)
class JavaScriptExpressionShortfall:
    code: str
    message: str

    def format(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass
class JavaScriptExpressionLowering:
    graph: Any
    root: int
    shortfalls: tuple[JavaScriptExpressionShortfall, ...]
    entry_identity: str | None = None
    binding_identities: tuple[Mapping[str, Any], ...] = ()

    @property
    def complete(self) -> bool:
        return not self.shortfalls


PORTABLE_NUMERIC_ASSUMPTIONS = frozenset({
    "boolean-operands",
    "number-operands",
    "number-truthiness",
    "pure-operands",
    "primitive-same-type-operands",
    "ecmascript-math-number-semantics",
    "ecmascript-number-semantics",
    "unshadowed-realm-intrinsics",
})


_BINARY_REQUIREMENTS: Mapping[str, frozenset[str]] = {
    **{
        spelling: frozenset({
            "number-operands", "ecmascript-number-semantics",
        })
        for spelling in ("+", "-", "*", "/", "<", "<=", ">", ">=")
    },
    "**": frozenset({
        "number-operands", "ecmascript-number-semantics",
    }),
    "&&": frozenset({"boolean-operands", "pure-operands"}),
    "||": frozenset({"boolean-operands", "pure-operands"}),
    "===": frozenset({"primitive-same-type-operands"}),
    "!==": frozenset({"primitive-same-type-operands"}),
}


def _callee_parts(callee: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not callee:
        return ()
    kind = callee.get("type")
    if kind == "Identifier":
        return (str(callee.get("name", "")),)
    if kind == "MemberExpression" and not callee.get("computed"):
        left = _callee_parts(callee.get("object"))
        right = callee.get("property") or {}
        if left and right.get("type") == "Identifier":
            return (*left, str(right.get("name", "")))
    return ()


def _callee_spelling(callee: Mapping[str, Any] | None) -> str:
    return ".".join(_callee_parts(callee))


#: Nodes whose ``break``/``continue`` do not leave an enclosing loop.
_OPAQUE_TO_LOOP_EXITS = frozenset({
    "ForStatement", "WhileStatement", "DoWhileStatement", "ForInStatement",
    "ForOfStatement", "SwitchStatement", "FunctionDeclaration",
    "FunctionExpression", "ArrowFunctionExpression",
})


_FUNCTION_NODES = frozenset({
    "FunctionDeclaration", "FunctionExpression", "ArrowFunctionExpression",
})


def _loop_of(statement: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The ``for``/``while`` loop ``statement`` is, or labels; else None."""

    kind = statement.get("type")
    if kind in {"ForStatement", "WhileStatement"}:
        return statement
    if kind == "LabeledStatement":
        body = statement.get("body") or {}
        if body.get("type") in {"ForStatement", "WhileStatement"}:
            return body
    return None


def _return_names(loop: Mapping[str, Any]) -> tuple[str, str]:
    """The hidden return flag and result of the loop at ``loop``."""

    position = f"{loop.get('start')}:{loop.get('end')}"
    return f"%return@{position}", f"%result@{position}"


def _return_check(loop: Mapping[str, Any]) -> dict[str, Any]:
    """``if (returned) { return result; }`` following a loop that returns."""

    flag, result = _return_names(loop)
    at = {"start": loop.get("end"), "end": loop.get("end")}
    return {
        "type": "IfStatement", **at,
        "test": {"type": "Identifier", "name": flag, **at},
        "consequent": {
            "type": "BlockStatement", **at,
            "body": [{
                "type": "ReturnStatement", **at,
                "argument": {"type": "Identifier", "name": result, **at},
            }],
        },
        "alternate": None,
    }


def _at(source: Mapping[str, Any]) -> dict[str, Any]:
    return {"start": source.get("start"), "end": source.get("end")}


def _flag_assignment(
    name: str,
    source: Mapping[str, Any],
    value: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """``name = true;`` standing for an exit (or ``name = value;``)."""

    right = value if value is not None else {
        "type": "Literal", "value": True, **_at(source),
    }
    return {
        "type": "ExpressionStatement", **_at(source),
        "expression": {
            "type": "AssignmentExpression", "operator": "=", **_at(source),
            "left": {"type": "Identifier", "name": name, **_at(source)},
            "right": right,
        },
    }


def _flag_declaration(name: str, source: Mapping[str, Any]) -> dict[str, Any]:
    """``let name = false;`` opening each iteration's body."""

    return {
        "type": "VariableDeclaration", "kind": "let", **_at(source),
        "declarations": [{
            "type": "VariableDeclarator", **_at(source),
            "id": {"type": "Identifier", "name": name, **_at(source)},
            "init": {"type": "Literal", "value": False, **_at(source)},
        }],
    }


def _flags_clear(names: Any, source: Mapping[str, Any]) -> dict[str, Any]:
    """``!a && !b ...`` over the hidden exit flags."""

    test: dict[str, Any] | None = None
    for name in names:
        clear = {
            "type": "UnaryExpression", "operator": "!", "prefix": True,
            **_at(source),
            "argument": {"type": "Identifier", "name": name, **_at(source)},
        }
        test = clear if test is None else {
            "type": "LogicalExpression", "operator": "&&", **_at(source),
            "left": test, "right": clear,
        }
    assert test is not None
    return test


#: Identifier fields that name no binding: labels and dotted property names.
_NOT_REFERENCES = frozenset({"label", "property", "key"})


def _var_as_let(
    statements: Sequence[Mapping[str, Any]],
    parameters: frozenset[str],
) -> list[Mapping[str, Any]]:
    """``var`` declarations rewritten into ``let`` where the two agree.

    A ``var`` is function scoped and hoisted (undefined until its
    declaration runs); a ``let`` is block scoped.  Without closures the two
    read alike when the first declaration of a name has an initializer and
    every read and later declaration of the name lies after it, inside the
    statement list that holds it.  That first declaration becomes a
    ``let`` (a ``for (var ...)`` initializer is hoisted to a ``let`` just
    before its loop, which still holds every read), later declarations
    become assignments (``var x;`` alone is dropped), and a declaration of
    a parameter is an assignment.  Any other ``var`` is left alone, and the
    interpreter refuses it.
    """

    declarations: dict[str, list[tuple[int, int, tuple[int, ...], bool]]] = {}
    references: dict[str, list[tuple[int, tuple[int, ...]]]] = {}
    lexical: set[str] = set()

    def visit(item: Any, lists: tuple[int, ...], field: str | None) -> None:
        if isinstance(item, list):
            if field in {"body", "consequent"} and all(
                isinstance(child, Mapping) for child in item
            ):
                lists = (*lists, id(item))
            for child in item:
                visit(child, lists, None)
            return
        if not isinstance(item, Mapping):
            return
        kind = item.get("type")
        if kind in _FUNCTION_NODES:
            return
        if kind == "Identifier":
            if field not in _NOT_REFERENCES:
                references.setdefault(str(item.get("name")), []).append(
                    (int(item.get("start") or 0), lists),
                )
            return
        if kind == "MemberExpression" and not item.get("computed"):
            visit(item.get("object"), lists, "object")
            return
        if kind == "VariableDeclaration":
            for declarator in item.get("declarations", ()):
                target = declarator.get("id") or {}
                if target.get("type") != "Identifier":
                    continue
                name = str(target.get("name"))
                if item.get("kind") != "var":
                    lexical.add(name)
                    continue
                declarations.setdefault(name, []).append((
                    int(target.get("start") or 0),
                    int(declarator.get("end") or 0), lists,
                    declarator.get("init") is not None,
                ))
            for declarator in item.get("declarations", ()):
                visit(declarator.get("init"), lists, "init")
            return
        for key, child in item.items():
            if key not in {"type", "start", "end"}:
                visit(child, lists, key)

    root = list(statements)
    visit(root, (), "body")
    lets: set[str] = set()
    for name, found in declarations.items():
        if name in parameters or name in lexical:
            continue
        start, end, lists, initialized = min(found, key=lambda entry: entry[0])
        scope = lists[-1] if lists else None
        if initialized and scope is not None and all(
            scope in where and position >= end
            for position, where in (
                *references.get(name, ()),
                *((entry[0], entry[2]) for entry in found if entry[0] != start),
            )
        ):
            lets.add(name)
    if not declarations:
        return root
    first: dict[str, int] = {
        name: min(entry[0] for entry in declarations[name]) for name in lets
    }

    def rewrite(declaration: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
        """The statements a ``var`` becomes, or None to leave it."""

        out: list[Mapping[str, Any]] = []
        for declarator in declaration.get("declarations", ()):
            target = declarator.get("id") or {}
            name = str(target.get("name"))
            initializer = declarator.get("init")
            if name in lets and int(target.get("start") or 0) == first[name]:
                out.append({
                    **declaration, "kind": "let", "declarations": [declarator],
                })
            elif name in lets or name in parameters:
                if initializer is not None:
                    out.append(_flag_assignment(name, declarator, initializer))
            else:
                return None
        return out

    def walk(item: Any) -> Any:
        if isinstance(item, list):
            result: list[Any] = []
            for child in item:
                if isinstance(child, Mapping) and child.get("type") == "VariableDeclaration" and child.get("kind") == "var":
                    replaced = rewrite(child)
                    result.extend([child] if replaced is None else replaced)
                    continue
                loop = _loop_of(child) if isinstance(child, Mapping) else None
                initializer = (loop or {}).get("init") or {}
                if (
                    initializer.get("type") == "VariableDeclaration"
                    and initializer.get("kind") == "var"
                ):
                    replaced = rewrite(initializer)
                    if replaced is not None and all(
                        statement.get("type") == "VariableDeclaration"
                        for statement in replaced
                    ):
                        result.extend(replaced)
                        child = _without_init(child)
                result.append(walk(child))
            return result
        if isinstance(item, Mapping):
            if item.get("type") in _FUNCTION_NODES:
                return item
            return {key: walk(child) for key, child in item.items()}
        return item

    return walk(root)


def _without_init(statement: Mapping[str, Any]) -> dict[str, Any]:
    """``statement`` (a ``for`` loop or a label on one) with no initializer."""

    if statement.get("type") == "LabeledStatement":
        return {**statement, "body": _without_init(statement["body"])}
    return {**statement, "init": None}


class _JavaScriptProcessInterpreter:
    def __init__(
        self, graph: Any, *, owner: str, assumptions: Sequence[str] = (),
        parameter_kinds: Mapping[str, str] | None = None,
    ) -> None:
        self.graph = graph
        self.owner = str(owner)
        self.assumptions = frozenset(map(str, assumptions))
        self.parameter_kinds = dict(parameter_kinds or {})
        self.last_version: dict[str, int] = {}
        self.loops: list[dict[str, Any]] = []
        self.loop_count = 0
        # Labeled loops being interpreted: label -> their exit flags.
        self.loop_labels: dict[str, dict[Any, str]] = {}
        self.shortfalls: list[JavaScriptExpressionShortfall] = []
        self.bindings: dict[str, dict[str, Any]] = {}
        self.binding_records: list[dict[str, Any]] = []
        self.identifier_occurrences: list[dict[str, Any]] = []
        self.entry_identity: str | None = None
        self.next_id = max(
            (item for item in graph.G if isinstance(item, int)), default=-1,
        ) + 1
        self.scope = begin_javascript_ingestion(graph, self.owner)

    def _span(self, source: Mapping[str, Any] | None) -> dict[str, Any]:
        source = source or {}
        return {
            "filename": self.owner,
            "start": source.get("start"),
            "end": source.get("end"),
            "layers": (),
        }

    def make_node(
        self,
        node_type: str,
        operation: str | None,
        label: str,
        parents: Sequence[tuple[int, str]],
        *,
        attributes: Mapping[str, Any] | None = None,
        constant: Any = None,
        source: Mapping[str, Any] | None = None,
        output_roles: Sequence[str] = ("result",),
        value_kind: str = "unknown",
    ) -> int:
        node_id = self.next_id
        self.next_id += 1
        attrs = {
            "source_language": "javascript",
            "javascript_owner": self.owner,
            "javascript_value_kind": value_kind,
            **dict(attributes or {}),
        }
        domain = DomainNode(shape=(1, 1, 1), unit_size=1)
        domain.id = id(domain)
        self.graph.G.add_node(
            node_id,
            type=node_type,
            op=operation,
            label=label,
            expr_obj=None,
            attributes=attrs,
            extra_args=attrs,
            constant=constant,
            tensor={},
            bit_quanta={},
            control={},
            source_span=self._span(source),
            input_roles=tuple(role for _parent, role in parents),
            output_roles=tuple(output_roles),
            schema_version=1,
            domain_node=domain,
            store_id=None,
            parents=[],
            children=[],
        )
        post_javascript_ingestion_identity(
            self.graph,
            node_id,
            owner=self.owner,
            qualname=self.entry_identity or "<expression>",
            kind=node_type,
            operation=str(operation or ""),
            label=label,
            source=source,
        )
        if parents:
            _set_operands(
                self.graph, node_id, list(parents), cause=INGEST_EDGE,
            )
        return node_id

    def unsupported(
        self,
        spelling: str,
        code: str,
        message: str,
        *,
        source: Mapping[str, Any] | None = None,
        parents: Sequence[tuple[int, str]] = (),
    ) -> int:
        self.shortfalls.append(JavaScriptExpressionShortfall(code, message))
        return self.make_node(
            "Untranslated",
            None,
            str(spelling),
            parents,
            attributes={"spelling": spelling, "shortfall_code": code},
            source=source,
        )

    def _facets(self, operands: Sequence[int]) -> frozenset[str]:
        """The caller's assumptions, corrected by the operands' proved kinds.

        A blanket assumption speaks only for values whose kind is unknown.
        "the node oracle: x && y on two numbers": with "boolean-operands"
        assumed for the region, ``&&`` lowered to LAnd and returned a bool,
        where JavaScript returns one of the two Numbers.  A proved kind
        therefore grants the facet it proves and revokes the ones it
        contradicts.
        """

        facets = set(self.assumptions)
        kinds = [self._kind(operand) for operand in operands]
        known = {kind for kind in kinds if kind != "unknown"}
        for facet, kind in (
            ("number-operands", "number"), ("boolean-operands", "boolean"),
        ):
            if known - {kind}:
                facets.discard(facet)
            elif known and len(known) == 1 and "unknown" not in kinds:
                facets.add(facet)
        if len(known) > 1:
            facets.discard("primitive-same-type-operands")
        elif known and "unknown" not in kinds and known <= {"number", "boolean"}:
            facets.add("primitive-same-type-operands")
        return frozenset(facets)

    def require(
        self,
        requirements: Sequence[str],
        *,
        spelling: str,
        code: str = "unproven-javascript-semantics",
        operands: Sequence[int] | None = None,
    ) -> bool:
        facets = (
            self.assumptions if operands is None else self._facets(operands)
        )
        missing = sorted(set(requirements) - facets)
        if not missing:
            return True
        self.shortfalls.append(JavaScriptExpressionShortfall(
            code,
            f"{spelling!r} requires proven facets: {', '.join(missing)}",
        ))
        return False

    def _post_binding(self, record: Mapping[str, Any]) -> Any:
        positions = (
            record.get("start"), None, record.get("end"), None,
        )
        context = json.dumps(
            {
                "owner": self.owner,
                "name": record["name"],
                "kind": record["kind"],
                "start": record.get("start"),
                "end": record.get("end"),
            },
            sort_keys=True,
        )
        return current_identity_book().post(
            NAME_BINDING,
            (self.scope, str(record["name"]), int(record["version"])),
            BindingFact(
                int(record["value_node"]),
                bool(record["kind"] != "free"),
                positions,
                hashlib.sha256(context.encode("utf-8")).hexdigest(),
            ),
            stage=INGESTION,
            provenance=Derived((
                node_identity_cell(self.graph, int(record["value_node"])),
            )),
            mode=Mode.CONCORD,
        )

    def bind(
        self,
        name: str,
        value_node: int,
        declaration: Mapping[str, Any],
        *,
        kind: str,
    ) -> dict[str, Any]:
        if name in self.bindings:
            self.shortfalls.append(JavaScriptExpressionShortfall(
                "duplicate-binding",
                f"binding {name!r} is already declared in this JavaScript region",
            ))
            return self.bindings[name]
        # One name may be declared in sibling scopes (each arm's ``let t``)
        # or read free after a block's own declaration went out of scope;
        # each is its own row on the book.  "if/else arms that both declare
        # let t": the second re-posted version 0 and the book refused it.
        version = self.last_version.get(name, -1) + 1
        self.last_version[name] = version
        record = {
            "declaration_identity": (
                f"{self.owner}#javascript-binding:{kind}:{name}:"
                f"{declaration.get('start', '?')}:{declaration.get('end', '?')}"
            ),
            "name": name,
            "kind": kind,
            "version": version,
            "value_node": int(value_node),
            "start": declaration.get("start"),
            "end": declaration.get("end"),
        }
        record["binding_cell"] = self._post_binding(record)
        self.bindings[name] = record
        self.binding_records.append(record)
        return record

    def read_identifier(self, node: Mapping[str, Any]) -> int:
        name = str(node.get("name", ""))
        binding = self.bindings.get(name)
        if binding is None:
            value_node = self.make_node(
                "Input",
                Handler.Load.name,
                name,
                (),
                attributes={"binding_name": name, "binding_kind": "free"},
                source=node,
                output_roles=("value",),
            )
            binding = self.bind(name, value_node, node, kind="free")
        self.identifier_occurrences.append({
            "name": name,
            "start": node.get("start"),
            "end": node.get("end"),
            "binding_cell": binding["binding_cell"],
            "version": binding["version"],
            "value_node": binding["value_node"],
        })
        return int(binding["value_node"])

    def _kind(self, node_id: int) -> str:
        return str(
            (self.graph.G.nodes[node_id].get("attributes") or {}).get(
                "javascript_value_kind", "unknown",
            )
        )

    def _number_falsy(
        self, operand: int, source: Mapping[str, Any],
    ) -> int:
        zero = self.make_node(
            "Constant", Handler.Const.name, "0", (),
            attributes={"value": 0.0}, constant=0.0, source=source,
            output_roles=("value",), value_kind="number",
        )
        equals_zero = self.make_node(
            Handler.Eq.name, Handler.Eq.name, Handler.Eq.name,
            ((operand, "left"), (zero, "right")), source=source,
            value_kind="boolean",
        )
        is_nan = self.make_node(
            "Call", Handler.Call.name, "isnan", ((operand, "arg0"),),
            attributes={
                "callee": "isnan", "tensor_operation": "isnan",
                "javascript_identity": "javascript.realm.Number.isNaN",
            },
            source=source,
            value_kind="boolean",
        )
        return self.make_node(
            Handler.LOr.name, Handler.LOr.name, Handler.LOr.name,
            ((equals_zero, "left"), (is_nan, "right")), source=source,
            value_kind="boolean",
        )

    def _node(
        self,
        handler: Handler,
        parents: Sequence[tuple[int, str]],
        source: Mapping[str, Any],
        kind: str,
    ) -> int:
        return self.make_node(
            handler.name, handler.name, handler.name, parents,
            source=source, value_kind=kind,
        )

    def _call(
        self, operation: str, operand: int, source: Mapping[str, Any], kind: str,
    ) -> int:
        return self.make_node(
            "Call", Handler.Call.name, operation, ((operand, "arg0"),),
            attributes={"callee": operation, "tensor_operation": operation},
            source=source, value_kind=kind,
        )

    def _boolean(self, value: bool, source: Mapping[str, Any]) -> int:
        return self.make_node(
            "Constant", Handler.Const.name, repr(bool(value)), (),
            attributes={"value": bool(value)}, constant=bool(value),
            source=source, output_roles=("value",), value_kind="boolean",
        )

    def _number(self, value: float, source: Mapping[str, Any]) -> int:
        return self.make_node(
            "Constant", Handler.Const.name, repr(value), (),
            attributes={"value": float(value)}, constant=float(value),
            source=source, output_roles=("value",), value_kind="number",
        )

    def _negative_sign(self, value: int, source: Mapping[str, Any]) -> int:
        """True for negative numbers and -0 (``1 / -0`` is -Infinity)."""

        zero = self._number(0.0, source)
        below = self._node(
            Handler.Lt, ((value, "left"), (zero, "right")), source, "boolean",
        )
        is_zero = self._node(
            Handler.Eq, ((value, "left"), (zero, "right")), source, "boolean",
        )
        reciprocal = self._node(
            Handler.Div, ((self._number(1.0, source), "left"), (value, "right")),
            source, "number",
        )
        negative_zero = self._node(
            Handler.LAnd,
            ((is_zero, "left"), (self._node(
                Handler.Lt, ((reciprocal, "left"), (zero, "right")),
                source, "boolean",
            ), "right")),
            source, "boolean",
        )
        return self._node(
            Handler.LOr, ((below, "left"), (negative_zero, "right")),
            source, "boolean",
        )

    def _remainder(
        self, dividend: int, divisor: int, source: Mapping[str, Any],
    ) -> int:
        """ECMAScript Number::remainder: truncated, with the dividend's sign.

        The repository Mod is floored (Python's ``%``).  On non-negative
        operands floored and truncated agree, and the remainder of
        magnitudes is exact, so ``|a| mod |b|`` carries the dividend's sign
        back, -0 included.  NaN, an infinite dividend and a zero divisor
        give NaN through Mod itself; an infinite divisor returns ``|a|``.
        """

        magnitude = self._node(
            Handler.Mod,
            ((self._call("abs", dividend, source, "number"), "left"),
             (self._call("abs", divisor, source, "number"), "right")),
            source, "number",
        )
        negated = self._node(
            Handler.Neg, ((magnitude, "operand"),), source, "number",
        )
        return self._node(
            Handler.Select,
            ((self._negative_sign(dividend, source), "condition"),
             (negated, "if_true"), (magnitude, "if_false")),
            source, "number",
        )

    def _sign(self, value: int, source: Mapping[str, Any]) -> int:
        """Math.sign: 1, -1, or the operand itself for +-0 and NaN."""

        zero = self._number(0.0, source)
        below = self._node(
            Handler.Select,
            ((self._node(Handler.Lt, ((value, "left"), (zero, "right")),
                         source, "boolean"), "condition"),
             (self._number(-1.0, source), "if_true"), (value, "if_false")),
            source, "number",
        )
        return self._node(
            Handler.Select,
            ((self._node(Handler.Gt, ((value, "left"), (zero, "right")),
                         source, "boolean"), "condition"),
             (self._number(1.0, source), "if_true"), (below, "if_false")),
            source, "number",
        )

    def _extremum(
        self, left: int, right: int, source: Mapping[str, Any], *, largest: bool,
    ) -> int:
        """Math.max / Math.min of two Numbers.

        Any NaN gives NaN, and among equal zeros max prefers +0 and min
        -0 -- C's fmax/fmin promise neither.  Equal zeros are resolved by
        addition (``-0 + -0`` is the only sum that is -0) for max, and by
        its mirror ``-((-a) + (-b))`` for min.
        """

        order = Handler.Gt if largest else Handler.Lt
        first = self._node(order, ((left, "left"), (right, "right")), source, "boolean")
        second = self._node(order, ((right, "left"), (left, "right")), source, "boolean")
        either_nan = self._node(
            Handler.LOr,
            ((self._call("isnan", left, source, "boolean"), "left"),
             (self._call("isnan", right, source, "boolean"), "right")),
            source, "boolean",
        )
        if largest:
            zeros = self._node(
                Handler.Add, ((left, "left"), (right, "right")), source, "number",
            )
        else:
            mirrored = self._node(
                Handler.Add,
                ((self._node(Handler.Neg, ((left, "operand"),), source, "number"), "left"),
                 (self._node(Handler.Neg, ((right, "operand"),), source, "number"), "right")),
                source, "number",
            )
            zeros = self._node(
                Handler.Neg, ((mirrored, "operand"),), source, "number",
            )
        equal = self._node(
            Handler.Select,
            ((self._node(Handler.Eq, ((left, "left"), (self._number(0.0, source), "right")),
                         source, "boolean"), "condition"),
             (zeros, "if_true"), (left, "if_false")),
            source, "number",
        )
        unordered = self._node(
            Handler.Select,
            ((either_nan, "condition"), (self._number(float("nan"), source), "if_true"),
             (equal, "if_false")),
            source, "number",
        )
        inner = self._node(
            Handler.Select,
            ((second, "condition"), (right, "if_true"), (unordered, "if_false")),
            source, "number",
        )
        return self._node(
            Handler.Select,
            ((first, "condition"), (left, "if_true"), (inner, "if_false")),
            source, "number",
        )

    def _exponentiate(
        self, base: int, exponent: int, source: Mapping[str, Any],
    ) -> int:
        """ECMAScript Number::exponentiate (``**`` and ``Math.pow``).

        "the node oracle: 1 ** NaN" -- C's pow, which the repository Pow
        lowers to, returns 1 for a base of +-1 with a NaN or infinite
        exponent; ECMAScript returns NaN.  Every other special case agrees,
        so those two are selected explicitly and the rest is Pow.  Pow
        itself is implementation-approximated (``ecmascript_approximated``).
        """

        power = self.make_node(
            Handler.Pow.name, Handler.Pow.name, Handler.Pow.name,
            ((base, "left"), (exponent, "right")), source=source,
            attributes={"ecmascript_approximated": True},
            value_kind="number",
        )
        exponent_nan = self.make_node(
            "Call", Handler.Call.name, "isnan", ((exponent, "arg0"),),
            attributes={"callee": "isnan", "tensor_operation": "isnan"},
            source=source, value_kind="boolean",
        )
        magnitude = self.make_node(
            "Call", Handler.Call.name, "abs", ((base, "arg0"),),
            attributes={"callee": "abs", "tensor_operation": "abs"},
            source=source, value_kind="number",
        )
        one = self.make_node(
            "Constant", Handler.Const.name, "1", (),
            attributes={"value": 1.0}, constant=1.0, source=source,
            output_roles=("value",), value_kind="number",
        )
        unit_base = self.make_node(
            Handler.Eq.name, Handler.Eq.name, Handler.Eq.name,
            ((magnitude, "left"), (one, "right")), source=source,
            value_kind="boolean",
        )
        exponent_infinite = self.make_node(
            "Call", Handler.Call.name, "isinf", ((exponent, "arg0"),),
            attributes={"callee": "isinf", "tensor_operation": "isinf"},
            source=source, value_kind="boolean",
        )
        unit_infinite = self.make_node(
            Handler.LAnd.name, Handler.LAnd.name, Handler.LAnd.name,
            ((unit_base, "left"), (exponent_infinite, "right")),
            source=source, value_kind="boolean",
        )
        undefined = self.make_node(
            Handler.LOr.name, Handler.LOr.name, Handler.LOr.name,
            ((exponent_nan, "left"), (unit_infinite, "right")),
            source=source, value_kind="boolean",
        )
        nan = self.make_node(
            "Constant", Handler.Const.name, "NaN", (),
            attributes={"value": float("nan")}, constant=float("nan"),
            source=source, output_roles=("value",), value_kind="number",
        )
        return self.make_node(
            Handler.Select.name, Handler.Select.name, Handler.Select.name,
            ((undefined, "condition"), (nan, "if_true"), (power, "if_false")),
            source=source, value_kind="number",
        )

    def truthy(self, operand: int, source: Mapping[str, Any]) -> int:
        kind = self._kind(operand)
        if kind == "boolean" or (
            kind == "unknown" and "boolean-operands" in self.assumptions
        ):
            return operand
        if kind == "number" and self.require(
            ("number-operands", "number-truthiness", "pure-operands"),
            spelling="Number truthiness",
            operands=(operand,),
        ):
            falsy = self._number_falsy(operand, source)
            return self.make_node(
                Handler.LNot.name, Handler.LNot.name, Handler.LNot.name,
                ((falsy, "operand"),), source=source, value_kind="boolean",
            )
        return self.unsupported(
            "truthiness",
            "unproven-javascript-truthiness",
            "condition has no proven JavaScript truthiness program",
            source=source,
            parents=((operand, "value"),),
        )

    def _number_logical(
        self, operator: str, left: int, right: int, expr: Mapping[str, Any],
    ) -> int:
        """``&&`` / ``||`` on two Numbers: one of the operands, not a bool.

        ``a && b`` is ``a`` when ``a`` is falsy (``0``, ``-0``, ``NaN``),
        else ``b``; ``a || b`` is ``a`` when truthy, else ``b``.  With pure
        operands both are evaluated and Selected by ``a``'s truthiness, so
        ``-0 && 5`` is ``-0`` and ``NaN || 0`` is ``0``.  An operand pair of
        different kinds has no single result kind and stays refused.
        """

        parents = ((left, "left"), (right, "right"))
        if not self.require(
            ("number-operands", "number-truthiness", "pure-operands"),
            spelling=operator, operands=(left, right),
        ):
            return self.unsupported(
                operator,
                "untranslated-binary-operator",
                f"{operator!r} has unproved JavaScript semantics",
                source=expr,
                parents=parents,
            )
        truthy = self.truthy(left, expr)
        chosen, other = (right, left) if operator == "&&" else (left, right)
        return self._node(
            Handler.Select,
            ((truthy, "condition"), (chosen, "if_true"), (other, "if_false")),
            expr, "number",
        )

    def binary(
        self, operator: str, left: int, right: int,
        expr: Mapping[str, Any],
    ) -> int:
        """One JavaScript binary operator on two interpreted operands.

        Shared by binary expressions and compound assignment (``x += y`` is
        ``x = x + y`` on the binding's current version).
        """

        parents = ((left, "left"), (right, "right"))
        if operator in {"&&", "||"} and (
            self._kind(left) == self._kind(right) == "number"
        ):
            return self._number_logical(operator, left, right, expr)
        if operator == "%":
            if not self.require(
                ("number-operands", "ecmascript-number-semantics"),
                spelling=operator, operands=(left, right),
            ):
                return self.unsupported(
                    operator,
                    "untranslated-binary-operator",
                    f"{operator!r} has unproved JavaScript semantics",
                    source=expr,
                    parents=parents,
                )
            return self._remainder(left, right, expr)
        handler = JAVASCRIPT_BINARY_TO_SSA.get(operator)
        if handler is None:
            reason = (
                "needs a JavaScript-specific reduction program"
                if operator in JAVASCRIPT_UNSUPPORTED_BINARY
                else "is not a recognized JavaScript binary operator"
            )
            return self.unsupported(
                operator,
                "unsupported-binary-operator",
                f"{operator!r} {reason}",
                source=expr,
                parents=parents,
            )
        if not self.require(
            _BINARY_REQUIREMENTS.get(operator, ()), spelling=operator,
            operands=(left, right),
        ):
            return self.unsupported(
                operator,
                "untranslated-binary-operator",
                f"{operator!r} has unproved JavaScript semantics",
                source=expr,
                parents=parents,
            )
        if handler is Handler.Pow:
            return self._exponentiate(left, right, expr)
        result_kind = (
            "boolean"
            if handler in {
                Handler.Eq, Handler.Ne, Handler.Lt, Handler.Le,
                Handler.Gt, Handler.Ge, Handler.LAnd, Handler.LOr,
            }
            else "number"
        )
        return self.make_node(
            handler.name, handler.name, handler.name, parents,
            source=expr, value_kind=result_kind,
        )

    def expression(self, expr: Mapping[str, Any]) -> int:
        kind = expr.get("type")
        if kind in ("BinaryExpression", "LogicalExpression"):
            operator = str(expr.get("operator"))
            left = self.expression(expr["left"])
            right = self.expression(expr["right"])
            return self.binary(operator, left, right, expr)
        if kind == "UnaryExpression":
            operator = str(expr.get("operator"))
            operand = self.expression(expr["argument"])
            if operator == "!":
                operand_kind = self._kind(operand)
                if operand_kind == "number":
                    if not self.require(
                        ("number-operands", "number-truthiness", "pure-operands"),
                        spelling="!",
                        operands=(operand,),
                    ):
                        return self.unsupported(
                            operator,
                            "unproven-javascript-semantics",
                            "number negation requires JavaScript Number truthiness",
                            source=expr,
                            parents=((operand, "operand"),),
                        )
                    return self._number_falsy(operand, expr)
                if self.require(
                    ("boolean-operands",), spelling="!", operands=(operand,),
                ):
                    return self.make_node(
                        Handler.LNot.name, Handler.LNot.name, Handler.LNot.name,
                        ((operand, "operand"),), source=expr,
                        value_kind="boolean",
                    )
            handler = JAVASCRIPT_UNARY_TO_SSA.get(operator)
            if handler is None:
                reason = (
                    "needs a JavaScript-specific reduction program"
                    if operator in JAVASCRIPT_UNSUPPORTED_UNARY
                    else "is not a recognized JavaScript unary operator"
                )
                return self.unsupported(
                    operator,
                    "unsupported-unary-operator",
                    f"{operator!r} {reason}",
                    source=expr,
                    parents=((operand, "operand"),),
                )
            if not self.require(
                ("number-operands", "ecmascript-number-semantics"),
                spelling=operator,
                operands=(operand,),
            ):
                return self.unsupported(
                    operator,
                    "untranslated-unary-operator",
                    f"{operator!r} has unproved JavaScript semantics",
                    source=expr,
                    parents=((operand, "operand"),),
                )
            return self.make_node(
                handler.name, handler.name, handler.name,
                ((operand, "operand"),), source=expr, value_kind="number",
            )
        if kind == "ConditionalExpression":
            condition_value = self.expression(expr["test"])
            condition = self.truthy(condition_value, expr["test"])
            yes = self.expression(expr["consequent"])
            no = self.expression(expr["alternate"])
            if self._kind(yes) != self._kind(no):
                return self.unsupported(
                    "?:",
                    "unproved-conditional-result",
                    "conditional result arms have different JavaScript value kinds",
                    source=expr,
                    parents=(
                        (condition, "condition"), (yes, "if_true"),
                        (no, "if_false"),
                    ),
                )
            return self.make_node(
                Handler.Select.name, Handler.Select.name, Handler.Select.name,
                ((condition, "condition"), (yes, "if_true"),
                 (no, "if_false")),
                source=expr,
                value_kind=self._kind(yes),
            )
        if kind == "CallExpression":
            parts = _callee_parts(expr.get("callee"))
            spelling = ".".join(parts)
            arguments = [
                self.expression(argument) for argument in expr.get("arguments", ())
            ]
            parents = tuple(
                (argument, f"arg{index}")
                for index, argument in enumerate(arguments)
            )
            if parts and parts[0] in self.bindings:
                return self.unsupported(
                    spelling,
                    "shadowed-javascript-realm",
                    f"{parts[0]!r} is a lexical binding, not a realm intrinsic",
                    source=expr,
                    parents=parents,
                )
            identity = JAVASCRIPT_SURFACE_IDENTITIES.get(spelling)
            program = resolve_javascript_identity(identity)
            if program is None:
                reason = JAVASCRIPT_UNSUPPORTED_CALLS.get(spelling)
                return self.unsupported(
                    spelling,
                    "unsupported-call" if reason else "unresolved-call",
                    reason or f"{spelling!r} has no grounded JavaScript identity",
                    source=expr,
                    parents=parents,
                )
            expected = int(program.attributes.get("javascript_arity", -1))
            if expected != len(arguments):
                return self.unsupported(
                    spelling,
                    "unsupported-call-arity",
                    f"{spelling!r} requires {expected} argument(s); got {len(arguments)}",
                    source=expr,
                    parents=parents,
                )
            if not self.require(
                program.requirements,
                spelling=spelling,
                code="unproven-javascript-identity",
                operands=arguments,
            ):
                return self.unsupported(
                    spelling,
                    "untranslated-call",
                    f"{spelling!r} identity requirements are unproved",
                    source=expr,
                    parents=parents,
                )
            if program.operator == "pow":
                return self._exponentiate(arguments[0], arguments[1], expr)
            if program.operator in {"maximum", "minimum"}:
                return self._extremum(
                    arguments[0], arguments[1], expr,
                    largest=program.operator == "maximum",
                )
            if program.operator == "sign":
                return self._sign(arguments[0], expr)
            attrs = {
                "callee": program.operator,
                "tensor_operation": program.operator,
                "javascript_identity": identity,
                "javascript_spelling": spelling,
                "javascript_requirements": tuple(sorted(program.requirements)),
                **dict(program.attributes),
            }
            return self.make_node(
                "Call", Handler.Call.name, str(program.operator), parents,
                attributes=attrs, source=expr, value_kind=(
                    "boolean" if program.operator == "isnan" else "number"
                ),
            )
        if kind == "AssignmentExpression":
            right = self.expression(expr.get("right") or {})
            return self.unsupported(
                str(expr.get("operator")),
                "unsupported-assignment",
                "mutable JavaScript bindings require the CFG/Phi region",
                source=expr,
                parents=((right, "right"),),
            )
        if kind == "Identifier":
            return self.read_identifier(expr)
        if kind == "Literal":
            value = expr.get("value")
            if value is None or isinstance(value, str):
                return self.unsupported(
                    repr(value),
                    "unsupported-literal",
                    "null and string storage are outside the numeric region",
                    source=expr,
                )
            value_kind = "boolean" if isinstance(value, bool) else "number"
            # A JavaScript Number is an IEEE double.  Acorn's JSON hands an
            # integral literal over as a Python int, which the repository's
            # scalar typing reads as int64: ``1 / 2`` and ``2 ** 64`` would
            # then lower as integer arithmetic.  The literal's value is the
            # double, so the constant carries the double.
            if value_kind == "number":
                value = float(value)
            return self.make_node(
                "Constant", Handler.Const.name, repr(value), (),
                attributes={"value": value}, constant=value, source=expr,
                output_roles=("value",), value_kind=value_kind,
            )
        return self.unsupported(
            str(kind),
            "unsupported-node-type",
            f"{kind!r} has no JavaScript ProcessGraph translation",
            source=expr,
        )

    @staticmethod
    def _statement_body(
        node: Mapping[str, Any] | None,
    ) -> list[Mapping[str, Any]]:
        if not node:
            return []
        if node.get("type") == "BlockStatement":
            return list(node.get("body", ()))
        return [node]

    def assign(
        self,
        name: str,
        value_node: int,
        source: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """A new version of an existing ``let`` or parameter binding.

        A JavaScript assignment rebinds the name; the region stays acyclic
        because the new value is a new node and later reads see the new
        version.  ``const`` and free names are refused, as JavaScript would
        throw (TypeError / ReferenceError in strict code).
        """

        binding = self.bindings.get(name)
        if binding is None or binding["kind"] not in {"let", "parameter"}:
            self.shortfalls.append(JavaScriptExpressionShortfall(
                "unsupported-assignment-target",
                f"{name!r} is "
                + ("not a binding of this region" if binding is None
                   else f"a {binding['kind']} binding")
                + "; only let and parameter bindings are reassignable",
            ))
            return None
        # Versions are numbered per name across the whole region, so the two
        # arms of an if never post different values under one version.
        version = self.last_version.get(name, int(binding["version"])) + 1
        self.last_version[name] = version
        record = {
            **binding,
            "version": version,
            "value_node": int(value_node),
            "start": source.get("start"),
            "end": source.get("end"),
        }
        record["binding_cell"] = self._post_binding(record)
        self.bindings[name] = record
        self.binding_records.append(record)
        return record

    def assignment_statement(self, expr: Mapping[str, Any]) -> None:
        """``x = e``, ``x op= e``, ``x++``/``x--`` as a whole statement."""

        kind = expr.get("type")
        target = (
            expr.get("left") if kind == "AssignmentExpression"
            else expr.get("argument")
        ) or {}
        if target.get("type") != "Identifier":
            self.unsupported(
                str(target.get("type")),
                "unsupported-assignment-target",
                "only plain names are assignable in the numeric region",
                source=expr,
            )
            return
        name = str(target["name"])
        if kind == "UpdateExpression":
            current = self.read_identifier(target)
            one = self.make_node(
                "Constant", Handler.Const.name, "1", (),
                attributes={"value": 1.0}, constant=1.0, source=expr,
                output_roles=("value",), value_kind="number",
            )
            operator = "+" if expr.get("operator") == "++" else "-"
            value = self.binary(operator, current, one, expr)
        else:
            operator = str(expr.get("operator"))
            right = self.expression(expr.get("right") or {})
            if operator == "=":
                value = right
            elif operator.endswith("=") and (
                operator[:-1] in JAVASCRIPT_BINARY_TO_SSA or operator == "%="
            ):
                if operator[:-1] in {"&&", "||"}:
                    # x &&= y assigns only when x is truthy: a branch, not
                    # a binary operator.
                    value = self.unsupported(
                        operator,
                        "unsupported-assignment",
                        f"{operator!r} is a conditional assignment",
                        source=expr,
                        parents=((right, "right"),),
                    )
                else:
                    value = self.binary(
                        operator[:-1], self.read_identifier(target), right, expr,
                    )
            else:
                value = self.unsupported(
                    operator,
                    "unsupported-assignment",
                    f"{operator!r} has no JavaScript translation here",
                    source=expr,
                    parents=((right, "right"),),
                )
        self.assign(name, value, expr)

    def _branch(
        self,
        statements: Sequence[Mapping[str, Any]],
        outer: Mapping[str, dict[str, Any]],
    ) -> tuple[int | None, bool, dict[str, dict[str, Any]]]:
        """Interpret one arm in its own block scope.

        Returns the arm's value and whether every path returned, and the
        arm's final versions of the bindings visible outside it.  Names the
        arm declares are dropped: ``let``/``const`` are block scoped.
        """

        self.bindings = dict(outer)
        value, returned = self.sequence(statements)
        visible = {name: self.bindings[name] for name in outer}
        return value, returned, visible

    def _select(
        self,
        condition: int,
        yes: int,
        no: int,
        source: Mapping[str, Any],
        *,
        control: str,
    ) -> int:
        if self._kind(yes) != self._kind(no):
            return self.unsupported(
                "if",
                "unproved-conditional-result",
                f"{control} arms have different JavaScript value kinds",
                source=source,
                parents=(
                    (condition, "condition"), (yes, "if_true"),
                    (no, "if_false"),
                ),
            )
        return self.make_node(
            Handler.Select.name, Handler.Select.name, Handler.Select.name,
            ((condition, "condition"), (yes, "if_true"), (no, "if_false")),
            attributes={"javascript_control": control},
            source=source,
            value_kind=self._kind(yes),
        )

    @staticmethod
    def _assigned_names(node: Any) -> set[str]:
        """Names a statement (re)assigns, not looking into nested functions."""

        names: set[str] = set()

        def visit(item: Any) -> None:
            if isinstance(item, Mapping):
                kind = item.get("type")
                if kind in {
                    "FunctionDeclaration", "FunctionExpression",
                    "ArrowFunctionExpression",
                }:
                    return
                target = (
                    item.get("left") if kind == "AssignmentExpression"
                    else item.get("argument") if kind == "UpdateExpression"
                    else None
                )
                if isinstance(target, Mapping) and target.get("type") == "Identifier":
                    names.add(str(target.get("name")))
                for key, child in item.items():
                    if key not in {"type", "start", "end"}:
                        visit(child)
            elif isinstance(item, list):
                for child in item:
                    visit(child)

        visit(node)
        return names

    @staticmethod
    def _loop_exit(node: Any) -> str | None:
        """The first return/break/continue/label inside a loop body."""

        def visit(item: Any) -> str | None:
            if isinstance(item, Mapping):
                kind = item.get("type")
                if kind in {
                    "FunctionDeclaration", "FunctionExpression",
                    "ArrowFunctionExpression",
                }:
                    return None
                if kind in {
                    "ReturnStatement", "BreakStatement", "ContinueStatement",
                    "LabeledStatement",
                }:
                    return str(kind)
                for key, child in item.items():
                    if key not in {"type", "start", "end"}:
                        found = visit(child)
                        if found:
                            return found
            elif isinstance(item, list):
                for child in item:
                    found = visit(child)
                    if found:
                        return found
            return None

        return visit(node)

    def _own_exits(self, node: Any, label: str | None = None) -> set[Any]:
        """Exits that leave THIS loop's iteration (the loop labeled ``label``).

        ``break``/``continue`` of nested loops are theirs, but a ``return``
        anywhere inside (nested loops included) leaves this loop too, as
        does an exit labeled with this loop's label.  An exit labeled with
        an enclosing loop's label (``self.loop_labels``) leaves this loop
        and continues or breaks that one: it is reported as
        ``(kind, label)``.  Any other labeled exit is ``LabeledExit``.
        Functions are opaque.
        """

        kinds: set[Any] = set()

        def visit(item: Any, nested: bool, inner_labels: frozenset) -> None:
            if isinstance(item, Mapping):
                kind = item.get("type")
                if kind in _FUNCTION_NODES:
                    return
                if kind == "ReturnStatement":
                    kinds.add("ReturnStatement")
                elif kind in {"BreakStatement", "ContinueStatement"}:
                    target = (item.get("label") or {}).get("name")
                    if target is None:
                        if not nested:
                            kinds.add(str(kind))
                    elif target == label:
                        kinds.add(str(kind))
                    elif target in inner_labels:
                        pass
                    elif target in self.loop_labels:
                        kinds.add((str(kind), str(target)))
                    else:
                        kinds.add("LabeledExit")
                    return
                if kind == "LabeledStatement":
                    inner_labels = inner_labels | {
                        (item.get("label") or {}).get("name")
                    }
                inner = nested or kind in _OPAQUE_TO_LOOP_EXITS
                for key, child in item.items():
                    if key not in {"type", "start", "end", "label"}:
                        visit(child, inner, inner_labels)
            elif isinstance(item, list):
                for child in item:
                    visit(child, nested, inner_labels)

        visit(node, False, frozenset())
        return kinds

    def _static_kind(
        self,
        expr: Mapping[str, Any],
        declared: Mapping[str, Mapping[str, Any]] | None = None,
        depth: int = 0,
    ) -> str | None:
        """The JavaScript value kind ``expr`` will have, read from its shape.

        Used only to type a hidden loop result before the loop is
        interpreted; a wrong guess is caught by the carried-kind check and
        refused, never miscompiled.
        """

        kind = expr.get("type")
        if kind == "Literal":
            value = expr.get("value")
            return (
                "boolean" if isinstance(value, bool)
                else "number" if isinstance(value, (int, float)) else None
            )
        if depth > 16:
            return None
        declared = declared or {}

        def of(child: Mapping[str, Any]) -> str | None:
            return self._static_kind(child, declared, depth + 1)

        if kind == "Identifier":
            name = str(expr.get("name"))
            if name in declared:
                return of(declared[name])
            record = self.bindings.get(name)
            if record is None:
                return "number" if name in {"NaN", "Infinity"} else None
            found = self._kind(int(record["value_node"]))
            return found if found in {"number", "boolean"} else None
        if kind == "UnaryExpression":
            return "boolean" if expr.get("operator") == "!" else "number"
        if kind == "BinaryExpression":
            return (
                "boolean"
                if expr.get("operator") in {
                    "<", "<=", ">", ">=", "===", "!==", "==", "!=",
                }
                else "number"
            )
        if kind == "LogicalExpression":
            left = of(expr["left"])
            return left if left == of(expr["right"]) else None
        if kind == "ConditionalExpression":
            yes = of(expr["consequent"])
            return yes if yes == of(expr["alternate"]) else None
        if kind == "CallExpression":
            spelling = _callee_spelling(expr.get("callee"))
            if spelling in {"Number.isNaN", "Number.isFinite", "isNaN", "isFinite"}:
                return "boolean"
            return "number" if spelling.startswith("Math.") else None
        return None

    def _exit_flags(
        self,
        statements: Sequence[Mapping[str, Any]],
        flags: Mapping[Any, str],
        label: str | None = None,
    ) -> list[Mapping[str, Any]]:
        """Rewrite ``break``/``continue`` as assignments to hidden flags.

        ``break`` sets ``flags["BreakStatement"]``, ``continue`` sets
        ``flags["ContinueStatement"]``; the rest of every statement list
        after a statement that may set one runs only while neither is set
        (an ``if`` on the flags around it, in its own block: the names it
        declares end with the list anyway).  Nested loops keep theirs.
        An exit labeled with an enclosing loop's label sets that loop's
        flag (``flags[(kind, label)]``), which this loop also stops on.
        """

        def again(items: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
            return self._exit_flags(items, flags, label)

        lowered: list[Mapping[str, Any]] = []
        for index, statement in enumerate(statements):
            kind = statement.get("type")
            if kind == "ReturnStatement" and "ReturnStatement" in flags:
                if statement.get("argument") is None:
                    # Left for ``sequence`` to refuse (undefined result).
                    lowered.append(statement)
                else:
                    lowered.append(_flag_assignment(
                        flags["%result"], statement, statement["argument"],
                    ))
                    lowered.append(
                        _flag_assignment(flags["ReturnStatement"], statement)
                    )
            elif kind in {"BreakStatement", "ContinueStatement"}:
                target = (statement.get("label") or {}).get("name")
                key = kind if target in {None, label} else (kind, target)
                lowered.append(_flag_assignment(flags[key], statement))
            elif _loop_of(statement) is not None and (
                "ReturnStatement" in flags
                and "ReturnStatement" in self._own_exits(statement, label)
            ):
                # The inner loop's own return check, rewritten into ours.
                lowered.append({**statement, "%returns_checked": True})
                lowered.extend(again([_return_check(_loop_of(statement))]))
            elif kind == "IfStatement":
                lowered.append({
                    **statement,
                    "consequent": {
                        "type": "BlockStatement",
                        "body": again(
                            self._statement_body(statement.get("consequent")),
                        ),
                    },
                    "alternate": None if statement.get("alternate") is None else {
                        "type": "BlockStatement",
                        "body": again(
                            self._statement_body(statement.get("alternate")),
                        ),
                    },
                })
            elif kind == "BlockStatement":
                lowered.append({
                    **statement,
                    "body": again(list(statement.get("body", ()))),
                })
            else:
                lowered.append(statement)
            if kind in {
                "BreakStatement", "ContinueStatement", "ReturnStatement",
            } or self._own_exits(statement, label):
                rest = again(statements[index + 1:])
                if rest:
                    lowered.append({
                        "type": "IfStatement",
                        "test": _flags_clear(
                            (
                                name for key, name in flags.items()
                                if key != "%result"
                            ),
                            statement,
                        ),
                        "consequent": {"type": "BlockStatement", "body": rest},
                        "alternate": None,
                    })
                break
        return lowered

    def loop(
        self, statement: Mapping[str, Any], label: str | None = None,
    ) -> None:
        """``for``/``while`` as a loop region with one Phi per carried name.

        Every binding the loop assigns that is visible at its head becomes a
        Phi (its value on entry, and its value at the end of the body).  The
        condition and body are interpreted once against the Phis.  The back
        edge is recorded in ``javascript_loops`` -- not as a graph edge, so
        the graph itself stays acyclic -- with the id ranges the header and
        body occupy (node ids are creation order).  After the loop each
        outer name reads its Phi: the loop leaves through the header.

        ``break`` and ``continue`` become hidden boolean flags: a carried
        break flag (the header's condition is ``!broken && test``) and a
        per-iteration continue flag; the statements after an exit run only
        while both are clear, and a ``for`` update only while not broken.
        So the iteration that breaks carries its values at the break to the
        header, which then exits with them.  A ``return`` is a break that
        also sets a hidden carried result (typed from the return arguments'
        shape); ``sequence`` follows the loop with ``if (returned) return
        result``.  A labeled exit to this loop is its own exit; one to an
        enclosing labeled loop sets that loop's flag, and this loop stops on
        it as on its own break.
        """

        kind = statement.get("type")
        own = self._own_exits(statement.get("body"), label)
        outward = {
            item: self.loop_labels[item[1]].get(item[0])
            for item in own if isinstance(item, tuple)
        }
        if None in outward.values():
            own.add("LabeledExit")
        if "LabeledExit" in own or (
            statement.get("test") is None
            and not own & {"BreakStatement", "ReturnStatement"}
            and not outward
        ):
            self.unsupported(
                str(kind),
                "unsupported-loop-exit",
                f"{'a labeled exit' if 'LabeledExit' in own else 'a loop without a condition, break or return'} "
                "leaves the loop other than through its condition",
                source=statement,
            )
            return
        outer = dict(self.bindings)
        self.bindings = dict(outer)
        initializer = statement.get("init")
        if initializer is not None:
            self.sequence([
                initializer if initializer.get("type") == "VariableDeclaration"
                else {"type": "ExpressionStatement", "expression": initializer},
            ])
        loop_id = self.loop_count
        # Keyed by position: a nested loop names an enclosing loop's flags
        # for its labeled exits.
        position = f"{statement.get('start')}:{statement.get('end')}"
        flags: dict[Any, str] = {
            exit: f"%{word}@{position}"
            for exit, word in (
                ("BreakStatement", "break"), ("ContinueStatement", "continue"),
            )
            if exit in own
        }
        if "ReturnStatement" in own:
            result_kind = self._loop_result_kind(statement.get("body"))
            if result_kind is None:
                self.unsupported(
                    str(kind),
                    "unproved-loop-return-kind",
                    "a return inside the loop has no statically known "
                    "JavaScript value kind",
                    source=statement,
                )
                self.bindings = outer
                return
            # Keyed by position: an enclosing loop names the same flags
            # when it rewrites this loop's return check into its own.
            flags["ReturnStatement"], flags["%result"] = _return_names(statement)
            self.bind(
                flags["%result"],
                self._number(0.0, statement) if result_kind == "number"
                else self._boolean(False, statement),
                statement, kind="let",
            )
        own_stops = tuple(
            flags[key] for key in ("BreakStatement", "ReturnStatement")
            if key in flags
        )
        for name in own_stops:
            self.bind(
                name, self._boolean(False, statement), statement, kind="let",
            )
        # An enclosing loop's flag set from in here stops this loop too.
        stops = (*own_stops, *sorted(set(outward.values())))
        guards = {**flags, **outward}
        body = self._statement_body(statement.get("body"))
        if guards:
            body = self._exit_flags(body, guards, label)
        if "ContinueStatement" in flags:
            body = [_flag_declaration(flags["ContinueStatement"], statement), *body]
        update = statement.get("update")
        if update is not None:
            step = {"type": "ExpressionStatement", "expression": update}
            if stops:
                step = {
                    "type": "IfStatement",
                    "test": _flags_clear(stops, update),
                    "consequent": step, "alternate": None,
                }
            body = [*body, step]
        # A nested loop's labeled exit sets this loop's (or an enclosing
        # loop's) flags only once that loop is interpreted, so every flag
        # this loop stops on is carried from the start.
        carried = sorted(
            (self._assigned_names([statement.get("test"), body]) | set(stops))
            & set(self.bindings)
        )
        # Numbered on entry: an inner loop finishes (and is recorded) first.
        self.loop_count += 1
        header_start = self.next_id
        phis: dict[str, int] = {}
        phi_records: dict[str, dict[str, Any]] = {}
        for name in carried:
            initial = int(self.bindings[name]["value_node"])
            phi = self.make_node(
                Handler.Phi.name, Handler.Phi.name, name, ((initial, "init"),),
                attributes={"javascript_loop": loop_id, "binding_name": name},
                source=statement, value_kind=self._kind(initial),
            )
            if self.assign(name, phi, statement) is None:
                self.bindings = outer
                return
            phis[name] = phi
            phi_records[name] = self.bindings[name]
        test = statement.get("test")
        condition = None
        if test is not None:
            condition = self.truthy(self.expression(test), test)
        for stop in reversed(stops):
            running = self._node(
                Handler.LNot, ((self.read_identifier(
                    {"type": "Identifier", "name": stop},
                ), "operand"),), statement, "boolean",
            )
            condition = running if condition is None else self._node(
                Handler.LAnd, ((running, "left"), (condition, "right")),
                statement, "boolean",
            )
        condition_end = self.next_id
        if label is not None:
            self.loop_labels[label] = flags
        try:
            _value, _returned, visible = self._branch(body, dict(self.bindings))
        finally:
            if label is not None:
                self.loop_labels.pop(label, None)
        carried_next = {
            name: int(visible[name]["value_node"]) for name in carried
        }
        for name in carried:
            if self._kind(carried_next[name]) != self._kind(phis[name]):
                self.unsupported(
                    name,
                    "unproved-loop-carried-kind",
                    f"{name!r} changes JavaScript value kind across the loop",
                    source=statement,
                )
        self.loops.append({
            "id": loop_id,
            "header_start": header_start,
            "condition": condition,
            "condition_end": condition_end,
            "body_end": self.next_id,
            "phis": dict(phis),
            "next": carried_next,
        })
        self.bindings = {
            name: phi_records.get(name, record) for name, record in outer.items()
        }
        # The return flag and result stay readable for the return check
        # that follows the loop.
        for key in ("ReturnStatement", "%result"):
            if key in flags and flags[key] in phi_records:
                self.bindings[flags[key]] = phi_records[flags[key]]

    def _loop_result_kind(self, body: Any) -> str | None:
        """The one kind every ``return`` argument inside ``body`` has.

        Names declared inside the loop are typed by their initializers.
        """

        arguments: list[Mapping[str, Any]] = []
        declared: dict[str, Mapping[str, Any]] = {}

        def visit(item: Any) -> None:
            if isinstance(item, Mapping):
                kind = item.get("type")
                if kind in _FUNCTION_NODES:
                    return
                if kind == "ReturnStatement" and item.get("argument") is not None:
                    arguments.append(item["argument"])
                if kind == "VariableDeclarator" and (
                    (item.get("id") or {}).get("type") == "Identifier"
                    and item.get("init") is not None
                ):
                    declared.setdefault(str(item["id"]["name"]), item["init"])
                for key, child in item.items():
                    if key not in {"type", "start", "end"}:
                        visit(child)
            elif isinstance(item, list):
                for child in item:
                    visit(child)

        visit(body)
        saved = self.bindings
        kinds = set()
        try:
            for argument in arguments:
                kind = self._static_kind(argument, declared)
                kinds.add(kind)
        finally:
            self.bindings = saved
        return kinds.pop() if len(kinds) == 1 and None not in kinds else None

    def sequence(
        self, statements: Sequence[Mapping[str, Any]],
    ) -> tuple[int | None, bool]:
        index = 0
        while index < len(statements):
            statement = statements[index]
            kind = statement.get("type")
            if kind == "VariableDeclaration":
                if statement.get("kind") not in {"const", "let"}:
                    self.unsupported(
                        str(statement.get("kind")),
                        "unsupported-binding-kind",
                        "this var can be read while still undefined (before "
                        "its declaration, outside its block, or without an "
                        "initializer); only a var that reads like a let is "
                        "translated",
                        source=statement,
                    )
                    index += 1
                    continue
                for declarator in statement.get("declarations", ()):
                    target = declarator.get("id") or {}
                    initializer = declarator.get("init")
                    if target.get("type") != "Identifier" or initializer is None:
                        self.unsupported(
                            str(target.get("type")),
                            "unsupported-declaration",
                            "destructuring and uninitialized declarations are unsupported",
                            source=declarator,
                        )
                        continue
                    value = self.expression(initializer)
                    self.bind(
                        str(target["name"]), value, target,
                        kind=str(statement.get("kind")),
                    )
                index += 1
                continue
            if kind == "ReturnStatement":
                argument = statement.get("argument")
                if argument is None:
                    return self.unsupported(
                        "return",
                        "unsupported-undefined-return",
                        "JavaScript undefined has no numeric-region value",
                        source=statement,
                    ), True
                return self.expression(argument), True
            if kind == "BlockStatement":
                outer = dict(self.bindings)
                value, returned, visible = self._branch(
                    list(statement.get("body", ())), outer,
                )
                self.bindings = visible
                if returned:
                    return value, True
                index += 1
                continue
            if kind == "IfStatement":
                # Both arms are interpreted; their versions of each outer
                # binding meet in a Select on the condition.  An arm that
                # returns takes the rest of the function with the other arm
                # only: the rest runs on that path alone.
                outer = dict(self.bindings)
                condition_value = self.expression(statement["test"])
                condition = self.truthy(condition_value, statement["test"])
                yes, yes_returns, yes_bindings = self._branch(
                    self._statement_body(statement.get("consequent")), outer,
                )
                no, no_returns, no_bindings = self._branch(
                    self._statement_body(statement.get("alternate")), outer,
                )
                rest = list(statements[index + 1:])
                if yes_returns and not no_returns:
                    self.bindings = no_bindings
                    no, no_returns = self.sequence(rest)
                elif no_returns and not yes_returns:
                    self.bindings = yes_bindings
                    yes, yes_returns = self.sequence(rest)
                if yes_returns and no_returns:
                    self.bindings = outer
                    if yes is None or no is None:
                        return None, False
                    return self._select(
                        condition, yes, no, statement,
                        control="early-return-if",
                    ), True
                if yes_returns or no_returns:
                    self.bindings = outer
                    return self.unsupported(
                        "if",
                        "unsupported-control-flow",
                        "both early-return paths must produce a value",
                        source=statement,
                    ), False
                self.bindings = dict(outer)
                for name, record in outer.items():
                    a = yes_bindings[name]
                    b = no_bindings[name]
                    if a["value_node"] == b["value_node"]:
                        self.bindings[name] = a if a["version"] >= b["version"] else b
                        continue
                    merged = self._select(
                        condition, int(a["value_node"]), int(b["value_node"]),
                        statement, control="if-merge",
                    )
                    self.bindings[name] = record
                    self.assign(name, merged, statement)
                index += 1
                continue
            if kind == "ExpressionStatement":
                expression = statement["expression"]
                if expression.get("type") in {
                    "AssignmentExpression", "UpdateExpression",
                }:
                    self.assignment_statement(expression)
                    index += 1
                    continue
                value = self.expression(expression)
                self.unsupported(
                    "expression-statement",
                    "unsupported-effect",
                    "the numeric region has no statement-level effects",
                    source=statement,
                    parents=((value, "value"),),
                )
                index += 1
                continue
            loop_statement = _loop_of(statement)
            if loop_statement is not None:
                label = None
                if kind == "LabeledStatement":
                    label = str((statement.get("label") or {}).get("name"))
                self.loop(loop_statement, label)
                checked = statement.get("%returns_checked")
                statement = loop_statement
                if not checked and (
                    "ReturnStatement" in self._own_exits(statement)
                ):
                    # A loop that may return is followed by its check;
                    # the rest of the function runs on the other path.
                    test = statement.get("test")
                    endless = "BreakStatement" not in self._own_exits(
                        statement.get("body"),
                    ) and (test is None or (
                        test.get("type") == "Literal" and test.get("value") is True
                    ))
                    if endless:
                        # Left only through a return: nothing after it runs.
                        check = _return_check(statement)["consequent"]["body"]
                        return self.sequence(check)
                    return self.sequence([
                        _return_check(statement), *statements[index + 1:],
                    ])
                index += 1
                continue
            if kind == "EmptyStatement":
                index += 1
                continue
            self.unsupported(
                str(kind),
                "unsupported-statement",
                f"{kind!r} has no JavaScript statement translation",
                source=statement,
            )
            index += 1
        return None, False

    def function(
        self,
        function: Mapping[str, Any],
        *,
        declaration: Mapping[str, Any],
        name: str,
    ) -> int:
        self.entry_identity = (
            f"{self.owner}#javascript-function:{name}:"
            f"{declaration.get('start', '?')}:{declaration.get('end', '?')}"
        )
        for parameter in function.get("params", ()):
            if parameter.get("type") != "Identifier":
                self.unsupported(
                    str(parameter.get("type")),
                    "unsupported-parameter",
                    "destructured/default/rest parameters are unsupported",
                    source=parameter,
                )
                continue
            parameter_name = str(parameter["name"])
            value_node = self.make_node(
                "Input",
                Handler.Load.name,
                parameter_name,
                (),
                attributes={
                    "binding_name": parameter_name,
                    "binding_kind": "parameter",
                    "javascript_function_identity": self.entry_identity,
                },
                source=parameter,
                output_roles=("value",),
                # A declared ABI kind ("number"/"boolean") is the parameter's
                # proof; an undeclared one stays "unknown" and is admitted
                # only through the caller's assumptions, as before.
                value_kind=self.parameter_kinds.get(parameter_name, "unknown"),
            )
            self.bind(parameter_name, value_node, parameter, kind="parameter")
        body = function.get("body") or {}
        if body.get("type") == "BlockStatement":
            value, returned = self.sequence(_var_as_let(
                body.get("body", ()),
                frozenset(
                    str(parameter.get("name"))
                    for parameter in function.get("params", ())
                ),
            ))
        else:
            value, returned = self.expression(body), True
        if not returned or value is None:
            value = self.unsupported(
                name,
                "missing-return",
                f"JavaScript entry {name!r} has no translated return",
                source=function,
            )
        return self.make_node(
            Handler.Ret.name,
            Handler.Ret.name,
            name,
            ((value, "value"),),
            attributes={"javascript_function_identity": self.entry_identity},
            source=function,
            output_roles=(),
            value_kind=self._kind(value),
        )

    def finish(self, root: int) -> JavaScriptExpressionLowering:
        self.graph.roots = [root]
        self.graph.domain_shape = (1,)
        self.graph.G.graph["javascript_entry_identity"] = self.entry_identity
        self.graph.G.graph["javascript_loops"] = tuple(
            dict(record) for record in self.loops
        )
        self.graph.G.graph["javascript_binding_identities"] = tuple(
            dict(record) for record in self.binding_records
        )
        self.graph.G.graph["javascript_identifier_occurrences"] = tuple(
            dict(record) for record in self.identifier_occurrences
        )
        return JavaScriptExpressionLowering(
            self.graph,
            root,
            tuple(self.shortfalls),
            self.entry_identity,
            tuple(dict(record) for record in self.binding_records),
        )


def ingest_javascript_expression(
    graph: Any,
    node: dict,
    *,
    owner: str,
    assumptions: Sequence[str] = (),
) -> JavaScriptExpressionLowering:
    interpreter = _JavaScriptProcessInterpreter(
        graph, owner=owner, assumptions=assumptions,
    )
    return interpreter.finish(interpreter.expression(node))


def _function_candidates(
    value: Any,
) -> list[tuple[str, Mapping[str, Any], Mapping[str, Any]]]:
    candidates: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            kind = item.get("type")
            if kind == "FunctionDeclaration":
                name = str((item.get("id") or {}).get("name", ""))
                if name:
                    candidates.append((name, item, item))
            elif kind == "VariableDeclarator":
                initializer = item.get("init") or {}
                target = item.get("id") or {}
                if (
                    target.get("type") == "Identifier"
                    and initializer.get("type") in {
                        "ArrowFunctionExpression", "FunctionExpression",
                    }
                ):
                    candidates.append((str(target["name"]), initializer, item))
            for key, child in item.items():
                if key not in {"type", "start", "end", "name"}:
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return candidates


def ingest_javascript_program(
    graph: Any,
    program: Mapping[str, Any],
    *,
    owner: str,
    entry: str | None = None,
    assumptions: Sequence[str] = (),
    parameter_kinds: Mapping[str, str] | None = None,
) -> JavaScriptExpressionLowering:
    """Interpret one selected function, including nested archive functions."""

    candidates = _function_candidates(program)
    selected = [item for item in candidates if entry is None or item[0] == entry]
    if len(selected) != 1:
        wanted = entry if entry is not None else "a unique JavaScript function"
        raise ValueError(
            f"expected {wanted!r}; found {[name for name, _, _ in selected]!r}"
        )
    name, function, declaration = selected[0]
    interpreter = _JavaScriptProcessInterpreter(
        graph, owner=owner, assumptions=assumptions,
        parameter_kinds=parameter_kinds,
    )
    return interpreter.finish(interpreter.function(
        function, declaration=declaration, name=name,
    ))


def ingest_javascript_source(
    graph: Any,
    source: str,
    *,
    owner: str,
    entry: str | None = None,
    assumptions: Sequence[str] = (),
    parameter_kinds: Mapping[str, str] | None = None,
) -> JavaScriptExpressionLowering:
    """Parse source with Turing's vendored Acorn and interpret one function."""

    from .dream_document import _run_javascript_ast_parser

    return ingest_javascript_program(
        graph,
        _run_javascript_ast_parser(source),
        owner=owner,
        entry=entry,
        assumptions=assumptions,
        parameter_kinds=parameter_kinds,
    )


def _javascript_literal_spelling(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return json.dumps(value)
    return repr(value)


def javascript_source_from_graph(graph: Any, node_id: int) -> str:
    data = graph.G.nodes[node_id]
    node_type = data.get("type")
    if node_type == "Constant":
        return _javascript_literal_spelling(data["attributes"].get("value"))
    if node_type == "Input":
        return str(data["attributes"].get("binding_name", node_id))
    if node_type == "Call":
        attributes = data.get("attributes") or {}
        callee = str(attributes.get("tensor_operation") or attributes.get("callee") or "")
        spelling = _REVERSE_DIRECT_CALLS.get(callee, callee)
        by_role = {role: parent for parent, role in data["parents"]}
        arguments = [
            javascript_source_from_graph(graph, by_role[role])
            for role in sorted(by_role, key=lambda item: int(item[3:]))
        ]
        return f"{spelling}({', '.join(arguments)})"
    try:
        handler = Handler[node_type]
    except KeyError as error:
        raise ValueError(
            f"cannot render node type {node_type!r} to javascript"
        ) from error
    by_role = {role: parent for parent, role in data["parents"]}
    if handler is Handler.Select:
        condition = javascript_source_from_graph(graph, by_role["condition"])
        yes = javascript_source_from_graph(graph, by_role["if_true"])
        no = javascript_source_from_graph(graph, by_role["if_false"])
        return f"({condition} ? {yes} : {no})"
    if handler in SSA_TO_JAVASCRIPT_UNARY:
        value = javascript_source_from_graph(graph, by_role["operand"])
        return f"{SSA_TO_JAVASCRIPT_UNARY[handler]}({value})"
    if handler in SSA_TO_JAVASCRIPT_BINARY:
        left = javascript_source_from_graph(graph, by_role["left"])
        right = javascript_source_from_graph(graph, by_role["right"])
        return f"({left} {SSA_TO_JAVASCRIPT_BINARY[handler]} {right})"
    raise ValueError(f"cannot render Handler {handler} to javascript")


__all__ = [
    "JavaScriptExpressionLowering",
    "JavaScriptExpressionShortfall",
    "PORTABLE_NUMERIC_ASSUMPTIONS",
    "ingest_javascript_expression",
    "ingest_javascript_program",
    "ingest_javascript_source",
    "javascript_source_from_graph",
]
