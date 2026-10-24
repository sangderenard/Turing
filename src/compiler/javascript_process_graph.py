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


class _JavaScriptProcessInterpreter:
    def __init__(
        self, graph: Any, *, owner: str, assumptions: Sequence[str] = (),
        parameter_kinds: Mapping[str, str] | None = None,
    ) -> None:
        self.graph = graph
        self.owner = str(owner)
        self.assumptions = frozenset(map(str, assumptions))
        self.parameter_kinds = dict(parameter_kinds or {})
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
        record = {
            "declaration_identity": (
                f"{self.owner}#javascript-binding:{kind}:{name}:"
                f"{declaration.get('start', '?')}:{declaration.get('end', '?')}"
            ),
            "name": name,
            "kind": kind,
            "version": 0,
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

    def expression(self, expr: Mapping[str, Any]) -> int:
        kind = expr.get("type")
        if kind in ("BinaryExpression", "LogicalExpression"):
            operator = str(expr.get("operator"))
            left = self.expression(expr["left"])
            right = self.expression(expr["right"])
            parents = ((left, "left"), (right, "right"))
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

    def sequence(
        self, statements: Sequence[Mapping[str, Any]],
    ) -> tuple[int | None, bool]:
        index = 0
        while index < len(statements):
            statement = statements[index]
            kind = statement.get("type")
            if kind == "VariableDeclaration":
                if statement.get("kind") != "const":
                    self.unsupported(
                        str(statement.get("kind")),
                        "unsupported-binding-kind",
                        "let/var require lexical lifetime and CFG handling",
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
                    self.bind(str(target["name"]), value, target, kind="const")
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
            if kind == "IfStatement":
                if statement.get("alternate") is not None:
                    return self.unsupported(
                        "if",
                        "unsupported-control-flow",
                        "the acyclic region accepts early-return if without else",
                        source=statement,
                    ), False
                saved = dict(self.bindings)
                condition_value = self.expression(statement["test"])
                condition = self.truthy(condition_value, statement["test"])
                self.bindings = dict(saved)
                yes, yes_returns = self.sequence(
                    self._statement_body(statement.get("consequent"))
                )
                self.bindings = dict(saved)
                no, no_returns = self.sequence(statements[index + 1:])
                self.bindings = saved
                if yes_returns and no_returns and yes is not None and no is not None:
                    if self._kind(yes) != self._kind(no):
                        return self.unsupported(
                            "if",
                            "unproved-conditional-result",
                            "early-return arms have different JavaScript value kinds",
                            source=statement,
                            parents=(
                                (condition, "condition"), (yes, "if_true"),
                                (no, "if_false"),
                            ),
                        ), False
                    return self.make_node(
                        Handler.Select.name,
                        Handler.Select.name,
                        Handler.Select.name,
                        ((condition, "condition"), (yes, "if_true"),
                         (no, "if_false")),
                        attributes={"javascript_control": "early-return-if"},
                        source=statement,
                        value_kind=self._kind(yes),
                    ), True
                return self.unsupported(
                    "if",
                    "unsupported-control-flow",
                    "both early-return paths must produce a value",
                    source=statement,
                ), False
            if kind == "ExpressionStatement":
                value = self.expression(statement["expression"])
                self.unsupported(
                    "expression-statement",
                    "unsupported-effect",
                    "the numeric region has no statement-level effects",
                    source=statement,
                    parents=((value, "value"),),
                )
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
            value, returned = self.sequence(body.get("body", ()))
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
