"""Declarative JavaScript identities expressed in ProcessGraph vocabulary.

These records are the JavaScript counterpart of ``python_identity_programs``:
they say which realm-owned JavaScript identity may be replaced by which
existing repository operation.  A surface spelling alone is never enough --
``Math`` may be shadowed and JavaScript operators carry coercion rules -- so
every replacement states the proof requirements the JavaScript region must
already possess.

The catalogue is inert.  It does not execute JavaScript, add Handler opcodes,
or reach into the generic compiler.  These strings are intentionally the
source-language vocabulary, just as they are in ``python_identity_programs``;
the ordinary ProcessGraph lowering owns structural tokenization downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from ...common.tensors.operator_catalog import (
    CANONICAL_ABSTRACT_TENSOR_OPERATORS,
)


@dataclass(frozen=True, slots=True)
class JavaScriptIdentityProgram:
    identities: tuple[str, ...]
    kind: str
    operator: str | None = None
    requirements: frozenset[str] = frozenset()
    effects: tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        if not self.identities:
            raise ValueError("a JavaScript identity program requires an identity")
        if self.kind not in {"operator", "object", "host"}:
            raise ValueError(
                f"unknown JavaScript identity replacement kind {self.kind!r}"
            )
        if self.kind == "operator":
            if self.operator not in CANONICAL_ABSTRACT_TENSOR_OPERATORS:
                raise ValueError(
                    f"unknown ProcessGraph operator {self.operator!r}"
                )
        elif self.operator is not None:
            raise ValueError(
                f"{self.kind} JavaScript identity cannot claim an operator"
            )
        object.__setattr__(self, "attributes", MappingProxyType(
            dict(self.attributes)
        ))

    def mapping(self) -> dict[str, Any]:
        return {
            "identities": self.identities,
            "kind": self.kind,
            "operator": self.operator,
            "requirements": tuple(sorted(self.requirements)),
            "effects": self.effects,
            "attributes": dict(self.attributes),
            "description": self.description,
        }


_MATH_REQUIREMENTS = frozenset({
    "number-operands", "pure-operands",
    "ecmascript-math-number-semantics", "unshadowed-realm-intrinsics",
})


def _math(
    identity: str, operator: str, *, arity: int = 1, **attributes: Any,
) -> JavaScriptIdentityProgram:
    return JavaScriptIdentityProgram(
        identities=(f"javascript.realm.Math.{identity}",),
        kind="operator",
        operator=operator,
        requirements=_MATH_REQUIREMENTS,
        attributes={
            "javascript_spelling": f"Math.{identity}",
            "javascript_arity": int(arity),
            **attributes,
        },
        description=f"realm Math.{identity} on proven Number operands",
    )


JAVASCRIPT_IDENTITY_PROGRAMS: tuple[JavaScriptIdentityProgram, ...] = (
    *(
        _math(identity, operator)
        for identity, operator in (
            ("abs", "abs"), ("acos", "acos"), ("acosh", "acosh"),
            ("asin", "asin"), ("asinh", "asinh"), ("atan", "atan"),
            ("atanh", "atanh"), ("ceil", "ceil"), ("cos", "cos"),
            ("cosh", "cosh"), ("exp", "exp"), ("floor", "floor"),
            ("log", "log"), ("sign", "sign"),
            ("sin", "sin"), ("sinh", "sinh"), ("sqrt", "sqrt"),
            ("tan", "tan"), ("tanh", "tanh"), ("trunc", "trunc"),
        )
    ),
    _math("pow", "pow", arity=2),
    # The repository operation is binary.  The interpreter admits exactly
    # two arguments; JavaScript's zero/one/variadic cases remain visible
    # shortfalls until they have their own reduction program.
    _math("max", "maximum", arity=2),
    _math("min", "minimum", arity=2),
    JavaScriptIdentityProgram(
        identities=("javascript.realm.Number.isNaN",),
        kind="operator",
        operator="isnan",
        requirements=frozenset({
            "number-operands", "pure-operands",
            "ecmascript-math-number-semantics",
            "unshadowed-realm-intrinsics",
        }),
        attributes={
            "javascript_spelling": "Number.isNaN", "javascript_arity": 1,
        },
        description="non-coercing Number.isNaN on a proven Number operand",
    ),
    # Known archive boundaries.  They are deliberately not operators: the
    # JavaScript region planner must keep their storage/effect/host semantics.
    JavaScriptIdentityProgram(
        identities=("javascript.Array.prototype.push",),
        kind="object",
        requirements=frozenset({"resident-sequence-receiver"}),
        effects=("write:receiver",),
        attributes={"object_type": "resident_sequence_append"},
        description="grow a resident JavaScript sequence",
    ),
    JavaScriptIdentityProgram(
        identities=("javascript.DataView.prototype.getUint16",),
        kind="object",
        requirements=frozenset({"dataview-receiver", "explicit-endianness"}),
        effects=("read:receiver",),
        attributes={"object_type": "binary_memory_read", "width": 16},
        description="unsigned 16-bit DataView read with explicit byte order",
    ),
    JavaScriptIdentityProgram(
        identities=(
            "javascript.host.FileReader", "javascript.host.Worker",
            "javascript.host.document", "javascript.host.CanvasRenderingContext2D",
        ),
        kind="host",
        effects=("javascript-host",),
        attributes={"region": "browser-javascript"},
        description="browser-owned capability retained as a deployment boundary",
    ),
)


_IDENTITY_INDEX = MappingProxyType({
    identity: program
    for program in JAVASCRIPT_IDENTITY_PROGRAMS
    for identity in program.identities
})


def resolve_javascript_identity(
    identity: str | None,
) -> JavaScriptIdentityProgram | None:
    """Resolve one already-grounded JavaScript realm identity."""

    if not identity:
        return None
    return _IDENTITY_INDEX.get(str(identity))


def javascript_surface_identities() -> Mapping[str, str]:
    """Surface spelling -> realm identity for unshadowed intrinsic calls."""

    return MappingProxyType({
        str(program.attributes["javascript_spelling"]): program.identities[0]
        for program in JAVASCRIPT_IDENTITY_PROGRAMS
        if "javascript_spelling" in program.attributes
    })


def begin_javascript_ingestion(graph: Any, owner: str) -> tuple[str, int]:
    """Mint the identity scope for one JavaScript graph product."""

    from ...compiler.concordance_declarations import INGESTION
    from ...compiler.identity_concordance import current_identity_book

    scope = graph.G.graph.get("ingestion_value_scope")
    if scope is None:
        scope = current_identity_book().mint_scope(
            f"javascript-ingestion:{owner}", stage=INGESTION,
        )
        graph.G.graph["ingestion_value_scope"] = scope
    graph.G.graph["source_language"] = "javascript"
    graph.G.graph["javascript_owner"] = str(owner)
    return tuple(scope)


def post_javascript_ingestion_identity(
    graph: Any, node_id: int, *, owner: str, qualname: str,
    kind: str, operation: str, label: str,
    source: Mapping[str, Any] | None = None,
) -> Any:
    """Post one ESTree/canonical-node source chain before authoring edges."""

    from ...compiler.concordance_declarations import (
        INGESTION, INGESTION_VALUE, INGEST_SOURCE, NodeFact, SOURCE_SPAN,
        SpanFact,
    )
    from ...compiler.identity_concordance import (
        Derived, Mode, Novel, current_identity_book,
    )

    scope = begin_javascript_ingestion(graph, owner)
    source = dict(source or {})
    start = int(source.get("start", -1) if source.get("start") is not None else -1)
    end = int(source.get("end", -1) if source.get("end") is not None else -1)
    # The ESTree construct is the authored source identity.  Several graph
    # operations may be derived from one construct (for example JavaScript
    # Number truthiness), so operation and label must not mint extra roots.
    import hashlib
    import json
    encoded = json.dumps(source, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    path = f"estree:{start}:{end}:{digest}"
    span_fact = SpanFact(
        str(source.get("type") or kind), -1, -1, -1, -1,
        digest,
    )
    book = current_identity_book()
    span = book.post(
        SOURCE_SPAN, (str(owner), str(qualname), path), span_fact,
        stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()),
        mode=Mode.CONCORD,
    )
    return book.post(
        INGESTION_VALUE, (scope, int(node_id)),
        NodeFact(str(kind), str(operation), str(label)),
        stage=INGESTION, provenance=Derived((span,)), mode=Mode.CONCORD,
    )


__all__ = [
    "JAVASCRIPT_IDENTITY_PROGRAMS",
    "JavaScriptIdentityProgram",
    "begin_javascript_ingestion",
    "javascript_surface_identities",
    "post_javascript_ingestion_identity",
    "resolve_javascript_identity",
]
