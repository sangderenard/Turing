"""Single switch block for ingestion special cases.

The Python-AST and SymPy front ends converge on the *same* node interpreter:
``ProcessGraph.build_graph`` dispatches on ``type(node).__name__`` for both
(which is why ``role_schemas`` carries SymPy spellings like ``IndexedBase`` /
``Sum`` right beside AST spellings like ``Module`` / ``Assign``).  This module
is the one place that interpreter consults, and -- via :func:`dissolve_spans`
-- the one place the *ingestion seam* consults before any preprocessing pass
walks the tree.

The deal it makes: the instant we can recognise a node as collapsible, we
dissolve it to a single leaf and never descend.  A ``[0.0] * 153600`` span (or
its already-expanded ``repr`` form baked into source text) becomes one ``fill``
leaf -- so no pass ever counts, allocates, or mutates per element.  Every quirk
we ever run across goes in the switch below as a new case; this is the
community list.

Contract: :func:`interpret_special_case` returns a :class:`SpecialCase`
describing the collapsed leaf, or ``None`` to defer to the normal schema path.
Returning ``None`` must be behaviour-preserving -- a special case only ever
*replaces* work the generic path would have done more expensively.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Optional

from ...common.tensors.operator_catalog import (
    CANONICAL_ABSTRACT_TENSOR_OPERATORS,
    REDUCTION_AND_LINALG_OPERATORS,
    canonical_operator_name,
)


# Only *bulk* aggregates are dissolved.  A shape tuple like ``(-1,)`` or
# ``(1, -1)``, an index tuple, or a small parameter list is a structural
# argument -- collapsing it would corrupt the call (e.g. ``reshape((-1,))``
# would receive a fill instead of a shape).  Dissolving only pays off at
# scale anyway, so anything below this element count is left untouched.
_MIN_SPAN_ELEMENTS = 64


@dataclass(frozen=True)
class SpecialCase:
    """A node the interpreter collapses to a single leaf.

    ``type`` is the canonical node/op name written onto the graph node;
    ``attributes`` is copied to both ``attributes`` and ``extra_args``;
    ``constant`` is the folded constant payload (or ``None``).
    """

    type: str
    attributes: dict
    constant: Any


def _as_numeric_constant(element: ast.AST) -> tuple[bool, Any]:
    """Return ``(True, value)`` for a numeric literal (incl. unary sign).

    Booleans are intentionally excluded: ``True``/``False`` are ``int``
    subclasses in Python but are not numeric span material.
    """

    if (
        isinstance(element, ast.Constant)
        and isinstance(element.value, (int, float))
        and not isinstance(element.value, bool)
    ):
        return True, element.value
    if (
        isinstance(element, ast.UnaryOp)
        and isinstance(element.op, (ast.USub, ast.UAdd))
        and isinstance(element.operand, ast.Constant)
        and isinstance(element.operand.value, (int, float))
        and not isinstance(element.operand.value, bool)
    ):
        magnitude = element.operand.value
        return True, (-magnitude if isinstance(element.op, ast.USub) else magnitude)
    return False, None


def _constant_span(node: ast.AST) -> Optional[SpecialCase]:
    """A flat aggregate of numeric constants is one span value, not N nodes.

    Single streaming pass, no per-element allocation on the hot (uniform)
    path: bail on the first non-numeric element, and never build a values
    list or a set just to test uniformity.  Uniform run -> ``fill`` (the
    calloc case when the value is zero); heterogeneous run -> a single
    ``tensor_from_list`` constant (only that rare path materialises values).
    """

    elts = getattr(node, "elts", None)
    if not elts or len(elts) < _MIN_SPAN_ELEMENTS:
        return None
    first_ok, first_value = _as_numeric_constant(elts[0])
    if not first_ok:
        return None
    uniform = True
    for element in elts[1:]:
        ok, value = _as_numeric_constant(element)
        if not ok:
            return None
        if uniform and value != first_value:
            uniform = False
    shape = (len(elts),)
    if uniform:
        return SpecialCase("fill", {"shape": shape, "fill_value": first_value}, None)
    values = tuple(_as_numeric_constant(element)[1] for element in elts)
    return SpecialCase("tensor_from_list", {"shape": shape, "values": values}, values)


def _broadcast_literal(node: ast.BinOp) -> Optional[SpecialCase]:
    """``[x] * N`` / ``N * [x]`` is a fill of shape ``(N,)`` -- recognised in
    O(1), before any expansion ever happens.

    Only a single-element aggregate collapses to a uniform ``fill``; a
    multi-element ``[a, b] * N`` would *tile* into ``len*N`` values, so it is
    deliberately left alone rather than expanded here.
    """

    if not isinstance(node.op, ast.Mult):
        return None

    def _count(candidate: ast.AST) -> Optional[int]:
        if (
            isinstance(candidate, ast.Constant)
            and isinstance(candidate.value, int)
            and not isinstance(candidate.value, bool)
            and candidate.value >= _MIN_SPAN_ELEMENTS
        ):
            return candidate.value
        return None

    def _single(candidate: ast.AST) -> tuple[bool, Any]:
        if isinstance(candidate, (ast.List, ast.Tuple)) and len(candidate.elts) == 1:
            return _as_numeric_constant(candidate.elts[0])
        return False, None

    for aggregate, scalar in ((node.left, node.right), (node.right, node.left)):
        count = _count(scalar)
        if count is None:
            continue
        ok, value = _single(aggregate)
        if ok:
            return SpecialCase("fill", {"shape": (count,), "fill_value": value}, None)
    return None


def interpret_special_case(node: Any) -> Optional[SpecialCase]:
    """The switch block: recognise a node's special case, or defer (``None``).

    Dispatch is on ``type(node).__name__`` so AST and SymPy nodes are handled
    by the same interpreter.  Add new quirks as new cases.
    """

    kind = type(node).__name__

    # ── AST literal aggregates: List / Tuple / Set ───────────────────────
    # A large uniform feed array (e.g. ``[0.0] * 153600`` baked into source)
    # collapses to a single fill leaf instead of one mutation per element.
    if kind in {"List", "Tuple", "Set"}:
        return _constant_span(node)

    # ── AST broadcast literal: ``[x] * N`` ───────────────────────────────
    # The un-expanded form of the same span; dissolved in O(1) so it never
    # becomes a literal aggregate in the first place.
    if kind == "BinOp":
        return _broadcast_literal(node)

    # ── AST attribute access: ``obj.field`` ──────────────────────────────
    # A read (``Load`` context) is normalized to the canonical ``GetAttr``
    # operator here, at ingestion, instead of being left as a generic
    # ``Attribute`` node for later passes to individually recognise (or
    # fail to).  This does not collapse the node -- its receiver
    # (``node.value``) still descends normally as this node's operand -- it
    # only stamps the canonical operator name, the same way a tensor-op
    # ``Call`` is flagged without collapsing just above.  Only ``Load`` is
    # claimed: a ``Store``-context ``Attribute`` is an assignment target,
    # already correctly built into a ``SetAttr`` node (with its own
    # ``object``/``value`` wiring) by ``bind_target`` -- reinterpreting it
    # here too would fight that already-correct, already-tested
    # construction over the same AST node identity.
    if kind == "Attribute" and isinstance(
        getattr(node, "ctx", None), ast.Load
    ):
        return SpecialCase("GetAttr", {"attribute": node.attr}, None)

    # ── (future cases go here) ───────────────────────────────────────────
    # e.g. SymPy ImmutableDenseMatrix of constants -> tensor_from_list,
    #      Call to zeros/ones/full/empty -> fill, etc.  Each is one branch.

    return None


class _SpanDissolver(ast.NodeTransformer):
    """Replace every recognised span with a compact marker at the seam.

    The marker is a bare ``ast.Constant`` carrying the resolved
    :class:`SpecialCase` on ``_special_case`` -- inert to every downstream
    pass (parent-expansion, IR mapping, state-machine planning, the AnnAssign
    normalizer) yet trivially collapsed by ``build_graph``.  A recognised
    node is never descended into: the expansion is walked exactly once, here.
    """

    def _marker(self, node: ast.AST) -> Optional[ast.AST]:
        special = interpret_special_case(node)
        if special is None:
            return None
        marker = ast.Constant(value=None)
        marker._special_case = special  # consumed by ProcessGraph.build_graph
        return ast.copy_location(marker, node)

    def visit_List(self, node):  # noqa: N802 - ast visitor naming
        return self._marker(node) or self.generic_visit(node)

    visit_Tuple = visit_List
    visit_Set = visit_List

    def visit_BinOp(self, node):  # noqa: N802 - ast visitor naming
        return self._marker(node) or self.generic_visit(node)


def dissolve_spans(tree: ast.AST) -> ast.AST:
    """Dissolve recognised spans in ``tree`` in one pass, in place.

    Run at the ingestion seam, before any preprocessing pass walks the tree,
    so a ``repr``-expanded feed array is collapsed once and never seen
    expanded again.
    """

    _SpanDissolver().visit(tree)
    ast.fix_missing_locations(tree)
    return tree


class _WalrusHoister(ast.NodeTransformer):
    """Hoist a walrus ``(n := expr)`` out of a once-evaluated statement position
    into a preceding ordinary ``n = expr``, replacing the expression with a load
    of ``n``.

    The reducer lowers a ``NamedExpr`` when ``resolve_expression`` reaches one,
    but a walrus buried in an ``if``/``return``/assignment expression is not
    always traversed there and then leaks to the deep compiler as a raw
    ``NamedExpr`` (``if (name := f()) is not None:`` is the shape that leaks).
    Hoisting it into a plain assignment -- a position always lowered -- fixes it
    at the syntax level.

    Only positions evaluated **exactly once and unconditionally** are hoisted
    (an ``if`` test, a ``return`` value, an assignment's value, a bare
    expression statement). A walrus inside a re-evaluated test (``while``), a
    short-circuit ``and``/``or``, a ternary, a comprehension, or a lambda is
    left untouched -- hoisting those would change evaluation semantics.
    """

    _SCOPE_OR_SHORTCIRCUIT = (
        ast.BoolOp, ast.Lambda, ast.IfExp,
        ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
    )

    def _hoist(self, expression):
        assigns: list = []
        outer = self

        class _Extract(ast.NodeTransformer):
            def visit_NamedExpr(self, node):  # noqa: N802
                node.value = self.visit(node.value)  # nested walrus first
                assigns.append(
                    ast.copy_location(
                        ast.Assign(targets=[node.target], value=node.value),
                        node,
                    )
                )
                return ast.copy_location(
                    ast.Name(id=node.target.id, ctx=ast.Load()), node
                )

            def visit(self, node):
                if isinstance(node, outer._SCOPE_OR_SHORTCIRCUIT):
                    return node
                return super().visit(node)

        rewritten = _Extract().visit(expression)
        return assigns, rewritten

    def _prefix(self, node, assigns):
        for assign in assigns:
            ast.copy_location(assign, node)
        return [*assigns, node] if assigns else node

    def visit_If(self, node):  # noqa: N802
        node = self.generic_visit(node)
        assigns, node.test = self._hoist(node.test)
        return self._prefix(node, assigns)

    def visit_Return(self, node):  # noqa: N802
        node = self.generic_visit(node)
        if node.value is None:
            return node
        assigns, node.value = self._hoist(node.value)
        return self._prefix(node, assigns)

    def visit_Assign(self, node):  # noqa: N802
        node = self.generic_visit(node)
        assigns, node.value = self._hoist(node.value)
        return self._prefix(node, assigns)

    def visit_Expr(self, node):  # noqa: N802
        node = self.generic_visit(node)
        assigns, node.value = self._hoist(node.value)
        return self._prefix(node, assigns)


def hoist_walrus_assignments(tree: ast.AST) -> ast.AST:
    """Hoist safely-hoistable walruses in ``tree`` (in place), so no raw
    ``NamedExpr`` reaches the deep compiler from a once-evaluated position."""

    _WalrusHoister().visit(tree)
    ast.fix_missing_locations(tree)
    return tree


class _TypeAnnotator(ast.NodeTransformer):
    """Turn an annotated assignment into an ordinary assignment PLUS captured
    type metadata -- a real type annotator, not a discard.

    ``x: T = v`` becomes ``x = v`` and records ``x -> T``; ``x: T`` (declaration
    only) is dropped as runtime code but still records ``x -> T``. The captured
    annotation is the source's own declared type, exactly the information the
    dtype system needs (it is where an ``int`` parameter's integer-ness comes
    from), so it is preserved as metadata instead of thrown away.

    Only executable bodies are rewritten. A class body's own ``AnnAssign`` is its
    field schema -- name/type/default, read off the untouched ``ClassDef`` by
    the class-table builder -- so ``visit_ClassDef`` normalizes each method body
    without touching the class's own direct-child statements.
    """

    def __init__(self) -> None:
        self.annotations: dict[str, str] = {}

    def _record(self, target, annotation) -> None:
        if isinstance(target, ast.Name) and annotation is not None:
            try:
                self.annotations[target.id] = ast.unparse(annotation)
            except Exception:  # noqa: BLE001 -- metadata only, never fatal
                pass

    def visit_ClassDef(self, node):  # noqa: N802
        node.body = [
            self.visit(member)
            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
            else member
            for member in node.body
        ]
        return node

    def visit_AnnAssign(self, node):  # noqa: N802
        node = self.generic_visit(node)
        self._record(node.target, node.annotation)
        if node.value is None:
            return ast.copy_location(ast.Pass(), node)
        return ast.copy_location(
            ast.Assign(targets=[node.target], value=node.value), node
        )


def annotate_types(tree: ast.AST) -> dict:
    """Rewrite annotated assignments in ``tree`` (in place) to ordinary
    assignments and return the captured ``name -> annotation`` type metadata."""

    annotator = _TypeAnnotator()
    annotator.visit(tree)
    ast.fix_missing_locations(tree)
    return annotator.annotations


class _DeclaredAttributeCollector(ast.NodeVisitor):
    """Every attribute name the source declares: a class's methods/properties,
    its annotated or assigned class-level fields, and any ``obj.name = ...``
    write. The union is what a constant-name ``getattr`` may be folded against --
    a name written or defined somewhere is a real attribute, not a dynamic probe.
    """

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_ClassDef(self, node):  # noqa: N802
        for member in node.body:
            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.names.add(member.name)
            elif isinstance(member, ast.AnnAssign) and isinstance(
                member.target, ast.Name
            ):
                self.names.add(member.target.id)
            elif isinstance(member, ast.Assign):
                for target in member.targets:
                    if isinstance(target, ast.Name):
                        self.names.add(target.id)
        self.generic_visit(node)

    def visit_Attribute(self, node):  # noqa: N802
        if isinstance(getattr(node, "ctx", None), ast.Store):
            self.names.add(node.attr)
        self.generic_visit(node)


class _GetattrFolder(ast.NodeTransformer):
    """Fold ``getattr(obj, "name"[, default])`` into the attribute access
    ``obj.name`` when ``"name"`` is a constant identifier the source declares.

    A constant-name ``getattr`` is not a dynamic lookup -- it names a specific
    attribute at compile time, exactly what ``obj.name`` means -- but left as a
    call it carries the name as a string constant a numeric backend cannot
    express, and never resolves to the attribute's structure (``x.shape`` -> its
    dimension extents). Folding routes it through the ordinary ``GetAttr`` path.
    The ``default`` argument is dropped: for an attribute the class declares it
    is provably present, so the default is dead in typed AOT (a genuinely
    missing attribute is a type error that should surface, not be defaulted).
    """

    def __init__(self, declared: set[str]) -> None:
        self.declared = declared

    def visit_Call(self, node):  # noqa: N802
        node = self.generic_visit(node)  # fold nested getattrs first
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) in (2, 3)
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
            and node.args[1].value.isidentifier()
            and node.args[1].value in self.declared
        ):
            return ast.copy_location(
                ast.Attribute(
                    value=node.args[0],
                    attr=node.args[1].value,
                    ctx=ast.Load(),
                ),
                node,
            )
        return node


def fold_constant_getattr(tree: ast.AST) -> ast.AST:
    """Fold constant-name ``getattr`` calls into attribute accesses (in place),
    gated to attribute names the source declares, so a static attribute lookup
    resolves structurally instead of surviving as an inexpressible string."""

    collector = _DeclaredAttributeCollector()
    collector.visit(tree)
    _GetattrFolder(collector.names).visit(tree)
    ast.fix_missing_locations(tree)
    return tree


def tensor_operation_name(node: Any) -> Optional[str]:
    """Canonical tensor-op name for a call node, else ``None``.

    Recognises ``x.op(...)`` (method) and ``op(...)`` (free function) against
    the authoritative :data:`CANONICAL_ABSTRACT_TENSOR_OPERATORS` catalog so
    ingestion can flag a node as tensor-bearing without collapsing it.
    """

    if type(node).__name__ != "Call":
        return None
    func = getattr(node, "func", None)
    if isinstance(func, ast.Attribute):
        name = func.attr
    elif isinstance(func, ast.Name):
        name = func.id
    else:
        return None
    canonical = canonical_operator_name(name)
    if canonical in CANONICAL_ABSTRACT_TENSOR_OPERATORS:
        return canonical
    return None


def is_reduction_operation(name: str) -> bool:
    """Whether a canonical op name reduces an axis (needs unroll off-tensor)."""

    return canonical_operator_name(name) in REDUCTION_AND_LINALG_OPERATORS


def graph_has_tensor_operation(graph: Any, node_ids: Any = None) -> bool:
    """Whether any node in ``node_ids`` (default: the whole graph) is
    tensor-bearing -- the ``attributes["tensor"]`` stamp ``build_graph``
    already writes from :func:`tensor_operation_name`, an ingestion-time,
    AST-derived fact, not a runtime observation.

    This is the qualification test for kernel reduction: a subgraph with no
    tensor operation anywhere has nothing for the flatten-and-optimize
    reduction path to parallelize across, so it does not qualify -- its
    already-correct ``ProcessGraph`` structure can be lowered directly
    instead, one node at a time, with no algebraic fusion and no runtime
    tape. A subgraph with at least one tensor operation still qualifies for
    the ordinary tensor-kernel reduction path, unchanged.
    """

    nodes = graph.G.nodes(data=True)
    if node_ids is not None:
        wanted = frozenset(int(node_id) for node_id in node_ids)
        nodes = (
            (node_id, data)
            for node_id, data in nodes
            if int(node_id) in wanted
        )
    return any(
        (data.get("attributes") or {}).get("tensor") is not None
        for _node_id, data in nodes
    )
