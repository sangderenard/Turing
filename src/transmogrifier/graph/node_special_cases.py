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
