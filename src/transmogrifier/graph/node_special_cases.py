"""Shared structural compactions for ProcessGraph ingestion.

The Python-AST and SymPy front ends converge on the *same* node interpreter:
``ProcessGraph.build_graph`` dispatches on ``type(node).__name__`` for both
(which is why ``role_schemas`` carries SymPy spellings like ``IndexedBase`` /
``Sum`` right beside AST spellings like ``Module`` / ``Assign``).  This module
consults this module for cross-frontend compactions. Language meaning is kept
in sibling adapters; Python call and attribute semantics live in
``python_special_cases.py``. :func:`dissolve_spans` still runs here before any
preprocessing pass walks the tree.

The deal it makes: the instant we can recognise a node as collapsible, we
dissolve it to a single leaf and never descend.  A ``[0.0] * 153600`` span (or
its already-expanded ``repr`` form baked into source text) becomes one ``fill``
leaf -- so no pass ever counts, allocates, or mutates per element. Only cases
safe across the shared structural walker belong in the switch below.

Contract: :func:`interpret_special_case` returns a :class:`SpecialCase`
describing the structural replacement, or ``None`` to defer to the normal
schema path. Sibling language adapters may also use ``terminal=False`` to
decorate a node while retaining its ordinary child-role traversal.
"""

from __future__ import annotations

import ast
import copy
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
    ``terminal=False`` retains ordinary role-schema child traversal.
    """

    type: str
    attributes: dict
    constant: Any
    terminal: bool = True


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


class _ContextInliner(ast.NodeTransformer):
    """Replace a ``with`` by the statements its context manager actually runs.

    A context manager is not control flow. It reads a slot, writes a value,
    runs the body, and writes the old value back -- no branch, no iteration,
    no continuation to merge. ``extended_precision.precision`` is exactly
    that and nothing else::

        previous = getattr(_state, "limbs", 1)   # get
        _state.limbs = limbs                     # set
        yield limbs                              # body
        _state.limbs = previous                  # restore

    Treating it as control flow is what blocks it: loop composition forbids
    ``ast.With`` in a loop body, grouped with ``raise`` and ``await`` as
    control divergence -- right for those, a category error for this. So the
    statement is turned into the statements it already is, at ingestion,
    exactly as a walrus is hoisted above so no raw ``NamedExpr`` reaches the
    deep compiler. Nothing downstream then needs to learn what a context
    manager is.

    The halves are READ FROM THE CONTEXT MANAGER'S OWN SOURCE rather than
    matched against a table of known names. Whatever it does before its
    ``yield`` is the set-up, whatever it does after is the clean-up, and the
    body goes between them. A manager this cannot read keeps its ``with``,
    and therefore keeps its blocker, because an unread context may carry real
    effects and dropping them silently is the failure this avoids.
    """

    def __init__(self, resolver) -> None:
        self.resolver = resolver
        self.counter = 0
        self.inlined: list[str] = []

    @staticmethod
    def _halves(definition: ast.FunctionDef):
        """Set-up and clean-up: the statements either side of the ``yield``."""

        def holds_yield(node) -> bool:
            return any(isinstance(inner, (ast.Yield, ast.YieldFrom))
                       for inner in ast.walk(node))

        setup: list[ast.stmt] = []
        for statement in definition.body:
            if isinstance(statement, ast.Try) and holds_yield(statement):
                for inner in statement.body:
                    if holds_yield(inner):
                        break
                    setup.append(inner)
                return setup, list(statement.finalbody)
            if holds_yield(statement):
                return setup, []
            setup.append(statement)
        return None

    def _bind(self, definition: ast.FunctionDef, call: ast.Call,
              setup: list[ast.stmt], cleanup: list[ast.stmt]):
        """Both halves, in ONE renaming scope, parameters bound by assignment.

        The halves must share a scope: the set-up's temporary is what the
        clean-up restores from, so renaming them apart leaves the restore
        reading a name nothing defined.

        Parameters are bound by an assignment rather than substituted into
        their uses. A manager may reassign its own parameter -- ``precision``
        opens with ``limbs = int(limbs)`` -- and substituting the call's
        argument for every load would silently discard that, so the argument
        is assigned once and the manager's own code proceeds normally.
        """

        self.counter += 1
        tag = f"__ctx{self.counter}_"
        parameters = [argument.arg for argument in definition.args.args]
        owned = set(parameters) | {
            node.id
            for statement in (*setup, *cleanup)
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }

        class Rebind(ast.NodeTransformer):
            def visit_Name(self, node):  # noqa: N802
                if node.id in owned:
                    return ast.copy_location(
                        ast.Name(id=tag + node.id, ctx=node.ctx), node,
                    )
                return node

        def rebind(statements):
            kept = []
            for statement in statements:
                # A docstring is documentation, not a step the manager runs.
                if (isinstance(statement, ast.Expr)
                        and isinstance(statement.value, ast.Constant)
                        and isinstance(statement.value.value, str)):
                    continue
                kept.append(Rebind().visit(copy.deepcopy(statement)))
            return kept

        binding = [
            ast.Assign(targets=[ast.Name(id=tag + name, ctx=ast.Store())],
                       value=copy.deepcopy(value))
            for name, value in zip(parameters, call.args)
        ]
        return binding + rebind(setup), rebind(cleanup)

    def visit_With(self, node):  # noqa: N802
        node = self.generic_visit(node)
        prologue: list[ast.stmt] = []
        epilogue: list[ast.stmt] = []
        for item in node.items:
            call = item.context_expr
            if not isinstance(call, ast.Call):
                return node
            definition = self.resolver(call)
            if definition is None:
                return node
            halves = self._halves(definition)
            if halves is None:
                return node
            if item.optional_vars is not None:
                # ``as`` names the yielded value, which is the one thing the
                # halves do not carry between them. Left alone rather than
                # guessed at.
                return node
            setup, cleanup = self._bind(definition, call, *halves)
            prologue.extend(setup)
            # Clean-ups undo in reverse, innermost first.
            epilogue[:0] = cleanup
            self.inlined.append(ast.unparse(call))
        return prologue + list(node.body) + epilogue


def inline_context_managers(tree: ast.AST, resolver) -> ast.AST:
    """Turn every readable ``with`` in ``tree`` into its equivalent statements.

    ``resolver`` maps a call node to the ``FunctionDef`` it references, or to
    ``None`` when it cannot be read. Resolution is the caller's because only
    the caller knows how names in this tree bind.
    """

    inliner = _ContextInliner(resolver)
    inliner.visit(tree)
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


def _slice_value(element: ast.AST) -> ast.AST:
    """An index element as a value expression. An ``ast.Slice`` is only legal
    directly inside a subscript; once the index is built as an ordinary tuple
    value (to splice the ellipsis expansion in) each slice must become an
    explicit ``slice(a, b, c)`` call."""

    if isinstance(element, ast.Slice):
        return ast.Call(
            ast.Name("slice", ast.Load()),
            [
                element.lower or ast.Constant(None),
                element.upper or ast.Constant(None),
                element.step or ast.Constant(None),
            ],
            [],
        )
    return element


def _full_slice_call() -> ast.AST:
    """``slice(None, None, None)`` -- a full ``:`` as a value expression."""

    return ast.Call(
        ast.Name("slice", ast.Load()),
        [ast.Constant(None), ast.Constant(None), ast.Constant(None)],
        [],
    )


class _EllipsisExpander(ast.NodeTransformer):
    """Expand ``obj[..., k]`` into an explicit, ndim-driven full-slice index.

    ``...`` means "as many full slices as ``obj`` has dimensions, minus the
    explicit index elements". That count is ``obj.ndim - (len(before) + len(
    after))``, so the ellipsis becomes ``[slice(None)] * that`` and the index is
    ``(before) + tuple(that) + (after)``. When ``obj.ndim`` is known at compile
    time the multiply folds to concrete slices; when it is not, it stays an
    ndim-driven expression rather than a wrong fixed guess (``... -> :`` would
    miscompile ``x[..., k]`` on rank > 2). ``obj[...]`` alone becomes
    ``obj[tuple([slice(None)] * obj.ndim)]`` -- correct for every rank, 0-d
    included (an empty index, i.e. the whole value).
    """

    def visit_Subscript(self, node):  # noqa: N802
        node = self.generic_visit(node)
        index = node.slice
        elements = list(index.elts) if isinstance(index, ast.Tuple) else [index]
        positions = [
            position
            for position, element in enumerate(elements)
            if isinstance(element, ast.Constant) and element.value is Ellipsis
        ]
        if len(positions) != 1:
            # No ellipsis, or the illegal multi-ellipsis form -- leave as is.
            return node
        cut = positions[0]
        before = elements[:cut]
        after = elements[cut + 1:]
        import copy

        count = ast.BinOp(
            ast.Attribute(copy.deepcopy(node.value), "ndim", ast.Load()),
            ast.Sub(),
            ast.Constant(len(before) + len(after)),
        )
        expansion = ast.Call(
            ast.Name("tuple", ast.Load()),
            [ast.BinOp(ast.List([_full_slice_call()], ast.Load()), ast.Mult(), count)],
            [],
        )
        index_expr = ast.BinOp(
            ast.BinOp(
                ast.Tuple([_slice_value(e) for e in before], ast.Load()),
                ast.Add(),
                expansion,
            ),
            ast.Add(),
            ast.Tuple([_slice_value(e) for e in after], ast.Load()),
        )
        # Keep the authored Subscript object's identity.  ProcessGraph node
        # ids are the AST object ids captured by callers before this seam;
        # replacing the node here makes the expanded operation unreachable by
        # that stable source identity even though its semantics are unchanged.
        # The expansion only changes the index expression, so mutate that
        # field in place and retain the original value/context/location.
        node.slice = index_expr
        return node


def expand_ellipsis_subscripts(tree: ast.AST) -> ast.AST:
    """Rewrite ``...`` subscripts into explicit ndim-driven full-slice indices
    (in place), so no inexpressible ``Ellipsis`` literal reaches a backend."""

    _EllipsisExpander().visit(tree)
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
