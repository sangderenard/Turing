"""Dispatch metadata classifier: the per-node rule deciding which nodes are dispatch metadata rather than executable work, its lazily built executable-rule set and the inert-routing node scan."""

from __future__ import annotations

import ast
from typing import Any

from ..common.tensors.fused_ir import flatten_tensor_constant
from ..common.tensors.operator_catalog import ACCESSOR_OPERATORS
from ..transmogrifier.operator_defs import abstract_tensor_funcs
from .source_node_facts import _constant_value
from .region_scheduling import _dependency_order
from .tensor_descriptor_query import _DTYPE_CAST_OPERATIONS


def _is_ast_metadata_node(graph: Any, node_id: int) -> bool:
    """Return whether a structural AST node carries no runtime value."""

    data = graph.G.nodes[node_id]
    node_type = data.get("type")
    map_ir = graph.G.graph.get("map_ir") or {}
    return (
        node_id in set(map_ir.get("schema_node_ids", ()))
        or
        node_type in {"str", "NoneType"}
        or isinstance(
            data.get("expr_obj"),
            (
                ast.Module,
                ast.ClassDef,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.Import,
                ast.ImportFrom,
                ast.alias,
                ast.arguments,
                ast.arg,
                ast.keyword,
            ),
        )
    )


def _semantic_children(graph: Any) -> dict[int, set[int]]:
    """Union the edge and parent-reference views of who consumes each node."""

    children: dict[int, set[int]] = {
        node_id: set(graph.G.successors(node_id)) for node_id in graph.G
    }
    for node_id, data in graph.G.nodes(data=True):
        for parent, _role in data.get("parents", ()):
            if parent in children:
                children[parent].add(node_id)
    return children


def _inert_routing_nodes(graph: Any) -> frozenset[int]:
    """Find attribute lookups whose result nothing ever reads.

    Ingesting ``x.foo()`` records the attribute lookup beside the call even
    when the call already references ``x`` directly, so the lookup is left with
    no consumer.  Evaluating it anyway is not free: it demands a materialized
    value for ``x``, which forces an otherwise shader-local intermediate to
    become a published region output.  Naming these nodes lets both the region
    extractor and the coordinator ignore work no one observes.  Only the
    side-effect-free lookup itself qualifies: only a load, never an assignment
    or deletion target, and never a call or statement with no consumer.
    """

    fingerprint = (
        graph.G.number_of_nodes(),
        graph.G.number_of_edges(),
        tuple(map(int, getattr(graph, "roots", ()) or ())),
    )
    cached = graph.G.graph.get("_inert_routing_nodes_cache")
    if cached is not None and cached[0] == fingerprint:
        return cached[1]
    roots = set(getattr(graph, "roots", ()) or ())
    children = _semantic_children(graph)
    inert: set[int] = set()
    for node_id in reversed(_dependency_order(graph)):
        if node_id in roots:
            continue
        expression = graph.G.nodes[node_id].get("expr_obj")
        if not isinstance(expression, ast.Attribute):
            continue
        if not isinstance(getattr(expression, "ctx", None), ast.Load):
            # ``obj.field = value`` mutates the receiver.  Its node has no
            # consumer by construction, which is exactly why it must still run.
            continue
        if all(child in inert for child in children[node_id]):
            inert.add(node_id)
    result = frozenset(inert)
    graph.G.graph["_inert_routing_nodes_cache"] = (fingerprint, result)
    return result


# Serialized ProcessGraphs retain this derived cache. Bump its schema when
# classification semantics change, so a compiler replay recomputes old facts.
_DISPATCH_METADATA_CACHE_SCHEMA = 3


def _dispatch_metadata_node_classifier(graph: Any):
    """Return a cached classifier for one pass over a stable planning graph.

    NetworkX counts edges by walking node degrees. Validate the cache once
    per pass, rather than doing that graph-wide walk for every queried node.
    Obtain a new classifier after changing the graph's structure.
    """

    from ..common.tensors.topological_reducer import node_identity_cell
    from .concordance_declarations import (
        EXECUTABLE_NODE, ExecutionKind, MetadataRule, NodeExecution,
        PLANNER_DISPATCH_CLASSIFICATION, PLANNER_SCOPE, SCALAR_PARAMETER,
        SYNTHESIZED_NO_SOURCE,
    )
    from .identity_concordance import (
        Derived, Mode, Unsourced, current_identity_book,
    )

    G = graph.G
    fingerprint = (_DISPATCH_METADATA_CACHE_SCHEMA,
                   G.number_of_nodes(), G.number_of_edges())
    cache = G.graph.get("_dispatch_metadata_cache")
    book = current_identity_book()
    if cache is None or cache.get("__fingerprint__") != fingerprint:
        # A new pass over a changed graph classifies again under a new
        # planning scope (plan 80, A2.3): every verdict of this pass is a row
        # ``executable_node (planning scope, node)`` and the cache below is
        # that scope's read view.  Readers take the scope from
        # ``graph.G.graph["planning_scope"]``, never join rows of two passes
        # by node id (R2).
        planning_scope = book.mint_scope("plan", PLANNER_SCOPE)
        G.graph["planning_scope"] = planning_scope
        cache = {"__fingerprint__": fingerprint,
                 "__scope__": planning_scope,
                 "__carried_initials__": frozenset(
                     int(initial)
                     for _, data in G.nodes(data=True)
                     for initial, _updated in (data.get("attributes") or {}).get(
                         "loop_carried_bindings", {}
                     ).values()
                 )}
        G.graph["_dispatch_metadata_cache"] = cache
    planning_scope = cache["__scope__"]

    def post_verdict(node_id: int, rule: MetadataRule) -> None:
        kind = (
            ExecutionKind.EXECUTABLE
            if rule in _EXECUTABLE_RULES
            else ExecutionKind.DISPATCH_METADATA
        )
        fact = NodeExecution(kind, rule)
        row = (planning_scope, int(node_id))
        try:
            cell = node_identity_cell(graph, int(node_id))
        except ValueError:
            # A graph the reducer never scoped (a synthetic wrapper): the
            # verdict is recorded, its source is not.
            book.post(
                EXECUTABLE_NODE, row, fact,
                stage=PLANNER_DISPATCH_CLASSIFICATION,
                provenance=Unsourced(SYNTHESIZED_NO_SOURCE),
                mode=Mode.CONCORD,
            )
            return
        cells = [cell]
        if rule is MetadataRule.COORDINATOR_NODE_TYPE:
            # An Input the reducer declared scalar: the rule read that row.
            scalar = book.latest_ref(SCALAR_PARAMETER, (cell.row[0], int(node_id)))
            if scalar is not None:
                cells.append(scalar)
        book.post(
            EXECUTABLE_NODE, row, fact,
            stage=PLANNER_DISPATCH_CLASSIFICATION,
            provenance=Derived(tuple(cells)), mode=Mode.CONCORD,
        )

    def classify(node_id: int) -> bool:
        key = int(node_id)
        cached = cache.get(key)
        if cached is not None:
            return cached
        rule = _dispatch_metadata_rule(graph, node_id)
        result = rule not in _EXECUTABLE_RULES
        post_verdict(key, rule)
        cache[key] = result
        return result

    return classify


def _executable_rules():
    from .concordance_declarations import MetadataRule

    return frozenset((
        MetadataRule.PRECISION_OPERATOR, MetadataRule.CATALOGUED_BUILTIN_CALL,
        MetadataRule.GROUNDED_TENSOR_PROPERTY, MetadataRule.SCALAR_INTRINSIC,
        MetadataRule.NUMERICAL_WORK,
    ))


class _ExecutableRules:
    """The ``MetadataRule`` members whose verdict is EXECUTABLE (the
    ``return False`` branches of the classifier and "no rule fired"),
    resolved on first use so the declarations import stays lazy."""

    _rules = None

    def __contains__(self, rule: Any) -> bool:
        if self._rules is None:
            type(self)._rules = _executable_rules()
        return rule in self._rules


_EXECUTABLE_RULES = _ExecutableRules()


def _is_dispatch_metadata_node(graph: Any, node_id: int) -> bool:
    """Classify a single node, validating the graph cache before the query."""
    return _dispatch_metadata_node_classifier(graph)(node_id)


def _is_dispatch_metadata_node_impl(graph: Any, node_id: int) -> bool:
    return _dispatch_metadata_rule(graph, node_id) not in _EXECUTABLE_RULES


def _dispatch_metadata_rule(graph: Any, node_id: int) -> Any:
    """Which rule classifies ``node_id``: a ``MetadataRule`` member, one per
    ``return`` (or disjunct of the final ``return``) of the classifier, in
    evaluation order.  ``NUMERICAL_WORK`` when no rule fires.  The verdict
    is a function of the rule: ``_EXECUTABLE_RULES`` are EXECUTABLE, every
    other member is DISPATCH_METADATA."""

    from .concordance_declarations import MetadataRule as Rule

    data = graph.G.nodes[node_id]
    precision_operator = (data.get("attributes") or {}).get(
        "python_precision_operator"
    )
    if precision_operator is True:
        # The authored syntax is a method call, but the concorded rewrite has
        # already made this a repository precision operator. It is numerical
        # region work, not a coordinator call. Its detached Attribute node is
        # separately marked ``*-selector`` and remains metadata.
        return Rule.PRECISION_OPERATOR
    if isinstance(precision_operator, str) and precision_operator.endswith(
        "-selector"
    ):
        return Rule.PRECISION_SELECTOR
    if (data.get("attributes") or {}).get("authored_call_result_projection"):
        # Specialization publishes typed leaves of a callee's tuple return.
        # Their tensor descriptors describe the payload, not a tensor gather
        # to dispatch. Keep the projection at its owning call/branch; a free
        # numerical region here escapes the loop that produces the return.
        return Rule.CALL_RESULT_PROJECTION
    if (data.get("attributes") or {}).get("bound_method_ref") is not None:
        return Rule.BOUND_METHOD
    if (data.get("attributes") or {}).get("conditional_result_of") is not None:
        return Rule.CONDITIONAL_RESULT
    node_type = str(data.get("type"))
    expression = data.get("expr_obj")
    parents = tuple(data.get("parents") or ())
    indexed_base = next(
        (
            parent
            for parent, role in parents
            if str(role) == "base"
        ),
        None,
    )
    python_routing_index = (
        node_type == "Indexed"
        and indexed_base in graph.G
        and (
            str(data.get("label", "")).startswith("unpack[")
            or graph.G.nodes[indexed_base].get("type")
            in {"Tuple", "List", "Set", "Dict"}
            or any(
                parent in graph.G
                and graph.G.nodes[parent].get("type")
                in {"Const", "const", "Constant"}
                and isinstance(
                    _constant_value(graph.G.nodes[parent]),
                    (str, bytes),
                )
                for parent, role in parents
                if str(role) == "index"
            )
        )
    )
    compares_none = (
        isinstance(expression, ast.Compare)
        and any(
            parent in graph.G
            and graph.G.nodes[parent].get("type")
            in {"Const", "const", "Constant"}
            and graph.G.nodes[parent].get("constant") is None
            for parent, _role in parents
        )
    )
    non_numeric_constant_operand = False
    def basic_index_literal(value):
        # ``...`` is a basic-index literal: the full-extent selector that
        # ``normalize_basic_index`` resolves against the parent's rank.
        if value is None or value is Ellipsis or isinstance(value, int):
            return True
        if isinstance(value, slice):
            return all(part is None or isinstance(part, int)
                       for part in (value.start, value.stop, value.step))
        if isinstance(value, tuple):
            return all(basic_index_literal(part) for part in value)
        return False

    for parent, role in parents:
        if parent not in graph.G:
            continue
        parent_data = graph.G.nodes[parent]
        if str(parent_data.get("type")) not in {
            "Const", "const", "Constant",
        }:
            continue
        # A slice tuple is index metadata for a numerical load/store, not
        # a non-numeric data operand that makes the operation coordinator-only.
        if (
            node_type in {"Indexed", "IndexedStore"}
            and str(role) == "index"
            and basic_index_literal(_constant_value(parent_data))
        ):
            continue
        # A dtype spelling (``x.to_dtype("int64")``) is operation metadata
        # for a numerical cast, not a non-numeric data operand.  Treating it
        # as one severed every cast from its region: ``cell_index`` and
        # ``upper`` in the periodic terrain became unproduced region feeds.
        if (
            str(data.get("op") or node_type).casefold()
            in _DTYPE_CAST_OPERATIONS
            and isinstance(_constant_value(parent_data), str)
        ):
            continue
        try:
            flatten_tensor_constant(parent_data.get("constant"))
        except (TypeError, ValueError):
            non_numeric_constant_operand = True
            break
    # REMOVED 2026-09-30: the ``static_scalar_expression`` deferral.
    #
    # Until this date a BinOp/UnaryOp/Compare whose operands were ALL scalars
    # (constants, accessors, or an Input marked ``value_kind == "scalar"``) was
    # classified as dispatch metadata: coordinator bookkeeping, left out of
    # every region and expected to be evaluated in place by whichever consumer
    # read it (an index, a condition).  It arrived 2026-07-28 with commit
    # 87a867ea ("Enforce complete demo execution inside AST root") and stated
    # no reason beyond sitting in this list of Python-syntax constructs.
    #
    # What it did wrong.  A scalar Input is marked by the reducer for a
    # parameter annotated bool/bytes/complex/float/int/str OR given a scalar
    # default (``dt=0.5``).  For such a parameter ``k + 1`` was deferred to a
    # LATER phase: ``fortran_c_shell._graph_control_expression`` rebuilds an
    # expression tree from the node's graph parents where a consumer needs it.
    # Only some consumers call it (branch predicates, scalar field writes).
    # A returned ``k + 1``, or a scalar feeding a tensor op
    # (``x * (dt * 0.5)``), reaches no such consumer and had no producer.
    # That phase re-derives the value per consumer, which its own docstring
    # warns gives one source value a second physical definition and, for a
    # loop-carried update, a stale one.  The function then carried an
    # unnamed, unused formal and the
    # full-native gate rejected it.  ``def f(k): return k + 1`` compiled and
    # ``def f(k: int): return k + 1`` did not, though they are one program.
    # The decision to defer lived only in this classifier's private cache, so
    # no row on the identity book told the consumer it owed the computation:
    # a morph with no edge (tools/compiler_probes/
    # probe_annotated_scalar_parameter.py; docs/
    # DECISION_scalar_expression_deferral_2026-09-30.md).
    #
    # Rule now: a scalar expression is ordinary numerical work.  It resolves
    # at compile time when its operands are known (constant folding is
    # another pass) and otherwise is computed ONCE by the region that owns it,
    # as soon as its operands are available (the planner's dependency
    # order), and every consumer reads that one published value.  Nothing is
    # re-derived per consumer and nothing is deferred to the latest use.
    #
    # Evidence for removing it (all compile-only, 2026-09-30): with the rule
    # off, 14 programs compiled and each annotated form matched its plain
    # twin; with it on, 7 annotated/defaulted forms were rejected.  The
    # concordance audit tool gave identical rows and findings on all six of
    # its cases either way.
    #
    # If a scalar op turns out to cost a needless shader dispatch on a GPU
    # backend, fix that in the deployment profile of that backend, with the
    # producer recorded on the book.  Do not restore a deferral no row
    # records.
    coordinator_accessor = (
        str(data.get("op") or node_type) in ACCESSOR_OPERATORS
    )
    # ``<<``/``>>`` are ordinary numeric tensor operators (bitfield extraction
    # in a byte decoder, for example), not inherently coordinator-side. A
    # genuine *scalar* shift -- address/index arithmetic -- was once caught by
    # the scalar-expression deferral (removed 2026-09-30, see above), exactly
    # as a scalar ``&`` was; there is deliberately no ``coordinator_bitand``.
    # Special-casing every shift as coordinator metadata additionally swallowed
    # tensor-parallel shifts, which then never reached a numeric region.
    coordinator_boolean_not = (
        isinstance(expression, ast.UnaryOp)
        and isinstance(expression.op, ast.Not)
    )
    chained_comparison = (
        isinstance(expression, ast.Compare)
        and len(expression.ops) > 1
    )
    python_shape_index = (
        isinstance(expression, ast.Subscript)
        and isinstance(expression.value, ast.Attribute)
        and expression.value.attr == "shape"
    )
    # Loop-target initializers are coordinator state setup.  Their values may
    # use ordinary tensor operations, but emitting them as an independent GPU
    # region duplicates work before the retained loop owns the binding.
    loop_target_initializers = getattr(
        graph,
        "_dispatch_loop_target_initializers",
        None,
    )
    if loop_target_initializers is None:
        loop_target_initializers = frozenset(
            int(initializer)
            for _candidate_id, candidate in graph.G.nodes(data=True)
            if isinstance(
                candidate.get("expr_obj"),
                (ast.For, ast.While),
            )
            for initializer in (
                (candidate.get("attributes") or {})
                .get("loop_target_initials", {})
                .values()
            )
        )
        graph._dispatch_loop_target_initializers = (
            loop_target_initializers
        )
    loop_target_initializer = node_id in loop_target_initializers
    attributes = data.get("attributes") or {}
    ungrounded_tensor_method = (
        attributes.get("tensor_candidate") is not None
        and attributes.get("tensor") is None
        and str(attributes.get("tensor_candidate")) in {"split"}
        and isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Attribute)
    )
    # A builtin Call naming a catalogued operator in ``tensor`` computes
    # exactly as the operator it names: ``max(a, b)`` is numeric work as
    # surely as ``a + b``.  Classifying these coordination severed every
    # carried max reduction -- and every post-loop ``abs``/``max``
    # temporary -- from its region while the adds flowed.  Method calls
    # (an Attribute callee: ``d.get``) and linked references stay
    # coordinator work; those have their own machinery.
    if (
        node_type in {"Call", "call"}
        and str(
            (data.get("attributes") or {}).get("tensor") or ""
        ) in abstract_tensor_funcs
        and not isinstance(
            getattr(expression, "func", None), ast.Attribute
        )
        and not any(
            (data.get("attributes") or {}).get(name) is not None
            for name in ("callee_ref", "method_ref", "class_ref")
        )
    ):
        return Rule.CATALOGUED_BUILTIN_CALL
    if (
        str(data.get("op") or node_type).casefold() == "getattr"
        and str(attributes.get("tensor") or "")
        in abstract_tensor_funcs
        and attributes.get("tensor_candidate") == attributes.get("tensor")
        and not any(
            attributes.get(name) is not None
            for name in ("callee_ref", "method_ref", "class_ref")
        )
    ):
        # A receiver-grounded NumPy/Torch tensor property is numerical work,
        # despite retaining its authored Attribute syntax.  The source
        # candidate plus the exact receiver tensor descriptor is the proof;
        # an ordinary object field with the same name never reaches here.
        return Rule.GROUNDED_TENSOR_PROPERTY
    if node_type == "BitLength" and bool(
        (data.get("attributes") or {}).get("python_scalar_intrinsic")
    ):
        return Rule.SCALAR_INTRINSIC
    # The disjuncts of the former single ``return``, in their order: the
    # first that holds names the rule.
    if bool(
        (data.get("attributes") or {}).get(
            "coordinator_short_circuit"
        )
    ):
        return Rule.COORDINATOR_SHORT_CIRCUIT
    if any(
        (data.get("attributes") or {}).get(name) is not None
        for name in (
            "callee_ref",
            "method_ref",
            "class_ref",
        )
    ):
        return Rule.LINKED_REFERENCE
    if (
        (data.get("attributes") or {}).get(
            "static_python_reference"
        ) is not None
        and (
            node_type not in abstract_tensor_funcs
            or (data.get("attributes") or {}).get(
                "operator_reference_node"
            ) is None
        )
    ):
        return Rule.STATIC_PYTHON_REFERENCE
    if node_type in {
        "Input",
        "input",
        "Const",
        "const",
        "Constant",
        "Store",
        "store",
        "Output",
        "output",
        "Return",
        "return",
        "Call",
        "StaticReference",
        "SetAttr",
        "DelAttr",
        "DelItem",
        "Phi",
        "LoopExit",
        "LoopStateTransition",
        "LoopResult",
        "LoopStatePort",
        "LoopAggregateResult",
        "Yield",
        "YieldFrom",
        "no_grad",
    }:
        return Rule.COORDINATOR_NODE_TYPE
    if (
        isinstance(data.get("expr_obj"), ast.Call)
        and isinstance(data["expr_obj"].func, ast.Attribute)
        and (
            node_type not in abstract_tensor_funcs
            or ungrounded_tensor_method
        )
    ):
        return Rule.METHOD_CALL
    if bool(
        (data.get("attributes") or {}).get(
            "contextual_requirement"
        )
    ):
        return Rule.CONTEXTUAL_REQUIREMENT
    if _is_ast_metadata_node(graph, node_id):
        return Rule.AST_METADATA
    if isinstance(
        expression,
        (
            ast.expr_context,
            ast.operator,
            ast.unaryop,
            ast.boolop,
            ast.cmpop,
            ast.Slice,
            ast.With,
            ast.withitem,
            ast.If,
            ast.Match,
            ast.Raise,
            ast.Assert,
            ast.Pass,
            ast.Break,
            ast.Continue,
            ast.Try,
            ast.ExceptHandler,
            ast.For,
            ast.While,
            ast.comprehension,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
            ast.GeneratorExp,
            ast.JoinedStr,
            ast.FormattedValue,
            ast.IfExp,
            ast.BoolOp,
            ast.Lambda,
            ast.Starred,
            ast.Tuple,
            ast.List,
            ast.Set,
            ast.Dict,
        ),
    ):
        return Rule.PYTHON_SYNTAX
    if isinstance(expression, ast.Attribute):
        return Rule.ATTRIBUTE
    if python_routing_index:
        return Rule.PYTHON_ROUTING_INDEX
    if python_shape_index:
        return Rule.PYTHON_SHAPE_INDEX
    if compares_none:
        return Rule.COMPARES_NONE
    if non_numeric_constant_operand:
        return Rule.NON_NUMERIC_CONSTANT_OPERAND
    if coordinator_accessor:
        return Rule.COORDINATOR_ACCESSOR
    if coordinator_boolean_not:
        return Rule.COORDINATOR_BOOLEAN_NOT
    if chained_comparison:
        return Rule.CHAINED_COMPARISON
    if loop_target_initializer:
        return Rule.LOOP_TARGET_INITIALIZER
    if (
        data.get("type") == "Load"
        and (data.get("attributes") or {}).get("source_type") == "Name"
    ):
        return Rule.NAME_LOAD
    return Rule.NUMERICAL_WORK
