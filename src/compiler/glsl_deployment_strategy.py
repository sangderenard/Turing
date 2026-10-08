"""Deployment-strategy stage for ProcessGraphs.

Despite the module's filename, ``strategize_shell_deployment`` (formerly
``strategize_glsl_deployment``) is not GLSL-specific: it is the single
compilation choke point every backend passes through -- c, python, glsl,
fortran-via-precompile_only, webgl, and (once registered) webgpu all
funnel through this one control-planning stage before diverging into
backend-specific emission later in the pipeline. See
``accelerator_backends/aot_compile.py`` for the full pipeline this stage
sits in.
"""

from __future__ import annotations

import ast
import builtins
import concurrent.futures
import copy
import gc
import hashlib
import operator
import os
import re
import sys
import time
import traceback
from collections import deque
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence
from types import ModuleType, SimpleNamespace

import networkx as nx
import numpy as np


from .deployment_fifo import DeploymentFIFO
from .control_source import (
    CallBlock,
    ControlExpression,
    ControlOverlayScopeError,
    ControlProgram,
    LoopBlock,
    ParallelDeployment,
    SequenceBlock,
    StatementBlock,
    StateMachineTick,
    StreamPublishBlock,
    ValidationBlock,
    WhileBlock,
    overlay_scheduled_control,
    place_validations_after_region_producers,
    project_control_regions,
)
from .hierarchical_control import compose_hierarchical_control
from .process_graph_value_ids import next_process_value_id
from .hierarchical_plan import (
    PlanCall,
    PlanClosure,
    PlanLine,
    assign_hierarchy_ids,
    reduce_hierarchy_identities,
)

from ..common.tensors.topological_reducer import (
    _append_operand,
    _set_operands,
    _widest_element,
    indexed_store_site_span,
)
# Step 9 (plan 100, 1.2): the causes this module's operand rewrites record.
from .concordance_declarations import (
    AGGREGATE_FORMAL_MEMBERS as _AGGREGATE_FORMAL_MEMBERS,
    AGGREGATE_MEMBER as _AGGREGATE_MEMBER,
    BOUND_RECEIVER as _BOUND_RECEIVER,
    CALLSITE_FOLD_LITERAL as _CALLSITE_FOLD_LITERAL,
    CALLSITE_FOLD_REMOVE_NODE as _CALLSITE_FOLD_REMOVE_NODE,
    CALLSITE_FOLD_REPLACE_ALIAS as _CALLSITE_FOLD_REPLACE_ALIAS,
    DISPATCH_STORE as _DISPATCH_STORE_TRANSFORM,
    PROJECTION_TO_LEAF as _PROJECTION_TO_LEAF,
    REGION_BOUNDARY_INPUT as _REGION_BOUNDARY_INPUT,
    SCALAR_INTRINSIC_RECEIVER as _SCALAR_INTRINSIC_RECEIVER,
    UNBROADCAST_CHAIN as _UNBROADCAST_CHAIN,
)
from .loop_composer import (
    LoopBackendCapabilities,
    LoopComposer,
    LoopStrategy,
    _destructure_loop_target,
    _retarget_cached_value_ids,
    analyze_shader_loop_reductions,
    bind_control_deployments_to_regions,
    evaporate_unrolled_loops,
    materialize_retained_loop_ports,
)
from .loop_ir import MODELLED_STATE_OPERATORS as _MODELLED_STATE_OPERATORS
from .loop_ir import LoopStateEffectMode
from .process_graph_callable import EphemeralProcessGraphCallable
from .process_graph_fusion import (
    DispatchRegion,
    dispatch_region_to_fused_program,
    extract_clean_process_subgraph,
    reduce_scheduled_shader_regions,
)
from .shell_reference_tables import build_shell_reference_tables
from ..transmogrifier.function_table import FunctionReference
from ..common.tensors.abstraction import AbstractTensor, tensor_identity
from ..common.tensors.accelerator_backends.glsl_fused_network import (
    GLSLFusedProgramNetwork,
)
from ..common.tensors.accelerator_backends.c_primitive_program import (
    CapturedFusedProgram,
    compile_recorded_fused_tape,
)
from ..common.tensors.accelerator_backends.glsl_backend import (
    InstalledGLSLControlShell,
    build_control_shader_artifact,
    compile_captured_fused_program,
    compose_control_shader,
    dispatch_stats,
    emit_multi_output_program_source,
    execute_captured_fused_program,
)
from ..common.tensors.accelerator_backends.glsl_tensor_backend import (
    GLSLTensorOperations,
)
from ..common.tensors.accelerator_backends.precompile_observer_backend import (
    PrecompileObserverTensorOperations,
)
from ..common.tensors.autograd import GradTape, autograd
from ..common.tensors.fused_ir import (
    FusedProgram,
    Meta,
    OpStep,
    canonical_elementwise_op,
    flatten_tensor_constant,
    ordered_feed_ids,
)
from ..common.tensors.operator_catalog import (
    ACCESSOR_OPERATORS,
    CREATION_OPERATORS,
)
from ..transmogrifier.graph.edge_roles import (
    ordered_arguments,
    positional_argument_index,
)
from ..transmogrifier.graph.graph_deep_compiler import GraphDeepCompiler
from ..transmogrifier.operator_defs import (
    abstract_tensor_funcs,
    abstract_tensor_sigs,
)

# Relocated modules: names re-exported so existing importers keep working.
from .source_node_facts import (
    _ast_source_location as _ast_source_location,
    _ast_source_signature as _ast_source_signature,
    _constant_value as _constant_value,
    _static_python_value as _static_python_value,
    _is_runtime_value_id as _is_runtime_value_id,
)
from .call_argument_binding import (
    _positional_argument_index as _positional_argument_index,
    _method_parameter_layout as _method_parameter_layout,
    _callsite_parameter_name as _callsite_parameter_name,
    _record_aggregate_ledger_lookup as _record_aggregate_ledger_lookup,
    _authored_aggregate_leaves as _authored_aggregate_leaves,
    _expanded_callsite_argument_edges as _expanded_callsite_argument_edges,
    _declared_output_terminals as _declared_output_terminals,
    _call_arguments as _call_arguments,
)
from .region_scheduling import (
    _dependency_order as _dependency_order,
    _topological_region_order as _topological_region_order,
    _topological_region_schedule as _topological_region_schedule,
    _control_dependency_value_ids as _control_dependency_value_ids,
    _DEPENDENCY_LEVEL_CACHE as _DEPENDENCY_LEVEL_CACHE,
    _dependency_levels as _dependency_levels,
)
from .deployment_profiler import (
    DeploymentErrorBuffer as DeploymentErrorBuffer,
    DeploymentProfiler as DeploymentProfiler,
    _shell_profile_name as _shell_profile_name,
    _attach_profiler as _attach_profiler,
    _profile_event as _profile_event,
)
from .formal_shape_ledger import (
    _FORMAL_LITERAL_CONFLICT as _FORMAL_LITERAL_CONFLICT,
    _publish_formal_literal as _publish_formal_literal,
    _publish_formal_shape as _publish_formal_shape,
    _proven_formal_shape as _proven_formal_shape,
    _proven_formal_literal as _proven_formal_literal,
)
from .planner_concordance_posts import (
    _post_deployment_region as _post_deployment_region,
    _post_if_changed as _post_if_changed,
    _argument_identity_cells as _argument_identity_cells,
    _formal_identity_cells as _formal_identity_cells,
    _post_planner_specialization as _post_planner_specialization,
    _post_call_binding as _post_call_binding,
    _post_callsite_activation as _post_callsite_activation,
    _post_control_specialization as _post_control_specialization,
    _post_pruned_return_sites as _post_pruned_return_sites,
    _fold_literal_source_cells as _fold_literal_source_cells,
    _post_identity_table_mutation as _post_identity_table_mutation,
    _post_copy_planner_specializations as _post_copy_planner_specializations,
    _materialize_planner_specializations as _materialize_planner_specializations,
    _callsite_descriptor_receipt as _callsite_descriptor_receipt,
    _post_callsite_return_member as _post_callsite_return_member,
)
from .planned_shell_tree import (
    _walk_planned_shells as _walk_planned_shells,
    _deployment_program_table_lines as _deployment_program_table_lines,
)
from .receiver_resolution_prepasses import (
    _lower_python_scalar_intrinsics as _lower_python_scalar_intrinsics,
    _refresh_specialized_python_precision as _refresh_specialized_python_precision,
    _resolve_bound_function_references as _resolve_bound_function_references,
    _resolve_grounded_method_references as _resolve_grounded_method_references,
    _resolve_grounded_tensor_operations as _resolve_grounded_tensor_operations,
)
from .source_control_retention import (
    _retain_source_control_records as _retain_source_control_records,
    _retain_source_sequence_mutation_records as _retain_source_sequence_mutation_records,
    _source_control_records as _source_control_records,
    _source_control_expression as _source_control_expression,
    _retained_control_value_id as _retained_control_value_id,
    _optional_presence_control_expression as _optional_presence_control_expression,
    _record_field_state_keys as _record_field_state_keys,
    _repair_missing_phi_initial_identities as _repair_missing_phi_initial_identities,
)
from .tensor_descriptor_query import (
    _authored_aggregate_descriptor_tree as _authored_aggregate_descriptor_tree,
    _invalidate_tensor_descriptor_dependents as _invalidate_tensor_descriptor_dependents,
    _structured_output_descriptor as _structured_output_descriptor,
    _operand_descriptor as _operand_descriptor,
    _DTYPE_CAST_OPERATIONS as _DTYPE_CAST_OPERATIONS,
    _attribute_axis as _attribute_axis,
    _ELEMENTWISE_BINARY_OPERATIONS as _ELEMENTWISE_BINARY_OPERATIONS,
    _deepest_fact as _deepest_fact,
    descriptor_states_a_shape as descriptor_states_a_shape,
    _ARITHMETIC_LOOP_TARGET_DTYPE as _ARITHMETIC_LOOP_TARGET_DTYPE,
    _arithmetic_loop_target_owner as _arithmetic_loop_target_owner,
    _POLYMORPHIC_FORMAL_SCANS as _POLYMORPHIC_FORMAL_SCANS,
    _owner_has_polymorphic_formal as _owner_has_polymorphic_formal,
    _DescriptorQuery as _DescriptorQuery,
    _tensor_descriptor as _tensor_descriptor,
    _promoting_sides as _promoting_sides,
    _dtype_argument_name as _dtype_argument_name,
    _tensor_descriptor_rule as _tensor_descriptor_rule,
)
from .callsite_literal_specialization import (
    _source_static_value as _source_static_value,
    _source_static_literal as _source_static_literal,
    _propagate_callsite_planner_specializations as _propagate_callsite_planner_specializations,
)
from .captured_region_programs import (
    _captured_storage_meta as _captured_storage_meta,
    _compiler_input_name as _compiler_input_name,
    _capture_storage_identity as _capture_storage_identity,
    _remap_captured_program as _remap_captured_program,
    _remap_captured_all_ids as _remap_captured_all_ids,
    _wire_planned_step_inputs as _wire_planned_step_inputs,
    _collapse_planned_collection_materializations as _collapse_planned_collection_materializations,
    _project_captured_program as _project_captured_program,
    _resolve_binding_name as _resolve_binding_name,
    _capture_feed_aliases as _capture_feed_aliases,
    _unique_runtime_feed_aliases as _unique_runtime_feed_aliases,
)
from .validation_control import (
    _identity_parameter_projection as _identity_parameter_projection,
    _synthetic_device_scalar_shell as _synthetic_device_scalar_shell,
    _validation_control_blocks as _validation_control_blocks,
    _place_validation_blocks as _place_validation_blocks,
)
from .conditional_control_programs import (
    CompilationSubdivisionRequired as CompilationSubdivisionRequired,
    _loop_induction_name as _loop_induction_name,
    _find_nested_loop_node_ids_in_block as _find_nested_loop_node_ids_in_block,
    _loop_reduction_nesting_hints as _loop_reduction_nesting_hints,
    _branch_compartments as _branch_compartments,
    _ordinary_conditional_control_programs as _ordinary_conditional_control_programs,
    _nest_region_less_conditionals as _nest_region_less_conditionals,
    _ordinary_conditional_nesting as _ordinary_conditional_nesting,
    _source_control_nesting_hints as _source_control_nesting_hints,
    _overlay_control_or_require_subdivision as _overlay_control_or_require_subdivision,
    _control_partition_keys as _control_partition_keys,
)
from .dispatch_metadata_classifier import (
    _is_ast_metadata_node as _is_ast_metadata_node,
    _semantic_children as _semantic_children,
    _inert_routing_nodes as _inert_routing_nodes,
    _DISPATCH_METADATA_CACHE_SCHEMA as _DISPATCH_METADATA_CACHE_SCHEMA,
    _dispatch_metadata_node_classifier as _dispatch_metadata_node_classifier,
    _executable_rules as _executable_rules,
    _ExecutableRules as _ExecutableRules,
    _EXECUTABLE_RULES as _EXECUTABLE_RULES,
    _is_dispatch_metadata_node as _is_dispatch_metadata_node,
    _is_dispatch_metadata_node_impl as _is_dispatch_metadata_node_impl,
    _dispatch_metadata_rule as _dispatch_metadata_rule,
)
from .dispatch_region_partition import (
    _atomic_region_node_order as _atomic_region_node_order,
    _shader_fusible_node_ids as _shader_fusible_node_ids,
    _precision_indivisible_node_groups as _precision_indivisible_node_groups,
    _closure_routing_dependencies as _closure_routing_dependencies,
)
from .dispatch_subgraph_extraction import (
    _subgraph_reduction_digest as _subgraph_reduction_digest,
    _structural_region_program_from_subgraph as _structural_region_program_from_subgraph,
    _dispatch_subgraph as _dispatch_subgraph,
)
from .structural_fold import (
    _repair_missing_aggregate_leaf_projections as _repair_missing_aggregate_leaf_projections,
    _UNRESOLVED_STRUCTURAL_VALUE as _UNRESOLVED_STRUCTURAL_VALUE,
    _StructuralValueAlias as _StructuralValueAlias,
    _ProgramABIValueFact as _ProgramABIValueFact,
    _contains_program_abi_fact as _contains_program_abi_fact,
    _fold_callsite_structural_values as _fold_callsite_structural_values,
    _follow_declared_value_source as _follow_declared_value_source,
    _retarget_all_cached_value_ids as _retarget_all_cached_value_ids,
)
from .scheduled_capture_coordinator import (
    _scheduled_capture_backend as _scheduled_capture_backend,
    _use_scheduled_capture_backend as _use_scheduled_capture_backend,
    _planned_capture_context as _planned_capture_context,
    _observe_process_graph_node as _observe_process_graph_node,
    _diagnostic_value_summary as _diagnostic_value_summary,
    _graph_source_binding_name as _graph_source_binding_name,
    _bind_capture_tape as _bind_capture_tape,
    _SourceReturnSignal as _SourceReturnSignal,
    _tensorize_graph_input as _tensorize_graph_input,
    _coordinate_scheduled_capture_impl as _coordinate_scheduled_capture_impl,
    _coordinate_scheduled_capture as _coordinate_scheduled_capture,
    _CompiledStructuralObject as _CompiledStructuralObject,
    _CompiledStructuralMethod as _CompiledStructuralMethod,
    _CompiledStructuralFunction as _CompiledStructuralFunction,
    _CompiledStructuralClass as _CompiledStructuralClass,
    _GreedyGeneratorStream as _GreedyGeneratorStream,
)


def _concord_consumer_operands(
    graph: Any, consumer_nodes: Any, values: Any,
) -> None:
    """Commit, per consumer node and value, which operands read it.

    Page ``consumer_operand`` row ``(read scope, node, value)`` holds the
    ``(role, ordinal)`` positions of ``node``'s operands that read
    ``value``.  Structure only: the binding each operand read is the base
    fact ``lexical_read_binding``, at ``(read scope, node, role, ordinal)``.
    """

    from ..common.tensors.topological_reducer import _operand_positions
    from .identity_concordance import current_identity_book

    scope = graph.G.graph.get("lexical_read_scope")
    if scope is None:
        return
    wanted = set(map(int, values))
    book = current_identity_book()
    page = book.page("consumer_operand")
    # A read row naming a position its consumer no longer has was orphaned
    # by an operand rewrite that bypassed ``_set_operands``: the binding the
    # operand read is lost.  Record it where the live graph and the rows
    # meet; ``CorrelationTable`` reports page ``operand_position_orphan``.
    read_page = book.page("lexical_read_binding")
    orphan_page = book.page("operand_position_orphan")
    read_rows: dict[int, list[tuple[Any, int]]] = {}
    for row in read_page.scope_rows(tuple(scope)):
        if (
            len(row) == 4 and isinstance(row[1], int)
            and read_page.latest(row) is not None
        ):
            read_rows.setdefault(int(row[1]), []).append((row[2], row[3]))
    for node_id in consumer_nodes:
        if int(node_id) not in graph.G:
            continue
        positions: dict[int, list[tuple[Any, int]]] = {}
        present: set[tuple[Any, int]] = set()
        for role, ordinal, parent in _operand_positions(
            graph.G.nodes[int(node_id)].get("parents") or ()
        ):
            present.add((role, ordinal))
            if int(parent) in wanted:
                positions.setdefault(int(parent), []).append((role, ordinal))
        for position in read_rows.get(int(node_id), ()):
            if position not in present:
                orphan_page.concord(
                    (tuple(scope), int(node_id), *position),
                    read_page.latest((tuple(scope), int(node_id), *position)),
                )
        for parent, operand_positions in positions.items():
            page.concord(
                (tuple(scope), int(node_id), parent), tuple(operand_positions),
            )


def _concord_item_operands(graph: Any, captures: Any) -> None:
    """Record, per captured ``item()`` node, the operand position it reads.

    ``float(total.item())`` in a loop body: no region computes the ``item``,
    so the region captures the item's id.  The loop carries ``total``, not
    the item id; a later id-keyed identity rewrite
    (``_rebind_recorded_scalar_identities``) replaced the capture with
    ``total``'s only definition -- the pre-loop one -- and ``total_value``
    froze at its first value (silent wrong answer).

    Page ``item_operand`` row ``(read scope, item id)`` holds ``(operand id,
    role, ordinal)``: structure only.  The control lowering reads the
    operand by the binding at that position (``lexical_read_binding``) in
    each read site's loop generation, and decides the rank-0 identity from
    the value it reads.
    """

    from ..common.tensors.topological_reducer import _operand_positions
    from .identity_concordance import current_identity_book

    scope = graph.G.graph.get("lexical_read_scope")
    if scope is None:
        return
    page = current_identity_book().page("item_operand")
    for value_id in captures:
        if int(value_id) not in graph.G:
            continue
        data = graph.G.nodes[int(value_id)]
        if str(data.get("op") or data.get("type") or "").casefold() != "item":
            continue
        operands = [
            (role, ordinal, parent)
            for role, ordinal, parent in _operand_positions(
                data.get("parents") or ()
            )
            if str(role) in {"operand", "arg:0", "value"}
        ]
        if len(operands) != 1:
            continue
        role, ordinal, operand = operands[0]
        page.concord(
            (tuple(scope), int(value_id)),
            (int(operand), str(role), int(ordinal)),
        )


def _concord_call_argument_operands(
    graph: Any, callsite_id: int, operand_edges: Mapping[int, tuple[int, str]],
) -> None:
    """Commit, per argument position of one call, which operand it is.

    Page ``call_argument_operand`` row ``(read scope, callsite, position)``
    holds the ``(role, ordinal)`` of the call node's operand that produced
    that argument binding.  One value passed twice is two operands; the
    binding each read is the base fact ``lexical_read_binding``.
    """

    from ..common.tensors.topological_reducer import _operand_positions
    from .identity_concordance import current_identity_book

    scope = graph.G.graph.get("lexical_read_scope")
    if scope is None or int(callsite_id) not in graph.G:
        return
    positions = list(_operand_positions(
        graph.G.nodes[int(callsite_id)].get("parents") or ()
    ))
    page = current_identity_book().page("call_argument_operand")
    for index, (parent, role) in operand_edges.items():
        ordinal = next((
            edge_ordinal for edge_role, edge_ordinal, edge_parent in positions
            if str(edge_role) == str(role) and int(edge_parent) == int(parent)
        ), None)
        if ordinal is not None:
            page.concord(
                (tuple(scope), int(callsite_id), int(index)), (role, ordinal),
            )


def _build_shell_hierarchy_plan(
    shell: Any,
    _memo: dict[int, PlanClosure] | None = None,
) -> PlanClosure:
    """Freeze call/region ownership before backend source composition."""

    memo = {} if _memo is None else _memo
    incumbent_plan = memo.get(id(shell))
    if incumbent_plan is not None:
        return incumbent_plan

    graph = shell.process_graph
    # The plan freezes tensor descriptors into ``PlanClosure.value_shapes``.
    # Some catalogue and callsite shells are constructed before their
    # ProgramABI descriptors arrive; freezing them at that point permanently
    # records reshape results as scalars even though the graph can now resolve
    # their structural shape tuples. Make the documented ordering invariant
    # local to the freeze operation itself. The fold is structural,
    # idempotent, and executes no numerical source code.
    if graph.G.graph.get("function_name"):
        _fold_callsite_structural_values(graph)
    # An Indexed successor used to unpack a multi-result source call is a
    # call-boundary projection, not a numerical gather owned by a region.
    # Dispatch regions can be carved before callsite shells are attached, so
    # remove these exact projection nodes when freezing final ownership. If
    # both the PlanCall and a stale region produce the same caller value, the
    # repository-SSA type linker sees two physical definitions and can
    # oscillate between their shapes indefinitely.
    def call_result_projections(value_id: int) -> set[int]:
        # Every literal-index projection below a call result is a structural
        # path onto that call's outputs: ``r[1]`` (an intermediate aggregate
        # with no physical value of its own) as much as ``r[1][0]`` (a bound
        # leaf).  None of them is an operator a region may compute.
        found: set[int] = set()
        pending = [int(value_id)]
        while pending:
            current = pending.pop()
            for successor in graph.G.successors(current):
                successor_data = graph.G.nodes[successor]
                if str(
                    successor_data.get("op")
                    or successor_data.get("type")
                    or ""
                ).casefold() != "indexed":
                    continue
                if not any(
                    str(role) in {"value", "base", "operand", "object"}
                    and int(parent) == current
                    for parent, role in successor_data.get("parents") or ()
                ):
                    continue
                # Only a literal integer index is a structural path onto a
                # call's outputs.  A slice or a tuple of slices over a tensor
                # call result (``moment[:, :, 1]``, ``m[:, :, 0:2]``) is a
                # numerical view an owning region must compute; treating it
                # as a projection left it with no producer at all.
                index_parents = tuple(
                    int(parent)
                    for parent, role in successor_data.get("parents") or ()
                    if str(role) == "index"
                )
                if len(index_parents) != 1 or index_parents[0] not in graph.G:
                    continue
                index_data = graph.G.nodes[index_parents[0]]
                if str(index_data.get("type")) not in {
                    "Const", "const", "Constant",
                }:
                    continue
                try:
                    index_value = _constant_value(index_data)
                except KeyError:
                    continue
                if not isinstance(index_value, int) or isinstance(
                    index_value, bool
                ):
                    continue
                if int(successor) in found:
                    continue
                found.add(int(successor))
                # Keep descending only while the projection is itself an
                # intermediate aggregate.  ``r[1][0]`` is a structural path
                # when ``r[1]`` is another tuple, and an ordinary element read
                # when ``r[1]`` is a tensor.  Walking through a tensor removed
                # that read from every region, so nothing computed it and the
                # caller kept a formal for it.
                if _tensor_descriptor(graph, int(successor)) is None:
                    pending.append(int(successor))
        return found

    call_result_projection_ids = {
        projection
        for callsite_id in shell.callsite_function_shells
        if int(callsite_id) in graph.G
        for projection in call_result_projections(int(callsite_id))
    }
    region_by_node = {
        int(node_id): int(region_index)
        for region_index, subgraph in enumerate(shell.dispatch_subgraphs)
        for node_id in subgraph.G.graph.get("deployment_nodes", ())
        if int(node_id) not in call_result_projection_ids
    }
    emitted_regions = set()
    items = []
    topological_nodes = _dependency_order(graph)
    order_index = {
        int(node_id): index
        for index, node_id in enumerate(topological_nodes)
    }
    def source_position(node_id: int) -> tuple[int, int] | None:
        expression = graph.G.nodes[int(node_id)].get("expr_obj")
        line = getattr(expression, "lineno", None)
        column = getattr(expression, "col_offset", None)
        if line is None:
            return None
        return int(line), int(column or 0)

    caller_identities = graph.G.graph.get("identity_table") or {}
    # Regions are emitted atomically at the position of their first member,
    # so the walk must already respect region-level dependencies.
    walk_order = _atomic_region_node_order(
        graph, topological_nodes, region_by_node,
    )

    def direct_return_value_ids(child_graph: Any) -> tuple[int, ...]:
        """Resolve one direct return by deterministic source coordinates.

        ProcessGraph return_value_nodes historically held ``id(ast_node)``
        correlations, which do not survive graph reduction/reconstruction.
        The reduced nodes retain exact source spans, so correlate the authored
        Return.value to its graph value without restoring transient ids.
        """

        return_signatures = set()
        for _candidate_id, candidate_data in child_graph.nodes(data=True):
            expression = candidate_data.get("expr_obj")
            if not isinstance(expression, ast.AST):
                continue
            for nested in ast.walk(expression):
                if not isinstance(nested, ast.Return) or nested.value is None:
                    continue
                return_signatures.add(_ast_source_signature(nested.value))
        matched = []
        for candidate_id, candidate_data in child_graph.nodes(data=True):
            expression = candidate_data.get("expr_obj")
            if not isinstance(expression, ast.AST):
                continue
            if _ast_source_signature(expression) not in return_signatures:
                continue
            value_id = int(candidate_data.get("value_id", candidate_id))
            attributes = candidate_data.get("attributes") or {}
            if (
                str(attributes.get("extraction_identity") or "") in {
                    "builtins.bytes", "builtins.bytearray", "builtins.list",
                    "builtins.tuple", "builtins.frozenset", "builtins.set",
                }
                or str(attributes.get("producer_kind") or "")
                == "aggregate_materialization"
            ):
                backing = tuple(
                    int(parent)
                    for parent, role in candidate_data.get("parents") or ()
                    if str(role) in {"arg:0", "operand", "value"}
                )
                if len(backing) == 1:
                    value_id = backing[0]
            matched.append(value_id)
        unique = tuple(dict.fromkeys(matched))
        # Multiple lexical returns need control-aware result merging; never
        # guess that they are a tuple-valued return surface.
        return unique if len(unique) == 1 else ()

    for node_id in walk_order:
        child = shell.callsite_function_shells.get(node_id)
        if child is not None:
            call_attributes = graph.G.nodes[node_id].get("attributes") or {}
            constructor_call = call_attributes.get("constructor_ref") is not None
            call_parents = _expanded_callsite_argument_edges(graph, node_id)
            parents = tuple(parent for parent, _role in call_parents)
            child_graph = child.process_graph.G
            child_identities = (
                child_graph.graph.get("identity_table") or {}
            )
            receiver_parameter, positional_parameters, _parameters = (
                _method_parameter_layout(child_graph)
            )
            argument_bindings = []
            if constructor_call and receiver_parameter is not None:
                receiver_identities = child_identities.get(
                    receiver_parameter, ()
                )
                if receiver_identities:
                    # The class-call result is the caller-owned record identity
                    # initialized by __init__/__new__.  It is both the
                    # constructor callsite and the concrete ``self`` binding;
                    # its fields remain ordinary caller-owned arena values.
                    argument_bindings.append((
                        int(node_id), int(receiver_identities[0])
                    ))
            # Which call edge produced each argument binding, whatever path
            # the body below takes: the operand's authored read is the edge's.
            operand_edges: dict[int, tuple[int, str]] = {}

            def attributed_call_parents():
                for edge_parent, edge_role in call_parents:
                    start = len(argument_bindings)
                    yield edge_parent, edge_role
                    for index in range(start, len(argument_bindings)):
                        operand_edges[index] = (int(edge_parent), str(edge_role))

            for parent, role in attributed_call_parents():
                position = _positional_argument_index(role)
                # The call's edge is the argument's identity.  (A former
                # name-history override replaced it with the source-latest
                # definition of the argument's name, which chose a continue
                # arm's value over the edge to the pre-branch value.)
                if position is not None:
                    name = (
                        positional_parameters[position]
                        if position < len(positional_parameters)
                        else None
                    )
                elif (
                    role == "operand"
                    and receiver_parameter is not None
                ):
                    name = receiver_parameter
                elif role.startswith("kw:"):
                    name = role.split(":", 1)[1]
                elif role.startswith("closure:"):
                    name = role.split(":", 1)[1]
                else:
                    name = None
                identities = child_identities.get(name, ())
                caller_leaves = _authored_aggregate_leaves(
                    graph, int(parent),
                )
                # Member formals materialized for an aggregate parameter
                # carry their exact member path.  They outlive the
                # shapeless aggregate formal, which is dropped once every
                # authored projection reads its member directly, so the
                # member binding must not depend on that formal's identity.
                member_inputs = {
                    tuple(map(int, member_attributes.get(
                        "aggregate_path",
                        (member_attributes["aggregate_index"],),
                    ))): int(member_id)
                    for member_id, member_data in child_graph.nodes(data=True)
                    if member_data.get("type") == "Input"
                    and str(
                        (member_data.get("attributes") or {}).get(
                            "aggregate_parent_binding"
                        )
                    ) == str(name)
                    for member_attributes in (
                        member_data.get("attributes") or {},
                    )
                    if member_attributes.get("aggregate_index") is not None
                }
                if member_inputs:
                    caller_member_ids: dict[tuple[int, ...], int] = {}

                    def record_caller_members(
                        value_id: int, path: tuple[int, ...] = (),
                    ) -> None:
                        member_leaves = _authored_aggregate_leaves(
                            graph, int(value_id),
                        )
                        if not member_leaves:
                            if path:
                                caller_member_ids[path] = int(value_id)
                            return
                        for member_index, member_id in enumerate(member_leaves):
                            record_caller_members(
                                int(member_id), (*path, int(member_index)),
                            )

                    record_caller_members(int(parent))
                    missing_paths = tuple(
                        path for path in member_inputs
                        if path not in caller_member_ids
                    )
                    if missing_paths:
                        raise ValueError(
                            f"aggregate call binding for {name!r} names "
                            f"paths {missing_paths!r} absent from caller "
                            f"paths {tuple(caller_member_ids)!r}"
                        )
                    for path in sorted(member_inputs):
                        argument_bindings.append((
                            int(caller_member_ids[path]), member_inputs[path],
                        ))
                if identities and not member_inputs:
                    # With member formals bound above, the aggregate itself
                    # has no physical storage to pass: the callee's formal is
                    # the tuple of its members.  Binding it too would turn an
                    # authored tuple node into an untyped caller argument.
                    child_input = int(identities[0])
                    child_input_data = child_graph.nodes.get(child_input, {})
                    if (child_input_data.get("attributes") or {}).get(
                        "structural_specialization"
                    ):
                        # This exact formal has already become a callee-owned
                        # literal. Its source identity remains in the ledger,
                        # but it no longer accepts a runtime argument frame.
                        continue
                    argument_bindings.append((parent, child_input))
                    child_leaves = _authored_aggregate_leaves(
                        child.process_graph, child_input,
                    )
                    if caller_leaves or child_leaves:
                        if len(caller_leaves) != len(child_leaves):
                            raise ValueError(
                                f"aggregate call binding for {name!r} has "
                                "different caller/callee arity: "
                                f"{len(caller_leaves)} != {len(child_leaves)}"
                            )
                        argument_bindings.extend(zip(
                            caller_leaves, child_leaves, strict=True,
                        ))
            bound_child_inputs = {
                int(child_input)
                for _caller, child_input in argument_bindings
            }
            for child_input, child_data in child_graph.nodes(data=True):
                child_attributes = child_data.get("attributes") or {}
                if (
                    child_data.get("type") != "Input"
                    or child_attributes.get("binding_kind") != "external"
                    or int(child_input) in bound_child_inputs
                ):
                    continue
                name = child_attributes.get("binding_name")
                call_position = source_position(int(node_id))
                candidates = [
                    int(definition)
                    for definition in caller_identities.get(name, ())
                    if (
                        int(definition) in order_index
                        and not nx.has_path(
                            graph.G, int(node_id), int(definition)
                        )
                        and (
                            (
                                call_position is not None
                                and source_position(int(definition))
                                is not None
                                and source_position(int(definition))
                                < call_position
                            )
                            or (
                                (
                                    call_position is None
                                    or source_position(int(definition))
                                    is None
                                )
                                and order_index[int(definition)]
                                < order_index[int(node_id)]
                            )
                        )
                    )
                ]
                if candidates:
                    argument_bindings.append((
                        max(
                            candidates,
                            key=lambda definition: (
                                source_position(definition)
                                or (-1, order_index[definition])
                            ),
                        ),
                        int(child_input),
                    ))
            child_outputs = tuple(
                int(child_identities[name][-1])
                for name in child_graph.graph.get(
                    "function_outputs", ()
                )
                if child_identities.get(name)
            )
            if not child_outputs:
                # ``return bytes(out)`` and other direct aggregate-return
                # expressions have no authored assignment name, so
                # function_outputs is legitimately empty.  Capture analysis
                # already records their exact ProcessGraph value ids; use
                # that structural return surface for the call binding instead
                # of silently turning a value-returning call into void.
                child_outputs = tuple(map(int, getattr(
                    child, "_captured_return_value_ids", (),
                )))
            if not child_outputs:
                child_outputs = direct_return_value_ids(child_graph)
            # A returned aggregate is a structural path, not one position in
            # a flat list.  ``return total, (hub, angle), valid`` has leaf
            # paths (0,), (1, 0), (1, 1), (2,); the caller's ``r[1][0]`` is
            # the projection path (1, 0).  Expanding the callee leaves and
            # zipping them against the caller's first-level projections
            # shifts every member after the aggregate onto the wrong result.
            child_output_paths: dict[tuple[int, ...], int] = {}

            def expand_child_output(
                path: tuple[int, ...], value_id: int,
            ) -> None:
                leaves = _authored_aggregate_leaves(
                    child.process_graph, int(value_id),
                )
                if not leaves:
                    child_output_paths[path] = int(value_id)
                    return
                for index, leaf in enumerate(leaves):
                    expand_child_output((*path, int(index)), int(leaf))

            for position, child_output in enumerate(child_outputs):
                expand_child_output((int(position),), int(child_output))

            def literal_projection_index(successor: int) -> int | None:
                successor_data = graph.G.nodes[int(successor)]
                if str(successor_data.get("op", "")).lower() != "indexed":
                    return None
                index_parents = tuple(
                    int(parent)
                    for parent, role in (
                        successor_data.get("parents") or ()
                    )
                    if str(role) == "index"
                )
                if len(index_parents) != 1:
                    return None
                index_data = graph.G.nodes[index_parents[0]]
                index_value = index_data.get("constant")
                if index_value is None:
                    index_value = (
                        index_data.get("attributes") or {}
                    ).get("value")
                if isinstance(index_value, bool) or not isinstance(
                    index_value, int
                ):
                    return None
                return int(index_value)

            caller_projection_paths: dict[tuple[int, ...], int] = {}

            def collect_caller_projections(
                path: tuple[int, ...], value_id: int,
            ) -> None:
                for successor in graph.G.successors(int(value_id)):
                    index = literal_projection_index(int(successor))
                    if index is None:
                        continue
                    projection_path = (*path, index)
                    caller_projection_paths[projection_path] = int(successor)
                    collect_caller_projections(projection_path, int(successor))

            collect_caller_projections((), int(node_id))
            # This is the operation that owns the exact relation between a
            # source call's authored result path and the caller projection
            # value.  Publish it now.  Later SSA stages must not reconstruct
            # that relation from target names or from dependency enumeration,
            # either of which can change when a projection becomes loop state.
            from .identity_concordance import current_identity_book

            call_result_projection_page = current_identity_book().page(
                "call_result_projection_concordance"
            )
            for projection_path, caller_projection_id in (
                caller_projection_paths.items()
            ):
                call_result_projection_page.concord(
                    (int(node_id), int(caller_projection_id)),
                    tuple(map(int, projection_path)),
                )
            if tuple(child_output_paths) == ((0,),) and not (
                graph.G.nodes[int(node_id)].get("attributes") or {}
            ).get("authored_return_container"):
                result_bindings = (
                    (child_output_paths[(0,)], int(node_id)),
                )
            else:
                result_bindings = tuple(
                    (child_output_paths[path], caller_projection_paths[path])
                    for path in child_output_paths
                    if path in caller_projection_paths
                )
            child_outputs = tuple(child_output_paths.values())
            _concord_call_argument_operands(graph, int(node_id), operand_edges)
            items.append(PlanCall(
                int(node_id),
                _build_shell_hierarchy_plan(child, memo),
                parents,
                tuple(caller for _callee, caller in result_bindings),
                tuple(argument_bindings),
                tuple(result_bindings),
                tuple(
                    int(plan.loop.node_id)
                    for plan in sorted(
                        (
                            plan
                            for plan in shell.loop_plans
                            if int(node_id) in plan.loop.body_nodes
                        ),
                        key=lambda plan: -len(plan.loop.body_nodes),
                    )
                ),
            ))
            continue
        region_index = region_by_node.get(int(node_id))
        if (
            region_index is None
            or region_index in emitted_regions
        ):
            continue
        emitted_regions.add(region_index)
        subgraph = shell.dispatch_subgraphs[region_index]
        # The dispatch subgraph was carved before loop evaporation could
        # replace the nodes it named.  The live graph is the authority for
        # what this region still contains, so drop members it no longer
        # holds instead of indexing them and raising a bare KeyError.
        region_nodes = tuple(
            int(value)
            for value in subgraph.G.graph.get("deployment_nodes", ())
            if int(value) not in call_result_projection_ids
            and int(value) in graph.G
        )
        original_region_node_set = set(region_nodes)
        carried_initial_boundaries = {
            int(initial)
            for loop_plan in getattr(shell, "loop_plans", ())
            for _name, initial, updated in loop_plan.loop.carried_bindings
            if int(updated) in original_region_node_set
            and int(initial) != int(updated)
        }
        # A carried initializer is the entry arm of the coordinator-owned Phi.
        # If the numerical compartment also owns that initializer, its body
        # recomputes from the iteration-zero value on every trip.  Cut it from
        # this region so the ordinary capture calculation below binds the
        # current Phi value into the update operation.
        region_nodes = tuple(
            value_id for value_id in region_nodes
            if value_id not in carried_initial_boundaries
        )
        region_node_set = set(region_nodes)

        def numerical_region_parents(
            node_id: int,
        ) -> tuple[tuple[int, str], ...]:
            """Expose structural tensor-call leaves at the numerical ABI.

            A Python list/tuple is one source argument, but stack/cat consume
            each tensor element as a separate numerical operand.  The graph's
            aggregate leaf ledger is the exact authored correspondence; the
            container identity itself is coordinator storage and must not
            replace those producers at a numerical region cut.
            """

            node_data = graph.G.nodes.get(int(node_id))
            if node_data is None:
                # Loop evaporation removes the nodes it replaced, and a region
                # view carved before that still names them.  The graph is the
                # authority this recomputation exists to consult, so a node it
                # no longer holds contributes no parents rather than raising.
                return ()
            operation = str(
                node_data.get("op") or node_data.get("type") or ""
            )
            expanded: list[tuple[int, str]] = []
            for parent, role in node_data.get("parents") or ():
                parent = int(parent)
                parent_data = graph.G.nodes.get(parent, {})
                attributes = parent_data.get("attributes") or {}
                leaves = tuple(map(
                    int,
                    attributes.get("aggregate_leaf_value_ids", ()),
                ))
                if (
                    operation in {"stack", "cat", "concat"}
                    and str(role).startswith("arg")
                    and attributes.get("producer_kind") == "aggregate"
                    and attributes.get("aggregate_kind") in {"list", "tuple"}
                    and leaves
                ):
                    expanded.extend(
                        (leaf, f"{role}:element:{index}")
                        for index, leaf in enumerate(leaves)
                    )
                else:
                    expanded.append((parent, str(role)))
            return tuple(expanded)

        # Loop-port materialization can rewrite a region's live parents after
        # its dispatch subgraph was first carved. Recompute region captures
        # from the authoritative current graph instead of
        # reusing stale deployment_inputs from the numeric view.
        region_captures = tuple(dict.fromkeys((
            *(
                int(parent)
                for node_id in region_nodes
                for parent, _role in numerical_region_parents(node_id)
                if int(parent) not in region_node_set
            ),
        )))

        # A region capture is not necessarily a runtime input. Structural
        # arguments such as ``transpose(-2, -1)`` often sit just outside the
        # tensor region as a tiny constant-expression chain (Const -> neg, or
        # several constants -> Tuple). Preserve that source closure in the
        # region instead of promoting its terminal to an unexplained ABI
        # parameter. This is graph-source pursuit, not value specialization:
        # only expressions whose complete producer closure is literal are
        # admitted here.
        # Keep evaluator state separate from its values.  A Python ``object()``
        # sentinel used to be written into this runtime-indexed cache for an
        # unavailable producer.  When this planner itself is ingested, that is
        # a static Python reference flowing through dynamic table storage --
        # something complete SSA correctly refuses to pretend is data.
        _CONSTANT_KNOWN = 1
        _CONSTANT_UNAVAILABLE = 2
        _constant_status: dict[int, int] = {}
        _constant_values: dict[int, Any] = {}
        # A loop-carried initial is the entry arm of a runtime Phi, not a
        # literal every consumer may fold: inside the loop the name holds
        # the previous iteration's update.  Region extraction already
        # captures such a seed as an Input; a capture derived from it
        # (``i + 1`` over the seed ``0``) is likewise a runtime input, never
        # a region-local constant.  Seeding the evaluator state here keeps
        # every closure through a carried seed unavailable.
        for _node_id, _node_data in graph.G.nodes(data=True):
            for _initial, _updated in (
                (_node_data.get("attributes") or {}).get(
                    "loop_carried_bindings"
                ) or {}
            ).values():
                _constant_status[int(_initial)] = _CONSTANT_UNAVAILABLE

        def _constant_expression(value_id: int):
            """Evaluate a literal producer closure without Python recursion.

            A retained loop can expose feedback in the ProcessGraph, and a
            large whole-object compile can contain a perfectly acyclic closure
            deeper than Python's call-stack limit. Neither makes the capture a
            constant. Use an explicit post-order walk, marking every active
            frame nonconstant if a feedback edge is encountered.
            """

            root = int(value_id)
            if root in _constant_status:
                return (
                    _constant_status[root] == _CONSTANT_KNOWN,
                    _constant_values.get(root),
                )

            # 0/absent = unseen, 1 = active DFS frame, 2 = evaluated.
            states: dict[int, int] = {}
            stack: list[tuple[int, bool]] = [(root, False)]
            while stack:
                current, expanded = stack.pop()
                if current in _constant_status:
                    states[current] = 2
                    continue
                node = graph.G.nodes.get(current, {})
                opcode = str(node.get("op") or node.get("type") or "")
                attributes = dict(node.get("attributes") or {})
                parents = tuple(
                    int(parent)
                    for parent, _role in (node.get("parents") or ())
                )

                if not expanded:
                    if states.get(current) == 1:
                        # This node is already on the active producer path: a
                        # feedback closure cannot be a literal expression.
                        for active, state in tuple(states.items()):
                            if state == 1:
                                _constant_status[active] = (
                                    _CONSTANT_UNAVAILABLE
                                )
                                states[active] = 2
                        continue
                    states[current] = 1
                    if opcode in {"const", "Const", "Constant"}:
                        expression = node.get("expr_obj")
                        if "value" in attributes:
                            value = attributes["value"]
                            known = True
                        elif "constant" in node:
                            value = node["constant"]
                            known = True
                        elif isinstance(expression, ast.Constant):
                            value = expression.value
                            known = True
                        else:
                            value = None
                            known = False
                        if known:
                            _constant_values[current] = value
                            _constant_status[current] = _CONSTANT_KNOWN
                        else:
                            _constant_status[current] = (
                                _CONSTANT_UNAVAILABLE
                            )
                        states[current] = 2
                        continue
                    stack.append((current, True))
                    for parent in reversed(parents):
                        if parent in _constant_status:
                            continue
                        if states.get(parent) == 1:
                            for active, state in tuple(states.items()):
                                if state == 1:
                                    _constant_status[active] = (
                                        _CONSTANT_UNAVAILABLE
                                    )
                                    states[active] = 2
                            break
                        stack.append((parent, False))
                    continue

                if states.get(current) == 2:
                    continue
                if any(
                    _constant_status.get(parent) != _CONSTANT_KNOWN
                    for parent in parents
                ):
                    known = False
                elif opcode in {"neg", "USub"} and len(parents) == 1:
                    values = tuple(_constant_values[parent] for parent in parents)
                    try:
                        value = -values[0]
                    except (ArithmeticError, TypeError, ValueError):
                        known = False
                    else:
                        known = True
                elif opcode in {"pos", "UAdd"} and len(parents) == 1:
                    values = tuple(_constant_values[parent] for parent in parents)
                    try:
                        value = +values[0]
                    except (ArithmeticError, TypeError, ValueError):
                        known = False
                    else:
                        known = True
                elif opcode in {"Tuple", "tuple"}:
                    values = tuple(_constant_values[parent] for parent in parents)
                    value = tuple(values)
                    known = True
                elif opcode in {"List", "list"}:
                    values = tuple(_constant_values[parent] for parent in parents)
                    value = list(values)
                    known = True
                elif opcode in {"add", "Add"} and len(parents) == 2:
                    values = tuple(_constant_values[parent] for parent in parents)
                    try:
                        value = values[0] + values[1]
                    except (ArithmeticError, TypeError, ValueError):
                        known = False
                    else:
                        known = True
                elif opcode in {"sub", "Sub"} and len(parents) == 2:
                    values = tuple(_constant_values[parent] for parent in parents)
                    try:
                        value = values[0] - values[1]
                    except (ArithmeticError, TypeError, ValueError):
                        known = False
                    else:
                        known = True
                elif opcode in {"mul", "Mul"} and len(parents) == 2:
                    values = tuple(_constant_values[parent] for parent in parents)
                    try:
                        value = values[0] * values[1]
                    except (ArithmeticError, TypeError, ValueError):
                        known = False
                    else:
                        known = True
                else:
                    known = False
                if known:
                    _constant_values[current] = value
                    _constant_status[current] = _CONSTANT_KNOWN
                else:
                    _constant_status[current] = _CONSTANT_UNAVAILABLE
                states[current] = 2
            return (
                _constant_status.get(root) == _CONSTANT_KNOWN,
                _constant_values.get(root),
            )

        # A loop target's value is per-iteration even though its INITIALIZER
        # is a literal; folding it froze ``row`` at 0 inside every region a
        # nested body formed, so value = row*4+col compiled as value = col.
        loop_target_value_ids = {
            int(bound_target)
            for _loop_id, loop_data in graph.G.nodes(data=True)
            for bound_target in (
                (loop_data.get("attributes") or {}).get(
                    "loop_target_bindings", {}
                ) or {}
            ).values()
        }
        capture_constants = {}
        for value_id in region_captures:
            if value_id in carried_initial_boundaries:
                continue
            if int(value_id) in loop_target_value_ids:
                continue
            known, value = _constant_expression(value_id)
            if known:
                capture_constants[value_id] = value
        region_captures = tuple(
            value_id
            for value_id in region_captures
            if value_id not in capture_constants
        )
        compute_lines = []
        for value in region_nodes:
            node_data = graph.G.nodes[value]
            expression = node_data.get("expr_obj")
            parents = numerical_region_parents(value)
            opcode = str(node_data.get("op") or node_data.get("type"))
            line_attributes = dict(node_data.get("attributes") or {})
            if (
                isinstance(expression, ast.Attribute)
                or (
                    str(node_data.get("op") or node_data.get("type") or "")
                    .casefold() == "getattr"
                    and line_attributes.get("tensor") is not None
                )
            ):
                tensor_property = line_attributes.get("tensor")
                if tensor_property in abstract_tensor_funcs:
                    # NumPy/Torch expose ``real``/``imag`` as properties;
                    # after receiver grounding they are the same abstract
                    # tensor operations as ``AbstractTensor.real(x)`` and
                    # ``AbstractTensor.imag(x)``.
                    opcode = str(tensor_property)
                else:
                    # The graph may retain the frontend spelling ``Attribute``;
                    # repository SSA uses the semantic memory operation and must
                    # retain the selected field name in its record ABI.
                    opcode = "GetAttr"
                    line_attributes.setdefault("attribute", expression.attr)
            elif opcode in {"Call", "call"} and str(
                line_attributes.get("tensor") or ""
            ) in abstract_tensor_funcs:
                # A builtin arriving as a Call names its catalogued operator
                # in ``tensor`` (``max``/``min``); the operator IS the
                # semantics, exactly as ``float`` respells to a cast.
                opcode = str(line_attributes["tensor"])
            compute_lines.append(PlanLine.create(
                opcode,
                inputs=tuple(parent for parent, _role in parents),
                outputs=(int(value),),
                attributes={
                    **line_attributes,
                    "region": region_index,
                },
                input_roles=tuple(role for _parent, role in parents),
            ))
        compute_lines = tuple(compute_lines)
        # A constant operand (``self.n + 1``) is a leaf the region consumes but
        # neither produces nor captures -- it is not a runtime input, so it never
        # entered ``deployment_inputs``. The fused capture folds it into a scalar
        # attribute; the plan lines reference it by value id, so without a
        # defining line it is a dangling SSA value the backend emits as an
        # undeclared variable. Materialise each such constant as its own ``const``
        # line, ahead of the consumers, so the region is self-contained.
        produced = set(region_nodes)
        captured = set(region_captures)
        const_lines = []
        materialised: set[int] = set()
        for value_id, value in capture_constants.items():
            materialised.add(int(value_id))
            const_lines.append(PlanLine.create(
                "Const",
                inputs=(),
                outputs=(int(value_id),),
                attributes={"value": value, "region": region_index},
            ))
        for value in region_nodes:
            for parent, _role in (graph.G.nodes[value].get("parents") or ()):
                parent = int(parent)
                if (
                    parent in produced
                    or parent in captured
                    or parent in materialised
                ):
                    continue
                parent_data = graph.G.nodes.get(parent, {})
                if (
                    parent_data.get("op") or parent_data.get("type")
                ) not in ("const", "Constant"):
                    continue
                materialised.add(parent)
                const_lines.append(PlanLine.create(
                    "Const",
                    inputs=(),
                    outputs=(parent,),
                    attributes={
                        **dict(parent_data.get("attributes") or {}),
                        "region": region_index,
                    },
                ))
        # Carry each region value's shape/dtype from its process-graph node's
        # domain so the lowered SSA is shaped: a value that is an array lowers
        # as an array, not a scalar. A domain padded with size-1 dims (a scalar
        # is (1, 1, 1)) squeezes to its logical shape -- () for a scalar.
        _shape_dtype_cache: dict[int, tuple[tuple[int, ...], str]] = {}

        def _value_shape_dtype(value_id):
            """Infer one value record with a bounded explicit producer stack.

            Feedback is a retained-loop boundary.  When the work stack sees a
            parent already active, it seeds that loop-carried record from the
            parent's declared tensor/domain metadata and does not expand the
            edge again.  This is finite even for irreducible recursion and does
            not turn a loop into an unbounded source expansion.
            """

            root = int(value_id)

            def declared(current: int):
                node = graph.G.nodes.get(int(current), {})
                tensor = node.get("tensor") or {}
                domain = node.get("domain_node")
                attributes = dict(node.get("attributes") or {})
                binding_name = attributes.get("binding_name")
                specialization = (
                    (graph.G.graph.get("planner_specializations") or {}).get(
                        str(binding_name)
                    )
                    if binding_name is not None else None
                )
                specialized_shape = tuple(
                    int(extent)
                    for extent in (
                        getattr(specialization, "shape", ()) or ()
                    )
                )
                # Extents the source-program boundary states for this
                # parameter. Rank alone cannot serve here: the scalar-versus-
                # tensor decision reads EXTENTS (an operation is spelled with
                # its scalar opcode when result and operands all have empty
                # shape), so a parameter that declares `storage: span,
                # rank: 1` and nothing else still arrives shapeless and its
                # arithmetic is compiled as scalar arithmetic. A declared
                # shape is authoritative the same way a planner
                # specialization is -- it is a statement by the boundary, not
                # an inference from a default domain.
                abi_shape: tuple[int, ...] = ()
                if binding_name is not None:
                    abi_entry = (
                        graph.G.graph.get("parameter_value_abi") or {}
                    ).get(str(binding_name))
                    if isinstance(abi_entry, Mapping):
                        abi_shape = tuple(
                            int(extent)
                            for extent in (abi_entry.get("shape") or ())
                        )
                tensor_shape_declared = "shape" in tensor
                if tensor_shape_declared:
                    # An empty static extent tuple can be the exact companion
                    # to a nonzero dynamic rank.  It means "runtime extents",
                    # not "consult the padded DomainNode default".
                    shape = tuple(tensor.get("shape") or ())
                elif specialized_shape:
                    shape = specialized_shape
                elif abi_shape:
                    shape = abi_shape
                else:
                    shape = tuple(getattr(domain, "shape", ()) or ())
                logical = (
                    tuple(map(int, shape))
                    if tensor_shape_declared
                    or specialized_shape
                    or abi_shape
                    else tuple(int(dim) for dim in shape if int(dim) != 1)
                )
                dtype = str(
                    tensor.get("dtype")
                    or getattr(specialization, "dtype", None)
                    or node.get("dtype")
                    or getattr(domain, "dtype", None)
                    or "float64"
                )
                field_name = attributes.get("attribute")
                if (
                    not logical
                    and field_name is not None
                    and str(node.get("type")) in {"GetAttr", "Attribute"}
                ):
                    # A record field read through a mutable, non-array-typed
                    # receiver (``state.next_height``) has no tensor/domain
                    # shape of its own -- ``specialized_shape`` above only
                    # covers a receiver bound *directly* to an array. By this
                    # pass, ``expr_obj`` is gone (nodes created during SSA
                    # normalization carry none), so the receiver is found via
                    # the graph edge instead of the AST. The receiver's own
                    # bound specialization (a real object, set by
                    # ``propagate_bound_planner_specializations`` before this
                    # pass runs) still has the real field on it; read it
                    # structurally, the same safe, side-effect-free
                    # ``getattr`` this module already uses for call-argument
                    # specialization. Without this, every whole-array read of
                    # a record field silently collapses to ``()`` and a
                    # downstream array op is emitted as a scalar op on one
                    # element (see tools/HANDOFF_fluid_c_shell.md, Defect 1).
                    receiver_id = next((
                        int(parent)
                        for parent, role in (node.get("parents") or ())
                        if str(role) in {"value", "base", "object"}
                    ), None)
                    receiver_binding = (
                        (
                            graph.G.nodes.get(receiver_id, {}).get(
                                "attributes"
                            ) or {}
                        ).get("binding_name")
                        if receiver_id is not None else None
                    )
                    receiver_value = (
                        (graph.G.graph.get("planner_specializations") or {})
                        .get(str(receiver_binding))
                        if receiver_binding is not None else None
                    )
                    if receiver_value is not None:
                        try:
                            field_value = getattr(
                                receiver_value, str(field_name)
                            )
                        except AttributeError:
                            field_value = None
                        field_shape = getattr(field_value, "shape", None)
                        if field_shape:
                            logical = tuple(int(dim) for dim in field_shape)
                            field_dtype = getattr(field_value, "dtype", None)
                            if field_dtype is not None:
                                dtype = str(field_dtype)
                operation = str(
                    attributes.get("tensor")
                    or attributes.get("tensor_operation")
                    or attributes.get("tensor_candidate")
                    or node.get("op")
                    or node.get("type")
                    or ""
                ).casefold()
                if operation in {"const", "constant"}:
                    literal = attributes.get("value", node.get("constant"))
                    if isinstance(literal, bool):
                        dtype = "bool"
                    elif isinstance(literal, int):
                        dtype = "int"
                    elif isinstance(literal, float):
                        dtype = "float64"
                # Callable/operator-reference edges are provenance, not
                # numeric operands.  Letting their default domain dtype enter
                # scalar promotion made an all-integer ``min`` or arithmetic
                # chain appear float-valued merely because ``builtins.min``
                # itself was represented by a graph node.
                parents = tuple(
                    int(parent)
                    for parent, role in (node.get("parents") or ())
                    if int(parent) in graph.G
                    and str(role).casefold() not in {
                        "callee", "func", "function", "definition",
                        "operator", "operator_reference",
                    }
                )
                return node, tensor, attributes, logical, dtype, operation, parents

            states: dict[int, int] = {}
            stack: list[tuple[int, bool]] = [(root, False)]
            while stack:
                current, expanded = stack.pop()
                if current in _shape_dtype_cache:
                    states[current] = 2
                    continue
                (
                    _node,
                    tensor,
                    attributes,
                    logical,
                    dtype,
                    operation,
                    parents,
                ) = declared(current)
                if not expanded:
                    states[current] = 1
                    stack.append((current, True))
                    for parent in reversed(parents):
                        if parent in _shape_dtype_cache:
                            continue
                        if states.get(parent) == 1:
                            # The edge closes a retained loop. Its declared
                            # record is the finite boundary condition.
                            (_n, _t, _a, parent_shape, parent_dtype, _o, _p) = (
                                declared(parent)
                            )
                            _shape_dtype_cache[parent] = (
                                parent_shape, parent_dtype
                            )
                            states[parent] = 2
                            continue
                        stack.append((parent, False))
                    continue
                if states.get(current) == 2:
                    continue
                parent_records = [
                    (parent, *_shape_dtype_cache[parent])
                    for parent in parents
                    if parent in _shape_dtype_cache
                ]
                parent_shapes = [record[1] for record in parent_records]
                parent_dtypes = [record[2] for record in parent_records]
                if operation == "cast_like" and len(parents) >= 2:
                    # ``value.cast_like(reference)`` has the REFERENCE's
                    # element dtype (AbstractTensor.cast_like: the destination
                    # dtype is the reference operand's), not the domain's
                    # default.  Left at the default every float32 solve typed
                    # its one-hot selectors float64 while the formals they
                    # feed were float32.
                    reference_record = _shape_dtype_cache.get(parents[1])
                    if reference_record is not None and reference_record[1]:
                        dtype = reference_record[1]
                if (
                    operation in {
                        "unsqueeze", "squeeze", "expand", "broadcast_to",
                        "sum", "prod", "min", "max", "cumsum", "cumprod",
                    }
                    and not tensor.get("dtype")
                ):
                    # These keep their VALUE operand's element dtype (the
                    # descriptor rules above read it from the operand).  With
                    # no dtype posted on the node, the domain default
                    # (float64) is not a declaration: take the posted
                    # descriptor's dtype, else the operand's own record.
                    posted = _tensor_descriptor(graph, int(current))
                    value_parent = next((
                        int(parent)
                        for parent, role in (_node.get("parents") or ())
                        if str(role).casefold() in {
                            "operand", "value", "base", "input", "self",
                            "receiver",
                        }
                        and int(parent) in _shape_dtype_cache
                    ), next((p for p in parents[:1]), None))
                    if posted is not None and posted.get("dtype"):
                        dtype = str(posted["dtype"])
                    elif value_parent in _shape_dtype_cache:
                        dtype = _shape_dtype_cache[value_parent][1]
                # DomainNode historically pads scalars with unit axes. If it
                # says scalar but a canonical tensor op consumes a shaped
                # tensor, carry that shape through.
                if operation in {
                    "neg", "abs", "sqrt", "exp", "log", "round", "trunc",
                    "floor", "ceil", "isfinite", "isnan", "isinf",
                    "logical_not", "tanh", "sin", "cos", "tan", "asin",
                    "acos", "atan", "sinh", "cosh", "asinh", "acosh",
                    "atanh", "sign", "add", "sub", "mul", "truediv",
                    "floordiv", "mod", "pow", "less", "less_equal",
                    "greater", "greater_equal", "equal", "not_equal",
                    "maximum", "minimum", "where", "float", "double",
                    "long", "int", "to", "to_dtype", "astype", "cbrt",
                    "zeros_like", "empty_like", "ones_like", "full_like",
                }:
                    shaped = [candidate for candidate in parent_shapes if candidate]
                    if shaped:
                        try:
                            logical = tuple(np.broadcast_shapes(*shaped))
                        except ValueError:
                            # Preserve the declared descriptor for an invalid
                            # expression.  Kernel lowering will report the
                            # incompatible operands precisely; choosing one
                            # tuple here would fabricate a region ABI.
                            pass
                    if parent_dtypes and not tensor.get("dtype"):
                        # A literal Python float is a weak operand: it names
                        # no element width, so beside a typed float operand
                        # it does not widen the result (``float32 * 2.0`` is
                        # float32, as in the eager lane).  The float rank of
                        # the result is the widest the operands DECLARE; an
                        # explicit ``float``/``double`` cast is float64 by
                        # its own source.
                        weak_literals = {
                            parent for parent, _s, _d in parent_records
                            if declared(parent)[5] in {"const", "constant"}
                            and isinstance(
                                declared(parent)[2].get(
                                    "value", declared(parent)[0].get("constant")
                                ),
                                float,
                            )
                        }
                        numeric_dtypes = {
                            "int" if candidate in {"i32", "int32"}
                            else "int64" if candidate == "i64"
                            else "float64" if candidate in {
                                "float", "double", "f64",
                            }
                            else str(candidate)
                            for parent, _s, candidate in parent_records
                            if parent not in weak_literals
                        }
                        # The declared promotion rule is the precision
                        # layer's: the wider element type wins.
                        declared_float = _widest_element(*numeric_dtypes)
                        if declared_float is None and weak_literals:
                            # Only literal floats are float: the literal's
                            # own default width is all the source says.
                            declared_float = "float64"
                        if declared_float is not None:
                            dtype = (
                                "float64"
                                if operation in {"float", "double"}
                                else declared_float
                            )
                        elif "int64" in numeric_dtypes:
                            dtype = "int64"
                        elif "int" in numeric_dtypes:
                            dtype = "int"
                        elif "bool" in numeric_dtypes:
                            dtype = "bool"
                if operation in {
                    "sum", "prod", "min", "max", "any", "all",
                    "unsqueeze", "squeeze",
                    # ``cast_like`` keeps its VALUE's extents (its reference
                    # operand only names the dtype), so broadcasting the
                    # parents' shapes would answer the reference's; a scan
                    # keeps its operand's.  Both are settled by the
                    # descriptor.
                    "cast_like", "cumsum", "cumprod",
                }:
                    # The result extents of a reduction or an axis insertion
                    # are settled by the graph's tensor descriptor
                    # (``_tensor_descriptor``, which also posts them on the
                    # book) -- the same record ``value_ranks`` below reads.
                    # Re-deriving them here read only an ``axis``/``dim``
                    # ATTRIBUTE: the orbital quadrature's positional
                    # ``t26.sum(t3)`` became rank 0 while the descriptor said
                    # (4,), and ``t1.unsqueeze(t2)`` kept its source's (5,)
                    # while the descriptor said (5, 1).
                    settled = _tensor_descriptor(graph, int(current))
                    if settled is not None and descriptor_states_a_shape(settled):
                        logical = tuple(int(e) for e in (settled.get("shape") or ()))
                        _shape_dtype_cache[current] = (logical, dtype)
                        states[current] = 2
                        continue
                if operation in {"sum", "prod", "min", "max", "any", "all"}:
                    axis = attributes.get("axis", attributes.get("dim"))
                    source_shape = next(
                        (candidate for candidate in parent_shapes if candidate),
                        (),
                    )
                    if axis is None:
                        logical = ()
                    elif source_shape:
                        normalized = int(axis) % len(source_shape)
                        logical = (
                            source_shape[:normalized]
                            + source_shape[normalized + 1:]
                        )
                if operation == "matmul" and len(parent_shapes) >= 2:
                    left_shape, right_shape = parent_shapes[:2]
                    if (
                        len(left_shape) >= 2
                        and len(right_shape) >= 2
                        and left_shape[-1] == right_shape[-2]
                    ):
                        try:
                            batch_shape = tuple(np.broadcast_shapes(
                                left_shape[:-2], right_shape[:-2]
                            ))
                        except ValueError:
                            pass
                        else:
                            logical = (
                                *batch_shape, left_shape[-2], right_shape[-1]
                            )
                _shape_dtype_cache[current] = (logical, dtype)
                states[current] = 2
            logical, dtype = _shape_dtype_cache.get(
                root, declared(root)[3:5]
            )
            return (root, logical, dtype)

        value_shapes = tuple(
            _value_shape_dtype(value_id)
            for value_id in (*region_nodes, *region_captures)
        )
        value_ranks = tuple(
            (
                int(value_id),
                int((descriptor or {}).get(
                    "rank", len(tuple((descriptor or {}).get("shape") or ()))
                )),
            )
            for value_id in (*region_nodes, *region_captures)
            for descriptor in (_tensor_descriptor(graph, int(value_id)),)
            if descriptor is not None
        )
        # A capture produced by a shape-only operation outside the region is
        # fed its source's storage (the operation has no runtime existence);
        # the formal is a view of it (``PlanClosure.value_views``).
        value_views = []
        for value_id in region_captures:
            capture = graph.G.nodes.get(int(value_id), {})
            capture_operation = str(
                (capture.get("attributes") or {}).get("tensor_operation")
                or capture.get("op") or capture.get("type") or ""
            ).casefold()
            if capture_operation not in _SHAPE_ONLY_VIEW_OPERATIONS:
                continue
            storage = next((
                int(parent) for parent, role in capture.get("parents") or ()
                if str(role).casefold() in {
                    "operand", "value", "base", "input", "self", "receiver",
                } and int(parent) in graph.G
            ), None)
            descriptor = _tensor_descriptor(graph, int(value_id))
            if storage is None or descriptor is None or not descriptor_states_a_shape(descriptor):
                continue
            value_views.append((
                int(value_id), storage,
                tuple(int(e) for e in descriptor.get("shape") or ()),
                capture_operation,
            ))
        _concord_consumer_operands(graph, region_nodes, region_captures)
        _concord_item_operands(graph, region_captures)
        items.append(PlanClosure(
            name=f"region_{region_index}",
            captures=region_captures,
            items=(*const_lines, *compute_lines),
            value_shapes=value_shapes,
            value_ranks=value_ranks,
            value_views=tuple(value_views),
        ))
    control_values = set(_control_dependency_value_ids(
        getattr(shell, "shell_control_program", None)
    ))

    planned = PlanClosure(
        name=str(
            graph.G.graph.get("function_name")
            or type(shell).__name__
        ),
        captures=tuple(dict.fromkeys((
            *sorted(control_values),
            *(
            int(node_id)
            for node_id, data in graph.G.nodes(data=True)
            if data.get("type") == "Input"
            and graph.G.out_degree(node_id)
            ),
        ))),
        items=tuple(items),
    )
    memo[id(shell)] = planned
    return planned


def _planned_operator_node_ids(hierarchy: PlanClosure) -> tuple[int, ...]:
    """Return this closure's operators in planner-owned segment order."""

    return tuple(
        int(output_id)
        for item in hierarchy.items
        if isinstance(item, PlanClosure)
        and item.name.startswith("region_")
        for line in item.items
        if isinstance(line, PlanLine)
        for output_id in line.outputs
    )


@dataclass(frozen=True)
class PlannedOperatorImplementation:
    node_id: int
    kind: str
    fused_step_ids: tuple[int, ...] = ()


def _build_planned_operator_implementations(
    hierarchy: PlanClosure,
    captured_regions: Mapping[int, CapturedFusedProgram],
    observed_plain_node_ids: Iterable[int],
) -> dict[int, tuple[PlannedOperatorImplementation, ...]]:
    """Attach lowering evidence to planner-owned operators by node ID."""

    observed_plain = set(map(int, observed_plain_node_ids))
    implementations = {}
    for item in hierarchy.items:
        if not (
            isinstance(item, PlanClosure)
            and item.name.startswith("region_")
        ):
            continue
        region_index = int(item.name.split("_", 1)[1])
        captured = captured_regions.get(region_index)
        fused_steps: dict[int, list[int]] = {}
        if captured is not None:
            for program in captured.execution_programs:
                for step in program.steps:
                    fused_steps.setdefault(
                        int(step.result_id), []
                    ).append(int(step.step_id))
        region_implementations = []
        for line in item.items:
            if not isinstance(line, PlanLine):
                continue
            for node_id in line.outputs:
                node_id = int(node_id)
                step_ids = tuple(fused_steps.get(node_id, ()))
                region_implementations.append(
                    PlannedOperatorImplementation(
                        node_id,
                        "plain"
                        if node_id in observed_plain
                        else "fused"
                        if step_ids
                        else "structural",
                        step_ids,
                    )
                )
        implementations[region_index] = tuple(
            region_implementations
        )
    return implementations


def _refresh_hierarchy_control_captures(
    closure: PlanClosure,
    shell: Any,
) -> PlanClosure:
    """Update final control captures without rediscovering source structure.

    The pre-capture hierarchy already freezes every callsite, argument/result
    binding and region closure.  Numerical lowering only adds control ABI
    values.  Re-running graph topological discovery for the entire call tree
    at that point is redundant and becomes quadratic on deeply composed
    programs, so preserve the structural plan and refresh capture identities
    directly from each shell's final ControlProgram.
    """

    from .control_source import (
        CallBlock,
        LoopControlBlock,
        LoopBlock,
        ParallelDeployment,
        SequenceBlock,
        StateMachineTick,
        WhileBlock,
    )

    values = set(_control_dependency_value_ids(shell.shell_control_program))

    refreshed_items = []
    for item in closure.items:
        if not isinstance(item, PlanCall):
            refreshed_items.append(item)
            continue
        child = shell.callsite_function_shells.get(item.callsite_id)
        if child is None:
            refreshed_items.append(item)
            continue
        refreshed_items.append(PlanCall(
            item.callsite_id,
            _refresh_hierarchy_control_captures(item.callee, child),
            item.argument_value_ids,
            item.result_value_ids,
            item.argument_bindings,
            item.result_bindings,
            item.enclosing_loop_ids,
        ))
    return PlanClosure(
        closure.name,
        tuple(dict.fromkeys((*closure.captures, *sorted(values)))),
        tuple(refreshed_items),
        closure.closure_id,
        closure.value_shapes,
        closure.value_ranks,
        closure.value_views,
    )



def _build_hierarchical_glsl_artifact(shell: Any):
    """Lower a captured call tree into one globally-namespaced shell."""

    compose_started = time.perf_counter()
    shell._profiler.trace(
        path=shell.profile_path,
        section="hierarchical-artifact",
        label="begin",
        fields={},
    )
    shell.hierarchical_compose_failure = None
    hierarchy = shell.hierarchy_plan
    value_table = shell.hierarchy_value_table

    identity_closures: dict[int, tuple[int, str]] = {}
    identity_call_sources: dict[
        tuple[int, int], tuple[int, int]
    ] = {}

    def find_identities(closure: PlanClosure, owner: Any) -> None:
        graph = owner.process_graph.G
        identity = _identity_parameter_projection(graph)
        if identity is not None:
            identity_closures[int(closure.closure_id)] = identity
        for item in closure.items:
            if not isinstance(item, PlanCall):
                continue
            child = owner.callsite_function_shells.get(item.callsite_id)
            if child is not None:
                find_identities(item.callee, child)
                identity = identity_closures.get(
                    int(item.callee.closure_id)
                )
                if identity is not None:
                    source_parameter, _kind = identity
                    caller_source = next((
                        int(caller)
                        for caller, callee in item.argument_bindings
                        if int(callee) == int(source_parameter)
                    ), None)
                    if caller_source is not None:
                        identity_call_sources[(
                            int(closure.closure_id),
                            int(item.callsite_id),
                        )] = (
                            int(closure.closure_id),
                            caller_source,
                        )

    find_identities(hierarchy, shell)
    identity_reduction = reduce_hierarchy_identities(
        hierarchy, set(identity_closures)
    )
    hierarchy = identity_reduction.root
    shell._profiler.trace(
        path=shell.profile_path,
        section="hierarchical-artifact",
        label="identity reduction complete",
        fields={
            "elapsed_ms": round(
                (time.perf_counter() - compose_started) * 1e3,
                3,
            ),
        },
    )
    shell.hierarchy_identity_collapses = (
        identity_reduction.collapsed_callsites
    )
    shell.hierarchy_identity_rounds = identity_reduction.rounds
    shell.hierarchy_remaining_callsite_ids = tuple(
        int(item.callsite_id)
        for item in hierarchy.items
        if isinstance(item, PlanCall)
    )
    controls: dict[int, ControlProgram] = {}
    shells: dict[int, Any] = {}
    closures: dict[int, PlanClosure] = {}
    calls: dict[tuple[int, int], PlanCall] = {}
    argument_sources: dict[tuple[int, int], tuple[int, int]] = {}

    def gather(closure: PlanClosure, owner: Any) -> bool:
        closure_id = int(closure.closure_id)
        closures[closure_id] = closure
        shells[closure_id] = owner
        control = owner.shell_control_program
        if control is None:
            if owner.dispatch_subgraphs:
                shell.hierarchical_compose_failure = {
                    "reason": "missing-control-program",
                    "closure_id": closure_id,
                    "function": owner.process_graph.G.graph.get(
                        "function_name"
                    ),
                    "dispatch_regions": len(owner.dispatch_subgraphs),
                }
                return False
            control = ControlProgram(SequenceBlock(()), ())
        controls[closure_id] = control
        import os as _os, sys as _sys
        if _os.environ.get("TURING_DEBUG_REGION_ORDER"):
            _fn = owner.process_graph.G.graph.get("function_name")
            print(
                f"DEBUGGATHER fn={_fn} closure_id={closure_id} "
                f"region_indices={getattr(control, 'region_indices', None)}",
                file=_sys.stderr,
            )
        for item in closure.items:
            if not isinstance(item, PlanCall):
                continue
            calls[(closure_id, int(item.callsite_id))] = item
            for caller_local, callee_local in item.argument_bindings:
                argument_sources[(
                    int(item.callee.closure_id),
                    int(callee_local),
                )] = (closure_id, int(caller_local))
            child = owner.callsite_function_shells.get(item.callsite_id)
            if child is None or not gather(item.callee, child):
                return False
        return True

    if not gather(hierarchy, shell):
        return None
    # The loop planner has already performed the control-flow/SSA analysis.
    # Its aliases are therefore authoritative value edges, not hints for a
    # later runtime reconstruction.  Carry them into hierarchy identity
    # resolution before numerical programs are globally namespaced.  In
    # particular, a LoopExit names the planner-selected carried value; mapping
    # it back to the syntactic list initializer would sever the real producer
    # edge and manufacture an external shader input.
    collection_owner_endpoints = {
        (int(closure_id), int(collection))
        for closure_id, control in controls.items()
        for _source, collection, _induction, _start
        in control.collection_bindings
    }
    loop_carried_update_endpoints: set[tuple[int, int]] = set()

    def collect_loop_carried_updates(
        closure_id: int,
        block: Any,
    ) -> None:
        carried = (
            block.carried_aliases
            if isinstance(block, (LoopBlock, WhileBlock))
            else ()
        )
        loop_carried_update_endpoints.update(
            (int(closure_id), int(updated))
            for updated, _initial in carried
        )
        if isinstance(block, SequenceBlock):
            for child in block.blocks:
                collect_loop_carried_updates(closure_id, child)
        elif isinstance(block, LoopBlock):
            collect_loop_carried_updates(closure_id, block.body)
        elif isinstance(block, WhileBlock):
            collect_loop_carried_updates(closure_id, block.condition)
            collect_loop_carried_updates(closure_id, block.body)

    for closure_id, control in controls.items():
        collect_loop_carried_updates(int(closure_id), control.root)
    control_alias_sources = {
        (int(closure_id), int(updated)): (
            int(closure_id),
            int(initial),
        )
        for closure_id, control in controls.items()
        for updated, initial in control.value_aliases
        if (int(closure_id), int(updated))
        not in collection_owner_endpoints
        and (int(closure_id), int(updated))
        not in loop_carried_update_endpoints
    }
    for closure_id, owner in shells.items():
        selected_returns = tuple(getattr(
            owner, "_captured_return_value_ids", ()
        ))
        output_names = tuple(
            owner.process_graph.G.graph.get("function_outputs", ())
        )
        if len(selected_returns) != 1 or len(output_names) != 1:
            continue
        selected = int(selected_returns[0])
        identities = owner.process_graph.G.graph.get(
            "identity_table", {}
        ) or {}
        for output_id in identities.get(output_names[0], ()):
            output_id = int(output_id)
            if output_id == selected:
                continue
            # A statically selected early return is the sole producer of a
            # single-output invocation.  Normalization may still correlate
            # the function output with a later, unexecuted return expression;
            # collapse that stale endpoint before global value projection so
            # it cannot veto the caller-argument redirect.
            control_alias_sources[(int(closure_id), output_id)] = (
                int(closure_id), selected
            )
    for call in calls.values():
        child_id = int(call.callee.closure_id)
        child_owner = shells[child_id]
        selected_returns = tuple(getattr(
            child_owner, "_captured_return_value_ids", ()
        ))
        if len(selected_returns) != 1 or len(call.result_bindings) != 1:
            continue
        callee_result, _caller_result = call.result_bindings[0]
        selected = int(selected_returns[0])
        if int(callee_result) != selected:
            control_alias_sources[(child_id, int(callee_result))] = (
                child_id, selected
            )
    for closure_id, owner in shells.items():
        parameter_inputs: dict[str, list[int]] = {}
        for local_id, data in owner.process_graph.G.nodes(data=True):
            attributes = data.get("attributes") or {}
            if (
                data.get("type") != "Input"
                or attributes.get("binding_kind") != "parameter"
            ):
                continue
            name = attributes.get("binding_name") or data.get("label")
            if name is not None:
                parameter_inputs.setdefault(str(name), []).append(
                    int(local_id)
                )
        for local_ids in parameter_inputs.values():
            bound = [
                local_id
                for local_id in local_ids
                if (int(closure_id), int(local_id)) in argument_sources
            ]
            if len(bound) != 1:
                continue
            canonical = int(bound[0])
            for local_id in local_ids:
                if int(local_id) == canonical:
                    continue
                # AST normalization may retain multiple Input occurrences for
                # one source parameter while PlanCall binds only its canonical
                # occurrence.  They are the same lexical SSA source; route
                # every duplicate through the explicit argument edge.
                control_alias_sources[(
                    int(closure_id), int(local_id)
                )] = (int(closure_id), canonical)
    shell._profiler.trace(
        path=shell.profile_path,
        section="hierarchical-artifact",
        label="control gather complete",
        fields={
            "closures": len(closures),
            "elapsed_ms": round(
                (time.perf_counter() - compose_started) * 1e3,
                3,
            ),
        },
    )
    global_redirects = {}
    for identity_endpoint, source_endpoint in identity_call_sources.items():
        try:
            identity_global = value_table.global_id(*identity_endpoint)
            source_global = value_table.global_id(*source_endpoint)
        except KeyError:
            continue
        if identity_global != source_global:
            global_redirects[int(identity_global)] = int(source_global)

    def redirected(global_id: int) -> int:
        seen = set()
        current = int(global_id)
        while current in global_redirects and current not in seen:
            seen.add(current)
            current = global_redirects[current]
        return current

    if global_redirects:
        value_table = type(value_table)(tuple(
            (closure_id, local_id, redirected(global_id))
            for closure_id, local_id, global_id
            in value_table.correlations
        ))
    shell.hierarchical_effective_value_table = value_table
    hierarchical = compose_hierarchical_control(
        hierarchy, controls, value_table
    )
    shell._profiler.trace(
        path=shell.profile_path,
        section="hierarchical-artifact",
        label="control composition complete",
        fields={
            "regions": len(hierarchical.program.region_indices),
            "elapsed_ms": round(
                (time.perf_counter() - compose_started) * 1e3,
                3,
            ),
        },
    )
    region_lookup = {
        (closure_id, local_region): global_region
        for closure_id, local_region, global_region
        in hierarchical.region_correlations
    }
    next_value = 1 + max(
        (
            global_id
            for _closure, _local, global_id
            in value_table.correlations
        ),
        default=-1,
    )
    private_values: dict[tuple[int, int], int] = {}
    capture_values: dict[int, int] = {}
    endpoint_meta: dict[tuple[int, int], Any] = {}
    for closure_id, owner in shells.items():
        for value_id, meta in owner.compiled_feed_meta.items():
            endpoint_meta.setdefault(
                (int(closure_id), int(value_id)),
                meta,
            )
        for captured in owner.captured_region_programs.values():
            for program in (captured.program, *captured.stages):
                for value_id, meta in (program.meta or {}).items():
                    endpoint_meta.setdefault(
                        (int(closure_id), int(value_id)),
                        meta,
                    )

    def global_value(closure_id: int, local_id: int) -> int:
        nonlocal next_value
        try:
            return value_table.global_id(closure_id, local_id)
        except KeyError:
            endpoint = (int(closure_id), int(local_id))
            owner = shells.get(int(closure_id))
            is_synthetic = endpoint in synthetic_field_paths
            if (
                owner is not None
                and int(local_id) not in owner.process_graph.G
                and not is_synthetic
            ):
                # The discovery tape is one global program observation.  A
                # transient result identity therefore means the same value
                # when it crosses function compartments; closure-namespacing
                # it severs the producer/consumer edge and fabricates a shell
                # input.  Graph-local IDs and compiler-created field
                # endpoints remain scoped, but raw tape identities are
                # canonical across the hierarchy.
                capture_id = int(local_id)
                if capture_id not in capture_values:
                    capture_values[capture_id] = next_value
                    next_value += 1
                return capture_values[capture_id]
            key = endpoint
            if key not in private_values:
                private_values[key] = next_value
                next_value += 1
            return private_values[key]

    # Python classes crossing function boundaries are logical aggregate SSA
    # values, not runtime buffers and never host callables.  Resolve only the
    # field that an ordinary Attribute expression requests, following call
    # arguments and returns through the hierarchy.  This keeps source programs
    # normal Python while making aggregate erasure a general compiler rule.
    resolving: set[tuple[int, int]] = set()
    resolved_leaves: dict[
        tuple[int, int],
        dict[tuple[object, ...], tuple[int, int]],
    ] = {}
    synthetic_field_endpoints: dict[
        tuple[int, int, tuple[str, ...]], tuple[int, int]
    ] = {}
    synthetic_field_paths: dict[
        tuple[int, int], tuple[tuple[int, int], tuple[str, ...]]
    ] = {}
    synthetic_static_values: dict[tuple[int, int], Any] = {}
    next_synthetic_local = -1
    unresolved_static = object()
    metadata_links = {
        **identity_call_sources,
        **argument_sources,
    }
    changed = True
    while changed:
        changed = False
        for target, source in metadata_links.items():
            target_meta = endpoint_meta.get(target)
            source_meta = endpoint_meta.get(source)
            if target_meta is None and source_meta is not None:
                endpoint_meta[target] = source_meta
                changed = True
            elif source_meta is None and target_meta is not None:
                endpoint_meta[source] = target_meta
                changed = True
        by_global: dict[int, list[tuple[int, int]]] = {}
        for closure_id, local_id, global_id in value_table.correlations:
            by_global.setdefault(int(global_id), []).append(
                (int(closure_id), int(local_id))
            )
        for endpoints in by_global.values():
            known = next(
                (
                    endpoint_meta[endpoint]
                    for endpoint in endpoints
                    if endpoint in endpoint_meta
                ),
                None,
            )
            if known is None:
                continue
            for endpoint in endpoints:
                if endpoint not in endpoint_meta:
                    endpoint_meta[endpoint] = known
                    changed = True
    global_meta: dict[int, Any] = {}
    for closure_id, local_id, global_id in value_table.correlations:
        meta = endpoint_meta.get((int(closure_id), int(local_id)))
        if meta is not None:
            global_meta.setdefault(int(global_id), meta)

    def value_meta(
        endpoint: tuple[int, int],
        visiting: set[tuple[int, int]] | None = None,
    ) -> Any | None:
        endpoint = (int(endpoint[0]), int(endpoint[1]))
        visiting = set() if visiting is None else set(visiting)
        if endpoint in visiting:
            return None
        visiting.add(endpoint)
        direct = endpoint_meta.get(endpoint)
        if direct is not None:
            return direct
        source = metadata_links.get(endpoint)
        if source is not None:
            inherited = value_meta(source, visiting)
            if inherited is not None:
                return inherited
        try:
            correlated = value_table.global_id(*endpoint)
        except KeyError:
            correlated = None
        if correlated is not None:
            inherited = global_meta.get(int(correlated))
            if inherited is not None:
                return inherited
        owner = shells.get(endpoint[0])
        graph = None if owner is None else owner.process_graph.G
        if graph is None or endpoint[1] not in graph:
            return None
        data = graph.nodes[endpoint[1]]
        parents = tuple(
            (int(parent), str(role))
            for parent, role in (data.get("parents") or ())
            if str(role) not in {"callee", "func", "definition"}
        )
        if data.get("type") in {"Indexed", "indexed"}:
            base = next((
                parent for parent, role in parents if role == "base"
            ), None)
            index_node = next((
                parent for parent, role in parents if role == "index"
            ), None)
            index = None
            if index_node is not None and index_node in graph:
                try:
                    index = _constant_value(graph.nodes[index_node])
                except KeyError:
                    pass
            if base is not None and isinstance(index, int):
                call = calls.get((endpoint[0], base))
                if call is not None:
                    child_id = int(call.callee.closure_id)
                    child_graph = shells[child_id].process_graph.G
                    output_names = tuple(
                        child_graph.graph.get("function_outputs", ())
                    )
                    if -len(output_names) <= index < len(output_names):
                        output_name = output_names[index]
                        identities = (
                            child_graph.graph.get("identity_table") or {}
                        )
                        output_ids = identities.get(output_name, ())
                        if output_ids:
                            inherited = value_meta(
                                (child_id, int(output_ids[-1])),
                                visiting,
                            )
                            if inherited is not None:
                                return inherited
        call = calls.get(endpoint)
        if call is not None:
            child_id = int(call.callee.closure_id)
            child_graph = shells[child_id].process_graph.G
            output_names = tuple(
                child_graph.graph.get("function_outputs", ())
            )
            if len(output_names) == 1:
                identities = child_graph.graph.get("identity_table") or {}
                output_ids = identities.get(output_names[0], ())
                if output_ids:
                    return value_meta(
                        (child_id, int(output_ids[-1])),
                        visiting,
                    )
        return None

    def field_endpoint(
        source: tuple[int, int],
        field: str,
    ) -> tuple[int, int]:
        nonlocal next_synthetic_local
        rooted = synthetic_field_paths.get(source)
        if rooted is None:
            root, path = source, ()
        else:
            root, path = rooted
        full_path = (*path, str(field))
        key = (int(root[0]), int(root[1]), full_path)
        endpoint = synthetic_field_endpoints.get(key)
        if endpoint is None:
            endpoint = (int(root[0]), next_synthetic_local)
            next_synthetic_local -= 1
            synthetic_field_endpoints[key] = endpoint
            synthetic_field_paths[endpoint] = (root, full_path)
        return endpoint

    def static_endpoint_value(
        endpoint: tuple[int, int],
        visiting: set[tuple[int, int]] | None = None,
    ) -> Any:
        visiting = set() if visiting is None else set(visiting)
        endpoint = (int(endpoint[0]), int(endpoint[1]))
        if endpoint in visiting:
            return unresolved_static
        visiting.add(endpoint)
        rooted = synthetic_field_paths.get(endpoint)
        if rooted is not None:
            if endpoint in synthetic_static_values:
                return synthetic_static_values[endpoint]
            root, path = rooted
            root_owner = shells.get(int(root[0]))
            root_graph = (
                None if root_owner is None
                else root_owner.process_graph.G
            )
            root_data = (
                None
                if root_graph is None or int(root[1]) not in root_graph
                else root_graph.nodes[int(root[1])]
            )
            root_name = (
                None
                if root_data is None
                else (root_data.get("attributes") or {}).get(
                    "binding_name"
                )
            )
            static_fields = (
                {}
                if root_owner is None
                else getattr(
                    root_owner, "_capture_input_static_fields", {}
                )
            )
            static_key = (str(root_name), tuple(map(str, path)))
            if root_name is not None and static_key in static_fields:
                return static_fields[static_key]
            projected = leaves(*root)
            target = projected.get(tuple(path))
            if target is not None and target != endpoint:
                return static_endpoint_value(target, visiting)
            return unresolved_static
        source = identity_call_sources.get(endpoint)
        if source is None:
            source = argument_sources.get(endpoint)
        if source is not None:
            return static_endpoint_value(source, visiting)
        owner = shells.get(endpoint[0])
        if owner is None or endpoint[1] not in owner.process_graph.G:
            return unresolved_static
        graph = owner.process_graph.G
        data = graph.nodes[endpoint[1]]
        if data.get("type") in {
            "Constant", "Const", "const", "StaticReference"
        }:
            try:
                return _constant_value(data)
            except KeyError:
                return unresolved_static
        if data.get("type") == "Input":
            name = (data.get("attributes") or {}).get("binding_name")
            specializations = (
                graph.graph.get("planner_specializations") or {}
            )
            if name in specializations:
                return specializations[name]
            defaults = graph.graph.get("parameter_defaults") or {}
            if name in defaults:
                return defaults[name]
        parents = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
            if str(role) not in {"callee", "func", "definition"}
        }
        expression = data.get("expr_obj")
        if data.get("type") == "Phi":
            test = parents.get("test")
            body = parents.get("body")
            orelse = parents.get("orelse")
            predicate = (
                unresolved_static
                if test is None
                else static_endpoint_value((endpoint[0], test), visiting)
            )
            selected = (
                body
                if predicate is not unresolved_static and bool(predicate)
                else (
                    orelse
                    if predicate is not unresolved_static
                    else None
                )
            )
            if selected is not None:
                return static_endpoint_value(
                    (endpoint[0], selected), visiting
                )
            if body is not None and orelse is not None:
                body_value = static_endpoint_value(
                    (endpoint[0], body), visiting
                )
                else_value = static_endpoint_value(
                    (endpoint[0], orelse), visiting
                )
                if (
                    body_value is not unresolved_static
                    and else_value is not unresolved_static
                    and type(body_value) is type(else_value)
                    and body_value == else_value
                ):
                    return body_value
        if isinstance(expression, ast.Attribute) and expression.attr in {
            "shape", "ndim", "ndims", "dtype", "device",
        }:
            base = next((
                parent
                for role, parent in parents.items()
                if role in {"value", "base", "operand", "object"}
            ), None)
            if base is not None:
                meta = value_meta((endpoint[0], base))
                if meta is None:
                    projected = leaves(endpoint[0], base)
                    if len(projected) == 1 and () in projected:
                        meta = value_meta(projected[()])
                if meta is not None:
                    if expression.attr == "shape" and meta.shape is not None:
                        return tuple(int(size) for size in meta.shape)
                    if expression.attr in {"ndim", "ndims"} and (
                        meta.shape is not None
                    ):
                        return len(meta.shape)
                    if expression.attr == "dtype" and meta.dtype is not None:
                        return str(meta.dtype)
                    if expression.attr == "device":
                        device = getattr(meta, "device", None)
                        if device is not None:
                            return device
        if (
            isinstance(expression, ast.Subscript)
            or data.get("type") in {"Indexed", "indexed"}
        ):
            base = next((
                parent
                for role, parent in parents.items()
                if role in {"value", "base", "operand", "object"}
            ), None)
            index = next((
                parent
                for role, parent in parents.items()
                if role in {"slice", "index", "subscript"}
            ), None)
            if base is not None and index is not None:
                container = static_endpoint_value(
                    (endpoint[0], base), visiting
                )
                position = static_endpoint_value(
                    (endpoint[0], index), visiting
                )
                if (
                    container is not unresolved_static
                    and position is not unresolved_static
                    and isinstance(position, (int, slice))
                ):
                    try:
                        return container[position]
                    except (IndexError, KeyError, TypeError):
                        pass
        if isinstance(expression, ast.UnaryOp):
            operand = next((
                parent
                for role, parent in parents.items()
                if role in {"operand", "value"}
            ), None)
            fixed = (
                unresolved_static
                if operand is None
                else static_endpoint_value((endpoint[0], operand), visiting)
            )
            if fixed is not unresolved_static:
                if isinstance(expression.op, ast.USub):
                    return -fixed
                if isinstance(expression.op, ast.UAdd):
                    return +fixed
                if isinstance(expression.op, ast.Not):
                    return not fixed
        if isinstance(expression, ast.BinOp):
            left = parents.get("lhs", parents.get("left"))
            right = parents.get("rhs", parents.get("right"))
            lhs = (
                unresolved_static
                if left is None
                else static_endpoint_value((endpoint[0], left), visiting)
            )
            rhs = (
                unresolved_static
                if right is None
                else static_endpoint_value((endpoint[0], right), visiting)
            )
            if lhs is not unresolved_static and rhs is not unresolved_static:
                operation = expression.op
                try:
                    if isinstance(operation, ast.Add):
                        return lhs + rhs
                    if isinstance(operation, ast.Sub):
                        return lhs - rhs
                    if isinstance(operation, ast.Mult):
                        return lhs * rhs
                    if isinstance(operation, ast.Div):
                        return lhs / rhs
                    if isinstance(operation, ast.FloorDiv):
                        return lhs // rhs
                    if isinstance(operation, ast.Mod):
                        return lhs % rhs
                    if isinstance(operation, ast.Pow):
                        return lhs ** rhs
                except (ArithmeticError, TypeError, ValueError):
                    pass
        if isinstance(expression, ast.Compare) and len(expression.ops) == 1:
            left = parents.get("lhs", parents.get("left"))
            right = parents.get("rhs", parents.get("right"))
            lhs = (
                unresolved_static
                if left is None
                else static_endpoint_value((endpoint[0], left), visiting)
            )
            rhs = (
                unresolved_static
                if right is None
                else static_endpoint_value((endpoint[0], right), visiting)
            )
            if lhs is not unresolved_static and rhs is not unresolved_static:
                operation = expression.ops[0]
                if isinstance(operation, ast.Eq):
                    return lhs == rhs
                if isinstance(operation, ast.NotEq):
                    return lhs != rhs
                if isinstance(operation, ast.Lt):
                    return lhs < rhs
                if isinstance(operation, ast.LtE):
                    return lhs <= rhs
                if isinstance(operation, ast.Gt):
                    return lhs > rhs
                if isinstance(operation, ast.GtE):
                    return lhs >= rhs
                if isinstance(operation, ast.Is):
                    return lhs is rhs
                if isinstance(operation, ast.IsNot):
                    return lhs is not rhs
        if isinstance(expression, ast.BoolOp):
            operands = [
                parent
                for role, parent in sorted(parents.items())
                if role.startswith("value")
            ]
            if operands:
                result = (
                    False if isinstance(expression.op, ast.Or) else True
                )
                for operand in operands:
                    fixed = static_endpoint_value(
                        (endpoint[0], operand), visiting
                    )
                    if fixed is unresolved_static:
                        return unresolved_static
                    result = bool(fixed)
                    if (
                        isinstance(expression.op, ast.Or) and result
                    ) or (
                        isinstance(expression.op, ast.And) and not result
                    ):
                        return result
                return result
        if (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Attribute)
            and expression.func.attr in {"numel", "ndim", "ndims"}
        ):
            base = next((
                parent
                for role, parent in parents.items()
                if role in {"operand", "value", "base", "object"}
            ), None)
            if base is not None:
                meta = value_meta((endpoint[0], base))
                if meta is None:
                    projected = leaves(endpoint[0], base)
                    if len(projected) == 1 and () in projected:
                        meta = value_meta(projected[()])
                shape = None if meta is None else meta.shape
                if shape is not None:
                    if expression.func.attr in {"ndim", "ndims"}:
                        return len(shape)
                    count = 1
                    for extent in shape:
                        count *= int(extent)
                    return count
        if (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
        ):
            builtin_name = expression.func.id
            positional = tuple(
                parent
                for role, parent in sorted(
                    parents.items(),
                    key=lambda item: (
                        _positional_argument_index(item[0])
                        if _positional_argument_index(item[0]) is not None
                        else 1 << 30
                    ),
                )
                if _positional_argument_index(role) is not None
            )
            fixed_arguments = tuple(
                static_endpoint_value((endpoint[0], argument), visiting)
                for argument in positional
            )
            if builtin_name == "getattr" and len(fixed_arguments) >= 2:
                attribute_name = fixed_arguments[1]
                base_id = positional[0]
                if (
                    isinstance(attribute_name, str)
                    and not attribute_name.startswith("_")
                ):
                    meta = value_meta((endpoint[0], base_id))
                    if meta is not None:
                        if attribute_name == "shape" and meta.shape is not None:
                            return tuple(int(size) for size in meta.shape)
                        if attribute_name in {"ndim", "ndims"} and (
                            meta.shape is not None
                        ):
                            return len(meta.shape)
                        if attribute_name == "dtype" and meta.dtype is not None:
                            return str(meta.dtype)
                        if attribute_name == "device":
                            device = getattr(meta, "device", None)
                            if device is not None:
                                return device
                    owner_value = fixed_arguments[0]
                    if owner_value is not unresolved_static and isinstance(
                        owner_value,
                        (tuple, list, dict, range, str, bytes),
                    ):
                        try:
                            return getattr(owner_value, attribute_name)
                        except AttributeError:
                            pass
                if (
                    len(fixed_arguments) >= 3
                    and fixed_arguments[2] is not unresolved_static
                ):
                    return fixed_arguments[2]
            if builtin_name == "isinstance" and positional:
                candidate = fixed_arguments[0]
                type_expression = (
                    expression.args[1] if len(expression.args) > 1 else None
                )

                def safe_types(node: ast.AST | None):
                    if isinstance(node, ast.Name):
                        return {
                            "bool": (bool,), "int": (int,),
                            "float": (float,), "str": (str,),
                            "bytes": (bytes,), "tuple": (tuple,),
                            "list": (list,), "dict": (dict,),
                            "set": (set,), "range": (range,),
                        }.get(node.id)
                    if isinstance(node, ast.Tuple):
                        collected = tuple(
                            item
                            for element in node.elts
                            for item in (safe_types(element) or ())
                        )
                        return collected or None
                    return None

                accepted = safe_types(type_expression)
                if candidate is not unresolved_static and accepted is not None:
                    return isinstance(candidate, accepted)
            if all(
                argument is not unresolved_static
                for argument in fixed_arguments
            ):
                try:
                    if builtin_name in {"bool", "int", "float"}:
                        return {"bool": bool, "int": int, "float": float}[
                            builtin_name
                        ](*fixed_arguments)
                    if builtin_name in {"tuple", "list", "set"}:
                        constructor = {
                            "tuple": tuple, "list": list, "set": set,
                        }[builtin_name]
                        return constructor(*fixed_arguments)
                    if builtin_name == "len":
                        return len(*fixed_arguments)
                    if builtin_name == "range":
                        return range(*fixed_arguments)
                    if builtin_name == "enumerate":
                        return tuple(enumerate(*fixed_arguments))
                    if builtin_name == "zip":
                        return tuple(zip(*fixed_arguments))
                    if builtin_name == "sorted":
                        return sorted(*fixed_arguments)
                    if builtin_name == "max":
                        return max(*fixed_arguments)
                    if builtin_name == "min":
                        return min(*fixed_arguments)
                    if builtin_name == "all":
                        return all(*fixed_arguments)
                    if builtin_name == "any":
                        return any(*fixed_arguments)
                    if builtin_name == "slice":
                        return slice(*fixed_arguments)
                except (TypeError, ValueError, OverflowError):
                    pass
        if isinstance(data.get("expr_obj"), ast.Attribute):
            projected = leaves(*endpoint)
            if len(projected) == 1 and () in projected:
                target = projected[()]
                if target != endpoint:
                    return static_endpoint_value(target, visiting)
        return unresolved_static

    def static_predicate(
        closure_id: int,
        local_id: int,
    ) -> bool | None:
        owner = shells[int(closure_id)]
        graph = owner.process_graph.G
        data = graph.nodes[int(local_id)]
        expression = data.get("expr_obj")
        # Ordinary Python selection such as ``cached or build()`` becomes a
        # Phi whose test is the cached SSA value itself.  If that endpoint is
        # a source-static scalar (most importantly an optional parameter with
        # default None), resolve its truth value before considering comparison
        # syntax.  Tensor truth remains deliberately unresolved.
        direct_leaves = leaves(int(closure_id), int(local_id))
        if len(direct_leaves) == 1 and () in direct_leaves:
            direct = static_endpoint_value(direct_leaves[()])
            if direct is not unresolved_static and (
                direct is None
                or isinstance(direct, (bool, int, float, str, bytes))
            ):
                return bool(direct)
        if isinstance(expression, ast.UnaryOp) and isinstance(
            expression.op, ast.Not
        ):
            operand = next(
                (
                    int(parent)
                    for parent, role in (data.get("parents") or ())
                    if str(role) in {"operand", "value"}
                ),
                None,
            )
            if operand is not None:
                resolved = static_predicate(int(closure_id), operand)
                if resolved is not None:
                    return not resolved
        if (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
            and expression.func.id == "isinstance"
            and len(expression.args) == 2
            and isinstance(expression.args[1], ast.Name)
            and expression.args[1].id == "AbstractTensor"
        ):
            argument = next(
                (
                    int(parent)
                    for parent, role in (data.get("parents") or ())
                    if _positional_argument_index(str(role)) == 0
                ),
                None,
            )
            if argument is not None:
                projected = leaves(int(closure_id), argument)
                if len(projected) == 1 and () in projected:
                    # Tensor metadata is the compiler's structural type fact;
                    # this does not inspect or specialize the captured value.
                    if value_meta(projected[()]) is not None:
                        return True
                argument_data = graph.nodes[int(argument)]
                argument_name = (
                    argument_data.get("attributes") or {}
                ).get("binding_name")
                if (
                    argument_data.get("type") == "Input"
                    and argument_name in getattr(
                        owner, "_capture_tensor_input_names", ()
                    )
                ):
                    return True
        if not (
            isinstance(expression, ast.Compare)
            and len(expression.ops) == 1
            and len(expression.comparators) == 1
        ):
            return None
        if (
            isinstance(expression.left, ast.Name)
            and isinstance(expression.comparators[0], ast.Constant)
            and expression.comparators[0].value is None
            and isinstance(expression.ops[0], (ast.Is, ast.IsNot))
        ):
            identities = graph.graph.get("identity_table") or {}
            parameter_ids = identities.get(expression.left.id, ())
            if parameter_ids and (
                int(closure_id), int(parameter_ids[0])
            ) in argument_sources:
                # An explicitly bound argument is a value edge, regardless of
                # whether its runtime payload is a tensor or aggregate.  It
                # cannot be the omitted/default None branch -- unless the
                # bound value itself is declared None: a caller forwarding a
                # parameter whose ProgramABI fact is ``builtins.NoneType``
                # (``distribution`` through ``_propose_dt_pen``), or passing
                # the literal ``None``.  Treating that edge as "not None"
                # selected ``return distribution(...)`` over an opaque
                # callable and emptied the specialized callee.
                bound = leaves(int(closure_id), int(parameter_ids[0]))
                endpoint = bound.get(()) if len(bound) == 1 else None
                fact = (
                    unresolved_static if endpoint is None
                    else static_endpoint_value(endpoint)
                )
                declared_none = fact is None or (
                    isinstance(fact, _ProgramABIValueFact)
                    and str(fact.python_type) in {
                        "builtins.NoneType", "NoneType",
                    }
                )
                if declared_none:
                    return isinstance(expression.ops[0], ast.Is)
                return isinstance(expression.ops[0], ast.IsNot)
        parents = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
        }
        left = parents.get("lhs")
        right = parents.get("rhs")
        if left is None or right is None:
            return None
        left_leaves = leaves(int(closure_id), left)
        right_leaves = leaves(int(closure_id), right)
        if not (
            len(left_leaves) == len(right_leaves) == 1
            and () in left_leaves
            and () in right_leaves
        ):
            if isinstance(expression.ops[0], (ast.Is, ast.IsNot)):
                shell._profiler.trace(
                    path=owner.profile_path,
                    section="hierarchy-static-predicate",
                    label="non-scalar-identity-comparison",
                    fields={
                        "node": int(local_id),
                        "lhs": tuple(left_leaves.items()),
                        "rhs": tuple(right_leaves.items()),
                    },
                )
            return None
        lhs = static_endpoint_value(left_leaves[()])
        rhs = static_endpoint_value(right_leaves[()])
        if (
            lhs is unresolved_static or rhs is unresolved_static
        ) and isinstance(operation := expression.ops[0], (ast.Is, ast.IsNot)):
            shell._profiler.trace(
                path=owner.profile_path,
                section="hierarchy-static-predicate",
                label="unresolved-identity-comparison",
                fields={
                    "node": int(local_id),
                    "lhs_endpoint": left_leaves[()],
                    "rhs_endpoint": right_leaves[()],
                    "lhs": (
                        "<unresolved>"
                        if lhs is unresolved_static
                        else repr(lhs)
                    ),
                    "rhs": (
                        "<unresolved>"
                        if rhs is unresolved_static
                        else repr(rhs)
                    ),
                },
            )
        if lhs is unresolved_static or rhs is unresolved_static:
            return None
        operation = expression.ops[0]
        if isinstance(operation, ast.Is):
            return lhs is rhs
        if isinstance(operation, ast.IsNot):
            return lhs is not rhs
        if isinstance(operation, ast.Eq):
            return lhs == rhs
        if isinstance(operation, ast.NotEq):
            return lhs != rhs
        return None

    def leaves(
        closure_id: int,
        local_id: int,
    ) -> dict[tuple[object, ...], tuple[int, int]]:
        key = (int(closure_id), int(local_id))
        cached = resolved_leaves.get(key)
        if cached is not None:
            return cached
        if key in resolving:
            return {(): key}
        resolving.add(key)
        owner = shells[key[0]]
        graph = owner.process_graph.G
        result: dict[tuple[object, ...], tuple[int, int]] = {}
        source = identity_call_sources.get(key)
        if source is None:
            source = argument_sources.get(key)
        if source is None:
            source = control_alias_sources.get(key)
        if key in collection_owner_endpoints:
            # Planner collection/publication is a producer boundary.  In
            # particular, a LoopExit may retain the source initializer on its
            # syntactic ``value`` edge, but that initializer neither owns nor
            # produces the resident collection.
            result = {(): key}
        elif source is not None:
            result = leaves(*source)
        elif key[1] not in graph:
            result = {(): key}
        else:
            data = graph.nodes[key[1]]
            expression = data.get("expr_obj")
            attributes = data.get("attributes") or {}
            parents = tuple(
                (int(parent), str(role))
                for parent, role in (data.get("parents") or ())
                if str(role) not in {"callee", "func", "definition"}
            )
            if data.get("type") in {"LoopExit", "LoopResult"}:
                # LoopExit is a control-qualified SSA identity, not a new
                # numerical allocation.  The planner owns the control edge;
                # hierarchy/value reduction must carry the value edge through
                # so callers, returns, stream publications, and numerical
                # regions all name the same resident range.  Leaving the
                # wrapper as a distinct value fabricates an unproduced shell
                # slot and loses the producer's storage contract.
                value_parent = next(
                    (
                        parent
                        for parent, role in parents
                        if role == "value"
                    ),
                    None,
                )
                control_parent = next((
                    parent
                    for parent, role in parents
                    if role == "control"
                ), None)
                if (
                    data.get("type") == "LoopResult"
                    and control_parent is not None
                    and getattr(
                        owner, "_captured_loop_iterations", {}
                    ).get(int(control_parent)) == 0
                ):
                    binding_name = attributes.get("binding_name")
                    control_attributes = (
                        graph.nodes[int(control_parent)].get("attributes")
                        or {}
                    )
                    carried = (
                        control_attributes.get("loop_carried_bindings")
                        or {}
                    ).get(binding_name)
                    if carried is not None:
                        value_parent = int(carried[0])
                if (
                    value_parent is not None
                    and key not in collection_owner_endpoints
                ):
                    result = leaves(key[0], value_parent)
            elif data.get("type") == "Phi":
                test_id = next((
                    parent for parent, role in parents if role == "test"
                ), None)
                predicate = (
                    None
                    if test_id is None
                    else static_predicate(key[0], test_id)
                )
                selected_role = (
                    None
                    if predicate is None
                    else ("body" if predicate else "orelse")
                )
                selected = next((
                    parent
                    for parent, role in parents
                    if role == selected_role
                ), None)
                if selected is None:
                    phi_meta = value_meta(key)
                    if phi_meta is not None:
                        candidates = [
                            parent
                            for parent, role in parents
                            if role in {"body", "orelse"}
                            and (
                                candidate_meta := value_meta(
                                    (key[0], parent)
                                )
                            ) is not None
                            and tuple(candidate_meta.shape or ())
                            == tuple(phi_meta.shape or ())
                            and str(candidate_meta.dtype)
                            == str(phi_meta.dtype)
                        ]
                        if len(candidates) == 1:
                            selected = candidates[0]
                if selected is not None:
                    result = leaves(key[0], selected)
            elif isinstance(expression, ast.BoolOp):
                # Python ``a or b`` / ``a and b`` returns one operand; it is
                # not a numerical operation.  When operand truth is a source
                # fact (for example an optional parameter defaulting to
                # None), preserve the selected operand's SSA identity so
                # aggregate fields and tensor dependencies remain visible.
                operands = [
                    parent
                    for parent, role in parents
                    if (
                        str(role).startswith("value")
                        or str(role).startswith("arg")
                        or str(role) in {"operand", "values"}
                    )
                ]
                if not operands:
                    operands = [parent for parent, _role in parents]
                selected = None
                for index, operand in enumerate(operands):
                    projected = leaves(key[0], operand)
                    fixed = (
                        unresolved_static
                        if len(projected) != 1 or () not in projected
                        else static_endpoint_value(projected[()])
                    )
                    last = index == len(operands) - 1
                    if last:
                        selected = operand
                        break
                    if fixed is unresolved_static:
                        break
                    truth = bool(fixed)
                    if (
                        isinstance(expression.op, ast.Or) and truth
                    ) or (
                        isinstance(expression.op, ast.And) and not truth
                    ):
                        selected = operand
                        break
                if selected is not None:
                    result = leaves(key[0], selected)
            elif isinstance(expression, ast.IfExp):
                # Optional-parameter selection is a hierarchy fact when the
                # caller explicitly supplied that parameter.  Collapse
                # ``x if x is not None else ...`` to the supplied SSA
                # reference without consulting the captured tensor value.
                test = expression.test
                selected_role = None
                if (
                    isinstance(test, ast.Compare)
                    and len(test.ops) == 1
                    and len(test.comparators) == 1
                    and isinstance(test.left, ast.Name)
                    and isinstance(test.comparators[0], ast.Constant)
                    and test.comparators[0].value is None
                    and isinstance(test.ops[0], (ast.Is, ast.IsNot))
                ):
                    identities = graph.graph.get("identity_table") or {}
                    parameter_ids = identities.get(test.left.id, ())
                    if parameter_ids:
                        parameter_id = int(parameter_ids[0])
                        explicitly_supplied = (
                            (key[0], parameter_id) in argument_sources
                        )
                        if explicitly_supplied:
                            selected_role = (
                                "body"
                                if isinstance(test.ops[0], ast.IsNot)
                                else "orelse"
                            )
                        elif (
                            test.left.id
                            in (
                                graph.graph.get("parameter_defaults")
                                or {}
                            )
                        ):
                            default_is_none = (
                                graph.graph["parameter_defaults"][
                                    test.left.id
                                ]
                                is None
                            )
                            predicate = (
                                not default_is_none
                                if isinstance(test.ops[0], ast.IsNot)
                                else default_is_none
                            )
                            selected_role = (
                                "body" if predicate else "orelse"
                            )
                if selected_role is None:
                    test_id = next((
                        parent
                        for parent, role in parents
                        if role == "test"
                    ), None)
                    predicate = (
                        None
                        if test_id is None
                        else static_predicate(key[0], test_id)
                    )
                    if predicate is not None:
                        selected_role = (
                            "body" if predicate else "orelse"
                        )
                selected = next((
                    parent
                    for parent, role in parents
                    if role == selected_role
                ), None)
                if selected is not None:
                    result = leaves(key[0], selected)
            elif (
                attributes.get("materialization_kind")
                in {"unrolled_loop", "retained_loop_aggregate"}
            ):
                materialized = tuple(
                    int(value_id)
                    for value_id in attributes.get(
                        "materialized_value_ids", ()
                    )
                )
                for index, parent in enumerate(materialized):
                    for path, endpoint in leaves(
                        key[0], parent
                    ).items():
                        result[(int(index), *path)] = endpoint
            elif attributes.get("producer_kind") == "aggregate":
                elements = tuple(
                    int(value_id)
                    for value_id in attributes.get(
                        "aggregate_leaf_value_ids",
                        (),
                    )
                )
                for index, parent in enumerate(elements):
                    for path, endpoint in leaves(
                        key[0], parent
                    ).items():
                        result[(int(index), *path)] = endpoint
            elif data.get("type") in {"Indexed", "indexed"}:
                base = next((
                    parent for parent, role in parents
                    if role in {"base", "value", "object"}
                ), None)
                index = next((
                    parent for parent, role in parents
                    if role in {"index", "slice", "subscript"}
                ), None)
                fixed_index = (
                    unresolved_static
                    if index is None
                    else static_endpoint_value((key[0], index))
                )
                if (
                    base is not None
                    and isinstance(fixed_index, int)
                ):
                    projected = leaves(key[0], base)
                    result = {
                        path[1:]: endpoint
                        for path, endpoint in projected.items()
                        if path and path[0] == fixed_index
                    }
            elif isinstance(expression, ast.Attribute):
                base = next((
                    parent for parent, role in parents
                    if role in {"value", "base", "operand", "object"}
                ), None)
                projected = {} if base is None else leaves(key[0], base)
                field = str(expression.attr)
                result = {
                    path[1:]: endpoint
                    for path, endpoint in projected.items()
                    if path and str(path[0]) == field
                }
                if (
                    not result
                    and len(projected) == 1
                    and () in projected
                ):
                    # The aggregate may be a public/captured parameter whose
                    # Python class is intentionally absent.  A field reference
                    # is still a precise SSA identity: aggregate endpoint plus
                    # field path.  Canonicalize that identity instead of
                    # assigning every repeated Attribute AST node a new fake
                    # buffer ID.
                    result = {
                        (): field_endpoint(projected[()], field)
                    }
            elif attributes.get("producer_kind") in {
                "aggregate_materialization",
                "loop_materialization",
            }:
                # Materializing an iterable container does not create new
                # numerical values.  Preserve the source aggregate's ordered
                # leaves so downstream stack/cat operations can correlate the
                # one tape's transient leaf IDs with planner-owned identities.
                sources = tuple(
                    int(value_id)
                    for value_id in attributes.get(
                        "materialized_source_value_ids",
                        (),
                    )
                )
                if len(sources) == 1:
                    result = leaves(key[0], sources[0])
                else:
                    for index, source_id in enumerate(sources):
                        for path, endpoint in leaves(
                            key[0], source_id
                        ).items():
                            result[(int(index), *path)] = endpoint
            elif isinstance(expression, ast.Call) and (
                attributes.get("class_ref") is not None
                or (
                    isinstance(expression.func, ast.Name)
                    and expression.func.id
                    in (graph.graph.get("class_table") or {})
                )
                or (
                    graph.graph.get("method_binding") == "class"
                    and any(
                        role in {"callee", "func"}
                        and graph.nodes[parent].get("type") == "Input"
                        and (
                            graph.nodes[parent].get("attributes") or {}
                        ).get("binding_name") == "cls"
                        for parent, role in (
                            data.get("parents") or ()
                        )
                    )
                )
            ):
                # A classmethod commonly returns ``cls(...)``.  AST ingestion
                # cannot attach a concrete class_ref to that call because
                # ``cls`` is deliberately an SSA parameter, but the method's
                # lexical owner is still a compiler fact.  Treating the result
                # as opaque makes later ``result.field`` accesses escape as
                # fake shell inputs.  Resolve the aggregate schema from the
                # method owner without evaluating or retaining the Python
                # class object.
                class_name = (
                    attributes.get("class_ref")
                    or (
                        expression.func.id
                        if isinstance(expression.func, ast.Name)
                        and expression.func.id
                        in (graph.graph.get("class_table") or {})
                        else None
                    )
                    or graph.graph.get("method_owner")
                )
                descriptor = (
                    graph.graph.get("class_table", {}).get(
                        class_name
                    ) or {}
                )
                fields = tuple(descriptor.get("fields") or ())
                field_defaults = descriptor.get("field_defaults") or {}
                positional = {
                    position: parent
                    for parent, role in parents
                    if (
                        position := _positional_argument_index(role)
                    ) is not None
                }
                keywords = {
                    role.split(":", 1)[1]: parent
                    for parent, role in parents
                    if role.startswith("kw:")
                }
                for index, field in enumerate(fields):
                    parent = keywords.get(field, positional.get(index))
                    if parent is None:
                        if field in field_defaults:
                            endpoint = field_endpoint(key, str(field))
                            synthetic_static_values[endpoint] = (
                                field_defaults[field]
                            )
                            result[(str(field),)] = endpoint
                        continue
                    for path, endpoint in leaves(key[0], parent).items():
                        result[(str(field), *path)] = endpoint
            elif isinstance(expression, ast.Call) and (
                call := calls.get(key)
            ) is not None:
                child_id = int(call.callee.closure_id)
                child_owner = shells[child_id]
                child = child_owner.process_graph.G
                selected_returns = tuple(
                    getattr(
                        child_owner,
                        "_captured_return_value_ids",
                        (),
                    )
                )
                if len(selected_returns) == 1:
                    result = dict(leaves(
                        child_id, int(selected_returns[0])
                    ))
                    resolved_leaves[key] = result
                    resolving.remove(key)
                    return result
                output_names = tuple(
                    child.graph.get("function_outputs", ())
                )
                identities = child.graph.get("identity_table") or {}
                for output_index, name in enumerate(output_names):
                    values = identities.get(name, ())
                    if not values:
                        continue
                    prefix = (
                        () if len(output_names) == 1 else (output_index,)
                    )
                    for path, endpoint in leaves(
                        child_id, int(values[-1])
                    ).items():
                        result[(*prefix, *path)] = endpoint
            if not result:
                result = {(): key}
        resolving.remove(key)
        resolved_leaves[key] = result
        return result

    def canonical_global(closure_id: int, local_id: int) -> int:
        if (int(closure_id), int(local_id)) in loop_carried_update_endpoints:
            # A loop backedge is a cut in the projection graph.  Structural
            # leaf resolution walks value/alias/call edges as if the program
            # were acyclic, so it happily fuses the two endpoints of a cycle:
            # a LoopResult carries its value edge through to the very update
            # it publishes, and a caller result, local copy, or IndexedStore
            # correlated with the same source name resolves to that same leaf.
            # Both endpoints then land on one global and the header Phi
            # degenerates to Phi(x, x), leaving no body producer to find.
            # Storage reuse across the backedge stays an arena decision; the
            # update keeps its own SSA identity here so the pair survives to
            # Phi construction.  Same exclusion the alias table already
            # applies, extended to the projection that assigns identity.
            return global_value(closure_id, local_id)
        projected = leaves(closure_id, local_id)
        if len(projected) == 1 and () in projected:
            return global_value(*projected[()])
        return global_value(closure_id, local_id)

    # Canonicalization must apply to the hierarchy's shared value table, not
    # only to numerical region payloads.  A structural Phi/call/attribute and
    # a raw tape endpoint can already share one pre-reduction global ID.  If
    # only the numerical side follows the resolved source, composed control
    # retains the stale ID and fabricates a second, external shader input for
    # the same value.
    #
    # First discover an unambiguous redirect for every pre-reduction global
    # identity.  Then apply that redirect to *all* correlated endpoints and
    # compose control again from the unified table.  This is graph identity
    # reduction over immutable IR; it neither consults captured values nor
    # creates/replays a tape.
    projected_correlations = tuple(
        (
            int(closure_id),
            int(local_id),
            int(global_id),
            (
                canonical_global(int(closure_id), int(local_id))
                if int(closure_id) in shells
                else int(global_id)
            ),
        )
        for closure_id, local_id, global_id
        in value_table.correlations
    )
    projection_targets: dict[int, set[int]] = {}
    for _closure_id, _local_id, original, projected in (
        projected_correlations
    ):
        if projected != original:
            projection_targets.setdefault(original, set()).add(projected)
    aggregate_redirect_diagnostics = {}
    for closure_id, owner in shells.items():
        for aggregate_map in owner.compiled_aggregate_feed_paths:
            for capture_id, graph_input_id, path in aggregate_map:
                projected = leaves(
                    int(closure_id), int(graph_input_id)
                )
                target = projected.get(tuple(path))
                if target is None:
                    aggregate_redirect_diagnostics.setdefault(
                        int(closure_id), []
                    ).append({
                        "capture": int(capture_id),
                        "input": int(graph_input_id),
                        "path": tuple(path),
                        "projected": tuple(projected.items()),
                        "target": None,
                    })
                    continue
                original = global_value(
                    int(closure_id), int(capture_id)
                )
                destination = global_value(*target)
                aggregate_redirect_diagnostics.setdefault(
                    int(closure_id), []
                ).append({
                    "capture": int(capture_id),
                    "input": int(graph_input_id),
                    "path": tuple(path),
                    "projected": tuple(projected.items()),
                    "target": tuple(target),
                    "original": int(original),
                    "destination": int(destination),
                })
                if original != destination:
                    projection_targets.setdefault(
                        original, set()
                    ).add(destination)
    canonical_redirects = {
        original: next(iter(targets))
        for original, targets in projection_targets.items()
        if len(targets) == 1
    }
    shell.hierarchical_aggregate_redirect_diagnostics = {
        closure_id: tuple(rows)
        for closure_id, rows in aggregate_redirect_diagnostics.items()
    }

    def canonical_redirect(value_id: int) -> int:
        current = int(value_id)
        seen = set()
        while current in canonical_redirects and current not in seen:
            seen.add(current)
            current = int(canonical_redirects[current])
        return current

    def composed_global(closure_id: int, local_id: int) -> int:
        return canonical_redirect(canonical_global(closure_id, local_id))

    # A loop over an aggregate needs one resident binding per field actually
    # consumed by its body.  The local loop plan initially names the aggregate
    # target (for example ``packet``), while numerical regions name precise
    # field endpoints (``packet.octets`` and ``packet.byte_count``).  Expand
    # that one structural binding into field-wise source lists after hierarchy
    # leaves are known.  This is identity projection only: it neither reads a
    # captured object nor reconstructs the container at runtime.
    aggregate_binding_expansions: dict[
        tuple[int, int, str],
        tuple[tuple[int, int, str, tuple[int, ...]], ...],
    ] = {}
    aggregate_resident_bindings: set[tuple[int, int, str]] = set()
    aggregate_loop_bounds: dict[
        tuple[str, str, str],
        int,
    ] = {}
    aggregate_candidate_diagnostics: dict[int, list[dict[str, Any]]] = {}
    shell.hierarchical_control_binding_diagnostics = {
        int(closure_id): {
            "function": (
                shells[int(closure_id)].process_graph.G.graph.get(
                    "function_name"
                )
                if int(closure_id) in shells
                else None
            ),
            "iterable": tuple(control.iterable_bindings),
            "static": tuple(control.static_iterable_bindings),
            "closure": tuple(control.closure_iterable_bindings),
            "root": repr(control.root),
            "loop_plans": tuple(
                (
                    int(plan.loop.node_id),
                    plan.strategy.value,
                    int(plan.loop.iterable_node)
                    if plan.loop.iterable_node is not None
                    else None,
                    tuple(plan.loop.target_bindings),
                    tuple(plan.loop.publication_nodes),
                )
                for plan in shells[int(closure_id)].loop_plans
            ),
            "loop_reductions": tuple(
                (
                    int(reduction.loop_node_id),
                    bool(reduction.collapsible),
                    tuple(reduction.blockers),
                    reduction.control_program is not None,
                )
                for reduction
                in shells[int(closure_id)].loop_shader_reductions
            ),
        }
        for closure_id, control in controls.items()
        if int(closure_id) in shells
        and shells[int(closure_id)].process_graph.G.graph.get(
            "function_name"
        ) == "concatenate_resident_byte_packets"
    }
    for closure_id, local_control in controls.items():
        closure_id = int(closure_id)
        owner = shells.get(closure_id)
        if owner is None:
            continue
        owner_graph = owner.process_graph.G
        aggregate_candidates = (
            *tuple(
                (
                    iterable_id,
                    target_id,
                    induction,
                    source_ids,
                    False,
                )
                for iterable_id, target_id, induction, source_ids
                in local_control.closure_iterable_bindings
            ),
            *tuple(
                (
                    iterable_id,
                    target_id,
                    induction,
                    (iterable_id,),
                    True,
                )
                for iterable_id, target_id, induction
                in local_control.iterable_bindings
            ),
        )
        for (
            iterable_id,
            target_id,
            induction,
            _source_ids,
            was_resident,
        ) in aggregate_candidates:
            composed_induction = (
                f"{str(induction)}_closure_{closure_id}"
            )
            target_endpoint = (closure_id, int(target_id))
            projected_iterable = leaves(
                closure_id, int(iterable_id)
            )
            # Demand only attribute nodes downstream of this lexical target so
            # field_endpoint() has established every field the loop body uses.
            if int(target_id) in owner_graph:
                for descendant in nx.descendants(
                    owner_graph, int(target_id)
                ):
                    if isinstance(
                        owner_graph.nodes[descendant].get("expr_obj"),
                        ast.Attribute,
                    ):
                        leaves(closure_id, int(descendant))
            field_targets = {
                tuple(str(part) for part in path): endpoint
                for endpoint, (root, path)
                in synthetic_field_paths.items()
                if root == target_endpoint and path
            }
            if not field_targets:
                # A scalar loop target over an authored positional aggregate
                # is the ordinary generator/tuple case. The hierarchy leaf
                # walk already knows the exact ordered members, including
                # members specialized from a caller argument. Bind the target
                # from that resident series and replace the extent by its
                # proven arity. Requiring an attribute projection here left
                # ``tuple(f(x) for x in values)`` on the stale whole-formal
                # identity after the concordance had split ``values``.
                positional_sources = sorted(
                    (int(path[0]), endpoint)
                    for path, endpoint in projected_iterable.items()
                    if len(path) == 1 and isinstance(path[0], int)
                )
                indices = tuple(index for index, _ in positional_sources)
                if (
                    positional_sources
                    and indices == tuple(range(len(indices)))
                    and len(positional_sources) == len(projected_iterable)
                ):
                    global_key = (
                        composed_global(closure_id, int(iterable_id)),
                        composed_global(closure_id, int(target_id)),
                        composed_induction,
                    )
                    aggregate_binding_expansions[global_key] = ((
                        global_key[0], global_key[1], composed_induction,
                        tuple(
                            composed_global(*source)
                            for _index, source in positional_sources
                        ),
                    ),)
                    if was_resident:
                        aggregate_resident_bindings.add(global_key)
                    aggregate_loop_bounds[(
                        composed_induction,
                        f"__iterable_extent_{int(iterable_id)}__",
                        f"__iterable_extent_{global_key[0]}__",
                    )] = len(positional_sources)
                aggregate_candidate_diagnostics.setdefault(
                    closure_id, []
                ).append({
                    "iterable": int(iterable_id),
                    "target": int(target_id),
                    "induction": str(induction),
                    "resident": bool(was_resident),
                    "argument_source": argument_sources.get(
                        (closure_id, int(iterable_id))
                    ),
                    "projected": (),
                    "fields": (),
                })
                continue
            aggregate_candidate_diagnostics.setdefault(
                closure_id, []
            ).append({
                "iterable": int(iterable_id),
                "target": int(target_id),
                "induction": str(induction),
                "resident": bool(was_resident),
                "argument_source": argument_sources.get(
                    (closure_id, int(iterable_id))
                ),
                "projected": tuple(projected_iterable.items()),
                "fields": tuple(field_targets.items()),
            })
            expanded = []
            for field_path, field_target in sorted(
                field_targets.items()
            ):
                indexed_sources = sorted(
                    (
                        int(path[0]),
                        endpoint,
                    )
                    for path, endpoint in projected_iterable.items()
                    if (
                        path
                        and isinstance(path[0], int)
                        and tuple(str(part) for part in path[1:])
                        == field_path
                    )
                )
                if not indexed_sources:
                    continue
                indices = tuple(index for index, _ in indexed_sources)
                if indices != tuple(range(len(indices))):
                    continue
                expanded.append((
                    composed_global(closure_id, int(iterable_id)),
                    composed_global(*field_target),
                    composed_induction,
                    tuple(
                        composed_global(*source)
                        for _index, source in indexed_sources
                    ),
                ))
            if expanded:
                global_key = (
                    composed_global(closure_id, int(iterable_id)),
                    composed_global(closure_id, int(target_id)),
                    composed_induction,
                )
                aggregate_binding_expansions[global_key] = tuple(expanded)
                if was_resident:
                    aggregate_resident_bindings.add(global_key)
                aggregate_loop_bounds[(
                    composed_induction,
                    f"__iterable_extent_{int(iterable_id)}__",
                    f"__iterable_extent_{global_key[0]}__",
                )] = len(expanded[0][3])

    value_table = type(value_table)(tuple(
        (
            closure_id,
            local_id,
            canonical_redirect(projected),
        )
        for closure_id, local_id, _original, projected
        in projected_correlations
    ))
    shell.hierarchical_effective_value_table = value_table
    shell.hierarchical_aggregate_candidate_diagnostics = {
        closure_id: tuple(rows)
        for closure_id, rows in aggregate_candidate_diagnostics.items()
    }
    hierarchical = compose_hierarchical_control(
        hierarchy, controls, value_table
    )
    if aggregate_binding_expansions:
        expanded_bindings = []
        expanded_keys = set()
        for binding in hierarchical.program.closure_iterable_bindings:
            binding_key = (
                int(binding[0]),
                int(binding[1]),
                str(binding[2]),
            )
            replacement = aggregate_binding_expansions.get(binding_key)
            if replacement is None:
                expanded_bindings.append(binding)
            else:
                expanded_bindings.extend(replacement)
                expanded_keys.add(binding_key)
        for binding_key in aggregate_resident_bindings:
            if binding_key in expanded_keys:
                continue
            expanded_bindings.extend(
                aggregate_binding_expansions[binding_key]
            )

        def fix_aggregate_loop_bounds(block):
            from .control_source import (
                CallBlock,
                LoopControlBlock,
                LoopBlock,
                ParallelDeployment,
                StateMachineTick,
                WhileBlock,
            )

            if isinstance(block, LoopBlock):
                stop = str(block.stop)
                for (
                    induction,
                    local_marker,
                    global_marker,
                ), count in aggregate_loop_bounds.items():
                    if (
                        str(block.induction) == induction
                        and (
                            local_marker in stop
                            or global_marker in stop
                        )
                    ):
                        stop = str(int(count))
                return LoopBlock(
                    block.induction,
                    block.start,
                    stop,
                    block.step,
                    fix_aggregate_loop_bounds(block.body),
                    carried_aliases=block.carried_aliases,
                    result_ports=block.result_ports,
                    carried_seeds=block.carried_seeds,
                    parallel_iterations=block.parallel_iterations,
                    dispatch_shell=block.dispatch_shell,
                    recursion_region_id=block.recursion_region_id,
                    schedule_preference=block.schedule_preference,
                    sequence_mutations=block.sequence_mutations,
                    comparison=block.comparison,
                    terminal_controls=block.terminal_controls,
                    source_loop_node_id=block.source_loop_node_id,
                    control_site_ids=block.control_site_ids,
                )
            if isinstance(block, WhileBlock):
                return WhileBlock(
                    block.predicate_value_id,
                    fix_aggregate_loop_bounds(block.condition),
                    fix_aggregate_loop_bounds(block.body),
                    carried_aliases=block.carried_aliases,
                    result_ports=block.result_ports,
                    carried_seeds=block.carried_seeds,
                    recursion_region_id=block.recursion_region_id,
                    predicate_expression=block.predicate_expression,
                    sequence_mutations=block.sequence_mutations,
                    source_loop_node_id=block.source_loop_node_id,
                    terminal_controls=block.terminal_controls,
                    control_site_ids=block.control_site_ids,
                )
            if isinstance(block, LoopControlBlock):
                return block
            if isinstance(block, SequenceBlock):
                return SequenceBlock(tuple(
                    fix_aggregate_loop_bounds(child)
                    for child in block.blocks
                ))
            if isinstance(block, StateMachineTick):
                return replace(
                    block,
                    cases=tuple(
                        (case, fix_aggregate_loop_bounds(body))
                        for case, body in block.cases
                    ),
                    default=(
                        None if block.default is None
                        else fix_aggregate_loop_bounds(block.default)
                    ),
                )
            if isinstance(block, ParallelDeployment):
                return ParallelDeployment(
                    tuple(
                        fix_aggregate_loop_bounds(lane)
                        for lane in block.lanes
                    ),
                    block.schedule_preference,
                )
            if isinstance(block, CallBlock):
                return CallBlock(
                    block.callsite_id,
                    fix_aggregate_loop_bounds(block.callee),
                    block.argument_bindings,
                    block.result_bindings,
                )
            return block

        program = hierarchical.program
        hierarchical = type(hierarchical)(
            replace(
                program,
                root=fix_aggregate_loop_bounds(program.root),
                iterable_bindings=tuple(
                    binding
                    for binding in program.iterable_bindings
                    if (
                        int(binding[0]),
                        int(binding[1]),
                        str(binding[2]),
                    ) not in aggregate_resident_bindings
                ),
                closure_iterable_bindings=tuple(
                    dict.fromkeys(expanded_bindings)
                ),
            ),
            hierarchical.region_correlations,
        )
    shell.hierarchical_composed_closure_iterable_bindings = tuple(
        hierarchical.program.closure_iterable_bindings
    )
    shell.hierarchical_region_correlations = tuple(
        hierarchical.region_correlations
    )
    region_lookup = {
        (closure_id, local_region): global_region
        for closure_id, local_region, global_region
        in hierarchical.region_correlations
    }

    captured_regions = {}
    specialized_values = {}
    control_uniform_ids = {
        int(uniform.value_id)
        for uniform in hierarchical.program.uniforms
    }

    def add_specialization(global_id: int, fixed: Any) -> None:
        previous = specialized_values.get(global_id, fixed)
        if previous != fixed:
            raise ValueError(
                "hierarchical GLSL specialization conflict for "
                f"value {global_id}: {previous!r} != {fixed!r}"
            )
        specialized_values[global_id] = fixed

    for closure_id, owner in shells.items():
        for region_key, captured in owner.captured_region_programs.items():
            local_region = int(
                region_key
                if not isinstance(region_key, tuple)
                else region_key[-2]
            )
            global_region = region_lookup.get(
                (closure_id, local_region)
            )
            if global_region is None:
                continue
            programs = (captured.program, *captured.stages)
            all_ids = {
                int(value_id)
                for program in programs
                for value_id in (
                    *program.feeds,
                    *program.outputs.values(),
                    *(step.result_id for step in program.steps),
                    *(
                        input_id
                        for step in program.steps
                        for input_id in step.input_ids
                    ),
                )
            }
            id_map = {
                value_id: composed_global(closure_id, value_id)
                for value_id in all_ids
            }
            captured_regions[global_region] = (
                _remap_captured_all_ids(
                    captured, id_map, source_closure_id=closure_id,
                )
            )
        # Nested callsite shells are logical compartments of this artifact,
        # not separately finalized shaders.  Carry their typed specialization
        # facts directly from local lowering; requiring a throwaway child
        # shader merely to recover this map creates enormous compile-time CPU
        # churn and falsely treats every source call as a deployment.
        for value_id, fixed in (
            owner.composed_shell_specialized_values.items()
        ):
            global_id = composed_global(closure_id, int(value_id))
            add_specialization(global_id, fixed)
        owner_graph = owner.process_graph.G
        # Literal scalar nodes are already part of the source program.  When
        # hierarchy composition promotes one to control-uniform syntax, carry
        # the literal into the artifact specialization map instead of
        # inventing a host ABI input for it.
        for local_id, data in owner_graph.nodes(data=True):
            if data.get("type") not in {
                "Constant", "Const", "const", "StaticReference"
            }:
                continue
            try:
                fixed = _constant_value(data)
            except KeyError:
                continue
            if isinstance(fixed, (bool, int, float)):
                global_id = composed_global(
                    closure_id, int(local_id)
                )
                if global_id in control_uniform_ids:
                    add_specialization(global_id, fixed)
    # Defaulted scalar parameters and source-static predicate values are
    # compiler facts even when they are not literal nodes.  Resolve them
    # through explicit call bindings/defaults and specialize their canonical
    # identities; leaving them in the external set fabricates root ABI inputs
    # for nested optional arguments.
    inferred_static: dict[int, list[Any]] = {}
    for closure_id, local_id, _global_id in value_table.correlations:
        fixed = static_endpoint_value((closure_id, local_id))
        if isinstance(fixed, (bool, int, float)):
            inferred_static.setdefault(
                composed_global(closure_id, local_id), []
            ).append(fixed)
    for global_id, candidates in inferred_static.items():
        first = candidates[0]
        if not all(
            type(candidate) is type(first) and candidate == first
            for candidate in candidates[1:]
        ):
            continue
        previous = specialized_values.get(global_id, first)
        if type(previous) is type(first) and previous == first:
            specialized_values[global_id] = first
    shell._profiler.trace(
        path=shell.profile_path,
        section="hierarchical-artifact",
        label="numerical remap complete",
        fields={
            "regions": len(captured_regions),
            "elapsed_ms": round(
                (time.perf_counter() - compose_started) * 1e3,
                3,
            ),
        },
    )
    program_value_origins: dict[int, list[dict[str, Any]]] = {}
    region_owners = {
        int(global_region): {
            "closure": int(closure_id),
            "function": shells[int(closure_id)].process_graph.G.graph.get(
                "function_name"
            ),
            "local_region": int(local_region),
        }
        for closure_id, local_region, global_region
        in hierarchical.region_correlations
        if int(closure_id) in shells
    }
    for global_region, captured in captured_regions.items():
        programs = (
            tuple(captured.stages)
            if captured.stages
            else (captured.program,)
        )
        for stage_index, program in enumerate(programs):
            def note(value_id: int, role: str, **extra: Any) -> None:
                program_value_origins.setdefault(
                    int(value_id), []
                ).append({
                    "region": int(global_region),
                    **region_owners.get(int(global_region), {}),
                    "stage": int(stage_index),
                    "role": str(role),
                    **extra,
                })

            for value_id in ordered_feed_ids(program):
                note(value_id, "feed")
            for name, value_id in program.outputs.items():
                note(value_id, "output", name=str(name))
            for step_index, step in enumerate(program.steps):
                for input_index, value_id in enumerate(step.input_ids):
                    note(
                        value_id,
                        "step-input",
                        step=int(step_index),
                        input=int(input_index),
                        op=str(step.op_name),
                    )
                note(
                    step.result_id,
                    "step-result",
                    step=int(step_index),
                    op=str(step.op_name),
                )
    shell.hierarchical_program_value_origins = {
        value_id: tuple(origins)
        for value_id, origins in program_value_origins.items()
    }
    hierarchical_value_ids = {
        int(value_id)
        for captured in captured_regions.values()
        for program in (
            tuple(captured.stages)
            if captured.stages
            else (captured.program,)
        )
        for value_id in (
            *ordered_feed_ids(program),
            *tuple(program.outputs.values()),
        )
    }
    hierarchical = type(hierarchical)(
        project_control_regions(
            hierarchical.program,
            hierarchical.program.region_indices,
            retained_value_ids=hierarchical_value_ids,
        ),
        hierarchical.region_correlations,
    )
    shell.hierarchical_control_program = hierarchical.program
    shell.hierarchical_captured_region_programs = dict(captured_regions)
    if set(captured_regions) != set(
        hierarchical.program.region_indices
    ):
        shell.hierarchical_compose_failure = {
            "reason": "captured-control-region-mismatch",
            "captured_only": tuple(sorted(
                set(captured_regions)
                - set(hierarchical.program.region_indices)
            )),
            "control_only": tuple(sorted(
                set(hierarchical.program.region_indices)
                - set(captured_regions)
            )),
            "captured_only_origins": tuple(
                (
                    int(global_region),
                    int(closure_id),
                    int(local_region),
                    shells[int(closure_id)].process_graph.G.graph.get(
                        "function_name"
                    ),
                )
                for closure_id, local_region, global_region
                in hierarchical.region_correlations
                if global_region in (
                    set(captured_regions)
                    - set(hierarchical.program.region_indices)
                )
            )[:12],
        }
        return None
    root_id = int(hierarchy.closure_id)
    root_locals = {
        int(local_id)
        for closure_id, local_id, _global_id
        in value_table.correlations
        if int(closure_id) == root_id
    }
    # Identity-call reduction may erase a parameter's explicit correlation
    # after all of its uses move into a child closure.  It remains a public
    # root ABI value and must still receive its canonical global identity.
    # Derive root inputs from the source graph, not merely from the reduced
    # correlation table, or valid parameters surface as anonymous private
    # shader feeds.
    root_locals.update(
        int(local_id)
        for local_id, data in shell.process_graph.G.nodes(data=True)
        if data.get("type") == "Input"
        and shell.process_graph.G.out_degree(local_id)
    )
    shell.hierarchical_root_value_ids = {
        local_id: composed_global(root_id, local_id)
        for local_id in root_locals
    }
    # Preserve the compiler-generated portion of the global namespace as
    # identities only.  These are not captured values and carry no Python
    # objects; the table makes any accidentally exposed private operand
    # diagnosable as its owning closure/local SSA endpoint instead of an
    # inscrutable integer in the shell ABI.
    shell.hierarchical_private_value_ids = {
        int(global_id): (int(closure_id), int(local_id))
        for (closure_id, local_id), global_id in private_values.items()
    }
    shell.hierarchical_capture_value_ids = {
        int(global_id): int(capture_id)
        for capture_id, global_id in capture_values.items()
    }
    shell.hierarchical_synthetic_field_paths = {
        int(global_id): (
            (int(root[0]), int(root[1])),
            tuple(str(part) for part in path),
        )
        for endpoint, (root, path) in synthetic_field_paths.items()
        if (global_id := private_values.get(endpoint)) is not None
    }
    endpoint_details: dict[int, list[dict[str, Any]]] = {}

    def describe_endpoint(
        global_id: int,
        closure_id: int,
        local_id: int,
    ) -> None:
        owner = shells.get(int(closure_id))
        graph = None if owner is None else owner.process_graph.G
        data = (
            {}
            if graph is None or int(local_id) not in graph
            else graph.nodes[int(local_id)]
        )
        meta = value_meta((int(closure_id), int(local_id)))
        endpoint_details.setdefault(int(global_id), []).append({
            "closure": int(closure_id),
            "function": (
                None
                if graph is None
                else graph.graph.get("function_name")
            ),
            "local": int(local_id),
            "type": data.get("type"),
            "op": data.get("op"),
            "label": data.get("label"),
            "parents": tuple(
                (int(parent), str(role))
                for parent, role in (data.get("parents") or ())
            ),
            "attributes": {
                str(name): value
                for name, value in (data.get("attributes") or {}).items()
                if isinstance(value, (bool, int, float, str, type(None)))
            },
            "meta": (
                None
                if meta is None
                else {
                    "shape": tuple(meta.shape or ()),
                    "dtype": str(meta.dtype),
                }
            ),
            "compiled_feed_meta": (
                {}
                if owner is None
                else {
                    int(value_id): {
                        "shape": tuple(value_meta.shape or ()),
                        "dtype": str(value_meta.dtype),
                    }
                    for value_id, value_meta
                    in owner.compiled_feed_meta.items()
                }
            ),
            "captured_return_value_ids": (
                ()
                if owner is None
                else tuple(sorted(getattr(
                    owner, "_captured_return_value_ids", ()
                )))
            ),
            "hierarchy_call": (
                int(closure_id), int(local_id)
            ) in calls,
            "call_result_bindings": tuple(
                getattr(
                    calls.get((int(closure_id), int(local_id))),
                    "result_bindings",
                    (),
                )
            ),
            "argument_source": argument_sources.get((
                int(closure_id), int(local_id)
            )),
            "control_alias_source": control_alias_sources.get((
                int(closure_id), int(local_id)
            )),
            "parent_details": (
                ()
                if graph is None
                else tuple(
                    (
                        int(parent),
                        str(role),
                        graph.nodes[int(parent)].get("type"),
                        graph.nodes[int(parent)].get("label"),
                        dict(
                            graph.nodes[int(parent)].get("attributes") or {}
                        ),
                    )
                    for parent, role in (data.get("parents") or ())
                    if int(parent) in graph
                )
            ),
            "capture_input_types": (
                {}
                if owner is None
                else {
                    str(name): tuple(sorted(types))
                    for name, types in getattr(
                        owner, "_capture_input_type_names", {}
                    ).items()
                }
            ),
            "captured_source_branches": (
                ()
                if owner is None
                else tuple(getattr(
                    owner, "_captured_source_branches", ()
                ))
            ),
            "expr": (
                None
                if not isinstance(data.get("expr_obj"), ast.AST)
                else ast.dump(
                    data["expr_obj"],
                    annotate_fields=False,
                    include_attributes=False,
                )[:240]
            ),
        })

    for closure_id, local_id, global_id in value_table.correlations:
        describe_endpoint(global_id, closure_id, local_id)
    for (closure_id, local_id), global_id in private_values.items():
        describe_endpoint(global_id, closure_id, local_id)
        rooted = synthetic_field_paths.get((closure_id, local_id))
        if rooted is not None:
            (root_closure, root_local), field_path = rooted
            describe_endpoint(global_id, root_closure, root_local)
            endpoint_details[int(global_id)][-1]["field_path"] = tuple(
                str(part) for part in field_path
            )
    shell.hierarchical_endpoint_details = {
        global_id: tuple(details)
        for global_id, details in endpoint_details.items()
    }
    # Preserve source provenance through graph-inert tensor accommodation.
    # A common example is ``x if isinstance(x, AbstractTensor) else
    # AbstractTensor.tensor(x)``: both arms name the same semantic input, but
    # the IfExp owns a private ProcessGraph endpoint.  It must not become a
    # fabricated native ABI input merely because tensor discovery selected
    # one arm.  This is derived exclusively from scoped graph edges and the
    # hierarchy value table, never from observed values or wrapper identity.
    transparent_cache: dict[tuple[int, int], int | None] = {}

    def transparent_origin(
        closure_id: int,
        local_id: int,
        visiting: frozenset[tuple[int, int]] = frozenset(),
    ) -> int | None:
        endpoint = (int(closure_id), int(local_id))
        if endpoint in transparent_cache:
            return transparent_cache[endpoint]
        if endpoint in visiting:
            return None
        owner = shells.get(endpoint[0])
        graph = None if owner is None else owner.process_graph.G
        if graph is None or endpoint[1] not in graph:
            return None
        data = graph.nodes[endpoint[1]]
        parents = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
        }
        next_visiting = visiting | {endpoint}
        origin = None
        if data.get("type") == "Input":
            origin = composed_global(*endpoint)
        elif isinstance(data.get("expr_obj"), ast.IfExp):
            body = parents.get("body")
            orelse = parents.get("orelse")
            body_origin = (
                None if body is None else transparent_origin(
                    endpoint[0], body, next_visiting
                )
            )
            orelse_origin = (
                None if orelse is None else transparent_origin(
                    endpoint[0], orelse, next_visiting
                )
            )
            if body_origin is not None and body_origin == orelse_origin:
                origin = body_origin
        elif isinstance(data.get("expr_obj"), ast.Call):
            expression = data["expr_obj"]
            function = expression.func
            is_tensor_accommodation = (
                isinstance(function, ast.Attribute)
                and function.attr == "tensor"
                and isinstance(function.value, ast.Name)
                and function.value.id == "AbstractTensor"
            )
            if is_tensor_accommodation:
                argument = next((
                    int(parent)
                    for parent, role in (data.get("parents") or ())
                    if str(role) in {"arg", "args", "arg:0", "arg0"}
                ), None)
                if argument is not None:
                    origin = transparent_origin(
                        endpoint[0], argument, next_visiting
                    )
        transparent_cache[endpoint] = origin
        return origin

    hierarchical_value_aliases = {}
    for global_id, details in shell.hierarchical_endpoint_details.items():
        for detail in details:
            origin = transparent_origin(
                int(detail["closure"]), int(detail["local"])
            )
            if origin is not None and int(origin) != int(global_id):
                hierarchical_value_aliases[int(global_id)] = int(origin)
                break
    shell.hierarchical_value_aliases = hierarchical_value_aliases
    root_parameter_names = {
        str((data.get("attributes") or {}).get("binding_name"))
        for _node_id, data in shell.process_graph.G.nodes(data=True)
        if data.get("type") == "Input"
        and (data.get("attributes") or {}).get("binding_kind")
        == "parameter"
    }

    def attribute_path(local_id: int) -> str | None:
        if int(local_id) not in shell.process_graph.G:
            return None
        data = shell.process_graph.G.nodes[int(local_id)]
        expression = data.get("expr_obj")
        if not isinstance(expression, ast.Attribute):
            return None
        parts = [str(expression.attr)]
        current = expression.value
        while isinstance(current, ast.Attribute):
            parts.append(str(current.attr))
            current = current.value
        if (
            not isinstance(current, ast.Name)
            or current.id not in root_parameter_names
        ):
            return None
        return ".".join((current.id, *reversed(parts)))

    shell.hierarchical_root_field_value_ids = {
        path: int(global_id)
        for global_id, (closure_id, local_id)
        in shell.hierarchical_private_value_ids.items()
        if int(closure_id) == root_id
        and (path := attribute_path(local_id)) is not None
    }
    for global_id, endpoint in (
        shell.hierarchical_private_value_ids.items()
    ):
        rooted = synthetic_field_paths.get(endpoint)
        if rooted is None:
            continue
        (closure_id, local_id), field_path = rooted
        if int(closure_id) != root_id:
            continue
        root_graph = shell.process_graph.G
        if int(local_id) not in root_graph:
            continue
        root_data = root_graph.nodes[int(local_id)]
        root_attributes = root_data.get("attributes") or {}
        if (
            root_data.get("type") != "Input"
            or root_attributes.get("binding_kind") != "parameter"
        ):
            continue
        name = root_attributes.get("binding_name")
        if name is not None:
            shell.hierarchical_root_field_value_ids[
                ".".join((str(name), *field_path))
            ] = int(global_id)
    graph = shell.process_graph.G
    identities = graph.graph.get("identity_table") or {}
    terminals = {}
    root_output_leaves = set()
    for name in graph.graph.get("function_outputs", ()):
        if not identities.get(name):
            continue
        for path, endpoint in leaves(
            root_id, int(identities[name][-1])
        ).items():
            global_id = global_value(*endpoint)
            suffix = "".join(f".{part}" for part in path)
            terminals[f"{name}{suffix}"] = global_id
            root_output_leaves.add(global_id)
    resident_values = {
        int(value_id)
        for captured in captured_regions.values()
        for program in (
            tuple(captured.stages)
            if captured.stages
            else (captured.program,)
        )
        for value_id in (
            *ordered_feed_ids(program),
            *tuple(program.outputs.values()),
        )
    }
    resident_values.update(
        int(collection_id)
        for _source_id, collection_id, _induction, _start
        in hierarchical.program.collection_bindings
    )
    terminals = {
        name: value_id
        for name, value_id in terminals.items()
        if value_id in resident_values
    }
    shell.hierarchical_public_output_ids = set(terminals.values())
    shell.hierarchical_terminal_outputs = dict(terminals)
    shell.hierarchical_specialized_values = dict(specialized_values)
    stream_outputs = {}
    for plan in shell.loop_plans:
        for statement_id, published_value_id, _count_id in (
            plan.loop.publication_nodes
        ):
            for name in graph.graph.get("function_outputs", ()):
                output_ids = tuple(identities.get(name, ()))
                if not output_ids:
                    continue
                output_id = int(output_ids[-1])
                if (
                    statement_id in graph
                    and output_id in graph
                    and nx.has_path(graph, statement_id, output_id)
                ):
                    stream_outputs[str(name)] = int(
                        shell.hierarchical_root_value_ids[
                            int(published_value_id)
                        ]
                    )
    shell._profiler.trace(
        path=shell.profile_path,
        section="hierarchical-artifact",
        label="shader artifact begin",
        fields={
            "regions": len(captured_regions),
            "terminals": len(terminals),
        },
    )
    hierarchical_storage_meta = dict(global_meta)
    for endpoint, global_id in private_values.items():
        meta = value_meta(endpoint)
        if meta is not None:
            hierarchical_storage_meta.setdefault(int(global_id), meta)
    for closure_id, local_id, global_id in value_table.correlations:
        meta = value_meta((int(closure_id), int(local_id)))
        if meta is not None:
            hierarchical_storage_meta.setdefault(int(global_id), meta)
    for capture_id, global_id in capture_values.items():
        meta = next(
            (
                value_meta((int(closure_id), int(capture_id)))
                for closure_id in shells
                if value_meta((int(closure_id), int(capture_id))) is not None
            ),
            None,
        )
        if meta is not None:
            hierarchical_storage_meta.setdefault(int(global_id), meta)
    hierarchy_contract_diagnostics = {
        int(global_id): ({
            "kind": "hierarchy",
            "private_endpoint": (
                shell.hierarchical_private_value_ids.get(int(global_id))
            ),
            "capture_endpoint": (
                shell.hierarchical_capture_value_ids.get(int(global_id))
            ),
            "endpoints": tuple(details),
            "program_origins": tuple(
                shell.hierarchical_program_value_origins.get(
                    int(global_id), ()
                )
            ),
            "synthetic_field": (
                shell.hierarchical_synthetic_field_paths.get(
                    int(global_id)
                )
            ),
        },)
        for global_id, details in shell.hierarchical_endpoint_details.items()
    }
    try:
        # Retain the target-neutral whole-program form alongside the optional
        # GLSL artifact. AOT consumers (notably WebAssembly) need the same
        # hierarchy and region identities without reverse-engineering shader
        # source or selecting one nested shell as a surrogate root.
        shell.hierarchical_control_program = hierarchical.program
        shell.hierarchical_captured_region_programs = dict(captured_regions)
        artifact = build_control_shader_artifact(
            hierarchical.program,
            captured_regions,
            value_meta=hierarchical_storage_meta,
            value_contract_diagnostics=hierarchy_contract_diagnostics,
            instrumentation=shell._profiler.verbose,
            terminal_outputs=terminals or None,
            stream_outputs=stream_outputs,
            specialized_values=specialized_values,
            device_resident=(shell.shell_language == "glsl"),
        )
    except Exception as error:
        program = getattr(error, "program", None)
        value_ids = (
            ()
            if program is None
            else tuple(dict.fromkeys((
                *ordered_feed_ids(program),
                *program.outputs.values(),
                *(step.result_id for step in program.steps),
            )))
        )
        raise RuntimeError(
            f"{error}; endpoint_details="
            f"{ {int(value_id): shell.hierarchical_endpoint_details.get(int(value_id), ()) for value_id in value_ids}!r}"
        ) from error
    shell._profiler.trace(
        path=shell.profile_path,
        section="hierarchical-artifact",
        label="complete",
        fields={
            "elapsed_ms": round(
                (time.perf_counter() - compose_started) * 1e3,
                3,
            ),
        },
    )
    return artifact


def _compile_whole_process_graph(
    shell: Any,
    *,
    device: Any = None,
    prepare_ephemerals: bool = False,
    _visited: set[int] | None = None,
) -> Any:
    if getattr(shell, "process_graph_boundary", None):
        shell.whole_program_compiled = True
        shell.whole_program_device = device
        return shell
    """Prepare every planned dispatch and validate complete graph coverage.

    ``GraphDeepCompiler`` produces Python callables composed from the
    AbstractTensor operator table.  Selecting the GLSL backend therefore
    delegates their numerical operations to GLSL, but this stage does not
    claim that the complete graph has become one fused shader.
    """

    visited = set() if _visited is None else _visited
    identity = id(shell)
    if identity in visited:
        return shell
    visited.add(identity)

    if (
        getattr(shell, "runtime_closure_only", False)
        and getattr(shell, "_owns_function_shells", False)
    ):
        # The outer shell owns the complete source catalogue and map records;
        # it is not a second execution of the module containing the requested
        # entrypoint.  Compile the proven activation roots.  Their complete
        # call topology is represented by callsite_function_shells and is
        # recursively validated by the ordinary path below.
        activation_roots = tuple(
            getattr(shell, "activation_root_references", ())
        )
        if not activation_roots:
            raise RuntimeError(
                "runtime-root ProcessGraph has no activation root"
            )
        for reference in activation_roots:
            function_shell = shell.function_shells.get(int(reference))
            if function_shell is None:
                raise RuntimeError(
                    "ProcessGraph activation root has no deployment shell: "
                    f"{reference}"
                )
            _compile_whole_process_graph(
                function_shell,
                device=device,
                prepare_ephemerals=prepare_ephemerals,
                _visited=visited,
            )
        shell.compiled_dispatch_functions = ()
        shell.compiled_dispatch_sources = ()
        shell.whole_program_compiled = True
        shell.whole_program_device = device
        return shell

    covered = {
        node_id
        for subgraph in shell.dispatch_subgraphs
        for node_id in subgraph.G.graph.get("deployment_nodes", ())
    }
    is_dispatch_metadata = _dispatch_metadata_node_classifier(shell.process_graph)
    uncovered = [
        node_id
        for node_id in shell.process_graph.G
        if node_id not in covered
        and not is_dispatch_metadata(node_id)
    ]
    if uncovered:
        details = ", ".join(
            f"{node_id}:{shell.process_graph.G.nodes[node_id].get('type')}"
            for node_id in uncovered
        )
        raise RuntimeError(
            "whole ProcessGraph compilation has uncovered runtime nodes: "
            + details
        )

    functions = []
    sources = []
    # An ephemeral is a GraphDeepCompiler-built PYTHON callable for a numeric
    # region -- only useful when a numeric region has to stay executable as
    # Python (runtime execution). Deriving the dual IR (the whole-program,
    # no-bake precompile) does not execute anything, so it does not need them,
    # and eagerly building them here forces the deep compiler over the whole
    # graph -- including control/binding constructs (a walrus, an annotation)
    # that are not tensor operators and have no op-table entry -- which is the
    # wrong question to ask of them. So prepare them only when explicitly asked.
    # The runtime consumer falls back to the ephemeral itself (prepared lazily
    # on first call) when compiled_dispatch_functions is empty, so leaving them
    # unprepared here is safe.
    if prepare_ephemerals:
        for ephemeral in shell.ephemeral_callables:
            ephemeral.prepare(device=device)
            functions.append(ephemeral)
            sources.append(ephemeral.generated_source)
    direct_callees = [
        (node_id, int(reference))
        for node_id, data in shell.process_graph.G.nodes(data=True)
        for reference in (
            (data.get("attributes") or {}).get("callee_ref")
            or (data.get("attributes") or {}).get("method_ref"),
        )
        if reference is not None
    ]
    for node_id, reference in direct_callees:
        function_shell = (
            getattr(shell, "callsite_function_shells", {}).get(node_id)
            or shell.function_shells.get(reference)
        )
        if function_shell is None:
            try:
                referenced_entry = shell.process_graph.function_table.entry(
                    reference
                )
            except (AttributeError, KeyError):
                referenced_entry = None
            if (
                referenced_entry is not None
                and referenced_entry.metadata.get("host_ssa_module") is not None
            ):
                # The callee is already fully represented as repository SSA;
                # it does not need or admit a source-planning shell.
                continue
            if (
                referenced_entry is not None
                and referenced_entry.metadata.get("implementation_kind")
                == "authored-source-fallback"
            ):
                # A bounded function-shell subdivision deliberately compiles
                # only one authored function's numerical regions. Calls that
                # leave the cut remain exact authored source and are restored
                # whenever recursive compilation dives through them. This is
                # an explicit implementation contract attached by the
                # subdivision orchestrator, not permission to overlook an
                # accidentally missing deployment shell in an ordinary build.
                continue
            function_name = shell.process_graph.G.graph.get(
                "function_name", "?"
            )
            try:
                referenced_name = (
                    shell.process_graph.function_table.entry(reference)
                    .qualified_name
                )
            except (AttributeError, KeyError):
                referenced_name = "?"
            node_data = shell.process_graph.G.nodes[node_id]
            expression = node_data.get("expr_obj")
            raise RuntimeError(
                "ProcessGraph references a function with no deployment "
                f"shell: {reference}; owner={function_name!r}; "
                f"referenced={referenced_name!r}; node={node_id}; "
                f"type={node_data.get('type')!r}; "
                f"source={ast.dump(expression, include_attributes=False) if isinstance(expression, ast.AST) else repr(expression)}; "
                "planned_references="
                f"{tuple(sorted(map(int, shell.function_shells)))}"
            )
        _compile_whole_process_graph(
            function_shell,
            device=device,
            prepare_ephemerals=prepare_ephemerals,
            _visited=visited,
        )
    shell.compiled_dispatch_functions = tuple(functions)
    shell.compiled_dispatch_sources = tuple(sources)
    shell.whole_program_compiled = True
    shell.whole_program_device = device
    return shell


def _source_fallthrough_predicate_facts(graph: Any, controls, predicate_fact):
    """Facts reaching source expressions, excluding paths which already exit.

    Predicates are graph value identities. Loop exits and exception handlers
    do not export newly inferred facts; nested function bodies are separate
    scopes. Unknown control remains conservative rather than guessing a path.
    """
    body = graph.G.graph.get("function_body")
    if not body:
        return None
    expressions = {}
    for node_id, data in graph.G.nodes(data=True):
        expression = data.get("expr_obj")
        if isinstance(expression, ast.AST):
            expressions.setdefault(_ast_source_signature(expression), []).append(int(node_id))
    predicates = {}
    for record in controls.values():
        if record.get("predicate_id") in graph.G:
            predicates[_ast_source_signature(record["expression"])] = int(record["predicate_id"])
    reached = {}

    def observed(node, facts):
        for node_id in expressions.get(_ast_source_signature(node), ()):
            reached[node_id] = facts if node_id not in reached else reached[node_id] & facts

    def branch_facts(node, facts, truth):
        predicate = predicates.get(_ast_source_signature(node))
        return facts if predicate is None else facts | {predicate_fact(predicate, truth)}

    def expression(node, facts):
        if node is None or isinstance(node, (ast.Lambda, ast.GeneratorExp)):
            return
        observed(node, facts)
        if isinstance(node, ast.IfExp):
            expression(node.test, facts)
            expression(node.body, branch_facts(node, facts, True))
            expression(node.orelse, branch_facts(node, facts, False))
        else:
            for child in ast.iter_child_nodes(node):
                if not isinstance(child, ast.stmt):
                    expression(child, facts)

    def sequence(statements, facts):
        for statement in statements:
            if facts is None:
                break
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            observed(statement, facts)
            if isinstance(statement, ast.If):
                expression(statement.test, facts)
                left = sequence(statement.body, branch_facts(statement, facts, True))
                right = sequence(statement.orelse, branch_facts(statement, facts, False))
                facts = right if left is None else left if right is None else left & right
            elif isinstance(statement, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                for child in ast.iter_child_nodes(statement):
                    expression(child, facts)
                facts = None
            elif isinstance(statement, (ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith)):
                for child in ast.iter_child_nodes(statement):
                    if not isinstance(child, ast.stmt):
                        expression(child, facts)
                sequence(statement.body, facts)
                sequence(getattr(statement, "orelse", ()), facts)
            elif isinstance(statement, (ast.Try, ast.TryStar)):
                sequence(statement.body, facts)
                sequence(statement.orelse, facts)
                sequence(statement.finalbody, facts)
                for handler in statement.handlers:
                    sequence(handler.body, facts)
            else:
                expression(statement, facts)
        return facts

    sequence(body, frozenset())
    return reached


def _publish_conditional_tuple_members(graph: Any) -> bool:
    """Publish position-matched tuple members as control-owned merge values."""
    changed = False
    memberships = _branch_compartments(graph)
    controls = _source_control_records(graph.G)

    def predicate_fact(value_id, truth):
        data = graph.G.nodes.get(int(value_id), {})
        if isinstance(data.get("expr_obj"), ast.UnaryOp) and isinstance(data["expr_obj"].op, ast.Not):
            operand = next((int(parent) for parent, role in data.get("parents", ())
                            if str(role) == "operand"), None)
            if operand is not None:
                return predicate_fact(operand, not truth)
        return int(value_id), bool(truth)

    fallthrough_facts = _source_fallthrough_predicate_facts(graph, controls, predicate_fact)
    for node_id, data in tuple(graph.G.nodes(data=True)):
        if not isinstance(data.get("expr_obj"), ast.IfExp):
            continue
        attributes = dict(data.get("attributes") or {})
        if attributes.get("conditional_member_bindings"):
            continue
        parents = {str(role): int(parent) for parent, role in data.get("parents", ())}
        if not all(parents.get(role) in graph.G for role in ("test", "body", "orelse")):
            continue
        arms = [graph.G.nodes[parents[role]].get("attributes") or {}
                for role in ("body", "orelse")]
        leaves = [_authored_aggregate_leaves(graph, parents[role]) for role in ("body", "orelse")]
        optional = None
        absent_arms = [
            graph.G.nodes[parents[role]].get("type") in {"Constant", "Const", "const"}
            and _constant_value(graph.G.nodes[parents[role]]) is None
            for role in ("body", "orelse")
        ]
        if sum(absent_arms) == 1:
            present_arm = 1 if absent_arms[0] else 0
            if arms[present_arm].get("aggregate_kind") != "tuple" or not leaves[present_arm]:
                continue
            descriptors = [_tensor_descriptor(graph, leaf) for leaf in leaves[present_arm]]
            if any(item is None for item in descriptors):
                continue
            required = predicate_fact(parents["test"], present_arm == 0)
            consumers = tuple(graph.G.successors(node_id))
            guarded = []
            for consumer in consumers:
                facts = {
                    predicate_fact(controls[control]["predicate_id"], arm == "body")
                    for control, arm in memberships.get(int(consumer), ())
                    if control in controls and controls[control].get("predicate_id") in graph.G
                    and arm in {"body", "orelse"}
                }
                if fallthrough_facts is not None:
                    facts = fallthrough_facts.get(int(consumer), frozenset())
                if required not in facts:
                    raise ValueError(
                        f"optional tuple {node_id} consumer {consumer} has no "
                        "proven presence guard in function "
                        f"{graph.G.graph.get('function_name')!r}"
                    )
                guarded.append(int(consumer))
            if not consumers or any(node_id in values for values in
                                    (graph.G.graph.get("return_slot_values") or {}).values()):
                raise ValueError(f"optional tuple {node_id} escapes without a presence ABI")
            inactive = []
            for descriptor in descriptors:
                value_id = next_process_value_id(graph)
                graph.G.add_node(value_id, type="Constant", op="const", value_id=value_id,
                                 constant=0, parents=[], children=[],
                                 tensor={"shape": (), "dtype": "ptr" if descriptor["shape"] else descriptor["dtype"]},
                                 attributes={"value": 0, "conditional_result_of": int(node_id),
                                             "optional_inactive_payload": True})
                inactive.append(value_id)
            leaves[1 - present_arm] = tuple(inactive)
            optional = {"predicate_id": required[0], "present_when": required[1],
                        "guarded_consumers": tuple(guarded), "inactive_value_ids": tuple(inactive)}
            descriptors = [descriptors, descriptors]
        else:
            if any(arm.get("aggregate_kind") != "tuple" for arm in arms):
                continue
            descriptors = [[_tensor_descriptor(graph, leaf) for leaf in arm] for arm in leaves]
        if not leaves[0] or len(leaves[0]) != len(leaves[1]):
            continue
        if descriptors[0] != descriptors[1] or any(item is None for item in descriptors[0]):
            continue
        bindings = []
        for left, right, descriptor in zip(*leaves, descriptors[0]):
            member = next_process_value_id(graph)
            edges = [(left, "body"), (right, "orelse"), (parents["test"], "test")]
            graph.G.add_node(member, type="Phi", op="Phi", value_id=member,
                             parents=[], children=[], tensor=copy.deepcopy(descriptor),
                             attributes={"conditional_result_of": int(node_id)})
            _set_operands(graph, member, edges, cause=_AGGREGATE_MEMBER)
            _append_operand(graph, int(node_id), member, "elts")
            bindings.append((int(left), int(right), member))
        attributes.update({
            "producer_kind": "aggregate", "aggregate_kind": "tuple",
            "aggregate_leaf_value_ids": tuple(item[2] for item in bindings),
            "tensor_output_descriptors": tuple(copy.deepcopy(descriptors[0])),
            "conditional_member_bindings": tuple(bindings),
        })
        if optional is not None:
            attributes["optional_presence"] = optional
        data["attributes"] = attributes
        data.pop("tensor", None)
        changed = True
    return changed


def _publish_callsite_return_members(
    caller: Any, node_id: int, descriptors: Any, kind: str,
    return_cells: tuple = (),
) -> bool:
    """Materialize or enrich the exact projections of one aggregate call.

    Projection ids are stable identities.  Their descriptors are fixed-point
    facts and may become more precise after those ids are first materialized.
    Existing ids therefore receive the newer fact in place; node existence is
    not evidence that their metadata has settled.
    """

    node_id = int(node_id)
    data = caller.G.nodes[node_id]
    attributes = dict(data.get("attributes") or {})
    descriptors = tuple(copy.deepcopy(tuple(descriptors)))
    incumbent_leaves = tuple(map(
        int, attributes.get("aggregate_leaf_value_ids") or (),
    ))

    from .concordance_declarations import PLANNER_TENSOR_SPECIALIZATION
    from .identity_concordance import current_identity_book

    caller_name = str(caller.G.graph.get("function_name") or "")
    return_cells = tuple(return_cells)

    def record(index: int, leaf_id: int, action: str, descriptor: Any) -> None:
        # DERIVED from the callee return value's cells for this slot (its
        # identity cell and its shape-state cell): a refined member is a
        # changed source; a changed member from an unchanged source is not.
        _post_callsite_return_member(
            caller, node_id, int(index), int(leaf_id), descriptor,
            tuple(return_cells[index]) if index < len(return_cells) else (),
            stage=PLANNER_TENSOR_SPECIALIZATION,
        )

    if (
        len(incumbent_leaves) == len(descriptors)
        and incumbent_leaves
        and all(leaf_id in caller.G for leaf_id in incumbent_leaves)
    ):
        changed = False
        for index, (leaf_id, descriptor) in enumerate(zip(
            incumbent_leaves, descriptors,
        )):
            if not isinstance(descriptor, Mapping):
                continue
            replacement = copy.deepcopy(dict(descriptor))
            leaf = caller.G.nodes[leaf_id]
            if dict(leaf.get("tensor") or {}) == replacement:
                continue
            leaf["tensor"] = replacement
            record(index, leaf_id, "enriched", descriptor)
            _invalidate_tensor_descriptor_dependents(
                caller, (leaf_id,), "aggregate-return-projection-enriched",
            )
            changed = True
        if tuple(attributes.get("tensor_output_descriptors") or ()) != descriptors:
            attributes["tensor_output_descriptors"] = descriptors
            data["attributes"] = attributes
            changed = True
        return changed

    stale_leaves = tuple(
        leaf_id for leaf_id in incumbent_leaves if leaf_id not in caller.G
    )

    def resident_direct_projection(index: int) -> int | None:
        """Return the one authored ``call[index]`` value, when exact.

        Lexical tuple unpacking creates these projections before callsite
        return descriptors settle.  Reusing that identity is essential:
        loop/effect metadata already names it, so minting a parallel leaf
        makes the call publish one value while the effect consumes another.
        """

        candidates = []
        for successor in caller.G.successors(node_id):
            projection = caller.G.nodes[int(successor)]
            if str(
                projection.get("op") or projection.get("type") or ""
            ).casefold() != "indexed":
                continue
            if not any(
                int(parent) == node_id
                and str(role) in {"base", "value", "operand", "object"}
                for parent, role in projection.get("parents") or ()
            ):
                continue
            indices = tuple(
                int(parent)
                for parent, role in projection.get("parents") or ()
                if str(role) == "index" and int(parent) in caller.G
            )
            if len(indices) != 1:
                continue
            try:
                projected_index = _constant_value(
                    caller.G.nodes[indices[0]]
                )
            except KeyError:
                continue
            if (
                isinstance(projected_index, int)
                and not isinstance(projected_index, bool)
                and int(projected_index) == int(index)
            ):
                candidates.append(int(successor))
        unique = tuple(dict.fromkeys(candidates))
        return unique[0] if len(unique) == 1 else None

    # Materialize exact member projections of the authored return. These are
    # call results, never extra caller inputs or copies of the payload.
    leaves = []
    for index, descriptor in enumerate(descriptors):
        resident_projection = resident_direct_projection(index)
        if resident_projection is not None:
            member = caller.G.nodes[int(resident_projection)]
            member.setdefault("attributes", {})[
                "authored_call_result_projection"
            ] = True
            member["tensor"] = (
                {} if descriptor is None or isinstance(descriptor, tuple)
                else copy.deepcopy(dict(descriptor))
            )
            leaves.append(int(resident_projection))
            record(index, int(resident_projection), "reused", descriptor)
            continue
        index_id = next_process_value_id(caller)
        caller.G.add_node(index_id, type="Constant", op="const",
                          value_id=index_id, constant=index,
                          attributes={"value": index}, parents=[], children=[])
        member_id = next_process_value_id(caller)
        caller.G.add_node(
            member_id, type="Indexed", op="Indexed", value_id=member_id,
            parents=[], children=[],
            attributes={"authored_call_result_projection": True},
            # A member that is itself an aggregate has no tensor fact of its
            # own; its structure stays in the ordered output descriptors,
            # where the nested projection path reads it.
            tensor=(
                {} if descriptor is None or isinstance(descriptor, tuple)
                else copy.deepcopy(dict(descriptor))
            ),
        )
        _set_operands(
            caller, member_id, [(node_id, "base"), (index_id, "index")],
            cause=_AGGREGATE_MEMBER,
        )
        leaves.append(member_id)
        record(index, member_id, "materialized", descriptor)
    attributes.update({
        "authored_return_container": kind,
        "producer_kind": "aggregate", "aggregate_kind": kind,
        "aggregate_leaf_value_ids": tuple(leaves),
        "tensor_output_descriptors": descriptors,
    })
    if stale_leaves:
        # The old leaf ids are not merely descriptor cache entries.  Source
        # control can name them from cached loop/effect receipts even after a
        # specialized call republishes the same return slots under fresh
        # resident ids.  Concord the exact slot transition, then move every
        # cached occurrence through that recorded identity before exposing
        # the replacement ledger.  Otherwise ``append(result_record)`` can
        # retain the dead pre-specialization projection while the call itself
        # publishes the new one.
        projection_identity = current_identity_book().page(
            "callsite_projection_identity_concordance"
        )
        projection_scope = (
            caller_name,
            int(node_id),
            id(caller.G),
        )
        for index, (stale_id, replacement_id) in enumerate(zip(
            incumbent_leaves, leaves,
        )):
            if int(stale_id) == int(replacement_id):
                continue
            row = (projection_scope, int(index), int(stale_id))
            fact = int(replacement_id)
            incumbent = projection_identity.latest(row)
            if incumbent is not None and int(incumbent) != fact:
                raise ValueError(
                    "callsite projection identity concordance disagreement "
                    f"for {row!r}: recorded={incumbent!r}, "
                    f"replacement={fact!r}"
                )
            if incumbent is None:
                projection_identity.set(row, 0, fact)
            _retarget_all_cached_value_ids(
                caller, int(stale_id),
                int(projection_identity.latest(row)),
            )
            if os.environ.get("TURING_DEBUG_SEQUENCE_EFFECT_IDENTITY"):
                print(
                    "DEBUG-CALLSITE-PROJECTION-IDENTITY "
                    f"function={caller_name!r} callsite={int(node_id)} "
                    f"slot={int(index)} stale={int(stale_id)} "
                    f"resident={int(replacement_id)}",
                    file=sys.stderr,
                    flush=True,
                )
        attributes["aggregate_leaf_republication"] = {
            "replaced_missing_value_ids": stale_leaves,
            "replacement_value_ids": tuple(map(int, leaves)),
            "rule": "exact-stored-output-descriptors",
            "tie_policy": "incumbent",
        }
    data["attributes"] = attributes
    data.pop("tensor", None)
    return True


#: Per compile book: the (callee graph, signature) answers a callsite-
#: specialized copy's propagation has computed, and the callees whose answer
#: is being computed now.
_CHILD_SIGNATURE_RESULTS: dict[int, tuple[Any, dict]] = {}
_CHILD_CALLS_IN_PROGRESS: set[tuple] = set()


def _child_signature_results(book: Any) -> dict:
    held = _CHILD_SIGNATURE_RESULTS.get(id(book))
    if held is None or held[0] is not book:
        _CHILD_SIGNATURE_RESULTS.clear()
        held = _CHILD_SIGNATURE_RESULTS[id(book)] = (book, {})
    return held[1]


def _propagate_callsite_tensor_specializations(
    graph: Any,
    *,
    runtime_references: frozenset[int] | None = None,
    publish_to_function_table: bool = True,
    _progress: Any = None,
) -> None:
    """Carry consistent tensor descriptors through the function table.

    This is the descriptor analogue of literal planner specialization.  It
    settles to a fixed point so a root call can shape ``bw_matmul`` and that
    shaped wrapper can in turn shape ``matmul_vjp`` before either function's
    logical control is partitioned. Conflicting callsite shapes remain
    parametric and are left for per-callsite specialization.
    """

    function_table = getattr(graph, "function_table", None)
    if function_table is None:
        return
    # A source-catalogue graph owns the table but is not itself an executable
    # caller.  Its per-definition graphs contain the actual parameter ABI,
    # call edges, and numerical dependencies.  Including the catalogue here
    # both duplicates all callsites and lets tuple/return publication rewrite
    # a map container which the structural-fold stage explicitly excludes.
    graphs = (
        [graph] if graph.G.graph.get("function_name") else []
    )
    graphs.extend(
        entry.graph
        for entry in function_table
        if entry.graph is not None
        # A root that is itself an executable function is registered in its
        # own table: it is already ``graph`` above, and a second visit of
        # the same callsite in one round would post its decision row twice.
        and entry.graph is not graph
        and (
            runtime_references is None
            or int(entry.reference.address) in runtime_references
        )
    )

    #: (caller graph, call node) -> per-slot callee return value cells of
    #: the latest ``call_result_descriptor`` answer for that call.
    return_cells_by_call: dict[tuple[int, int], tuple] = {}

    from .concordance_declarations import (
        CALLSITE_DESCRIPTOR_REUSE, COMPILE_POLICY, POLICY_DECLARATION,
        PLANNER_TENSOR_SPECIALIZATION,
    )
    from .identity_concordance import (
        Derived, Mode, Novel, current_identity_book, fork_shape_scope,
        shape_scope_of,
    )
    from .work_contract import active_contract

    # The work contract's callsite_descriptor_reuse policy, declared on the
    # book as a root row before the first decision reads it: a different
    # policy arriving mid-compile is a disagreement the book refuses.
    reuse_mode = str(active_contract().callsite_descriptor_reuse)
    _book = current_identity_book()
    _book.post(
        COMPILE_POLICY, ("callsite_descriptor_reuse",), reuse_mode,
        stage=PLANNER_TENSOR_SPECIALIZATION,
        provenance=Novel(POLICY_DECLARATION, ()), mode=Mode.CONCORD,
    )
    reuse_policy_cell = _book.latest_ref(
        COMPILE_POLICY, ("callsite_descriptor_reuse",),
    )
    #: This round's (callee graph id, signature) -> (callee graph, output
    #: descriptors, output cells, the computing callsite's decision cell).
    #: Cleared at every round: the callee graphs change between rounds.
    #: A callsite-specialized copy's call edges (``publish_to_function_table``
    #: False) reach callees whose own call edges are not settled on the
    #: function table either, so each propagation copy settles its own
    #: (below); that nested answer is a function of the same inputs, so the
    #: memo outlives the round and the invocation, per book.
    signature_results: dict[tuple, tuple] = (
        {} if publish_to_function_table
        else _child_signature_results(_book)
    )

    def call_result_descriptor(
        caller: Any,
        node_id: int,
        callee: Any,
    ) -> tuple[dict[str, Any] | None, ...]:
        """Infer the ordered tensor returns of an exact callsite."""

        receiver, positional, _all = _method_parameter_layout(callee.G)
        descriptors: dict[str, dict[str, Any]] = {}
        aggregate_descriptors: dict[str, tuple[Any, ...]] = {}
        specializations: dict[str, Any] = {}
        specialization_cells: dict[str, tuple] = {}
        bound_parameters: set[str] = set()
        for parent, role_value in _expanded_callsite_argument_edges(
            caller, int(node_id),
        ):
            role = str(role_value)
            parameter = _callsite_parameter_name(
                role, receiver, positional,
            )
            if parameter is None:
                continue
            parameter = str(parameter)
            bound_parameters.add(parameter)
            descriptor = _tensor_descriptor(caller, int(parent))
            if descriptor is not None:
                descriptors[parameter] = descriptor
            static_argument = _source_static_value(caller, int(parent))
            leaves = _authored_aggregate_leaves(caller, int(parent))
            if leaves and not static_argument and all(
                int(leaf) in caller.G for leaf in leaves
            ):
                # Same member-identity rule as callsite specialization: the
                # callee's return descriptors may depend on aggregate members
                # (``previous_hub, ... = history``), so the propagation copy
                # must see those members bound, not an opaque formal.
                aggregate_descriptors[parameter] = tuple(
                    _authored_aggregate_descriptor_tree(caller, int(leaf))
                    for leaf in leaves
                )
            if static_argument:
                try:
                    specializations[parameter] = _source_static_literal(
                        caller, int(parent)
                    )
                except ValueError:
                    pass
                else:
                    specialization_cells[parameter] = (
                        _argument_identity_cells(caller, int(parent))
                    )
        for parameter, default in (
            callee.G.graph.get("parameter_defaults") or {}
        ).items():
            if str(parameter) not in bound_parameters:
                specializations[str(parameter)] = copy.deepcopy(default)
        if not descriptors and not aggregate_descriptors and not callee.G.graph.get("parameter_record_abi"):
            return ()

        def any_descriptor(item: Any) -> bool:
            if isinstance(item, tuple):
                return any(any_descriptor(member) for member in item)
            return item is not None

        def record_and_return(
            output_descriptors: Any, output_cells: Any,
        ) -> tuple[dict[str, Any] | None, ...]:
            # Record this round's decision in the concordance.  The
            # specialization fixed point below has no bound of its own: it
            # repeats until nothing changes.  Writing each round's published
            # return shape to its own page means
            # ``IdentityPage.oscillating_rows`` can name a row that leaves a
            # shape and comes back to it -- a round trip, which a settling
            # fixed point never makes -- instead of the whole compile merely
            # looking slow from outside.
            _page = _book.page("callsite_return_specialization")
            _row = (
                shape_scope_of(caller), shape_scope_of(callee), int(node_id),
            )
            _page.set(_row, len(_page.history(_row)), tuple(
                None if item is None or not isinstance(item, Mapping)
                else (tuple(item.get("shape") or ()), str(item.get("dtype") or ""))
                for item in output_descriptors
            ))
            return_cells_by_call[(id(caller.G), int(node_id))] = tuple(
                output_cells
            )
            return (
                tuple(copy.deepcopy(output_descriptors))
                if any(any_descriptor(item) for item in output_descriptors)
                else ()
            )

        # The callee copy below is a function of exactly these inputs: the
        # callee graph, the argument descriptors, the aggregate member
        # descriptors, the literal specializations and the callee's declared
        # record ABI.  Under ``callsite_descriptor_reuse="signature"`` (the
        # work contract) the first callsite of a round with this signature
        # derives the copy and every later one in the same round reads its
        # answer: the same descriptors, and the same callee return cells for
        # the member publication to derive from.  ``"callsite"`` derives a
        # copy per callsite, as every round did before.  Either way the
        # decision is a row: ``callsite_descriptor_reuse`` (caller, callee,
        # call, round) = ("computed"|"reused", signature digest), DERIVED
        # from the policy cell (and, for a reuse, the computing callsite's
        # row).  A generator callee publishes its yield rows on the caller's
        # call value while deriving, so it is always derived per callsite.
        signature = (
            _stable_signature_value(descriptors),
            _stable_signature_value(aggregate_descriptors),
            _stable_signature_value(specializations),
            _stable_signature_value(
                callee.G.graph.get("parameter_record_abi") or {}
            ),
        )
        reuse_key = (id(callee.G), signature)
        signature_digest = hashlib.sha256(
            repr(signature).encode("utf-8")
        ).hexdigest()[:12]
        reuse_row = (
            shape_scope_of(caller), shape_scope_of(callee),
            int(node_id), int(round_index),
        )
        held = (
            signature_results.get(reuse_key)
            if reuse_mode == "signature"
            and not callee.G.graph.get("generator_stream")
            else None
        )
        if held is not None and held[0] is callee.G:
            _book.post(
                CALLSITE_DESCRIPTOR_REUSE, reuse_row,
                ("reused", signature_digest),
                stage=PLANNER_TENSOR_SPECIALIZATION,
                provenance=Derived((reuse_policy_cell, held[3])),
                mode=Mode.CONCORD,
            )
            return record_and_return(copy.deepcopy(held[1]), tuple(held[2]))
        specialized = extract_clean_process_subgraph(callee, callee.G)
        # A propagation copy states the shapes of ITS callsite: its own shape
        # scope, never the callee's (or another callsite's).
        fork_shape_scope(specialized, "callsite_tensor_specialization")
        specialized.G.graph["planner_specializations"] = copy.deepcopy(
            specializations
        )
        # This propagation copy's own rows, under its forked scope (7.1).
        _post_copy_planner_specializations(
            specialized, specializations, specialization_cells,
        )
        specialized.G.graph["planner_tensor_descriptors"] = copy.deepcopy(
            descriptors
        )
        _apply_callsite_tensor_descriptors(specialized, descriptors)
        _apply_callsite_aggregate_descriptors(
            specialized, aggregate_descriptors,
        )
        if not _expand_specialized_unbroadcast_identity(specialized):
            _fold_callsite_structural_values(specialized)
        if not publish_to_function_table and reuse_key not in (
            _CHILD_CALLS_IN_PROGRESS
        ):
            # The copy's own call edges are settled the same way (a
            # recursive callee is entered once: its re-entry answers
            # nothing, never loops).
            _CHILD_CALLS_IN_PROGRESS.add(reuse_key)
            try:
                _propagate_callsite_tensor_specializations(
                    specialized,
                    runtime_references=frozenset(),
                    publish_to_function_table=False,
                )
            finally:
                _CHILD_CALLS_IN_PROGRESS.discard(reuse_key)
        output_descriptors: list[dict[str, Any] | None] = []
        # Per published slot, the callee return value's cells: its identity
        # cell and its shape-state cell, read after the descriptor query
        # that (re)states it.  The member publication derives from these.
        output_cells: list[tuple] = []

        def return_value_cells(value_id: int) -> tuple:
            from ..common.tensors.topological_reducer import node_identity_cell
            from .identity_concordance import (
                SHAPE_STATE_PAGE, current_identity_book, shape_scope_of,
            )

            cells = []
            try:
                cells.append(node_identity_cell(specialized, int(value_id)))
            except ValueError:
                pass
            state = current_identity_book().latest_ref(SHAPE_STATE_PAGE, (
                shape_scope_of(specialized),
                int(specialized.G.nodes[int(value_id)].get(
                    "value_id", value_id,
                )),
            ))
            if state is not None:
                cells.append(state)
            return tuple(cells)
        identities = specialized.G.graph.get("identity_table") or {}
        for output_name in specialized.G.graph.get("function_outputs") or ():
            output_ids = tuple(identities.get(str(output_name), ()))
            value_id = next((
                int(candidate)
                for candidate in reversed(output_ids)
                if int(candidate) in specialized.G
            ), None)
            if value_id is None:
                output_descriptors.append(None)
                output_cells.append(())
                continue
            nested = (
                specialized.G.nodes[value_id].get("attributes") or {}
            ).get("tensor_output_descriptors")
            if nested:
                output_descriptors.extend(
                    copy.deepcopy(tuple(nested))
                )
                nested_cells = return_value_cells(value_id)
                output_cells.extend(nested_cells for _item in tuple(nested))
            else:
                described_id = value_id
                descriptor = _structured_output_descriptor(
                    specialized, value_id,
                )
                if not descriptor_states_a_shape(descriptor):
                    # The identity table lists every value that IS this
                    # binding; they are one storage by construction.  The last
                    # resident one is what the return names, but a store or a
                    # loop-carried port among them can answer the descriptor
                    # query with nothing at all.  Read the shape from whichever
                    # identity states it rather than publish a factless return
                    # that every caller installs as a rank-0 scalar.
                    for candidate in reversed(output_ids):
                        if int(candidate) == value_id:
                            continue
                        alternative = _structured_output_descriptor(
                            specialized, int(candidate),
                        )
                        if isinstance(alternative, Mapping) and (
                            descriptor_states_a_shape(alternative)
                        ):
                            descriptor = alternative
                            described_id = int(candidate)
                            break
                output_descriptors.append(descriptor)
                output_cells.append(return_value_cells(described_id))

        # A generator returns no value; its product is the row it yields.
        # Describe every yield site's tuple members on this exact callsite
        # specialization, require the sites to agree per column, and publish
        # the row on the caller's call value through the concordance, so a
        # ``for a, b, c in gen(...)`` target reads its column in the graph
        # phase instead of acquiring a shape only after SSA linking.
        generator_stream = (
            specialized.G.graph.get("generator_stream")
            or callee.G.graph.get("generator_stream")
            or {}
        )
        yield_rows: list[tuple[dict[str, Any] | None, ...]] = []
        for yield_node in generator_stream.get("yield_nodes") or ():
            if int(yield_node) not in specialized.G or not isinstance(
                specialized.G.nodes[int(yield_node)].get("expr_obj"),
                ast.Yield,
            ):
                continue
            yielded = next((
                int(parent)
                for parent, role in (
                    specialized.G.nodes[int(yield_node)].get("parents") or ()
                )
                if str(role) in {"value", "operand"}
                and int(parent) in specialized.G
            ), None)
            if yielded is None:
                continue
            yielded_data = specialized.G.nodes[yielded]
            members = (
                tuple(
                    int(parent)
                    for parent, _role in yielded_data.get("parents") or ()
                    if int(parent) in specialized.G
                )
                if isinstance(yielded_data.get("expr_obj"), ast.Tuple)
                else (yielded,)
            )
            yield_rows.append(tuple(
                _tensor_descriptor(specialized, member) for member in members
            ))
        if yield_rows and len({len(row) for row in yield_rows}) == 1:
            from .identity_concordance import record_shape_transformation

            caller_scope = shape_scope_of(caller)
            callee_scope = shape_scope_of(specialized)
            call_value = int(
                caller.G.nodes[int(node_id)].get("value_id", node_id)
            )
            for column, members in enumerate(zip(*yield_rows)):
                # Only a column every yield site states is a fact; a record
                # handle or a not-yet-described member claims nothing (an
                # empty shape would read as a proven scalar downstream).
                if not all(
                    isinstance(member, Mapping)
                    and descriptor_states_a_shape(member)
                    for member in members
                ):
                    continue
                facts = {
                    (
                        tuple(map(int, member.get("shape") or ())),
                        str(member.get("dtype") or "unknown"),
                    )
                    for member in members
                }
                if len(facts) != 1:
                    continue
                shape, dtype = next(iter(facts))
                state = {"shape": shape, "dtype": dtype, "rank": len(shape)}
                record_shape_transformation(
                    callee_scope, ("yield", column),
                    caller_scope, ("yield_row", call_value, column),
                    stage="callsite_yield_observation",
                    operation="yield_row",
                    source_state=state,
                    target_state=state,
                    role=f"column:{column}",
                )

        computing_cell = _book.post(
            CALLSITE_DESCRIPTOR_REUSE, reuse_row,
            ("computed", signature_digest),
            stage=PLANNER_TENSOR_SPECIALIZATION,
            provenance=Derived((reuse_policy_cell,)),
            mode=Mode.CONCORD,
        )
        if reuse_mode == "signature" and not callee.G.graph.get(
            "generator_stream"
        ):
            signature_results[reuse_key] = (
                callee.G, copy.deepcopy(output_descriptors),
                tuple(output_cells), computing_cell,
            )
        return record_and_return(output_descriptors, output_cells)

    round_index = 0
    seen_round_states: dict[str, int] = {}

    def specialization_state() -> tuple[str, int]:
        facts = []
        tensor_fact_count = 0
        for item in graphs:
            function_name = str(item.G.graph.get("function_name") or "")
            planner = item.G.graph.get("planner_tensor_descriptors") or {}
            facts.append((function_name, "planner", repr(planner)))
            for value_id, node in item.G.nodes(data=True):
                tensor = node.get("tensor")
                if tensor:
                    tensor_fact_count += 1
                    facts.append((
                        function_name,
                        int(value_id),
                        repr(_callsite_descriptor_receipt(tensor)),
                    ))
        digest = hashlib.sha256(repr(tuple(facts)).encode("utf-8")).hexdigest()
        return digest[:16], tensor_fact_count

    def _planner_progress(message: str) -> None:
        if _progress is not None:
            _progress({
                "specialization_state": "fixed-point-guard",
                "message": message,
            })

    from .bounded_fixed_point import BoundedFixedPoint, bound_for

    specialization_guard = BoundedFixedPoint(
        "callsite-specialization",
        bound_for(
            "planner",
            nodes=sum(item.G.number_of_nodes() for item in graphs),
            callsites=sum(
                1
                for item in graphs
                for _node_id, data in item.G.nodes(data=True)
                if (data.get("attributes") or {}).get("callee_ref")
                is not None
                or (data.get("attributes") or {}).get("method_ref")
                is not None
            ),
        ),
        scope=("whole-program",), progress=_planner_progress,
        state=specialization_state()[0],
    )
    changed = True
    while changed:
        round_index += 1
        # A round's reuse table is that round's: the callee graphs it was
        # derived on are mutated by the publications below.  (A specialized
        # copy's memo is not: it publishes nowhere.)
        if publish_to_function_table:
            signature_results.clear()
        round_started = time.monotonic()
        callsites = 0
        mutation_counts = {
            "conditional_members": 0,
            "return_members": 0,
            "single_returns": 0,
            "structured_returns": 0,
            "formal_shapes": 0,
            "callee_descriptors": 0,
            "settled_graph_shapes": 0,
        }
        if _progress is not None:
            _progress({
                "specialization_state": "round-begin",
                "round": round_index,
                "graphs": len(graphs),
                "graph_nodes": sum(item.G.number_of_nodes() for item in graphs),
            })
        changed = False
        # The outer catalogue graph owns the function table, but its entry
        # graphs own the parameter ABI and numerical dependencies.  Settle
        # each caller locally before reading its call arguments; otherwise a
        # shaped ``material_state[:, :, 3]`` still looks scalar here and only
        # a later backend happens to rediscover its extent after the call ABI
        # has already been frozen.
        for caller_index, caller in enumerate(graphs):
            if caller.G.graph.get("function_name"):
                fold_started = time.monotonic()
                if _progress is not None:
                    _progress({
                        "specialization_state": "prefold-begin",
                        "round": round_index,
                        "caller_index": int(caller_index),
                        "caller": str(caller.G.graph.get("function_name")),
                        "caller_nodes": caller.G.number_of_nodes(),
                    })
                _fold_callsite_structural_values(
                    caller,
                    _progress=(
                        None if _progress is None else
                        lambda facts, caller=caller: _progress({
                            "specialization_state": "prefold-fixed-point",
                            "round": round_index,
                            "caller": str(
                                caller.G.graph.get("function_name") or ""
                            ),
                            **facts,
                        })
                    ),
                )
                if _progress is not None:
                    _progress({
                        "specialization_state": "prefold-end",
                        "round": round_index,
                        "caller_index": int(caller_index),
                        "caller": str(caller.G.graph.get("function_name")),
                        "caller_nodes": caller.G.number_of_nodes(),
                        "duration_seconds": round(
                            time.monotonic() - fold_started, 6
                        ),
                    })
        candidates: dict[tuple[int, str], list[dict[str, Any]]] = {}
        for caller in graphs:
            conditional_changed = _publish_conditional_tuple_members(caller)
            changed |= conditional_changed
            mutation_counts["conditional_members"] += int(conditional_changed)
            for _node_id, data in tuple(caller.G.nodes(data=True)):
                attributes = data.get("attributes") or {}
                reference = attributes.get("callee_ref")
                if reference is None:
                    reference = attributes.get("method_ref")
                if reference is None:
                    continue
                try:
                    callee = function_table.entry(int(reference)).graph
                except (KeyError, TypeError, ValueError):
                    continue
                if callee is None:
                    continue
                callsites += 1
                if _progress is not None and (
                    callsites == 1 or callsites % 16 == 0
                ):
                    _progress({
                        "specialization_state": "callsite-progress",
                        "round": round_index,
                        "callsites": callsites,
                        "caller": str(
                            caller.G.graph.get("function_name") or ""
                        ),
                        "callee": str(
                            callee.G.graph.get("function_name") or reference
                        ),
                        "round_elapsed_seconds": round(
                            time.monotonic() - round_started, 6
                        ),
                    })
                # A call node can carry an early catalogue descriptor that is
                # no longer exact after its operands are specialized.  Calls
                # are equality edges, so recompute their return contract on
                # every fixed-point round and replace only when the exact
                # callee result differs.  The decision itself is recorded by
                # ``call_result_descriptor`` in the concordance.
                result_descriptors = call_result_descriptor(
                    caller, int(_node_id), callee
                )
                if result_descriptors:
                    if (
                        len(result_descriptors) == 1
                        and isinstance(result_descriptors[0], Mapping)
                    ):
                        exact_result = result_descriptors[0]
                        exact_shape = tuple(
                            exact_result.get("shape") or ()
                        )
                        if exact_shape:
                            from .identity_concordance import (
                                record_proven_shape,
                                record_shape_transformation,
                                shape_scope_of,
                            )

                            # Publish on every fixed-point observation, not
                            # only when the graph-node dictionary changes.
                            # Another callsite later in the same round may
                            # invalidate a dependent proof while leaving this
                            # exact call descriptor textually unchanged; the
                            # final stable round must restore the call-edge
                            # fact to the concordance.
                            record_proven_shape(
                                shape_scope_of(caller),
                                int(data.get("value_id", _node_id)),
                                exact_shape,
                                exact_result.get("dtype"),
                                int(_dependency_levels(caller).get(
                                    int(_node_id), 0,
                                )),
                            )
                            # The restoration is itself a morph of the call
                            # value: record it as the callee-return edge so
                            # anything derived from the call value while its
                            # proof was withdrawn is superseded along the
                            # transformation graph instead of keeping the
                            # gap's answer.
                            record_shape_transformation(
                                shape_scope_of(callee),
                                ("return",),
                                shape_scope_of(caller),
                                int(data.get("value_id", _node_id)),
                                stage="callsite_return_observation",
                                operation="call_result",
                                source_state=exact_result,
                                target_state=exact_result,
                                role="return",
                            )
                    return_kinds = set((callee.G.graph.get("return_container_kinds") or {}).values())
                    if (
                        len(return_kinds) == 1
                        and return_kinds <= {"tuple", "list"}
                        and result_descriptors
                        # An authored return may mix tensors with nested
                        # aggregates -- the vehicle tick returns fifteen
                        # tensors and one four-member history tuple.  Refusing
                        # the whole publication because one member is nested
                        # left every sibling projection unresolved, so each
                        # ``result[k]`` became a formal no caller could fill.
                        and all(item is None
                                or isinstance(item, (Mapping, tuple))
                                for item in result_descriptors)
                    ):
                        return_members_changed = _publish_callsite_return_members(
                            caller, int(_node_id), result_descriptors,
                            next(iter(return_kinds)),
                            return_cells=return_cells_by_call.get(
                                (id(caller.G), int(_node_id)), (),
                            ),
                        )
                        changed |= return_members_changed
                        mutation_counts["return_members"] += int(
                            return_members_changed
                        )
                    elif len(result_descriptors) == 1 and isinstance(
                        result_descriptors[0], Mapping
                    ):
                        # A single returned aggregate still has a nested
                        # descriptor tree. Only a tensor leaf belongs in
                        # the node's tensor slot; preserve aggregate returns
                        # in the structured output ledger below.
                        replacement = copy.deepcopy(dict(
                            result_descriptors[0]
                        ))
                        if dict(data.get("tensor") or {}) != replacement:
                            previous = copy.deepcopy(data.get("tensor"))
                            data["tensor"] = replacement
                            _invalidate_tensor_descriptor_dependents(
                                caller, (int(_node_id),),
                                "call-result-specialization-changed",
                            )
                            from .identity_concordance import (
                                current_identity_book,
                                record_shape_transformation,
                                record_proven_shape,
                                shape_scope_of,
                            )

                            # The call edge has just replaced a provisional
                            # descriptor with the exact specialized return.
                            # Invalidation withdraws facts derived from the
                            # old result; publish the replacement on the call
                            # value itself in the same causal operation.  A
                            # retained comprehension may use that exact value
                            # as its appended row, and sequence physicalization
                            # consults the shared proof rather than a private
                            # graph-node cache.
                            replacement_shape = tuple(
                                replacement.get("shape") or ()
                            )
                            if replacement_shape:
                                record_proven_shape(
                                    shape_scope_of(caller),
                                    int(data.get("value_id", _node_id)),
                                    replacement_shape,
                                    replacement.get("dtype"),
                                    int(_dependency_levels(caller).get(
                                        int(_node_id), 0,
                                    )),
                                )
                            record_shape_transformation(
                                shape_scope_of(callee),
                                ("return",),
                                shape_scope_of(caller),
                                int(data.get("value_id", _node_id)),
                                stage="callsite_return_specialization",
                                operation="call_result",
                                source_state=replacement,
                                target_state=replacement,
                                role="return",
                            )

                            mutation_page = current_identity_book().page(
                                "callsite_tensor_result_specialization"
                            )
                            mutation_row = (
                                shape_scope_of(caller), int(_node_id),
                            )
                            mutation_page.set(
                                mutation_row,
                                len(mutation_page.history(mutation_row)),
                                (
                                    _callsite_descriptor_receipt(previous),
                                    _callsite_descriptor_receipt(replacement),
                                ),
                            )
                            # Invalidation above withdrew every descriptor
                            # derived from the old call result.  Re-evaluate
                            # those exact transformation descendants only
                            # after installing the replacement, so a
                            # comprehension materializer reads the new row
                            # shape rather than immediately republishing the
                            # provisional scalar that was just retired.
                            def descriptor_dependents(source_id: int):
                                return tuple(
                                    int(candidate_id)
                                    for candidate_id, candidate
                                    in caller.G.nodes(data=True)
                                    if any(
                                        int(parent) == int(source_id)
                                        and str(role) not in {
                                            "callee", "func", "definition",
                                        }
                                        for parent, role in candidate.get(
                                            "parents", ()
                                        )
                                    )
                                )

                            pending_descriptor_nodes = list(
                                descriptor_dependents(int(_node_id))
                            )
                            seen_descriptor_nodes = {int(_node_id)}
                            while pending_descriptor_nodes:
                                dependent_id = int(
                                    pending_descriptor_nodes.pop(0)
                                )
                                if dependent_id in seen_descriptor_nodes:
                                    continue
                                seen_descriptor_nodes.add(dependent_id)
                                _tensor_descriptor(caller, dependent_id)
                                pending_descriptor_nodes.extend(
                                    descriptor_dependents(dependent_id)
                                )
                            changed = True
                            mutation_counts["single_returns"] += 1
                    elif result_descriptors:
                        attributes = dict(data.get("attributes") or {})
                        if attributes.get(
                            "tensor_output_descriptors"
                        ) != result_descriptors:
                            attributes["tensor_output_descriptors"] = (
                                result_descriptors
                            )
                            data["attributes"] = attributes
                            changed = True
                            mutation_counts["structured_returns"] += 1
                receiver, positional, _all = _method_parameter_layout(callee.G)
                for parent, role_value in _expanded_callsite_argument_edges(
                    caller, int(_node_id),
                ):
                    role = str(role_value)
                    parameter = _callsite_parameter_name(
                        role, receiver, positional,
                    )
                    descriptor = _tensor_descriptor(caller, int(parent))
                    if parameter is not None and descriptor is not None:
                        # This fixed point is the first stage that knows the
                        # exact caller value and the callee formal together.
                        # Publish that agreement now, under the authored
                        # identity, so recursive return specialization can
                        # recover the formal shape from every clean or hashed
                        # graph copy on its next round.  Keeping the fact only
                        # in ``planner_tensor_descriptors`` made it local to
                        # one catalogue graph; a fresh specialization then
                        # reverted the same formal to an unknown scalar.
                        # A rank-0 descriptor with a known dtype is a
                        # stated shape (``descriptor_states_a_shape``), not
                        # "recovery stopped".  Publishing only non-empty
                        # extents hid the disagreement between unbroadcast's
                        # G = () (bw_add) and G = (2, 3) (bw_mul) callsites:
                        # no FORMAL_SHAPE_CONFLICT, so every expansion copy
                        # read the shared ('unbroadcast', 1) row and the
                        # (2, 3) reshape answered (), which spun the
                        # return-member fixed point forever.
                        if publish_to_function_table and (
                            descriptor_states_a_shape(descriptor)
                        ):
                            formal_shape_changed = _publish_formal_shape(
                                str(callee.G.graph.get("function_name")),
                                str(parameter),
                                descriptor,
                                str(caller.G.graph.get("function_name")),
                            )
                            changed |= formal_shape_changed
                            mutation_counts["formal_shapes"] += int(
                                formal_shape_changed
                            )
                        candidates.setdefault(
                            (int(reference), str(parameter)), []
                        ).append(copy.deepcopy(dict(descriptor)))
                # A nested function reads bindings its caller owns, and those
                # captures are ordinary values of this exact caller.  The
                # callee had no contract for them, so every shape expression
                # over a capture -- ``wrench_k.shape[:2]`` inside
                # ``_wrench_force`` -- stayed unresolved and escaped as an
                # anonymous formal.  Offer the enclosing value's descriptor
                # under the capture's own name; the agreement check below
                # applies to it exactly as it does to a bound argument, so a
                # shared catalogue graph is only specialized when every
                # callsite names the same contract.
                caller_identities = caller.G.graph.get("identity_table") or {}
                for _capture_id, capture in callee.G.nodes(data=True):
                    capture_attributes = capture.get("attributes") or {}
                    if str(capture.get("type") or "") != "Input":
                        continue
                    if str(
                        capture_attributes.get("binding_kind") or ""
                    ) not in {"closure", "external"}:
                        continue
                    capture_name = str(
                        capture_attributes.get("binding_name") or ""
                    )
                    if not capture_name:
                        continue
                    capture_values = tuple(dict.fromkeys(
                        int(value_id)
                        for value_id in caller_identities.get(capture_name, ())
                        if int(value_id) in caller.G
                    ))
                    capture_descriptor = (
                        None if len(capture_values) != 1
                        else _tensor_descriptor(caller, capture_values[0])
                    )
                    if os.environ.get("TURING_DEBUG_CAPTURE_DESCRIPTOR"):
                        print(
                            "DEBUG-CAPTURE-DESCRIPTOR "
                            f"caller={caller.G.graph.get('function_name')!r} "
                            f"callee_ref={int(reference)} name={capture_name!r} "
                            f"values={capture_values!r} "
                            f"descriptor={capture_descriptor!r}",
                            file=sys.stderr, flush=True,
                        )
                    if capture_descriptor is None:
                        continue
                    candidates.setdefault(
                        (int(reference), capture_name), []
                    ).append(copy.deepcopy(dict(capture_descriptor)))

        # Calls are processed in source-graph order, but that order must not
        # decide which descendant proof survives.  Replacing one exact call
        # result invalidates every shape downstream from that identity.  A
        # later call in the same round can therefore withdraw a proof that an
        # earlier call already republished.  Once all call edges for the round
        # are known, walk each transformation graph in dependency order and
        # let the one concordance re-prove every derivable value.  This is a
        # settlement pass over the graph itself, not a parallel descriptor
        # cache; a newly recovered proof requests another callsite round so
        # callers and callees can observe it symmetrically.
        from .identity_concordance import (
            current_identity_book,
            proven_shape_contract_of,
            shape_scope_of,
        )

        settlement_page = current_identity_book().page(
            "tensor_shape_settlement_concordance"
        )

        for caller in graphs:
            function_name = shape_scope_of(caller)
            for value_id in _dependency_order(caller):
                data = caller.G.nodes[int(value_id)]
                semantic_id = int(data.get("value_id", value_id))
                before = proven_shape_contract_of(
                    function_name, semantic_id,
                )
                _tensor_descriptor(caller, int(value_id))
                after = proven_shape_contract_of(
                    function_name, semantic_id,
                )
                if before is None and after is not None:
                    # A later callsite in this round may invalidate a proof
                    # that this graph walk then restores. Restoration is
                    # necessary, but after the first settled receipt it is
                    # not a new fixed-point mutation. Keep that distinction
                    # on the one concordance: a local ``seen`` cache would
                    # fork identity authority and made identical graph state
                    # spin forever with ``changed=True``.
                    settlement_row = (function_name, int(semantic_id))
                    settlement_fact = (
                        tuple(map(int, after[0])), str(after[1]),
                    )
                    incumbent = settlement_page.latest(settlement_row)
                    if incumbent is None:
                        settlement_page.concord(
                            settlement_row, settlement_fact,
                        )
                        changed = True
                        mutation_counts["settled_graph_shapes"] += 1
                    elif tuple(incumbent) != settlement_fact:
                        raise ValueError(
                            "tensor shape settlement disagreement for "
                            f"{settlement_row!r}: recorded={incumbent!r}, "
                            f"proposed={settlement_fact!r}"
                        )
        by_reference: dict[int, dict[str, dict[str, Any]]] = {}
        for (reference, parameter), descriptors in candidates.items():
            if not descriptors or any(
                descriptor != descriptors[0] for descriptor in descriptors[1:]
            ):
                continue
            by_reference.setdefault(reference, {})[parameter] = descriptors[0]
        if not publish_to_function_table:
            # A callsite-specialized copy settles only its own call edges;
            # the shared function-table graphs belong to every callsite and
            # are not narrowed to one copy's arguments.
            by_reference = {}
        for reference, descriptors in by_reference.items():
            callee = function_table.entry(reference).graph
            if callee is None:
                continue
            existing = callee.G.graph.setdefault(
                "planner_tensor_descriptors", {}
            )
            additions = {
                name: descriptor
                for name, descriptor in descriptors.items()
                if existing.get(name) != descriptor
            }
            if not additions:
                continue
            existing.update(copy.deepcopy(additions))
            _apply_callsite_tensor_descriptors(callee, additions)
            # The catalogue graph is shared by every callsite.  A tensor
            # descriptor alone is not a complete structural specialization:
            # helpers such as ``unbroadcast(G, target_shape)`` also require a
            # callsite-local shape value before their source control can be
            # selected.  Folding the shared graph here permanently discarded
            # branches before that value was known.  The copied graph is
            # folded below by ``_callsite_specialized_shell_type`` after both
            # descriptor and literal arguments have been collected.
            changed = True
            mutation_counts["callee_descriptors"] += len(additions)

        state_digest, tensor_fact_count = specialization_state()
        repeated_round = seen_round_states.get(state_digest)
        seen_round_states.setdefault(state_digest, round_index)
        from .identity_concordance import current_identity_book

        return_page = current_identity_book().page(
            "callsite_return_specialization"
        )
        oscillating_rows = return_page.oscillating_rows()
        if _progress is not None:
            _progress({
                "specialization_state": "round-end",
                "round": round_index,
                "duration_seconds": round(
                    time.monotonic() - round_started, 6
                ),
                "changed": bool(changed),
                "callsites": callsites,
                "tensor_facts": tensor_fact_count,
                "state_digest": state_digest,
                "repeats_round": repeated_round,
                "oscillating_rows": len(oscillating_rows),
                **mutation_counts,
            })
        specialization_guard.round(changed, state=state_digest)


def propagate_bound_planner_specializations(
    graph: Any,
    entrypoint: str,
    bindings: Mapping[str, Any],
    mutable_parameters: Iterable[str] = (),
) -> None:
    """Carry safe structural feed values through the linked call graph.

    This is planning metadata, not constant folding of numerical tensors.
    Only syntax that reads an already-supplied object without invoking
    arbitrary user code is resolved: names, public attributes, subscripts,
    literal containers, and standard structural views such as ``items``.
    """

    function_table = getattr(graph, "function_table", None)
    if function_table is None:
        return
    reference = function_table.reference(entrypoint)
    if reference is None:
        return

    def resolve(
        node: ast.AST, environment: Mapping[str, Any], used: set | None = None,
    ) -> Any:
        # ``used`` collects the caller bindings this expression read, so the
        # callee's row can derive from the caller's rows for exactly them.
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name) and node.id in environment:
            if used is not None:
                used.add(str(node.id))
            return environment[node.id]
        if isinstance(node, ast.Attribute) and not node.attr.startswith("_"):
            return getattr(resolve(node.value, environment, used), node.attr)
        if isinstance(node, ast.Subscript):
            return resolve(node.value, environment, used)[
                resolve(node.slice, environment, used)
            ]
        if isinstance(node, ast.Tuple):
            return tuple(resolve(item, environment, used) for item in node.elts)
        if isinstance(node, ast.List):
            return [resolve(item, environment, used) for item in node.elts]
        if isinstance(node, ast.Dict):
            return {
                resolve(key, environment, used): resolve(value, environment, used)
                for key, value in zip(node.keys, node.values)
            }
        if isinstance(node, ast.Call):
            args = [
                resolve(argument, environment, used) for argument in node.args
            ]
            if isinstance(node.func, ast.Name) and node.func.id in {
                "tuple", "list", "range", "enumerate", "zip"
            }:
                return {
                    "tuple": tuple, "list": list, "range": range,
                    "enumerate": enumerate, "zip": zip,
                }[node.func.id](*args)
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"items", "keys", "values"}
                and not node.keywords
                and not args
            ):
                owner = resolve(node.func.value, environment, used)
                if isinstance(owner, Mapping):
                    return getattr(owner, node.func.attr)()
        raise ValueError("expression is not a safe structural binding")

    mutable = frozenset(map(str, mutable_parameters))
    entry = function_table.entry(reference.address).graph
    if entry is None:
        return

    # Mutable public tensors are values, not planner literals, but their
    # extents and element type are still the fixed Program-ABI contract for
    # this compilation.  Previously the mutable filter below discarded both
    # facts together.  The root web shell consequently entered hierarchy
    # planning without descriptors, specialized every helper as rank zero,
    # and only rediscovered the real shapes after its regions had already
    # been cut.  Attach the non-value boundary facts to the exact formal
    # identities and publish them through the same concordance used by
    # callsite specialization.  No payload value is retained or folded.
    mutable_descriptors: dict[str, dict[str, Any]] = {}
    for name in mutable:
        value = bindings.get(name)
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is None or callable(shape) or dtype is None:
            continue
        try:
            extents = tuple(map(int, shape))
        except (TypeError, ValueError):
            continue
        descriptor = {
            "shape": extents,
            "dtype": str(dtype),
            "rank": len(extents),
        }
        mutable_descriptors[name] = descriptor
        _publish_formal_shape(
            str(entry.G.graph.get("function_name") or entrypoint),
            name,
            descriptor,
            "program-abi",
        )
    if mutable_descriptors:
        _apply_callsite_tensor_descriptors(entry, mutable_descriptors)

    from ..common.tensors.topological_reducer import node_identity_cell
    from .concordance_declarations import (
        EXTERNAL_DECLARATION, PLANNER_SPECIALIZATION,
        PLANNER_SPECIALIZATION_PAGE, SpecializationFact,
        SpecializationSource,
    )
    from .identity_concordance import (
        Mode, Unsourced, current_identity_book,
    )

    book = current_identity_book()

    def post_bound(graph_obj: Any, environment: Mapping[str, Any],
                   sources: Mapping[str, tuple]) -> None:
        """One ``SpecializationFact(value, BOUND)`` row per forwarded
        binding under ``graph_obj``'s scope: DERIVED from the call node's
        cell and the caller's rows the binding was resolved from; the
        entrypoint's own bindings come from the caller of the compile (no
        cell on the book) and are posted ``Unsourced(EXTERNAL_DECLARATION)``
        under the latch."""

        scope = graph_obj.G.graph.get("lexical_read_scope")
        if scope is None:
            return
        for name, value in environment.items():
            fact = SpecializationFact(value, SpecializationSource.BOUND)
            cells = tuple(sources.get(str(name)) or ())
            if cells:
                _post_planner_specialization(graph_obj, str(name), fact, cells)
                continue
            row = (tuple(scope), str(name))
            if book.pages.get(PLANNER_SPECIALIZATION_PAGE.name) is not None and (
                book.pages[PLANNER_SPECIALIZATION_PAGE.name].latest(row) == fact
            ):
                continue
            book.post(
                PLANNER_SPECIALIZATION_PAGE, row, fact,
                stage=PLANNER_SPECIALIZATION,
                provenance=Unsourced(EXTERNAL_DECLARATION), mode=Mode.REVISE,
            )
        _materialize_planner_specializations(graph_obj)

    queue: list[tuple[Any, dict[str, Any], dict[str, tuple]]] = [(
        entry,
        {
            str(name): value
            for name, value in bindings.items()
            if str(name) not in mutable
        },
        {},
    )]
    visited: set[tuple[int, tuple[str, ...]]] = set()
    while queue:
        current, environment, sources = queue.pop(0)
        key = (id(current), tuple(sorted(environment)))
        if key in visited:
            continue
        visited.add(key)
        current.G.graph.setdefault("planner_specializations", {}).update(
            environment
        )
        post_bound(current, environment, sources)
        current_scope = current.G.graph.get("lexical_read_scope")

        def caller_cells(node_id: int, used: set) -> tuple:
            """The call node's identity cell plus the caller's rows for the
            bindings the argument expression read."""

            cells = [node_identity_cell(current, int(node_id))]
            if current_scope is not None:
                for name in sorted(used):
                    cell = book.latest_ref(
                        PLANNER_SPECIALIZATION_PAGE,
                        (tuple(current_scope), str(name)),
                    )
                    if cell is not None:
                        cells.append(cell)
            return tuple(cells)

        for _node_id, data in current.G.nodes(data=True):
            attributes = data.get("attributes") or {}
            callee_reference = attributes.get(
                "callee_ref", attributes.get("method_ref")
            )
            expression = data.get("expr_obj")
            if callee_reference is None or not isinstance(expression, ast.Call):
                continue
            try:
                callee = function_table.entry(int(callee_reference)).graph
            except (KeyError, TypeError, ValueError):
                continue
            if callee is None:
                continue
            receiver, positional, _all = _method_parameter_layout(callee.G)
            resolved: dict[str, Any] = {}
            resolved_sources: dict[str, tuple] = {}
            if receiver is not None and isinstance(expression.func, ast.Attribute):
                used: set = set()
                try:
                    resolved[receiver] = resolve(
                        expression.func.value, environment, used,
                    )
                except (AttributeError, KeyError, TypeError, ValueError):
                    pass
                else:
                    resolved_sources[receiver] = caller_cells(_node_id, used)
            for position, argument in enumerate(expression.args):
                if position >= len(positional):
                    break
                used = set()
                try:
                    resolved[positional[position]] = resolve(
                        argument, environment, used,
                    )
                except (AttributeError, KeyError, TypeError, ValueError):
                    continue
                resolved_sources[positional[position]] = caller_cells(
                    _node_id, used,
                )
            for keyword in expression.keywords:
                if keyword.arg is None:
                    continue
                used = set()
                try:
                    resolved[str(keyword.arg)] = resolve(
                        keyword.value, environment, used,
                    )
                except (AttributeError, KeyError, TypeError, ValueError):
                    continue
                resolved_sources[str(keyword.arg)] = caller_cells(
                    _node_id, used,
                )
            if resolved:
                queue.append((callee, resolved, resolved_sources))


_CALLSITE_SHELL_TYPE_CACHE: dict[tuple[object, ...], type] = {}
def _stable_signature_value(value: Any) -> object:
    """A hashable, order-independent spelling of a descriptor/specialization
    mapping, for keying one callsite signature against another (the same
    spelling ``_callsite_specialized_shell_type`` keys planned shells by)."""

    if isinstance(value, dict):
        return tuple(sorted(
            (_stable_signature_value(key), _stable_signature_value(item))
            for key, item in value.items()
        ))
    if isinstance(value, (tuple, list, set, frozenset)):
        return tuple(_stable_signature_value(item) for item in value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


#: Shape-only operations (no runtime existence, alias in every target): the
#: likeness table's ``_SHAPE_ONLY`` set (``ssa_llvm_backend``).
def _shape_only_view_operations() -> frozenset[str]:
    from .ssa_llvm_backend import _SHAPE_ONLY

    return frozenset(_SHAPE_ONLY)


_SHAPE_ONLY_VIEW_OPERATIONS = _shape_only_view_operations()


def _apply_callsite_tensor_descriptors(
    graph: Any,
    descriptors: Mapping[str, Mapping[str, Any]],
) -> None:
    graph.G.graph["callsite_tensor_descriptor_names"] = tuple(sorted({
        *map(
            str,
            graph.G.graph.get("callsite_tensor_descriptor_names") or (),
        ),
        *map(str, descriptors),
    }))
    authoritative = dict(
        graph.G.graph.get("planner_tensor_descriptors") or {}
    )
    authoritative.update({
        str(name): copy.deepcopy(dict(descriptor))
        for name, descriptor in descriptors.items()
    })
    graph.G.graph["planner_tensor_descriptors"] = authoritative
    identities = graph.G.graph.get("identity_table") or {}
    changed_inputs = set()
    for name, descriptor in descriptors.items():
        candidates = tuple(dict.fromkeys((
            *tuple(identities.get(str(name), ())),
            *(
                int(node_id)
                for node_id, data in graph.G.nodes(data=True)
                if data.get("type") == "Input"
                and (data.get("attributes") or {}).get("binding_name") == name
            ),
        )))
        for node_id in candidates:
            if int(node_id) not in graph.G:
                continue
            data = graph.G.nodes[int(node_id)]
            if data.get("type") != "Input":
                continue
            if data.get("tensor") != dict(descriptor):
                changed_inputs.add(int(node_id))
            data["tensor"] = copy.deepcopy(dict(descriptor))

    # Cached intermediate shapes were derived before these exact argument
    # descriptors arrived. Invalidate their dependency closure, including the
    # concordance proof that would otherwise immediately repopulate the cache.
    _invalidate_tensor_descriptor_dependents(
        graph, changed_inputs, "callsite-input-specialization-changed",
    )


def _alias_projection_to_member(
    graph: Any, projection: int, leaf_id: int,
) -> None:
    """Give one authored ``name[index]`` projection its member's identity."""

    projection = int(projection)
    leaf_id = int(leaf_id)
    if projection not in graph.G or leaf_id not in graph.G:
        return
    projection_data = graph.G.nodes[projection]
    for successor in tuple(graph.G.successors(projection)):
        successor_data = graph.G.nodes[int(successor)]
        _set_operands(graph, int(successor), [
            (
                leaf_id if int(parent) == projection else int(parent),
                str(role),
            )
            for parent, role in successor_data.get("parents") or ()
        ], cause=_PROJECTION_TO_LEAF, same={projection: leaf_id})
        _follow_declared_value_source(successor_data, projection, leaf_id)
    index_ids = tuple(
        int(parent)
        for parent, role in projection_data.get("parents") or ()
        if str(role) in {"index", "slice", "subscript"}
    )
    # The projection's own positions retire (its parents' ``children``
    # entries and edges with them) before the node is removed below.
    _set_operands(graph, projection, [], cause=_PROJECTION_TO_LEAF)
    # A loop that carries this projection (``a, b = history`` bound as its
    # initial) caches the projection id without being its successor; move
    # every cached copy before the node is gone.
    _retarget_all_cached_value_ids(graph, projection, leaf_id)
    graph.roots = [leaf_id if int(root) == projection else int(root)
                   for root in graph.roots]
    graph.G.graph["return_value_nodes"] = {
        site: leaf_id if value == projection else value
        for site, value in (graph.G.graph.get("return_value_nodes") or {}).items()
    }
    graph.G.graph["return_slot_values"] = {
        site: tuple(leaf_id if value == projection else value for value in values)
        for site, values in (graph.G.graph.get("return_slot_values") or {}).items()
    }
    graph.G.remove_node(projection)
    for index_id in index_ids:
        if (
            index_id in graph.G
            and graph.G.out_degree(index_id) == 0
            and graph.G.nodes[index_id].get("type") in {
                "Constant", "Const", "const",
            }
        ):
            graph.G.remove_node(index_id)
    identities = graph.G.graph.get("identity_table") or {}
    # The projection's ``name_binding`` versions (and the return slots that
    # named it) revise to the member's identity (plan 80, A2.7).
    from .concordance_declarations import PLANNER_PROJECTION_ALIAS

    _post_identity_table_mutation(
        graph, aliases={projection: leaf_id}, stage=PLANNER_PROJECTION_ALIAS,
    )
    graph.G.graph["identity_table"] = {
        str(key): tuple(
            leaf_id if int(value_id) == projection else int(value_id)
            for value_id in history
        )
        for key, history in identities.items()
    }


def _apply_callsite_aggregate_descriptors(
    graph: Any,
    descriptors: Mapping[str, tuple[Any, ...]],
) -> None:
    """Bind an aggregate formal to its exact authored projection identities."""

    identities = graph.G.graph.get("identity_table") or {}
    for name, root_members in descriptors.items():
        inputs = tuple(dict.fromkeys((
            *tuple(identities.get(str(name), ())),
            *(
                int(node_id)
                for node_id, data in graph.G.nodes(data=True)
                if data.get("type") == "Input"
                and (data.get("attributes") or {}).get("binding_name") == name
            ),
        )))
        inputs = tuple(
            int(node_id) for node_id in inputs
            if int(node_id) in graph.G
            and graph.G.nodes[int(node_id)].get("type") == "Input"
        )
        if len(inputs) != 1:
            raise ValueError(
                f"aggregate parameter {name!r} does not have one input identity: "
                f"{inputs!r}"
            )
        def materialize(
            input_id: int,
            binding_path: str,
            members: tuple[Any, ...],
            *,
            root_binding: str,
            aggregate_path: tuple[int, ...] = (),
        ) -> None:
            """Materialize one level, retaining nested member ledgers."""

            projections: dict[int, list[int]] = {}
            for node_id, data in tuple(graph.G.nodes(data=True)):
                if str(
                    data.get("op") or data.get("type") or ""
                ).casefold() not in {"indexed", "getitem", "subscript"}:
                    continue
                base = tuple(
                    int(parent)
                    for parent, role in data.get("parents") or ()
                    if str(role) in {"base", "value", "object"}
                )
                if base != (int(input_id),):
                    continue
                index_ids = tuple(
                    int(parent)
                    for parent, role in data.get("parents") or ()
                    if str(role) in {"index", "slice", "subscript"}
                )
                if len(index_ids) != 1 or index_ids[0] not in graph.G:
                    continue
                index_data = graph.G.nodes[index_ids[0]]
                index = index_data.get("constant")
                if index is None:
                    index = (index_data.get("attributes") or {}).get("value")
                if (
                    not isinstance(index, int) or isinstance(index, bool)
                    or not members
                ):
                    continue
                normalized = int(index) % len(members)
                projections.setdefault(normalized, []).append(int(node_id))

            leaves = []
            nested = []
            for index, descriptor in enumerate(members):
                leaf_id = next_process_value_id(graph)
                member_path = f"{binding_path}[{index}]"
                full_path = (*aggregate_path, int(index))
                is_nested = isinstance(descriptor, tuple)
                graph.G.add_node(
                    leaf_id,
                    type="Input", op="input", label=member_path,
                    expr_obj=None, value_id=leaf_id,
                    parents=[], children=[],
                    attributes={
                        "binding_name": member_path,
                        "binding_kind": "parameter",
                        "aggregate_parent_binding": str(root_binding),
                        "aggregate_path": tuple(full_path),
                        "aggregate_index": int(index),
                    },
                    tensor=(
                        copy.deepcopy(dict(descriptor))
                        if isinstance(descriptor, Mapping) else {}
                    ),
                )
                leaves.append(leaf_id)
                for projection in projections.get(index, ()):
                    _alias_projection_to_member(graph, projection, leaf_id)
                if is_nested:
                    nested.append((
                        leaf_id, member_path, tuple(descriptor), full_path,
                    ))

            leaves = tuple(leaves)
            input_attributes = dict(
                graph.G.nodes[input_id].get("attributes") or {}
            )
            input_attributes.update({
                "producer_kind": "aggregate",
                "aggregate_kind": "tuple",
                "aggregate_leaf_value_ids": leaves,
                "tensor_output_descriptors": copy.deepcopy(tuple(members)),
                "sequence_key_columns": (),
                "sequence_column_count": 1,
                "sequence_writable": False,
            })
            formal = graph.G.nodes[input_id]
            formal["attributes"] = input_attributes
            formal["type"] = "Tuple"
            formal["op"] = None
            formal["tensor"] = {}
            _set_operands(graph, int(input_id), [
                (int(leaf), "elts") for leaf in leaves
            ], cause=_AGGREGATE_FORMAL_MEMBERS)
            for leaf_id, member_path, nested_members, nested_path in nested:
                materialize(
                    leaf_id,
                    member_path,
                    nested_members,
                    root_binding=root_binding,
                    aggregate_path=tuple(nested_path),
                )

        materialize(
            int(inputs[0]),
            str(name),
            tuple(root_members),
            root_binding=str(name),
        )


def _expand_specialized_unbroadcast_identity(graph: Any) -> bool:
    """Lower the authored ``unbroadcast`` helper to canonical tensor nodes.

    The helper's Python loop is only structural shape logic.  Once a callsite
    supplies both shapes, retain its exact BACKWARD_RULES meaning as a finite
    sequence of existing ``sum`` and ``reshape`` operators instead of asking
    the control planner to carry a tensor through a Python ``for`` loop.
    """

    if str(graph.G.graph.get("function_name") or "") != "unbroadcast":
        return False
    descriptors = graph.G.graph.get("planner_tensor_descriptors") or {}
    specializations = graph.G.graph.get("planner_specializations") or {}
    source_descriptor = descriptors.get("G")
    target_shape = specializations.get("target_shape")
    if source_descriptor is None or not isinstance(target_shape, (tuple, list)):
        return False
    source_shape = tuple(map(int, source_descriptor.get("shape") or ()))
    target_shape = tuple(map(int, target_shape))
    input_id = next((
        int(node_id)
        for node_id, data in graph.G.nodes(data=True)
        if data.get("type") == "Input"
        and (data.get("attributes") or {}).get("binding_name") == "G"
    ), None)
    if input_id is None:
        return False
    metadata = copy.deepcopy(dict(graph.G.graph))
    input_data = copy.deepcopy(dict(graph.G.nodes[input_id]))
    graph.G.clear()
    graph.G.graph.update(metadata)
    input_data["parents"] = []
    input_data["children"] = []
    input_data["tensor"] = copy.deepcopy(dict(source_descriptor))
    graph.G.add_node(input_id, **input_data)
    next_id = input_id + 1
    current = input_id
    current_shape = list(source_shape)

    def append(op: str, attributes: Mapping[str, Any], shape: tuple[int, ...]):
        nonlocal next_id, current
        node_id = next_id
        next_id += 1
        role = "operand"
        graph.G.add_node(
            node_id, op=op, type=op, label=op,
            parents=[], children=[],
            attributes={
                **dict(attributes),
                "tensor_candidate": op,
                "authored_identity": "backward_registry.unbroadcast",
            },
            extra_args=dict(attributes),
            tensor={
                "shape": tuple(shape),
                "dtype": str(source_descriptor.get("dtype") or "float64"),
            },
            control={}, constant=None, expr_obj=None, store_id=None,
        )
        _set_operands(graph, node_id, [(current, role)], cause=_UNBROADCAST_CHAIN)
        current = node_id

    while len(current_shape) > len(target_shape):
        current_shape.pop(0)
        append(
            "sum", {"dim": 0, "keepdim": False}, tuple(current_shape),
        )
    for axis, (actual, target) in enumerate(zip(current_shape, target_shape)):
        if target == 1 and actual != 1:
            current_shape[axis] = 1
            append(
                "sum", {"dim": axis, "keepdim": True},
                tuple(current_shape),
            )
    append("reshape", {"shape": target_shape}, target_shape)
    graph.roots = [current]
    identities = copy.deepcopy(metadata.get("identity_table") or {})
    identities["G"] = (input_id,)
    identities["result_0"] = (current,)
    graph.G.graph.update({
        "identity_table": identities,
        "function_outputs": ("result_0",),
        "function_parameters": ("G",),
        "positional_parameters": ("G",),
        "keyword_only_parameters": (),
        "authored_identity_expansion": "backward_registry.unbroadcast",
    })
    return True


def _callsite_specialized_shell_type(
    owner: Any,
    node_id: int,
    reference: int,
    fallback: type,
    max_nodes_per_dispatch: int,
) -> type:
    """Plan one literal callsite once, before its shell is instantiated."""

    caller = owner.process_graph
    if int(node_id) not in caller.G:
        return fallback
    function_table = getattr(caller, "function_table", None)
    if function_table is None:
        return fallback
    try:
        original = function_table.entry(int(reference)).graph
    except (KeyError, TypeError, ValueError):
        return fallback
    if original is None:
        return fallback
    receiver, positional, _all = _method_parameter_layout(original.G)
    specializations = {}
    tensor_descriptors: dict[str, dict[str, Any]] = {}
    aggregate_descriptors: dict[str, tuple[Any, ...]] = {}
    record_descriptors: dict[str, dict[str, Any]] = {}
    # What each argument is, as the caller's concordance states it.  The
    # specialization's parameters are exactly these values.
    parameter_classes: dict[str, tuple[str, int]] = {}
    #: Per literal parameter: the caller argument cells that proved it; the
    #: copy's ``planner_specialization`` rows derive from them.
    specialization_cells: dict[str, tuple] = {}
    from ..common.tensors.topological_reducer import _source_numeric_scope
    from .identity_concordance import current_identity_book

    caller_scope = _source_numeric_scope(caller.G)
    caller_class_page = current_identity_book().page(
        "source_value_class_concordance"
    )
    bound_parameters: set[str] = set()
    data = caller.G.nodes[int(node_id)]
    caller_record_abi = dict(
        caller.G.graph.get("parameter_record_abi") or {}
    )
    caller_identities = caller.G.graph.get("identity_table") or {}

    def caller_record_for_value(value_id: int) -> dict[str, Any] | None:
        """Resolve a concrete caller record from its exact parameter value."""

        value_id = int(value_id)
        matches = [
            dict(record)
            for parameter_name, record in caller_record_abi.items()
            if value_id in set(map(
                int, caller_identities.get(str(parameter_name), ())
            ))
        ]
        if len(matches) == 1:
            return matches[0]
        node = caller.G.nodes.get(value_id, {})
        binding_name = str(
            (node.get("attributes") or {}).get("binding_name") or ""
        )
        record = caller_record_abi.get(binding_name)
        return None if record is None else dict(record)

    for parent, role_value in _expanded_callsite_argument_edges(
        caller, int(node_id),
    ):
        role = str(role_value)
        parameter = _callsite_parameter_name(
            role, receiver, positional,
        )
        if parameter is None:
            continue
        parameter = str(parameter)
        bound_parameters.add(parameter)
        class_fact = caller_class_page.latest((caller_scope, int(parent)))
        if isinstance(class_fact, tuple) and len(class_fact) >= 2:
            parameter_classes[parameter] = (
                str(class_fact[0]), max(int(class_fact[1]), 1),
            )
        declared_record_abi = dict(
            original.G.graph.get("parameter_record_abi") or {}
        ).get(parameter)
        callsite_record_abi = caller_record_for_value(int(parent))
        record_abi = (
            callsite_record_abi
            if callsite_record_abi is not None
            else declared_record_abi
        )
        if record_abi is not None and record_abi.get("identity") is not None:
            # A declared record identity is a structural type fact even when
            # the callsite carries no scalar literal or tensor descriptor.
            # Replanning the clean callee lets isinstance(record, Schema)
            # fold from that contract before regions/branches are carved.
            record_descriptors[parameter] = copy.deepcopy(dict(record_abi))
        descriptor = _tensor_descriptor(caller, int(parent))
        if descriptor is not None:
            tensor_descriptors[parameter] = descriptor
        static_argument = _source_static_value(caller, int(parent))
        leaves = _authored_aggregate_leaves(caller, int(parent))
        if leaves and not static_argument:
            # Every authored member is a scoped callee identity whether or
            # not its tensor contract is already known.  A member without a
            # descriptor is materialized undescribed and fails loudly where a
            # shape is needed; it must not erase the whole aggregate contract
            # and leave the callee formal with zero members.
            missing = tuple(
                int(leaf) for leaf in leaves if int(leaf) not in caller.G
            )
            if missing:
                raise ValueError(
                    f"aggregate argument {parameter!r} names leaf identities "
                    f"{missing!r} that no longer exist in "
                    f"{caller.G.graph.get('function_name')!r}"
                )
            aggregate_descriptors[parameter] = tuple(
                _authored_aggregate_descriptor_tree(caller, int(leaf))
                for leaf in leaves
            )
        if static_argument:
            try:
                specializations[parameter] = _source_static_literal(
                    caller, int(parent)
                )
            except ValueError:
                pass
            else:
                # The cells of THIS callsite's argument prove the copy's
                # literal (design 7.1) and the authored formal's literal.
                specialization_cells[parameter] = _argument_identity_cells(
                    caller, int(parent),
                )
                _publish_formal_literal(
                    str(original.G.graph.get("function_name")),
                    parameter,
                    specializations[parameter],
                    str(caller.G.graph.get("function_name")),
                    specialization_cells[parameter],
                )
        proven_descriptor = tensor_descriptors.get(parameter)
        if proven_descriptor is not None and descriptor_states_a_shape(
            proven_descriptor
        ):
            # A stated rank-0 shape is a fact too; see the fixed-point
            # publication in _propagate_callsite_tensor_specializations.
            _publish_formal_shape(
                str(original.G.graph.get("function_name")),
                parameter,
                proven_descriptor,
                str(caller.G.graph.get("function_name")),
                _argument_identity_cells(caller, int(parent))[:1],
            )

    # Nested functions receive enclosing values through closure identities,
    # not through the Call node's positional edges.  They are nevertheless
    # exact callsite inputs and must carry the same recursive aggregate
    # ledger.  Without this, ``selected`` saw ``left_limbs`` as an opaque
    # zero-member input even though its caller had two concorded limb leaves.
    for _capture_id, capture in original.G.nodes(data=True):
        capture_attributes = capture.get("attributes") or {}
        if str(capture.get("type") or "") != "Input" or str(
            capture_attributes.get("binding_kind") or ""
        ) not in {"closure", "external"}:
            continue
        capture_name = str(capture_attributes.get("binding_name") or "")
        if not capture_name or capture_name in bound_parameters:
            continue
        capture_values = tuple(dict.fromkeys(
            int(value_id)
            for value_id in caller_identities.get(capture_name, ())
            if int(value_id) in caller.G
        ))
        if len(capture_values) != 1:
            continue
        capture_value = int(capture_values[0])
        capture_leaves = _authored_aggregate_leaves(
            caller, capture_value,
        )
        if capture_leaves:
            aggregate_descriptors[capture_name] = tuple(
                _authored_aggregate_descriptor_tree(caller, int(leaf))
                for leaf in capture_leaves
            )
            bound_parameters.add(capture_name)
            continue
        capture_descriptor = _tensor_descriptor(caller, capture_value)
        if capture_descriptor is not None:
            tensor_descriptors[capture_name] = capture_descriptor
            bound_parameters.add(capture_name)
    for parameter, default in (
        original.G.graph.get("parameter_defaults") or {}
    ).items():
        parameter = str(parameter)
        if parameter in bound_parameters:
            continue
        if default is None or isinstance(
            default, (bool, int, float, str, bytes)
        ) or (
            isinstance(default, tuple)
            and all(
                item is None or isinstance(
                    item, (bool, int, float, str, bytes)
                )
                for item in default
            )
        ):
            specializations[parameter] = copy.deepcopy(default)
    if not (
        specializations or tensor_descriptors
        or aggregate_descriptors or record_descriptors
    ):
        return fallback

    def stable(value: Any) -> object:
        if isinstance(value, dict):
            return tuple(sorted(
                (stable(key), stable(item))
                for key, item in value.items()
            ))
        if isinstance(value, (tuple, list, set, frozenset)):
            return tuple(stable(item) for item in value)
        try:
            hash(value)
        except TypeError:
            return repr(value)
        return value

    key = (
        id(function_table),
        # A planned shell's rows were posted on the book of the compile that
        # built it; a later compile (a new book) must plan again so its own
        # book holds them (plan 80, R3).
        id(current_identity_book()),
        int(reference),
        tuple(sorted(
            (name, stable(value))
            for name, value in specializations.items()
        )),
        tuple(sorted(
            (name, stable(value))
            for name, value in tensor_descriptors.items()
        )),
        tuple(sorted(
            (name, stable(value))
            for name, value in aggregate_descriptors.items()
        )),
        tuple(sorted(
            (name, stable(value))
            for name, value in record_descriptors.items()
        )),
        tuple(sorted(parameter_classes.items())),
        int(max_nodes_per_dispatch),
    )
    def publish_call_result_shape(specialized_graph: Any) -> None:
        """Join an exact specialized return to its caller value identity."""

        outputs = tuple(
            specialized_graph.G.graph.get("function_outputs") or ()
        )
        if len(outputs) != 1:
            return
        identities = specialized_graph.G.graph.get("identity_table") or {}
        output_ids = tuple(identities.get(str(outputs[0]), ()))
        descriptor = None
        inspected = []
        for candidate in reversed(output_ids):
            if int(candidate) not in specialized_graph.G:
                continue
            inferred = _structured_output_descriptor(
                specialized_graph, int(candidate),
            )
            inspected.append((int(candidate), copy.deepcopy(inferred)))
            if isinstance(inferred, Mapping) and (
                descriptor_states_a_shape(inferred)
            ):
                descriptor = inferred
                break
        if os.environ.get("TURING_DEBUG_CALL_RESULT_SHAPE"):
            print(
                "CALL-RESULT-SHAPE "
                f"caller={caller.G.graph.get('function_name')!r} "
                f"callee={specialized_graph.G.graph.get('function_name')!r} "
                f"outputs={outputs!r} output_ids={output_ids!r} "
                "planner="
                f"{specialized_graph.G.graph.get('planner_tensor_descriptors')!r} "
                f"inspected={inspected!r}",
                file=sys.stderr, flush=True,
            )
        if descriptor is None:
            return
        extents = tuple(descriptor.get("shape") or ())
        if not extents:
            return
        from .identity_concordance import record_proven_shape, shape_scope_of

        record_proven_shape(
            shape_scope_of(caller),
            int(data.get("value_id", node_id)),
            extents,
            descriptor.get("dtype"),
            int(_dependency_levels(caller).get(int(node_id), 0)),
        )
        # A Precision operation may have been identified before this callee's
        # exact return extents existed.  The concordance publication above is
        # the causal event that makes its tensor identity complete; refresh
        # the caller from that fact now, so the later SSA copy does not retain
        # the provisional rank-zero descriptor.
        from ..common.tensors.topological_reducer import (
            specialize_python_precision_widths,
        )

        specialize_python_precision_widths(caller)

    cached = _CALLSITE_SHELL_TYPE_CACHE.get(key)
    if cached is not None:
        publish_call_result_shape(cached.process_graph)
        return cached
    specialized = extract_clean_process_subgraph(
        original,
        original.G,
    )
    # A callsite specialization is a variant of the function: it states
    # shapes in its own shape scope, never in the original's or a sibling
    # specialization's.
    from .identity_concordance import fork_shape_scope

    fork_shape_scope(specialized, "callsite_specialization")
    # These are callsite facts, not accumulators.  A shared function-table
    # graph may already carry metadata from an earlier specialization; using
    # ``setdefault().update()`` allowed a scalar literal from one occurrence
    # to survive into a later tensor-valued occurrence of the same parameter.
    specialized.G.graph["planner_specializations"] = copy.deepcopy(
        specializations
    )
    # The copy's own ``planner_specialization`` rows, under the read scope
    # ``fork_read_scope`` just minted for it (which inherited none of the
    # shared callee's rows): each literal DERIVED from this callsite's
    # argument cells, each default from the formal it binds (design 7.1).
    _post_copy_planner_specializations(
        specialized, specializations, specialization_cells,
    )
    specialized.G.graph["planner_parameter_classes"] = dict(
        parameter_classes
    )
    specialized.G.graph["planner_tensor_descriptors"] = copy.deepcopy(
        tensor_descriptors
    )
    if record_descriptors:
        parameter_records = copy.deepcopy(dict(
            specialized.G.graph.get("parameter_record_abi") or {}
        ))
        parameter_records.update(copy.deepcopy(record_descriptors))
        specialized.G.graph["parameter_record_abi"] = parameter_records
    _apply_callsite_tensor_descriptors(specialized, tensor_descriptors)
    _apply_callsite_aggregate_descriptors(
        specialized, aggregate_descriptors,
    )
    if not _expand_specialized_unbroadcast_identity(specialized):
        _fold_callsite_structural_values(specialized)
    planned = strategize_shell_deployment(
        specialized,
        max_nodes_per_dispatch=int(max_nodes_per_dispatch),
        _function_table_stack=(id(function_table),),
    )
    # Recursive planning above specializes the callee's own calls first.  At
    # this point its return descriptor is therefore the exact callsite
    # contract, including facts that did not exist during the earlier global
    # catalogue fixed point.  Publish it to the caller's authored value row;
    # every later graph copy and SSA object asks that row instead of relying
    # on mutation of this one specialized graph.
    publish_call_result_shape(specialized)
    _CALLSITE_SHELL_TYPE_CACHE[key] = planned
    return planned


# Above this an unroll is refused rather than attempted. It is a guard
# against a runaway trip count turning into an unbounded program, not a
# judgement about what is reasonable: a caller who wants this much detail is
# spending their own compute, and the emitted program is theirs to hold.
MAX_UNROLL_LIMIT = 4096


class ProcessGraphGLSLDeployment:
    # Per-compile values. A fresh subclass is created for each compiled
    # ProcessGraph (see strategize_shell_deployment) with these overridden
    # via ordinary class-attribute assignment -- no source-text templating.
    process_graph = None
    external_function_table = None
    static_python_bindings = None
    dispatch_plan = None
    loop_plans = None
    loop_region_indices = None
    loop_shader_reductions = None
    control_deployment_regions = None
    process_graph_boundary = None
    python_callable = None
    dispatch_subgraphs = None
    deep_compilers = None
    ephemeral_callables = None
    reference_table_template = None
    function_shell_types = {}
    runtime_closure_only = False
    planned_function_references = ()
    activation_root_references = ()
    catalogued_function_references = ()
    catalogue_only_function_references = ()
    source_node_count = None
    primitive_count = None
    loop_count = None
    dispatch_count = None
    deployment_batches = None
    max_nodes_per_dispatch = None
    selection_progress = None

    def __init__(
        self,
        *,
        batch_size=None,
        profiling=False,
        verbose_profile=False,
        input_slots=64,
        output_slots=64,
        legacy_fused_network=False,
        shell_language="glsl",
        _defer_callsite_planning=False,
        **tuning,
    ):
        tuning = dict(tuning)
        tuning["legacy_fused_network"] = bool(legacy_fused_network)
        tuning["shell_language"] = str(shell_language)
        self.state = {}
        self.tuning = dict(tuning)
        self.legacy_fused_network = bool(legacy_fused_network)
        self.shell_language = str(shell_language)
        if self.shell_language not in {"python", "c", "glsl"}:
            raise ValueError(
                "shell_language must be python, c, or glsl"
            )
        self.control_runtime = (
            "legacy_fused_network"
            if self.legacy_fused_network
            else "composed_control"
        )
        self.batch_size = dict(
            self.deployment_batches
            if batch_size is None
            else batch_size
        )
        self.profiling = bool(profiling or verbose_profile)
        self.verbose_profile = bool(verbose_profile)
        self._profiler = DeploymentProfiler(
            self.profiling,
            verbose=self.verbose_profile,
        )
        self.error_buffer = self._profiler.error_buffer
        self.profile = self._profiler
        self.profile_path = _shell_profile_name(self)
        self.inputs = DeploymentFIFO(input_slots)
        self.outputs = DeploymentFIFO(output_slots)
        self.reference_tables = self.reference_table_template.copy()
        self.function_references = self.reference_tables.functions
        self.constant_references = self.reference_tables.constants
        self.memory_references = self.reference_tables.memory
        self.recursion_references = self.reference_tables.recursion
        self.reference_correlations = self.reference_tables.correlations
        # Specialization can prove additional method callees. Keep the
        # growing registry on this deployment, not on its cached class.
        self.function_shell_types = dict(self.function_shell_types)
        self.specialized_dependency_extensions = []
        self._callsite_planning_deferred = bool(
            _defer_callsite_planning
        )
        # Only the outer deployment owns and instantiates the function
        # catalogue.  Catalogue members and literal-callsite shells are
        # constructed shallowly, attached to that one shared catalogue, and
        # then traversed by ``plan_callsites`` with its active reference
        # chain intact.  Eagerly constructing a catalogue from one of those
        # children re-entered this constructor before the chain could be
        # passed down, making recursive call topologies expand forever.
        self.function_shells = (
            {}
            if self._callsite_planning_deferred
            else {
                reference: shell_type(
                    **tuning,
                    _defer_callsite_planning=True,
                )
                for reference, shell_type in self.function_shell_types.items()
            }
        )
        self._owns_function_shells = bool(self.function_shells)
        for function_shell in self.function_shells.values():
            function_shell.function_shells = self.function_shells
            function_shell._owns_function_shells = False
        compilation_unit_receipt = dict(
            self.process_graph.G.graph.get("compilation_unit_plan") or {}
        )
        callsite_unit_by_reference = {
            int(reference): int(unit_index)
            for reference, unit_index in (
                compilation_unit_receipt.get("reference_to_unit") or {}
            ).items()
        }
        recursive_callsite_units = {
            int(unit_index)
            for unit_index, unit in enumerate(
                compilation_unit_receipt.get("units") or ()
            )
            if bool(unit.get("recursive"))
        }
        callsite_planning_started = time.monotonic()
        callsite_planning_visits: dict[
            tuple[str, int, int], int
        ] = {}
        planned_callsite_shells: dict[
            tuple[type, int, int, type], Any
        ] = {}
        callsite_planning_total = 0

        def recursive_unit_backedge(owner, reference):
            owner_reference = owner.process_graph.G.graph.get("function_ref")
            if owner_reference is None:
                return None
            owner_unit = callsite_unit_by_reference.get(int(owner_reference))
            callee_unit = callsite_unit_by_reference.get(int(reference))
            if (
                owner_unit is None
                or owner_unit != callee_unit
                or owner_unit not in recursive_callsite_units
            ):
                return None
            return int(owner_unit)

        def plan_callsites(owner, active_references=()):
            nonlocal callsite_planning_total
            owner.callsite_function_shells = {}
            for node_id, data in owner.process_graph.G.nodes(data=True):
                attributes = data.get("attributes") or {}
                # A resolved method selector and its invocation both carry
                # the same method-table reference, but only the Call is an
                # activation.  Planning the selector as a second PlanCall
                # creates a marker with no source-call record and publishes
                # the selector's projection values as fictitious results.
                # The authored AST node is the exact distinction; no name,
                # position, or later linker repair is involved.
                #
                # A graph-native node has no authored AST (``expr_obj`` is
                # None): the adjoint builder's ``Call bw_mul`` (op "Call",
                # ``callee_ref`` -> the BACKWARD_RULES graph) in a fused
                # training motion.  Its declared op is its only declaration
                # of an activation.  Requiring an ``ast.Call`` here dropped
                # every backward rule call: no callsite shell, the call
                # result settled ``call-result-unavailable``, its seed input
                # left the ABI and its Indexed gradients left the outputs
                # (left*right VJP: buffer_order (0, 1, 2), grads 7/8 gone).
                # Rule: an AST node is an activation iff it is an
                # ``ast.Call``; a node with no AST iff its declared op is
                # "call".  AST nodes keep the AST rule because an AST graph
                # also holds op "call" nodes that are not activations
                # (short-circuited boolop operands).
                expression = data.get("expr_obj")
                if expression is None:
                    if str(
                        data.get("op") or data.get("type") or ""
                    ).casefold() != "call":
                        continue
                elif not isinstance(expression, ast.Call):
                    continue
                reference = attributes.get("callee_ref")
                if reference is None:
                    reference = attributes.get("method_ref")
                if reference is None and attributes.get("class_ref") is not None:
                    class_record = (
                        owner.process_graph.G.graph.get("class_table") or {}
                    ).get(str(attributes["class_ref"]), {})
                    methods = class_record.get("methods") or {}
                    reference = methods.get(
                        "__new__", methods.get("__init__")
                    )
                    if reference is not None:
                        attributes = dict(attributes)
                        attributes["constructor_ref"] = int(reference)
                        data["attributes"] = attributes
                if reference is None:
                    continue
                reference = int(reference)
                recursive_unit = recursive_unit_backedge(owner, reference)
                activation_mode = (
                    "recursive_scc_backedge"
                    if recursive_unit is not None else "callsite_shell"
                )
                from .identity_concordance import current_identity_book

                callsite_identity = current_identity_book().page(
                    "source_callsite_activation_concordance"
                )
                caller_metadata = owner.process_graph.G.graph
                caller_name = str(
                    caller_metadata.get("function_name") or "<module>"
                )
                method_owner = caller_metadata.get("method_owner")
                caller_identity = str(
                    caller_metadata.get("qualified_name")
                    or (
                        f"{method_owner}.{caller_name}"
                        if method_owner else caller_name
                    )
                )
                planning_key = (
                    caller_identity, int(node_id), reference,
                )
                planning_visit = int(
                    callsite_planning_visits.get(planning_key, 0)
                ) + 1
                callsite_planning_visits[planning_key] = planning_visit
                callsite_planning_total += 1
                if (
                    self.selection_progress is not None
                    and (
                        callsite_planning_total <= 16
                        or callsite_planning_total % 64 == 0
                    )
                ):
                    self.selection_progress(
                        "callsite-plan: "
                        f"total={callsite_planning_total} "
                        f"elapsed_seconds={round(time.monotonic() - callsite_planning_started, 6)} "
                        f"depth={len(active_references)} "
                        f"owner={caller_identity} node={int(node_id)} "
                        f"callee_ref={reference} visit={planning_visit} "
                        f"owner_ref={owner.process_graph.G.graph.get('function_ref')} "
                        f"unit={recursive_unit} mode={activation_mode}"
                    )
                identity_row = (caller_identity, int(node_id))
                activation_fact = (
                    reference,
                    activation_mode,
                    recursive_unit,
                )
                incumbent_reference = callsite_identity.latest(identity_row)
                if (
                    incumbent_reference is not None
                    and tuple(incumbent_reference) != activation_fact
                ):
                    raise ValueError(
                        "source callsite activation concordance "
                        f"disagreement for {identity_row!r}: "
                        f"recorded={incumbent_reference!r}, "
                        f"resolved={activation_fact!r}"
                    )
                if incumbent_reference is None:
                    # The call's ``call_binding`` row (caller scope, call
                    # node) DERIVED from the call node's cell and the
                    # callee's ``function_address`` cell; the activation
                    # record DERIVED from that binding (plan 80, A2.8).  The
                    # recursion unit stays in the fact tuple (R6).
                    binding_cell = _post_call_binding(
                        owner.process_graph, int(node_id), reference,
                        "constructor" if attributes.get("constructor_ref")
                        is not None else "callsite_shell",
                    )
                    _post_callsite_activation(
                        identity_row, activation_fact, binding_cell,
                    )
                if recursive_unit is not None:
                    # The compilation-unit planner has already proven this
                    # edge belongs to one recursive SCC.  Its call remains in
                    # the owner's ProcessGraph and function-reference table;
                    # expanding it into another shell would enumerate every
                    # simple path around the SCC instead of representing the
                    # one authored backedge.
                    continue
                shell_type = self.function_shell_types.get(reference)
                if shell_type is None and self._owns_function_shells:
                    # A receiver class can first become known at this exact
                    # callsite. The initial catalogue closure cannot include
                    # that method yet; extend it once by function identity.
                    table = owner.process_graph.function_table
                    entry = table.entry(reference)
                    if (entry.graph is not None
                            and entry.metadata.get("host_ssa_module") is None
                            and entry.metadata.get("implementation_kind") != "authored-source-fallback"):
                        function_graph = extract_clean_process_subgraph(entry.graph, entry.graph.G)
                        function_graph.G.graph.setdefault("loop_settings", dict(
                            owner.process_graph.G.graph.get("loop_settings") or {}))
                        shell_type = strategize_shell_deployment(
                            function_graph,
                            max_nodes_per_dispatch=self.max_nodes_per_dispatch,
                            _function_table_stack=(id(table),),
                        )
                        self.function_shell_types[reference] = shell_type
                        shared = shell_type(
                            **tuning,
                            _defer_callsite_planning=True,
                        )
                        shared.function_shells = self.function_shells
                        shared._owns_function_shells = False
                        self.function_shells[reference] = shared
                        self.planned_function_references = tuple(sorted({
                            *self.planned_function_references, reference}))
                        self.specialized_dependency_extensions.append({
                            "caller": owner.process_graph.G.graph.get("qualified_name")
                                      or owner.process_graph.G.graph.get("function_name"),
                            "callsite_id": int(node_id),
                            "callee_reference": reference,
                            "provenance": "callsite_specialization",
                        })
                if shell_type is None or reference in active_references:
                    continue
                shell_type = _callsite_specialized_shell_type(
                    owner,
                    node_id,
                    reference,
                    shell_type,
                    self.max_nodes_per_dispatch,
                )
                planned_identity = (
                    type(owner), int(node_id), reference, shell_type,
                )
                incumbent_planned = planned_callsite_shells.get(
                    planned_identity
                )
                if incumbent_planned is not None:
                    owner.callsite_function_shells[node_id] = (
                        incumbent_planned
                    )
                    continue
                planned = shell_type(
                    **tuning,
                    _defer_callsite_planning=True,
                )
                planned.function_shells = self.function_shells
                planned._owns_function_shells = False
                # Publish before descending.  The active-reference rule owns
                # recursive backedges, while this table turns repeated
                # arrivals at one exact specialized callsite into a DAG node
                # instead of unfolding the same subtree once per path.
                planned_callsite_shells[planned_identity] = planned
                owner.callsite_function_shells[node_id] = planned
                plan_callsites(
                    planned,
                    (*active_references, reference),
                )
        activation_roots = (
            tuple(self.activation_root_references)
            if self.runtime_closure_only
            else tuple(self.function_shells)
        )
        for reference in activation_roots:
            function_shell = self.function_shells.get(int(reference))
            if function_shell is None:
                continue
            # Seed the active chain with the activation root itself.  A
            # direct recursive call is therefore a backedge, not one more
            # shell instance; mutual recursion terminates on the first
            # already-active function identity for the same reason.
            plan_callsites(function_shell, (int(reference),))
            (
                function_shell.hierarchy_plan,
                function_shell.hierarchy_value_table,
            ) = assign_hierarchy_ids(
                _build_shell_hierarchy_plan(function_shell),
                shell=function_shell,
            )
        if self._callsite_planning_deferred:
            self.callsite_function_shells = {}
        elif self.runtime_closure_only:
            # The module shell owns the definition catalogue, not another
            # execution of every function body in the source file.  Submitted
            # target shells above are the activation roots in this mode.
            self.callsite_function_shells = {}
        else:
            plan_callsites(self)
        (
            self.hierarchy_plan,
            self.hierarchy_value_table,
        ) = assign_hierarchy_ids(_build_shell_hierarchy_plan(self), shell=self)
        _attach_profiler(
            self,
            self._profiler,
            self.profile_path,
        )
        self.fused_network = None
        self.compiled_shell_callable = None
        self.whole_program_compiled = False
        self.whole_program_device = None
        self.compiled_dispatch_functions = ()
        self.compiled_dispatch_sources = ()
        self.forward_tapes = ()
        self.forward_feed_ids = ()
        self.forward_aggregate_feed_paths = ()
        self.compiled_aggregate_feed_paths = ()
        self.compiled_feed_meta = {}
        self.forward_subgraphs = ()
        self.forward_compilers = ()
        self.forward_region_indices = ()
        self.forward_output_values = ()
        self.forward_region_capture_node_ids = ()
        self.forward_region_planned_capture_ids = ()
        self.forward_region_planned_input_ids = ()
        self.forward_planned_collection_materializations = {}
        # ``Precision.of(tensor, n)`` introduces no independent high-limb
        # storage: its high limb is exactly the authored tensor and the
        # precision pass supplies the remaining zero limbs.  Source
        # specialization records that identity on the concordance.  Install
        # it in the ordinary capture alias table so a region feed is wired to
        # the authored producer instead of escaping as compiler-owned frame
        # storage.
        from .identity_concordance import current_identity_book

        numeric_scope = str(
            self.process_graph.G.graph.get("source_numeric_scope")
            or self.process_graph.G.graph.get("function_name")
            or self.process_graph.G.graph.get("qualified_name")
            or "<module>"
        )
        boundary_page = current_identity_book().page(
            "source_precision_boundary_concordance"
        )
        self.compiled_process_graph_aliases = {
            int(row[1]): int(fact[1])
            for row in boundary_page.rows()
            for fact in (boundary_page.latest(row),)
            if (
                isinstance(row, tuple)
                and len(row) == 2
                and str(row[0]) == numeric_scope
                and int(row[1]) in self.process_graph.G
                and isinstance(fact, tuple)
                and len(fact) == 3
                and str(fact[0]) == "promote"
                and int(fact[1]) in self.process_graph.G
            )
        }
        self.compiled_tapes = ()
        self.compiled_shell_program = None
        self.compiled_region_indices = ()
        self.captured_region_programs = {}
        self.planned_operator_implementations = {}
        # A reference operator (SetAttr/GetAttr) has an unambiguous identity
        # from its own graph node id -- no tensor primitive to correlate,
        # no ambiguity the tape machinery above exists to resolve.  Recorded
        # here, in exact execution order, as each one runs: not a strategy,
        # not a region to score for launch profitability, just what happened,
        # in the order it happened -- the sequential memory operations this
        # represents have real causality that reordering or fusing would
        # break.
        self.reference_operator_sequence = []
        self._capture_invocations = 0
        # Which source parameter each captured input tensor came from, by
        # object identity. Feeds are identified downstream by id(), and
        # without this the name is lost at the boundary -- leaving a
        # descriptor that can say "feed0" but not "cx", so a caller has to
        # guess which array goes where.
        self._capture_input_names = {}
        # The same names, keyed by the resident range instead of by object
        # identity, so a value that was rewrapped can still be recognised.
        self._capture_input_storage = {}
        # Scalar configuration fields carried by structural input objects are
        # compile-time control facts.  Their scoped field paths let hierarchy
        # composition resolve predicates such as ``ctrl.dt_min is not None``
        # without retaining or inspecting the object later.
        self._capture_input_static_fields = {}
        self._captured_return_value_ids = set()
        self._capture_input_type_names = {}
        self._capture_tensor_input_names = set()
        self._captured_source_branches = []
        self._captured_loop_iterations = {}
        self._execute_invocations = 0
        self._discovery_tape_creations = 0
        self._discovery_tape_lowerings = 0
        self._discovery_tape = None
        self._discovery_tape_owner = False
        self._discovery_session = None
        self._discovery_tape_bindings = {}
        self._discovery_tape_complete = False
        self.collection_observations = {}
        self.collection_collapse_report = {}
        self.planned_invocation_slots = 1
        self.coordinator_region_indices = set()
        self.compile_time_region_indices = set()
        self.compile_failures = ()
        self.glsl_sources = ()
        self.composed_loop_sources = {}
        self.composed_loop_artifacts = {}
        self.installed_control_shell = None
        self.composed_shell_artifact = None
        self.composed_shell_specialized_values = {}
        self.shell_control_program = None
        self.hierarchical_control_program = None
        self.hierarchical_captured_region_programs = {}
        self.hierarchical_shell_composed = False
        self.hierarchical_root_value_ids = {}
        self.hierarchical_private_value_ids = {}
        self.hierarchical_capture_value_ids = {}
        self.hierarchical_synthetic_field_paths = {}
        self.hierarchical_program_value_origins = {}
        self.hierarchical_aggregate_redirect_diagnostics = {}
        self.hierarchical_aggregate_candidate_diagnostics = {}
        self.hierarchical_control_binding_diagnostics = {}
        self.hierarchical_composed_closure_iterable_bindings = ()
        self.hierarchical_region_correlations = ()
        self.hierarchical_endpoint_details = {}
        self.hierarchical_value_aliases = {}
        self.hierarchical_root_field_value_ids = {}
        self.hierarchical_effective_value_table = None
        self.hierarchical_compose_failure = None
        self.hierarchical_public_output_ids = set()
        self.hierarchical_terminal_outputs = {}
        self.hierarchical_specialized_values = {}
        self.hierarchy_identity_collapses = ()
        self.hierarchy_identity_rounds = 0
        self.hierarchy_remaining_callsite_ids = None
        self.composed_shell_complete = False
        self.composed_shell_blockers = ()
        self.input_bindings = {}
        self.output_bindings = {}
        self.output_aggregate_bindings = {}
        self.output_loop_aggregate_bindings = {}

    @property
    def ready(self):
        if not self.legacy_fused_network:
            return self.installed_control_shell is not None
        return (
            self.installed_control_shell is not None
            or self.whole_program_compiled
            or self.fused_network is not None
        )

    @property
    def programs(self):
        if self.fused_network is None:
            return ()
        return self.fused_network.programs

    def enable_profiling(self, enabled=True):
        self._profiler.enabled = bool(enabled)
        self.profiling = bool(enabled)
        return self

    def enable_verbose_profiling(self, enabled=True):
        self._profiler.verbose = bool(enabled)
        if enabled:
            self.enable_profiling(True)
        self.verbose_profile = bool(enabled)
        return self

    def trace_report(self, *, limit=None):
        records = tuple(self._profiler.trace_history)
        if limit is not None:
            records = records[-max(0, int(limit)):]
        return records

    def profile_report(self):
        return self._profiler.report()

    def exception_report(self):
        return self.error_buffer.snapshot()

    def exception_lines(self, *, limit=8):
        records = self.error_buffer.snapshot()
        if limit is not None:
            records = records[-max(0, int(limit)):]
        lines = []
        for record in records:
            lines.append(
                f"error {record['sequence']} | {record['path']} | "
                f"{record['phase']} | {record['exception_type']}: "
                f"{record['message']}"
            )
            lines.extend(
                f"  {line}"
                for line in record["traceback"].rstrip().splitlines()
            )
        return tuple(lines)

    def profile_summary(self, *, window=60):
        return self._profiler.summary(window=window)

    def profile_lines(self, *, limit=None, window=1):
        report = self.profile_summary(window=window)
        rows = sorted(
            report["rows"],
            key=lambda row: (
                row["gpu_mean_ms"],
                row["cpu_mean_ms"],
            ),
            reverse=True,
        )
        if limit is not None:
            rows = rows[:max(0, int(limit))]
        lines = [
            f"shell profile {report['frames']} invocation(s) | "
            f"total mean {report['total_mean_ms']:.3f} ms | "
            f"p95 {report['total_p95_ms']:.3f} ms"
        ]
        for row in rows:
            lines.append(
                f"  {row['path']} | {row['section']} | "
                f"{row['label']} | calls {row['calls_mean']:.1f} | "
                f"cpu {row['cpu_mean_ms']:.3f}/"
                f"{row['cpu_p95_ms']:.3f} ms mean/p95 | "
                f"gpu {row['gpu_mean_ms']:.3f}/"
                f"{row['gpu_p95_ms']:.3f} ms mean/p95 | "
                f"dispatches {row['dispatches_mean']:.1f}"
            )
        for row in report["device_rows"]:
            lines.append(
                f"  {row['path']} | logging-ssbo | "
                f"{row['label']}[{row['code']}] | "
                f"events {row['events_mean']:.1f}/invocation | "
                f"payload {row['payload0_mean']:.1f},"
                f"{row['payload1_mean']:.1f}"
            )
        return tuple(lines)

    def program_table_lines(self):
        return _deployment_program_table_lines(self)

    def require_ready(self):
        if not self.ready:
            raise RuntimeError(
                "GLSL deployment is planned but not installed or prepared; "
                "call compile_process_graph() for scheduled AbstractTensor "
                "execution or install fused programs"
            )
        return self

    def call_external(self, reference_or_name, *args, **kwargs):
        return self.external_function_table.invoke(
            reference_or_name,
            *args,
            **kwargs,
        )

    def capture_forward_tapes(
        self,
        dispatch_inputs,
        *,
        device=None,
    ):
        raise RuntimeError(
            "per-region tape capture has been retired: call "
            "capture_fused_programs() once to discover the complete source "
            "program, compile its shell, and destroy the discovery tape"
        )

    def capture_scheduled_forward_tapes(
        self,
        initial_values,
        *,
        device=None,
    ):
        raise RuntimeError(
            "per-region scheduled tape capture has been retired: call "
            "capture_fused_programs() once; the planner's region boundaries "
            "are cut points within that single discovery tape, not separate "
            "executions"
        )

    def coordinate_first_invocation(self, initial_values, *, device=None):
        return _coordinate_scheduled_capture(
            self,
            initial_values,
            device=device,
            capture=True,
        )

    def refresh_hierarchy_plan(self):
        # Freeze hierarchy only after all callsite shells are attached.

        (
            self.hierarchy_plan,
            self.hierarchy_value_table,
        ) = assign_hierarchy_ids(_build_shell_hierarchy_plan(self), shell=self)
        return self.hierarchy_plan

    def prepare_graph_precompile(
        self, *, device=None, reduction_cache=None, progress=None,
        structural_ssa_only: bool = False,
    ):
        # Prepare graph/control IR without observing runtime parameter values.
        #
        # ``reduction_cache`` (a ``ReductionArtifactStore``), when given,
        # persists each region program under its own content key as it is
        # lowered -- an incremental backup of this portion, so an interrupted
        # prepare reloads the regions it already built instead of redoing all
        # of them, and each region becomes an independently addressable
        # catalogue entry. ``progress``, if given, is called with a message
        # string per shell and per region so this otherwise-silent phase is
        # visible.

        def _note(message):
            if progress is not None:
                progress(message)

        hierarchy_started = time.perf_counter()
        _note(
            "aot: refreshing root hierarchy before structural precompile"
        )
        self.refresh_hierarchy_plan()
        _note(
            "aot: refreshed root hierarchy before structural precompile "
            f"in {time.perf_counter() - hierarchy_started:.3f}s"
        )
        if not self.whole_program_compiled:
            self.compile_process_graph(device=device)
        # A selected-entrypoint compile prepares only its proven activation
        # tree.  The canonical whole-source caller explicitly requests the
        # complete definition catalogue: class identity is retained by
        # ClassNavigationTable, while every constructor/method shell supplies
        # the executable contents of that one class record.  Do not infer this
        # from ``runtime_closure_only``: older selected-entrypoint adapters also
        # use broad planning but still have one executable activation tree.
        planned_shells = tuple(_walk_planned_shells(
            self,
            include_function_registry=bool(getattr(
                self, "prepare_complete_catalogue", False
            )),
        ))
        _note(
            f"aot: preparing {len(planned_shells)} planned shell(s)"
        )
        for shell_index, target in enumerate(planned_shells, 1):
            shell_started = time.perf_counter()
            shell_name = str(
                target.process_graph.G.graph.get("function_name") or "?"
            )
            _note(
                f"aot: shell {shell_index}/{len(planned_shells)} "
                f"{shell_name} structural preparation begin"
            )
            # A callsite-specialized shell (below, in
            # ``_specialized_shell_type``) already gets this fold; a
            # top-level planned shell like the one this loop walks never
            # did. That left ``state.height.shape[0]`` -- an ordinary
            # record-field read feeding a loop bound -- structurally
            # unresolved for exactly the whole-program compile path this
            # loop drives, with no error: the loop bound silently reads as
            # 0 and the loop it bounds never runs. Fold it here too, before
            # the hierarchy plan (and the region shapes/order derived from
            # it) gets built from this graph.
            # The module graph retains the definition catalogue with AST
            # identities. It is not a function activation: folding expressions
            # inside those definitions here invalidates its carved regions.
            # Match hierarchy freezing, which folds only function graphs.
            if target.process_graph.G.graph.get("function_name"):
                _fold_callsite_structural_values(target.process_graph)
            target.refresh_hierarchy_plan()
            _note(
                f"aot: shell {shell_index}/{len(planned_shells)} "
                f"{shell_name} hierarchy ready in "
                f"{time.perf_counter() - shell_started:.3f}s"
            )
            complete_regions, region_dependencies = _topological_region_schedule(
                target, range(len(target.dispatch_subgraphs))
            )
            retained_value_ids = {
                int(value_id)
                for item in target.hierarchy_plan.items
                if isinstance(item, PlanClosure)
                and item.name.startswith("region_")
                for line in item.items
                if isinstance(line, PlanLine)
                for value_id in (*line.inputs, *line.outputs)
            }
            considered_reductions = tuple(
                reduction
                for reduction in target.loop_shader_reductions
                if reduction.control_program is not None
            )
            structurally_owned_regions = frozenset(
                int(region)
                for reduction in considered_reductions
                for region in reduction.structurally_owned_region_indices
            ) | frozenset(
                int(region)
                for reduction in target.loop_shader_reductions
                for region in reduction.domain_region_indices
            )
            runtime_regions = tuple(
                int(region) for region in complete_regions
                if int(region) not in structurally_owned_regions
            )
            # A blocked loop (real blockers, e.g. an authored ``raise``) has
            # no control_program and is silently excluded above. Its BODY
            # REGIONS were still compiled as ordinary numerical work and
            # remain in complete_regions -- unless something refuses here,
            # they get scheduled FLAT with no loop wrapping them at all,
            # which does not fail: it silently runs the loop's body exactly
            # once, dropping both the iteration count and the raise
            # statement itself. Measured: a two-line authored program with
            # `if cond: ... else: raise` inside a `for` loop compiled with
            # `complete: True`, no shortfall, no materializer refusal, and
            # ran to a plausible but wrong answer. Refuse instead: a region
            # scheduled without its owning loop is exactly the "lowering
            # must not fabricate control" case the conditional overlay's own
            # duplicate-region guard already refuses for a different cause.
            complete_region_set = frozenset(int(index) for index in complete_regions)
            orphaned = tuple(
                reduction
                for reduction in target.loop_shader_reductions
                if reduction.control_program is None
                and complete_region_set.intersection(
                    map(int, reduction.region_indices)
                )
            )
            if orphaned:
                loop_descriptors = {
                    int(plan.loop.node_id): plan.loop
                    for plan in target.loop_plans
                }

                def opaque_effects(reduction):
                    descriptor = loop_descriptors.get(int(reduction.loop_node_id))
                    if descriptor is None:
                        return ()
                    details = []
                    for effect in descriptor.state_effects:
                        if effect.mode is not LoopStateEffectMode.OPAQUE:
                            continue
                        data = target.process_graph.G.nodes.get(
                            int(effect.effect_node_id), {}
                        )
                        expression = data.get("expr_obj")
                        # SAY WHY THIS ONE IS OPAQUE.
                        # `opaque` is the DEFAULT mode, not a detection:
                        # the classifier recognises a loop-body mutation
                        # only when the state's aggregate_kind is a known
                        # container AND the operator is one of a short
                        # fixed vocabulary. Everything else -- every
                        # `.step()` ever written -- falls through to it.
                        # Reporting the bare word sends a reader looking
                        # for what was detected, when the fact is that
                        # nothing was.
                        attributes = data.get("attributes") or {}
                        receiver_class = (
                            attributes.get("result_class_ref")
                            or attributes.get("class_ref")
                        )
                        operator = str(effect.operator)
                        modelled = operator in _MODELLED_STATE_OPERATORS
                        # THE CALL MAY HAVE RESOLVED AND BEEN REFUSED.  Source
                        # pursuit follows ``self.<field>.<method>`` to the
                        # real method; the extraction contract then decides
                        # whether that method's origin may be ingested.  A
                        # file under no declared root is ``unknown`` and
                        # rejected as ``provenance_not_declared`` -- the call
                        # node carries that exact decision.  Reporting only
                        # "class unresolved" sent readers into the resolver
                        # when the missing fact was a root in the contract.
                        boundary = attributes.get("extraction_contract")
                        boundary = (
                            dict(boundary)
                            if isinstance(boundary, Mapping)
                            and str(boundary.get("action") or "")
                            not in {"ingest_python", ""}
                            else None
                        )
                        if boundary is not None:
                            parameters = dict(
                                boundary.get("parameters") or {}
                            )
                            why = (
                                "the receiver resolved: this call is "
                                f"{boundary.get('identity')!r} defined in "
                                f"{boundary.get('origin')!r}, but the "
                                "extraction contract classified that "
                                f"origin {boundary.get('classification')!r} "
                                f"and chose {boundary.get('action')!r} "
                                f"(rule {boundary.get('rule_id')!r}"
                                + (
                                    f", reason {parameters['reason']!r}"
                                    if parameters.get("reason") else ""
                                )
                                + "), so its source was not pursued and no "
                                "transition can be modelled. Declare that "
                                "one file by module name and path "
                                "(`ExtractionContract.with_sources([(name, "
                                "path)])`) if the program is meant to use "
                                "it; if it is not, the program is using "
                                "something it should not"
                            )
                        elif not modelled and receiver_class is None:
                            why = (
                                "the receiver's class is unresolved AND "
                                f"{operator!r} is outside the modelled "
                                "operator vocabulary, so no state "
                                "transition could be formed"
                            )
                        elif not modelled:
                            why = (
                                f"{operator!r} is outside the modelled "
                                "operator vocabulary "
                                f"({sorted(_MODELLED_STATE_OPERATORS)})"
                            )
                        elif receiver_class is None:
                            why = (
                                f"{operator!r} is modellable, but the "
                                "receiver's class is unresolved so its "
                                "aggregate_kind is unknown"
                            )
                        else:
                            why = (
                                "operator and receiver are both known; the "
                                "mode was not upgraded from the default"
                            )
                        details.append({
                            "effect_node_id": int(effect.effect_node_id),
                            "state_name": str(effect.state_name),
                            "operator": operator,
                            "source": (
                                ast.unparse(expression)
                                if isinstance(expression, ast.AST) else None
                            ),
                            "method_ref": attributes.get("method_ref"),
                            "receiver_class": (
                                None if receiver_class is None
                                else str(receiver_class)
                            ),
                            "why_opaque": why,
                            "has_state_output": (
                                effect.state_output_id is not None
                            ),
                            **(
                                {"extraction_contract": boundary}
                                if boundary is not None else {}
                            ),
                        })
                    return tuple(details)

                owner_reference = target.process_graph.G.graph.get(
                    "function_ref"
                )
                owner_qualified_name = None
                owner_table = getattr(
                    target.process_graph, "function_table", None,
                )
                if owner_reference is not None and owner_table is not None:
                    try:
                        owner_qualified_name = str(
                            owner_table.entry(
                                int(owner_reference)
                            ).qualified_name
                        )
                    except (IndexError, KeyError, TypeError, ValueError):
                        owner_qualified_name = None
                boundaries = tuple({
                    "kind": "loop-control-owner",
                    "loop_node_id": int(reduction.loop_node_id),
                    "region_indices": list(map(
                        int, sorted(reduction.region_indices),
                    )),
                    "blockers": list(map(str, reduction.blockers)),
                    "opaque_effects": list(opaque_effects(reduction)),
                    **({
                        "function_reference": int(owner_reference),
                    } if owner_reference is not None else {}),
                    **({
                        "qualified_name": owner_qualified_name,
                    } if owner_qualified_name is not None else {}),
                } for reduction in orphaned)
                raise CompilationSubdivisionRequired(
                    "a loop's body regions are scheduled but the loop "
                    "itself could not compile, which would otherwise "
                    "silently run the body once with no iteration and no "
                    "effect from its blockers. Note that 'opaque-state-"
                    "effect' is the DEFAULT classification, not a "
                    "detection, and that an opaque effect normally still "
                    "HAS a state output -- what is missing is a model of "
                    "the transition, not a value. Each effect below says "
                    "why it stayed at the default: "
                    + "; ".join(
                        f"loop_node={reduction.loop_node_id} "
                        f"regions={tuple(sorted(reduction.region_indices))} "
                        f"blockers={reduction.blockers} "
                        f"opaque_effects={opaque_effects(reduction)!r}"
                        for reduction in orphaned
                    ),
                    boundaries=boundaries,
                )
            loop_controls = tuple(
                project_control_regions(
                    reduction.control_program,
                    runtime_regions,
                    retained_value_ids=retained_value_ids,
                    preserve_source_loop_carries=True,
                )
                for reduction in considered_reductions
            )
            conditional_controls = _ordinary_conditional_control_programs(
                target.process_graph,
                runtime_regions,
                target.dispatch_subgraphs,
            )
            validations = _validation_control_blocks(target)
            if os.environ.get("TURING_DEBUG_CONTROL_OVERLAY"):
                from .control_source import ConditionalBlock as _DebugConditional
                _debug_conditionals = [
                    (
                        next((
                            int(block.source_node_id)
                            for block in control.root.blocks
                            if isinstance(block, _DebugConditional)
                            and block.source_node_id is not None
                        ), None),
                        control.region_indices,
                    )
                    for control in conditional_controls
                ]
                print(
                    "DEBUGCONTROL "
                    f"fn={target.process_graph.G.graph.get('function_name')} "
                    f"regions={complete_regions} "
                    f"loops={[control.region_indices for control in loop_controls]} "
                    f"conditionals={_debug_conditionals} "
                    f"validations={len(validations)} "
                    f"if_nodes={sum(isinstance(data.get('expr_obj'), ast.If) for _node, data in target.process_graph.G.nodes(data=True))} "
                    f"source_conditional_ids={sorted({int((data.get('attributes') or {}).get('source_conditional_id')) for _node, data in target.process_graph.G.nodes(data=True) if (data.get('attributes') or {}).get('source_conditional_id') is not None})}",
                    file=sys.stderr,
                )
            nesting = _source_control_nesting_hints(
                considered_reductions,
                conditional_controls,
                target.loop_plans,
                target.process_graph,
            )
            shell_control = _overlay_control_or_require_subdivision(
                target.process_graph,
                runtime_regions,
                considered_reductions,
                loop_controls,
                conditional_controls,
                nesting,
                region_dependencies=region_dependencies,
            )
            shell_control = _place_validation_blocks(
                shell_control,
                validations,
                runtime_regions,
                target.dispatch_subgraphs,
            )
            target.shell_control_program = replace(
                shell_control,
                deployment_regions=tuple(dict.fromkeys((
                    *shell_control.deployment_regions,
                    *target.control_deployment_regions,
                ))),
            )
            # The complete-source SSA path consumes ``hierarchy_plan`` and
            # ``shell_control_program`` directly.  It must not manufacture a
            # nominal FusedProgram merely to satisfy the older dual-IR shape:
            # doing so makes a non-projecting compile indistinguishable from
            # numerical capture.  Keep the legacy sentinel only for callers
            # that explicitly request the dual-IR precompile product.
            target.compiled_shell_program = (
                None
                if structural_ssa_only
                else CapturedFusedProgram(
                    FusedProgram(
                        version=1,
                        feeds=set(),
                        steps=[],
                        outputs={},
                        meta={},
                        extras={"kernel_kind": "graph-precompile"},
                    ),
                    {},
                )
            )
            target.compiled_tapes = ()
            target.compiled_region_indices = ()
            # An unfed mutable parameter is left a symbolic SSA input; its
            # region programs come from the planner-isolated dispatch subgraphs
            # via the same structural builder the tape path uses, not from a
            # bespoke plan transcription. Layout/cast ops ride through under
            # their own names for each backend to lower.
            subgraphs = tuple(enumerate(target.dispatch_subgraphs))
            _note(
                f"aot: lowering {len(subgraphs)} region(s) for shell "
                f"{shell_name}"
            )
            # NOTE: regions are NOT deduplicated by sharing program objects
            # here. Structurally-identical regions are distinct *invocations*
            # with their own value-ids and field bindings; sharing one object
            # collides their value-ids and destroys the coordinator's per-region
            # wiring (producer[value_id] would overwrite). The correct IR-level
            # reduction is the kernel/invocation split -- one shared kernel
            # structure plus a per-region binding table -- not object sharing.
            # Byte-identical *kernel files* are still deduped later in the bake.
            region_programs = (
                {}
                if structural_ssa_only
                else {
                    region_index: _structural_region_program_from_subgraph(
                        subgraph
                    )
                    for region_index, subgraph in subgraphs
                }
            )
            target.captured_region_programs = region_programs
            target.compile_time_region_indices = set()
            target.planned_operator_implementations = (
                _build_planned_operator_implementations(
                    target.hierarchy_plan,
                    {},
                    (),
                )
            )
            _note(
                f"aot: shell {shell_index}/{len(planned_shells)} "
                f"{shell_name} structural preparation end in "
                f"{time.perf_counter() - shell_started:.3f}s"
            )
        (
            self.hierarchy_plan,
            self.hierarchy_value_table,
        ) = assign_hierarchy_ids(
            _refresh_hierarchy_control_captures(
                self.hierarchy_plan,
                self,
            ),
            self.hierarchy_value_table,
            shell=self,
        )
        return self

    def capture_fused_programs(
        self,
        initial_values,
        *,
        device=None,
        precompile_only=False,
    ):
        if self._discovery_tape_complete:
            raise RuntimeError(
                "this source program has already consumed its one discovery "
                "tape; execute the installed compiled shell"
            )
        self.refresh_hierarchy_plan()
        if not self.whole_program_compiled:
            self.compile_process_graph(device=device)
        observer_backend = (
            "precompile_observer" if precompile_only else "glsl"
        )
        with _use_scheduled_capture_backend(observer_backend):
            result = _coordinate_scheduled_capture(
                self,
                initial_values,
                device=device,
                capture=True,
            )
        targets = []
        seen = set()
        for target in _walk_planned_shells(
            self, include_function_registry=False
        ):
            identity = id(target)
            if identity in seen or target._discovery_tape is None:
                continue
            seen.add(identity)
            targets.append(target)
        observed_target_ids = {id(target) for target in targets}
        for target in _walk_planned_shells(
            self, include_function_registry=False
        ):
            if id(target) in observed_target_ids:
                continue
            # The program's one discovery sequence did not enter this
            # callsite (for example the grayscale side of a statically color
            # branch).  Record reachability as an explicit empty structural
            # compartment.  Do not execute it, create a tape for it, or infer
            # numerical constants.  If pruning it removes required numerical
            # output, the root artifact's public-output completeness check
            # still rejects the compiled shell.
            target.compiled_shell_program = CapturedFusedProgram(
                FusedProgram(
                    version=1,
                    feeds=set(),
                    steps=[],
                    outputs={},
                    meta={},
                    extras={"kernel_kind": "structural-unreached"},
                ),
                {},
            )
            target.compiled_tapes = ()
            target.compiled_region_indices = ()
            target.compile_time_region_indices = set(
                range(len(target.dispatch_subgraphs))
            )
            target.shell_control_program = ControlProgram(
                SequenceBlock(()),
                (),
            )
        incomplete_shells = []
        # Freeze the one discovery run's wrapper identities before any child
        # shell is lowered and discarded.  This table exists only during
        # compilation so a nested structured argument can be correlated with
        # its caller's SSA result; it is cleared with ``_discovery_session``
        # before the installed shell can execute.
        discovery_capture_objects = {}

        def collect_discovery_capture_objects(value):
            if isinstance(value, dict):
                for item in value.values():
                    collect_discovery_capture_objects(item)
                return
            if isinstance(value, (tuple, list)):
                for item in value:
                    collect_discovery_capture_objects(item)
                return
            if isinstance(value, AbstractTensor):
                discovery_capture_objects[id(value)] = value
                storage_value = getattr(value, "data", None)
                if storage_value is not None:
                    discovery_capture_objects[id(storage_value)] = (
                        storage_value
                    )

        for target in _walk_planned_shells(
            self, include_function_registry=False
        ):
            collect_discovery_capture_objects(
                getattr(target, "captured_values", {})
            )
        if self._discovery_session is not None:
            self._discovery_session["capture_objects"] = (
                discovery_capture_objects
            )
        # Lower the compilation tree's single observation exactly once at its
        # owner.  Child shells below only project their planner cut points from
        # this immutable IR; they never receive the tape and cannot lower it.
        lowered_program = self.compile_discovery_program(
            strict=True,
            emit_glsl=not precompile_only,
        )
        if lowered_program is None:
            raise RuntimeError(
                "the compilation owner did not produce lowered program IR"
            )

        def discard_discovery_payloads(target):
            # Remove observed values while retaining only compiled IR.  This
            # class is generated from an outer triple-quoted source string, so
            # keep this as a comment rather than an embedded docstring.

            target.forward_tapes = ()
            target.forward_feed_ids = ()
            target.forward_aggregate_feed_paths = ()
            target.forward_subgraphs = ()
            target.forward_compilers = ()
            target.forward_region_indices = ()
            target.forward_output_values = ()
            target.forward_region_capture_node_ids = ()
            target.forward_region_planned_capture_ids = ()
            target.forward_region_planned_input_ids = ()
            target.forward_planned_collection_materializations = {}
            target.captured_values = {}
            target.collection_observations = {}
            target._discovery_tape_bindings = {}
            target.last_result = None
            assert not target.forward_tapes
            assert not target.forward_aggregate_feed_paths
            assert not target.captured_values
            assert not target.collection_observations
            assert not target._discovery_tape_bindings
            assert target._discovery_tape_creations == (
                1 if target is self else 0
            )
            assert target._discovery_tape_lowerings == (
                1 if target is self else 0
            )
            assert target.compiled_shell_program is not None

        for target in reversed(targets):
            if target is not self:
                target.compile_discovery_program(
                    strict=True,
                    lowered_program=lowered_program,
                    emit_glsl=not precompile_only,
                )
            # Numerical lowering establishes the final control ABI (uniform
            # bounds, carried aliases, iterable bindings and validations).
            # The hierarchy is a property of the complete source program, not
            # of every suffix subtree.  Rebuilding and re-numbering it once
            # per nested shell repeatedly traverses the same descendants and
            # turns deep ordinary-Python call trees into quadratic compiler
            # work.  Rebuild it exactly once at the discovery owner, which is
            # visited after its children in this reversed traversal and can
            # therefore see every child's final control ABI.
            if target is self:
                hierarchy_refresh_started = time.perf_counter()
                target._profiler.trace(
                    path=target.profile_path,
                    section="hierarchy-refresh",
                    label="begin",
                    fields={},
                )
                (
                    target.hierarchy_plan,
                    target.hierarchy_value_table,
                ) = assign_hierarchy_ids(
                    _refresh_hierarchy_control_captures(
                        target.hierarchy_plan,
                        target,
                    ),
                    target.hierarchy_value_table,
                    shell=target,
                )
                target._profiler.trace(
                    path=target.profile_path,
                    section="hierarchy-refresh",
                    label="complete",
                    fields={
                        "elapsed_ms": round(
                            (
                                time.perf_counter()
                                - hierarchy_refresh_started
                            ) * 1e3,
                            3,
                        ),
                    },
                )
            target.planned_invocation_slots = max(
                1, target._capture_invocations
            )
            target._execute_invocations = 0
            captured = {
                (
                    region_key
                    if not isinstance(region_key, tuple)
                    else region_key[-2]
                )
                for region_key in target.compiled_region_indices
            }
            missing = (
                set(range(len(target.ephemeral_callables))) - captured
            )
            unexplained_missing = missing - set(
                target.coordinator_region_indices
            )
            for region_index in unexplained_missing:
                subgraph = target.dispatch_subgraphs[region_index]
                tensor_outputs = [
                    output_id
                    for output_id in subgraph.G.graph[
                        "deployment_outputs"
                    ]
                    if isinstance(
                        target.captured_values.get(output_id),
                        AbstractTensor,
                    )
                    and int(output_id) not in (
                        target.compiled_process_graph_aliases
                    )
                ]
                if tensor_outputs:
                    raise RuntimeError(
                        "every tensor-producing ProcessGraph callable must "
                        "become one CapturedFusedProgram; "
                        f"{target.process_graph.G.graph.get('function_name', '?')} "
                        f"region {region_index} left tensor outputs "
                        f"{tensor_outputs}; operations="
                        f"{tuple(str(subgraph.G.nodes[node].get('op') or subgraph.G.nodes[node].get('type')) for node in subgraph.G.graph.get('deployment_nodes', ()))!r}; "
                        f"recorded_regions={target.forward_region_indices!r}; "
                        f"compiled_regions={target.compiled_region_indices!r}; "
                        f"shell_id={id(target)}"
                    )
            target.coordinator_region_indices = missing
            target.captured_region_programs = dict(zip(
                target.compiled_region_indices,
                target.compiled_tapes,
            ))
            target.planned_operator_implementations = (
                _build_planned_operator_implementations(
                    target.hierarchy_plan,
                    target.captured_region_programs,
                    target.reference_operator_sequence,
                )
            )
            if target.reference_operator_sequence and not precompile_only:
                # Observation classifies the lowering selected at runtime;
                # it does not define segment membership or order.  Those are
                # already fixed by the ProcessGraph hierarchy plan.
                observed_node_ids = set(
                    map(int, target.reference_operator_sequence)
                )
                ordered_node_ids = [
                    node_id
                    for node_id in _planned_operator_node_ids(
                        target.hierarchy_plan
                    )
                    if node_id in observed_node_ids
                    if node_id in target.process_graph.G
                ]
                missing_planned_nodes = observed_node_ids - set(
                    ordered_node_ids
                )
                if missing_planned_nodes:
                    raise RuntimeError(
                        "observed plain operators are absent from the "
                        "ProcessGraph segment plan: "
                        f"{tuple(sorted(missing_planned_nodes))!r}"
                    )
                def _reference_step_input_ids(node_id: int) -> list[int]:
                    # A field's real identity is ``attribute_slot`` (see
                    # ``bind_target``/``resolve_expression``) -- a
                    # (class_identity, slot) pair grounded in the class's
                    # own declared layout, not a node id.  It carries
                    # through automatically via ``attrs`` below (a full copy
                    # of this node's attributes), so nothing needs pulling in
                    # here beyond the ordinary ``parents`` wiring.
                    node_data = target.process_graph.G.nodes[node_id]
                    return [
                        int(parent)
                        for parent, _role in (node_data.get("parents") or ())
                    ]

                reference_steps = [
                    OpStep(
                        step_id=index,
                        op_name=str(
                            target.process_graph.G.nodes[node_id].get("op")
                            or target.process_graph.G.nodes[node_id].get(
                                "type"
                            )
                        ),
                        input_ids=_reference_step_input_ids(node_id),
                        attrs=dict(
                            target.process_graph.G.nodes[node_id].get(
                                "attributes"
                            )
                            or {}
                        ),
                        result_id=int(node_id),
                    )
                    for index, node_id in enumerate(ordered_node_ids)
                ]
                reference_result_ids = {
                    step.result_id for step in reference_steps
                }
                reference_feeds = {
                    input_id
                    for step in reference_steps
                    for input_id in step.input_ids
                    if input_id not in reference_result_ids
                }
                reference_feed_origins = {
                    feed_id: {"binding_name": binding_name}
                    for feed_id in reference_feeds
                    if (
                        binding_name := (
                            target.process_graph.G.nodes.get(feed_id, {})
                            .get("attributes", {})
                            .get("binding_name")
                        )
                    )
                    is not None
                }
                reference_program = FusedProgram(
                    version=1,
                    feeds=reference_feeds,
                    steps=reference_steps,
                    outputs={},
                    extras={"capture_feed_origins": reference_feed_origins},
                )
                next_index = (
                    max(target.compiled_region_indices, default=-1) + 1
                )
                target.captured_region_programs[next_index] = (
                    CapturedFusedProgram(
                        program=reference_program,
                        feeds={},
                    )
                )
                target.compiled_region_indices = (
                    *target.compiled_region_indices, next_index,
                )
            if target is self and precompile_only:
                # SSA and other IR consumers stop at the Turing precompile
                # boundary.  They need the complete numerical manifest,
                # projected regions, and planned control, but must not build,
                # cache, emit, install, or driver-compile a GLSL artifact.
                target.composed_shell_blockers = (
                    "precompile-only",
                )
                target.composed_shell_complete = False
                discard_discovery_payloads(target)
                continue
            if target is not self:
                # A nested source function is a planner compartment, not a
                # separately deployed shell.  Its local control, projected
                # numerical IR and specialization map are all the root
                # hierarchy consumes.  Auditing a synthetic child ABI walks
                # large source subgraphs, invents "missing shell" blockers,
                # and repeats deployment work that has no runtime meaning.
                target.composed_shell_blockers = (
                    "planner-compartment-awaits-root-composition",
                )
                target.composed_shell_complete = False
                cleanup_started = time.perf_counter()
                target._profiler.trace(
                    path=target.profile_path,
                    section="discovery-cleanup",
                    label="begin",
                    fields={},
                )
                discard_discovery_payloads(target)
                target._profiler.trace(
                    path=target.profile_path,
                    section="discovery-cleanup",
                    label="complete",
                    fields={
                        "elapsed_ms": round(
                            (
                                time.perf_counter() - cleanup_started
                            ) * 1e3,
                            3,
                        ),
                    },
                )
                continue
            if target is self and target.callsite_function_shells:
                hierarchical_artifact = (
                    _build_hierarchical_glsl_artifact(target)
                )
                if hierarchical_artifact is not None:
                    target.composed_shell_artifact = (
                        hierarchical_artifact
                    )
                    target.hierarchical_shell_composed = True
            artifact = target.composed_shell_artifact
            required_inputs = (
                set()
                if artifact is None
                else set(artifact.external_value_ids)
                | set(artifact.uniform_value_ids.values())
            )
            public_inputs = {
                int(node_id)
                for node_id, data in target.process_graph.G.nodes(data=True)
                if data.get("type") == "Input"
                and (
                    data.get("attributes") or {}
                ).get("binding_kind") == "parameter"
                and target.process_graph.G.out_degree(node_id)
            }
            if target.hierarchical_shell_composed:
                public_inputs = {
                    target.hierarchical_root_value_ids.get(
                        node_id, node_id
                    )
                    for node_id in public_inputs
                }
            if artifact is not None:
                # Only specializations originating at the shell's public
                # parameter boundary are runtime constraints.  Specialized
                # lexical constants inside nested closures are compiled facts,
                # not fabricated root inputs.
                required_inputs.update(
                    set(artifact.specialized_values) & public_inputs
                )
            if artifact is not None:
                # A source-level parameter is not necessarily a runtime shader
                # operand.  Parameters such as AbstractTensor's `like=` carry
                # construction/type context and can legitimately disappear
                # during typed lowering.  Requiring an SSBO merely because the
                # source graph retains that informational edge invents runtime
                # work and makes metadata look like executable data.  Only
                # parameters referenced by the emitted artifact belong to its
                # runtime ABI; every such operand must still be covered below.
                artifact_references = set(artifact.slot_value_ids) | set(
                    artifact.uniform_value_ids.values()
                ) | set(artifact.specialized_values)
                public_inputs &= artifact_references
            locally_produced_values = {
                int(value_id)
                for captured in target.captured_region_programs.values()
                for program in (
                    tuple(captured.stages)
                    if captured.stages
                    else (captured.program,)
                )
                for value_id in program.outputs.values()
            }
            public_outputs = set(_declared_output_terminals(
                target.process_graph,
                produced_values=locally_produced_values,
            ).values())
            if target.hierarchical_shell_composed:
                public_outputs = set(
                    target.hierarchical_public_output_ids
                )
            artifact_outputs = (
                set()
                if artifact is None
                else set(artifact.terminal_outputs.values())
            )
            hierarchy_inputs = None
            if (
                artifact is not None
                and target.hierarchical_shell_composed
            ):
                # Nested lexical specializations are compiler-owned constants
                # and are filled from the artifact.  A specialization at the
                # public parameter boundary remains part of the ABI so execute
                # can reject a caller that violates the compiled contract.
                root_runtime_inputs = (
                    required_inputs
                    - set(artifact.specialized_values)
                    | (
                        set(artifact.specialized_values)
                        & public_inputs
                    )
                )
                hierarchy_inputs = {
                    str(data.get("label")): global_id
                    for local_id, data in (
                        target.process_graph.G.nodes(data=True)
                    )
                    if data.get("type") == "Input"
                    and (
                        global_id := (
                            target.hierarchical_root_value_ids.get(
                                int(local_id)
                            )
                        )
                    ) in root_runtime_inputs
                }
                hierarchy_inputs.update({
                    str(path): int(global_id)
                    for path, global_id in (
                        target.hierarchical_root_field_value_ids.items()
                    )
                    if int(global_id) in root_runtime_inputs
                })
            blockers = []
            compile_time_only = (
                target.compiled_shell_program is not None
                and not target.compiled_shell_program.program.steps
                and not target.compiled_shell_program.program.feeds
                and not target.compiled_shell_program.program.outputs
                and (
                    target.compiled_shell_program.program.extras or {}
                ).get("kernel_kind") == "structural"
            )
            if artifact is None and not compile_time_only:
                blockers.append("no-composed-artifact")
            if (
                target.callsite_function_shells
                and not target.hierarchical_shell_composed
                and not compile_time_only
            ):
                remaining_calls = (
                    tuple(sorted(target.callsite_function_shells))
                    if target.hierarchy_remaining_callsite_ids is None
                    else target.hierarchy_remaining_callsite_ids
                )
                blockers.append(
                    "callee-artifacts-not-absorbed:"
                    + ",".join(map(str, remaining_calls))
                    + (
                        ""
                        if target.hierarchical_compose_failure is None
                        else "[compose-failure="
                        f"{target.hierarchical_compose_failure!r}]"
                    )
                )
            if (
                artifact is not None
                and not compile_time_only
                and not public_inputs.issubset(
                required_inputs
                )
            ):
                blockers.append(
                    "public-inputs-not-covered:"
                    + ",".join(map(str, sorted(
                        public_inputs - required_inputs
                    )))
                )
            if (
                artifact is not None
                and not compile_time_only
                and not public_outputs.issubset(
                artifact_outputs
                )
            ):
                blockers.append(
                    "public-outputs-not-covered:"
                    + ",".join(map(str, sorted(
                        public_outputs - artifact_outputs
                    )))
                )
            if (
                hierarchy_inputs is not None
                and not compile_time_only
                and set(hierarchy_inputs.values()) != root_runtime_inputs
            ):
                missing_hierarchy_inputs = (
                    root_runtime_inputs
                    - set(hierarchy_inputs.values())
                )
                parameter_id_map = {
                    int(local_id): int(global_id)
                    for local_id, global_id in (
                        target.hierarchical_root_value_ids.items()
                    )
                    if local_id in target.process_graph.G
                    and target.process_graph.G.nodes[local_id].get("type")
                    == "Input"
                }
                target.hierarchical_unbound_diagnostics = {
                    "missing": tuple(sorted(missing_hierarchy_inputs)),
                    "parameter_ids": parameter_id_map,
                    "field_ids": dict(
                        target.hierarchical_root_field_value_ids
                    ),
                    "private_endpoints": {
                        int(value_id): (
                            target.hierarchical_private_value_ids.get(
                                int(value_id)
                            )
                        )
                        for value_id in missing_hierarchy_inputs
                    },
                    "capture_endpoints": {
                        int(value_id): (
                            target.hierarchical_capture_value_ids.get(
                                int(value_id)
                            )
                        )
                        for value_id in missing_hierarchy_inputs
                    },
                    "endpoint_details": {
                        int(value_id): (
                            target.hierarchical_endpoint_details.get(
                                int(value_id), ()
                            )
                        )
                        for value_id in missing_hierarchy_inputs
                    },
                    "synthetic_fields": {
                        int(value_id): (
                            target.hierarchical_synthetic_field_paths.get(
                                int(value_id)
                            )
                        )
                        for value_id in missing_hierarchy_inputs
                    },
                    "correlated_endpoints": {
                        int(value_id): tuple(
                            (int(closure_id), int(local_id))
                            for closure_id, local_id, global_id
                            in (
                                target.hierarchical_effective_value_table
                                .correlations
                            )
                            if int(global_id) == int(value_id)
                        )
                        for value_id in missing_hierarchy_inputs
                    },
                    "program_origins": {
                        int(value_id): (
                            target.hierarchical_program_value_origins.get(
                                int(value_id), ()
                            )
                        )
                        for value_id in missing_hierarchy_inputs
                    },
                    "aggregate_redirects": {
                        int(value_id): tuple(
                            row
                            for rows in (
                                target
                                .hierarchical_aggregate_redirect_diagnostics
                                .values()
                            )
                            for row in rows
                            if row.get("original") == int(value_id)
                        )
                        for value_id in missing_hierarchy_inputs
                    },
                    "aggregate_closure_rows": {
                        int(closure_id): tuple(
                            target
                            .hierarchical_aggregate_redirect_diagnostics
                            .get(int(closure_id), ())
                        )
                        for closure_id in {
                            int(origin["closure"])
                            for value_id in missing_hierarchy_inputs
                            for origin in (
                                target
                                .hierarchical_program_value_origins.get(
                                    int(value_id), ()
                                )
                            )
                            if "closure" in origin
                        }
                    },
                }
                blockers.append(
                    "hierarchical-inputs-not-root-bound:"
                    + ",".join(map(str, sorted(
                        missing_hierarchy_inputs
                    )))
                    + f"[parameter-map={parameter_id_map!r},"
                    f"field-map={target.hierarchical_root_field_value_ids!r},"
                    "private="
                    f"{target.hierarchical_unbound_diagnostics['private_endpoints']!r},"
                    "capture="
                    f"{target.hierarchical_unbound_diagnostics['capture_endpoints']!r},"
                    "details="
                    f"{target.hierarchical_unbound_diagnostics['endpoint_details']!r},"
                    "fields="
                    f"{target.hierarchical_unbound_diagnostics['synthetic_fields']!r},"
                    "correlated="
                    f"{target.hierarchical_unbound_diagnostics['correlated_endpoints']!r},"
                    "program-origins="
                    f"{target.hierarchical_unbound_diagnostics['program_origins']!r},"
                    "aggregate-redirects="
                    f"{target.hierarchical_unbound_diagnostics['aggregate_redirects']!r},"
                    "aggregate-closure-rows="
                    f"{target.hierarchical_unbound_diagnostics['aggregate_closure_rows']!r},"
                    "aggregate-candidates="
                    f"{target.hierarchical_aggregate_candidate_diagnostics!r},"
                    "control-bindings="
                    f"{target.hierarchical_control_binding_diagnostics!r}]"
                )
            target.composed_shell_blockers = tuple(blockers)
            target.composed_shell_complete = not blockers
            if not target.legacy_fused_network:
                if (
                    target.composed_shell_complete
                    and artifact is not None
                    and (
                        target is self
                        or not target.hierarchical_shell_composed
                    )
                ):
                    # A hierarchically composed child is source owned by its
                    # enclosing shell, not another runtime deployment.
                    # Installing every complete subtree allocates competing
                    # resident lifetimes and turns composition back into
                    # orchestration.  Leaf artifacts remain installable for
                    # differential tests while the parent is incomplete.
                    target.install_composed_control(
                        target.composed_shell_artifact,
                        input_bindings=hierarchy_inputs,
                    )
                elif (
                    target is self
                    and not target.composed_shell_complete
                ):
                    # Nested callsite shells are planner compartments inside
                    # the one source program, not separately deployable
                    # programs.  A method subtree may legitimately expose an
                    # aggregate field or unresolved branch which its caller
                    # later binds/collapses.  Requiring every subtree to have
                    # a standalone public ABI recreates function calls and
                    # defeats whole-shell hierarchy reduction.  Preserve each
                    # child's blockers for diagnostics, but gate installation
                    # only on the compilation root after all descendants have
                    # been absorbed.
                    incomplete_shells.append((
                        target.profile_path,
                        target.composed_shell_blockers,
                    ))

            # A tape is compile-time evidence of operation order, not retained
            # runtime state.  Its nodes hold every concrete tensor produced
            # while revealing the sequence; keeping them after lowering pins
            # all corresponding GL buffers and causes GPU memory to grow until
            # the driver terminates the process.  The compiled programs above
            # contain identities, metadata, and operations only—their captured
            # feed payloads have already been discarded.
            discard_discovery_payloads(target)

        # Structural helpers may legitimately produce no tensor region and
        # therefore never enter ``targets`` above.  They still participated in
        # the one discovery execution and must lose their tape here.  Runtime
        # ownership begins only after every function shell's discovery object
        # has been destroyed.
        for target in _walk_planned_shells(
            self, include_function_registry=False
        ):
            if target._discovery_tape_creations:
                target._discovery_tape = None
                target._discovery_tape_complete = True
                assert target._discovery_tape is None
            elif target._discovery_tape is not None:
                target._discovery_tape = None
                target._discovery_tape_complete = True
                assert target._discovery_tape is None
        if self._discovery_session is not None:
            self._discovery_session.clear()
            self._discovery_session = None

        # Drop the root capture result as well.  Returning it would keep the
        # observed frame tensors alive even though capture exists solely to
        # produce the reusable programs.
        result = None
        gc.collect()
        if incomplete_shells and not self.legacy_fused_network:
            details = "; ".join(
                f"{path}: {', '.join(blockers)}"
                for path, blockers in incomplete_shells[:12]
            )
            if len(incomplete_shells) > 12:
                details += (
                    f"; ... {len(incomplete_shells) - 12} more shells"
                )
            error = RuntimeError(
                "cannot install a truthful complete GLSL shell; "
                f"{len(incomplete_shells)} function shells retain uncovered "
                f"public ABI or control: {details}"
            )
            self._profiler.record_exception(
                error,
                path=self.profile_path,
                phase="shell-completeness",
            )
            raise error
        return None

    def compile_process_graph(self, *, device=None, prepare_ephemerals=False):
        try:
            return _compile_whole_process_graph(
                self, device=device, prepare_ephemerals=prepare_ephemerals
            )
        except Exception as error:
            self._profiler.record_exception(
                error,
                path=self.profile_path,
                phase="compile",
            )
            raise

    def execute_process_graph(self, feeds):
        if not self.whole_program_compiled:
            raise RuntimeError(
                "compile the whole ProcessGraph before executing it"
            )
        return _coordinate_scheduled_capture(
            self,
            feeds,
            device=self.whole_program_device,
            capture=False,
        )

    def compile_discovery_program(
        self,
        *,
        dynamic_scalar_ids=(),
        strict=False,
        lowered_program=None,
        emit_glsl=True,
    ):
        projection_started = time.perf_counter()
        self._profiler.trace(
            path=self.profile_path,
            section="numerical-projection",
            label="begin",
            fields={
                "regions": len(self.dispatch_subgraphs),
                "owner": bool(self._discovery_tape_owner),
                "shared_ir": lowered_program is not None,
                "emit_glsl": bool(emit_glsl),
            },
        )
        if self._discovery_tape is None:
            raise RuntimeError(
                "capture the source shell's discovery tape before compiling"
            )
        if dynamic_scalar_ids and len(dynamic_scalar_ids) != 1:
            raise ValueError(
                "dynamic_scalar_ids must provide one sequence for the one "
                "discovery tape"
            )
        tape = self._discovery_tape
        if lowered_program is None:
            if not self._discovery_tape_owner:
                raise RuntimeError(
                    "a nested shell attempted to lower the compilation "
                    "owner's discovery tape"
                )
            if self._discovery_tape_lowerings:
                raise RuntimeError(
                    "the source program's discovery tape has already been "
                    "lowered; planner work must use compiled_shell_program"
                )
            self._discovery_tape_lowerings += 1
            if self._discovery_tape_lowerings != 1:
                raise RuntimeError(
                    "one source program lowered its discovery tape more than "
                    "once"
                )
        if not tape._nodes:
            # A structural-only shell still consumed its single discovery
            # observation, but there is no numerical program to lower or
            # fabricate.  Preserve an explicit empty IR manifest so cleanup
            # can prove the tape was consumed exactly once.
            self.compiled_shell_program = CapturedFusedProgram(
                FusedProgram(
                    version=1,
                    feeds=set(),
                    steps=[],
                    outputs={},
                    meta={},
                    extras={"kernel_kind": "structural"},
                ),
                {},
            )
            self.compiled_tapes = ()
            self.compiled_region_indices = ()
            self.compile_failures = ()
            self.glsl_sources = ()
            self.compile_time_region_indices = set(
                range(len(self.dispatch_subgraphs))
            )
            self.shell_control_program = ControlProgram(
                SequenceBlock(()),
                (),
            )
            return self
        compiled = []
        compiled_region_indices = []
        failures = []
        sources = []
        # Projection is a cut through the one observed program IR.  Never
        # substitute every planned region when a shell contributed no cut
        # points: that fabricates executions (especially beneath lazy
        # generator boundaries) and attempts to infer feeds from an example
        # that never ran.  The planner still owns those regions and the
        # completeness audit must report them until generalized control
        # lowering makes them reachable.
        subgraphs = tuple(self.forward_subgraphs)
        region_indices = tuple(self.forward_region_indices)
        captured_outputs = tuple(
            self.forward_output_values
            or (None,) * len(subgraphs)
        )
        if not (
            len(self.forward_feed_ids)
            == len(subgraphs)
            == len(region_indices)
            == len(captured_outputs)
        ):
            raise RuntimeError(
                "discovery cut-point tables have inconsistent lengths; "
                "the tape is singular while region metadata is per cut point; "
                f"shell={self.profile_path!r}; "
                f"feeds={len(self.forward_feed_ids)} "
                f"subgraphs={len(subgraphs)} "
                f"regions={len(region_indices)} "
                f"outputs={len(captured_outputs)}"
            )

        if lowered_program is not None and not subgraphs:
            # This child contributed no numerical cut point to the program's
            # one observation.  It is a structural compartment, not an owner
            # of the root numerical manifest.  Preserve an explicit empty
            # artifact so hierarchy composition can erase it without
            # pretending it lowered or executed another program.
            self.compiled_shell_program = CapturedFusedProgram(
                FusedProgram(
                    version=1,
                    feeds=set(),
                    steps=[],
                    outputs={},
                    meta={},
                    extras={"kernel_kind": "structural"},
                ),
                {},
            )
            self.compiled_tapes = ()
            self.compiled_region_indices = ()
            self.compile_failures = ()
            self.glsl_sources = ()
            self.compile_time_region_indices = set(
                range(len(self.dispatch_subgraphs))
            )
            self.shell_control_program = ControlProgram(
                SequenceBlock(()),
                (),
            )
            self._profiler.trace(
                path=self.profile_path,
                section="numerical-projection",
                label="structural complete",
                fields={
                    "elapsed_ms": round(
                        (time.perf_counter() - projection_started) * 1e3,
                        3,
                    ),
                },
            )
            return lowered_program

        # Lower the complete observed numerical program exactly once.  Nothing
        # below this call receives ``tape``: region programs are dependency
        # projections of this backend-neutral IR, never repeated tape solves.
        try:
            whole = lowered_program
            if whole is None:
                whole = compile_recorded_fused_tape(
                    tape,
                    dynamic_scalar_ids=(
                        tuple(dynamic_scalar_ids[0])
                        if dynamic_scalar_ids
                        else ()
                    ),
                )
            whole = _collapse_planned_collection_materializations(
                whole,
                self.forward_planned_collection_materializations,
            )
            # Discovery observations are diagnostic samples, not compiler
            # identity.  In particular, matching a stack's transient operand
            # IDs to observed loop items must never erase the stack producer
            # and alias its tensor result to the resident collection.  The
            # planner-owned mapping above rewires only the producer's inputs
            # and deliberately preserves its distinct result ID.
            self.collection_collapse_report = {
                int(result_id): {
                    "collection_id": int(collection_id),
                    "producer_preserved": True,
                }
                for result_id, collection_id in (
                    self.forward_planned_collection_materializations.items()
                )
            }
        except ValueError as error:
            if strict:
                raise RuntimeError(
                    "the source shell's one discovery tape did not lower to "
                    "one complete FusedProgram"
                ) from error
            self.compile_failures = ({
                "region_index": None,
                "outputs": (),
                "unsupported_ops": (),
                "reason": str(error),
            },)
            return self

        whole.program.extras = {
            **(whole.program.extras or {}),
            "capture_feed_origins": {
                feed_id: {
                    "op": (
                        None
                        if tape._nodes.get(feed_id) is None
                        else str(tape._nodes[feed_id].op)
                    ),
                    # The source parameter this feed came from, recorded
                    # when the graph's Input nodes were bound. A feed is a
                    # trace root and has no tape node, so the old lookup
                    # through tape params could only ever return empty.
                    "binding_name": _resolve_binding_name(self, whole, feed_id),
                    "parameter_names": (
                        tuple(sorted(
                            tape._nodes[feed_id].ctx.get("params") or {}
                        ))
                        if tape._nodes.get(feed_id) is not None
                        else ()
                    ),
                }
                for feed_id in whole.program.feeds
            },
        }
        boundary_aliases = {}
        for feed_map in self.forward_feed_ids:
            boundary_aliases.update(feed_map)
        self.compiled_feed_meta = {
            int(graph_id): meta
            for feed_map in self.forward_feed_ids
            for transient_id, graph_id in feed_map.items()
            if (
                meta := (whole.program.meta or {}).get(
                    int(transient_id)
                )
            ) is not None
        }
        boundary_aliases = _capture_feed_aliases(
            whole,
            boundary_aliases,
        )
        # Keep occurrence identities unique in the whole-shell IR.  A loop
        # revisits the same ProcessGraph node, so globally replacing every
        # observed result with that graph ID would create several producers
        # for one value and corrupt dependency/shape selection.  Stable graph
        # IDs are assigned only after each planner cut point is projected.
        self.compiled_shell_program = _remap_captured_all_ids(whole, {})
        # Aggregate leaf paths are identities derived while lowering the one
        # discovery tape.  Freeze only those integer/path facts into compiled
        # IR before capture payload disposal.  The tape, tensor wrappers and
        # observed values remain forbidden after this point; hierarchical
        # composition must not rediscover them or run Python again.
        compiled_feed_values = {
            int(capture_id): value
            for capture_id, value in whole.feeds.items()
        }
        capture_objects = {}

        def collect_capture_objects(value):
            if isinstance(value, dict):
                for item in value.values():
                    collect_capture_objects(item)
                return
            if isinstance(value, (tuple, list)):
                for item in value:
                    collect_capture_objects(item)
                return
            if (
                isinstance(value, AbstractTensor)
                or _capture_storage_identity(value) is not None
            ):
                capture_objects[id(value)] = value
                if isinstance(value, AbstractTensor):
                    collect_capture_objects(getattr(value, "data", None))

        for node in tape._nodes.values():
            collect_capture_objects(node.ctx.get("result"))
            collect_capture_objects(node.ctx.get("inputs", ()))
            collect_capture_objects(node.ctx.get("params") or {})
        discovery_owner = (
            None
            if self._discovery_session is None
            else self._discovery_session.get("owner")
        )
        if discovery_owner is not None:
            for planned_shell in _walk_planned_shells(
                discovery_owner, include_function_registry=False
            ):
                collect_capture_objects(
                    getattr(planned_shell, "captured_values", {})
                )
        capture_objects.update(compiled_feed_values)
        if self._discovery_session is not None:
            capture_objects.update(
                self._discovery_session.get("capture_objects") or {}
            )
        self.compiled_aggregate_feed_paths = tuple(
            tuple(dict.fromkeys(
                (
                    int(alias_id),
                    int(graph_input_id),
                    tuple(path),
                )
                for capture_id, (
                    graph_input_id,
                    path,
                    storage,
                ) in aggregate_map.items()
                for alias_id in (
                    int(capture_id),
                    *(
                        int(candidate_id)
                        for candidate_id, candidate
                        in capture_objects.items()
                        if (
                            storage is not None
                            and _capture_storage_identity(candidate) == storage
                        )
                    ),
                )
            ))
            for aggregate_map in self.forward_aggregate_feed_paths
        )

        def is_parameter_aggregate_origin(
            graph_input_id: int,
            visiting=None,
        ) -> bool:
            visiting = set() if visiting is None else set(visiting)
            graph_input_id = int(graph_input_id)
            if graph_input_id in visiting or graph_input_id not in self.process_graph.G:
                return False
            visiting.add(graph_input_id)
            data = self.process_graph.G.nodes[graph_input_id]
            attributes = data.get("attributes") or {}
            if data.get("type") == "Input":
                return attributes.get("binding_kind") == "parameter"
            expression = data.get("expr_obj")
            function_expression = getattr(expression, "func", None)
            if not (
                type(expression).__name__ == "Call"
                and type(function_expression).__name__ == "Name"
                and getattr(function_expression, "id", None)
                in {"tuple", "list"}
            ):
                return False
            source = next((
                int(parent)
                for parent, role in (data.get("parents") or ())
                if str(role) in {"arg", "args", "arg:0", "arg0"}
            ), None)
            return (
                source is not None
                and is_parameter_aggregate_origin(source, visiting)
            )

        # Convert the once-lowered program into forward SSA ownership facts
        # before region projection.  The AST planner has already decided the
        # logic and the region boundaries; the lowered IR now supplies the
        # exact numerical edges between those regions.
        #
        # Do not reconstruct these edges by inspecting captured Python
        # wrappers, object identity, storage identity, values, or autograd
        # ancestry.  A backend may legitimately replace a wrapper while
        # retaining the same SSA value, and storage may be shared or reused.
        # Both cases made real producers appear terminal and then left their
        # consumers as unbound shell inputs.  Once FusedProgram exists, its
        # integer input/result IDs are the sole numerical dependency
        # authority.  Forward capture has finished revealing primitive
        # operations at this point and is never asked to rediscover logic.
        ssa_consumers: dict[int, set[int]] = {}
        whole_result_ids: set[int] = set()
        for execution_program in whole.execution_programs:
            for step in execution_program.steps:
                consumer_id = int(step.result_id)
                whole_result_ids.add(consumer_id)
                ssa_consumers.setdefault(consumer_id, set())
                for producer_id in step.input_ids:
                    ssa_consumers.setdefault(
                        int(producer_id), set()
                    ).add(consumer_id)

        # Equal typed literals are a mathematical identity already stated by
        # the ProcessGraph itself.  AbstractTensor may materialize one backend
        # constant primitive and reuse it at several source occurrences.  Use
        # the lowest existing ProcessGraph node ID as their canonical wire;
        # never invent an ID and never inspect a captured runtime value.
        endpoint_identity: dict[int, int] = {}
        literal_ids: dict[tuple[type, Any], int] = {}
        attribute_ids: dict[tuple[int, str], int] = {}
        for node_id, data in sorted(
            self.process_graph.G.nodes(data=True),
            key=lambda item: int(item[0]),
        ):
            node_id = int(node_id)
            if data.get("type") == "Attribute":
                parents = tuple(
                    int(parent)
                    for parent, role in (data.get("parents") or ())
                    if str(role) == "value"
                )
                expression = data.get("expr_obj")
                attribute = getattr(expression, "attr", None)
                if len(parents) == 1 and attribute is not None:
                    parent = endpoint_identity.get(
                        parents[0], parents[0]
                    )
                    key = (parent, str(attribute))
                    endpoint_identity[node_id] = (
                        attribute_ids.setdefault(key, node_id)
                    )
                continue
            if data.get("type") not in {"Constant", "Const", "const"}:
                continue
            try:
                literal = _constant_value(data)
            except KeyError:
                continue
            if not isinstance(
                literal, (type(None), bool, int, float, str, bytes)
            ):
                continue
            key = (type(literal), literal)
            canonical = literal_ids.setdefault(key, node_id)
            endpoint_identity[node_id] = canonical

        for (
            feed_ids,
            aggregate_feed_paths,
            region_capture_node_ids,
            region_planned_capture_ids,
            region_planned_input_ids,
            subgraph,
            region_index,
            captured_output_values,
        ) in zip(
            self.forward_feed_ids,
            self.compiled_aggregate_feed_paths,
            self.forward_region_capture_node_ids,
            self.forward_region_planned_capture_ids,
            self.forward_region_planned_input_ids,
            subgraphs,
            region_indices,
            captured_outputs,
        ):
            output_ids = tuple(
                subgraph.G.graph["deployment_outputs"]
            )
            parameter_aggregate_capture_ids = tuple(
                int(capture_id)
                for capture_id, input_id, _path in aggregate_feed_paths
                if is_parameter_aggregate_origin(input_id)
            )
            owned_result_ids = tuple(dict.fromkeys(
                int(value_id)
                for value_id in region_capture_node_ids
                if (
                    int(value_id) in tape._nodes
                    and int(value_id) in whole_result_ids
                )
            ))
            if not owned_result_ids:
                continue
            owned_set = set(owned_result_ids)
            # Generated region code reports this correspondence at the exact
            # planned operation invocation.  It is tab-A-to-slot-B wiring:
            # no wrapper/storage/value comparison and no reconstruction from
            # tape topology is involved.
            direct_output_ids = {
                int(capture_id): int(graph_node_id)
                for graph_node_id, capture_ids
                in region_planned_capture_ids
                if int(graph_node_id) in output_ids
                for capture_id in capture_ids
            }
            # Apply ProcessGraph identities only after the planner-owned
            # region has been projected from the complete primitive program.
            # Local node IDs are scoped to a function/closure and therefore
            # cannot safely be substituted into the whole discovery capture:
            # unrelated compartments routinely reuse the same integers.
            #
            # The positional rows and result rows below are emitted at the
            # exact generated ProcessGraph operation.  They are the complete
            # tab-A-to-slot-B correspondence for this compartment.  Remapping
            # both sides together also keeps literal tensor constructors and
            # intermediate results as real producers instead of fabricating
            # their ProcessGraph IDs as external shell inputs.
            planned_region_id_map: dict[int, int] = {}

            def compiled_alias_identity(graph_id: int) -> int:
                current = int(graph_id)
                seen = set()
                while (
                    current not in seen
                    and current in self.compiled_process_graph_aliases
                ):
                    seen.add(current)
                    current = int(
                        self.compiled_process_graph_aliases[current]
                    )
                return current

            def add_planned_id(primitive_id: int, graph_id: int) -> None:
                primitive_id = int(primitive_id)
                graph_id = compiled_alias_identity(graph_id)
                graph_id = endpoint_identity.get(
                    int(graph_id), int(graph_id)
                )
                previous = planned_region_id_map.get(
                    primitive_id, graph_id
                )
                if previous != graph_id:
                    raise RuntimeError(
                        "one primitive occurrence was assigned to two "
                        "ProcessGraph endpoints in region "
                        f"{region_index}: primitive={primitive_id}, "
                        f"first={previous}, second={graph_id}, "
                        f"function={self.process_graph.G.graph.get('function_name', '?')!r}, "
                        f"primitive_node={tape._nodes.get(primitive_id)!r}, "
                        f"capture_rows={region_planned_capture_ids!r}, "
                        f"input_rows={region_planned_input_ids!r}, "
                        f"first_node={dict(self.process_graph.G.nodes[previous]) if previous in self.process_graph.G else None!r}, "
                        f"second_node={dict(self.process_graph.G.nodes[graph_id]) if graph_id in self.process_graph.G else None!r}"
                    )
                planned_region_id_map[primitive_id] = graph_id

            for graph_node_id, capture_ids in region_planned_capture_ids:
                for capture_id in capture_ids:
                    add_planned_id(capture_id, graph_node_id)
            for _result_capture_id, positional_inputs in (
                region_planned_input_ids
            ):
                for primitive_id, graph_id in positional_inputs:
                    previous = planned_region_id_map.get(int(primitive_id))
                    previous_node = (
                        self.process_graph.G.nodes[int(previous)]
                        if previous is not None
                        and int(previous) in self.process_graph.G
                        else {}
                    )
                    if (
                        int(primitive_id) == int(_result_capture_id)
                        and previous is not None
                        and int(previous) != int(graph_id)
                        and previous_node.get("type")
                        not in {"Store", "IndexedStore"}
                    ):
                        # Object-keyed capture represents an identity adapter
                        # with the same transient ID at its input and result.
                        # Keep that transient ID wired to the real input and
                        # publish the planned result endpoint as an SSA alias.
                        self.compiled_process_graph_aliases[
                            int(previous)
                        ] = int(graph_id)
                        planned_region_id_map[int(primitive_id)] = int(
                            graph_id
                        )
                        continue
                    add_planned_id(primitive_id, graph_id)
            live_result_ids = tuple(
                value_id
                for value_id in owned_result_ids
                if (
                    value_id in direct_output_ids
                    or not ssa_consumers.get(value_id)
                    or any(
                        consumer_id not in owned_set
                        for consumer_id in ssa_consumers.get(value_id, ())
                    )
                )
            )
            if not live_result_ids:
                # This region's numerical definitions are dead within its
                # own AST compartment.  It needs no shader region, and no
                # observed payload or host fallback is retained.
                continue
            try:
                captured = _project_captured_program(
                    whole,
                    output_ids=live_result_ids,
                    boundary_ids=tuple(dict.fromkeys((
                        *feed_ids,
                        *parameter_aggregate_capture_ids,
                    ))),
                    # AST/ProcessGraph planning has already assigned logic and
                    # scope before the tape is lowered.  The tape contributes
                    # only the tensor operations observed while this region
                    # executed; it is not allowed to walk backward into an
                    # operation owned by a caller, callee, sibling, or loop
                    # compartment merely because lazy materialization joined
                    # their backend expressions.  Repeated occurrences are
                    # accumulated into this one region-owned set above, so
                    # planner-owned loops remain complete without allowing
                    # forward capture to rediscover control flow.
                    allowed_result_ids=tuple(region_capture_node_ids),
                )
                captured = _remap_captured_program(
                    captured,
                    feed_ids={
                        **{
                            transient_id: graph_id
                            for transient_id, graph_id
                            in boundary_aliases.items()
                            if (
                                transient_id in captured.program.feeds
                                or transient_id in feed_ids
                            )
                        },
                        # Exact planned correspondences override legacy
                        # boundary aliases.  The latter remain temporarily
                        # below for untouched aggregate paths, but may never
                        # override an explicit ProcessGraph endpoint.
                        **planned_region_id_map,
                    },
                    output_ids=tuple(
                        direct_output_ids.get(value_id, value_id)
                        for value_id in live_result_ids
                    ),
                )
                canonical_literals = set(literal_ids.values())
                for program in captured.execution_programs:
                    metadata = program.meta or {}
                    for literal_id in canonical_literals:
                        meta = metadata.get(int(literal_id))
                        if meta is None:
                            continue
                        metadata[int(literal_id)] = Meta(
                            (),
                            meta.dtype,
                            meta.device,
                        )
            except ValueError as error:
                if strict:
                    raise RuntimeError(
                        "ProcessGraph numerical region projection "
                        f"{self.process_graph.G.graph.get('function_name', '?')} "
                        f"region {region_index} did not project to one "
                        f"CapturedFusedProgram; operations="
                        f"{tuple(str(subgraph.G.nodes[node].get('op') or subgraph.G.nodes[node].get('type')) for node in subgraph.G.graph.get('deployment_nodes', ()))!r}; "
                        "owned_results="
                        f"{owned_result_ids!r}; live_results="
                        f"{live_result_ids!r}; direct_outputs="
                        f"{tuple(sorted(direct_output_ids.items()))!r}"
                    ) from error
                failures.append({
                    "region_index": region_index,
                    "outputs": tuple(
                        output_ids
                    ),
                    "unsupported_ops": (),
                    "reason": str(error),
                })
                continue
            compiled.append(captured)
            compiled_region_indices.append(region_index)
            if not (
                self._discovery_tape_owner
                and not self.callsite_function_shells
            ):
                # This projected region belongs to a compartment inside a
                # larger source shell.  Keep the CapturedFusedProgram IR for
                # hierarchical composition, but do not emit a throwaway
                # standalone shader.  The one root artifact emits every
                # snippet in its real control/dependency context and is the
                # authoritative GLSL validation point.
                continue
            if not emit_glsl:
                continue
            try:
                sources.append(compile_captured_fused_program(captured))
            except Exception as error:
                stage_shapes = tuple(
                    {
                        "steps": tuple(
                            (
                                step.op_name,
                                tuple(step.input_ids),
                                int(step.result_id),
                            )
                            for step in stage.steps
                        ),
                        "feeds": {
                            int(value_id): (
                                None
                                if (stage.meta or {}).get(value_id) is None
                                else tuple(
                                    (stage.meta or {})[value_id].shape or ()
                                )
                            )
                            for value_id in stage.feeds
                        },
                        "outputs": {
                            name: (
                                int(value_id),
                                None
                                if (stage.meta or {}).get(value_id) is None
                                else tuple(
                                    (stage.meta or {})[value_id].shape or ()
                                ),
                            )
                            for name, value_id in stage.outputs.items()
                        },
                    }
                    for stage in captured.execution_programs
                )
                raise RuntimeError(
                    "failed to compile captured GLSL source for ProcessGraph "
                    f"{self.process_graph.G.graph.get('function_name', '?')} "
                    f"region {region_index}; operations="
                    f"{tuple(str(subgraph.G.nodes[node].get('op') or subgraph.G.nodes[node].get('type')) for node in subgraph.G.graph.get('deployment_nodes', ()))!r}; "
                    f"projected_stages={stage_shapes!r}"
                ) from error
        self.compiled_tapes = tuple(compiled)
        self.compiled_region_indices = tuple(compiled_region_indices)
        self.compile_failures = tuple(failures)
        self.glsl_sources = tuple(sources)
        self._profiler.trace(
            path=self.profile_path,
            section="numerical-projection",
            label="complete",
            fields={
                "compiled_regions": len(compiled_region_indices),
                "elapsed_ms": round(
                    (time.perf_counter() - projection_started) * 1e3,
                    3,
                ),
            },
        )
        if self._discovery_tape_owner:
            self._discovery_session["lowered_program"] = whole
        by_region = {
            int(
                region_index
                if not isinstance(region_index, tuple)
                else region_index[-2]
            ): captured
            for region_index, captured in zip(
                compiled_region_indices,
                compiled,
            )
        }
        composed_loops = {}
        for reduction in self.loop_shader_reductions:
            if not emit_glsl:
                continue
            if not reduction.collapsible or reduction.control_program is None:
                continue
            missing = tuple(
                region_index
                for region_index in reduction.region_indices
                if region_index not in by_region
            )
            if missing:
                self._profiler.trace(
                    path=self.profile_path,
                    section="control-lowering",
                    label=f"loop {reduction.loop_node_id} not composed",
                    fields={
                        "reason": "regions have no captured GLSL program",
                        "regions": missing,
                    },
                )
                continue
            loop_regions = {
                region_index: by_region[region_index]
                for region_index in reduction.region_indices
            }
            available_loop_values = {
                int(value_id)
                for captured in loop_regions.values()
                for program in (
                    captured.program,
                    *tuple(captured.stages),
                )
                for value_id in (
                    *tuple(program.feeds),
                    *tuple(program.outputs.values()),
                    *(step.result_id for step in program.steps),
                )
            }
            deferred_sources = tuple(
                int(source_id)
                for source_id, _collection_id, _induction, _start
                in reduction.control_program.collection_bindings
                if int(source_id) not in available_loop_values
            )
            if deferred_sources:
                self._profiler.trace(
                    path=self.profile_path,
                    section="control-lowering",
                    label=(
                        f"loop {reduction.loop_node_id} deferred "
                        "to hierarchy"
                    ),
                    fields={
                        "collection_sources": deferred_sources,
                    },
                )
                continue
            artifact = build_control_shader_artifact(
                reduction.control_program,
                loop_regions,
                value_meta=_captured_storage_meta(self.captured_values),
                instrumentation=self._profiler.verbose,
            )
            composed_loops[reduction.loop_node_id] = artifact.source
            self.composed_loop_artifacts[reduction.loop_node_id] = artifact
        self.composed_loop_sources = composed_loops
        # A ProcessGraph deployment region is not automatically a runtime
        # shader.  Shape/type predicates and other structural calculations can
        # be fully resolved while discovering the program and produce no
        # tensor output.  Requiring such a region to fabricate an SSBO result
        # prevents the real numerical regions from composing and resurrects a
        # host coordinator for work that no longer exists.
        #
        # Only tensor-producing regions enter the shader schedule.  The
        # capture_fused_programs completeness audit below independently rejects
        # any omitted region that actually has a tensor output, so this cannot
        # hide failed numerical lowering.
        complete_regions, region_dependencies = _topological_region_schedule(
            self,
            (
                region_index
                for region_index in range(len(self.dispatch_subgraphs))
                if region_index in by_region
            ),
        )
        self.compile_time_region_indices = (
            set(range(len(self.dispatch_subgraphs))) - set(complete_regions)
        )
        retained_value_ids = {
            int(value_id)
            for captured in by_region.values()
            for program in (
                tuple(captured.stages)
                if captured.stages
                else (captured.program,)
            )
            for value_id in (
                *ordered_feed_ids(program),
                *tuple(program.outputs.values()),
            )
        }
        all_loops_lowered = all(
            not reduction.region_indices
            or (
                reduction.control_program is not None
                and (
                    not emit_glsl
                    or reduction.collapsible
                )
            )
            for reduction in self.loop_shader_reductions
        )
        if by_region and all_loops_lowered:
            control_compose_started = time.perf_counter()
            self._profiler.trace(
                path=self.profile_path,
                section="control-compose",
                label="overlay begin",
                fields={
                    "regions": len(complete_regions),
                    "loops": len(self.loop_shader_reductions),
                },
            )
            considered_reductions = tuple(
                reduction
                for reduction in self.loop_shader_reductions
                if reduction.control_program is not None
                and (
                    not emit_glsl
                    or reduction.collapsible
                )
            )
            structurally_owned_regions = frozenset(
                int(region)
                for reduction in considered_reductions
                for region in reduction.structurally_owned_region_indices
            ) | frozenset(
                int(region)
                for reduction in self.loop_shader_reductions
                for region in reduction.domain_region_indices
            )
            runtime_regions = tuple(
                int(region) for region in complete_regions
                if int(region) not in structurally_owned_regions
            )
            conditional_controls = _ordinary_conditional_control_programs(
                self.process_graph,
                runtime_regions,
                self.dispatch_subgraphs,
            )
            loop_controls = tuple(
                project_control_regions(
                    reduction.control_program,
                    runtime_regions,
                    retained_value_ids=retained_value_ids,
                    preserve_source_loop_carries=True,
                )
                for reduction in considered_reductions
            )
            nesting = _source_control_nesting_hints(
                considered_reductions,
                conditional_controls,
                self.loop_plans,
                self.process_graph,
            )
            shell_control = _overlay_control_or_require_subdivision(
                self.process_graph,
                runtime_regions,
                considered_reductions,
                loop_controls,
                conditional_controls,
                nesting,
                region_dependencies=region_dependencies,
            )
            shell_control = replace(
                shell_control,
                deployment_regions=tuple(dict.fromkeys((
                    *shell_control.deployment_regions,
                    *self.control_deployment_regions,
                ))),
            )
            validations = _validation_control_blocks(self)
            shell_control = _place_validation_blocks(
                shell_control,
                validations,
                runtime_regions,
                self.dispatch_subgraphs,
            )
            shell_control = project_control_regions(
                shell_control,
                runtime_regions,
                retained_value_ids=retained_value_ids,
            )
            if self.compiled_process_graph_aliases:
                shell_control = replace(
                    shell_control,
                    value_aliases=tuple(dict.fromkeys((
                        *shell_control.value_aliases,
                        *(
                            (int(alias), int(owner))
                            for alias, owner in
                            self.compiled_process_graph_aliases.items()
                        ),
                    ))),
                )
            self.shell_control_program = shell_control
            self._profiler.trace(
                path=self.profile_path,
                section="control-compose",
                label="overlay complete",
                fields={
                    "elapsed_ms": round(
                        (
                            time.perf_counter()
                            - control_compose_started
                        ) * 1e3,
                        3,
                    ),
                },
            )
            produced_values = {
                int(value_id)
                for captured in by_region.values()
                for program in (
                    tuple(captured.stages)
                    if captured.stages
                    else (captured.program,)
                )
                for value_id in program.outputs.values()
            }
            declared_terminals = _declared_output_terminals(
                self.process_graph,
                produced_values=produced_values,
            )
            used_value_ids = {
                int(value_id)
                for captured in by_region.values()
                for program in (
                    tuple(captured.stages)
                    if captured.stages
                    else (captured.program,)
                )
                for value_id in (
                    *program.feeds,
                    *program.outputs.values(),
                )
            }
            specialized_values = {
                int(node_id): self.captured_values[node_id]
                for node_id, data in self.process_graph.G.nodes(data=True)
                if data.get("type") == "Input"
                and str(data.get("label")) not in (
                    self.process_graph.G.graph.get(
                        "planner_specializations", {}
                    )
                )
                and (
                    data.get("attributes") or {}
                ).get("value_kind") == "scalar"
                and node_id not in used_value_ids
                and node_id not in {
                    int(uniform.value_id)
                    for uniform in shell_control.uniforms
                }
                and node_id in self.captured_values
            }
            self.composed_shell_specialized_values = dict(
                specialized_values
            )
            # A child callsite is an IR compartment awaiting absorption into
            # its source program's root hierarchy.  Finalizing it here would
            # generate a shader that is never deployed, then repeat that work
            # for every ancestor.  Only a standalone leaf source is complete
            # at this level; a root with callees is finalized once by
            # _build_hierarchical_glsl_artifact after every compartment has
            # supplied control and numerical IR.
            if (
                emit_glsl
                and self._discovery_tape_owner
                and not self.callsite_function_shells
            ):
                artifact_started = time.perf_counter()
                self._profiler.trace(
                    path=self.profile_path,
                    section="control-compose",
                    label="artifact begin",
                    fields={
                        "regions": len(complete_regions),
                        "terminals": len(declared_terminals),
                    },
                )
                self.composed_shell_artifact = build_control_shader_artifact(
                    shell_control,
                    {
                        region_index: by_region[region_index]
                        for region_index in complete_regions
                    },
                    value_meta=_captured_storage_meta(self.captured_values),
                    instrumentation=self._profiler.verbose,
                    terminal_outputs=(
                        declared_terminals or None
                    ),
                    specialized_values=specialized_values,
                    device_resident=(self.shell_language == "glsl"),
                )
                self._profiler.trace(
                    path=self.profile_path,
                    section="control-compose",
                    label="artifact complete",
                    fields={
                        "elapsed_ms": round(
                            (
                                time.perf_counter() - artifact_started
                            ) * 1e3,
                            3,
                        ),
                    },
                )
        return whole

    def install_composed_control(
        self,
        artifact,
        *,
        input_bindings=None,
        output_bindings=None,
    ):
        if self.installed_control_shell is not None:
            self.installed_control_shell.release()
        try:
            installed = InstalledGLSLControlShell(artifact)
        except Exception as error:
            raise RuntimeError(
                "failed to install composed GLSL closure "
                f"{self.profile_path}"
            ) from error
        public_parameter_ids = {
            int(node_id)
            for node_id, data in self.process_graph.G.nodes(data=True)
            if data.get("type") == "Input"
            and (
                data.get("attributes") or {}
            ).get("binding_kind") == "parameter"
            and self.process_graph.G.out_degree(node_id)
        }
        if self.hierarchical_shell_composed:
            public_parameter_ids = {
                self.hierarchical_root_value_ids.get(node_id, node_id)
                for node_id in public_parameter_ids
            }
        specialized_inputs = (
            set(artifact.specialized_values) & public_parameter_ids
        )
        inferred_inputs = {
            (
                str(self.process_graph.G.nodes[value_id].get("label"))
                if value_id in self.process_graph.G
                and self.process_graph.G.nodes[value_id].get("label")
                else f"value_{value_id}"
            ): value_id
            for value_id in (
                *artifact.external_value_ids,
                *artifact.uniform_value_ids.values(),
                *specialized_inputs,
            )
        }
        self.input_bindings = dict(
            inferred_inputs if input_bindings is None else input_bindings
        )
        required_inputs = (
            set(artifact.external_value_ids)
            | (
                set(artifact.uniform_value_ids.values())
                - set(artifact.specialized_values)
            )
            | specialized_inputs
        )
        if set(self.input_bindings.values()) != required_inputs:
            installed.release()
            raise ValueError(
                "composed-control input bindings must cover every external "
                "buffer and dynamic control uniform exactly once; "
                f"declared={tuple(sorted(self.input_bindings.values()))!r}; "
                f"required={tuple(sorted(required_inputs))!r}; "
                f"missing={tuple(sorted(required_inputs - set(self.input_bindings.values())))!r}; "
                f"extra={tuple(sorted(set(self.input_bindings.values()) - required_inputs))!r}"
            )
        terminal_by_value = {
            int(value_id): terminal_name
            for terminal_name, value_id
            in artifact.terminal_outputs.items()
        }
        identity_table = (
            self.process_graph.G.graph.get("identity_table") or {}
        )
        inferred_outputs = {}
        for public_name in (
            self.process_graph.G.graph.get("function_outputs") or ()
        ):
            identities = tuple(identity_table.get(public_name, ()))
            if not identities:
                continue
            candidate_values = tuple(
                int(
                    self.hierarchical_root_value_ids.get(
                        int(value_id),
                        int(value_id),
                    )
                )
                if self.hierarchical_shell_composed
                else int(value_id)
                for value_id in reversed(identities)
            )
            # A retained loop can append a structural result port after the
            # numerical value actually published by the artifact.  Select the
            # newest identity that is a real terminal instead of discarding
            # the public name merely because the final structural alias is
            # not itself a shader slot.
            terminal_name = next(
                (
                    terminal_by_value[value_id]
                    for value_id in candidate_values
                    if value_id in terminal_by_value
                ),
                None,
            )
            if terminal_name is not None:
                inferred_outputs[str(public_name)] = terminal_name
        if not inferred_outputs:
            inferred_outputs = {
                name: name for name in artifact.terminal_outputs
            }
        self.output_bindings = dict(
            inferred_outputs if output_bindings is None else output_bindings
        )
        self.output_aggregate_bindings = {}
        self.output_loop_aggregate_bindings = {}
        for public_name in (
            self.process_graph.G.graph.get("function_outputs") or ()
        ):
            identities = tuple(identity_table.get(public_name, ()))
            output_id = int(identities[-1]) if identities else None
            output_attributes = (
                self.process_graph.G.nodes[output_id].get("attributes") or {}
                if output_id is not None
                and output_id in self.process_graph.G
                else {}
            )
            if (
                output_attributes.get("materialization_kind")
                == "retained_loop_aggregate"
            ):
                terminal_names = []
                for leaf_id in output_attributes.get(
                    "materialized_value_ids", ()
                ):
                    value_id = int(leaf_id)
                    if self.hierarchical_shell_composed:
                        value_id = int(
                            self.hierarchical_root_value_ids.get(
                                value_id,
                                value_id,
                            )
                        )
                    terminal_name = terminal_by_value.get(value_id)
                    if terminal_name is None:
                        break
                    terminal_names.append(terminal_name)
                else:
                    if terminal_names:
                        self.output_loop_aggregate_bindings[
                            str(public_name)
                        ] = (
                            int(output_attributes.get(
                                "loop_aggregate_axis", 0
                            )),
                            tuple(terminal_names),
                        )
                        continue
            prefix = f"{public_name}."
            leaves = []
            for terminal_name in artifact.terminal_outputs:
                if not str(terminal_name).startswith(prefix):
                    continue
                path = tuple(
                    int(part) if part.isdigit() else part
                    for part in str(terminal_name)[len(prefix):].split(".")
                )
                leaves.append((path, str(terminal_name)))
            if leaves:
                self.output_aggregate_bindings[str(public_name)] = tuple(
                    leaves
                )
        unknown = (
            set(self.output_bindings.values())
            - set(artifact.terminal_outputs)
        )
        if unknown:
            installed.release()
            raise ValueError(
                "composed-control output bindings name unknown terminals: "
                + ", ".join(sorted(map(str, unknown)))
            )
        self.installed_control_shell = installed
        transient_callables = tuple(self.ephemeral_callables)
        for transient in transient_callables:
            transient.discard_python_callable()
            assert not hasattr(transient, "_callable")
        self.ephemeral_callables = ()
        self.compiled_dispatch_functions = ()
        self.compiled_shell_callable = installed.execute
        assert callable(self.compiled_shell_callable)
        return self

    def compile_forward_tapes(self, *args, **kwargs):
        raise RuntimeError(
            "compile_forward_tapes is retired: a source shell owns one "
            "discovery tape, which compile_discovery_program lowers once"
        )

    def install_fused_programs(
        self,
        programs,
        *,
        fifo_slots=2,
        input_bindings=None,
        output_bindings=None,
    ):
        if not self.legacy_fused_network:
            raise RuntimeError(
                "GLSLFusedProgramNetwork is retired by the "
                "composed-control runtime flag. Install the planner-composed "
                "control artifact; pass legacy_fused_network=True only for "
                "explicit transition diagnostics."
            )
        normalized = tuple(
            getattr(program, "program", program)
            for program in programs
        )
        if not normalized:
            raise ValueError("a deployment needs at least one fused program")
        if self.fused_network is not None:
            self.fused_network.release()
        self.fused_network = GLSLFusedProgramNetwork(
            normalized,
            fifo_slots=fifo_slots,
        )
        external_ids = {
            route.value_id
            for route in self.fused_network.routes
            if route.producer is None
        }
        inferred_inputs = {
            f"value_{value_id}": value_id
            for value_id in external_ids
        }
        for value_id in external_ids:
            if value_id not in self.process_graph.G:
                continue
            label = str(
                self.process_graph.G.nodes[value_id].get("label", "")
            )
            if label and label not in inferred_inputs:
                inferred_inputs[label] = value_id
        self.input_bindings = dict(
            inferred_inputs
            if input_bindings is None
            else input_bindings
        )
        if set(self.input_bindings.values()) != external_ids:
            self.fused_network.release()
            self.fused_network = None
            raise ValueError(
                "input bindings must cover every external fused value "
                "exactly once by value identity"
            )
        terminal_names = {
            route.output_name
            for route in self.fused_network.routes
            if route.consumer is None and route.output_name is not None
        }
        self.output_bindings = dict(
            {name: name for name in terminal_names}
            if output_bindings is None
            else output_bindings
        )
        unknown_outputs = set(self.output_bindings.values()) - terminal_names
        if unknown_outputs:
            self.fused_network.release()
            self.fused_network = None
            raise ValueError(
                "output bindings name unknown fused terminals: "
                + ", ".join(sorted(map(str, unknown_outputs)))
            )
        # ------------------------------------------------------------------
        # NON-NEGOTIABLE COMPILED-SHELL INVARIANT
        #
        # ProcessGraph-derived Python callables exist only to *make* the
        # compiled program.  They are compile-time lowering instruments.  They
        # are not the deployed program, they are not an acceptable fallback,
        # and executing them after compilation is not compiled-shell behavior.
        #
        # Agents have repeatedly violated this invariant by retaining these
        # Python objects, running them through a Python coordinator, setting a
        # flag named ``whole_program_compiled``, and then claiming that the
        # shell was compiled.  That claim is false.  Producing correct output,
        # or dispatching compiled numerical fragments from Python, does not
        # make the controlling shell a compiled program.
        #
        # Anyone who changes this code to bypass, weaken, mock, or remove the
        # deletion and assertions below while continuing to claim
        # compiled-shell execution is lying about what the program does and
        # is ruining the compiler boundary this code exists to enforce.  This
        # has happened repeatedly and it must stop now.
        #
        # The proof is deliberately mechanical:
        #
        #   1. Every transient Python callable is physically discarded.
        #   2. The deployment retains no ephemeral callable collection.
        #   3. A real installed compiled-shell callable exists.
        #
        # Do not replace deletion with ``None`` while retaining another
        # reference.  Do not rename a Python coordinator and store it in
        # ``compiled_shell_callable``.  Do not make the assertions conditional.
        # Fix the compiler if these assertions fail.
        #
        # Likewise, do not evade these assertions earlier in the pipeline by
        # wrapping a tensor algorithm in an opaque host-boundary call.  A
        # Python codec that internally performs thousands of tensor/GLSL
        # operations is not one compiled operation.  Marking it external only
        # blinds the planner, leaves the work eager, and falsely makes the
        # remaining graph look compiled.  The numerical topology must remain
        # visible through planning and lowering; only the true terminal host
        # effect may cross the boundary.
        # ------------------------------------------------------------------
        transient_callables = tuple(self.ephemeral_callables)
        for transient in transient_callables:
            transient.discard_python_callable()
            assert not hasattr(transient, "_callable")
        self.ephemeral_callables = ()
        self.compiled_dispatch_functions = ()
        self.compiled_shell_callable = self.fused_network.execute

        # These assertions are the executable definition of the deployment
        # boundary described above.  A failure means compilation is incomplete;
        # it must never be hidden by falling back to ProcessGraph execution.
        assert not self.ephemeral_callables
        assert callable(self.compiled_shell_callable)
        return self

    def install_compiled_tapes(
        self,
        *,
        fifo_slots=2,
        input_bindings=None,
        output_bindings=None,
    ):
        if not self.compiled_tapes:
            raise RuntimeError("compile forward tapes before installing them")
        return self.install_fused_programs(
            self.compiled_tapes,
            fifo_slots=fifo_slots,
            input_bindings=input_bindings,
            output_bindings=output_bindings,
        )

    def execute(self, feeds):
        if (
            not self.legacy_fused_network
            and self.installed_control_shell is None
        ):
            raise RuntimeError(
                "composed GLSL execution requires the installed compiled "
                "shell; consume the program's one discovery tape with "
                "capture_fused_programs() before runtime"
            )
        if not self.ready:
            self.compile_process_graph()
        self.require_ready()
        if self.installed_control_shell is not None:
            return self.installed_control_shell.execute(feeds)
        if self.whole_program_compiled:
            return self.execute_process_graph(feeds)
        return self.fused_network.execute(feeds)

    def execute_named(self, feeds):
        if (
            not self.legacy_fused_network
            and self.installed_control_shell is None
        ):
            raise RuntimeError(
                "composed GLSL execution requires the installed compiled "
                "shell; consume the program's one discovery tape with "
                "capture_fused_programs() before runtime"
            )
        if not self.ready:
            self.compile_process_graph()
        self.require_ready()
        if self.installed_control_shell is not None:
            unknown = set(feeds) - set(self.input_bindings)
            if unknown:
                raise KeyError(
                    "unknown deployment inputs: "
                    + ", ".join(sorted(map(str, unknown)))
                )
            token = self._profiler.begin_shell(self.profile_path)
            started_ns = time.perf_counter_ns()
            try:
                installed = self.installed_control_shell
                outputs = installed.execute({
                    self.input_bindings[name]: value
                    for name, value in feeds.items()
                })
                gpu_ms = installed.last_gpu_ms
                dispatches = installed.last_dispatches
                self._profiler.record_device_trace(
                    path=self.profile_path,
                    records=installed.last_debug_records,
                    header=installed.last_debug_header,
                )
                streamed = {}
                if installed.artifact.stream_outputs:
                    while True:
                        drained = installed.drain_stream()
                        for item in drained:
                            streamed.setdefault(
                                int(item["stream_id"]),
                                [],
                            ).append(
                                item["values"]
                                if getattr(
                                    item["values"], "nbytes", 0
                                )
                                or not item["words"].size
                                else item["words"].astype(
                                    "uint8",
                                    copy=False,
                                )
                            )
                        if installed.last_stream_status != 1:
                            break
                        if not drained:
                            raise RuntimeError(
                                "compiled GLSL shell suspended for "
                                "downstream capacity without publishing a "
                                "drainable resident prefix"
                            )
                        outputs = installed.resume()
                        gpu_ms += installed.last_gpu_ms
                        dispatches += installed.last_dispatches
                        self._profiler.record_device_trace(
                            path=self.profile_path,
                            records=installed.last_debug_records,
                            header=installed.last_debug_header,
                        )
                self._profiler.record(
                    path=self.profile_path,
                    section="compiled-glsl",
                    label="composed-shell-dispatch",
                    cpu_ms=(
                        time.perf_counter_ns() - started_ns
                    ) / 1e6,
                    gpu_ms=gpu_ms,
                    dispatches=dispatches,
                )
                named_outputs = {
                    public_name: outputs[terminal_name]
                    for public_name, terminal_name
                    in self.output_bindings.items()
                }
                for public_name, leaves in (
                    self.output_aggregate_bindings.items()
                ):
                    tree = {}
                    for path, terminal_name in leaves:
                        current = tree
                        for part in path[:-1]:
                            current = current.setdefault(part, {})
                        current[path[-1]] = outputs[terminal_name]

                    def freeze_aggregate(value):
                        if not isinstance(value, dict):
                            return value
                        if value and all(
                            isinstance(key, int) for key in value
                        ):
                            return tuple(
                                freeze_aggregate(value[index])
                                for index in range(max(value) + 1)
                            )
                        return {
                            key: freeze_aggregate(item)
                            for key, item in value.items()
                        }

                    named_outputs[public_name] = freeze_aggregate(tree)
                for public_name, (
                    axis,
                    terminal_names,
                ) in self.output_loop_aggregate_bindings.items():
                    leaf_values = [
                        outputs[terminal_name]
                        for terminal_name in terminal_names
                    ]
                    leaf_arrays = [
                        (
                            value.numpy()
                            if callable(getattr(value, "numpy", None))
                            else np.asarray(value)
                        )
                        for value in leaf_values
                    ]
                    extents = {
                        int(array.shape[axis])
                        for array in leaf_arrays
                    }
                    if len(extents) != 1:
                        raise RuntimeError(
                            "retained loop aggregate leaves disagree on "
                            f"axis {axis} extent: {tuple(sorted(extents))!r}"
                        )
                    named_outputs[public_name] = tuple(
                        tuple(
                            np.take(array, iteration, axis=axis)
                            for array in leaf_arrays
                        )
                        for iteration in range(next(iter(extents)))
                    )
                if installed.artifact.stream_outputs:
                    for public_name, stream_id in (
                        installed.artifact.stream_outputs.items()
                    ):
                        named_outputs[str(public_name)] = tuple(
                            streamed.get(int(stream_id), ())
                        )
                return named_outputs
            finally:
                self._profiler.end_shell(self.profile_path, token)
        if self.whole_program_compiled:
            result = self.execute_process_graph(feeds)
            values = result if isinstance(result, tuple) else (result,)
            names = tuple(
                self.process_graph.G.graph.get(
                    "function_outputs",
                    (),
                )
            )
            if not names:
                names = tuple(
                    f"result_{index}" for index in range(len(values))
                )
            if len(names) != len(values):
                raise RuntimeError(
                    "compiled ProcessGraph output names do not match "
                    "the returned values"
                )
            return dict(zip(names, values))
        missing = set(feeds) - set(self.input_bindings)
        if missing:
            raise KeyError(
                "unknown deployment inputs: "
                + ", ".join(sorted(map(str, missing)))
            )
        outputs = self.execute({
            self.input_bindings[name]: value
            for name, value in feeds.items()
        })
        return {
            public_name: outputs[network_name]
            for public_name, network_name in self.output_bindings.items()
        }

    def submit(self, feeds):
        if not self.inputs.publish(feeds):
            raise BufferError("GLSL deployment input FIFO is full")
        return self

    def run_pending(self):
        if self.outputs.unread >= self.outputs.slots:
            raise BufferError("GLSL deployment output FIFO is full")
        available, feeds = self.inputs.consume()
        if not available:
            return False
        outputs = self.execute_named(feeds)
        if not self.outputs.publish(outputs):
            raise BufferError("GLSL deployment output FIFO is full")
        return True

    def receive(self):
        return self.outputs.consume()

    def __call__(self, feeds):
        return self.execute_named(feeds)

    def release(self):
        if self.installed_control_shell is not None:
            self.installed_control_shell.release()
            self.installed_control_shell = None
        if self.fused_network is not None:
            self.fused_network.release()
            self.fused_network = None
        if self._owns_function_shells:
            for shell in _walk_planned_shells(self):
                if shell is self:
                    continue
                shell.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.release()
        return False


def strategize_shell_deployment(
    graph: Any,
    *,
    # Was 256, which was a GLSL dispatch-sizing constraint: a shader had to
    # be split into chunks a driver would accept. That is no longer what the
    # number means now that emission goes SPIR-V -> GLSL, and for a target
    # that wants one flat program it was actively harmful -- a program past
    # the bound was split into regions, so a caller asking for a flat
    # unrolled loop silently got a retained one instead.
    max_nodes_per_dispatch: int = 65536,
    backend: str | None = None,
    remove_loops: bool | None = None,
    unroll_limit: int | None = None,
    schedule_preference: str | None = None,
    runtime_closure_only: bool = False,
    _function_table_stack: tuple[int, ...] = (),
    _selection_progress: Any = None,
    _selection_trace: dict[str, Any] | None = None,
) -> type:
    """Build a stateful shell around the graph's flat dispatch schedule.

    This is the compilation choke point: every backend -- c, python, glsl,
    fortran, webgl, webgpu -- funnels its ProcessGraph through this one
    control-planning stage before any backend-specific emission happens.
    The name is generic on purpose; nothing below is GLSL-specific.

    ``backend`` only tags the loop-composition capability profile below --
    it does not change GLSL emission, which stays gated behind
    ``capture_fused_programs(precompile_only=...)`` regardless of this
    argument. Callers that only want the FusedProgram/ControlProgram/SSA
    stages (any backend) should pass ``precompile_only=True`` there and
    never reach GLSL-specific emission at all.

    ``remove_loops`` disables native loop retention, so
    ``evaporate_unrolled_loops`` unrolls every discovered loop into a flat
    instruction sequence instead of a real ``LoopBlock``.

    ``unroll_limit`` is the largest trip count that may be unrolled.  It is a
    representation-choice threshold, never an execution bound.  A loop above
    the threshold is retained with its complete source domain; if a target
    cannot lower retained control, compilation must fail visibly rather than
    dropping iterations.
    """

    from .identity_concordance import shape_scope_is_variant

    if _selection_trace is None:
        _selection_trace = {
            "started_at": time.monotonic(),
            "sequence": 0,
            "dropped_events": 0,
            "visits": {},
            "events": [],
        }
    trace = _selection_trace
    graph_metadata = graph.G.graph
    function_identity = str(
        graph_metadata.get("qualified_name")
        or graph_metadata.get("function_name")
        or graph_metadata.get("function_ref")
        or "<source-catalogue>"
    )
    visit_key = "|".join((
        function_identity,
        str(graph.G.number_of_nodes()),
        str(graph.G.number_of_edges()),
    ))
    visits = trace.setdefault("visits", {})
    visits[visit_key] = int(visits.get(visit_key, 0)) + 1

    def selection_event(stage: str, state: str, **facts: Any) -> None:
        trace["sequence"] = int(trace.get("sequence", 0)) + 1
        event = {
            "sequence": int(trace["sequence"]),
            "elapsed_seconds": round(
                time.monotonic() - float(trace["started_at"]), 6
            ),
            "function": function_identity,
            "function_depth": len(_function_table_stack),
            "visit": int(visits[visit_key]),
            "stage": str(stage),
            "state": str(state),
            "nodes": int(graph.G.number_of_nodes()),
            "edges": int(graph.G.number_of_edges()),
            **facts,
        }
        events = trace.setdefault("events", [])
        events.append(event)
        if len(events) > 512:
            del events[: len(events) - 512]
            trace["dropped_events"] = int(
                trace.get("dropped_events", 0)
            ) + 1
        # Keep the evidence on every graph which participates in selection.
        # The shared object means the root and extracted function graphs expose
        # one ordered account rather than unrelated print-only diagnostics.
        graph_metadata["deployment_selection_trace"] = trace
        if _selection_progress is not None:
            detail = " ".join(
                f"{name}={value}"
                for name, value in facts.items()
            )
            _selection_progress(
                "deployment-select: "
                f"function={function_identity} "
                f"depth={len(_function_table_stack)} "
                f"visit={visits[visit_key]} "
                f"stage={stage} state={state} "
                f"nodes={event['nodes']} edges={event['edges']}"
                + (f" {detail}" if detail else "")
            )

    def selection_phase(name: str, action: Any) -> Any:
        phase_started = time.monotonic()
        selection_event(name, "begin")
        try:
            result = action()
        except BaseException as exc:
            selection_event(
                name,
                "failed",
                duration_seconds=round(time.monotonic() - phase_started, 6),
                exception=type(exc).__name__,
            )
            raise
        selection_event(
            name,
            "end",
            duration_seconds=round(time.monotonic() - phase_started, 6),
        )
        return result

    selection_event("deployment", "begin", repeated_visit=visits[visit_key] > 1)
    selection_phase(
        "lower-python-scalar-intrinsics",
        lambda: _lower_python_scalar_intrinsics(graph),
    )

    # Region planning may legitimately erase structural AST nodes after their
    # dataflow has been reduced.  Capture authored branch/guard identity before
    # any such planning so every extracted function shell retains the control
    # catalogue needed for later SSA overlay.
    selection_phase(
        "retain-source-control-records",
        lambda: _retain_source_control_records(graph.G),
    )
    selection_phase(
        "retain-source-sequence-mutation-records",
        lambda: _retain_source_sequence_mutation_records(graph.G),
    )

    # Inherit whatever the compilation asked for, then record it so the
    # function shells planned from subgraphs of this one see the same thing.
    # Without this a nested function -- which is where a loop usually lives,
    # since the entrypoint is a thin wrapper -- silently planned with the
    # defaults, and a caller asking for a flat program got a retained loop.
    inherited = dict(graph.G.graph.get("loop_settings") or {})
    backend = inherited.get("backend", "glsl") if backend is None else backend
    remove_loops = (
        inherited.get("remove_loops", False) if remove_loops is None
        else remove_loops
    )
    if unroll_limit is None:
        from .work_contract import active_contract

        unroll_limit = inherited.get(
            "unroll_limit", active_contract().loops.unroll_limit,
        )
    schedule_preference = (
        inherited.get("schedule_preference", "alap")
        if schedule_preference is None else schedule_preference
    )
    schedule_preference = str(schedule_preference).lower()
    if schedule_preference not in {"asap", "alap"}:
        raise ValueError(
            "deployment schedule preference must be 'asap' or 'alap'"
        )
    unroll_limit = max(1, min(int(unroll_limit), MAX_UNROLL_LIMIT))
    graph.G.graph["loop_settings"] = {
        "backend": backend,
        "remove_loops": bool(remove_loops),
        "unroll_limit": int(unroll_limit),
        "schedule_preference": schedule_preference,
    }
    graph.G.graph["deployment_schedule_preference"] = schedule_preference
    runtime_catalogue = bool(
        runtime_closure_only
        and not graph.G.graph.get("function_name")
    )
    specialization_dependency_regions = (
        (graph.G.graph.get("map_ir") or {}).get("dependency_regions") or {}
    )
    runtime_specialization_references = (
        frozenset(map(int, specialization_dependency_regions["runtime"]))
        if runtime_closure_only
        and "runtime" in specialization_dependency_regions
        else None
    )

    selection_phase(
        "resolve-bound-function-references",
        lambda: _resolve_bound_function_references(graph),
    )
    selection_phase(
        "propagate-callsite-planner-specializations",
        lambda: _propagate_callsite_planner_specializations(graph),
    )
    # A concrete function graph always has literal structural facts that can
    # remove or alias nodes even when there is no callsite-specific scalar
    # specialization. Perform that fixed point before its dispatch regions are
    # carved. The file/module catalogue graph has no ``function_name`` and is
    # intentionally excluded: folding that shared discovery graph mutates the
    # function-table sources before their individual specialization facts are
    # applied. ``prepare_graph_precompile`` repeats the fold defensively after
    # callsite shells are attached; by then it must be idempotent.
    if graph.G.graph.get("function_name"):
        selection_phase(
            "fold-callsite-structural-values-before-tensors",
            lambda: _fold_callsite_structural_values(graph),
        )
    if not _function_table_stack:
        selection_phase(
            "propagate-callsite-tensor-specializations",
            lambda: _propagate_callsite_tensor_specializations(
                graph,
                runtime_references=runtime_specialization_references,
                _progress=lambda facts: selection_event(
                    "callsite-tensor-specialization",
                    "progress",
                    **facts,
                ),
            ),
        )
    elif graph.G.graph.get("planner_tensor_descriptors") and (
        shape_scope_is_variant(graph)
    ):
        # A callsite-specialized copy (a child shell) has its own call
        # edges, and the values its callees receive derive from its own
        # specialized arguments: the root pass cannot have settled them.
        # Settle this copy's call edges alone -- no function-table graph is
        # entered or narrowed -- so each callee reached from it is
        # specialized for ITS arguments.
        selection_phase(
            "propagate-callsite-tensor-specializations-specialized-copy",
            lambda: _propagate_callsite_tensor_specializations(
                graph,
                runtime_references=frozenset(),
                publish_to_function_table=False,
                _progress=lambda facts: selection_event(
                    "callsite-tensor-specialization",
                    "progress",
                    **facts,
                ),
            ),
        )
    else:
        # The root pass settled the entire proven runtime closure directly on
        # the shared function-table entries before any clean function graph is
        # extracted. Re-running that whole-table fixed point from every child
        # shell is redundant and can let one child mutate sibling catalogue
        # graphs after their shells have already been planned.
        selection_event(
            "propagate-callsite-tensor-specializations",
            "inherited",
        )
    if graph.G.graph.get("function_name"):
        # Tensor descriptors learned across call edges can make another local
        # shape-only expression decidable.  The structural fixed point is
        # idempotent and still executes no numerical source code.
        selection_phase(
            "fold-callsite-structural-values-after-tensors",
            lambda: _fold_callsite_structural_values(graph),
        )
        # Branch folding can remove synthesized call-result projections while
        # retaining the aggregate producer selected by the surviving arm.
        # The fold itself now repairs that ledger invariant before returning,
        # including for recursive callsite specializations.
        selection_phase(
            "refresh-specialized-python-precision",
            lambda: _refresh_specialized_python_precision(graph),
        )
    # Loop discovery snapshots state effects into immutable descriptors. Link
    # methods before that snapshot so an authored record mutation is owned by
    # its compiled callee instead of remaining a second opaque callsite effect.
    selection_phase(
        "resolve-grounded-method-references-before-loops",
        lambda: _resolve_grounded_method_references(graph),
    )
    canonical_value_ids = bool(
        graph.G.graph.get("canonical_value_ids")
    )
    loop_composer = LoopComposer(
        LoopBackendCapabilities(
            backend=backend,
            native_for=not remove_loops,
            native_while=not remove_loops,
            dynamic_bounds=not remove_loops,
            kpn=False,
            unroll_limit=int(unroll_limit),
        )
    )
    discovered_loop_plans = selection_phase(
        "discover-loops",
        lambda: (
            loop_composer.discover(graph)
            if canonical_value_ids and not runtime_catalogue
            else ()
        ),
    )
    evaporated_loop_plans = selection_phase(
        "evaporate-unrolled-loops",
        lambda: (
            evaporate_unrolled_loops(graph, discovered_loop_plans)
            if discovered_loop_plans else ()
        ),
    )
    if evaporated_loop_plans and (
        graph.G.graph.get("planner_specializations")
        or graph.G.graph.get("planner_tensor_descriptors")
    ):
        # Unrolling turns each induction/tuple component into a literal node.
        # Re-run callsite specialization before the local structural fixed
        # point. A cloned call consumes the fresh literal induction value and
        # owns a fresh callsite id; retaining the pre-unroll dynamic callee
        # would replace an exact aggregate member with private frame storage.
        selection_phase(
            "propagate-callsite-planner-specializations-after-unroll",
            lambda: _propagate_callsite_planner_specializations(graph),
        )
        selection_phase(
            "fold-callsite-structural-values-after-unroll",
            lambda: _fold_callsite_structural_values(graph),
        )
    evaporated_loop_ids = {
        int(plan.loop.node_id) for plan in evaporated_loop_plans
    }
    retained_loop_plans = tuple(
        (
            replace(
                plan,
                strategy=LoopStrategy.NATIVE_SOURCE,
                reason=(
                    "resident sequence mutation requires iterative SSA "
                    "memory effects"
                ),
            )
            if (
                plan.strategy in {LoopStrategy.UNROLL, LoopStrategy.CONSTANT}
                and (
                    bool(plan.loop.control_sites)
                    or any(
                        effect.mode in {LoopStateEffectMode.SEQUENCE_MUTATION,
                                        LoopStateEffectMode.MAPPING_MUTATION}
                        for effect in plan.loop.state_effects
                    )
                    or any(
                        any((graph.G.nodes[node_id].get("attributes") or {}).get(key) is not None
                            for key in ("callee_ref", "method_ref", "constructor_ref"))
                        for node_id in plan.loop.body_nodes if node_id in graph.G
                    )
                )
            )
            else plan
        )
        for plan in discovered_loop_plans
        if int(plan.loop.node_id) not in evaporated_loop_ids
        and int(plan.loop.node_id) in graph.G
    )
    semantic_loop_plans = selection_phase(
        "materialize-loop-semantic-ir",
        lambda: (
            loop_composer.materialize_semantic_ir(
                graph,
                retained_loop_plans,
            )
            if retained_loop_plans else ()
        ),
    )
    loop_plans = selection_phase(
        "materialize-retained-loop-ports",
        lambda: (
            materialize_retained_loop_ports(graph, semantic_loop_plans)
            if semantic_loop_plans else ()
        ),
    )
    function_entry = None
    function_reference = graph.G.graph.get("function_ref")
    function_table = getattr(graph, "function_table", None)
    if function_reference is not None and function_table is not None:
        function_entry = function_table.entry(function_reference)
    process_graph_boundary = (
        None
        if function_entry is None
        else function_entry.metadata.get("process_graph_boundary")
    )
    python_callable = (
        None if function_entry is None else function_entry.python_callable
    )
    selection_phase(
        "resolve-grounded-method-references-after-loops",
        lambda: _resolve_grounded_method_references(graph),
    )
    selection_phase(
        "resolve-grounded-tensor-operations",
        lambda: _resolve_grounded_tensor_operations(graph),
    )
    reference_tables = selection_phase(
        "build-shell-reference-tables",
        lambda: build_shell_reference_tables(graph),
    )
    ordered_executable_nodes = selection_phase(
        "dependency-order",
        lambda: _dependency_order(graph),
    )
    is_dispatch_metadata = _dispatch_metadata_node_classifier(graph)
    executable_nodes = tuple(
        node_id
        for node_id in ordered_executable_nodes
        if not is_dispatch_metadata(node_id)
    )
    if runtime_catalogue:
        # The runtime root is the source catalogue and map owner. Execution
        # starts at ``activation_root_references`` in its function shells;
        # _compile_whole_process_graph already treats this shell as a catalogue
        # rather than a second execution of the module. Keep the same boundary
        # here during region selection: planning catalogue syntax as numerical
        # work created hundreds of dispatch subgraphs which no runtime could
        # ever execute.
        selection_event(
            "runtime-catalogue-boundary",
            "applied",
            excluded_executable_nodes=len(executable_nodes),
        )
        executable_nodes = ()
    # Control membership is a semantic partition: numerical work may be
    # reduced freely inside one planner-owned loop body or conditional branch,
    # but never fused across that control boundary.  All other dispatch
    # boundaries are produced by the fixed-point shader-region identities
    # rather than inherited from schedule levels or operator spelling.
    partition_keys = selection_phase(
        "control-partition-keys",
        lambda: _control_partition_keys(
            graph,
            loop_plans,
            executable_nodes,
        ),
    )
    inert_nodes = selection_phase(
        "inert-routing-nodes",
        lambda: _inert_routing_nodes(graph),
    )
    closure_edges, closure_outputs = selection_phase(
        "closure-routing-dependencies",
        lambda: _closure_routing_dependencies(graph),
    )
    # A method may ``return`` a compile-time reference (a type or module, e.g.
    # ``return torch.float32``). That resolves to a _StaticPythonReference, not a
    # runtime value node id -- it is a compile-time constant, not a runtime
    # output -- so it is excluded from the runtime return-liveness set here
    # (int() would fail on it), the same way compile-time references are not
    # runtime roots.
    return_outputs = frozenset(
        int(value_id)
        for value_id in (
            *(
                parent
                for _node_id, data in graph.G.nodes(data=True)
                if isinstance(data.get("expr_obj"), ast.Return)
                for parent, role in (data.get("parents") or ())
                if str(role) in {"result", "value", "operand"}
            ),
            *(graph.G.graph.get("return_value_nodes", {}) or {}).values(),
        )
        if _is_runtime_value_id(value_id) and int(value_id) in graph.G
    )
    recursion_control_nodes = frozenset(
        int(node_id)
        for record in (
            graph.G.graph.get("recursion_table") or {}
        ).values()
        if record.get("control_ir", True)
        for node_id in record.get("control_members", ())
    )
    dispatch_plan = selection_phase(
        "reduce-scheduled-shader-regions",
        lambda: reduce_scheduled_shader_regions(
            graph,
            executable_nodes,
            max_nodes_per_region=max_nodes_per_dispatch,
            partition_keys=partition_keys,
            extra_dependency_edges=closure_edges,
            fusible_node_ids=_shader_fusible_node_ids(
                graph, executable_nodes,
            ),
            indivisible_node_groups=_precision_indivisible_node_groups(
                graph, executable_nodes,
            ),
            control_node_ids=recursion_control_nodes,
            _progress=lambda facts: selection_event(
                "shader-region-fixed-point",
                "progress",
                **facts,
            ),
        ),
    )
    executable_dispatch_nodes = tuple(
        dispatch.node_ids
        for dispatch in dispatch_plan.dispatches
    )
    control_deployment_regions = bind_control_deployments_to_regions(
        graph.G.graph.get("control_deployment_regions", ()),
        executable_dispatch_nodes,
    )
    deployment_region_preferences: dict[int, str] = {}
    for deployment in control_deployment_regions:
        for lane in deployment.lanes:
            for region_index in lane.region_indices:
                previous = deployment_region_preferences.setdefault(
                    int(region_index), deployment.schedule_preference
                )
                if previous != deployment.schedule_preference:
                    raise ValueError(
                        "scheduled region has conflicting deployment "
                        f"preferences: region={region_index}, "
                        f"preferences={(previous, deployment.schedule_preference)!r}"
                    )
    dispatch_subgraphs = selection_phase(
        "extract-dispatch-subgraphs",
        lambda: tuple(
            _dispatch_subgraph(
                graph,
                node_ids,
                required_outputs=frozenset((*closure_outputs, *return_outputs)),
                inert_nodes=inert_nodes,
                schedule_preference=deployment_region_preferences.get(
                    region_index, "asap"
                ),
            )
            for region_index, node_ids in enumerate(executable_dispatch_nodes)
            if node_ids
        ),
    )
    # The region ordinal every later reader uses is the subgraph's position
    # in this tuple; post the rows under that ordinal (plan 80, A2.4).
    for region_index, subgraph in enumerate(dispatch_subgraphs):
        _post_deployment_region(graph, subgraph, int(region_index))
    for subgraph, dispatch in zip(
        dispatch_subgraphs,
        dispatch_plan.dispatches,
    ):
        subgraph.G.graph["rewrite_history"] = dispatch.rewrite_history
    loop_region_indices = {
        plan.loop.node_id: tuple(
            region_index
            for region_index, subgraph in enumerate(dispatch_subgraphs)
            if set(
                subgraph.G.graph.get("deployment_nodes", ())
            ).intersection(plan.loop.body_nodes)
        )
        for plan in loop_plans
    }
    loop_shader_reductions = selection_phase(
        "analyze-shader-loop-reductions",
        lambda: analyze_shader_loop_reductions(
            graph,
            loop_plans,
            executable_dispatch_nodes,
        ),
    )
    deep_compilers = tuple(
        GraphDeepCompiler(
            subgraph,
            dict(abstract_tensor_funcs),
            abstract_tensor_sigs,
            node_observer=_observe_process_graph_node,
        )
        for subgraph in dispatch_subgraphs
    )
    ephemeral_callables = tuple(
        EphemeralProcessGraphCallable(
            subgraph,
            compiler=compiler,
            eager=False,
        )
        for subgraph, compiler in zip(
            dispatch_subgraphs,
            deep_compilers,
        )
    )

    executable_node_ids = {
        node_id
        for node_ids in executable_dispatch_nodes
        for node_id in node_ids
    }
    node_locations = {
        node_id: location
        for node_id, location in dispatch_plan.node_locations.items()
        if node_id in executable_node_ids
    }
    deployment_class = type(
        "ProcessGraphGLSLDeployment",
        (ProcessGraphGLSLDeployment,),
        {
            "process_graph": graph,
            "external_function_table": getattr(
                graph, "external_function_table", None
            ),
            "static_python_bindings": dict(
                getattr(graph, "python_bindings", {}) or {}
            ),
            "dispatch_plan": dispatch_plan,
            "loop_plans": loop_plans,
            "loop_shader_reductions": loop_shader_reductions,
            "control_deployment_regions": control_deployment_regions,
            "loop_region_indices": loop_region_indices,
            "process_graph_boundary": process_graph_boundary,
            "python_callable": staticmethod(python_callable),
            "dispatch_subgraphs": dispatch_subgraphs,
            "deep_compilers": deep_compilers,
            "ephemeral_callables": ephemeral_callables,
            "reference_table_template": reference_tables,
            "source_node_count": graph.G.number_of_nodes(),
            "primitive_count": sum(
                len(node_ids) for node_ids in executable_dispatch_nodes
            ),
            "loop_count": sum(
                1
                for _node_id, data in graph.G.nodes(data=True)
                if str(data.get("type")) in {"For", "AsyncFor", "While"}
            ),
            "dispatch_count": len(dispatch_subgraphs),
            "max_nodes_per_dispatch": int(max_nodes_per_dispatch),
            "selection_progress": (
                None if _selection_progress is None
                else staticmethod(_selection_progress)
            ),
            "deployment_batches": node_locations,
        },
    )
    function_shell_types = {}
    function_table = getattr(graph, "function_table", None)
    table_identity = id(function_table)
    dependency_regions = (
        (graph.G.graph.get("map_ir") or {}).get("dependency_regions") or {}
    )
    runtime_references = (
        frozenset(map(int, dependency_regions["runtime"]))
        if runtime_closure_only and "runtime" in dependency_regions
        else None
    )
    if (
        function_table is not None
        and table_identity not in _function_table_stack
    ):
        nested_stack = (*_function_table_stack, table_identity)
        for function_index, entry in enumerate(function_table):
            if entry.graph is None:
                continue
            reference = int(entry.reference.address)
            if (
                runtime_references is not None
                and reference not in runtime_references
            ):
                # ``build_map_dependency_regions`` has already proved the
                # strict call closure of every submitted compile target.
                # Map-only definitions stay in FunctionTable/map/class
                # navigation records, but they are not executable members of
                # this program and must not receive deployment shells.  The
                # old all-table expansion turned a 27-function program into
                # 103 function types and thousands of callsite activations,
                # lowering unrelated class methods as though they ran.
                continue
            child_identity = str(
                getattr(entry, "qualified_name", None)
                or getattr(entry, "name", None)
                or reference
            )
            function_started = time.monotonic()
            selection_event(
                "function-shell",
                "begin",
                child_function=child_identity,
                child_reference=reference,
                function_index=int(function_index),
            )
            function_graph = selection_phase(
                "extract-function-subgraph",
                lambda entry=entry: extract_clean_process_subgraph(
                    entry.graph,
                    entry.graph.G,
                ),
            )
            # A function's graph is built independently, not carved out of
            # the caller's, so it does not inherit the compilation's loop
            # settings the way an extracted subgraph does. Seeding them here
            # is what makes unroll_limit/remove_loops mean anything for a
            # callee -- and the callee is where loops usually live, since the
            # entrypoint is a thin wrapper by convention.
            function_graph.G.graph.setdefault(
                "loop_settings",
                dict(graph.G.graph.get("loop_settings") or {}),
            )
            function_shell_types[entry.reference.address] = (
                strategize_shell_deployment(
                    function_graph,
                    max_nodes_per_dispatch=max_nodes_per_dispatch,
                    _function_table_stack=nested_stack,
                    _selection_progress=_selection_progress,
                    _selection_trace=trace,
                )
            )
            selection_event(
                "function-shell",
                "end",
                child_function=child_identity,
                child_reference=reference,
                function_index=int(function_index),
                duration_seconds=round(
                    time.monotonic() - function_started, 6
                ),
            )
    activation_root_references = []
    if runtime_closure_only and function_table is not None:
        for target in graph.G.graph.get("compile_targets", ()):
            try:
                reference = int(
                    function_table.entry(str(target)).reference.address
                )
            except (KeyError, TypeError, ValueError):
                continue
            if reference in function_shell_types:
                activation_root_references.append(reference)
        if not activation_root_references:
            # A caller may provide a precomputed runtime closure without the
            # source-level target names (for example an older checkpoint).
            # In that case retain the complete proven runtime set rather than
            # guessing one privileged root.
            activation_root_references.extend(sorted(function_shell_types))
    deployment_class.function_shell_types = function_shell_types
    deployment_class.runtime_closure_only = bool(runtime_closure_only)
    deployment_class.planned_function_references = tuple(sorted(
        function_shell_types
    ))
    deployment_class.activation_root_references = tuple(dict.fromkeys(
        activation_root_references
    ))
    deployment_class.catalogued_function_references = tuple(sorted(
        int(entry.reference.address)
        for entry in function_table or ()
        if entry.graph is not None
    ))
    deployment_class.catalogue_only_function_references = tuple(sorted(
        set(deployment_class.catalogued_function_references)
        - set(deployment_class.planned_function_references)
    ))
    selection_event(
        "deployment",
        "end",
        dispatches=len(dispatch_subgraphs),
        planned_functions=len(function_shell_types),
    )
    return deployment_class


__all__ = [
    "DeploymentErrorBuffer",
    "DeploymentProfiler",
    "strategize_shell_deployment",
]