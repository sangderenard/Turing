"""Which stage last touches each declared identity-book page: when it goes cold.

``PAGE_LIFECYCLE`` classifies every page ``concordance_declarations`` (and
``identity_concordance``) declares by

* ``written_in`` -- the LATEST compile stage whose code posts to the page,
* ``read_in``    -- the latest stage whose code reads it,
* ``cold_after`` -- the stage after which no code touches the page again; once
  that stage has completed the page may be spilled (``identity_spill``).  A
  write reloads a spilled page exactly as a read does, so ``cold_after`` is the
  later of the two.  ``None``: never spilled (the policy and receipt pages, the
  scope bookkeeping every post consults, and the book's private edge pages,
  whose rows follow the page that owns them).

The stages are the progress layer's (``shell_telemetry.COMPILE_STAGES``) with
one declared refinement: ``"ssa-lowering:functions"``, the per-function lowering
loop that ends at the ``"ssa-program: dispatch region copies released"``
boundary and is the only stage with its own boundary inside ``ssa-lowering``.

HOW IT WAS READ.  By static reference, not by trace.  Every code reference
(a ``Name``/``Attribute`` of the page's constant, alias-aware, or the page's
name as a string constant; comments and docstrings do not count) was mapped to
the function that makes it, and each function to the stage it runs in: whole
modules for the single-stage ones (``graph_express2``: ingestion;
``precompile_to_ssa``, ``hierarchical_plan``, ``loop_composer``: the
per-function lowering loop; the emission and backend modules), function by
function for the multi-stage ones (``fortran_c_shell``,
``identity_concordance``: the late repairs and the compile-tail detector are
``pre-native-repairs``; everything under ``_class_surface_ssa_program`` is
``ssa-lowering``).  The reducer, the planner posts and the callsite
specialization run in several stages and are assigned their LATEST.  A wrong
stage can therefore only make a page spill later than it could (a spilled page
is read back when touched, so a too-early assignment costs a reload, never a
wrong answer).

Readers that walk the whole book do not pin a page: the compile-tail unsourced
detector and the log read every page through ``IdentityBook.borrowed``.  The
book's own ``post`` / ``edges_into`` / ``edges_out_of`` / ``mint_of`` touch the
private edge pages, which follow their owners.

Completed stages by boundary (``BOUNDARY_COMPLETES``):

* ``"ssa-source: topology reduced, program ABI reattached"``: through topology-reduction
* ``"ssa-source: call topology validated"``: through call-topology
* ``"ssa-source: control/operator graph planned"``: through graph-planning
* ``"ssa-program: dispatch region copies released"``: through ssa-lowering:functions
* ``"ssa-source: repository SSA lowered"``, ``"ssa-source: deployment released"``: through ssa-lowering
* ``"compile: end"``: through pre-native-repairs
* emission and build are not lowering boundaries; ``spill_cold_pages(book,
  "build")`` is what a caller that owns them calls when the build is done.

Pages by ``cold_after`` (this table is generated from ``PAGE_LIFECYCLE``; the
counts are pages, not cells):

* never spilled (10)
* source-closure (5)
* ssa-lowering:functions (28)
* ssa-lowering (133)
* pre-native-repairs (37)
* emission (11)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .shell_telemetry import COMPILE_STAGES

#: The stages in order: the progress layer's, plus the per-function lowering
#: loop inside ssa-lowering.
SPILL_STAGES: tuple[str, ...] = tuple(
    item
    for stage in COMPILE_STAGES
    for item in (
        ("ssa-lowering:functions", stage.key) if stage.key == "ssa-lowering"
        else (stage.key,)
    )
)
_INDEX = {key: position for position, key in enumerate(SPILL_STAGES)}

#: Stage-boundary label (``memory_regulation.stage_boundary``) -> the last
#: stage that has COMPLETED when it is reached.
BOUNDARY_COMPLETES: dict[str, str] = {
    "ssa-source: topology reduced, program ABI reattached": "topology-reduction",
    "ssa-source: call topology validated": "call-topology",
    "ssa-source: control/operator graph planned": "graph-planning",
    "ssa-program: dispatch region copies released": "ssa-lowering:functions",
    "ssa-source: repository SSA lowered": "ssa-lowering",
    "ssa-source: deployment released": "ssa-lowering",
    "compile: end": "pre-native-repairs",
}


@dataclass(frozen=True)
class PageLifecycle:
    written_in: str | None
    read_in: str | None
    cold_after: str | None


CLOSURE = "source-closure"
PLANNING = "graph-planning"
FUNCTIONS = "ssa-lowering:functions"
LOWERING = "ssa-lowering"
REPAIRS = "pre-native-repairs"
EMISSION = "emission"

PAGE_LIFECYCLE: dict[str, PageLifecycle] = {
    "abstract_tensor_program_value": PageLifecycle(LOWERING, None, LOWERING),
    "aggregate_ledger": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "aggregate_passthrough_rebinding": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "alias_application_concordance": PageLifecycle(None, LOWERING, LOWERING),
    "annotation_declaration": PageLifecycle(CLOSURE, None, CLOSURE),
    "argument_binding": PageLifecycle(None, REPAIRS, REPAIRS),
    "argument_binding_resolution": PageLifecycle(LOWERING, REPAIRS, REPAIRS),
    "backward_rule_definition": PageLifecycle(None, LOWERING, LOWERING),
    "bounded_fixed_point_round": PageLifecycle(LOWERING, None, LOWERING),
    "call_argument_operand": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "call_binding": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "call_binding_input": PageLifecycle(None, LOWERING, LOWERING),
    "call_edge": PageLifecycle(None, LOWERING, LOWERING),
    "call_input_conversion": PageLifecycle(None, LOWERING, LOWERING),
    "call_input_storage_concordance": PageLifecycle(None, LOWERING, LOWERING),
    "call_link_order_concordance": PageLifecycle(None, LOWERING, LOWERING),
    "call_record": PageLifecycle(None, LOWERING, LOWERING),
    "call_record_pair_concordance": PageLifecycle(None, LOWERING, LOWERING),
    "callable_identity_concordance": PageLifecycle(LOWERING, REPAIRS, REPAIRS),
    "callsite_argument": PageLifecycle(FUNCTIONS, FUNCTIONS, FUNCTIONS),
    "callsite_argument_role": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "callsite_descriptor_reuse": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "callsite_return_member": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "callsite_return_specialization": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "callsite_tensor_result_specialization": PageLifecycle(LOWERING, None, LOWERING),
    "canonical_value": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "carried_port_value": PageLifecycle(FUNCTIONS, FUNCTIONS, FUNCTIONS),
    "carried_snapshot": PageLifecycle(FUNCTIONS, FUNCTIONS, FUNCTIONS),
    "cell_set": PageLifecycle(LOWERING, REPAIRS, REPAIRS),
    "class_declaration": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "class_field_declaration": PageLifecycle(CLOSURE, LOWERING, LOWERING),
    "class_method_declaration": PageLifecycle(CLOSURE, None, CLOSURE),
    "compile_policy": PageLifecycle(LOWERING, LOWERING, None),
    "concordance_dependents": PageLifecycle(None, None, None),
    "concordance_edge": PageLifecycle(None, REPAIRS, None),
    "concordance_mint": PageLifecycle(None, REPAIRS, None),
    "concordance_unsourced": PageLifecycle(None, REPAIRS, None),
    "constant_role": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "consumer_operand": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "contract_demand": PageLifecycle(CLOSURE, LOWERING, LOWERING),
    "control_block": PageLifecycle(PLANNING, FUNCTIONS, FUNCTIONS),
    "control_block_placement": PageLifecycle(PLANNING, FUNCTIONS, FUNCTIONS),
    "control_program": PageLifecycle(PLANNING, EMISSION, EMISSION),
    "control_uniform_dtype": PageLifecycle(FUNCTIONS, FUNCTIONS, FUNCTIONS),
    "control_value_alias": PageLifecycle(FUNCTIONS, FUNCTIONS, FUNCTIONS),
    "control_value_binding": PageLifecycle(FUNCTIONS, EMISSION, EMISSION),
    "control_value_concordance": PageLifecycle(FUNCTIONS, LOWERING, LOWERING),
    "copy_value_shape": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "cross_function_references": PageLifecycle(LOWERING, None, LOWERING),
    "dead_structural_retirement": PageLifecycle(REPAIRS, None, REPAIRS),
    "declared_parameter": PageLifecycle(FUNCTIONS, FUNCTIONS, FUNCTIONS),
    "deployment_region": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "deployment_region_member": PageLifecycle(LOWERING, None, LOWERING),
    "dispatch_store": PageLifecycle(LOWERING, None, LOWERING),
    "emission_artifact": PageLifecycle(EMISSION, EMISSION, EMISSION),
    "emission_function": PageLifecycle(EMISSION, EMISSION, EMISSION),
    "emission_unit": PageLifecycle(EMISSION, EMISSION, EMISSION),
    "entry_record_handle_concordance": PageLifecycle(None, LOWERING, LOWERING),
    "exact_formal_dtype": PageLifecycle(None, None, REPAIRS),
    "exact_region_feed_dtype": PageLifecycle(LOWERING, REPAIRS, REPAIRS),
    "executable_node": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "external_call_lowering": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "external_callsite": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "external_derivative": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "external_function": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "external_function_name": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "external_leaf": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "external_leaf_value": PageLifecycle(LOWERING, None, LOWERING),
    "field_slot_storage_concordance": PageLifecycle(None, FUNCTIONS, FUNCTIONS),
    "formal_actual_concordance": PageLifecycle(None, None, REPAIRS),
    "formal_actual_occurrence_concordance": PageLifecycle(None, REPAIRS, REPAIRS),
    "formal_literal": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "formal_shape": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "frame_lease_link": PageLifecycle(None, LOWERING, LOWERING),
    "function_address": PageLifecycle(CLOSURE, LOWERING, LOWERING),
    "function_output": PageLifecycle(FUNCTIONS, EMISSION, EMISSION),
    "function_parameter": PageLifecycle(FUNCTIONS, EMISSION, EMISSION),
    "hierarchy_global_value": PageLifecycle(FUNCTIONS, None, FUNCTIONS),
    "hierarchy_value": PageLifecycle(FUNCTIONS, None, FUNCTIONS),
    "identity_transition": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "indexed_store_site": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "ingestion_value": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "kernel_by_value_formal_concordance": PageLifecycle(None, LOWERING, LOWERING),
    "kernel_input_conversion": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "layout_state": PageLifecycle(None, REPAIRS, REPAIRS),
    "layout_supersession": PageLifecycle(None, LOWERING, LOWERING),
    "lexical_read_binding": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "linked_caller_member": PageLifecycle(None, LOWERING, LOWERING),
    "loop_carried_binding": PageLifecycle(FUNCTIONS, FUNCTIONS, FUNCTIONS),
    "loop_carried_entry": PageLifecycle(FUNCTIONS, FUNCTIONS, FUNCTIONS),
    "loop_continuation_rewire_concordance": PageLifecycle(None, FUNCTIONS, FUNCTIONS),
    "loop_entry_state": PageLifecycle(FUNCTIONS, FUNCTIONS, FUNCTIONS),
    "loop_record_layout_concordance": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "loop_record_schema_concordance": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "loop_region_membership": PageLifecycle(FUNCTIONS, FUNCTIONS, FUNCTIONS),
    "loop_result_port_binding": PageLifecycle(FUNCTIONS, FUNCTIONS, FUNCTIONS),
    "loop_result_reconciliation": PageLifecycle(FUNCTIONS, FUNCTIONS, FUNCTIONS),
    "loop_result_use_rebinding": PageLifecycle(FUNCTIONS, None, FUNCTIONS),
    "loop_scope": PageLifecycle(None, REPAIRS, REPAIRS),
    "loop_scope_inner_transition": PageLifecycle(None, REPAIRS, REPAIRS),
    "member_formals": PageLifecycle(None, REPAIRS, REPAIRS),
    "memory_release_receipt": PageLifecycle(None, None, None),
    "name_binding": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "native_loop_value": PageLifecycle(EMISSION, EMISSION, EMISSION),
    "numeral_leaf_materialization_concordance": PageLifecycle(None, LOWERING, LOWERING),
    "numeral_record_literal_concordance": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "numeral_return_leaves_concordance": PageLifecycle(None, LOWERING, LOWERING),
    "operator_result_type_concordance": PageLifecycle(FUNCTIONS, REPAIRS, REPAIRS),
    "output_identity_concordance": PageLifecycle(LOWERING, REPAIRS, REPAIRS),
    "page_spill_receipt": PageLifecycle(None, None, None),
    "parameter_abi_kind": PageLifecycle(FUNCTIONS, FUNCTIONS, FUNCTIONS),
    "parameter_annotation": PageLifecycle(CLOSURE, LOWERING, LOWERING),
    "parameter_record_class": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "phi_edge_projection_placement_concordance": PageLifecycle(None, LOWERING, LOWERING),
    "phi_initial_binding": PageLifecycle(FUNCTIONS, FUNCTIONS, FUNCTIONS),
    "physical_call_input_adaptation_fixed_point": PageLifecycle(None, LOWERING, LOWERING),
    "planner_specialization": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "planner_tensor_descriptor": PageLifecycle(None, None, REPAIRS),
    "planning_alias_transition_concordance": PageLifecycle(None, REPAIRS, REPAIRS),
    "planning_value_concordance": PageLifecycle(None, REPAIRS, REPAIRS),
    "precision_channel_shape_concordance": PageLifecycle(None, REPAIRS, REPAIRS),
    "precision_declared_formal_width_concordance": PageLifecycle(None, REPAIRS, REPAIRS),
    "program_abi_field_slot": PageLifecycle(EMISSION, EMISSION, EMISSION),
    "program_abi_keyed_row_record": PageLifecycle(None, LOWERING, LOWERING),
    "propagated_frame_tail_concordance": PageLifecycle(None, REPAIRS, REPAIRS),
    "proven_literal": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "pruned_callee_formal_concordance": PageLifecycle(None, REPAIRS, REPAIRS),
    "receiver_field_slot_read": PageLifecycle(FUNCTIONS, LOWERING, LOWERING),
    "record_descriptor": PageLifecycle(None, REPAIRS, REPAIRS),
    "record_descriptor_merge": PageLifecycle(None, LOWERING, LOWERING),
    "record_field_decomposition": PageLifecycle(None, LOWERING, LOWERING),
    "record_field_incoming_slot": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "record_field_layout_concordance": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "record_field_resident_concordance": PageLifecycle(None, LOWERING, LOWERING),
    "record_field_storage_concordance": PageLifecycle(None, LOWERING, LOWERING),
    "record_forwarding_edge": PageLifecycle(None, LOWERING, LOWERING),
    "record_forwarding_unresolved_actual": PageLifecycle(None, LOWERING, LOWERING),
    "record_member": PageLifecycle(None, REPAIRS, REPAIRS),
    "record_parameter_row_handle": PageLifecycle(None, LOWERING, LOWERING),
    "record_parameter_value": PageLifecycle(None, LOWERING, LOWERING),
    "record_phi_expansion": PageLifecycle(LOWERING, None, LOWERING),
    "record_phi_temporal_fallback_concordance": PageLifecycle(LOWERING, None, LOWERING),
    "record_return_field_selection": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "record_return_layout": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "record_return_phi_input_concordance": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "record_storage_alias": PageLifecycle(None, LOWERING, LOWERING),
    "reducer_field_state": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "region_capture_binding": PageLifecycle(FUNCTIONS, FUNCTIONS, FUNCTIONS),
    "region_feed_consumer": PageLifecycle(FUNCTIONS, FUNCTIONS, FUNCTIONS),
    "region_signature": PageLifecycle(FUNCTIONS, FUNCTIONS, FUNCTIONS),
    "region_value_dtype": PageLifecycle(FUNCTIONS, FUNCTIONS, FUNCTIONS),
    "repository_kernel_definition": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "repository_kernel_value": PageLifecycle(LOWERING, None, LOWERING),
    "result_storage_binding": PageLifecycle(None, LOWERING, LOWERING),
    "return_site_container": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "return_site_field_state": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "return_site_slot": PageLifecycle(LOWERING, REPAIRS, REPAIRS),
    "row_leaf_view": PageLifecycle(None, LOWERING, LOWERING),
    "scalar_integer_width": PageLifecycle(EMISSION, None, EMISSION),
    "scalar_item_merge": PageLifecycle(FUNCTIONS, FUNCTIONS, FUNCTIONS),
    "scalar_kernel_operand_concordance": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "scalar_parameter": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "scheduled_call_argument": PageLifecycle(None, LOWERING, LOWERING),
    "schema_node": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "scope_origin": PageLifecycle(LOWERING, LOWERING, None),
    "scope_registry": PageLifecycle(LOWERING, LOWERING, None),
    "sequence_column_claims": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "sequence_contract_concordance": PageLifecycle(None, LOWERING, LOWERING),
    "sequence_descriptor": PageLifecycle(None, REPAIRS, REPAIRS),
    "sequence_member": PageLifecycle(None, REPAIRS, REPAIRS),
    "sequence_residency": PageLifecycle(None, None, REPAIRS),
    "sequence_row_dtype_concordance": PageLifecycle(None, LOWERING, LOWERING),
    "sequence_row_layout_concordance": PageLifecycle(None, LOWERING, LOWERING),
    "shape.linked": PageLifecycle(None, LOWERING, LOWERING),
    "shape.node": PageLifecycle(None, LOWERING, LOWERING),
    "shape_scope_function": PageLifecycle(LOWERING, LOWERING, None),
    "shape_transformation_concordance": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "shape_transformation_dependents": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "shape_transformation_state": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "single_input_phi_descriptor_concordance": PageLifecycle(None, REPAIRS, REPAIRS),
    "source_callsite_activation_concordance": PageLifecycle(LOWERING, REPAIRS, REPAIRS),
    "source_control_specialization_concordance": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "source_field_identity_concordance": PageLifecycle(CLOSURE, REPAIRS, REPAIRS),
    "source_function_reachability_concordance": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "source_method_identity_concordance": PageLifecycle(CLOSURE, CLOSURE, CLOSURE),
    "source_numeric_specialization_concordance": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "source_pursuit_activation_concordance": PageLifecycle(CLOSURE, CLOSURE, CLOSURE),
    "source_python_identity_concordance": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "source_record_class_concordance": PageLifecycle(CLOSURE, LOWERING, LOWERING),
    "source_span": PageLifecycle(CLOSURE, REPAIRS, REPAIRS),
    "source_value_class_concordance": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "ssa_block": PageLifecycle(FUNCTIONS, LOWERING, LOWERING),
    "ssa_call_shape": PageLifecycle(LOWERING, None, LOWERING),
    "ssa_call_shape_evidence": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "ssa_field_version": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "ssa_shape_materialization": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "ssa_value": PageLifecycle(LOWERING, EMISSION, EMISSION),
    "state_machine_declaration": PageLifecycle(CLOSURE, None, CLOSURE),
    "static_attribute_state": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "static_reference_use": PageLifecycle(LOWERING, None, LOWERING),
    "static_symbol_definition": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "storage_root": PageLifecycle(FUNCTIONS, FUNCTIONS, FUNCTIONS),
    "struct_descriptor": PageLifecycle(None, REPAIRS, REPAIRS),
    "struct_member": PageLifecycle(None, REPAIRS, REPAIRS),
    "structural_specialization_fixed_point": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "symbolic_cache_build": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "symbolic_cache_staleness": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "symbolic_equation": PageLifecycle(LOWERING, None, LOWERING),
    "symbolic_equation_output": PageLifecycle(LOWERING, None, LOWERING),
    "symbolic_ingested_expression": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "symbolic_subexpression": PageLifecycle(LOWERING, None, LOWERING),
    "symbolic_transform": PageLifecycle(LOWERING, None, LOWERING),
    "table_owner": PageLifecycle(None, None, REPAIRS),
    "tensor_operation_parameter": PageLifecycle(LOWERING, None, LOWERING),
    "tensor_shape_concordance": PageLifecycle(FUNCTIONS, LOWERING, LOWERING),
    "tensor_shape_settlement_concordance": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "transformation_decision": PageLifecycle(None, LOWERING, LOWERING),
    "transformation_event": PageLifecycle(None, LOWERING, LOWERING),
    "transformation_rejection": PageLifecycle(None, LOWERING, LOWERING),
    "union_descriptor": PageLifecycle(None, REPAIRS, REPAIRS),
    "union_member": PageLifecycle(None, REPAIRS, REPAIRS),
    "value_shape": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "value_shape_polymorphism": PageLifecycle(LOWERING, LOWERING, LOWERING),
    "while_carried_test": PageLifecycle(FUNCTIONS, None, FUNCTIONS),
}


def cold_after(page: str) -> str | None:
    """The stage after which ``page`` is cold, or None (never spilled / not
    classified: an unclassified page is never spilled)."""
    entry = PAGE_LIFECYCLE.get(page)
    return None if entry is None else entry.cold_after


def cold_pages(pages: Iterable[str], completed_through: str) -> list[str]:
    """The pages among ``pages`` whose ``cold_after`` stage is at or before
    ``completed_through`` (a ``SPILL_STAGES`` key): no code touches them again."""
    limit = _INDEX[completed_through]
    found = []
    for name in pages:
        stage = cold_after(name)
        if stage is not None and _INDEX[stage] <= limit:
            found.append(name)
    return found


def spill_cold_pages(
    book: Any, completed_through: str, *, trigger: str = "explicit",
    budget_bytes: int | None = None, boundary: str = "",
) -> Any:
    """Spill every page of ``book`` that is cold once ``completed_through`` has
    completed (see ``identity_spill.spill_pages``)."""
    names = cold_pages(tuple(dict.keys(book.pages)), completed_through)
    return book.spill_pages(
        names, trigger=trigger, budget_bytes=budget_bytes, boundary=boundary,
    )
