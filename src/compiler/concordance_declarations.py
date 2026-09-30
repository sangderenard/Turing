"""The declared vocabulary of the concordance for migration steps 2 and 3.

Every page, stage, transform and reason that the ingestion roots (step 2,
``docs/concordance_census/60_plan_step2_ingestion_roots.md``) and the reducer
field state (step 3, ``70_plan_step3_reducer_field_state.md``) post with is
declared here once, as an object, so that writers in ``graph_express2``,
``topological_reducer``, ``function_table``, ``precompile_to_ssa``,
``fortran_c_shell`` and ``ssa_record_return_state`` import the SAME object and
no string decides anything at runtime (design section 6, decision 3).

Declarations are idempotent (``Registry`` returns the existing object for the
same name and shape and refuses a different shape), so importing this module
from several writers is safe.  Fact types are declared as frozen dataclasses
here when the plan names a structured fact; ``object`` when the plan leaves it
open.  Sections are owned by the lane that writes the pages; a lane edits only
its own section.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .identity_concordance import (
    Ref,
    RowField,
    RowFieldKind as K,
    Unresolved,
    declare_page,
    declare_reason,
    declare_stage,
    declare_transform,
)

# ============================================================================
# Step 2: ingestion roots (plan 60, sections 1-2)
# ============================================================================

# -- stages ------------------------------------------------------------------
INGESTION = declare_stage("ingestion")
PURSUIT = declare_stage("pursuit")
REDUCTION = declare_stage("reduction")
CANONICAL_RELABEL = declare_stage("canonical_relabel")
FUNCTION_SUBGRAPH = declare_stage("function_subgraph")
FUNCTION_TABLE = declare_stage("function_table")
#: ``fork_read_scope``: a graph copy's read-scope rows, each DERIVED from
#: the cell it was forked from.
READ_SCOPE_FORK = declare_stage("read_scope_fork")

# -- transforms (roots have no minted id: Novel writes the origin edge only) --
INGEST_SOURCE = declare_transform("ingest_source", 0)
CONTRACT_DEMAND = declare_transform("contract_demand", 0)

# -- reasons -----------------------------------------------------------------
SYNTHESIZED_NO_SOURCE = declare_reason("synthesized_no_source")
EXTERNAL_DECLARATION = declare_reason("external_declaration")
HELPER_CALLER_UNROUTED = declare_reason("helper_caller_unrouted")


# -- facts -------------------------------------------------------------------
@dataclass(frozen=True)
class SpanFact:
    """What a source span IS: its construct kind, positions and content hash.

    Positions are a fact, not a key: pursuit deep-copies nodes and
    ``fix_missing_locations`` makes spans collide (plan 60, section 1.4).
    """

    kind: str
    lineno: int
    col_offset: int
    end_lineno: int
    end_col_offset: int
    dump_sha256: str


@dataclass(frozen=True)
class DemandFact:
    record: Any = None


@dataclass(frozen=True)
class NodeFact:
    type: str
    op: str
    label: str


@dataclass(frozen=True)
class BindingFact:
    value_id: int
    authored: bool
    span_positions: tuple
    context_sha256: str


@dataclass(frozen=True)
class ClassFact:
    class_name: str
    permissions: tuple


@dataclass(frozen=True)
class FieldFact:
    storage: str
    annotation: str
    permissions: tuple


@dataclass(frozen=True)
class MethodFact:
    graph_identity: Any
    parameters: tuple


@dataclass(frozen=True)
class AnnotationFact:
    annotation: str
    value: str


@dataclass(frozen=True)
class StateMachineFact:
    marker: str
    bases: tuple
    transition_identity: Any


class DemandKind(Enum):
    RETAIN = "retain"
    PARAMETER_RECORD = "parameter_record"
    PURSUIT_ROOT = "pursuit_root"


class ValueKind(Enum):
    SCALAR = "scalar"


# -- pages (plan 60, table 2.1) ----------------------------------------------
SOURCE_SPAN = declare_page("source_span", (
    RowField("module", K.SCOPE), RowField("qualname", K.NAME),
    RowField("path", K.LABEL),
), SpanFact)
CONTRACT_DEMAND_PAGE = declare_page("contract_demand", (
    RowField("kind", K.LABEL), RowField("identity", K.NAME),
), DemandFact)
INGESTION_VALUE = declare_page("ingestion_value", (
    RowField("ingestion_scope", K.SCOPE), RowField("ingestion_id", K.VALUE_ID),
), NodeFact)
CANONICAL_VALUE = declare_page("canonical_value", (
    RowField("read_scope", K.SCOPE), RowField("canonical_id", K.VALUE_ID),
), tuple)
NAME_BINDING = declare_page("name_binding", (
    RowField("read_scope", K.SCOPE), RowField("name", K.NAME),
    RowField("version", K.INDEX),
), BindingFact)
CLASS_DECLARATION = declare_page("class_declaration", (
    RowField("module", K.SCOPE), RowField("class_identity", K.NAME),
), ClassFact)
CLASS_FIELD_DECLARATION = declare_page("class_field_declaration", (
    RowField("module", K.SCOPE), RowField("class_identity", K.NAME),
    RowField("field", K.NAME),
), FieldFact)
CLASS_METHOD_DECLARATION = declare_page("class_method_declaration", (
    RowField("module", K.SCOPE), RowField("class_identity", K.NAME),
    RowField("method", K.NAME),
), MethodFact)
ANNOTATION_DECLARATION = declare_page("annotation_declaration", (
    RowField("module", K.SCOPE), RowField("owner_qualname", K.NAME),
    RowField("name", K.NAME),
), AnnotationFact)
STATE_MACHINE_DECLARATION = declare_page("state_machine_declaration", (
    RowField("module", K.SCOPE), RowField("class_name", K.NAME),
), StateMachineFact)
SCHEMA_NODE = declare_page("schema_node", (
    RowField("scope", K.SCOPE), RowField("id", K.VALUE_ID),
), bool)
PARAMETER_ANNOTATION = declare_page("parameter_annotation", (
    RowField("module", K.SCOPE), RowField("function_identity", K.NAME),
    RowField("parameter", K.NAME),
), str)
SCALAR_PARAMETER = declare_page("scalar_parameter", (
    RowField("scope", K.SCOPE), RowField("input_id", K.VALUE_ID),
), ValueKind)
FUNCTION_ADDRESS = declare_page("function_address", (
    RowField("qualified_name", K.NAME),
), int)

# Existing pages re-declared with edges (plan 60, table 2.2).  Fact types are
# the objects those pages hold today; ``ast.AST`` is admitted as a fact type.
SOURCE_FIELD_IDENTITY = declare_page("source_field_identity_concordance", (
    RowField("owner_key", K.SCOPE), RowField("attribute", K.NAME),
), object)
SOURCE_METHOD_IDENTITY = declare_page("source_method_identity_concordance", (
    RowField("source_identity", K.SCOPE), RowField("attribute", K.NAME),
), ast.AST)
SOURCE_RECORD_CLASS = declare_page("source_record_class_concordance", (
    RowField("qualified_class_identity", K.SCOPE),
), ast.AST)
SOURCE_PURSUIT_ACTIVATION = declare_page("source_pursuit_activation_concordance", (
    RowField("module", K.SCOPE), RowField("qualname", K.NAME),
), bool)
CALLABLE_IDENTITY = declare_page("callable_identity_concordance", (
    RowField("numeric_scope", K.SCOPE), RowField("node_id", K.VALUE_ID),
), int)
SOURCE_VALUE_CLASS = declare_page("source_value_class_concordance", (
    RowField("numeric_scope", K.SCOPE), RowField("value_id", K.VALUE_ID),
), tuple)

# ============================================================================
# Step 3: reducer field state and return sites (plan 70, sections 1-3, 6)
# ============================================================================

# -- stages ------------------------------------------------------------------
REDUCER_FIELD_WRITE = declare_stage("reducer_field_write")
REDUCER_FIELD_MERGE = declare_stage("reducer_field_merge")
REDUCER_RETURN = declare_stage("reducer_return")
OPERAND_POSITION = declare_stage("operand_position")
CONTROL_SSA = declare_stage("control_ssa")
RECORD_RETURN_VERSION = declare_stage("record_return_version")

# -- transforms --------------------------------------------------------------
RECORD_RETURN_FIELD_CONVERSION = declare_transform("record_return_field_conversion", 1)
OPERAND_APPEND = declare_transform("operand_append", 1)
OPERAND_MOVE = declare_transform("operand_move", 1)
OPERAND_RETIRE = declare_transform("operand_retire", 1)
OPERAND_FORK = declare_transform("operand_fork", 1)


# -- facts -------------------------------------------------------------------
class FieldStateKind(Enum):
    OBSERVED = "observed"
    WRITTEN = "written"
    ELEMENT_WRITTEN = "element_written"
    MERGED = "merged"
    ARM_SELECTED = "arm_selected"
    LOOP_EXIT = "loop_exit"


@dataclass(frozen=True)
class FieldState:
    """The field's current value and the effect that last touched it, as
    the CELLS identifying those nodes, never ids (plan 70, section 1.1)."""

    kind: FieldStateKind
    value: Ref
    effect: Ref


class ContainerKind(Enum):
    TUPLE = "tuple"
    LIST = "list"
    VALUE = "value"


# -- pages -------------------------------------------------------------------
REDUCER_FIELD_STATE = declare_page("reducer_field_state", (
    RowField("reduction_scope", K.SCOPE), RowField("receiver", K.VALUE_ID),
    RowField("field", K.NAME),
), FieldState)                         # mode REVISE: each write is a version
STATIC_ATTRIBUTE_STATE = declare_page("static_attribute_state", (
    RowField("reduction_scope", K.SCOPE), RowField("receiver_path", K.LABEL),
    RowField("field", K.NAME),
), Ref)                                # mode REVISE
RETURN_SITE_SLOT = declare_page("return_site_slot", (
    RowField("reduction_scope", K.SCOPE), RowField("return_site", K.PAGE_REF),
    RowField("slot", K.INDEX),
), object)                             # Ref | Unresolved; mode CONCORD
RETURN_SITE_CONTAINER = declare_page("return_site_container", (
    RowField("reduction_scope", K.SCOPE), RowField("return_site", K.PAGE_REF),
), ContainerKind)                      # mode CONCORD
RETURN_SITE_FIELD_STATE = declare_page("return_site_field_state", (
    RowField("reduction_scope", K.SCOPE), RowField("return_site", K.PAGE_REF),
    RowField("receiver", K.VALUE_ID), RowField("field", K.NAME),
), object)                             # Ref | Unresolved; mode CONCORD
SSA_FIELD_VERSION = declare_page("ssa_field_version", (
    RowField("function_scope", K.SCOPE), RowField("field_state_cell", K.PAGE_REF),
), object)                             # the SSA value published for a field-state cell
RECORD_RETURN_FIELD_SELECTION = declare_page("record_return_field_selection", (
    RowField("function_scope", K.SCOPE), RowField("return_merge_phi", K.PAGE_REF),
    RowField("field", K.NAME), RowField("position", K.INDEX),
    RowField("predecessor", K.LABEL),
), object)                             # SSA_FIELD_VERSION cell | Unresolved; mode REVISE

# -- reasons (plan 70, sections 1-2 and the twelve of section 6) --------------
ARM_VERSION_MISSING = declare_reason("arm_version_missing")
RETURN_SLOT_NOT_A_VALUE = declare_reason("return_slot_not_a_value")
FIELD_STATE_UNRESOLVED_AT_RETURN = declare_reason("field_state_unresolved_at_return")
LOOP_EXIT_FIELD_STATE_UNMERGED = declare_reason("loop_exit_field_state_unmerged")
FIELD_WRITE_DTYPE_UNPROVEN = declare_reason("field_write_dtype_unproven")

NO_RETURN_FIELD_RECEIPTS = declare_reason("no_return_field_receipts")
PREDECESSOR_BLOCK_EMPTY = declare_reason("predecessor_block_empty")
PREDECESSOR_NOT_A_RETURN_EDGE = declare_reason("predecessor_not_a_return_edge")
SITE_WITHOUT_FIELD_STATE = declare_reason("site_without_field_state")
SITES_DISAGREE_ON_VERSION = declare_reason("sites_disagree_on_version")
VERSION_NOT_UNIQUELY_DEFINED = declare_reason("version_not_uniquely_defined")
VERSION_NOT_CONST_OR_CARRIED_PHI = declare_reason("version_not_const_or_carried_phi")
VERSION_IS_FORMAL_SHAPED_OR_MISTYPED = declare_reason("version_is_formal_shaped_or_mistyped")
INTERVENING_CALL_NOT_READONLY = declare_reason("intervening_call_not_readonly")
INTERVENING_STORE = declare_reason("intervening_store")
VERSION_DOES_NOT_DOMINATE_RETURN = declare_reason("version_does_not_dominate_return")
RECORD_DESCRIPTORS_DIFFER = declare_reason("record_descriptors_differ")

RETURN_VERSION_REASONS: tuple = (
    NO_RETURN_FIELD_RECEIPTS, PREDECESSOR_BLOCK_EMPTY,
    PREDECESSOR_NOT_A_RETURN_EDGE, SITE_WITHOUT_FIELD_STATE,
    SITES_DISAGREE_ON_VERSION, VERSION_NOT_UNIQUELY_DEFINED,
    VERSION_NOT_CONST_OR_CARRIED_PHI, VERSION_IS_FORMAL_SHAPED_OR_MISTYPED,
    INTERVENING_CALL_NOT_READONLY, INTERVENING_STORE,
    VERSION_DOES_NOT_DOMINATE_RETURN, RECORD_DESCRIPTORS_DIFFER,
)

# ============================================================================
# Step 4: planner structure (plan 80, part A) -- owned by the step-4 lane
# ============================================================================

# -- stages ------------------------------------------------------------------
#: ``_propagate_callsite_planner_specializations``, the two copy writers
#: (``_callsite_specialized_shell_type``, ``call_result_descriptor``) and
#: ``propagate_bound_planner_specializations``.
PLANNER_SPECIALIZATION = declare_stage("planner_specialization")
#: ``_propagate_callsite_tensor_specializations``,
#: ``_apply_callsite_tensor_descriptors``, ``publish_call_result_shape``.
PLANNER_TENSOR_SPECIALIZATION = declare_stage("planner_tensor_specialization")
#: ``_fold_callsite_structural_values`` and its closures.
PLANNER_STRUCTURAL_FOLD = declare_stage("planner_structural_fold")
#: ``_alias_projection_to_member``, ``_apply_callsite_aggregate_descriptors``.
PLANNER_PROJECTION_ALIAS = declare_stage("planner_projection_alias")
#: ``_dispatch_metadata_node_classifier``: one ``executable_node`` row per
#: classified node under the planning scope.
PLANNER_DISPATCH_CLASSIFICATION = declare_stage("planner_dispatch_classification")
#: ``_dispatch_subgraph``: regions, members and stores.
PLANNER_REGION_CARVE = declare_stage("planner_region_carve")
#: ``assign_hierarchy_ids``.
PLANNER_HIERARCHY = declare_stage("planner_hierarchy")
#: ``link_process_graph_functions``, ``plan_callsites``.
PLANNER_CALL_BINDING = declare_stage("planner_call_binding")
LOOP_COMPOSER = declare_stage("loop_composer")
LOOP_CONTINUATION_REWIRE = declare_stage("loop_continuation_rewire")
#: The stage a planner pass mints its scopes under (``mint_scope``, N5).
PLANNER_SCOPE = declare_stage("planner_scope")

# -- transforms --------------------------------------------------------------
#: ``IdentityBook.mint_scope``: a ``scope_registry`` row is a root that
#: mints no id (the scope IS the row); arity 0.
MINT_SCOPE = declare_transform("mint_scope", 0)
#: ``materialize_retained_loop_ports.add_port``: the port's ``canonical_value``
#: row, from the ``loop_carried_binding`` cell of the binding it continues.
LOOP_RESULT_PORT = declare_transform("loop_result_port", 1)
LOOP_STATE_PORT = declare_transform("loop_state_port", 1)
#: The loop composer's unrolled-induction constant, from the loop cell.
LOOP_COMPOSER_CONSTANT = declare_transform("loop_composer_constant", 1)
#: ``_synthetic_device_scalar_shell``'s three nodes, from the predicate cell.
SYNTHETIC_DEVICE_SCALAR_PREDICATE = declare_transform(
    "synthetic_device_scalar_predicate", 1,
)
#: ``plan_region_to_ssa_instrs.fresh_like``: an SSA value "like" a result,
#: from the result's ``hierarchy_value`` cell.
FRESH_LIKE = declare_transform("fresh_like", 1)
#: ``_dispatch_subgraph``'s Store node for one region output: its id is the
#: subgraph's local counter, so the row carries no ``NEW``; from the OUTPUT
#: member cell.
DISPATCH_STORE = declare_transform("dispatch_store", 1)

# -- reasons -----------------------------------------------------------------
SPECIALIZATION_DYNAMIC_ARGUMENT = declare_reason("specialization_dynamic_argument")
SPECIALIZATION_CALLSITES_DISAGREE = declare_reason("specialization_callsites_disagree")
SPECIALIZATION_NOT_SOURCE_STATIC = declare_reason("specialization_not_source_static")
FORMAL_LITERAL_CONFLICT = declare_reason("formal_literal_conflict")
FORMAL_SHAPE_CONFLICT = declare_reason("formal_shape_conflict")
DESCRIPTORS_DISAGREE = declare_reason("descriptors_disagree")
#: The fold's IfExp whose test is absent from ``known`` (the truthy
#: ``unresolved`` sentinel used to select ``body``).
PREDICATE_NOT_KNOWN = declare_reason("predicate_not_known")
#: A ``name_binding`` version whose node the planner removed.
BINDING_VERSION_REMOVED = declare_reason("binding_version_removed")
RETURN_SITE_UNREACHABLE = declare_reason("return_site_unreachable")
CALLEE_UNRESOLVED = declare_reason("callee_unresolved")
#: ``place_loop_carried_region_producers.owned_regions``' marker-region
#: fallback.
REGION_OWNERSHIP_UNKNOWN = declare_reason("region_ownership_unknown")


# -- facts -------------------------------------------------------------------
class SpecializationSource(Enum):
    LITERAL = "literal"
    DEFAULT = "default"
    BOUND = "bound"


@dataclass(frozen=True)
class SpecializationFact:
    """A folded planner literal for one callee formal and how it was
    proven: an authored literal argument, the signature default of an
    omitted argument, or a binding forwarded from the caller."""

    value: Any
    source: SpecializationSource


class ExecutionKind(Enum):
    EXECUTABLE = "executable"
    DISPATCH_METADATA = "dispatch_metadata"


class MetadataRule(Enum):
    """The branch of ``_is_dispatch_metadata_node_impl`` that decided, one
    per ``return`` (or disjunct of the final ``return``) in that function,
    in evaluation order."""

    PRECISION_OPERATOR = "precision_operator"
    PRECISION_SELECTOR = "precision_selector"
    CALL_RESULT_PROJECTION = "call_result_projection"
    BOUND_METHOD = "bound_method"
    CONDITIONAL_RESULT = "conditional_result"
    CATALOGUED_BUILTIN_CALL = "catalogued_builtin_call"
    GROUNDED_TENSOR_PROPERTY = "grounded_tensor_property"
    SCALAR_INTRINSIC = "scalar_intrinsic"
    COORDINATOR_SHORT_CIRCUIT = "coordinator_short_circuit"
    LINKED_REFERENCE = "linked_reference"
    STATIC_PYTHON_REFERENCE = "static_python_reference"
    COORDINATOR_NODE_TYPE = "coordinator_node_type"
    METHOD_CALL = "method_call"
    CONTEXTUAL_REQUIREMENT = "contextual_requirement"
    AST_METADATA = "ast_metadata"
    PYTHON_SYNTAX = "python_syntax"
    ATTRIBUTE = "attribute"
    PYTHON_ROUTING_INDEX = "python_routing_index"
    PYTHON_SHAPE_INDEX = "python_shape_index"
    COMPARES_NONE = "compares_none"
    NON_NUMERIC_CONSTANT_OPERAND = "non_numeric_constant_operand"
    COORDINATOR_ACCESSOR = "coordinator_accessor"
    COORDINATOR_BOOLEAN_NOT = "coordinator_boolean_not"
    CHAINED_COMPARISON = "chained_comparison"
    LOOP_TARGET_INITIALIZER = "loop_target_initializer"
    NAME_LOAD = "name_load"
    #: No rule fired: ordinary numerical work (``ExecutionKind.EXECUTABLE``).
    NUMERICAL_WORK = "numerical_work"


@dataclass(frozen=True)
class NodeExecution:
    kind: ExecutionKind
    rule: MetadataRule


@dataclass(frozen=True)
class RegionFact:
    schedule_preference: str
    output_count: int
    store_count: int


class MemberRole(Enum):
    INPUT = "input"
    NODE = "node"
    OUTPUT = "output"


class CallResolution(Enum):
    LINKED_PROCESS_GRAPH = "linked_process_graph"
    CONSTRUCTOR = "constructor"
    EXTERNAL = "external"
    #: ``plan_callsites``: a call node whose ``callee_ref`` / ``method_ref``
    #: names a function-table graph shell.
    CALLSITE_SHELL = "callsite_shell"


@dataclass(frozen=True)
class CallBinding:
    callee: int
    resolution: CallResolution


# -- new pages (plan 80, A1.1) ------------------------------------------------
#: Per COPY (design section 7.1): the callee graph's ``lexical_read_scope``,
#: which ``fork_read_scope`` mints fresh for every copy and does not copy
#: these rows into.  Mode REVISE.
PLANNER_SPECIALIZATION_PAGE = declare_page("planner_specialization", (
    RowField("callee_scope", K.SCOPE), RowField("parameter", K.NAME),
), SpecializationFact)
PLANNER_TENSOR_DESCRIPTOR = declare_page("planner_tensor_descriptor", (
    RowField("callee_scope", K.SCOPE), RowField("parameter", K.NAME),
), dict)                               # mode REVISE
EXECUTABLE_NODE = declare_page("executable_node", (
    RowField("planning_scope", K.SCOPE), RowField("node", K.VALUE_ID),
), NodeExecution)                      # mode CONCORD
DEPLOYMENT_REGION = declare_page("deployment_region", (
    RowField("planning_scope", K.SCOPE), RowField("region", K.INDEX),
), RegionFact)                         # mode CONCORD
DEPLOYMENT_REGION_MEMBER = declare_page("deployment_region_member", (
    RowField("planning_scope", K.SCOPE), RowField("region", K.INDEX),
    RowField("node", K.VALUE_ID),
), MemberRole)                         # mode CONCORD
DISPATCH_STORE_PAGE = declare_page("dispatch_store", (
    RowField("planning_scope", K.SCOPE), RowField("region", K.INDEX),
    RowField("output", K.VALUE_ID),
), int)                                # mode CONCORD; NOVEL(DISPATCH_STORE)
HIERARCHY_VALUE = declare_page("hierarchy_value", (
    RowField("hierarchy_scope", K.SCOPE), RowField("closure", K.INDEX),
    RowField("local", K.VALUE_ID),
), int)                                # mode CONCORD
HIERARCHY_GLOBAL_VALUE = declare_page("hierarchy_global_value", (
    RowField("hierarchy_scope", K.SCOPE), RowField("global", K.VALUE_ID),
), tuple)                              # tuple of member Refs; mode CONCORD
CALL_BINDING = declare_page("call_binding", (
    RowField("caller_scope", K.SCOPE), RowField("call", K.VALUE_ID),
), CallBinding)                        # CallBinding | Unresolved; CONCORD
#: N5: every scope ``IdentityBook.mint_scope`` mints, NOVEL(MINT_SCOPE, ()).
SCOPE_REGISTRY = declare_page("scope_registry", (
    RowField("label", K.LABEL), RowField("serial", K.INDEX),
), bool)

# -- existing pages re-declared (plan 80, A1.2), shapes as written today ------
#: ``plan_callsites``: (caller identity, callsite) -> (reference, activation
#: mode, recursive unit), DERIVED from the call's ``call_binding`` cell; the
#: recursion unit stays in the fact tuple until ``recursion_table`` is on the
#: book (plan 80, R6).
SOURCE_CALLSITE_ACTIVATION = declare_page(
    "source_callsite_activation_concordance", (
        RowField("caller_identity", K.SCOPE), RowField("callsite", K.VALUE_ID),
    ), tuple,                          # mode CONCORD
)
#: ``("proven", value, caller)`` or ``Unresolved(FORMAL_LITERAL_CONFLICT)``.
FORMAL_LITERAL = declare_page("formal_literal", (
    RowField("function", K.NAME), RowField("parameter", K.NAME),
), tuple)                              # mode REVISE
#: ``("proven", extents, dtype, caller)`` or ``Unresolved(FORMAL_SHAPE_CONFLICT)``.
FORMAL_SHAPE = declare_page("formal_shape", (
    RowField("function", K.NAME), RowField("parameter", K.NAME),
), tuple)                              # mode REVISE
LOOP_CARRIED_BINDING = declare_page("loop_carried_binding", (
    RowField("read_scope", K.SCOPE), RowField("loop", K.VALUE_ID),
    RowField("updated", K.VALUE_ID), RowField("initial", K.VALUE_ID),
), tuple)                              # binding names; mode CONCORD
#: Fact: the region ordinals this loop owns (the read view the placement
#: pass consumes; plan 80 A1.2 names the ``deployment_region`` cells as the
#: fact -- those rows live under the planning scope, which this writer
#: reaches through ``graph.G.graph["planning_scope"]`` when it exists).
LOOP_REGION_MEMBERSHIP = declare_page("loop_region_membership", (
    RowField("read_scope", K.SCOPE), RowField("loop", K.VALUE_ID),
), tuple)                              # mode REVISE
LOOP_RESULT_PORT_BINDING = declare_page("loop_result_port_binding", (
    RowField("read_scope", K.SCOPE), RowField("port", K.VALUE_ID),
), str)                                # mode CONCORD
LOOP_CONTINUATION_REWIRE_PAGE = declare_page("loop_continuation_rewire_concordance", (
    RowField("read_scope", K.SCOPE), RowField("consumer", K.VALUE_ID),
    RowField("role", K.LABEL), RowField("ordinal", K.INDEX),
), tuple)                              # (old, new, binding); mode REVISE
PROVEN_LITERAL = declare_page("proven_literal", (
    RowField("function", K.NAME), RowField("value", K.VALUE_ID),
), object)                             # mode REVISE
SOURCE_CONTROL_SPECIALIZATION = declare_page(
    "source_control_specialization_concordance", (
        RowField("function", K.NAME), RowField("control", K.VALUE_ID),
    ), dict,                           # dict | Unresolved(PREDICATE_NOT_KNOWN)
)
STRUCTURAL_SPECIALIZATION_FIXED_POINT = declare_page(
    "structural_specialization_fixed_point", (
        RowField("function", K.NAME), RowField("round", K.INDEX),
    ), tuple,                          # (digest, changed, mutations); REVISE
)

# ============================================================================
# Step 5: control SSA builder (plan 80, part B) -- owned by the step-5 lane
# ============================================================================

# -- stages ------------------------------------------------------------------
CONTROL_SSA_ENTRY = declare_stage("control_ssa_entry")
CONTROL_SSA_CONDITIONAL = declare_stage("control_ssa_conditional")
CONTROL_SSA_LOOP = declare_stage("control_ssa_loop")
CONTROL_SSA_REGION = declare_stage("control_ssa_region")
CONTROL_SSA_FINISH = declare_stage("control_ssa_finish")
REDUCER_READ = declare_stage("reducer_read")

# -- transforms (what a ``fresh_value`` mint did to its operand) --------------
# Plan 80 B1 names the Phi, predicate and expression transforms variadic.
# The api's ``Transform.arity`` is fixed, so every transform below takes ONE
# operand cell; a mint made from several cells first posts one ``CELL_SET``
# row DERIVED from all of them and names that row as its operand, so the
# sources are reachable through one more edge, never lost.
CONTROL_CONST = declare_transform("control_const", 1)
CONTROL_PREDICATE = declare_transform("control_predicate", 1)
CONTROL_EXPRESSION = declare_transform("control_expression", 1)
CONTROL_CAST = declare_transform("control_cast", 1)
PHI_CONDITIONAL = declare_transform("phi_conditional", 1)
PHI_LOOP_HEADER = declare_transform("phi_loop_header", 1)
PHI_LOOP_EXIT = declare_transform("phi_loop_exit", 1)
PHI_RETURN_MERGE = declare_transform("phi_return_merge", 1)
LOAD = declare_transform("load", 1)
ADDRESS = declare_transform("address", 1)
SEQUENCE_LENGTH_CELL = declare_transform("sequence_length_cell", 1)
DESCRIPTOR_CELL = declare_transform("descriptor_cell", 1)
REGION_CALL_RESULT = declare_transform("region_call_result", 1)
VERSIONED_WRITE = declare_transform("versioned_write", 1)
LOOP_RESULT_VERSION = declare_transform("loop_result_version", 1)
REGION_FORMAL_SPLIT = declare_transform("region_formal_split", 1)
FIELD_SLOT_ACCESS = declare_transform("field_slot_access", 1)
TABLE_LOOKUP = declare_transform("table_lookup", 1)
ROW_COLUMN_PROJECTION = declare_transform("row_column_projection", 1)
#: The one root row a control lowering posts for itself (a Novel post with
#: no NEW mints nothing and writes only its origin edge, as ``source_span``
#: roots do).  A mint whose operand cell the builder cannot name (a literal
#: with no identity cell, a function the reducer never saw) names this root,
#: so the mint page still shows what made it and from where.
CONTROL_FUNCTION_ROOT = declare_transform("control_function_root", 0)
#: ``fork_read_scope``: the forked scope's ``SCOPE_ORIGIN`` row.
FORK_READ_SCOPE = declare_transform("fork_read_scope", 1)

# -- reasons -----------------------------------------------------------------
NO_PRODUCER_AT_USE = declare_reason("no_producer_at_use")
NAME_ARM_VERSION_MISSING = declare_reason("name_arm_version_missing")
WHILE_TEST_NO_READ_EXPRESSION = declare_reason("while_test_no_read_expression")
REGION_FEED_NO_OPERAND_ROW = declare_reason("region_feed_no_operand_row")
CARRIED_ENTRY_NOT_ATTRIBUTED = declare_reason("carried_entry_not_attributed")
RETURN_SLOT_UNRESOLVED_ON_EDGE = declare_reason("return_slot_unresolved_on_edge")
BINDING_WITHDRAWN = declare_reason("binding_withdrawn")
NO_OPERAND_POSITION_SCOPE = declare_reason("no_operand_position_scope")
#: An SSA value whose id IS a graph id (``_value_from_meta``) in a function
#: whose graph has no ``canonical_value`` row for that id.
GRAPH_ID_WITHOUT_CANONICAL_CELL = declare_reason("graph_id_without_canonical_cell")


# -- facts -------------------------------------------------------------------
class SSAValueOrigin(Enum):
    MINTED = "minted"
    ADOPTED_GRAPH_ID = "adopted_graph_id"


@dataclass(frozen=True)
class SSAValueFact:
    dtype: Any
    shape: tuple
    origin: SSAValueOrigin


class BindingKind(Enum):
    UNIFORM = "uniform"
    PARAMETER_SEED = "parameter_seed"
    FIELD_INCUMBENT = "field_incumbent"
    SEQUENCE_LENGTH = "sequence_length"
    PROVISIONAL_ARGUMENT = "provisional_argument"
    REFINED = "refined"
    REGION_RESULT = "region_result"
    VERSIONED_WRITE = "versioned_write"
    CALL_RESULT = "call_result"
    CONTROL_EXPRESSION = "control_expression"
    FIELD_WRITE = "field_write"
    SEQUENCE_RESULT = "sequence_result"
    RESTORED = "restored"
    CONDITIONAL_MERGE = "conditional_merge"
    LOOP_RESULT_PORT = "loop_result_port"
    LOOP_SEED = "loop_seed"
    LOOP_HEADER = "loop_header"
    LOOP_LATCH = "loop_latch"
    LOOP_CONTROL = "loop_control"
    #: A function the reducer never saw: no ``canonical_value`` cell exists
    #: to derive from (plan 80 B6 R8).
    UNSCOPED = "unscoped"


@dataclass(frozen=True)
class ControlBinding:
    """A graph id's current SSA value: the ``ssa_value`` cell and the
    writer family that bound it."""

    ssa_value: Ref
    kind: BindingKind


class AliasKind(Enum):
    PLANNING = "planning"
    LOOP_BODY_SPELLING = "loop_body_spelling"
    RESTORED = "restored"


@dataclass(frozen=True)
class AliasFact:
    source_id: Any                       # int, or None when the alias is withdrawn
    kind: AliasKind


class ParameterDeclaration(Enum):
    CALL_ONLY = "call_only"
    USED = "used"


@dataclass(frozen=True)
class RegionSignature:
    feeds: tuple
    outputs: tuple


@dataclass(frozen=True)
class OperandTransition:
    """One operator on an operand position (plan 70, section 3)."""

    cause: str


@dataclass(frozen=True)
class OperandMove(OperandTransition):
    consumer: Any
    role: Any
    ordinal: int


@dataclass(frozen=True)
class OperandRetire(OperandTransition):
    pass


@dataclass(frozen=True)
class OperandFork(OperandTransition):
    source_consumer: Any
    source_role: Any
    source_ordinal: int


@dataclass(frozen=True)
class OperandAppend(OperandTransition):
    operand: Any


@dataclass(frozen=True)
class ScopeFork:
    source_scope: Any
    cause: Any


# -- pages: new (plan 80, B1.1) ------------------------------------------------
SSA_VALUE = declare_page("ssa_value", (
    RowField("function_scope", K.SCOPE), RowField("ssa_id", K.VALUE_ID),
), SSAValueFact)                       # mode REVISE
CONTROL_VALUE_BINDING = declare_page("control_value_binding", (
    RowField("function_scope", K.SCOPE), RowField("graph_id", K.VALUE_ID),
), ControlBinding)                     # ControlBinding | Unresolved; mode REVISE
CONTROL_VALUE_ALIAS = declare_page("control_value_alias", (
    RowField("function_scope", K.SCOPE), RowField("alias", K.VALUE_ID),
), AliasFact)                          # mode REVISE
CARRIED_SNAPSHOT = declare_page("carried_snapshot", (
    RowField("function_scope", K.SCOPE), RowField("conditional", K.PAGE_REF),
    RowField("initial", K.VALUE_ID),
), Ref)                                # the binding cell at branch entry; CONCORD
CARRIED_PORT_VALUE = declare_page("carried_port_value", (
    RowField("function_scope", K.SCOPE), RowField("port", K.VALUE_ID),
), Ref)                                # the exit Phi's ssa_value cell; CONCORD
DECLARED_PARAMETER = declare_page("declared_parameter", (
    RowField("function_scope", K.SCOPE), RowField("graph_id", K.VALUE_ID),
), ParameterDeclaration)               # mode REVISE
REGION_SIGNATURE = declare_page("region_signature", (
    RowField("function_scope", K.SCOPE), RowField("region", K.INDEX),
), RegionSignature)                    # mode CONCORD
FUNCTION_PARAMETER = declare_page("function_parameter", (
    RowField("function_scope", K.SCOPE), RowField("name", K.NAME),
), int)                                # mode CONCORD
FUNCTION_OUTPUT = declare_page("function_output", (
    RowField("function_scope", K.SCOPE), RowField("slot", K.INDEX),
), tuple)                              # (name or None, ssa id); CONCORD
#: The operand set of a mint made from several cells: one row per mint,
#: DERIVED from every source cell; its fact repeats the sources as Ref keys.
CELL_SET = declare_page("cell_set", (
    RowField("function_scope", K.SCOPE), RowField("ordinal", K.INDEX),
), tuple)                              # mode CONCORD

# -- pages: declared from raw (plan 80, B1.3) -------------------------------
LEXICAL_READ_BINDING = declare_page("lexical_read_binding", (
    RowField("read_scope", K.SCOPE), RowField("consumer", K.LABEL),
    RowField("role", K.LABEL), RowField("ordinal", K.INDEX),
), str)                                # mode CONCORD
IDENTITY_TRANSITION = declare_page("identity_transition", (
    RowField("scope", K.SCOPE), RowField("consumer", K.LABEL),
    RowField("role", K.LABEL), RowField("ordinal", K.INDEX),
), OperandTransition)                  # mode REVISE
SCOPE_ORIGIN = declare_page("scope_origin", (
    RowField("scope", K.SCOPE),
), ScopeFork)                          # mode CONCORD
SCALAR_ITEM_MERGE = declare_page("scalar_item_merge", (
    RowField("function_scope", K.SCOPE), RowField("item", K.VALUE_ID),
), Ref)                                # the merged source's ssa_value cell; CONCORD

# -- pages: existing builder pages re-declared (plan 80, B1.2; census 75) ----
# The control scope element is the book-numbered scope string
# ``lower_control_sections_to_ssa`` mints (``<name>@control:<serial>``), no
# longer a process object id.
CONTROL_VALUE_CONCORDANCE = declare_page("control_value_concordance", (
    RowField("control_owner", K.SCOPE), RowField("alias", K.VALUE_ID),
), int)                                # mode REVISE (each rebinding is a version)
LOOP_CARRIED_ENTRY = declare_page("loop_carried_entry", (
    RowField("control_scope", K.SCOPE), RowField("loop_node", K.VALUE_ID),
    RowField("binding", K.NAME),
), int)                                # mode CONCORD
LOOP_ENTRY_STATE = declare_page("loop_entry_state", (
    RowField("control_scope", K.SCOPE), RowField("loop_node", K.VALUE_ID),
    RowField("initial", K.VALUE_ID),
), tuple)                              # (bindings, initial id); CONCORD
WHILE_CARRIED_TEST = declare_page("while_carried_test", (
    RowField("control_scope", K.SCOPE), RowField("loop_node", K.VALUE_ID),
), object)                             # bindings tuple | Unresolved; CONCORD
REGION_FEED_CONSUMER = declare_page("region_feed_consumer", (
    RowField("control_scope", K.SCOPE), RowField("region", K.INDEX),
    RowField("feed", K.VALUE_ID),
), tuple)                              # mode CONCORD
REGION_CAPTURE_BINDING = declare_page("region_capture_binding", (
    RowField("control_scope", K.SCOPE), RowField("region", K.INDEX),
    RowField("value_id", K.VALUE_ID),
), tuple)                              # (source value id, binding); CONCORD
CALLSITE_ARGUMENT = declare_page("callsite_argument", (
    RowField("function", K.SCOPE), RowField("callsite", K.VALUE_ID),
    RowField("position", K.INDEX),
), tuple)                              # mode REVISE
LOOP_RESULT_RECONCILIATION = declare_page("loop_result_reconciliation", (
    RowField("function", K.SCOPE), RowField("argument", K.VALUE_ID),
), tuple)                              # mode REVISE
CONTROL_UNIFORM_DTYPE = declare_page("control_uniform_dtype", (
    RowField("control_scope", K.SCOPE), RowField("value_id", K.VALUE_ID),
), str)                                # mode CONCORD
REGION_VALUE_DTYPE = declare_page("region_value_dtype", (
    RowField("control_scope", K.SCOPE), RowField("value_id", K.VALUE_ID),
), str)                                # mode REVISE
FIELD_SLOT_STORAGE = declare_page("field_slot_storage_concordance", (
    RowField("function", K.SCOPE), RowField("slot", K.LABEL),
), tuple)                              # mode CONCORD
SEQUENCE_CONTRACT_CONCORDANCE = declare_page("sequence_contract_concordance", (
    RowField("scope", K.SCOPE), RowField("sequence_id", K.VALUE_ID),
), tuple)                              # (policy, columns, writable, source); REVISE
TENSOR_SHAPE_CONCORDANCE = declare_page("tensor_shape_concordance", (
    RowField("control_scope", K.SCOPE), RowField("value_id", K.VALUE_ID),
), dict)                               # mode REVISE
LOOP_SCOPE = declare_page("loop_scope", (
    RowField("function", K.SCOPE), RowField("loop_node", K.VALUE_ID),
    RowField("key", K.LABEL),
), object)                             # fixed generation columns; CONCORD
LOOP_SCOPE_INNER_TRANSITION = declare_page("loop_scope_inner_transition", (
    RowField("function", K.SCOPE), RowField("loop_node", K.VALUE_ID),
    RowField("declared_inner", K.VALUE_ID),
), tuple)                              # mode REVISE

# -- pages: physical call-input adaptation and precision passes (census 75,
# section 6; ``ssa_call_input_adapters.py`` and ``ir_identities.py``) --------
EXACT_REGION_FEED_DTYPE = declare_page("exact_region_feed_dtype", (
    RowField("control_scope", K.SCOPE), RowField("value_id", K.VALUE_ID),
), tuple)                              # mode CONCORD
EXACT_FORMAL_DTYPE = declare_page("exact_formal_dtype", (
    RowField("callee", K.SCOPE), RowField("formal", K.VALUE_ID),
), tuple)                              # mode CONCORD
PHYSICAL_CALL_INPUT_ADAPTATION_ROUND = declare_page(
    "physical_call_input_adaptation_fixed_point", (
        RowField("program", K.SCOPE), RowField("round", K.INDEX),
    ), int,
)                                      # mode CONCORD
CALL_INPUT_CONVERSION = declare_page("call_input_conversion", (
    RowField("caller", K.SCOPE), RowField("actual", K.VALUE_ID),
    RowField("callee", K.NAME), RowField("formal", K.VALUE_ID),
), tuple)                              # mode REVISE
CALL_INPUT_STORAGE = declare_page("call_input_storage_concordance", (
    RowField("caller", K.SCOPE), RowField("actual", K.VALUE_ID),
), str)                                # mode CONCORD
KERNEL_INPUT_CONVERSION = declare_page("kernel_input_conversion", (
    RowField("function", K.SCOPE), RowField("block", K.NAME),
    RowField("result", K.VALUE_ID), RowField("position", K.INDEX),
), tuple)                              # mode REVISE
PRECISION_CHANNEL_SHAPE = declare_page("precision_channel_shape_concordance", (
    RowField("function", K.SCOPE), RowField("value_id", K.VALUE_ID),
), tuple)                              # mode CONCORD
PRECISION_DECLARED_FORMAL_WIDTH = declare_page(
    "precision_declared_formal_width_concordance", (
        RowField("function", K.SCOPE), RowField("formal", K.VALUE_ID),
    ), int,
)                                      # mode CONCORD
SINGLE_INPUT_PHI_DESCRIPTOR = declare_page(
    "single_input_phi_descriptor_concordance", (
        RowField("function", K.SCOPE), RowField("phi", K.VALUE_ID),
    ), tuple,
)                                      # mode CONCORD

# ============================================================================
# Steps 6-8: records, linker, tables (plan 90) -- owned by those lanes
# ============================================================================

# (steps 6-7 declarations go here)

# ----------------------------------------------------------------------------
# Step 8: the book-backed SSA tables (plan 90, section 4; census 75, section
# 5 and the step-8 DRAFT of section 10).  DECLARATION ONLY in this phase: the
# writers in ``src/transmogrifier/ssa.py`` (``_BookRows``,
# ``_revise_member_claims``, ``SSARecordTable.register``,
# ``SSASequenceTable.register``, ``_SSALayoutTable``, ``_BookCallList``,
# ``SSACallTable``) keep their raw writes until a caller hands them source
# cells; the row shapes below are the shapes those writers produce today.
# ----------------------------------------------------------------------------

# -- stages ------------------------------------------------------------------
#: Every table write made on behalf of a caller that names its source cells
#: (``register(..., sources=...)``, ``assign``, ``remove``, the call-list
#: mutators): the default stage of the sourced path.
TABLE_REGISTRATION = declare_stage("table_registration")
#: ``_mint_table_owner``: the scope one table's rows live under (E6.0 posts
#: the ``scope_registry`` row under the caller's stage; this is the tables').
SCOPE_MINT = declare_stage("scope_mint")
#: ``_SSALayoutTable.register``'s default: a layout row declared by the
#: frontend itself (today the string ``"declaration"``).
DECLARATION = declare_stage("declaration")
#: ``ctypes_layout``'s ``STAGE``: a layout row read from a ctypes type.
CTYPES_INTERCEPTION = declare_stage("ctypes_interception")

# -- transforms --------------------------------------------------------------
#: A table owner scope made for one function: operand is that function's
#: ``FUNCTION_SCOPE`` cell (plan 90, section 1.2).
TABLE_OWNER_SCOPE = declare_transform("table_owner_scope", 1)

# -- reasons -----------------------------------------------------------------
#: A table built with a label only (``"module"``, ``"table"``,
#: ``"call_records"``, a test constructor, a table rebuilt from a pickle):
#: no function scope to derive its owner row from.
NO_FUNCTION_SCOPE = declare_reason("no_function_scope")
#: ``SSASequenceTable.register`` called without ``sources``: the column
#: typing was offered by a site the book cannot name.
SEQUENCE_CLAIM_WITHOUT_PROPOSER = declare_reason("sequence_claim_without_proposer")
#: ``SSASequenceTable.register`` re-offered the same column typing from the
#: same, unchanged cells: recorded (every attempt is), but it has no cause
#: the api admits as a revision.
SEQUENCE_CLAIM_UNCHANGED = declare_reason("sequence_claim_unchanged")

# -- facts -------------------------------------------------------------------
class TableKind(Enum):
    """Which SSA table a ``TABLE_OWNER`` scope belongs to."""

    RECORD = "record"
    SEQUENCE = "sequence"
    STRUCT = "struct"
    UNION = "union"
    CALL = "call"


@dataclass(frozen=True)
class RecordMergeFact:
    """One complementary-view merge in ``SSARecordTable.register``.

    ``incumbent`` and ``incoming`` are the two descriptors as registered,
    ``merged`` the descriptor written; ``widened_fields`` names every field
    the merge made writable that the incumbent had read-only, and
    ``adopted_pool`` says whether the incoming view supplied the instance
    pool the incumbent lacked.  Silent ORs become named fields.
    """

    incumbent: Any
    incoming: Any
    merged: Any
    widened_fields: tuple[str, ...]
    adopted_pool: bool


# -- pages: the table pages as written today (census 75, section 5) ---------
# Owner is the ``(label, serial)`` scope ``_mint_table_owner`` mints; ids are
# SSA value ids; a removal is a ``None`` revision, so descriptor and
# call-record facts are ``object`` (descriptor | None, tuple | None) while
# member claims are always tuples.
RECORD_DESCRIPTOR = declare_page("record_descriptor", (
    RowField("owner", K.SCOPE), RowField("record", K.VALUE_ID),
), object)                             # SSARecordDescriptor | None; REVISE
RECORD_MEMBER = declare_page("record_member", (
    RowField("owner", K.SCOPE), RowField("member", K.VALUE_ID),
), tuple)                              # (record id, field, storage identity, role)...; REVISE
SEQUENCE_DESCRIPTOR = declare_page("sequence_descriptor", (
    RowField("owner", K.SCOPE), RowField("sequence", K.VALUE_ID),
), object)                             # SSASequenceDescriptor | None; REVISE
SEQUENCE_MEMBER = declare_page("sequence_member", (
    RowField("owner", K.SCOPE), RowField("member", K.VALUE_ID),
), tuple)                              # (sequence id, role)...; REVISE
#: Every column typing ever offered for a sequence, winner or not; the key
#: element is the literal label ``"column_dtypes"``.
SEQUENCE_COLUMN_CLAIMS = declare_page("sequence_column_claims", (
    RowField("owner", K.SCOPE), RowField("sequence", K.VALUE_ID),
    RowField("key", K.LABEL),
), tuple)                              # (column dtypes, key columns); REVISE
CALL_RECORD = declare_page("call_record", (
    RowField("owner", K.SCOPE), RowField("caller", K.NAME),
), object)                             # tuple of SSACallRecord | None; REVISE
STRUCT_DESCRIPTOR = declare_page("struct_descriptor", (
    RowField("owner", K.SCOPE), RowField("struct", K.VALUE_ID),
), object)                             # SSAStructDescriptor | None; REVISE
STRUCT_MEMBER = declare_page("struct_member", (
    RowField("owner", K.SCOPE), RowField("member", K.VALUE_ID),
), tuple)                              # (container kind, container id, name, offset, role)...; REVISE
UNION_DESCRIPTOR = declare_page("union_descriptor", (
    RowField("owner", K.SCOPE), RowField("union", K.VALUE_ID),
), object)                             # SSAUnionDescriptor | None; REVISE
UNION_MEMBER = declare_page("union_member", (
    RowField("owner", K.SCOPE), RowField("member", K.VALUE_ID),
), tuple)                              # REVISE
#: ``("resolved", descriptor, edge row | None)``, ``("superseded", (kind,
#: id), stage)`` or ``("invalidated", (kind, id), reason)``.
LAYOUT_STATE = declare_page("layout_state", (
    RowField("owner", K.SCOPE), RowField("kind", K.NAME),
    RowField("row_id", K.VALUE_ID),
), tuple)                              # REVISE when changed
#: One edge per re-declaration, at the page's next free column.
LAYOUT_SUPERSESSION = declare_page("layout_supersession", (
    RowField("owner", K.SCOPE), RowField("kind", K.NAME),
    RowField("target", K.VALUE_ID), RowField("source", K.VALUE_ID),
    RowField("stage", K.NAME),
), tuple)                              # (incumbent, replacement); CONCORD
#: ``SSARecordTable.register``'s complementary merge, one row per merge:
#: ``revision`` is the descriptor row's revision count at the merge.
#: DERIVED(incumbent ``record_descriptor`` cell, the incoming view's cells);
#: the descriptor revision that follows derives from this cell.
RECORD_DESCRIPTOR_MERGE = declare_page("record_descriptor_merge", (
    RowField("owner", K.SCOPE), RowField("record", K.VALUE_ID),
    RowField("revision", K.INDEX),
), RecordMergeFact)                    # mode CONCORD
#: The join from a table's owner scope to the function it was built for
#: (plan 90, section 4.2, row spelled as the lead directed: the function
#: symbol is a row element, the fact is the table kind).  DERIVED(the
#: function's ``FUNCTION_SCOPE`` cell) or ``Unresolved(NO_FUNCTION_SCOPE)``.
TABLE_OWNER = declare_page("table_owner", (
    RowField("owner", K.SCOPE), RowField("function", K.NAME),
), TableKind)                          # mode CONCORD

# -- pages: sequence-contract helpers nearest to step 8 (census 75, section
# 6; ``sequence_contract_concordance`` is declared in the step-5 section) --
#: ``(column shapes, column dtypes, source)`` or ``("invalidated", source, reason)``;
#: the scope element is the authored function name.
SEQUENCE_ROW_LAYOUT = declare_page("sequence_row_layout_concordance", (
    RowField("function", K.SCOPE), RowField("sequence", K.VALUE_ID),
), tuple)                              # mode REVISE
SEQUENCE_ROW_DTYPE = declare_page("sequence_row_dtype_concordance", (
    RowField("scope", K.SCOPE), RowField("sequence", K.VALUE_ID),
), tuple)                              # (dtypes, source) at the next column

__all__ = [name for name in dir() if not name.startswith("_") and name not in {
    "ast", "dataclass", "Enum", "Any", "Ref", "RowField", "K", "Unresolved",
    "declare_page", "declare_reason", "declare_stage", "declare_transform",
}]
