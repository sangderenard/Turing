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
#: A caller passes the formal as a described tensor (shape/dtype): the
#: literal belongs to that callsite's copy, never to the shared definition.
SPECIALIZATION_TENSOR_ARGUMENT = declare_reason("specialization_tensor_argument")
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

# ----------------------------------------------------------------------------
# Step 6: record materialization and return versions (plan 90, sections 1-2;
# census 75, section 3 and the step-6 DRAFT of section 10).
#
# The identity of every SSA value the record passes mint (a per-field Phi, a
# record-return Cast, a program-abi default, an optional presence or inactive
# payload, a loop record header or projection) is a NOVEL row on step 5's
# ``ssa_value`` page under the function's control scope
# (``metadata["tensor_shape_concordance_scope"]``), with the transform below
# and the cells it was made from as operands; plan 90's SSA_VALUE_IDENTITY is
# that page, not a second one.  A mint made from several cells names one
# ``cell_set`` row (step 5's page).  The pages below are the record passes'
# own statements, re-declared with the row shapes their writers produce.
# ----------------------------------------------------------------------------

# -- stages ------------------------------------------------------------------
#: ``materialize_program_abi_record_literals``.
RECORD_LITERAL_MATERIALIZATION = declare_stage("record_literal_materialization")
#: ``materialize_record_phis`` and ``materialize_loop_record_phis``.
RECORD_PHI_EXPANSION_STAGE = declare_stage("record_phi_expansion")
#: Every ``record_return_layouts`` writer.
RECORD_RETURN_LAYOUT_STAGE = declare_stage("record_return_layout")
#: ``_publish_concorded_output_identities``.
OUTPUT_IDENTITY_STAGE = declare_stage("output_identity")
#: ``materialize_parameter_record_abi`` (step 7 routes it).
RECORD_ABI_MATERIALIZATION = declare_stage("record_abi_materialization")
#: ``recover_structural_source_outputs``.
STRUCTURAL_RECOVERY = declare_stage("structural_recovery")
#: ``freshen_redefined_ssa_objects`` and the other
#: ``ssa_record_return_state`` repairs.
RECORD_RETURN_REPAIR = declare_stage("record_return_repair")

# -- transforms (arity 1: several sources become one ``cell_set`` row) -------
#: A per-field Phi of a record merge: operand = the ``cell_set`` of the record
#: Phi's identity cell and each incoming record's ``record_member`` cell.
RECORD_FIELD_PHI = declare_transform("record_field_phi", 1)
#: A constructor field's default Const: operand = the field's
#: ``class_field_declaration`` cell (else the function root).
PROGRAM_ABI_DEFAULT = declare_transform("program_abi_default", 1)
OPTIONAL_PRESENCE = declare_transform("optional_presence", 1)
OPTIONAL_INACTIVE_PAYLOAD = declare_transform("optional_inactive_payload", 1)
#: The conceptual record header Phi a loop acquires: operand = the
#: ``cell_set`` of the initial and updated descriptors' cells.
LOOP_RECORD_HEADER = declare_transform("loop_record_header", 1)
#: The projection of a richer callee record onto the loop's schema: operand
#: = the ``loop_record_schema_concordance`` cell that recorded the projection.
LOOP_RECORD_PROJECTION = declare_transform("loop_record_projection", 1)
#: ``freshen_redefined_ssa_objects``: operand = the redefined value's cell.
FRESHEN = declare_transform("freshen", 1)

# -- reasons -----------------------------------------------------------------
PLANNER_OUTPUT_UNROUTED = declare_reason("planner_output_unrouted")
#: A layout revision whose member ids have no identity cell yet, so the
#: changed layout cannot name a changed source.
LAYOUT_MEMBER_NOT_YET_DEFINED = declare_reason("layout_member_not_yet_defined")
#: A numeral literal whose coefficient record arrives only when the
#: function's calls are linked (the ``"deferred"`` row).
LITERAL_FIELD_DEFERRED = declare_reason("literal_field_deferred")
#: ``coalesce_record_field_storage`` took the first read as the resident.
RESIDENT_CHOSEN_BY_ORDER = declare_reason("resident_chosen_by_order")

# -- pages: new --------------------------------------------------------------
#: The physical layout returned for a record, one row per (function scope,
#: record id); REVISE, DERIVED(the record's ``record_descriptor`` cell, each
#: layout id's ``ssa_value`` cell).  ``metadata["record_return_layouts"]``
#: is written beside it until its readers move to this page.
RECORD_RETURN_LAYOUT = declare_page("record_return_layout", (
    RowField("function_scope", K.SCOPE), RowField("record", K.VALUE_ID),
), tuple)
#: One row per field Phi a record Phi expands into; the fact is the field
#: Phi's ``ssa_value`` cell.  CONCORD, DERIVED(the same cells the mint named).
RECORD_PHI_EXPANSION = declare_page("record_phi_expansion", (
    RowField("function_scope", K.SCOPE), RowField("record_phi", K.VALUE_ID),
    RowField("field", K.NAME), RowField("slot", K.INDEX),
), Ref)

# -- pages: re-declared with the shapes written today -------------------------
#: (value ids, sequence id, record id, offset, storage kind); CONCORD.
RECORD_FIELD_LAYOUT = declare_page("record_field_layout_concordance", (
    RowField("symbol", K.SCOPE), RowField("result", K.VALUE_ID),
    RowField("storage_identity", K.NAME),
), tuple)
#: (old physical layout, new layout, "record_loop_phi"); CONCORD.
LOOP_RECORD_LAYOUT = declare_page("loop_record_layout_concordance", (
    RowField("symbol", K.SCOPE), RowField("loop_node", K.VALUE_ID),
    RowField("result", K.VALUE_ID),
), tuple)
#: (initial signature, updated signature, projected, discarded,
#: "project_updated_to_initial"); CONCORD.
LOOP_RECORD_SCHEMA = declare_page("loop_record_schema_concordance", (
    RowField("symbol", K.SCOPE), RowField("loop_node", K.VALUE_ID),
    RowField("result", K.VALUE_ID),
), tuple)
#: Three statements on one page: (symbol, record, field) -> field value id;
#: (symbol, node, "deferred") -> True or Unresolved(LITERAL_FIELD_DEFERRED);
#: (symbol, node, "completed") -> the field tuple.  The third element is a
#: LABEL because it is a field name or a status; CONCORD.
NUMERAL_RECORD_LITERAL = declare_page("numeral_record_literal_concordance", (
    RowField("symbol", K.SCOPE), RowField("node_or_record", K.VALUE_ID),
    RowField("field_or_status", K.LABEL),
), object)
RECORD_FIELD_DECOMPOSITION = declare_page("record_field_decomposition", (
    RowField("owner", K.SCOPE), RowField("record", K.VALUE_ID),
    RowField("storage_identity", K.NAME),
), tuple)                              # CONCORD
PROGRAM_ABI_KEYED_ROW_RECORD = declare_page("program_abi_keyed_row_record", (
    RowField("owner", K.SCOPE), RowField("record", K.VALUE_ID),
    RowField("storage_identity", K.NAME),
), int)                                # CONCORD
#: The resident among a field's reads: its id, or
#: Unresolved(RESIDENT_CHOSEN_BY_ORDER) when the first read was taken.
RECORD_FIELD_RESIDENT = declare_page("record_field_resident_concordance", (
    RowField("symbol", K.SCOPE), RowField("parameter_root", K.VALUE_ID),
    RowField("field", K.NAME),
), object)                             # CONCORD
RECORD_FIELD_STORAGE = declare_page("record_field_storage_concordance", (
    RowField("caller", K.SCOPE), RowField("record", K.VALUE_ID),
    RowField("storage_identity", K.NAME),
), dict)                               # REVISE
NUMERAL_LEAF_MATERIALIZATION = declare_page("numeral_leaf_materialization_concordance", (
    RowField("symbol", K.SCOPE), RowField("parameter", K.NAME),
    RowField("path", K.LABEL),
), str)                                # CONCORD
NUMERAL_RETURN_LEAVES = declare_page("numeral_return_leaves_concordance", (
    RowField("symbol", K.SCOPE), RowField("argument", K.VALUE_ID),
), tuple)                              # CONCORD
#: ``_publish_concorded_output_identities``: alias value -> result value;
#: REVISE, DERIVED(the alias's and the result's ``ssa_value`` cells).
OUTPUT_IDENTITY = declare_page("output_identity_concordance", (
    RowField("function", K.NAME), RowField("alias", K.VALUE_ID),
), int)
#: ``_concord_record_return_phi_inputs``: (candidate, chosen, reason);
#: REVISE, DERIVED(the ``record_return_field_selection`` cell the choice
#: was made by, the chosen value's ``ssa_value`` cell).
RECORD_RETURN_PHI_INPUT = declare_page("record_return_phi_input_concordance", (
    RowField("function", K.SCOPE), RowField("phi", K.VALUE_ID),
    RowField("field", K.NAME), RowField("position", K.INDEX),
    RowField("predecessor", K.NAME),
), tuple)
RECORD_PHI_TEMPORAL_FALLBACK = declare_page("record_phi_temporal_fallback_concordance", (
    RowField("function", K.SCOPE), RowField("result", K.VALUE_ID),
    RowField("block", K.NAME), RowField("use_index", K.INDEX),
    RowField("position", K.INDEX),
), tuple)                              # CONCORD

# ----------------------------------------------------------------------------
# Step 7: the frame linker (plan 90, section 3; census 10, sections 1.1-1.6
# and 2; census 75, section 4 and the step-7 DRAFT of section 10).
#
# Every SSA value the linker mints (a leased result slot, a frame-tail slot,
# a replacement slot, a cloned frame value, a constructor remap, a scaffold
# constant, a structural fold intermediate, a declared row column, a nested
# record part) is a NOVEL row on step 5's ``ssa_value`` page under the
# function's control scope, with one of the transforms below and the cell it
# was made from as operand (several cells become one ``cell_set`` row; none
# names the function root, as step 5's ``fresh_value`` does).  The pages
# below are the linker's own statements.  The argument storage the binding
# walk mints from absence is RECORDED, never raised: the storage row is
# NOVEL and the binding row is ``Unresolved(STORAGE_MINTED_FROM_ABSENCE)``.
# ----------------------------------------------------------------------------

# -- stages ------------------------------------------------------------------
#: The binding walk of ``_class_surface_ssa_program`` (one linked call).
FRAME_BINDING = declare_stage("frame_binding")
#: The frame fixed point: leases, ledger proposals, replacement slots.
FRAME_LINK = declare_stage("frame_link")
#: ``_complete_propagated_frame_tails``.
FRAME_TAIL_STAGE = declare_stage("frame_tail")
#: The record-forwarding roots (``record_parameter_value`` / row handles).
RECORD_FORWARDING = declare_stage("record_forwarding")
#: Sequence residency decided while lowering sequence operations.
PLANNING_RESIDENCY = declare_stage("planning_residency")
#: The shell control handoff's planning aliases.
SHELL_HANDOFF = declare_stage("shell_handoff")

# -- transforms (arity 1) ----------------------------------------------------
#: Caller storage leased for a callee value: operand = the callee value's
#: ``ssa_value`` cell (its ``record_member`` cell joins through the binding).
RESULT_STORAGE_LEASE = declare_transform("result_storage_lease", 1)
#: A caller slot appended for a callee formal the call was short by.
FRAME_TAIL_SLOT = declare_transform("frame_tail_slot", 1)
#: A caller clone of linked frame storage (``clone_value`` on a lease).
FRAME_STORAGE_CLONE = declare_transform("frame_storage_clone", 1)
#: The slot an accepted ledger proposal replaces a shared slot with.
REPLACEMENT_SLOT = declare_transform("replacement_slot", 1)
#: ``remap[old_id]`` inside a constructor frame; pool ids and strides.
CONSTRUCTOR_REMAP = declare_transform("constructor_remap", 1)
#: Output index / address constants, derived lengths, literal values,
#: projected element addresses, child pool columns.
FRAME_SCAFFOLD = declare_transform("frame_scaffold", 1)
#: A structural Boolean / membership / load intermediate.
STRUCTURAL_FOLD = declare_transform("structural_fold", 1)
#: One physical leaf of a nested program-abi record (operand = the declared
#: field cell or the owning record's cell).
NESTED_RECORD_PART = declare_transform("nested_record_part", 1)
#: A pooled column of a declared keyed field's rows, and its row record.
DECLARED_ROW_COLUMN = declare_transform("declared_row_column", 1)

# -- reasons -----------------------------------------------------------------
#: The binding walk's final ``else``: the callee formal matched no caller
#: value, literal, default or linked member; storage was leased for it.
STORAGE_MINTED_FROM_ABSENCE = declare_reason("storage_minted_from_absence")
#: ``record_parameter_row_handle``: the abi record declares no ``identity``,
#: so the schema NAME stands in for the row identity.
ROW_IDENTITY_FROM_SCHEMA_NAME = declare_reason("row_identity_from_schema_name")
#: A frame tail completed for a formal no ``argument_binding`` row binds.
FORMAL_UNBOUND_AT_TAIL = declare_reason("formal_unbound_at_tail")
#: ``_linked_caller_member`` found the callee formal a member of no record
#: bound at this call.
MEMBER_NOT_BOUND_AT_CALL = declare_reason("member_not_bound_at_call")
#: An aggregate output whose position the projection table lacks.
AGGREGATE_POSITION_MISSING = declare_reason("aggregate_position_missing")
#: A planning alias built by matching an authored name (``singleton_name_aliases``).
RESIDENCY_FROM_NAME_MATCH = declare_reason("residency_from_name_match")
#: A linker statement whose source value has no ``ssa_value`` / ``canonical_value``
#: cell on the book (a function another lowering built).
FRAME_SOURCE_CELL_ABSENT = declare_reason("frame_source_cell_absent")

# -- facts -------------------------------------------------------------------
class ArgumentBindingFact(tuple):
    """``(kind, source)`` of one callee formal at one callsite: ``kind`` is
    the binding kind string the call record carries (``caller_value``,
    ``caller_literal``, ``caller_storage``, ``default_literal``), ``source``
    the caller value id or the literal.  A tuple, so every reader that reads
    ``(kind, source)`` pairs off the page keeps reading."""

    __slots__ = ()

    def __new__(cls, kind: Any, source: Any) -> "ArgumentBindingFact":
        return tuple.__new__(cls, (str(kind), source))

    @property
    def kind(self) -> str:
        return self[0]

    @property
    def source(self) -> Any:
        return self[1]


@dataclass(frozen=True)
class ResidencyFact:
    """One sequence value's resident and kind, and the helper that decided."""

    resident: int
    kind: str
    helper: str


class CallBindingSource(Enum):
    """How the binding walk learned a callee formal's caller value."""

    IDENTITY_ALIAS = "identity_alias"
    DEFAULT_LITERAL = "default_literal"
    DISCOVERY = "discovery"


# -- pages: new --------------------------------------------------------------
#: Caller storage leased for one callee value at one call; the fact is the
#: storage's ``ssa_value`` cell.  CONCORD, DERIVED(the callee value's cell).
#: A ``distinct_slot`` lease is its own row, keyed by its serial.
RESULT_STORAGE_BINDING = declare_page("result_storage_binding", (
    RowField("caller", K.SCOPE), RowField("callsite", K.LABEL),
    RowField("callee_value", K.VALUE_ID), RowField("serial", K.INDEX),
), Ref)
#: An input of the binding walk: the caller formal an identity alias names,
#: a Python default, or a discovered linked member.  CONCORD, DERIVED.
CALL_BINDING_INPUT = declare_page("call_binding_input", (
    RowField("caller", K.SCOPE), RowField("callsite", K.LABEL),
    RowField("callee_value", K.VALUE_ID), RowField("source", K.LABEL),
), object)
#: ``_linked_caller_member``'s decision: the caller value a callee formal is
#: at one call, or ``Unresolved(MEMBER_NOT_BOUND_AT_CALL)``.  CONCORD.
LINKED_CALLER_MEMBER = declare_page("linked_caller_member", (
    RowField("caller", K.NAME), RowField("callsite", K.LABEL),
    RowField("callee", K.NAME), RowField("formal", K.VALUE_ID),
), int)
#: Sequence residency per value: REVISE, DERIVED(the value's cell, the
#: operation's cell).  Declared here; the 104 dict writes route to it as
#: each helper is migrated.
SEQUENCE_RESIDENCY = declare_page("sequence_residency", (
    RowField("function", K.NAME), RowField("value", K.VALUE_ID),
), ResidencyFact)

# -- pages: re-declared (census 75, step 7) ----------------------------------
#: The callsite moves from the column into the row: one row per binding.
#: Fact ``ArgumentBindingFact`` or ``Unresolved(STORAGE_MINTED_FROM_ABSENCE)``.
#: CONCORD; rows written raw before this step keep ``"binding"`` as their
#: third element with the callsite as the column.
ARGUMENT_BINDING = declare_page("argument_binding", (
    RowField("callee", K.NAME), RowField("formal", K.VALUE_ID),
    RowField("callsite", K.LABEL),
), ArgumentBindingFact)
ARGUMENT_BINDING_RESOLUTION = declare_page("argument_binding_resolution", (
    RowField("callee", K.SCOPE), RowField("formal", K.VALUE_ID),
    RowField("source", K.VALUE_ID),
), tuple)                              # REVISE (next column)
#: (callsite, callee, formal); CONCORD, DERIVED(the slot's cell, the formal's cell).
FRAME_LEASE = declare_page("frame_lease_link", (
    RowField("caller", K.NAME), RowField("slot", K.VALUE_ID),
), tuple)
#: (slot, kind) or Unresolved(FORMAL_UNBOUND_AT_TAIL); CONCORD.
FRAME_TAIL = declare_page("propagated_frame_tail_concordance", (
    RowField("owner", K.NAME), RowField("callsite", K.LABEL),
    RowField("callee", K.NAME), RowField("formal", K.VALUE_ID),
), tuple)
#: value -> resident under ``mint_scope(("record_storage_alias", symbol))``;
#: REVISE, DERIVED(the ``record_field_resident_concordance`` cell).
RECORD_STORAGE_ALIAS = declare_page("record_storage_alias", (
    RowField("alias_scope", K.SCOPE), RowField("alias", K.VALUE_ID),
), int)
#: (symbol, parameter) for one (symbol, value id); CONCORD, DERIVED(the
#: parameter's ``name_binding`` cells, the record's declaration cell).
RECORD_PARAMETER_VALUE = declare_page("record_parameter_value", (
    RowField("access_scope", K.SCOPE), RowField("symbol_value", K.LABEL),
), tuple)
#: ((symbol, parameter), path prefix, row identity) or
#: Unresolved(ROW_IDENTITY_FROM_SCHEMA_NAME); CONCORD.
RECORD_PARAMETER_ROW_HANDLE = declare_page("record_parameter_row_handle", (
    RowField("access_scope", K.SCOPE), RowField("symbol_value", K.LABEL),
), tuple)
RECORD_FORWARDING_EDGE = declare_page("record_forwarding_edge", (
    RowField("access_scope", K.SCOPE), RowField("edge_key", K.LABEL),
), tuple)                              # REVISE
RECORD_FORWARDING_UNRESOLVED_ACTUAL = declare_page("record_forwarding_unresolved_actual", (
    RowField("access_scope", K.SCOPE), RowField("binding", K.LABEL),
), tuple)                              # CONCORD
#: caller child record for (caller, callsite, callee child record); CONCORD,
#: DERIVED(the bound pair's binding cell, both fields' ``record_member`` cells).
CALL_RECORD_PAIR = declare_page("call_record_pair_concordance", (
    RowField("caller", K.NAME), RowField("callsite", K.LABEL),
    RowField("callee_record", K.VALUE_ID),
), int)
PLANNING_VALUE = declare_page("planning_value_concordance", (
    RowField("function", K.SCOPE), RowField("alias", K.VALUE_ID),
), object)                             # int, None on retirement; REVISE
PLANNING_ALIAS_TRANSITION = declare_page("planning_alias_transition_concordance", (
    RowField("function", K.SCOPE), RowField("alias", K.VALUE_ID),
), tuple)                              # REVISE
ALIAS_APPLICATION = declare_page("alias_application_concordance", (
    RowField("function", K.SCOPE), RowField("block", K.NAME),
    RowField("instruction", K.INDEX), RowField("position", K.INDEX),
), tuple)                              # CONCORD
SCHEDULED_CALL_ARGUMENT = declare_page("scheduled_call_argument", (
    RowField("scope", K.SCOPE), RowField("call_formal", K.LABEL),
), object)                             # the resident SSAValue; REVISE
CALL_LINK_ORDER = declare_page("call_link_order_concordance", (
    RowField("artifact", K.SCOPE), RowField("symbol", K.NAME),
), int)                                # CONCORD
KERNEL_BY_VALUE_FORMAL = declare_page("kernel_by_value_formal_concordance", (
    RowField("callee", K.SCOPE), RowField("formal", K.VALUE_ID),
), str)                                # CONCORD
PHI_EDGE_PROJECTION_PLACEMENT = declare_page("phi_edge_projection_placement_concordance", (
    RowField("function", K.SCOPE), RowField("phi", K.VALUE_ID),
    RowField("position", K.INDEX), RowField("predecessor", K.NAME),
    RowField("value_id", K.VALUE_ID),
), tuple)                              # CONCORD
PRUNED_CALLEE_FORMAL = declare_page("pruned_callee_formal_concordance", (
    RowField("callee", K.SCOPE), RowField("formal", K.VALUE_ID),
), str)                                # CONCORD
ENTRY_RECORD_HANDLE = declare_page("entry_record_handle_concordance", (
    RowField("function", K.SCOPE), RowField("parameter", K.NAME),
), str)                                # CONCORD
MEMBER_FORMALS = declare_page("member_formals", (
    RowField("function", K.SCOPE), RowField("value_id", K.VALUE_ID),
    RowField("role", K.LABEL),
), tuple)                              # column = aggregate index
FORMAL_ACTUAL = declare_page("formal_actual_concordance", (
    RowField("callee", K.SCOPE), RowField("formal", K.VALUE_ID),
    RowField("caller", K.NAME), RowField("block", K.NAME),
    RowField("instruction", K.INDEX), RowField("position", K.INDEX),
), int)                                # CONCORD
#: ``TransformationLedger`` (routed by ``propose(..., sources=)``): the
#: retained (rule, proof, target) per identity, REVISE; every proposal as an
#: event dict, CONCORD; each rejection once, by its full key, CONCORD.
TRANSFORMATION_DECISION = declare_page("transformation_decision", (
    RowField("scope", K.SCOPE), RowField("identity", K.LABEL),
), tuple)
TRANSFORMATION_EVENT = declare_page("transformation_event", (
    RowField("scope", K.SCOPE), RowField("serial", K.INDEX),
), dict)
TRANSFORMATION_REJECTION = declare_page("transformation_rejection", (
    RowField("scope", K.SCOPE), RowField("identity", K.LABEL),
    RowField("rule", K.NAME), RowField("proof", K.LABEL),
    RowField("retained_rule", K.NAME), RowField("retained_proof", K.LABEL),
), int)

# ============================================================================
# Step 9, part A: the graphs as views of the book (plan 100, sections 1-2)
# -- owned by the step-9 lane
# ============================================================================

# -- stages ------------------------------------------------------------------
#: A builder that makes a ControlProgram from the graph
#: (``_ordinary_conditional_control_programs``, the loop composer's
#: loop-block construction, ``precompile_to_ssa``'s control-function assembly).
CONTROL_PROGRAM_BUILD = declare_stage("control_program_build")
#: Every rewriter that returns a new ControlProgram tree for an old one
#: (``control_source`` passes, ``_class_surface_ssa_program`` rewriters).
CONTROL_PROGRAM_REWRITE = declare_stage("control_program_rewrite")

# -- transforms: the causes ``_set_operands`` records (plan 70 section 3,
# plan 100 section 1.2).  ``cause`` is recorded by name on the
# ``OperandTransition`` fact; the Append post itself is DERIVED, so arity
# here is documentary (one consumer per edge).
INGEST_EDGE = declare_transform("ingest_edge", 1)
REDUCER_SYNTHESIS = declare_transform("reducer_synthesis", 1)
APPEND_OPERAND = declare_transform("append_operand", 1)
REPLACE_INPUTS = declare_transform("replace_inputs", 1)
REMOVE_NODE = declare_transform("remove_node", 1)
REDIRECT_VALUE = declare_transform("redirect_value", 1)
DISSOLVE_EXPR = declare_transform("dissolve_expr", 1)
DISSOLVE_RETURN = declare_transform("dissolve_return", 1)
DISSOLVE_WRAPPER = declare_transform("dissolve_wrapper", 1)
PARAMETER_INPUT = declare_transform("parameter_input", 1)
FUNCTION_SUBGRAPH_FILTER = declare_transform("function_subgraph_filter", 1)
CANONICAL_RELABEL_OPERANDS = declare_transform("canonical_relabel_operands", 1)
CLASS_TABLE_MEMBER = declare_transform("class_table_member", 1)
PROJECTION_TO_LEAF = declare_transform("projection_to_leaf", 1)
AGGREGATE_MEMBER = declare_transform("aggregate_member", 1)
AGGREGATE_FORMAL_MEMBERS = declare_transform("aggregate_formal_members", 1)
BOUND_RECEIVER = declare_transform("bound_receiver", 1)
SCALAR_INTRINSIC_RECEIVER = declare_transform("scalar_intrinsic_receiver", 1)
CALLSITE_FOLD_REMOVE_NODE = declare_transform("callsite_fold_remove_node", 1)
CALLSITE_FOLD_REPLACE_ALIAS = declare_transform("callsite_fold_replace_alias", 1)
CALLSITE_FOLD_LITERAL = declare_transform("callsite_fold_literal", 1)
REGION_BOUNDARY_INPUT = declare_transform("region_boundary_input", 1)
UNBROADCAST_CHAIN = declare_transform("unbroadcast_chain", 1)
LOOP_BODY_CLONE = declare_transform("loop_body_clone", 1)
LOOP_MATERIALIZER = declare_transform("loop_materializer", 1)
LOOP_EDGE_REBUILD = declare_transform("loop_edge_rebuild", 1)
LOOP_PARENT_REPLACEMENT = declare_transform("loop_parent_replacement", 1)
LOOP_CONTINUATION_REWIRE_OPERANDS = declare_transform(
    "loop_continuation_rewire_operands", 1,
)
#: A control block with no authored construct (the synthesized conditional
#: of ``_class_surface_ssa_program``, a planner-made loop): plan 100, 2.2.
SYNTHESIZED_CONTROL = declare_transform("synthesized_control", 1)

# -- reasons -----------------------------------------------------------------
#: A control block whose owner cell cannot be named (plan 100, 2.2).
CONTROL_OWNER_UNKNOWN = declare_reason("control_owner_unknown")
#: A region marker whose ``deployment_region`` cell is not on the book.
REGION_CELL_UNROUTED = declare_reason("region_cell_unrouted")
#: ``project_control_regions`` collapsed an empty construct: its placement
#: is withdrawn by an edge, never erased.
COLLAPSED_EMPTY_CONSTRUCT = declare_reason("collapsed_empty_construct")
#: A block row whose fact changed between two rewriters with no source cell
#: the api admits as the cause (same cells, none newer): recorded under the
#: latch so the audit lists the rewriter.
CONTROL_BLOCK_REVISION_UNCAUSED = declare_reason("control_block_revision_uncaused")
#: ``post_control_program`` called for a graph with no ``lexical_read_scope``
#: (step 6's FUNCTION_SCOPE row is not on the tree yet).
CONTROL_FUNCTION_SCOPE_UNKNOWN = declare_reason("control_function_scope_unknown")


# -- facts -------------------------------------------------------------------
class ControlBlockKind(Enum):
    """One member per control block class except ``SequenceBlock`` (a
    container, flattened for placement ordinals)."""

    STATEMENT = "statement"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    WHILE = "while"
    LOOP_CONTROL = "loop_control"
    STATE_MACHINE_TICK = "state_machine_tick"
    PARALLEL_DEPLOYMENT = "parallel_deployment"
    CALL = "call"
    DISPATCH = "dispatch"
    RESOURCE_SCOPE = "resource_scope"
    EXTERNAL_REFERENCE_CALL = "external_reference_call"
    VALIDATION = "validation"
    SEQUENCE_MUTATION = "sequence_mutation"
    SEQUENCE_QUERY = "sequence_query"
    SCALAR_FIELD_WRITE = "scalar_field_write"
    STREAM_PUBLISH = "stream_publish"


class Arm(Enum):
    """Which arm of its parent a placed block sits in (plan 100, 2.1)."""

    ROOT_SEQUENCE = "root_sequence"
    BODY = "body"
    ORELSE = "orelse"
    CONDITION = "condition"
    CALLEE = "callee"
    CLEANUP = "cleanup"
    CASE = "case"
    DEFAULT = "default"
    LANE = "lane"
    TERMINAL = "terminal"


#: ``Placement.parent`` for a block placed directly under the program root.
ROOT = "root"


@dataclass(frozen=True)
class ControlBlockFact:
    """A control block's identity-bearing fields as CELLS, never the tree.

    ``predicate``: the predicate value's node cell; ``carried``: the
    carried-alias cells (``control_carried_field`` cells when the block
    carries them, else the carried values' node cells); ``sites``: the
    break / continue / return site cells; ``regions``: the
    ``deployment_region`` cells a marker names (empty for a block whose
    region membership is placement, not identity); ``callsite``: the call's
    ``call_binding`` cell; ``extra``: the non-identity payload (``expect_true``,
    ``comparison``, ``schedule_preference``, ``dtype``, ``induction``, ...).
    """

    kind: ControlBlockKind
    predicate: Any
    carried: tuple
    sites: tuple
    regions: tuple
    callsite: Any
    extra: tuple


@dataclass(frozen=True)
class Placement:
    """Where a block sits: its parent block's cell (or ``ROOT``), the arm
    of that parent, and its ordinal among the arm's siblings after
    ``SequenceBlock`` flattening.  ``case`` / ``lane`` index the arm when
    the parent has several of that kind."""

    parent: Any
    arm: Arm
    ordinal: int
    index: int = 0


@dataclass(frozen=True)
class ControlProgramFact:
    """The program-level structure: region cells (or ordinals when no
    ``deployment_region`` row exists), uniform ids, value aliases, the
    anchor region and the specialized conditionals, as cells where the
    book has them."""

    regions: tuple
    uniforms: tuple
    value_aliases: tuple
    anchor_region: Any
    specialized_conditionals: tuple
    root_blocks: tuple


#: ``CONTROL_PROGRAM.program`` for the shell's one program.
SHELL = "shell"

# -- pages -------------------------------------------------------------------
CONTROL_BLOCK = declare_page("control_block", (
    RowField("function_scope", K.SCOPE), RowField("kind", K.LABEL),
    RowField("owner", K.PAGE_REF),
), ControlBlockFact)                   # CONCORD; a changed fact is a REVISE with its cause
CONTROL_BLOCK_PLACEMENT = declare_page("control_block_placement", (
    RowField("function_scope", K.SCOPE), RowField("block", K.PAGE_REF),
), Placement)                          # Placement | Unresolved; REVISE
CONTROL_PROGRAM = declare_page("control_program", (
    RowField("function_scope", K.SCOPE), RowField("program", K.LABEL),
), ControlProgramFact)                 # REVISE

# ----------------------------------------------------------------------------
# Step 9 Part B: emission as the last layer (plan 100, section 4).  Writers:
# ``emission_concordance.py`` (the recorder and the artifact helpers), called
# beside every append of ``ssa_c_backend`` (and the other backends as they
# are routed).  The function element of a unit / function row is the
# function's SYMBOL: a planned region carries its root's control scope, so
# the value scope cannot key one function's rows (see the module docstring
# of ``emission_concordance``).
# ----------------------------------------------------------------------------

# -- stages ------------------------------------------------------------------
EMISSION_C = declare_stage("emission_c")
EMISSION_LLVM = declare_stage("emission_llvm")
EMISSION_FORTRAN = declare_stage("emission_fortran")
EMISSION_WASM = declare_stage("emission_wasm")
EMISSION_JAVASCRIPT = declare_stage("emission_javascript")
#: ``compile`` / ``compile_standalone`` / ``write``: the files and the
#: command made from the module text.
ARTIFACT_BUILD = declare_stage("artifact_build")
#: ``ssa_llvm_backend._annotate_noalias``: rewrites a define line after its
#: FUNCTION_HEADER unit was posted (the unit is revised under this stage).
LLVM_NOALIAS_ANNOTATION = declare_stage("llvm_noalias_annotation")

# -- reasons -----------------------------------------------------------------
#: A unit spells an SSA value whose identity cell no pass posted (the
#: worklist of plan 100, 4.1 item 2).
VALUE_WITHOUT_IDENTITY_CELL = declare_reason("value_without_identity_cell")
#: The module reached a backend with no attached book (plan 100, R9.2).
NO_BOOK_AT_EMISSION = declare_reason("no_book_at_emission")
#: A BLOCK_LABEL unit while ``ssa_block`` (plan 100, 2.6) is not posted.
BLOCK_ORIGIN_UNROUTED = declare_reason("block_origin_unrouted")
#: ``emission_function`` posted before its units (revised by ``finish``).
FUNCTION_TEXT_PENDING = declare_reason("function_text_pending")
#: A linked LLVM piece whose own MODULE_TEXT is not on this book.
PIECE_ARTIFACT_UNROUTED = declare_reason("piece_artifact_unrouted")
WASM_REGION_UNROUTED = declare_reason("wasm_region_unrouted")
#: An instruction another unit spells (an aggregate projection bound by
#: its call): its own row reads the binding unit.
UNIT_ELIDED = declare_reason("unit_elided")
# ``NO_FUNCTION_SCOPE`` is declared in the step-8 block above.

# -- transforms --------------------------------------------------------------
#: A value ``with_native_sgd_loop`` / ``with_native_adam_loop`` mints for
#: its wrapper (operand: the wrapped root's function cell).
NATIVE_LOOP_WRAPPER_VALUE = declare_transform("native_loop_wrapper_value", 1)
#: The root of kernel / library text a backend pulls in by symbol
#: (``extract_llvm_function``, the intrinsic table, a bounded constant's
#: helper): authored input, as a source span is; no operands.
AUTHORED_KERNEL_TEXT = declare_transform("authored_kernel_text", 0)


# -- facts -------------------------------------------------------------------
class Backend(Enum):
    C_SCALAR = "c_scalar"
    C_MODULE = "c_module"
    LLVM_SCALAR = "llvm_scalar"
    LLVM_MODULE = "llvm_module"
    FORTRAN = "fortran"
    WASM_WAT = "wasm_wat"
    WASM_BINARY = "wasm_binary"
    JAVASCRIPT = "javascript"


class UnitKind(Enum):
    FUNCTION_HEADER = "function_header"
    FORMAL = "formal"
    BLOCK_LABEL = "block_label"
    STATEMENT = "statement"
    INLINED_EXPRESSION = "inlined_expression"
    LITERAL = "literal"
    PHI_EDGE_ASSIGNMENT = "phi_edge_assignment"
    BRANCH = "branch"
    RETURN = "return"
    CALL = "call"
    DECLARATION = "declaration"
    OUTPUT_STORE = "output_store"
    TABLE = "table"
    PROTOTYPE = "prototype"
    #: A kernel / library definition or declaration pulled in by symbol.
    KERNEL_TEXT = "kernel_text"


class ArtifactPart(Enum):
    """``emission_artifact.part``: a member, or ``(member, detail...)`` --
    ``(PIECE_FILE, symbol)``, ``(SOURCE_FILE, "host")``, and any part of a
    standalone build suffixed ``"standalone"``."""

    MODULE_TEXT = "module_text"
    SOURCE_FILE = "source_file"
    PIECE_FILE = "piece_file"
    COMPILE_COMMAND = "compile_command"
    LIBRARY = "library"
    BINARY = "binary"
    API_CONTRACT = "api_contract"
    BUFFER_ORDER = "buffer_order"
    #: ``(symbol, LLVM_MODULE, KERNEL_SOURCE)``: the authored text of a
    #: kernel pulled in by symbol, NOVEL(AUTHORED_KERNEL_TEXT).
    KERNEL_SOURCE = "kernel_source"


class NativeLoop(Enum):
    """Which native loop wrapper keys a wrapper's rows ``(symbol, loop)``."""

    SGD = "sgd"
    ADAM = "adam"


@dataclass(frozen=True)
class NativeLoopValue:
    """One id a native loop wrapper minted for its own buffer: its role
    (``steps``, ``first_moment``, ...) and the parameter it serves."""

    role: str
    parameter: int | None


@dataclass(frozen=True)
class EmittedUnit:
    """``text``: the exact line(s) appended (for LITERAL / INLINED_EXPRESSION
    the expression inlined, which has no line of its own); ``spelling``: the
    token the unit binds its result to (``t{id}``), "" when none."""

    kind: UnitKind
    text: str
    spelling: str


@dataclass(frozen=True)
class FunctionEmission:
    symbol: str
    unit_count: int
    text_sha256: str


@dataclass(frozen=True)
class ArtifactFact:
    """Never the text: its hash and length, and where it went (path parts,
    or the command tuple)."""

    sha256: str
    byte_length: int
    location: tuple


# -- pages -------------------------------------------------------------------
EMISSION_UNIT = declare_page("emission_unit", (
    RowField("function", K.SCOPE), RowField("backend", K.LABEL),
    RowField("unit", K.INDEX),
), EmittedUnit)                        # EmittedUnit | Unresolved(UNIT_ELIDED); CONCORD (REVISE on re-emission)
EMISSION_FUNCTION = declare_page("emission_function", (
    RowField("function", K.SCOPE), RowField("backend", K.LABEL),
), FunctionEmission)                   # Unresolved(FUNCTION_TEXT_PENDING) then FunctionEmission; REVISE
EMISSION_ARTIFACT = declare_page("emission_artifact", (
    RowField("artifact", K.NAME), RowField("backend", K.LABEL),
    RowField("part", K.LABEL),
), ArtifactFact)                       # REVISE
NATIVE_LOOP_VALUE = declare_page("native_loop_value", (
    RowField("wrapper", K.SCOPE), RowField("value", K.VALUE_ID),
), NativeLoopValue)                    # NOVEL(NATIVE_LOOP_WRAPPER_VALUE); CONCORD

# ----------------------------------------------------------------------------
# Step 9 identities: the cells upstream of emission (plan 100, 2.6 and 4.1
# item 2).  Writers: ``precompile_to_ssa`` -- ``_ControlSSABuilder.new_block``
# (every SSA block the control builder makes) and the planned-region assembly
# of ``lower_control_sections_to_ssa`` (the region's one block and the
# ``ssa_value`` row of every value the region body produces).  Reader:
# ``ssa_record_return_state.ssa_block_identity_cell``.
#
# ``ssa_block`` is keyed (function scope, function symbol, label): a planned
# region carries its root's control scope (step 6), so the scope alone would
# put the root's ``entry`` and the region's ``entry`` on one row -- the same
# reason step 9 Part B keys emission rows by the symbol.
# ----------------------------------------------------------------------------

#: The SSA block's role in the construct that made it: one member per block
#: stem ``_ControlSSABuilder.new_block`` is called with (the label is the stem,
#: suffixed ``.N`` for the N-th repeat).
class SSABlockKind(Enum):
    ENTRY = "entry"
    FUNCTION_EXIT = "function_exit"
    IF_TRUE = "if_true"
    IF_FALSE = "if_false"
    IF_MERGE = "if_merge"
    LOOP_HEADER = "loop_header"
    LOOP_BODY = "loop_body"
    LOOP_LATCH = "loop_latch"
    LOOP_EXIT = "loop_exit"
    WHILE_HEADER = "while_header"
    WHILE_BODY = "while_body"
    WHILE_LATCH = "while_latch"
    WHILE_EXIT = "while_exit"
    LOOP_CONTROL_NEXT = "loop_control_next"
    UNREACHABLE_LOOP_CONTROL = "unreachable_loop_control"
    RETURN_CONTROL_NEXT = "return_control_next"
    RETURN_EDGE = "return_edge"
    UNREACHABLE_RETURN_CONTROL = "unreachable_return_control"
    RESOURCE_EXIT = "resource_exit"
    RESOURCE_EXIT_NEXT = "resource_exit_next"
    VALIDATION_PASS = "validation_pass"
    VALIDATION_FAIL = "validation_fail"
    STATE_CASE = "state_case"
    STATE_NEXT = "state_next"
    STATE_MERGE = "state_merge"
    CHILD_TABLE_LIVE = "child_table_live"
    MAPPING_SETDEFAULT_FOUND = "mapping_setdefault_found"
    MAPPING_SETDEFAULT_MISSING = "mapping_setdefault_missing"
    MAPPING_SETDEFAULT_MERGE = "mapping_setdefault_merge"
    SEQUENCE_MUTATION_SELECTED = "sequence_mutation_selected"
    SEQUENCE_MUTATION_SKIPPED = "sequence_mutation_skipped"
    SEQUENCE_MUTATION_MERGE = "sequence_mutation_merge"
    SEQUENCE_MAXIMUM_SELECTED = "sequence_maximum_selected"
    SEQUENCE_MAXIMUM_RETAINED = "sequence_maximum_retained"
    SEQUENCE_MAXIMUM_MERGE = "sequence_maximum_merge"
    SEQUENCE_MAXIMUM_HEADER = "sequence_maximum_header"
    SEQUENCE_MAXIMUM_BODY = "sequence_maximum_body"
    SEQUENCE_MAXIMUM_LOOP_SELECTED = "sequence_maximum_loop_selected"
    SEQUENCE_MAXIMUM_LOOP_RETAINED = "sequence_maximum_loop_retained"
    SEQUENCE_MAXIMUM_LATCH = "sequence_maximum_latch"
    SEQUENCE_MAXIMUM_EXIT = "sequence_maximum_exit"
    SEQUENCE_QUERY_SELECTED = "sequence_query_selected"
    SEQUENCE_QUERY_DEFAULTED = "sequence_query_defaulted"
    SEQUENCE_QUERY_MERGE = "sequence_query_merge"
    SEQUENCE_POP_SELECTED = "sequence_pop_selected"
    SEQUENCE_POP_EMPTY = "sequence_pop_empty"
    SEQUENCE_REMOVE_SCAN = "sequence_remove_scan"
    SEQUENCE_REMOVE_COMPARE = "sequence_remove_compare"
    SEQUENCE_REMOVE_NEXT = "sequence_remove_next"
    SEQUENCE_REMOVE_SHIFT = "sequence_remove_shift"
    SEQUENCE_REMOVE_SHIFT_ROW = "sequence_remove_shift_row"
    SEQUENCE_REMOVE_SHIFT_NEXT = "sequence_remove_shift_next"
    SEQUENCE_REMOVE_STORE_LENGTH = "sequence_remove_store_length"
    SEQUENCE_REMOVE_ABSENT = "sequence_remove_absent"
    SEQUENCE_REMOVE_COMPLETE = "sequence_remove_complete"


#: A block made while lowering a control block that has no ``control_block``
#: row (a container's child the program poster could not own, a
#: ``StateMachineTick``): the fact is ``Unresolved`` and reads the nearest
#: enclosing control cell (else the program / function root cell).
SSA_BLOCK_OWNER_UNROUTED = declare_reason("ssa_block_owner_unrouted")
#: A planned-region value whose id carries the MINTED flag but whose minter
#: posted no mint record (``hierarchical_plan``'s variadic min/max fold,
#: ``fresh_like``): its ``ssa_value`` row is posted origin MINTED and
#: Unsourced with this reason, so the worklist names the minter's gap.
REGION_VALUE_MINT_UNRECORDED = declare_reason("region_value_mint_unrecorded")

#: ``tensor_ssa_lowering.lower_tensor_calls_to_repository_ssa``: replaces
#: tensor operations with explicit kernel calls.
TENSOR_SSA_LOWERING = declare_stage("tensor_ssa_lowering")
#: A value that lowering mints (a kernel's static shape / opcode / stride
#: constant, an ``extent``, a call result buffer): NOVEL on ``ssa_value``
#: from the cell of the tensor operation being lowered (its result's
#: ``ssa_value`` cell, else its first argument's), else the function root.
TENSOR_LOWERING_VALUE = declare_transform("tensor_lowering_value", 1)

SSA_BLOCK = declare_page("ssa_block", (
    RowField("function_scope", K.SCOPE), RowField("function", K.NAME),
    RowField("label", K.LABEL),
), SSABlockKind)                       # SSABlockKind | Unresolved(SSA_BLOCK_OWNER_UNROUTED); CONCORD

# ----------------------------------------------------------------------------
# Symbolic equation outputs (orbital benchmark, work item 1: matrices and
# complicated left-hand sides).  Writer: ``symbolic_equation_compiler.
# compile_sympy_equations``.  Each authored Equality is a NOVEL root on
# ``symbolic_equation`` keyed by (program, equation index); every declared
# output the equation produces -- one per component of a matrix lhs, one for
# a scalar lhs -- is DERIVED from that equation's cell, its fact naming the
# equation index, the component index and the form (assignment to a declared
# name, or the residual ``lhs - rhs`` of a non-name lhs).  ``program`` is
# (law name, digest of the authored set), so two different sets compiled
# under one name never share a row.
# ----------------------------------------------------------------------------

class SymbolicOutputForm(Enum):
    #: lhs is a declared name (a Symbol, or an element of a MatrixSymbol):
    #: the output is the rhs.
    ASSIGNMENT = "assignment"
    #: lhs is any other expression: the output is the residual lhs - rhs,
    #: zero where the equation holds.
    RESIDUAL = "residual"


@dataclass(frozen=True)
class SymbolicEquationFact:
    srepr_sha256: str
    lhs_shape: tuple


@dataclass(frozen=True)
class SymbolicOutputFact:
    equation_index: int
    component: tuple
    form: SymbolicOutputForm


SYMBOLIC_EQUATION = declare_page("symbolic_equation", (
    RowField("program", K.SCOPE), RowField("equation", K.INDEX),
), SymbolicEquationFact)               # NOVEL(INGEST_SOURCE) root; CONCORD
SYMBOLIC_EQUATION_OUTPUT = declare_page("symbolic_equation_output", (
    RowField("program", K.SCOPE), RowField("output", K.NAME),
), SymbolicOutputFact)                 # DERIVED(symbolic_equation cell); CONCORD

# ----------------------------------------------------------------------------
# SymPy ingestion and graph-native differentiation (collocation Jacobian,
# docs/DIFFERENTIATION_FEASIBILITY_2026-10-02.md item 1).
#
# Writers: ``symbolic_process_graph.ingest_sympy_expression(s)`` mints the
# graph's ``ingestion_value_scope`` (as ``graph_express2.build_from_ast``
# does) and posts, per ingested expression, one ``symbolic_ingested_
# expression`` row -- DERIVED from the caller's equation cell (the
# ``symbolic_equation_output`` cell ``compile_sympy_equations`` posts) or a
# NOVEL(INGEST_SOURCE) root when the caller has no equation -- and one
# ``symbolic_subexpression`` row per distinct authored subexpression, keyed
# by the argument path of its first occurrence and DERIVED from its parent's
# cell.  Every node's ``ingestion_value`` row is DERIVED from the cells of the
# authored subexpression it ingests (every expression it occurs in), or of
# the innermost authored subexpression whose ingestion made it (a pairwise
# fold link, a respelling, a declared lowering).  Only the construction
# envelope has no source: ``Unsourced(SYNTHESIZED_NO_SOURCE)``.
#
# ``process_graph_autograd``: the backward graph mints its own scope; each
# adjoint node is NOVEL(ADJOINT_OF, (forward node's cell,)) at stage ADJOINT.
# A forward graph with no identity scope has no cell to name:
# ``Unsourced(FORWARD_GRAPH_UNSCOPED)``.  ``fuse_forward_loss_backward``
# mints the motion's scope; every motion row is DERIVED from the forward or
# backward cell it was copied from (stage FORWARD_LOSS_BACKWARD_FUSION).
# ----------------------------------------------------------------------------

ADJOINT = declare_stage("adjoint")
FORWARD_LOSS_BACKWARD_FUSION = declare_stage("forward_loss_backward_fusion")

#: An adjoint node made for one forward node by its backward rule (the
#: seed, the rule call and its result projections, rule constants, gradient
#: accumulation, shape reductions).  Operand: the forward node's cell.
ADJOINT_OF = declare_transform("adjoint_of", 1)

FORWARD_GRAPH_UNSCOPED = declare_reason("forward_graph_unscoped")


@dataclass(frozen=True)
class SymbolicExpressionFact:
    #: The declared output name the expression is ingested for.
    output: str


@dataclass(frozen=True)
class SymbolicSubexpressionFact:
    #: The SymPy head of the subexpression (``type(value).__name__``).
    head: str


SYMBOLIC_INGESTED_EXPRESSION = declare_page("symbolic_ingested_expression", (
    RowField("ingestion_scope", K.SCOPE), RowField("expression", K.INDEX),
), SymbolicExpressionFact)             # DERIVED(equation cell) | NOVEL(INGEST_SOURCE) root; CONCORD
SYMBOLIC_SUBEXPRESSION = declare_page("symbolic_subexpression", (
    RowField("ingestion_scope", K.SCOPE), RowField("expression", K.INDEX),
    RowField("path", K.LABEL),
), SymbolicSubexpressionFact)          # DERIVED(parent subexpression / expression cell); CONCORD

__all__ =[name for name in dir() if not name.startswith("_") and name not in {
    "ast", "dataclass", "Enum", "Any", "Ref", "RowField", "K", "Unresolved",
    "declare_page", "declare_reason", "declare_stage", "declare_transform",
}]
