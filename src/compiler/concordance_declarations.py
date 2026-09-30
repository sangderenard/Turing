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

# (step-4 declarations go here)

# ============================================================================
# Step 5: control SSA builder (plan 80, part B) -- owned by the step-5 lane
# ============================================================================

# (step-5 declarations go here)

# ============================================================================
# Steps 6-8: records, linker, tables (plan 90) -- owned by those lanes
# ============================================================================

# (steps 6-8 declarations go here)

__all__ = [name for name in dir() if not name.startswith("_") and name not in {
    "ast", "dataclass", "Enum", "Any", "Ref", "RowField", "K", "Unresolved",
    "declare_page", "declare_reason", "declare_stage", "declare_transform",
}]
