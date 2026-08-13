##########################################################################
# Static Single Assignment (SSA) Intermediate Representation (IR) Syntax #
##########################################################################
#
# In SSA form:
#  - Each value is assigned exactly once.
#  - Values are named `%t<ID>`, where <ID> is a unique integer.
#    Internally, we store the integer `id` and reconstruct the textual
#    `%t{id}` when needed.
#  - Instructions (`Instr`) consist of:
#      * `op`: operation name (e.g. 'Add', 'Mul', 'Pow')
#      * `args`: list of input SSA values
#      * `res`: the result SSA value produced
#  - SSA enables clear dataflow analysis, liveness tracking,
#    and optimization passes (constant folding, dead-code elimination,
#    instruction scheduling, etc.).
#
# Example SSA sequence:
#    %t1 = Add %x, %y
#    %t2 = Mul %t1, %c
#    %t3 = Pow %t2, 2
#
# This file provides the core data structures for holding SSA in memory,
# using integer IDs for efficiency.
##########################################################################

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict, Callable, Union
from enum import Enum
from .function_table import FunctionTable
from ..compiler.deployment_frame import DeploymentFrame, DeploymentJoin

# -----------------------------------------------------------------------------
# Core SSA Data Structures
# -----------------------------------------------------------------------------
@dataclass
class SSAValue:
    """
    Represents a single SSA value.

    Attributes:
        id (int):
            Unique integer identifier (maps to textual `%t{id}`).
        dtype (Optional[str]):
            Optional type annotation (e.g. 'float32', 'int64').
    """
    id: int
    dtype: Optional[str] = None
    shape: tuple = ()
    device: Optional[str] = None
    accounting: Dict[str, Any] = field(default_factory=dict)

    def name(self) -> str:
        """Return the textual SSA name in `%t<ID>` form."""
        return f"%t{self.id}"


@dataclass
class Instr:
    """
    Represents a single SSA instruction in a linear sequence.

    Attributes:
        op (str):
            Operation name (e.g. 'Add', 'Mul', 'Pow').
        args (List[SSAValue]):
            Operands for the operation.
        res (SSAValue):
            The result value produced by this instruction.
    """
    op: str
    args: List[SSAValue]
    res: SSAValue
    arg_roles: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    source_span: Optional[Dict[str, Any]] = None

from enum import Enum

from .ssa_registry import Handler, sympy_ssa_name_map, sympy_ssa_disambig, SSARegistry

@dataclass
class BasicBlock:
    name: str
    instrs: List[Instr] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)  # for CFG edges

@dataclass
class Function:
    name: str
    args: List[SSAValue]
    blocks: Dict[str, BasicBlock]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SSAClassMethod:
    """One method of a class as the SSA layer holds it: its dot-name, the
    function-table reference to its body, and -- once that body is lowered into
    the module -- the SSA function that implements it."""

    name: str
    function_reference: int
    function_name: str | None = None


@dataclass(frozen=True)
class SSAClassField:
    """One instance field's addressable slot in a class's layout."""

    name: str
    slot: int


@dataclass(frozen=True)
class SSAClassDefinition:
    """A class definition the SSA module holds: its identity, its instance-field
    layout, and its methods (each pointing at the function that implements it)."""

    identity: str
    fields: tuple[SSAClassField, ...] = ()
    methods: tuple[SSAClassMethod, ...] = ()

    def method(self, name: str) -> "SSAClassMethod | None":
        for member in self.methods:
            if member.name == name:
                return member
        return None


@dataclass(frozen=True)
class SSAClassTable:
    """Every class definition carried into the SSA module -- the SSA-level
    counterpart of the frontend ``ClassNavigationTable``. Holding the definitions
    (not only reference LUTs) is what lets a backend emit a class's methods as
    real, individually linkable functions."""

    classes: tuple[SSAClassDefinition, ...] = ()

    def by_identity(self, identity: str) -> "SSAClassDefinition | None":
        for record in self.classes:
            if record.identity == identity:
                return record
        return None


class SSARecordFieldStorage(str, Enum):
    """Physical SSA storage named by one record field."""

    SCALAR = "scalar"
    SPAN = "span"
    SEQUENCE = "sequence"
    RECORD = "record"
    FUNCTION_TABLE = "function_table"
    CLASS_TABLE = "class_table"


@dataclass(frozen=True)
class SSARecordFieldDescriptor:
    """One typed field correlation inside a raw SSA record.

    This is not an object or a dispatch hook. ``value_ids`` point at ordinary
    SSA arguments/arenas; ``sequence_id`` points at an
    :class:`SSASequenceDescriptor` in the same function-scoped module table.
    """

    name: str
    storage: SSARecordFieldStorage
    storage_identity: str | None = None
    value_ids: tuple[int, ...] = ()
    sequence_id: int | None = None
    record_id: int | None = None
    offset: int | None = None
    dtype: str | None = None
    writable: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "storage", SSARecordFieldStorage(self.storage))
        if self.storage_identity is not None:
            object.__setattr__(
                self, "storage_identity", str(self.storage_identity)
            )
        object.__setattr__(
            self, "value_ids", tuple(map(int, self.value_ids))
        )
        if self.sequence_id is not None:
            object.__setattr__(self, "sequence_id", int(self.sequence_id))
        if self.record_id is not None:
            object.__setattr__(self, "record_id", int(self.record_id))
        if self.offset is not None:
            object.__setattr__(self, "offset", int(self.offset))
        if self.storage is SSARecordFieldStorage.SEQUENCE and self.sequence_id is None:
            raise ValueError("sequence record field requires sequence_id")
        if self.storage is SSARecordFieldStorage.RECORD and self.record_id is None:
            raise ValueError("nested record field requires record_id")

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "storage": self.storage.value,
            "storage_identity": self.storage_identity,
            "value_ids": list(self.value_ids),
            "sequence_id": self.sequence_id,
            "record_id": self.record_id,
            "offset": self.offset,
            "dtype": self.dtype,
            "writable": bool(self.writable),
        }


@dataclass(frozen=True)
class SSARecordDescriptor:
    """A typed grouping of independently stored SSA fields."""

    record_id: int
    identity: str
    fields: tuple[SSARecordFieldDescriptor, ...] = ()
    instance_pool: "SSARecordInstancePoolDescriptor | None" = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", int(self.record_id))
        object.__setattr__(self, "identity", str(self.identity))
        names = tuple(field.name for field in self.fields)
        if len(names) != len(set(names)):
            raise ValueError("SSA record field names must be unique")

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "identity": self.identity,
            "fields": [field.to_mapping() for field in self.fields],
            "instance_pool": (
                None
                if self.instance_pool is None
                else self.instance_pool.to_mapping()
            ),
        }


@dataclass(frozen=True)
class SSARecordInstancePoolField:
    """One physical field layout selected by a shared record handle."""

    storage_identity: str
    storage: SSARecordFieldStorage
    sequence_pool: "SSAChildTablePoolDescriptor | None" = None
    scalar_value_id: int | None = None
    scalar_stride_value_id: int | None = None
    scalar_offset: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "storage_identity", str(self.storage_identity))
        object.__setattr__(self, "storage", SSARecordFieldStorage(self.storage))
        if self.storage is SSARecordFieldStorage.SEQUENCE:
            if self.sequence_pool is None:
                raise ValueError("pooled sequence field requires a child pool")
        elif self.storage is SSARecordFieldStorage.SCALAR:
            if self.scalar_value_id is None or self.scalar_stride_value_id is None:
                raise ValueError("pooled scalar field requires arena and stride")

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "storage_identity": self.storage_identity,
            "storage": self.storage.value,
            "sequence_pool": (
                None if self.sequence_pool is None
                else self.sequence_pool.to_mapping()
            ),
            "scalar_value_id": self.scalar_value_id,
            "scalar_stride_value_id": self.scalar_stride_value_id,
            "scalar_offset": self.scalar_offset,
        }


@dataclass(frozen=True)
class SSARecordInstancePoolDescriptor:
    """Several field layouts addressed by one containing-sequence handle."""

    handle_sequence_id: int
    fields: tuple[SSARecordInstancePoolField, ...]

    def __post_init__(self) -> None:
        identities = tuple(field.storage_identity for field in self.fields)
        if len(identities) != len(set(identities)):
            raise ValueError("record instance-pool field identities must be unique")

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "handle_sequence_id": int(self.handle_sequence_id),
            "fields": [field.to_mapping() for field in self.fields],
        }


@dataclass
class SSARecordTable:
    records: Dict[int, SSARecordDescriptor] = field(default_factory=dict)

    def register(self, descriptor: SSARecordDescriptor) -> SSARecordDescriptor:
        existing = self.records.get(descriptor.record_id)
        if existing is not None and existing != descriptor:
            raise ValueError(f"conflicting SSA record descriptor {descriptor.record_id}")
        self.records[descriptor.record_id] = descriptor
        return descriptor


@dataclass(frozen=True)
class SSATensorDescriptor:
    """Compile-time identity and ABI facts for one logical SSA tensor.

    ``data_value_id`` names ordinary SSA storage. Shape/stride information is
    either static (the tuples) or itself ordinary SSA (the optional value ids).
    Nothing here is a backend object or runtime dispatch handle.
    """

    tensor_id: int
    data_value_id: int
    dtype: str = "float64"
    shape: tuple[int, ...] = ()
    strides: tuple[int, ...] = ()
    shape_value_id: int | None = None
    strides_value_id: int | None = None
    rank_value_id: int | None = None
    element_count_value_id: int | None = None
    layout: str = "dense-row-major"
    storage: str = "temporary"
    metadata_state: str = "static"
    arena_id: int | None = None
    allocation_owner: int | None = None
    owns_allocation: bool = True
    element_offset: int = 0
    byte_offset: int = 0
    byte_size: int | None = None
    alias_of: int | None = None
    writable: bool = True

    def __post_init__(self) -> None:
        if self.layout not in {"dense-row-major", "strided"}:
            raise ValueError(f"unsupported SSA tensor layout {self.layout!r}")
        if self.storage not in {
            "input", "constant", "temporary", "output", "view"
        }:
            raise ValueError(f"unsupported SSA tensor storage {self.storage!r}")
        if self.metadata_state not in {"static", "dynamic", "unresolved"}:
            raise ValueError(
                f"unsupported SSA tensor metadata state {self.metadata_state!r}"
            )
        if any(int(extent) < 0 for extent in self.shape):
            raise ValueError("static SSA tensor extents must be non-negative")
        if self.strides and len(self.strides) != len(self.shape):
            raise ValueError("SSA tensor strides must match static rank")
        if self.element_offset < 0 or self.byte_offset < 0:
            raise ValueError("SSA tensor arena offsets must be non-negative")
        if self.byte_size is not None and self.byte_size < 0:
            raise ValueError("SSA tensor byte size must be non-negative")
        if not self.owns_allocation and self.allocation_owner is None:
            raise ValueError("an SSA tensor view requires an allocation owner")
        if self.metadata_state == "dynamic" and (
            self.shape_value_id is None
            or self.rank_value_id is None
            or self.element_count_value_id is None
        ):
            raise ValueError(
                "dynamic SSA tensors require shape, rank, and element-count values"
            )
        if not self.shape and self.shape_value_id is not None and self.rank_value_id is None:
            raise ValueError("a dynamic SSA tensor shape requires a rank value")

    @property
    def static_rank(self) -> int | None:
        return len(self.shape) if self.metadata_state == "static" else None

    @property
    def static_element_count(self) -> int | None:
        if self.metadata_state != "static":
            return None
        count = 1
        for extent in self.shape:
            count *= int(extent)
        return count

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "tensor_id": int(self.tensor_id),
            "data_value_id": int(self.data_value_id),
            "dtype": self.dtype,
            "shape": list(self.shape),
            "strides": list(self.strides),
            "shape_value_id": self.shape_value_id,
            "strides_value_id": self.strides_value_id,
            "rank_value_id": self.rank_value_id,
            "element_count_value_id": self.element_count_value_id,
            "layout": self.layout,
            "storage": self.storage,
            "metadata_state": self.metadata_state,
            "arena_id": self.arena_id,
            "allocation_owner": self.allocation_owner,
            "owns_allocation": bool(self.owns_allocation),
            "element_offset": int(self.element_offset),
            "byte_offset": int(self.byte_offset),
            "byte_size": self.byte_size,
            "alias_of": self.alias_of,
            "writable": bool(self.writable),
        }


@dataclass
class SSATensorTable:
    """First-class tensor identities owned by one SSA function scope."""

    tensors: Dict[int, SSATensorDescriptor] = field(default_factory=dict)

    def register(self, descriptor: SSATensorDescriptor) -> SSATensorDescriptor:
        tensor_id = int(descriptor.tensor_id)
        existing = self.tensors.get(tensor_id)
        if existing is not None and existing != descriptor:
            raise ValueError(f"conflicting SSA tensor descriptor {tensor_id}")
        self.tensors[tensor_id] = descriptor
        return descriptor

    def by_id(self, tensor_id: int) -> SSATensorDescriptor | None:
        return self.tensors.get(int(tensor_id))

    def by_data_value(self, value_id: int) -> tuple[SSATensorDescriptor, ...]:
        value_id = int(value_id)
        return tuple(
            descriptor
            for descriptor in self.tensors.values()
            if int(descriptor.data_value_id) == value_id
        )


class SSASequenceCapacityPolicy(str, Enum):
    """How a row arena responds when its declared capacity is exhausted."""

    FIXED = "fixed"
    DYNAMIC = "dynamic"


@dataclass(frozen=True)
class SSASequenceDescriptor:
    """Raw SSA storage facts for a variable-length sequence or row table.

    The descriptor is compile-time information, not a runtime container or an
    object model.  ``column_value_ids`` name ordinary SSA arena pointers,
    ``length_address_id`` names the mutable length cell, and
    ``capacity_value_id`` names the available row count.  Empty
    ``key_columns`` means duplicates are allowed; populated key columns make
    insertion unique on those columns.  A live-flags arena is optional so row
    deletion can retain stable indices without mandatory compaction.
    """

    sequence_id: int
    column_value_ids: tuple[int, ...]
    length_address_id: int
    capacity_value_id: int
    status_address_id: int | None = None
    column_dtypes: tuple[str, ...] = ()
    key_columns: tuple[int, ...] = ()
    live_flags_value_id: int | None = None
    capacity_policy: SSASequenceCapacityPolicy = SSASequenceCapacityPolicy.FIXED
    writable: bool = True
    child_table_pool: "SSAChildTablePoolDescriptor | None" = None

    def __post_init__(self) -> None:
        storage_ids = (
            int(self.sequence_id),
            int(self.length_address_id),
            int(self.capacity_value_id),
            *(int(value_id) for value_id in self.column_value_ids),
        )
        if any(value_id < 0 for value_id in storage_ids):
            raise ValueError("SSA sequence value ids must be non-negative")
        if not self.column_value_ids:
            raise ValueError("an SSA sequence requires at least one data column")
        if self.column_dtypes and len(self.column_dtypes) != len(
            self.column_value_ids
        ):
            raise ValueError("SSA sequence dtypes must match its data columns")
        if len(set(self.key_columns)) != len(self.key_columns):
            raise ValueError("SSA sequence key columns must be unique")
        if any(
            int(column) < 0 or int(column) >= len(self.column_value_ids)
            for column in self.key_columns
        ):
            raise ValueError("SSA sequence key column is outside the row layout")
        if self.live_flags_value_id is not None and int(
            self.live_flags_value_id
        ) < 0:
            raise ValueError("SSA sequence live-flags id must be non-negative")
        if self.status_address_id is not None and int(
            self.status_address_id
        ) < 0:
            raise ValueError("SSA sequence status-cell id must be non-negative")
        if not isinstance(self.capacity_policy, SSASequenceCapacityPolicy):
            object.__setattr__(
                self,
                "capacity_policy",
                SSASequenceCapacityPolicy(str(self.capacity_policy)),
            )
        if self.child_table_pool is not None:
            if self.child_table_pool.handle_column < 0 or (
                self.child_table_pool.handle_column >= len(self.column_value_ids)
            ):
                raise ValueError(
                    "nested-table handle column is outside the outer row layout"
                )

    @property
    def allows_duplicates(self) -> bool:
        return not self.key_columns

    @property
    def retains_deleted_rows(self) -> bool:
        return self.live_flags_value_id is not None

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "sequence_id": int(self.sequence_id),
            "column_value_ids": [int(item) for item in self.column_value_ids],
            "length_address_id": int(self.length_address_id),
            "capacity_value_id": int(self.capacity_value_id),
            "status_address_id": self.status_address_id,
            "column_dtypes": list(self.column_dtypes),
            "key_columns": [int(item) for item in self.key_columns],
            "live_flags_value_id": self.live_flags_value_id,
            "capacity_policy": self.capacity_policy.value,
            "writable": bool(self.writable),
            "child_table_pool": (
                None
                if self.child_table_pool is None
                else self.child_table_pool.to_mapping()
            ),
        }


@dataclass(frozen=True)
class SSAChildTablePoolDescriptor:
    """Caller-owned arenas addressed by handles stored in an outer table.

    A handle is an integer child-table row. Child ``h`` owns the slice
    ``h * row_stride : (h + 1) * row_stride`` of every child data/live arena;
    its mutable length/status cells live at index ``h``. This is a raw memory
    contract only—no runtime collection object or dispatch is introduced.
    """

    handle_column: int
    column_value_ids: tuple[int, ...]
    length_value_id: int
    capacity_value_id: int
    row_stride_value_id: int
    status_value_id: int | None = None
    live_flags_value_id: int | None = None
    column_dtypes: tuple[str, ...] = ()
    key_columns: tuple[int, ...] = (0,)
    writable: bool = True

    def __post_init__(self) -> None:
        if not self.column_value_ids:
            raise ValueError("a child-table pool requires a data column")
        ids = (
            *self.column_value_ids,
            self.length_value_id,
            self.capacity_value_id,
            self.row_stride_value_id,
            *((self.status_value_id,) if self.status_value_id is not None else ()),
            *((self.live_flags_value_id,) if self.live_flags_value_id is not None else ()),
        )
        if any(int(value_id) < 0 for value_id in ids):
            raise ValueError("child-table pool value ids must be non-negative")
        if self.column_dtypes and len(self.column_dtypes) != len(
            self.column_value_ids
        ):
            raise ValueError("child-table pool dtypes must match data columns")
        if any(
            int(column) < 0 or int(column) >= len(self.column_value_ids)
            for column in self.key_columns
        ):
            raise ValueError("child-table key column is outside its row layout")

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "handle_column": int(self.handle_column),
            "column_value_ids": list(map(int, self.column_value_ids)),
            "length_value_id": int(self.length_value_id),
            "capacity_value_id": int(self.capacity_value_id),
            "row_stride_value_id": int(self.row_stride_value_id),
            "status_value_id": self.status_value_id,
            "live_flags_value_id": self.live_flags_value_id,
            "column_dtypes": list(self.column_dtypes),
            "key_columns": list(map(int, self.key_columns)),
            "writable": bool(self.writable),
        }


@dataclass
class SSASequenceTable:
    """Function-scoped sequence/table storage descriptions."""

    sequences: Dict[int, SSASequenceDescriptor] = field(default_factory=dict)

    def register(self, descriptor: SSASequenceDescriptor) -> SSASequenceDescriptor:
        sequence_id = int(descriptor.sequence_id)
        existing = self.sequences.get(sequence_id)
        if existing is not None and existing != descriptor:
            raise ValueError(f"conflicting SSA sequence descriptor {sequence_id}")
        self.sequences[sequence_id] = descriptor
        return descriptor

    def by_id(self, sequence_id: int) -> SSASequenceDescriptor | None:
        return self.sequences.get(int(sequence_id))


@dataclass(frozen=True)
class SSADeploymentLane:
    """One independently schedulable lane retained after SSA lowering."""

    index: int
    instruction_sites: tuple[tuple[str, int], ...] = ()
    callees: tuple[str, ...] = ()
    source_region_indices: tuple[int, ...] = ()
    source_value_ids: tuple[int, ...] = ()
    source_node_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class SSADeploymentRegion:
    """Backend-neutral proof that an SSA subgraph may deploy in parallel.

    This is a scheduling permission, not a selected GPU/API backend.  A later
    deployment pass can bind the region to GLSL, CUDA, SIMD, threads, or keep
    the recorded linear schedule without changing program semantics.
    """

    region_id: int
    function: str
    kind: str
    schedule: str
    schedule_preference: str = "alap"
    lanes: tuple[SSADeploymentLane, ...] = ()
    iteration_space: tuple[str, str, str] | None = None
    carried_aliases: tuple[tuple[int, int], ...] = ()
    recursion_region_id: int | None = None
    origin: str = "control_ir"
    source_loop_node_id: int | None = None
    scale: int = 1
    join: DeploymentJoin = DeploymentJoin()
    deploy_site: tuple[str, int] | None = None
    join_site: tuple[str, int] | None = None

    def __post_init__(self) -> None:
        preference = str(self.schedule_preference).lower()
        if preference not in {"asap", "alap"}:
            raise ValueError(
                "deployment schedule preference must be 'asap' or 'alap'"
            )
        object.__setattr__(self, "schedule_preference", preference)
        DeploymentFrame(self.region_id, self.scale, self.join)

    @property
    def frame(self) -> DeploymentFrame:
        return DeploymentFrame(self.region_id, self.scale, self.join)


@dataclass(frozen=True)
class SSACallRecord:
    """One source call occurrence and its complete repository-SSA status.

    A callee definition merely existing in ``functions`` is not execution.
    This record preserves the planner-owned argument/result edges and the
    callee-local storage values its call frame must supply.  ``resolution`` is
    deliberately explicit so a target cannot mistake an omitted call for a
    complete program.
    """

    caller: str
    callsite_id: int
    callee_reference: int | None
    callee_name: str
    callee_symbol: str | None
    argument_bindings: tuple[tuple[int, int], ...] = ()
    result_bindings: tuple[tuple[int, int], ...] = ()
    enclosing_loop_ids: tuple[int, ...] = ()
    callee_storage_value_ids: tuple[int, ...] = ()
    # Callee argument id -> source kind -> source value.  ``caller_value`` is
    # an exact PlanCall binding, ``caller_alias`` is the same binding through
    # the callee identity ledger, and ``default_literal`` is an authored
    # signature default.  Any callee argument absent here is deliberately
    # listed in ``unresolved_frame_value_ids`` and forbids native emission.
    frame_bindings: tuple[tuple[int, str, object], ...] = ()
    unresolved_frame_value_ids: tuple[int, ...] = ()
    resolution: str = "unresolved"
    decomposition: str | None = None

    def __post_init__(self) -> None:
        if self.resolution not in {"unresolved", "native_call", "decomposed"}:
            raise ValueError(f"unknown SSA call resolution {self.resolution!r}")


@dataclass(frozen=True)
class SSAMachineControlLink:
    """One exact machine-state transfer between separately owned CFG regions.

    This is not a source-language call. ``target_function`` names the owning
    repository function or machine funclet, while ``target_address`` preserves
    the architectural destination even when it is an interior PE address.
    """

    source_function: str
    source_block: str
    source_address: int
    edge_role: str
    target_address: int
    target_function: str | None = None
    target_block: str | None = None
    target_kind: str = "unresolved"

    def __post_init__(self) -> None:
        if self.edge_role not in {"true", "false", "direct"}:
            raise ValueError(f"unknown machine control edge role {self.edge_role!r}")
        if self.target_kind not in {
            "runtime-function-entry", "runtime-function-interior",
            "outside-image", "unresolved",
        }:
            raise ValueError(f"unknown machine control target kind {self.target_kind!r}")
        if self.target_kind.startswith("runtime-function") and not self.target_function:
            raise ValueError("resolved machine control link requires target_function")


@dataclass
class SSAMachineControlTable:
    links: tuple[SSAMachineControlLink, ...] = ()

    def from_source(self, function_name: str) -> tuple[SSAMachineControlLink, ...]:
        return tuple(
            link for link in self.links if link.source_function == function_name
        )

    def to_address(self, address: int) -> tuple[SSAMachineControlLink, ...]:
        return tuple(
            link for link in self.links if link.target_address == int(address)
        )


@dataclass(frozen=True)
class SSAMachineIndirectLink:
    """One indirect machine transfer with its strongest proved identity."""

    source_function: str
    source_address: int
    edge_kind: str
    operand_kind: str
    slot_address: int | None = None
    target_kind: str = "dynamic-state"
    target_address: int | None = None
    target_function: str | None = None
    external_identity: str | None = None

    def __post_init__(self) -> None:
        if self.edge_kind not in {"call", "jump"}:
            raise ValueError(f"unknown indirect edge kind {self.edge_kind!r}")
        if self.target_kind not in {
            "internal-function", "pe-import", "unresolved-slot", "dynamic-state",
        }:
            raise ValueError(f"unknown indirect target kind {self.target_kind!r}")
        if self.target_kind == "internal-function" and not self.target_function:
            raise ValueError("internal indirect link requires target_function")
        if self.target_kind == "pe-import" and not self.external_identity:
            raise ValueError("PE import link requires external_identity")


@dataclass
class SSAMachineIndirectTable:
    links: tuple[SSAMachineIndirectLink, ...] = ()

    def from_source(self, function_name: str) -> tuple[SSAMachineIndirectLink, ...]:
        return tuple(
            link for link in self.links if link.source_function == function_name
        )


@dataclass
class IRModule:
    functions: Dict[str, Function]
    function_table: FunctionTable = field(default_factory=FunctionTable)
    # Class definitions the module holds (identity -> fields + methods). Empty
    # for a plain function module; populated when a class navigation table is
    # lowered, so a backend can emit each method as its own function.
    class_table: SSAClassTable = field(default_factory=SSAClassTable)
    # Backend-neutral cache of CFG recursion regions.  Keys are function
    # names; region records identify loop headers, latches, Phi values, and
    # the ProcessGraph SCC from which each loop was lowered.
    recursion_table: Dict[str, Dict[int, Any]] = field(default_factory=dict)
    # Parallel-candidate regions survive lowering beside the ordinary CFG.
    # The SSA instruction stream remains a valid serial fallback.
    deployment_table: Dict[str, tuple[SSADeploymentRegion, ...]] = field(
        default_factory=dict
    )
    # Function-scoped logical tensors. Value ids are only unique within a
    # function, so each function owns its own descriptor table.
    tensor_tables: Dict[str, SSATensorTable] = field(default_factory=dict)
    # Raw variable-length row arenas used by lists, sets, dictionaries, and
    # graph tables.  Policy stays in these compile-time records; emitted SSA
    # contains only addresses, values, comparisons, branches, loads and stores.
    sequence_tables: Dict[str, SSASequenceTable] = field(default_factory=dict)
    # Function-scoped typed record correlations. Record fields point only to
    # ordinary SSA values or other published descriptor tables.
    record_tables: Dict[str, SSARecordTable] = field(default_factory=dict)
    # Every pursued source call occurrence, including those not yet supplied
    # with a complete call-frame storage ABI.  Targets must not silently omit
    # records whose resolution remains ``unresolved``.
    call_table: Dict[str, tuple[SSACallRecord, ...]] = field(default_factory=dict)
    # Exact cross-region machine control transfers. These carry full machine
    # state and must never be rewritten as source calls merely because their
    # destination happens to coincide with a function entry.
    machine_control_table: SSAMachineControlTable = field(
        default_factory=SSAMachineControlTable
    )
    machine_indirect_table: SSAMachineIndirectTable = field(
        default_factory=SSAMachineIndirectTable
    )
    def __post_init__(self) -> None:
        if not self.deployment_table:
            self.deployment_table = {
                name: tuple(function.metadata.get("deployment_regions", ()))
                for name, function in self.functions.items()
                if function.metadata.get("deployment_regions")
            }

# -----------------------------------------------------------------------------
# Correlator for Language <-> SSA Operation Mappings
# -----------------------------------------------------------------------------
class Correlator:
    """
    Bidirectional mapping between language-specific operator names
    and SSA `Handler` enum values.
    """
    def __init__(self):
        # language -> (lang_op_name -> Handler)
        self._lang_to_ssa: Dict[str, Dict[str, Handler]] = {}
        # language -> (Handler -> lang_op_name)
        self._ssa_to_lang: Dict[str, Dict[Handler, str]] = {}

    def register(self, language: str, mapping: Dict[str, Handler]) -> None:
        """
        Register a mapping for a specific language.

        Args:
            language: Identifier for the language (e.g. 'python', 'sympy').
            mapping: Dict of language operator names to `Handler` values.
        """
        lang_map = self._lang_to_ssa.setdefault(language, {})
        ssa_map = self._ssa_to_lang.setdefault(language, {})
        for lang_op, handler in mapping.items():
            lang_map[lang_op] = handler
            ssa_map[handler] = lang_op

    def to_ssa(self, language: str, lang_op: str) -> Optional[Handler]:
        """
        Convert a language-specific operator name to an SSA Handler.
        """
        return self._lang_to_ssa.get(language, {}).get(lang_op)

    def from_ssa(self, language: str, handler: Handler) -> Optional[str]:
        """
        Convert an SSA Handler to its language-specific operator name.
        """
        return self._ssa_to_lang.get(language, {}).get(handler)

# -----------------------------------------------------------------------------
# Pre-SSA Universal IR Schema, Signature, and Handler Registry
# -----------------------------------------------------------------------------
# This module defines the canonical data structures for the compiler’s
# pre-SSA stage and builds a registry of operator definitions.

try:  # heavy optional dependency
    from .operator_defs import operator_signatures, role_schemas, default_funcs
except Exception:  # pragma: no cover - optional
    operator_signatures = {}
    role_schemas = {}
    default_funcs = {}

@dataclass
class RoleSchema:
    """
    Defines how an operation wires its inputs ('up') and outputs ('down').
    """
    up: Dict[str, Union[int, str]]
    down: Dict[str, Union[int, str]]

@dataclass
class Signature:
    """
    Describes operation I/O counts and execution parameters.
    """
    min_inputs: int
    max_inputs: Optional[int]
    min_outputs: int
    max_outputs: Optional[int]
    concurrency: Optional[int]
    allows_inplace: bool
    parameters: List[str] = field(default_factory=list)

@dataclass
class OperatorDef:
    """
    Complete definition of an operation at pre-SSA stage.
    """
    name: str
    role_schema: RoleSchema
    signature: Signature
    handler: Optional[Callable]
    earliest_page: Optional[int] = None
    latest_page: Optional[int] = None

# Central registry: op-name -> OperatorDef
operator_definitions: Dict[str, OperatorDef] = {}

# Build the registry from existing definitions
for op_name, sig_dict in operator_signatures.items():
    schema_dict = role_schemas.get(op_name, {})
    schema = RoleSchema(
        up=schema_dict.get('up', {}),
        down=schema_dict.get('down', {})
    )
    # Prepare signature parameters
    sig_params = dict(sig_dict)
    sig_params.setdefault('parameters', [])
    signature = Signature(
        min_inputs=sig_params['min_inputs'],
        max_inputs=sig_params.get('max_inputs'),
        min_outputs=sig_params['min_outputs'],
        max_outputs=sig_params.get('max_outputs'),
        concurrency=sig_params.get('concurrency'),
        allows_inplace=sig_params.get('allows_inplace', False),
        parameters=sig_params.get('parameters', [])
    )
    handler = default_funcs.get(op_name)
    operator_definitions[op_name] = OperatorDef(
        name=op_name,
        role_schema=schema,
        signature=signature,
        handler=handler
    )

# -----------------------------------------------------------------------------
# Sympy <-> SSA Translation Correlator
# -----------------------------------------------------------------------------
class SympyToSSA:
    """
    Handles translation between SymPy node types and SSA handlers,
    including argument arrangement and schema/signature mapping.

    Attributes:
        name_map (Dict[str, Handler]):
            Maps SymPy node class names to SSA Handler enums.
        arg_order (Dict[str, List[str]]):
            For each SymPy node, specifies the order of argument names
            to extract from node properties for SSA arguments.
        schema_map (Dict[str, RoleSchema]):
            Maps SymPy node names to RoleSchema for argument directionality.
        signature_map (Dict[str, Signature]):
            Maps SymPy node names to Signature for I/O constraints.
        handler_map (Dict[str, Callable]):
            Maps SymPy node names to handler functions (if any).
    """
    def __init__(
        self,
        name_map: Dict[str, Handler],
        arg_order: Optional[Dict[str, List[str]]] = None,
        schema_map: Optional[Dict[str, 'RoleSchema']] = None,
        signature_map: Optional[Dict[str, 'Signature']] = None,
        handler_map: Optional[Dict[str, Callable]] = None,
    ):
        self.name_map = name_map
        self.arg_order = arg_order or {}
        self.schema_map = schema_map or {}
        self.signature_map = signature_map or {}
        self.handler_map = handler_map or {}

    def get_handler(self, sympy_node_name: str) -> Optional[Handler]:
        """Get the SSA Handler for a given SymPy node name."""
        return self.name_map.get(sympy_node_name)

    def get_arg_order(self, sympy_node_name: str) -> List[str]:
        """
        Get the argument extraction order for a SymPy node.
        Returns an empty list if not specified.
        """
        return self.arg_order.get(sympy_node_name, [])

    def get_schema(self, sympy_node_name: str) -> Optional['RoleSchema']:
        """Get the RoleSchema for a SymPy node."""
        return self.schema_map.get(sympy_node_name)

    def get_signature(self, sympy_node_name: str) -> Optional['Signature']:
        """Get the Signature for a SymPy node."""
        return self.signature_map.get(sympy_node_name)

    def get_handler_func(self, sympy_node_name: str) -> Optional[Callable]:
        """Get the handler function for a SymPy node."""
        return self.handler_map.get(sympy_node_name)

    def to_ssa_instr(self, node) -> Optional[Instr]:
        """
        Convert a SymPy node to an SSA Instr, using the mapping and argument arrangement.
        This is a stub; actual implementation depends on node structure.
        """
        handler = self.get_handler(type(node).__name__)
        if handler is None:
            return None
        # Extract arguments in the specified order, or default to node.args
        arg_names = self.get_arg_order(type(node).__name__)
        if arg_names:
            args = [getattr(node, name) for name in arg_names]
        else:
            args = getattr(node, 'args', [])
        # SSAValue wrapping and result assignment would be handled by the caller
        # Return a partially constructed Instr for demonstration
        return Instr(op=str(handler), args=args, res=None)  # res to be filled in by SSA builder



class CaseInsensitiveSympyToSSA(SympyToSSA):
    """
    Case-insensitive version of SympyToSSA for handling SymPy node names.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Convert all keys in name_map to lowercase for case-insensitivity
        self.name_map = {k.lower(): v for k, v in self.name_map.items()}

    def get_handler(self, sympy_node_name: str) -> Optional[Handler]:
        """Get the SSA Handler for a given SymPy node name (case-insensitive)."""
        return super().get_handler(sympy_node_name.lower())
import random
import sys
import json
import argparse
from .ssa_registry import Handler, SSARegistry

# Paths to persist dynamically generated helpers and structured metadata
HELPER_LOG_FILE = "generated_ssa_helpers.py"
METADATA_FILE = "ssa_helper_metadata.json"

# Ensure helper file exists
try:
    with open(HELPER_LOG_FILE, "x") as f:
        f.write("# Auto-generated SSA helper functions\n\n")
except FileExistsError:
    pass

# Ensure metadata file exists
try:
    with open(METADATA_FILE, "x") as f:
        json.dump({}, f, indent=2)
except FileExistsError:
    pass

class DummyBuilder:
    def __init__(self):
        self.instructions = []
    def record(self, instr):
        self.instructions.append(instr)
    def fresh(self, dtype=None):
        val = SSARegistry.new_value(dtype=dtype)
        return val

# Random command compounding: pick a random handler with dummy operands
def command_compound():
    handler = random.choice(list(Handler))
    arg_count = random.randint(1, 3)
    operands = [f"op{i}" for i in range(arg_count)]
    return handler, operands

import ast

def dump_ast_structure(node, indent=0):
    prefix = '  ' * indent
    node_type = type(node).__name__
    print(f"{prefix}{node_type}")
    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            print(f"{prefix}  {field}: [")
            for item in value:
                if isinstance(item, ast.AST):
                    dump_ast_structure(item, indent + 2)
                else:
                    print(f"{prefix}    {repr(item)}")
            print(f"{prefix}  ]")
        elif isinstance(value, ast.AST):
            print(f"{prefix}  {field}:")
            dump_ast_structure(value, indent + 2)
        else:
            print(f"{prefix}  {field}: {repr(value)}")


# Prompt helper to gather comprehensive metadata for future-proof, AST-aware conversion
def prompt_metadata(handler):
    print(f"=== Define metadata for handler: {handler.name} ===", file=sys.stderr)
    description = input("1) One-line description of this handler's semantics: ").strip()
    num_args = input("2) Number of positional operands: ").strip()

    # Keyword args schema
    print("3) Define keyword arguments (format: name:type:description), one per line; empty line to finish:")
    kwargs_schema = {}
    while True:
        line = input().strip()
        if not line:
            break
        name, typ, desc = [p.strip() for p in line.split(":", 2)]
        kwargs_schema[name] = {"type": typ, "description": desc}

    # AST node
    ast_node = input("4) Corresponding AST node class (e.g. ast.Cast), or leave blank: ").strip() or None

    # AST mappings
    print("5) Define AST field mappings (format: name=expression), one per line; empty line to finish:")
    ast_mapping = {}
    while True:
        line = input().strip()
        if not line:
            break
        key, expr = [p.strip() for p in line.split("=", 1)]
        ast_mapping[key] = expr

    return_dtype = input("6) Return dtype (e.g. 'int32', 'float64'): ").strip() or None
    usage_example = input("7) Provide a usage example (code snippet) for this handler: ").strip()
    print("", file=sys.stderr)

    # Load and update structured metadata
    with open(METADATA_FILE, "r+") as mf:
        data = json.load(mf)
        data.setdefault(handler.name, {})
        data[handler.name].update({
            "description": description,
            "num_args": int(num_args),
            "kwargs": kwargs_schema,
            "ast_node": ast_node,
            "ast_mapping": ast_mapping,
            "return_dtype": return_dtype,
            "usage_example": usage_example
        })
        mf.seek(0)
        json.dump(data, mf, indent=2)
        mf.truncate()
    print(f"[INFO] Structured metadata saved for {handler.name}", file=sys.stderr)
    return data[handler.name]

# Safe emit: prompt interactively if helper missing, plus structured metadata
def safe_emit_ssa(handler, builder, operands, **kwargs):
    try:
        return SSARegistry.emit_ssa(handler, builder, operands, **kwargs)
    except KeyError:
        print(f"[WARN] Missing SSA helper for handler: {handler.name}", file=sys.stderr)
        meta = prompt_metadata(handler)
        print("Enter the function body. Use args: builder, operands, **kwargs. End with an empty line.", file=sys.stderr)
        lines = []
        while True:
            line = sys.stdin.readline()
            if not line or not line.strip():
                break
            lines.append(line.rstrip("\n"))
        func_name = f"ssa_helper_{handler.name.lower()}"
        func_lines = [f"def {func_name}(builder, operands, **kwargs):",
                      f"    \"\"\"{meta['description']}\"\"\""]
        for ln in lines:
            func_lines.append(f"    {ln}")
        func_src = "\n".join(func_lines) + "\n"
        with open(HELPER_LOG_FILE, "a") as logf:
            logf.write(func_src + "\n")
        print(f"[INFO] Logged helper to {HELPER_LOG_FILE}:", file=sys.stderr)
        print(func_src, file=sys.stderr)
        local_ns = {}
        exec(func_src, globals(), local_ns)
        SSARegistry.register_helper(handler)(local_ns[func_name])
        print(f"[INFO] Registered new helper for {handler.name}", file=sys.stderr)
        return SSARegistry.emit_ssa(handler, builder, operands, **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSA helper tester with prioritized handlers and usage examples.")
    parser.add_argument('-o', '--operators', nargs='+', default=[],
                        help='List of handler names to include first in the test')
    parser.add_argument('-r', '--repeat', type=int, default=100,
                        help='Number of random commands to run')
    args = parser.parse_args()

    builder = DummyBuilder()
    
    # First, run specified handlers
    if args.operators:  
        print(f"[INFO] Including specified handlers first: {args.operators}", file=sys.stderr)
        for name in args.operators:
            try:
                handler = Handler[name]
            except KeyError:
                print(f"[ERROR] Unknown handler name: {name}", file=sys.stderr)
                continue
            arg_count = random.randint(1, 3)
            operands = [f"op{i}" for i in range(arg_count)]
            result = safe_emit_ssa(handler, builder, operands)
            print(f"Emitted SSA for {handler.name}: {result}")

    # Then, random commands
    for _ in range(args.repeat):
        handler, operands = command_compound()
        try:
            result = safe_emit_ssa(handler, builder, operands)
            print(f"Emitted SSA for {handler.name}: {result}")
        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            continue

