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
from collections.abc import Mapping
from .function_table import FunctionTable
from .dtype_layout import resolve as _resolve_dtype_layout
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


class SSAReferenceKind(str, Enum):
    """Identity domain carried by an opaque SSA reference handle.

    A reference is deliberately distinct from ``ptr``.  ``ptr`` addresses
    repository-owned memory and is valid for Load/Store; an opaque reference
    identifies a program object whose storage or implementation may remain on
    the Python host.  Backends may copy and compare its fixed-width handle but
    must not pretend that the handle is a dereferenceable target pointer.
    """

    STATIC_PYTHON = "static-python"
    FUNCTION = "function"
    OBJECT = "object"


@dataclass(frozen=True)
class SSAReferenceDescriptor:
    """One stable, backend-neutral opaque reference identity."""

    handle: int
    identity: str
    kind: SSAReferenceKind = SSAReferenceKind.OBJECT
    host_resident: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "handle", int(self.handle))
        object.__setattr__(self, "identity", str(self.identity))
        object.__setattr__(self, "kind", SSAReferenceKind(self.kind))

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "handle": self.handle,
            "identity": self.identity,
            "kind": self.kind.value,
            "host_resident": bool(self.host_resident),
        }


@dataclass
class SSAReferenceTable:
    """Opaque handles retained by one SSA function/module surface."""

    references: Dict[int, SSAReferenceDescriptor] = field(default_factory=dict)

    def register(
        self, descriptor: SSAReferenceDescriptor
    ) -> SSAReferenceDescriptor:
        existing = self.references.get(descriptor.handle)
        if existing is not None and existing != descriptor:
            raise ValueError(
                f"conflicting SSA reference handle {descriptor.handle}: "
                f"{existing.identity!r} != {descriptor.identity!r}"
            )
        self.references[descriptor.handle] = descriptor
        return descriptor


class SSARecordFieldStorage(str, Enum):
    """Physical SSA storage named by one record field."""

    SCALAR = "scalar"
    SPAN = "span"
    SEQUENCE = "sequence"
    RECORD = "record"
    REFERENCE = "reference"
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


def _mint_table_owner(book: Any, label: Any) -> tuple[str, int]:
    """The book scope one SSA table's rows live under.

    Value ids are not frame-unique (``id_space.SHARED``), and one function's
    ids reappear in helper tables and copies, so every table is its own
    scope, minted by the book.  ``label`` names the function the table was
    built for, for a reader of the book.
    """

    return book.mint_scope(label or "table")


def new_layout_tables() -> "tuple[SSAStructTable, SSAUnionTable]":
    """A struct table and a union table sharing one book scope.

    The two tables correlate through their member pages (a union embeds
    structs), so they must share a scope.  A frontend that meets its layout
    types before any module exists publishes them into this pair; the module
    assembled later is handed the same pair, not a copy.
    """

    from ..compiler.identity_concordance import current_identity_book

    book = current_identity_book()
    owner = _mint_table_owner(book, "module")
    return (
        SSAStructTable(owner=owner, book=book),
        SSAUnionTable(owner=owner, book=book),
    )


class _BookRows(__import__("collections.abc").abc.MutableMapping):
    """One SSA table's id -> descriptor storage, as rows of a book page.

    Row ``(owner, id)`` holds the descriptor; every write is a revision and a
    removal is a ``None`` revision, so the page is the storage and its whole
    history.  ``on_change(old, new)`` keeps the member page in step.
    """

    def __init__(self, book: Any, page: str, owner: Any, on_change: Callable):
        self._book = book
        self._page = book.page(page)
        self._owner = owner
        self._on_change = on_change

    def __getitem__(self, key: Any) -> Any:
        # Rows are keyed by value ids; anything else (``None`` from an
        # unbound lookup) names no row, exactly as a dict miss did.
        try:
            key = __import__("operator").index(key)
        except TypeError:
            raise KeyError(key) from None
        fact = self._page.latest((self._owner, int(key)))
        if fact is None:
            raise KeyError(key)
        return fact

    def __setitem__(self, key: Any, value: Any) -> None:
        row = (self._owner, int(key))
        old = self._page.latest(row)
        self._page.revise(row, value)
        self._on_change(old, value)

    def __delitem__(self, key: Any) -> None:
        row = (self._owner, int(key))
        old = self._page.latest(row)
        if old is None:
            raise KeyError(key)
        self._page.revise(row, None)
        self._on_change(old, None)

    def __iter__(self):
        for row in self._page.scope_rows(self._owner):
            if self._page.latest(row) is not None:
                yield row[1]

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __repr__(self) -> str:
        return repr(dict(self))


def _revise_member_claims(
    page: Any, owner: Any, old: dict[int, set], new: dict[int, set],
) -> None:
    """Move one descriptor's member claims from ``old`` to ``new``.

    Row ``(owner, member id)`` holds every claim live descriptors make on
    that value; a value can be a member of several records (record SSA
    versions share unchanged field storage).
    """

    for member in set(old) | set(new):
        before = old.get(member, set())
        after = new.get(member, set())
        if before == after:
            continue
        row = (owner, int(member))
        claims = (set(page.latest(row) or ()) - before) | after
        page.revise(row, tuple(sorted(claims, key=repr)))


def record_member_claims(
    descriptor: "SSARecordDescriptor | None",
) -> dict[int, set]:
    """member id -> {(record id, field name, storage identity, role)}.

    A SCALAR field's value ids are SSA versions of one slot (role
    ``("slot",)``); any other field's value ids are positional members
    (``("value", index)``).  A sequence field's descriptor handle is role
    ``("sequence",)`` and a nested record field's record id ``("record",)``.
    """

    claims: dict[int, set] = {}
    if descriptor is None:
        return claims
    record_id = int(descriptor.record_id)
    for field_ in descriptor.fields:
        base = (record_id, field_.name, field_.storage_identity)
        scalar = field_.storage is SSARecordFieldStorage.SCALAR
        for index, value_id in enumerate(field_.value_ids):
            role = ("slot",) if scalar else ("value", index)
            claims.setdefault(int(value_id), set()).add((*base, role))
        if field_.sequence_id is not None:
            claims.setdefault(int(field_.sequence_id), set()).add(
                (*base, ("sequence",))
            )
        if field_.record_id is not None:
            claims.setdefault(int(field_.record_id), set()).add(
                (*base, ("record",))
            )
    return claims


class SSARecordTable:
    """Function-scoped record descriptors, stored on the identity book.

    Page ``record_descriptor`` holds each descriptor at ``(owner, record
    id)``; page ``record_member`` holds, at ``(owner, value id)``, every claim
    a live descriptor makes on that value.  Nothing is kept beside the book.
    """

    def __init__(
        self,
        records: Dict[int, SSARecordDescriptor] | None = None,
        *,
        owner: Any = None,
        book: Any = None,
    ) -> None:
        from ..compiler.identity_concordance import current_identity_book

        self.book = current_identity_book() if book is None else book
        self.owner = (
            owner if isinstance(owner, tuple)
            else _mint_table_owner(self.book, owner)
        )
        member_page = self.book.page("record_member")
        self.records = _BookRows(
            self.book, "record_descriptor", self.owner,
            lambda old, new: _revise_member_claims(
                member_page, self.owner,
                record_member_claims(old), record_member_claims(new),
            ),
        )
        for record_id, descriptor in dict(records or {}).items():
            self.records[int(record_id)] = descriptor

    def member_claims(self, value_id: int) -> tuple:
        """Every (record id, field name, storage identity, role) naming
        ``value_id`` in this table, read from the book."""

        return tuple(
            self.book.page("record_member").latest((self.owner, int(value_id)))
            or ()
        )

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, SSARecordTable) and dict(self.records) == dict(
            other.records
        )

    def __repr__(self) -> str:
        return f"SSARecordTable(owner={self.owner!r}, records={self.records!r})"

    def __deepcopy__(self, memo: dict) -> "SSARecordTable":
        return SSARecordTable(
            dict(self.records), owner=self.owner[0], book=self.book,
        )

    def __reduce__(self):
        # The book scope's serial is not content; a pickle carries the label
        # and the descriptors and is rebuilt on the book current at load.
        return (SSARecordTable, (dict(self.records),), {"owner": self.owner[0]})

    def __setstate__(self, state: dict) -> None:
        self.__init__(dict(self.records), owner=state.get("owner"))

    def register(self, descriptor: SSARecordDescriptor) -> SSARecordDescriptor:
        existing = self.records.get(descriptor.record_id)
        if existing is not None and existing != descriptor:
            compatible_identity = existing.identity == descriptor.identity
            existing_fields = {field.name: field for field in existing.fields}
            incoming_fields = {field.name: field for field in descriptor.fields}
            def same_physical_field(left, right):
                return (
                    left.name == right.name
                    and left.storage == right.storage
                    and left.storage_identity == right.storage_identity
                    and left.value_ids == right.value_ids
                    and left.sequence_id == right.sequence_id
                    and left.record_id == right.record_id
                    and left.offset == right.offset
                    and left.dtype == right.dtype
                )

            compatible_overlap = all(
                same_physical_field(
                    existing_fields[name], incoming_fields[name]
                )
                for name in existing_fields.keys() & incoming_fields.keys()
            )
            compatible_pool = (
                existing.instance_pool is None
                or descriptor.instance_pool is None
                or existing.instance_pool == descriptor.instance_pool
            )
            if compatible_identity and compatible_overlap and compatible_pool:
                # One caller record is observed through several pursued
                # callees, each of which legitimately projects only the
                # fields it touches. Merge those complementary views under
                # the already-correlated record id; this is not an id
                # collision and no field spelling is reinterpreted.
                merged_fields = tuple(
                    SSARecordFieldDescriptor(
                        name=resident.name,
                        storage=resident.storage,
                        storage_identity=resident.storage_identity,
                        value_ids=resident.value_ids,
                        sequence_id=resident.sequence_id,
                        record_id=resident.record_id,
                        offset=resident.offset,
                        dtype=resident.dtype,
                        writable=(
                            bool(resident.writable)
                            or bool(incoming_fields[resident.name].writable)
                            if resident.name in incoming_fields
                            else bool(resident.writable)
                        ),
                    )
                    for resident in existing.fields
                )
                descriptor = SSARecordDescriptor(
                    descriptor.record_id,
                    descriptor.identity,
                    (
                        *merged_fields,
                        *(
                            field for field in descriptor.fields
                            if field.name not in existing_fields
                        ),
                    ),
                    existing.instance_pool or descriptor.instance_pool,
                )
            else:
                overlap_diagnostics = {
                    name: {
                        "existing": existing_fields[name].to_mapping(),
                        "incoming": incoming_fields[name].to_mapping(),
                    }
                    for name in existing_fields.keys() & incoming_fields.keys()
                    if not same_physical_field(
                        existing_fields[name], incoming_fields[name]
                    )
                }
                raise ValueError(
                    f"conflicting SSA record descriptor {descriptor.record_id}: "
                    f"existing_identity={existing.identity!r} "
                    f"existing_fields={tuple(field.name for field in existing.fields)!r} "
                    f"incoming_identity={descriptor.identity!r} "
                    f"incoming_fields={tuple(field.name for field in descriptor.fields)!r} "
                    f"overlap_mismatches={overlap_diagnostics!r} "
                    f"existing_instance_pool={existing.instance_pool!r} "
                    f"incoming_instance_pool={descriptor.instance_pool!r}"
                )
        self.records[descriptor.record_id] = descriptor
        return descriptor


# ---------------------------------------------------------------------------
# Struct and union types: layouts recorded once, spelled per backend.
#
# A record (``SSARecordDescriptor``) is decomposed: it names the SSA values
# that hold its fields and has no storage of its own.  A struct row is the
# opposite: it IS a layout -- size, alignment, and each field's offset -- and
# a value of that type is a base address into which every field is an offset.
#
# SSA does not know bytes.  Every number in a row (size, alignment, offset) is
# a count of UNITS of the row's ``SSALayoutSchema``, the memory system the
# numbers are counted in: its unit of division in bits, which end of a unit a
# narrower member occupies, and the direction offsets run.  Python delivers
# exactly one schema -- the host's -- and every row read from ctypes names it;
# it is not declared by anyone.  The schema a BACKEND targets is a contract
# fact for that backend, and absent it is the usual one, so conversion is the
# identity; another target is the backend's to realize or to refuse visibly.
#
# Rows are written from an intercepted ``ctypes.Structure`` /
# ``ctypes.Union`` subclass (``src/transmogrifier/ctypes_layout.py``): the
# layout ctypes computed for the eager program is the layout the native
# program gets, so the two cannot disagree about a unit.  Backends only spell
# the row: C as ``struct``/``union`` with ``_Alignas``; LLVM as a named struct
# type, or units at the row's alignment with typed access at the offsets;
# Fortran as a ``bind(C)`` derived type.  A union is a set of struct members
# sharing one payload: its ``storage_member`` is the strictest-aligned member
# (ties by declaration order), the member a backend without unions lays down
# first and pads to ``size`` (Clang's own lowering of a C union).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SSALayoutSchema:
    """The memory system a layout row's numbers are counted in.

    ``unit_bits`` is the unit of division: every size, alignment and offset in
    a row is a whole number of these.  ``justification`` says which end of a
    unit a member narrower than the unit occupies (``"low"`` or ``"high"``),
    which is what decides how a narrow member read through a wider one lands.
    ``direction`` is the way offsets run from the base.  ``preferred_spans``
    are the cache-ideal sizes in units: advisory, they never change what a
    row means, only what a backend may choose for placement.
    """

    unit_bits: int
    justification: str = "low"
    direction: str = "up"
    preferred_spans: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit_bits", int(self.unit_bits))
        object.__setattr__(self, "justification", str(self.justification))
        object.__setattr__(self, "direction", str(self.direction))
        object.__setattr__(
            self, "preferred_spans", tuple(int(v) for v in self.preferred_spans)
        )
        if self.unit_bits <= 0:
            raise ValueError("layout schema: unit_bits must be positive")
        if self.justification not in {"low", "high"}:
            raise ValueError(
                f"layout schema: justification must be 'low' or 'high', "
                f"not {self.justification!r}"
            )
        if self.direction not in {"up", "down"}:
            raise ValueError(
                f"layout schema: direction must be 'up' or 'down', "
                f"not {self.direction!r}"
            )
        if any(span <= 0 for span in self.preferred_spans):
            raise ValueError("layout schema: preferred_spans must be positive")

    def units(self, bits: int, what: str) -> int:
        """``bits`` as a whole number of units; refuse rather than round."""

        bits = int(bits)
        if bits % self.unit_bits:
            raise ValueError(
                f"{what}: {bits} bits is not a whole number of "
                f"{self.unit_bits}-bit units of this layout schema"
            )
        return bits // self.unit_bits

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "unit_bits": self.unit_bits,
            "justification": self.justification,
            "direction": self.direction,
            "preferred_spans": list(self.preferred_spans),
        }


def _check_leaf_sizes(
    identity: str, schema: "SSALayoutSchema", members: tuple,
) -> None:
    """A leaf member's size is its dtype's size counted in the schema's
    units (the dtype authority declares bits; the schema divides them)."""

    for member in members:
        if member.dtype is None:
            continue
        layout = _resolve_dtype_layout(member.dtype)
        expected = schema.units(
            layout.byte_size * 8,
            f"{identity}: member {member.name!r} of dtype {layout.name!r}",
        )
        if member.size != expected:
            raise ValueError(
                f"{identity}: member {member.name!r}: size {member.size} "
                f"disagrees with dtype {layout.name!r} ({expected} units)"
            )


@dataclass(frozen=True)
class SSAStructFieldDescriptor:
    """One member of a struct or union row.

    A leaf member has a repository ``dtype``; an aggregate member names the
    nested row through ``struct_id`` or ``union_id`` instead.  ``count`` > 1 is
    a fixed in-line array of ``count`` such elements (``c_double * 3``).
    ``offset`` is the member's offset inside the containing row -- zero for
    every member of a union -- and ``size`` the member's own (per-element)
    size, both in units of the containing row's schema.
    """

    name: str
    offset: int
    size: int
    dtype: str | None = None
    struct_id: int | None = None
    union_id: int | None = None
    count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "offset", int(self.offset))
        object.__setattr__(self, "size", int(self.size))
        object.__setattr__(self, "count", int(self.count))
        if self.struct_id is not None:
            object.__setattr__(self, "struct_id", int(self.struct_id))
        if self.union_id is not None:
            object.__setattr__(self, "union_id", int(self.union_id))
        kinds = sum(
            1 for marker in (self.dtype, self.struct_id, self.union_id)
            if marker is not None
        )
        if kinds != 1:
            raise ValueError(
                f"struct field {self.name!r} must be exactly one of a leaf "
                f"dtype, a nested struct or a nested union"
            )
        if self.offset < 0 or self.size <= 0 or self.count <= 0:
            raise ValueError(
                f"struct field {self.name!r}: offset/size/count must be "
                f"non-negative/positive/positive"
            )
        if self.dtype is not None:
            # A leaf's dtype is a spelling the one dtype authority declares
            # (``dtype_layout``); the row stores the canonical name, and the
            # container checks the member's size against that dtype's size
            # in the schema's units.  No local dtype table.
            layout = _resolve_dtype_layout(self.dtype)
            if layout is None:
                raise ValueError(
                    f"struct field {self.name!r}: dtype {self.dtype!r} is not "
                    f"declared in dtype_layout"
                )
            object.__setattr__(self, "dtype", layout.name)

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "offset": self.offset,
            "size": self.size,
            "dtype": self.dtype,
            "struct_id": self.struct_id,
            "union_id": self.union_id,
            "count": self.count,
        }


@dataclass(frozen=True)
class SSAStructDescriptor:
    """A struct row: a named layout, counted in its schema's units."""

    struct_id: int
    identity: str
    schema: SSALayoutSchema
    size: int
    alignment: int
    fields: tuple[SSAStructFieldDescriptor, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "struct_id", int(self.struct_id))
        object.__setattr__(self, "identity", str(self.identity))
        object.__setattr__(self, "size", int(self.size))
        object.__setattr__(self, "alignment", int(self.alignment))
        object.__setattr__(self, "fields", tuple(self.fields))
        names = tuple(field.name for field in self.fields)
        if len(names) != len(set(names)):
            raise ValueError(f"struct {self.identity}: field names must be unique")
        if self.size <= 0 or self.alignment <= 0:
            raise ValueError(f"struct {self.identity}: size and alignment must be positive")
        if self.alignment & (self.alignment - 1):
            raise ValueError(f"struct {self.identity}: alignment must be a power of two")
        if self.size % self.alignment:
            raise ValueError(f"struct {self.identity}: size must be a multiple of alignment")
        _check_leaf_sizes(f"struct {self.identity}", self.schema, self.fields)
        for field in self.fields:
            if field.offset + field.size * field.count > self.size:
                raise ValueError(
                    f"struct {self.identity}: field {field.name!r} overruns the row"
                )

    def field(self, name: str) -> "SSAStructFieldDescriptor | None":
        for member in self.fields:
            if member.name == name:
                return member
        return None

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "struct_id": self.struct_id,
            "identity": self.identity,
            "schema": self.schema.to_mapping(),
            "size": self.size,
            "alignment": self.alignment,
            "fields": [field.to_mapping() for field in self.fields],
        }


@dataclass(frozen=True)
class SSAUnionDescriptor:
    """A union row: a set of struct members sharing one payload.

    Every member is a struct row (a scalar alternative is a one-field struct,
    synthesized at interception, so the recipe is uniform).  ``storage_member``
    names the member a backend without unions lays down as the payload's
    storage type: the strictest-aligned member, ties broken by declaration
    order.  ``size`` is the largest member rounded up to ``alignment``, in
    units of the row's schema.
    """

    union_id: int
    identity: str
    schema: SSALayoutSchema
    size: int
    alignment: int
    members: tuple[SSAStructFieldDescriptor, ...] = ()
    storage_member: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "union_id", int(self.union_id))
        object.__setattr__(self, "identity", str(self.identity))
        object.__setattr__(self, "size", int(self.size))
        object.__setattr__(self, "alignment", int(self.alignment))
        object.__setattr__(self, "members", tuple(self.members))
        object.__setattr__(self, "storage_member", str(self.storage_member))
        if not self.members:
            raise ValueError(f"union {self.identity}: needs at least one member")
        names = tuple(member.name for member in self.members)
        if len(names) != len(set(names)):
            raise ValueError(f"union {self.identity}: member names must be unique")
        for member in self.members:
            if member.struct_id is None:
                raise ValueError(
                    f"union {self.identity}: member {member.name!r} is not a "
                    f"struct row (every union member is a struct)"
                )
            if member.offset != 0:
                raise ValueError(
                    f"union {self.identity}: member {member.name!r} is not at offset 0"
                )
            if member.size > self.size:
                raise ValueError(
                    f"union {self.identity}: member {member.name!r} overruns the payload"
                )
        if self.storage_member not in names:
            raise ValueError(
                f"union {self.identity}: storage_member {self.storage_member!r} "
                f"is not a member"
            )
        if self.size <= 0 or self.alignment <= 0 or (self.alignment & (self.alignment - 1)):
            raise ValueError(f"union {self.identity}: bad size/alignment")
        if self.size % self.alignment:
            raise ValueError(f"union {self.identity}: size must be a multiple of alignment")

    def member(self, name: str) -> "SSAStructFieldDescriptor | None":
        for candidate in self.members:
            if candidate.name == name:
                return candidate
        return None

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "union_id": self.union_id,
            "identity": self.identity,
            "schema": self.schema.to_mapping(),
            "size": self.size,
            "alignment": self.alignment,
            "members": [member.to_mapping() for member in self.members],
            "storage_member": self.storage_member,
        }


def _layout_member_claims(
    descriptor: Any, container_kind: str, members: tuple,
) -> dict[int, set]:
    """nested row id -> {(container id, member name, offset, role)}.

    The layout analogue of ``record_member_claims``: a struct or union row
    claims every nested row one of its members is laid out from.  Role
    ``("struct",)`` says the nested id is a struct row, ``("union",)`` a
    union row (a leaf member claims nothing; it has no row of its own).
    ``container_kind`` is carried in the claim so one member page can be
    read back to the row that made the claim.
    """

    claims: dict[int, set] = {}
    if descriptor is None:
        return claims
    container_id = int(
        descriptor.struct_id if container_kind == "struct" else descriptor.union_id
    )
    for member in members:
        base = (container_kind, container_id, member.name, int(member.offset))
        if member.struct_id is not None:
            claims.setdefault(int(member.struct_id), set()).add((*base, ("struct",)))
        if member.union_id is not None:
            claims.setdefault(int(member.union_id), set()).add((*base, ("union",)))
    return claims


def struct_member_claims(descriptor: "SSAStructDescriptor | None") -> dict[int, set]:
    """Every nested-row claim one struct row makes (see ``_layout_member_claims``)."""

    return _layout_member_claims(
        descriptor, "struct", () if descriptor is None else descriptor.fields,
    )


def union_member_claims(descriptor: "SSAUnionDescriptor | None") -> dict[int, set]:
    """Every nested-row claim one union row makes (see ``_layout_member_claims``)."""

    return _layout_member_claims(
        descriptor, "union", () if descriptor is None else descriptor.members,
    )


class _SSALayoutTable:
    """Module-wide byte-layout rows, stored on the identity book.

    The struct and union tables are the two instances.  Page
    ``<kind>_descriptor`` holds each row at ``(owner, row id)``; page
    ``<kind>_member`` holds, at ``(owner, nested row id)``, every claim a live
    row of this kind makes on a nested row; page ``layout_state`` holds, at
    ``(owner, kind, row id)``, whether the row's layout is ``resolved`` (with
    the edge that produced it), ``invalidated`` (a row it is laid out from
    changed underneath it) or ``superseded`` (its identity moved to another
    id); page ``layout_supersession`` is the edge page: one row per
    re-declaration, fact ``(incumbent, replacement)``.  Nothing is kept
    beside the book.

    A struct row and a union row that contain each other are correlated
    through the member pages, so the two tables of one module share one
    ``owner`` scope (``IRModule`` mints it); a table built alone mints its
    own.
    """

    kind: str
    descriptor_page: str
    member_page: str
    id_attribute: str

    def __init__(
        self,
        rows: Dict[int, Any] | None = None,
        *,
        owner: Any = None,
        book: Any = None,
    ) -> None:
        from ..compiler.identity_concordance import current_identity_book

        self.book = current_identity_book() if book is None else book
        self.owner = (
            owner if isinstance(owner, tuple)
            else _mint_table_owner(self.book, owner or "module")
        )
        member_page = self.book.page(self.member_page)
        claims_of = self._member_claims
        self._rows = _BookRows(
            self.book, self.descriptor_page, self.owner,
            lambda old, new: _revise_member_claims(
                member_page, self.owner, claims_of(old), claims_of(new),
            ),
        )
        for row_id, descriptor in dict(rows or {}).items():
            self._publish(int(row_id), descriptor, None)

    # -- book pages ---------------------------------------------------------

    @staticmethod
    def _member_claims(descriptor: Any) -> dict[int, set]:  # pragma: no cover
        raise NotImplementedError

    def _state_page(self) -> Any:
        return self.book.page("layout_state")

    def _row_id(self, descriptor: Any) -> int:
        return int(getattr(descriptor, self.id_attribute))

    def _publish(self, row_id: int, descriptor: Any, edge_row: Any) -> None:
        self._rows[row_id] = descriptor
        state_row = (self.owner, self.kind, row_id)
        fact = ("resolved", descriptor, edge_row)
        if self._state_page().latest(state_row) != fact:
            self._state_page().revise(state_row, fact)

    def layout_state(self, row_id: int) -> Any:
        """The latest ``layout_state`` fact for ``row_id`` (None if never declared)."""

        return self._state_page().latest((self.owner, self.kind, int(row_id)))

    def supersessions(self) -> tuple[tuple[Any, Any], ...]:
        """Every re-declaration edge this table recorded: ``(edge row, (incumbent, replacement))``."""

        page = self.book.page("layout_supersession")
        return tuple(
            (row, page.latest(row))
            for row in page.scope_rows(self.owner)
            if len(row) == 5 and row[1] == self.kind
        )

    def member_claims(self, row_id: int) -> tuple:
        """Every (container kind, container id, member name, offset, role)
        naming nested row ``row_id`` from a live row of this kind."""

        return tuple(
            self.book.page(self.member_page).latest((self.owner, int(row_id)))
            or ()
        )

    # -- registration -------------------------------------------------------

    def register(self, descriptor: Any, *, stage: Any = "declaration") -> Any:
        """Publish ``descriptor``; a re-declaration is an edge, never a rewrite.

        The same row (same id or same identity) declared again with the same
        layout is a no-op.  Declared again with a different layout, the row is
        revised (its history stays on the page), one edge is appended to
        ``layout_supersession`` and every row laid out from it has its
        ``layout_state`` withdrawn -- ``withdraw_superseded_layout_derivations``,
        the layout counterpart of ``withdraw_superseded_shape_derivations``.
        Two different identities under one id is an id collision and raises.
        """

        row_id = self._row_id(descriptor)
        incumbent = self._rows.get(row_id)
        superseded_id = row_id
        if incumbent is not None and incumbent.identity != descriptor.identity:
            raise ValueError(
                f"conflicting SSA {self.kind} descriptor {row_id} "
                f"({incumbent.identity!r} vs {descriptor.identity!r})"
            )
        if incumbent is None:
            incumbent = self.by_identity(descriptor.identity)
            if incumbent is not None:
                superseded_id = self._row_id(incumbent)
        if incumbent is not None and incumbent == descriptor:
            return incumbent
        edge_row = None
        if incumbent is not None:
            edge_row = self._record_supersession(
                superseded_id, incumbent, row_id, descriptor, stage,
            )
            if superseded_id != row_id:
                # The identity moved to a new id: the old row is removed (a
                # ``None`` revision -- its history stays) and its state says
                # where the identity went.
                del self._rows[superseded_id]
                self._state_page().revise(
                    (self.owner, self.kind, superseded_id),
                    ("superseded", (self.kind, row_id), str(stage)),
                )
        self._publish(row_id, descriptor, edge_row)
        if incumbent is not None:
            self.withdraw_superseded_layout_derivations(
                superseded_id, reason=stage,
            )
        return descriptor

    def _record_supersession(
        self, source_id: int, incumbent: Any, target_id: int, replacement: Any,
        stage: Any,
    ) -> tuple:
        """Append the edge ``incumbent -> replacement`` to ``layout_supersession``.

        Columns are compile time (monotonic across the page), as on
        ``shape_transformation_concordance``.
        """

        edge_page = self.book.page("layout_supersession")
        edge_row = (self.owner, self.kind, int(target_id), int(source_id), str(stage))
        edge_fact = (incumbent, replacement)
        if edge_page.latest(edge_row) != edge_fact:
            edge_page.set(
                edge_row, max(edge_page.columns, default=-1) + 1, edge_fact,
            )
        return edge_row

    def withdraw_superseded_layout_derivations(
        self, row_id: int, *, reason: Any,
    ) -> None:
        """Carry a changed layout along every row laid out from it.

        The member pages record which rows embed ``row_id``; each such row's
        size, alignment and member offsets were derived from the layout that
        just changed, so its ``layout_state`` is withdrawn (``invalidated``)
        in the same causal step and the withdrawal continues to the rows that
        embed those.  The frontend's next declaration of a withdrawn row
        appends its new generation.
        """

        state_page = self._state_page()
        pending = [(self.kind, int(row_id))]
        visited: set[tuple[str, int]] = set()
        while pending:
            kind, nested_id = pending.pop()
            if (kind, nested_id) in visited:
                continue
            visited.add((kind, nested_id))
            for member_page in ("struct_member", "union_member"):
                claims = self.book.page(member_page).latest(
                    (self.owner, nested_id)
                ) or ()
                for container_kind, container_id, _name, _offset, role in claims:
                    if role != (kind,):
                        continue
                    state_row = (self.owner, str(container_kind), int(container_id))
                    fact = state_page.latest(state_row)
                    if not (isinstance(fact, tuple) and fact and fact[0] == "resolved"):
                        # Already withdrawn (its dependents went with it) or
                        # never declared: nothing derived from it remains.
                        continue
                    state_page.revise(
                        state_row, ("invalidated", (kind, nested_id), str(reason)),
                    )
                    pending.append((str(container_kind), int(container_id)))

    # -- lookup -------------------------------------------------------------

    def by_id(self, row_id: int) -> Any:
        return self._rows.get(int(row_id))

    def by_identity(self, identity: str) -> Any:
        for descriptor in self._rows.values():
            if descriptor.identity == str(identity):
                return descriptor
        return None

    # -- identity -----------------------------------------------------------

    def __eq__(self, other: Any) -> bool:
        return type(other) is type(self) and dict(self._rows) == dict(other._rows)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(owner={self.owner!r}, rows={self._rows!r})"

    def __deepcopy__(self, memo: dict) -> "_SSALayoutTable":
        # A module's struct and union tables share one owner; one deepcopy of
        # the module copies both, so the scope the first copy mints is kept
        # in the memo for the second under the old owner.
        key = ("layout_owner", self.owner)
        if key not in memo:
            memo[key] = _mint_table_owner(self.book, self.owner[0])
        return type(self)(dict(self._rows), owner=memo[key], book=self.book)

    def __reduce__(self):
        # The book scope's serial is not content; a pickle carries the label
        # and the descriptors and is rebuilt on the book current at load.
        return (type(self), (dict(self._rows),), {"owner": self.owner[0]})

    def __setstate__(self, state: dict) -> None:
        self.__init__(dict(self._rows), owner=state.get("owner"))


class SSAStructTable(_SSALayoutTable):
    """Module-wide struct rows on the book.  A type is one row however many
    functions use it.  ``structs`` is the live id -> row view."""

    kind = "struct"
    descriptor_page = "struct_descriptor"
    member_page = "struct_member"
    id_attribute = "struct_id"
    _member_claims = staticmethod(struct_member_claims)

    @property
    def structs(self) -> Dict[int, SSAStructDescriptor]:
        return self._rows

    def register(
        self, descriptor: SSAStructDescriptor, *, stage: Any = "declaration",
    ) -> SSAStructDescriptor:
        if not isinstance(descriptor, SSAStructDescriptor):
            raise TypeError(f"struct table takes SSAStructDescriptor, not {descriptor!r}")
        return super().register(descriptor, stage=stage)


class SSAUnionTable(_SSALayoutTable):
    """Module-wide union rows on the book; every member points into the
    struct table.  ``unions`` is the live id -> row view."""

    kind = "union"
    descriptor_page = "union_descriptor"
    member_page = "union_member"
    id_attribute = "union_id"
    _member_claims = staticmethod(union_member_claims)

    @property
    def unions(self) -> Dict[int, SSAUnionDescriptor]:
        return self._rows

    def register(
        self, descriptor: SSAUnionDescriptor, *, stage: Any = "declaration",
    ) -> SSAUnionDescriptor:
        if not isinstance(descriptor, SSAUnionDescriptor):
            raise TypeError(f"union table takes SSAUnionDescriptor, not {descriptor!r}")
        return super().register(descriptor, stage=stage)


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
    # Logical shape of one row in each physical column.  Scalar columns use
    # ``()``.  A shaped column is still one contiguous arena: row ``i`` starts
    # at ``i * prod(column_shapes[column])``.  This preserves one storage
    # identity while allowing a resident comprehension of vectors to become
    # a matrix view without allocating a parallel object.
    column_shapes: tuple[tuple[int, ...], ...] = ()
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
        if self.column_shapes and len(self.column_shapes) != len(
            self.column_value_ids
        ):
            raise ValueError("SSA sequence shapes must match its data columns")
        if any(
            any(not isinstance(extent, int) or extent <= 0 for extent in shape)
            for shape in self.column_shapes
        ):
            raise ValueError("SSA sequence row shapes require positive static extents")
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
            "column_shapes": [list(shape) for shape in self.column_shapes],
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
    # A nested Tensor row carries the same three ordinary-SSA extent facts as
    # every other dynamic tensor. ``shape_value_id`` is a flattened shape
    # arena, ``rank_value_id`` is indexed by child handle, and
    # ``shape_stride_value_id`` gives the shape-arena row width.  Non-tensor
    # child tables leave all three absent.
    shape_value_id: int | None = None
    rank_value_id: int | None = None
    shape_stride_value_id: int | None = None
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
            *((self.shape_value_id,) if self.shape_value_id is not None else ()),
            *((self.rank_value_id,) if self.rank_value_id is not None else ()),
            *((self.shape_stride_value_id,) if self.shape_stride_value_id is not None else ()),
            *((self.status_value_id,) if self.status_value_id is not None else ()),
            *((self.live_flags_value_id,) if self.live_flags_value_id is not None else ()),
        )
        if any(int(value_id) < 0 for value_id in ids):
            raise ValueError("child-table pool value ids must be non-negative")
        extent_members = (
            self.shape_value_id,
            self.rank_value_id,
            self.shape_stride_value_id,
        )
        if any(value is not None for value in extent_members) and not all(
            value is not None for value in extent_members
        ):
            raise ValueError(
                "nested tensor child pool requires shape, rank, and shape-stride identities"
            )
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
            "shape_value_id": self.shape_value_id,
            "rank_value_id": self.rank_value_id,
            "shape_stride_value_id": self.shape_stride_value_id,
            "status_value_id": self.status_value_id,
            "live_flags_value_id": self.live_flags_value_id,
            "column_dtypes": list(self.column_dtypes),
            "key_columns": list(map(int, self.key_columns)),
            "writable": bool(self.writable),
        }


def sequence_member_roles(
    descriptor: "SSASequenceDescriptor | None",
) -> dict[int, set]:
    """member id -> {(sequence id, role)} for one sequence descriptor.

    Roles: ``("handle",)``, ``("column", position)`` and one per extent or
    status cell (``length_address_id``, ``capacity_value_id``,
    ``status_address_id``, ``live_flags_value_id``).
    """

    claims: dict[int, set] = {}
    if descriptor is None:
        return claims
    sequence_id = int(descriptor.sequence_id)

    def claim(value_id: Any, role: tuple) -> None:
        if value_id is not None:
            claims.setdefault(int(value_id), set()).add((sequence_id, role))

    claim(sequence_id, ("handle",))
    for position, column in enumerate(descriptor.column_value_ids):
        claim(column, ("column", position))
    for attribute in (
        "length_address_id", "capacity_value_id",
        "status_address_id", "live_flags_value_id",
    ):
        claim(getattr(descriptor, attribute, None), (attribute,))
    return claims


class SSASequenceTable:
    """Function-scoped sequence/table storage descriptions, on the book.

    Page ``sequence_descriptor`` holds each descriptor at ``(owner, sequence
    id)``; page ``sequence_member`` holds, at ``(owner, value id)``, every
    ``(sequence id, role)`` a live descriptor gives that value; page
    ``sequence_column_claims`` holds every column typing ever offered.
    """

    def __init__(
        self,
        sequences: Dict[int, SSASequenceDescriptor] | None = None,
        *,
        owner: Any = None,
        book: Any = None,
    ) -> None:
        from ..compiler.identity_concordance import current_identity_book

        self.book = current_identity_book() if book is None else book
        self.owner = (
            owner if isinstance(owner, tuple)
            else _mint_table_owner(self.book, owner)
        )
        member_page = self.book.page("sequence_member")
        self.sequences = _BookRows(
            self.book, "sequence_descriptor", self.owner,
            lambda old, new: _revise_member_claims(
                member_page, self.owner,
                sequence_member_roles(old), sequence_member_roles(new),
            ),
        )
        for sequence_id, descriptor in dict(sequences or {}).items():
            self.sequences[int(sequence_id)] = descriptor

    def member_claims(self, value_id: int) -> tuple:
        """Every (sequence id, role) naming ``value_id``, read from the book."""

        return tuple(
            self.book.page("sequence_member").latest(
                (self.owner, int(value_id))
            ) or ()
        )

    def offered_column_dtypes(self, sequence_id: int) -> list:
        """Every column-dtype tuple offered for ``sequence_id``, in order."""

        page = self.book.page("sequence_column_claims")
        return [
            fact[0] for _column, fact in page.history(
                (self.owner, int(sequence_id), "column_dtypes")
            )
        ]

    def register(self, descriptor: SSASequenceDescriptor) -> SSASequenceDescriptor:
        sequence_id = int(descriptor.sequence_id)
        existing = self.sequences.get(sequence_id)
        # Every attempt, not just the winner: a conflict report that shows
        # only incumbent-vs-newcomer cannot distinguish two sites that
        # stably disagree from a sequence of sites that flip a value back
        # and forth, and those need opposite fixes.
        self.book.page("sequence_column_claims").revise(
            (self.owner, sequence_id, "column_dtypes"),
            (
                tuple(descriptor.column_dtypes),
                tuple(descriptor.key_columns),
            ),
        )
        if existing is not None and existing != descriptor:
            # Name the id the way a reader can act on -- ``minted#1000013548``
            # rather than 2305843010213707500 -- and say WHICH fields the two
            # registrations disagree about, since "conflicting" alone sends
            # the reader back to re-derive that by hand from a build log.
            from ..compiler.id_space import label as _id_label

            differing = tuple(sorted(
                name for name in vars(descriptor)
                if getattr(existing, name, None) != getattr(descriptor, name)
            ))
            detail = "; ".join(
                f"{name}: incumbent={getattr(existing, name, None)!r} "
                f"vs new={getattr(descriptor, name)!r}"
                for name in differing
            )
            offered = self.offered_column_dtypes(sequence_id)
            raise ValueError(
                f"conflicting SSA sequence descriptor {_id_label(sequence_id)}"
                f" (differs in: {', '.join(differing) or 'identity only'})"
                + (f" [{detail}]" if detail else "")
                + f" dtypes offered in order: {offered}"
            )
        self.sequences[sequence_id] = descriptor
        return descriptor

    def by_id(self, sequence_id: int) -> SSASequenceDescriptor | None:
        return self.sequences.get(int(sequence_id))

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, SSASequenceTable) and dict(
            self.sequences
        ) == dict(other.sequences)

    def __repr__(self) -> str:
        return (
            f"SSASequenceTable(owner={self.owner!r}, "
            f"sequences={self.sequences!r})"
        )

    def __deepcopy__(self, memo: dict) -> "SSASequenceTable":
        return SSASequenceTable(
            dict(self.sequences), owner=self.owner[0], book=self.book,
        )

    def __reduce__(self):
        return (
            SSASequenceTable, (dict(self.sequences),),
            {"owner": self.owner[0]},
        )

    def __setstate__(self, state: dict) -> None:
        self.__init__(dict(self.sequences), owner=state.get("owner"))


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


class _BookCallList(__import__("collections.abc").abc.MutableSequence):
    """One caller's call records as a mutable list whose storage is a book
    row: every mutation revises the row to the whole new tuple."""

    def __init__(self, page: Any, row: Any) -> None:
        self._page = page
        self._row = row

    def _records(self) -> tuple:
        return tuple(self._page.latest(self._row) or ())

    def _commit(self, records: Any) -> None:
        self._page.revise(self._row, tuple(records))

    def __getitem__(self, index: Any) -> Any:
        records = self._records()
        return list(records[index]) if isinstance(index, slice) else records[index]

    def __setitem__(self, index: Any, value: Any) -> None:
        records = list(self._records())
        records[index] = value
        self._commit(records)

    def __delitem__(self, index: Any) -> None:
        records = list(self._records())
        del records[index]
        self._commit(records)

    def __len__(self) -> int:
        return len(self._records())

    def insert(self, index: int, value: Any) -> None:
        records = list(self._records())
        records.insert(index, value)
        self._commit(records)

    def sort(self, *, key: Any = None, reverse: bool = False) -> None:
        self._commit(sorted(self._records(), key=key, reverse=reverse))

    def copy(self) -> list:
        return list(self._records())

    def __eq__(self, other: Any) -> bool:
        return isinstance(
            other, (list, tuple, _BookCallList)
        ) and self._records() == tuple(other)

    def __repr__(self) -> str:
        return repr(list(self._records()))

    def __reduce__(self):
        return (list, (list(self._records()),))


class SSACallTable(__import__("collections.abc").abc.MutableMapping):
    """caller symbol -> that caller's SSACallRecords, stored on the book.

    Page ``call_record`` holds each caller's records at ``(owner, caller)``;
    every change is a revision.  ``mutable`` tables (the linker's working
    table) hand out list views whose in-place edits are revisions; module
    tables hand out tuples.
    """

    def __init__(
        self,
        records: Any = None,
        *,
        owner: Any = None,
        mutable: bool = False,
        book: Any = None,
    ) -> None:
        from ..compiler.identity_concordance import current_identity_book

        self.book = current_identity_book() if book is None else book
        self.owner = (
            owner if isinstance(owner, tuple)
            else _mint_table_owner(self.book, owner)
        )
        self.mutable = bool(mutable)
        self._page = self.book.page("call_record")
        for caller, caller_records in dict(records or {}).items():
            self[caller] = caller_records

    def __getitem__(self, caller: Any) -> Any:
        row = (self.owner, str(caller))
        records = self._page.latest(row)
        if records is None:
            raise KeyError(caller)
        return _BookCallList(self._page, row) if self.mutable else records

    def __setitem__(self, caller: Any, records: Any) -> None:
        row = (self.owner, str(caller))
        records = tuple(records)
        if self._page.latest(row) != records:
            self._page.revise(row, records)

    def __delitem__(self, caller: Any) -> None:
        row = (self.owner, str(caller))
        if self._page.latest(row) is None:
            raise KeyError(caller)
        self._page.revise(row, None)

    def __iter__(self):
        for row in self._page.scope_rows(self.owner):
            if self._page.latest(row) is not None:
                yield row[1]

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def setdefault(self, caller: Any, default: Any = ()) -> Any:
        # The stored view, never ``default`` itself: an append to the
        # returned list must land on the book.
        if caller not in self:
            self[caller] = default
        return self[caller]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Mapping):
            return False
        return {k: tuple(v) for k, v in self.items()} == {
            k: tuple(v) for k, v in other.items()
        }

    def __repr__(self) -> str:
        return f"SSACallTable(owner={self.owner!r}, {dict(self.items())!r})"

    def __reduce__(self):
        return (
            _rebuild_call_table,
            (
                {k: tuple(v) for k, v in self.items()},
                self.owner[0], self.mutable,
            ),
        )

    def __deepcopy__(self, memo: dict) -> "SSACallTable":
        return SSACallTable(
            {k: tuple(v) for k, v in self.items()},
            owner=self.owner[0], mutable=self.mutable, book=self.book,
        )


def _rebuild_call_table(records: Any, label: Any, mutable: bool) -> SSACallTable:
    return SSACallTable(records, owner=label, mutable=mutable)


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
    # Module-wide byte-layout types.  A struct row is a named layout (size,
    # alignment, member offsets) intercepted from a ctypes.Structure; a union
    # row is a set of struct members sharing one payload.  One row per type
    # however many functions hold a value of it; backends spell the rows.
    # Both tables live on the identity book under ONE owner scope minted in
    # ``__post_init__`` (a struct row and the union that embeds it are
    # correlated through the shared member pages); ``None`` here means "mint
    # them", a mapping assigned later is coerced onto the book like
    # ``call_table``.
    struct_table: SSAStructTable = None  # type: ignore[assignment]
    union_table: SSAUnionTable = None  # type: ignore[assignment]
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
    # Opaque program-object identities. Their signed i64 handles can be
    # copied, compared, and stored by native backends; ``host_resident``
    # records the explicit boundary at which dereference remains Python-owned.
    reference_tables: Dict[str, SSAReferenceTable] = field(default_factory=dict)
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
    # Compiler-wide receipts that describe source-to-module decisions.  This
    # is distinct from Function.metadata: a source transform can affect the
    # relationship among several lowered functions and must remain observable
    # even when a backend selects only one function for emission.
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __setattr__(self, name: str, value: Any) -> None:
        # The module's call table is stored on the identity book whatever a
        # caller assigns to it.
        if name == "call_table" and not isinstance(value, SSACallTable):
            value = SSACallTable(value, owner="module")
        # So are the struct and union tables: a plain mapping of rows is
        # published onto the book under the module's layout scope.
        if name == "struct_table" and value is not None and not isinstance(
            value, SSAStructTable
        ):
            value = SSAStructTable(dict(value), owner=self._layout_owner())
        if name == "union_table" and value is not None and not isinstance(
            value, SSAUnionTable
        ):
            value = SSAUnionTable(dict(value), owner=self._layout_owner())
        object.__setattr__(self, name, value)

    def _layout_owner(self) -> Any:
        """The one book scope this module's struct and union rows share.

        Taken from whichever layout table already exists, else a fresh
        ``("module", n)`` scope on the current book.  ``None`` while neither
        table exists yet lets the constructor mint one scope for both.
        """

        for table in (
            self.__dict__.get("struct_table"), self.__dict__.get("union_table"),
        ):
            if table is not None:
                return table.owner
        from ..compiler.identity_concordance import current_identity_book

        return _mint_table_owner(current_identity_book(), "module")

    def __post_init__(self) -> None:
        if self.struct_table is None or self.union_table is None:
            owner = self._layout_owner()
            book = (self.struct_table or self.union_table)
            book = None if book is None else book.book
            if self.struct_table is None:
                self.struct_table = SSAStructTable(owner=owner, book=book)
            if self.union_table is None:
                self.union_table = SSAUnionTable(owner=owner, book=book)
        if not self.deployment_table:
            self.deployment_table = {
                name: tuple(function.metadata.get("deployment_regions", ()))
                for name, function in self.functions.items()
                if function.metadata.get("deployment_regions")
            }

    # -- named byte-layout types ---------------------------------------------

    def declare_struct(
        self, descriptor: SSAStructDescriptor, *, stage: Any = "declaration",
    ) -> SSAStructDescriptor:
        """Publish one struct row on the book (see ``SSAStructTable.register``).

        ``stage`` names the pass declaring it (``"ctypes_interception"``, a
        frontend's own name); it is written on the supersession edge when the
        row re-declares an existing layout.
        """

        return self.struct_table.register(descriptor, stage=stage)

    def declare_union(
        self, descriptor: SSAUnionDescriptor, *, stage: Any = "declaration",
    ) -> SSAUnionDescriptor:
        """Publish one union row on the book (see ``SSAUnionTable.register``)."""

        return self.union_table.register(descriptor, stage=stage)

    def layout_row(
        self, kind: str, row_id: int,
    ) -> "SSAStructDescriptor | SSAUnionDescriptor | None":
        """The live row ``(kind, id)`` names -- ``kind`` is ``"struct"`` or ``"union"``."""

        if kind == "struct":
            return self.struct_table.by_id(row_id)
        if kind == "union":
            return self.union_table.by_id(row_id)
        raise ValueError(f"layout kind must be 'struct' or 'union', not {kind!r}")

    def layout_row_by_identity(
        self, identity: str,
    ) -> "tuple[str, SSAStructDescriptor | SSAUnionDescriptor] | None":
        """``(kind, row)`` for the authored type spelling ``identity``, or None."""

        row = self.struct_table.by_identity(identity)
        if row is not None:
            return ("struct", row)
        row = self.union_table.by_identity(identity)
        if row is not None:
            return ("union", row)
        return None

    def reachable_functions(
        self,
        *roots: str,
        follow_call: "Callable[[Instr], str | None] | None" = None,
    ) -> tuple[str, ...]:
        """Function names reachable from ``roots``, in this module's order.

        Reachability is MEMBERSHIP; ORDER belongs to the module.  The walk
        follows each instruction's ``callee`` attribute -- a backend narrows
        which calls are edges by passing ``follow_call``, returning the
        callee name to follow or ``None`` -- and the result is reported in
        ``functions`` insertion order.  Handing consumers a set here made
        every one of them invent an ordering, and one of them iterated the
        set itself: a per-process-random function order in emitted code.
        """

        def default_follow(instruction: "Instr") -> str | None:
            callee = instruction.attributes.get("callee")
            if callee is not None and str(callee) in self.functions:
                return str(callee)
            return None

        follow = follow_call or default_follow
        members: set[str] = set()
        pending = [str(root) for root in roots]
        while pending:
            name = pending.pop()
            if name in members or name not in self.functions:
                continue
            members.add(name)
            for block in self.functions[name].blocks.values():
                for instruction in block.instrs:
                    followed = follow(instruction)
                    if followed is not None:
                        pending.append(str(followed))
        return tuple(name for name in self.functions if name in members)

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

