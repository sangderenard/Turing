"""A language-agnostic class schema: the OOP input harness.

The purpose is to move *objects and classes* between ecosystems without
sacrificing them to a class-object table. SSA already holds a class table
(``SSAClassTable``/``SSAClassDefinition``), and it is deliberately minimal --
identity, an addressable field layout, and method-to-function links:

    SSAClassField   = (name, slot)
    SSAClassMethod  = (name, function_reference, function_name)

That is enough to *emit* a class whose methods are individually linkable
functions, and not enough to *reconstruct* one: it carries no field types, no
method signatures, no bases, no construction. A Java or C++ class reduced to
it arrives at the far side unable to say what its fields hold or what its
methods accept.

``ClassSchema`` is the richer form every source language reduces into and
every destination instantiates from. It is proved *against* the SSA table
rather than replacing it:

  * ``to_ssa_class_definition`` must produce exactly the SSA record a direct
    frontend lowering would produce -- the schema is a superset, never a
    divergence.
  * ``from_ssa_class_definition`` recovers everything the SSA form still
    carries, and marks the rest ``UNKNOWN`` rather than inventing it. A round
    trip through SSA is therefore lossy in a *stated* way, and
    ``ssa_projection_agrees`` checks the surviving part still matches.

Nothing here executes or dispatches. A schema is a description a destination
reads to build its own construct -- in nodus, a module whose rows are the
fields and whose bound tool carries the behavior. The destination supplies
its own implementations; only structure and canonical names travel.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from ..transmogrifier.ssa import (
    SSAClassDefinition,
    SSAClassField,
    SSAClassMethod,
    SSAClassTable,
)

# A type the source language stated but this layer has not mapped to a
# canonical spelling. Deliberately one marker, not a taxonomy: an unmapped
# type must be visible and must never be silently defaulted to a numeric.
UNKNOWN_TYPE = "unknown"

# Where a schema came from. Free text, not an enum: a new frontend should not
# need to amend a central list to be describable.
ORIGIN_UNSTATED = "unstated"


@dataclass(frozen=True)
class FieldSchema:
    """One instance field.

    ``slot`` is the addressable position the SSA layout assigns. It is carried
    (not recomputed) so a schema that came from SSA reproduces that layout
    exactly; ``slot=None`` means "the destination may assign one".
    """

    name: str
    type_name: str = UNKNOWN_TYPE
    slot: int | None = None
    # Source-stated initial value, when the language provides one literally.
    # Absent is different from zero, so this is Optional rather than 0.
    initial: Any | None = None
    readonly: bool = False

    def with_slot(self, slot: int) -> "FieldSchema":
        return replace(self, slot=int(slot))


@dataclass(frozen=True)
class ParameterSchema:
    """One formal parameter of a method."""

    name: str
    type_name: str = UNKNOWN_TYPE
    # A default the source stated; absent means the parameter is required.
    default: Any | None = None
    has_default: bool = False


@dataclass(frozen=True)
class MethodSchema:
    """One method: its signature, and a reference to the body that implements it.

    ``function_reference`` is the SSA function-table index when the body has
    been lowered; ``body_reference`` is a destination-agnostic name for the
    same body (a graph id, a tool id, a symbol). One of them being absent is
    normal -- a schema arriving from a source language has a body reference
    long before anything has been lowered into an SSA function table.
    """

    name: str
    parameters: tuple[ParameterSchema, ...] = ()
    returns: str = UNKNOWN_TYPE
    function_reference: int | None = None
    function_name: str | None = None
    body_reference: str | None = None
    is_static: bool = False
    is_constructor: bool = False


@dataclass(frozen=True)
class ClassSchema:
    """A class as something transportable between ecosystems."""

    identity: str
    fields: tuple[FieldSchema, ...] = ()
    methods: tuple[MethodSchema, ...] = ()
    bases: tuple[str, ...] = ()
    origin_language: str = ORIGIN_UNSTATED
    # Anything a frontend knows and this schema has no column for. Kept so a
    # language can round-trip its own detail without the schema growing a
    # field per language.
    annotations: Mapping[str, Any] = field(default_factory=dict)

    def field_named(self, name: str) -> FieldSchema | None:
        return next((f for f in self.fields if f.name == name), None)

    def method_named(self, name: str) -> MethodSchema | None:
        return next((m for m in self.methods if m.name == name), None)

    def constructor(self) -> MethodSchema | None:
        return next((m for m in self.methods if m.is_constructor), None)

    # -- reduction to the SSA form (the proving partner) -------------------

    def assign_slots(self) -> "ClassSchema":
        """Give every slotless field the next free layout position.

        Fields that already carry a slot keep it, so a schema recovered from
        SSA re-reduces to the identical layout.
        """

        taken = {f.slot for f in self.fields if f.slot is not None}
        assigned: list[FieldSchema] = []
        cursor = 0
        for member in self.fields:
            if member.slot is not None:
                assigned.append(member)
                continue
            while cursor in taken:
                cursor += 1
            taken.add(cursor)
            assigned.append(member.with_slot(cursor))
        return replace(self, fields=tuple(assigned))

    def to_ssa_class_definition(self) -> SSAClassDefinition:
        """The SSA record this class reduces to. Lossy by design, not by accident."""

        laid_out = self.assign_slots()
        return SSAClassDefinition(
            identity=str(laid_out.identity),
            fields=tuple(
                SSAClassField(name=str(f.name), slot=int(f.slot or 0))
                for f in laid_out.fields
            ),
            methods=tuple(
                SSAClassMethod(
                    name=str(m.name),
                    function_reference=int(m.function_reference or 0),
                    function_name=m.function_name,
                )
                for m in laid_out.methods
            ),
        )

    @staticmethod
    def from_ssa_class_definition(
        definition: SSAClassDefinition,
        *,
        origin_language: str = ORIGIN_UNSTATED,
    ) -> "ClassSchema":
        """Recover what the SSA form still carries; mark the rest unknown.

        Everything absent here is absent because SSA does not hold it, which
        is the fact this schema exists to make visible.
        """

        return ClassSchema(
            identity=str(definition.identity),
            fields=tuple(
                FieldSchema(name=str(f.name), type_name=UNKNOWN_TYPE, slot=int(f.slot))
                for f in definition.fields
            ),
            methods=tuple(
                MethodSchema(
                    name=str(m.name),
                    parameters=(),          # SSA holds no signature
                    returns=UNKNOWN_TYPE,
                    function_reference=int(m.function_reference),
                    function_name=m.function_name,
                )
                for m in definition.methods
            ),
            origin_language=origin_language,
            annotations={"recovered_from": "ssa_class_table"},
        )

    def ssa_projection_agrees(self, definition: SSAClassDefinition) -> tuple[str, ...]:
        """Differences between this schema's SSA reduction and ``definition``.

        Empty means the schema projects exactly onto the SSA record -- the
        agreement a frontend must satisfy before its classes are trusted to
        cross. Reported as messages rather than a bool so a mismatch names
        itself.
        """

        produced = self.to_ssa_class_definition()
        problems: list[str] = []
        if produced.identity != definition.identity:
            problems.append(
                f"identity {produced.identity!r} != {definition.identity!r}"
            )
        produced_fields = {f.name: f.slot for f in produced.fields}
        expected_fields = {f.name: f.slot for f in definition.fields}
        for name, slot in expected_fields.items():
            if name not in produced_fields:
                problems.append(f"field {name!r} missing from schema")
            elif produced_fields[name] != slot:
                problems.append(
                    f"field {name!r} slot {produced_fields[name]} != {slot}"
                )
        for name in produced_fields.keys() - expected_fields.keys():
            problems.append(f"field {name!r} not in the SSA definition")
        produced_methods = {m.name: m.function_reference for m in produced.methods}
        expected_methods = {m.name: m.function_reference for m in definition.methods}
        for name, reference in expected_methods.items():
            if name not in produced_methods:
                problems.append(f"method {name!r} missing from schema")
            elif produced_methods[name] != reference:
                problems.append(
                    f"method {name!r} function_reference "
                    f"{produced_methods[name]} != {reference}"
                )
        for name in produced_methods.keys() - expected_methods.keys():
            problems.append(f"method {name!r} not in the SSA definition")
        return tuple(problems)


def schemas_from_ssa_class_table(
    table: SSAClassTable, *, origin_language: str = ORIGIN_UNSTATED,
) -> tuple[ClassSchema, ...]:
    return tuple(
        ClassSchema.from_ssa_class_definition(
            definition, origin_language=origin_language
        )
        for definition in table.classes
    )


def ssa_class_table_from_schemas(
    schemas: Sequence[ClassSchema],
) -> SSAClassTable:
    return SSAClassTable(
        classes=tuple(schema.to_ssa_class_definition() for schema in schemas)
    )


# -- derivation from ingestion capture -------------------------------------
#
# ``ProcessGraph.build_from_ast`` captures every class definition at ingestion
# time -- before and independent of pursuit -- as a map_ir object record
# (``graph_express2._class_schema_from_ast``): attributes with instance/class
# storage and source annotations, methods with graph identities. That record
# is the authoritative source-side description of a class, so a ClassSchema
# derives from it rather than from live-object reflection.

def class_schema_from_map_ir_object(
    record: Mapping[str, Any], *, origin_language: str = "python",
) -> ClassSchema:
    """A transportable schema from one captured class record.

    Only what the capture states crosses: attribute names keep their source
    annotation when one was written, and ``UNKNOWN`` otherwise; methods carry
    their graph identity as the body reference the destination binds against.
    Parameter lists are not part of the capture, so none are invented.
    """

    return ClassSchema(
        identity=str(record.get("class_identity") or record["class_name"]),
        fields=tuple(
            FieldSchema(
                name=str(attribute["name"]),
                # Whitespace is meaningless in Python type syntax and fatal in
                # the whitespace-separated wire form, so it is collapsed.
                type_name="".join(
                    str(attribute.get("annotation") or UNKNOWN_TYPE).split()
                ),
            )
            for attribute in record.get("attributes", ())
        ),
        methods=tuple(
            MethodSchema(
                name=str(method["name"]),
                body_reference=str(method.get("graph_identity") or method["name"]),
                # The receiver is implicit at the destination (dispatch passes
                # the instance), so a leading ``self`` is not an argument.
                parameters=tuple(
                    ParameterSchema(name=str(parameter))
                    for parameter in method.get("parameters", ())
                    if parameter != "self"
                ),
            )
            for method in record.get("methods", ())
        ),
        origin_language=origin_language,
        annotations={"derived_from": "map_ir_object"},
    )


def class_schemas_from_process_graph(
    graph: Any, *, origin_language: str = "python",
) -> tuple[ClassSchema, ...]:
    """Every class an ingested ProcessGraph captured, as transportable schemas."""

    records = graph.G.graph.get("map_ir", {}).get("objects", ())
    return tuple(
        class_schema_from_map_ir_object(record, origin_language=origin_language)
        for record in records
    )


# -- SCHEMA V1 interchange -------------------------------------------------
#
# The wire form the destination reads (nodus: include/generic_object.h). One
# record per line, whitespace separated, in the same idiom as KIRTEXT and
# NODUSPKG: a producer needs nothing but the ability to write lines, and a
# malformed document fails by line rather than by guess.
#
#   schema 1
#   class Linear python
#   base Module
#   field W float32 0
#   field b float32 1 readonly
#   method forward abstract_tensor.matmul 1

SCHEMA_TEXT_VERSION = "1"


def serialize_schema_text(schema: "ClassSchema") -> str:
    """Emit ``schema`` in the destination's wire form.

    Slots are assigned first so the emitted layout is the one the schema
    actually reduces to -- a destination must never have to invent a layout
    the SSA side would have assigned differently.
    """

    laid_out = schema.assign_slots()
    lines = [f"schema {SCHEMA_TEXT_VERSION}"]
    identity = str(laid_out.identity)
    origin = str(laid_out.origin_language or "").strip()
    lines.append(f"class {identity} {origin}" if origin else f"class {identity}")
    for base in laid_out.bases:
        lines.append(f"base {base}")
    for member in laid_out.fields:
        record = f"field {member.name} {member.type_name} {int(member.slot or 0)}"
        if member.readonly:
            record += " readonly"
        lines.append(record)
    for member in laid_out.methods:
        # A method names the implementation the DESTINATION owns. Prefer the
        # destination-agnostic reference; fall back to the SSA function name.
        # Never emit a body: behaviour is bound at the far side, not shipped.
        reference = member.body_reference or member.function_name or member.name
        record = f"method {member.name} {reference} {len(member.parameters)}"
        if member.is_constructor:
            record += " constructor"
        lines.append(record)
    return "\n".join(lines) + "\n"


def parse_schema_text(text: str) -> "ClassSchema":
    """Read the wire form back. Mirrors nodus's parse_class_schema exactly."""

    identity = ""
    origin = ORIGIN_UNSTATED
    bases: list[str] = []
    fields: list[FieldSchema] = []
    methods: list[MethodSchema] = []
    saw_header = False

    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        record = parts[0]
        if not saw_header:
            if record != "schema" or len(parts) < 2 or parts[1] != SCHEMA_TEXT_VERSION:
                raise ValueError(f"schema line {number}: expected 'schema 1' header")
            saw_header = True
            continue
        if record == "class":
            if len(parts) < 2:
                raise ValueError(f"schema line {number}: class needs an identity")
            identity = parts[1]
            if len(parts) > 2:
                origin = parts[2]
        elif record == "base":
            if len(parts) < 2:
                raise ValueError(f"schema line {number}: base needs a name")
            bases.append(parts[1])
        elif record == "field":
            if len(parts) < 3:
                raise ValueError(f"schema line {number}: field needs a name and a type")
            slot = int(parts[3]) if len(parts) > 3 and parts[3].lstrip("-").isdigit() else None
            flags = parts[4:] if slot is not None else parts[3:]
            unknown = [f for f in flags if f != "readonly"]
            if unknown:
                raise ValueError(f"schema line {number}: unknown field flag {unknown[0]}")
            fields.append(FieldSchema(
                name=parts[1], type_name=parts[2], slot=slot,
                readonly="readonly" in flags,
            ))
        elif record == "method":
            if len(parts) < 3:
                raise ValueError(
                    f"schema line {number}: method needs a name and a body reference"
                )
            count = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0
            flags = parts[4:] if len(parts) > 3 and parts[3].isdigit() else parts[3:]
            unknown = [f for f in flags if f != "constructor"]
            if unknown:
                raise ValueError(f"schema line {number}: unknown method flag {unknown[0]}")
            methods.append(MethodSchema(
                name=parts[1],
                body_reference=parts[2],
                # The wire form states arity, not names: a destination binds by
                # position. Placeholders keep the count honest without
                # inventing parameter names the producer never sent.
                parameters=tuple(
                    ParameterSchema(name=f"arg{i}") for i in range(count)
                ),
                is_constructor="constructor" in flags,
            ))
        else:
            raise ValueError(f"schema line {number}: unknown record {record}")
    if not saw_header:
        raise ValueError("schema line 0: empty document")
    if not identity:
        raise ValueError("schema: no class record")
    return ClassSchema(
        identity=identity, fields=tuple(fields), methods=tuple(methods),
        bases=tuple(bases), origin_language=origin,
    )
