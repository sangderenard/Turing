"""Language-neutral class plans for source emitters.

Repository SSA deliberately separates three authorities:

* ``SSAClassTable`` owns class identity, field slots, and method links;
* ``FunctionTable`` owns source-qualified callable identity and ordered
  parameter transfer/access/storage contracts;
* ``IRModule.functions`` owns executable bodies.

A JavaScript, C++, or Java printer needs the correlation, not three private
readings of those tables.  This module performs that join once and retains the
evidence behind decisions such as "formal zero is the resident receiver".
It contains no destination syntax and does not lower a body.

An optional richer :class:`ClassSchema` may supply facts absent from the SSA
table (types, defaults, bases, static/constructor flags), but only after its
SSA projection is proved to agree with the module's definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .oop_schema import ClassSchema


@dataclass(frozen=True, slots=True)
class ClassEmissionIssue:
    severity: str
    code: str
    message: str
    class_identity: str | None = None
    method_name: str | None = None

    def format(self) -> str:
        location = ""
        if self.class_identity is not None:
            location = f" {self.class_identity}"
        if self.method_name is not None:
            location += f".{self.method_name}"
        return f"{self.severity}:{self.code}{location}: {self.message}"


@dataclass(frozen=True, slots=True)
class ClassFieldEmission:
    name: str
    slot: int
    type_name: str = "unknown"
    initial: Any = None
    has_initial: bool = False
    readonly: bool = False

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "slot": self.slot,
            "type_name": self.type_name,
            "has_initial": self.has_initial,
            **({"initial": self.initial} if self.has_initial else {}),
            "readonly": self.readonly,
        }


@dataclass(frozen=True, slots=True)
class MethodParameterEmission:
    name: str
    position: int
    value_id: int | None
    type_name: str = "unknown"
    transfer: str = "unknown"
    access: str = "unknown"
    storage: str = "unknown"
    scope: str = "unknown"
    default: Any = None
    has_default: bool = False

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "position": self.position,
            "value_id": self.value_id,
            "type_name": self.type_name,
            "transfer": self.transfer,
            "access": self.access,
            "storage": self.storage,
            "scope": self.scope,
            "has_default": self.has_default,
            **({"default": self.default} if self.has_default else {}),
        }


@dataclass(frozen=True, slots=True)
class ReceiverFieldEmission:
    field_slot: int
    formal_position: int
    value_id: int
    offset: int
    dtype: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "field_slot": self.field_slot,
            "formal_position": self.formal_position,
            "value_id": self.value_id,
            "offset": self.offset,
            "dtype": self.dtype,
        }


@dataclass(frozen=True, slots=True)
class ClassMethodEmission:
    name: str
    function_reference: int
    function_name: str | None
    qualified_name: str | None
    kind: str
    is_static: bool
    receiver_position: int | None
    receiver_fields: tuple[ReceiverFieldEmission, ...]
    receiver_evidence: str
    parameters: tuple[MethodParameterEmission, ...]
    body_available: bool

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "function_reference": self.function_reference,
            "function_name": self.function_name,
            "qualified_name": self.qualified_name,
            "kind": self.kind,
            "is_static": self.is_static,
            "receiver_position": self.receiver_position,
            "receiver_fields": [
                item.to_mapping() for item in self.receiver_fields
            ],
            "receiver_evidence": self.receiver_evidence,
            "parameters": [item.to_mapping() for item in self.parameters],
            "body_available": self.body_available,
        }


@dataclass(frozen=True, slots=True)
class ClassEmission:
    identity: str
    fields: tuple[ClassFieldEmission, ...]
    methods: tuple[ClassMethodEmission, ...]
    bases: tuple[str, ...] = ()
    origin_language: str = "unstated"

    def to_mapping(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "fields": [item.to_mapping() for item in self.fields],
            "methods": [item.to_mapping() for item in self.methods],
            "bases": list(self.bases),
            "origin_language": self.origin_language,
        }


@dataclass(frozen=True, slots=True)
class ClassEmissionPlan:
    classes: tuple[ClassEmission, ...]
    issues: tuple[ClassEmissionIssue, ...] = ()

    @property
    def complete(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": "turing.class-emission-plan.v1",
            "classes": [item.to_mapping() for item in self.classes],
            "issues": [
                {
                    "severity": issue.severity,
                    "code": issue.code,
                    "message": issue.message,
                    "class_identity": issue.class_identity,
                    "method_name": issue.method_name,
                }
                for issue in self.issues
            ],
        }


def _schemas_by_identity(
    schemas: Iterable[ClassSchema] | Mapping[str, ClassSchema] | None,
) -> dict[str, ClassSchema]:
    if schemas is None:
        return {}
    values = schemas.values() if isinstance(schemas, Mapping) else schemas
    result: dict[str, ClassSchema] = {}
    for schema in values:
        identity = str(schema.identity)
        if identity in result:
            raise ValueError(f"duplicate ClassSchema identity {identity!r}")
        result[identity] = schema
    return result


def _entry_by_reference(function_table: Any) -> dict[int, Any]:
    return {
        int(entry.reference.address): entry
        for entry in (function_table or ())
    }


def plan_class_emission(
    module: Any,
    *,
    schemas: Iterable[ClassSchema] | Mapping[str, ClassSchema] | None = None,
) -> ClassEmissionPlan:
    """Join a module's class, function, and optional rich-schema facts."""

    supplied = _schemas_by_identity(schemas)
    definitions = tuple(
        getattr(getattr(module, "class_table", None), "classes", ())
    )
    functions = dict(getattr(module, "functions", {}) or {})
    entries = _entry_by_reference(getattr(module, "function_table", None))
    issues: list[ClassEmissionIssue] = []
    planned: list[ClassEmission] = []

    defined_identities = {str(item.identity) for item in definitions}
    for extra in sorted(set(supplied) - defined_identities):
        issues.append(ClassEmissionIssue(
            "error", "schema-without-ssa-class",
            "a rich schema cannot create a class absent from SSA",
            extra,
        ))

    for definition in definitions:
        identity = str(definition.identity)
        schema = supplied.get(identity)
        if schema is not None:
            disagreements = schema.ssa_projection_agrees(definition)
            issues.extend(
                ClassEmissionIssue(
                    "error", "schema-ssa-disagreement", message, identity,
                )
                for message in disagreements
            )
        schema_fields = {
            field.name: field for field in (() if schema is None else schema.fields)
        }
        fields = tuple(
            ClassFieldEmission(
                name=str(field.name),
                slot=int(field.slot),
                type_name=str(getattr(schema_fields.get(field.name), "type_name", "unknown")),
                initial=getattr(schema_fields.get(field.name), "initial", None),
                has_initial=(
                    schema_fields.get(field.name) is not None
                    and schema_fields[field.name].initial is not None
                ),
                readonly=bool(getattr(schema_fields.get(field.name), "readonly", False)),
            )
            # SSAClassDefinition retains declaration order. Slots are the
            # physical ABI coordinate, not a request to reorder the public
            # class surface. C++, Java, and JavaScript all conventionally
            # preserve authored member order while laying fields out by their
            # separately recorded offsets/slots.
            for field in definition.fields
        )
        names = tuple(field.name for field in fields)
        slots = tuple(field.slot for field in fields)
        if len(names) != len(set(names)):
            issues.append(ClassEmissionIssue(
                "error", "duplicate-field-name", "field names are not unique", identity,
            ))
        if len(slots) != len(set(slots)):
            issues.append(ClassEmissionIssue(
                "error", "duplicate-field-slot", "field slots are not unique", identity,
            ))

        schema_methods = {
            method.name: method for method in (() if schema is None else schema.methods)
        }
        methods: list[ClassMethodEmission] = []
        for method in definition.methods:
            method_name = str(method.name)
            rich = schema_methods.get(method_name)
            body = (
                None if method.function_name is None
                else functions.get(str(method.function_name))
            )
            entry = entries.get(int(method.function_reference))
            contracts = tuple(getattr(entry, "parameter_contracts", ()) or ())
            formals = tuple(() if body is None else body.args)
            if entry is None:
                issues.append(ClassEmissionIssue(
                    "warning", "missing-function-table-entry",
                    f"reference {int(method.function_reference)} has no FunctionTable entry",
                    identity, method_name,
                ))
            if body is None:
                issues.append(ClassEmissionIssue(
                    "error", "missing-method-body",
                    f"SSA function {method.function_name!r} was not supplied",
                    identity, method_name,
                ))
            is_static = bool(getattr(rich, "is_static", False))
            receiver_position: int | None = None
            receiver_fields: tuple[ReceiverFieldEmission, ...] = ()
            receiver_evidence = "static-schema" if is_static else "unknown"
            receiver_contract_positions = [
                index
                for index, contract in enumerate(contracts)
                if str(contract.transfer.value) == "alias"
                and str(contract.storage.value) == "record"
                and str(contract.scope.value) in {"caller", "retained"}
            ]
            positions_by_id = {
                int(formal.id): position
                for position, formal in enumerate(formals)
            }
            if not is_static and body is not None:
                receiver_fields = tuple(
                    ReceiverFieldEmission(
                        field_slot=int(slot),
                        formal_position=positions_by_id[int(value_id)],
                        value_id=int(value_id),
                        offset=int(offset),
                        dtype=str(dtype),
                    )
                    for slot, value_id, offset, dtype in body.metadata.get(
                        "receiver_field_locations", ()
                    )
                    if int(value_id) in positions_by_id
                )
                direct_candidates = [
                    position
                    for position, formal in enumerate(formals)
                    if (formal.accounting or {}).get(
                        "linked_method_receiver_storage"
                    ) is not None
                ]
                if receiver_fields:
                    receiver_evidence = "ssa-receiver-field-locations"
                elif len(direct_candidates) == 1:
                    receiver_position = direct_candidates[0]
                    receiver_evidence = "linked-method-receiver-storage"
                elif contracts and len(receiver_contract_positions) != 1:
                    issues.append(ClassEmissionIssue(
                        "error", "ambiguous-logical-receiver-contract",
                        "record-alias logical receiver contracts are "
                        f"{receiver_contract_positions!r}",
                        identity, method_name,
                    ))
                elif formals:
                    # Minimal, manually-authored SSA has neither the function
                    # contract nor physical field-layout metadata. Preserve
                    # its established receiver-first convention as an
                    # explicit fallback receipt.
                    receiver_position = 0
                    receiver_evidence = "minimal-ssa-receiver-first-convention"
                    issues.append(ClassEmissionIssue(
                        "warning", "receiver-contract-fallback",
                        "physical receiver metadata is absent; using the "
                        "minimal SSA receiver-first convention",
                        identity, method_name,
                    ))
            if is_static:
                receiver_contract_positions = []

            physical_receiver_positions = {
                field.formal_position for field in receiver_fields
            }
            if receiver_position is not None:
                physical_receiver_positions.add(receiver_position)
            public_positions = [
                position for position in range(len(formals))
                if position not in physical_receiver_positions
            ]
            public_contracts = [
                contract
                for position, contract in enumerate(contracts)
                if position not in receiver_contract_positions
            ]
            if contracts and len(public_contracts) != len(public_positions):
                issues.append(ClassEmissionIssue(
                    "error", "parameter-contract-arity",
                    f"{len(public_contracts)} logical public parameter "
                    f"contracts do not match {len(public_positions)} physical "
                    "non-receiver formals",
                    identity, method_name,
                ))

            # Physical SSA order may differ from authored parameter order.
            # Use the function's explicit name->value ledger first, then pair
            # the remaining contracts and positions deterministically.
            named_ids = {
                str(name): int(value_id)
                for name, value_id in (
                    (() if body is None else body.metadata.get(
                        "parameter_names", ()
                    ))
                )
            }
            contract_by_position: dict[int, Any] = {}
            unused_contracts = list(public_contracts)
            for contract in tuple(unused_contracts):
                value_id = named_ids.get(str(contract.name))
                position = (
                    None if value_id is None else positions_by_id.get(value_id)
                )
                if position in public_positions:
                    contract_by_position[int(position)] = contract
                    unused_contracts.remove(contract)
            for position, contract in zip(
                (
                    position for position in public_positions
                    if position not in contract_by_position
                ),
                unused_contracts,
            ):
                contract_by_position[position] = contract

            # Public signature order comes from FunctionTable contracts (the
            # authored parameter sequence). Each item still carries its
            # possibly different physical SSA position for the call adapter.
            # Any contractless physical formals follow in their SSA order and
            # remain explicitly named fallbacks.
            ordered_positions = []
            for contract in public_contracts:
                position = next((
                    candidate
                    for candidate, bound in contract_by_position.items()
                    if bound is contract
                ), None)
                if position is not None:
                    ordered_positions.append(position)
            ordered_positions.extend(
                position for position in public_positions
                if position not in ordered_positions
            )

            rich_parameters = tuple(() if rich is None else rich.parameters)
            parameters: list[MethodParameterEmission] = []
            for public_position, position in enumerate(ordered_positions):
                formal = formals[position]
                contract = contract_by_position.get(position)
                rich_parameter = (
                    rich_parameters[public_position]
                    if public_position < len(rich_parameters)
                    else None
                )
                name = (
                    str(contract.name)
                    if contract is not None
                    else str(rich_parameter.name)
                    if rich_parameter is not None
                    else f"arg{position}"
                )
                parameters.append(MethodParameterEmission(
                    name=name,
                    position=position,
                    value_id=int(formal.id),
                    type_name=str(getattr(rich_parameter, "type_name", "unknown")),
                    transfer=("unknown" if contract is None else contract.transfer.value),
                    access=("unknown" if contract is None else contract.access.value),
                    storage=("unknown" if contract is None else contract.storage.value),
                    scope=("unknown" if contract is None else contract.scope.value),
                    default=getattr(rich_parameter, "default", None),
                    has_default=bool(getattr(rich_parameter, "has_default", False)),
                ))

            kind = (
                "allocator" if method_name == "__new__"
                else "initializer"
                if method_name in {"__init__", "constructor"}
                or bool(getattr(rich, "is_constructor", False))
                else "method"
            )
            methods.append(ClassMethodEmission(
                name=method_name,
                function_reference=int(method.function_reference),
                function_name=(
                    None if method.function_name is None else str(method.function_name)
                ),
                qualified_name=(
                    None if entry is None else str(entry.qualified_name)
                ),
                kind=kind,
                is_static=is_static,
                receiver_position=receiver_position,
                receiver_fields=receiver_fields,
                receiver_evidence=receiver_evidence,
                parameters=tuple(parameters),
                body_available=body is not None,
            ))

        planned.append(ClassEmission(
            identity=identity,
            fields=fields,
            methods=tuple(methods),
            bases=tuple(() if schema is None else map(str, schema.bases)),
            origin_language=(
                "unstated" if schema is None else str(schema.origin_language)
            ),
        ))

    return ClassEmissionPlan(tuple(planned), tuple(issues))


__all__ = [
    "ClassEmission",
    "ClassEmissionIssue",
    "ClassEmissionPlan",
    "ClassFieldEmission",
    "ClassMethodEmission",
    "MethodParameterEmission",
    "ReceiverFieldEmission",
    "plan_class_emission",
]
