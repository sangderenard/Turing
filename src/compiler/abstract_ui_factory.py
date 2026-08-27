"""Factory archetypes between class contracts and allocated instances.

A factory takes a Python or ClassSchema-shaped class, obtains a bounded heap,
holds construction defaults, dispenses request-specific instances, destroys
them safely, and keeps every live instance reachable for broadcast dispatch.
No host-language constructor or method is executed by this neutral layer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import inspect
from typing import Any, Mapping, Sequence


ABSTRACT_UI_FACTORY_VERSION = "abstract-ui-factory-v0"
_MISSING = object()


def _freeze(value: Mapping[str, Any] | None) -> tuple[tuple[str, Any], ...]:
    return tuple((str(name), item) for name, item in (value or {}).items())


def _type_name(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return getattr(value, "__qualname__", None) or str(value)


@dataclass(frozen=True, slots=True)
class FactoryField:
    name: str
    type_name: str | None = None
    has_default: bool = False
    default: Any = None


@dataclass(frozen=True, slots=True)
class FactoryClassContract:
    identity: str
    name: str
    source_kind: str
    fields: tuple[FactoryField, ...]
    methods: tuple[str, ...]


def _python_contract(subject: type) -> FactoryClassContract:
    namespace = vars(subject)
    annotations = namespace.get("__annotations__", {}) or {}
    fields = []
    for name, annotation in annotations.items():
        default = namespace.get(name, _MISSING)
        fields.append(FactoryField(
            str(name), _type_name(annotation), default is not _MISSING,
            None if default is _MISSING else default,
        ))
    methods = tuple(
        name for name, value in namespace.items()
        if not name.startswith("_") and (
            inspect.isfunction(value)
            or isinstance(value, (staticmethod, classmethod))
        )
    )
    return FactoryClassContract(
        f"python:{subject.__module__}.{subject.__qualname__}",
        subject.__name__,
        "python",
        tuple(fields),
        methods,
    )


def _schema_contract(subject: Any) -> FactoryClassContract:
    identity = str(subject.identity)
    return FactoryClassContract(
        identity,
        identity.rsplit(".", 1)[-1],
        str(getattr(subject, "origin_language", "schema")),
        tuple(
            FactoryField(str(field.name), _type_name(getattr(field, "type_name", None)))
            for field in subject.fields
        ),
        tuple(str(method.name) for method in subject.methods),
    )


def class_contract(subject: type | Any) -> FactoryClassContract:
    if isinstance(subject, type):
        return _python_contract(subject)
    if (
        isinstance(getattr(subject, "identity", None), str)
        and isinstance(getattr(subject, "fields", None), tuple)
        and isinstance(getattr(subject, "methods", None), tuple)
    ):
        return _schema_contract(subject)
    raise TypeError("factory subject must be a Python class or ClassSchema-shaped object")


@dataclass(frozen=True, slots=True)
class FactoryBroadcastMethod:
    name: str
    class_method: str


@dataclass(frozen=True, slots=True)
class FactoryArchetype:
    """Reusable class/default/broadcast recipe from which factories arise."""

    name: str
    contract: FactoryClassContract
    defaults: tuple[tuple[str, Any], ...]
    broadcasts: tuple[FactoryBroadcastMethod, ...]
    heap_capacity: int = 256

    @staticmethod
    def for_class(
        subject: type | Any,
        *,
        name: str | None = None,
        defaults: Mapping[str, Any] | None = None,
        broadcast_methods: Sequence[str] | None = None,
        heap_capacity: int = 256,
    ) -> "FactoryArchetype":
        if heap_capacity <= 0:
            raise ValueError("factory heap capacity must be positive")
        contract = class_contract(subject)
        field_names = {field.name for field in contract.fields}
        merged_defaults = {
            field.name: field.default for field in contract.fields if field.has_default
        }
        merged_defaults.update(defaults or {})
        unknown_defaults = set(merged_defaults).difference(field_names)
        if unknown_defaults:
            raise KeyError(f"factory defaults name unknown fields: {sorted(unknown_defaults)!r}")
        selected = tuple(contract.methods if broadcast_methods is None else broadcast_methods)
        unknown_methods = set(selected).difference(contract.methods)
        if unknown_methods:
            raise KeyError(f"factory broadcasts name unknown methods: {sorted(unknown_methods)!r}")
        return FactoryArchetype(
            str(name or f"{contract.name}-factory"),
            contract,
            tuple(
                (field.name, merged_defaults[field.name])
                for field in contract.fields if field.name in merged_defaults
            ),
            tuple(FactoryBroadcastMethod(method, method) for method in selected),
            int(heap_capacity),
        )

    def instantiate(
        self,
        *,
        identity: str,
        heap_identity: str | None = None,
    ) -> "AbstractUIFactory":
        heap = FactoryHeap.empty(
            heap_identity or f"{identity}/heap", self.heap_capacity,
        )
        return AbstractUIFactory(str(identity), self, heap)


@dataclass(frozen=True, slots=True)
class HeapAllocation:
    heap: str
    slot: int
    generation: int
    owner: str

    @property
    def address(self) -> str:
        return f"{self.heap}/slots/{self.slot}:{self.generation}"


@dataclass(frozen=True, slots=True)
class FactoryHeap:
    identity: str
    capacity: int
    generations: tuple[int, ...]
    allocations: tuple[HeapAllocation, ...] = ()

    @staticmethod
    def empty(identity: str, capacity: int) -> "FactoryHeap":
        if capacity <= 0:
            raise ValueError("heap capacity must be positive")
        return FactoryHeap(str(identity), int(capacity), (0,) * int(capacity))

    def allocate(self, owner: str) -> tuple["FactoryHeap", HeapAllocation]:
        occupied = {allocation.slot for allocation in self.allocations}
        slot = next((slot for slot in range(self.capacity) if slot not in occupied), None)
        if slot is None:
            raise MemoryError(f"factory heap is full: {self.identity}")
        allocation = HeapAllocation(
            self.identity, slot, self.generations[slot], str(owner),
        )
        return replace(self, allocations=(*self.allocations, allocation)), allocation

    def release(self, allocation: HeapAllocation) -> "FactoryHeap":
        if allocation.heap != self.identity:
            raise ValueError("allocation belongs to a different heap")
        if allocation not in self.allocations:
            raise ValueError(f"allocation is stale or already released: {allocation.address}")
        generations = list(self.generations)
        generations[allocation.slot] += 1
        return replace(
            self,
            generations=tuple(generations),
            allocations=tuple(item for item in self.allocations if item != allocation),
        )


@dataclass(frozen=True, slots=True)
class FactoryRequest:
    overrides: tuple[tuple[str, Any], ...] = ()
    requested_by: str = "system-root"
    organization: str | None = None

    @staticmethod
    def with_(
        overrides: Mapping[str, Any] | None = None,
        *,
        requested_by: str = "system-root",
        organization: str | None = None,
    ) -> "FactoryRequest":
        return FactoryRequest(_freeze(overrides), str(requested_by), organization)


@dataclass(frozen=True, slots=True)
class FactoryInstance:
    identity: str
    class_identity: str
    allocation: HeapAllocation
    values: tuple[tuple[str, Any], ...]
    created_revision: int
    requested_by: str
    organization: str | None = None

    def value(self, name: str) -> Any:
        values = dict(self.values)
        if name not in values:
            raise KeyError(name)
        return values[name]


@dataclass(frozen=True, slots=True)
class FactoryEdit:
    action: str
    factory: str
    instance: str
    allocation: str
    before_revision: int
    after_revision: int
    actor: str


@dataclass(frozen=True, slots=True)
class FactoryMutation:
    factory: "AbstractUIFactory"
    instance: FactoryInstance
    edit: FactoryEdit


@dataclass(frozen=True, slots=True)
class FactoryBroadcastCall:
    identity: str
    factory: str
    instance: str
    class_method: str
    arguments: tuple[Any, ...]
    keywords: tuple[tuple[str, Any], ...]


@dataclass(frozen=True, slots=True)
class FactoryBroadcastDispatch:
    identity: str
    factory: str
    method: str
    calls: tuple[FactoryBroadcastCall, ...]
    revision: int


@dataclass(frozen=True, slots=True)
class AbstractUIFactory:
    """One live factory, its heap lease, and all reachable instances."""

    identity: str
    archetype: FactoryArchetype
    heap: FactoryHeap
    instances: tuple[FactoryInstance, ...] = ()
    revision: int = 0

    def instance(self, identity: str) -> FactoryInstance:
        for instance in self.instances:
            if instance.identity == identity:
                return instance
        raise KeyError(identity)

    def dispense(
        self,
        request: FactoryRequest | None = None,
    ) -> FactoryMutation:
        active_request = request or FactoryRequest.with_()
        values = dict(self.archetype.defaults)
        overrides = dict(active_request.overrides)
        field_order = tuple(field.name for field in self.archetype.contract.fields)
        unknown = set(overrides).difference(field_order)
        if unknown:
            raise KeyError(f"factory request names unknown fields: {sorted(unknown)!r}")
        values.update(overrides)
        missing = tuple(name for name in field_order if name not in values)
        if missing:
            raise ValueError(f"factory request leaves fields unspecified: {missing!r}")
        heap, allocation = self.heap.allocate(self.identity)
        after = self.revision + 1
        instance_identity = f"{self.identity}/instances/{allocation.slot}:{allocation.generation}"
        instance = FactoryInstance(
            instance_identity,
            self.archetype.contract.identity,
            allocation,
            tuple((name, values[name]) for name in field_order),
            after,
            active_request.requested_by,
            active_request.organization,
        )
        factory = replace(
            self, heap=heap, instances=(*self.instances, instance), revision=after,
        )
        edit = FactoryEdit(
            "dispense", self.identity, instance.identity, allocation.address,
            self.revision, after, active_request.requested_by,
        )
        return FactoryMutation(factory, instance, edit)

    def destroy(
        self,
        instance_identity: str,
        *,
        actor: str = "system-root",
    ) -> FactoryMutation:
        instance = self.instance(instance_identity)
        heap = self.heap.release(instance.allocation)
        after = self.revision + 1
        factory = replace(
            self,
            heap=heap,
            instances=tuple(item for item in self.instances if item != instance),
            revision=after,
        )
        edit = FactoryEdit(
            "destroy", self.identity, instance.identity, instance.allocation.address,
            self.revision, after, str(actor),
        )
        return FactoryMutation(factory, instance, edit)

    def broadcast(
        self,
        method: str,
        *arguments: Any,
        **keywords: Any,
    ) -> FactoryBroadcastDispatch:
        definitions = {item.name: item for item in self.archetype.broadcasts}
        if method not in definitions:
            raise KeyError(f"factory has no broadcast method {method!r}")
        definition = definitions[method]
        calls = tuple(
            FactoryBroadcastCall(
                f"{self.identity}/broadcasts/{method}/{self.revision}/{index}",
                self.identity,
                instance.identity,
                definition.class_method,
                tuple(arguments),
                _freeze(keywords),
            )
            for index, instance in enumerate(self.instances)
        )
        return FactoryBroadcastDispatch(
            f"{self.identity}/broadcasts/{method}/{self.revision}",
            self.identity,
            method,
            calls,
            self.revision,
        )


def factory_model(factory: AbstractUIFactory) -> dict[str, Any]:
    """Transport form for namespace, heap, reachability, and broadcasts."""

    return {
        "schema": ABSTRACT_UI_FACTORY_VERSION,
        "identity": factory.identity,
        "revision": factory.revision,
        "class": {
            "identity": factory.archetype.contract.identity,
            "name": factory.archetype.contract.name,
            "source_kind": factory.archetype.contract.source_kind,
            "fields": [field.name for field in factory.archetype.contract.fields],
            "methods": list(factory.archetype.contract.methods),
        },
        "defaults": dict(factory.archetype.defaults),
        "broadcasts": [item.name for item in factory.archetype.broadcasts],
        "heap": {
            "identity": factory.heap.identity,
            "capacity": factory.heap.capacity,
            "allocations": [allocation.address for allocation in factory.heap.allocations],
        },
        "instances": [
            {
                "identity": instance.identity,
                "class_identity": instance.class_identity,
                "allocation": instance.allocation.address,
                "values": dict(instance.values),
                "requested_by": instance.requested_by,
                "organization": instance.organization,
            }
            for instance in factory.instances
        ],
        "reachability": [
            {
                "source": factory.identity,
                "target": instance.identity,
                "relationship": "dispensed-instance",
            }
            for instance in factory.instances
        ],
    }


__all__ = [
    "ABSTRACT_UI_FACTORY_VERSION",
    "AbstractUIFactory",
    "FactoryArchetype",
    "FactoryBroadcastCall",
    "FactoryBroadcastDispatch",
    "FactoryBroadcastMethod",
    "FactoryClassContract",
    "FactoryEdit",
    "FactoryField",
    "FactoryHeap",
    "FactoryInstance",
    "FactoryMutation",
    "FactoryRequest",
    "HeapAllocation",
    "class_contract",
    "factory_model",
]
