"""Factory archetype, heap, defaults, reachability, and broadcast tests."""

import pytest

from src.compiler.abstract_ui_factory import (
    FactoryArchetype,
    FactoryRequest,
    factory_model,
)
from src.compiler.oop_schema import ClassSchema, FieldSchema, MethodSchema


class Particle:
    radius: float = 4.0
    color: str = "amber"

    def move(self, x: float, y: float) -> None:
        raise AssertionError("neutral broadcast must not execute host methods")

    def illuminate(self, strength: float) -> None:
        raise AssertionError("neutral broadcast must not execute host methods")


def _factory(capacity=4):
    archetype = FactoryArchetype.for_class(
        Particle,
        defaults={"color": "gold"},
        heap_capacity=capacity,
    )
    return archetype.instantiate(identity="world/factories/particles")


def test_factory_contract_takes_class_defaults_and_obtains_heap():
    factory = _factory()
    assert factory.archetype.contract.identity.endswith(".Particle")
    assert factory.archetype.defaults == (("radius", 4.0), ("color", "gold"))
    assert factory.heap.identity == "world/factories/particles/heap"
    assert factory.heap.capacity == 4
    assert [method.name for method in factory.archetype.broadcasts] == [
        "move", "illuminate",
    ]


def test_dispense_merges_request_over_defaults_without_constructing_python_object():
    factory = _factory()
    result = factory.dispense(FactoryRequest.with_(
        {"radius": 9.0}, requested_by="pointer.primary", organization="sparks",
    ))
    assert result.instance.value("radius") == 9.0
    assert result.instance.value("color") == "gold"
    assert result.instance.requested_by == "pointer.primary"
    assert result.instance.organization == "sparks"
    assert result.factory.instance(result.instance.identity) is result.instance
    assert result.edit.action == "dispense"


def test_heap_capacity_destroy_and_generation_prevent_stale_instance_aliasing():
    factory = _factory(capacity=1)
    first = factory.dispense().factory
    instance = first.instances[0]
    with pytest.raises(MemoryError, match="heap is full"):
        first.dispense()
    destroyed = first.destroy(instance.identity, actor="pointer.primary")
    assert not destroyed.factory.instances
    replacement = destroyed.factory.dispense()
    assert replacement.instance.allocation.slot == instance.allocation.slot
    assert replacement.instance.allocation.generation == instance.allocation.generation + 1
    assert replacement.instance.identity != instance.identity
    with pytest.raises(KeyError):
        replacement.factory.instance(instance.identity)


def test_factory_broadcast_sits_between_class_method_and_all_live_instances():
    factory = _factory()
    first = factory.dispense().factory
    second = first.dispense(FactoryRequest.with_({"color": "blue"})).factory
    dispatch = second.broadcast("move", 3.0, 7.0)
    assert dispatch.method == "move"
    assert [call.instance for call in dispatch.calls] == [
        instance.identity for instance in second.instances
    ]
    assert all(call.class_method == "move" for call in dispatch.calls)
    assert all(call.arguments == (3.0, 7.0) for call in dispatch.calls)

    surviving = second.destroy(second.instances[0].identity).factory
    later = surviving.broadcast("move", 1.0, 2.0)
    assert [call.instance for call in later.calls] == [surviving.instances[0].identity]


def test_request_requires_all_fields_and_rejects_unknown_overrides():
    schema = ClassSchema(
        "demo.Required",
        (FieldSchema("value", "float", slot=0),),
        (MethodSchema("update", body_reference="demo.Required.update"),),
    )
    factory = FactoryArchetype.for_class(schema).instantiate(identity="required-factory")
    with pytest.raises(ValueError, match="leaves fields unspecified"):
        factory.dispense()
    with pytest.raises(KeyError, match="unknown fields"):
        factory.dispense(FactoryRequest.with_({"missing": 1}))
    result = factory.dispense(FactoryRequest.with_({"value": 2.5}))
    assert result.instance.class_identity == "demo.Required"
    assert result.instance.value("value") == 2.5


def test_factory_transport_keeps_instances_reachable_through_factory():
    factory = _factory().dispense().factory.dispense().factory
    model = factory_model(factory)
    assert model["heap"]["allocations"] == [
        instance.allocation.address for instance in factory.instances
    ]
    assert model["reachability"] == [
        {
            "source": factory.identity,
            "target": instance.identity,
            "relationship": "dispensed-instance",
        }
        for instance in factory.instances
    ]
