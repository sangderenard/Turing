# AbstractUI factory archetype

A factory is an identified intermediary between a class contract and its live
instances:

```text
class contract
  factory archetype: defaults + broadcast surface + heap requirement
    live factory
      leased heap
      dispensed instances
      broadcast dispatches
```

`FactoryArchetype.for_class` accepts a Python class or a transportable
`ClassSchema`-shaped object. Python reflection reads annotations, class-level
field defaults, and public methods without constructing the class or invoking
descriptors. Explicit factory defaults override class defaults.

Each factory invocation supplies a `FactoryRequest`. Request values override
the factory defaults for that single dispense operation. All declared fields
must be resolved and unknown fields are rejected.

The factory leases a bounded, generation-checked heap. Destroying an instance
releases its slot and increments that slot's generation, so a later instance
cannot accidentally inherit the destroyed instance's identity or address.

Every live instance remains reachable through the factory with a
`dispensed-instance` relation. Factory broadcast methods occupy a deliberate
layer between class methods and instance calls:

```text
factory.broadcast("move", x, y)
  -> call Class.move on instance 0
  -> call Class.move on instance 1
  -> call Class.move on instance N
```

The neutral layer produces identified call records; it does not execute Python
methods or hide callbacks. JavaScript, C++, Java, or SSA backends can lower the
same dispatch records according to their object and memory ABIs.
