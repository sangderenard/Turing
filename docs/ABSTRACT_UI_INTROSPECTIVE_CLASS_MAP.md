# AbstractUI introspective class map prototype

`abstract_ui_introspection.py` is the first bridge from ordinary program
structure into the existence/navigation/action vocabulary.

It does not instantiate the inspected class. It creates these `AbstractUI`
records:

```text
AbstractUIWorld
└── AbstractUIRegion             module or SSA namespace
    └── AbstractUIBuilding       class definition
        └── AbstractUIRoom       field, property, constructor, method, nested class

AbstractUITrack                  inheritance or contained-type navigation
ImpliedCode                      source/SSA meaning behind an affordance
UIIntention                     descriptive and free-form language
```

## Interpretation

- A Python module or SSA namespace is a region.
- A class is an enterable building.
- Declaration-order members are rooms on a deterministic compact grid.
- Fields become storage/observation rooms.
- Properties become reading rooms or instrument alcoves.
- Constructors become arrival or assembly rooms.
- Methods become workshops, control rooms, laboratories, or similar action
  spaces.
- Nested/composed classes and bases become other buildings connected by
  `contains-type` or `inherits` tracks.

The metaphor is fluid. A stable hash selects from caller-replaceable room and
building palettes. Changing the seed or palette changes the spatial story but
never changes canonical member identities or authored order.

## Depth

`depth_up` follows Python base classes. `depth_down` follows nested classes and
non-builtin class-valued annotations. Each selected class becomes a complete
building with its own member grid. SSA `ClassEmission` currently describes
only one class; requesting recursion is rejected until a correlated
`ClassEmissionPlan` supplies honest relationships.

## Intentions and implied code

Every room contains generated `UIIntention` phrases for identity, metaphor,
description, and affordance. Callers may add arbitrary boneyard intentions by
member name.

Every actionable/inspectable room also publishes `ImpliedCode`:

```text
Python field room   -> instance.field
Python method room  -> instance.method(parameters)
SSA field room      -> %value = load_field %receiver, slot N
SSA method room     -> %result = call @REFERENCE(%receiver, ...)
```

Python expressions are source-level executable expressions when supplied the
named receiver/arguments. SSA strings are explicitly marked non-executable:
they are readable intent receipts, while the actual SSA lowering remains
responsible for concrete value identifiers and types.

## What this establishes for AbstractUI

The prototype separates four facts which future backends must preserve:

1. **Program identity:** class/member identity and declaration order.
2. **Interpretation:** selected region/building/room metaphor.
3. **Expression:** controlled or free-form `UIIntention` descriptions.
4. **Consequence:** source/SSA operation implied by inspecting or activating a
   room.

Layout is therefore replaceable interpretation, not program identity. The
same building can be emitted as a planar game map, a prosaic HTML form, an
accessibility tree, or an introspection browser without changing the field or
method it denotes.

Reusable constructions that create new living program objects are documented
separately in
[`ABSTRACT_UI_ARCHETYPE_LIBRARY.md`](ABSTRACT_UI_ARCHETYPE_LIBRARY.md).
