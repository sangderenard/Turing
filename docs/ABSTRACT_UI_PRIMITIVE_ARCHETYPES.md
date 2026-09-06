# AbstractUI primitive archetypes

The first canonical constructible set is deliberately small:

```text
div       structural container
input     value/control object declaring interaction and destination
palette   layered presentation and edit-policy facet
bbox      geometry facet in an explicit coordinate space
graph     identified objects plus typed relations
```

All are backend-neutral. `div` does not mean a DOM object; it means the minimum
generic container operation which the HTML backend may realize as a `<div>`.
Likewise, `input` does not carry a callback or native input event. It declares
its value kind, interaction type, and destination identity.

Palettes carry `fg`, `bg`, named colors, margins, radii, font, decoration,
visibility, and locking. Named colors merge by role, so CSS and shader backends
can consume the same `room`, `building`, `sky`, or other material color without
copying it into backend source. Other fields are layered: unspecified values
inherit rather than being copied or guessed. The canonical default completes a
palette only when a backend or transport record needs resolved values.

`visible` is a presentation condition. `locked` is also an edit policy and is
therefore enforced by primitive construction methods; a backend must not
implement it only as disabled pointer events. A later authority system will
decide who may override or unlock it.

A bbox records geometry and coordinate-space identity. Geometry operations
refuse to compare boxes from unrelated spaces. A graph validates unique object
identities and refuses edges whose endpoints do not exist.

The existing room/building/inspector/entity/action-table demo can now be
reduced into library archetypes composed from these primitives rather than
remaining concepts embedded in its JavaScript renderer.
