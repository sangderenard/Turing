# AbstractUI entity mezzanine

Entities live directly beneath the AbstractUI system root. This mezzanine is
an identity and scheduling boundary between fundamental time/input, graphics
projection, and eventual action execution.

```text
system root
  entity mezzanine
    organization / corral
      entity
        archetype: geometry + texture + capabilities
        controller: native input, NPC dynamics, script, network, ...
        pose
        principal
    entity cycle
      control-input
      integration
      interaction
      presentation
```

An entity archetype does not decide who controls an instance, and an entity is
not required to originate from an archetype. An archetype places objects in a
universe of reusable recipes; direct objects remain valid graph citizens. The
living map now spawns one `player-being` in data-world space under the
`players` organization. Its `world-player` controller reads game controls;
mouse coordinates never become its pose. The same world pose drives both the
first-person camera and its sprite in the top-down div map.

The older `pointer-being` and follower helpers remain available for explicit
simulations and tests, but the default page no longer spawns them:

- `pointer.primary` binds to the neutral `mouse.primary` control-input ABI.
- `pointer.first` through `pointer.fourth` bind to first-, second-, third-, and
  fourth-order followers targeting the primary entity's pose.

They are members of the same `pointer-beings` organization. Their embodiment
is identical; their identities, principals, poses, colors, and controller
bindings are instance facts. Color is consequently present in each neutral
entity description rather than hidden in browser CSS.

Every pose also carries a normalized facing vector. Native actors derive it
from consecutive control positions; integrated followers derive it from their
velocity. A viewer camera can therefore follow an actor without reaching into
mouse events or inventing an extra follower whose separation happens to imply
orientation.

The general follower integrates a derivative chain. For order `n`, its highest
derivative is selected by the critically damped characteristic polynomial
`(D + omega)^n`, using binomial coefficients. This makes follower order data,
not a collection of bespoke algorithms. The original stiffness/damping form
remains accepted for existing second-order controller records.

The reference entity cycle is deterministic and pure. `inline` and `worker`
policies describe where the entire cycle may be hosted without changing its
phase order or numerical result. A graphics backend consumes immutable
presentation snapshots rather than reaching into live controller state.

Interactions are conceptual records containing actor, type, and destination.
They contain no callbacks. Later action systems can consume those records in a
separate phase without coupling entity integration to application code.

The first such consumer is the action-edge table documented in
`ABSTRACT_UI_ACTION_EDGES.md`. It shares the mezzanine but is clocked by the
system-root timer, keeping event recency outside entity integration and outside
graphics rendering.

Color selection, inventory membership, and the active tool are neutral records
defined in `abstract_ui_tools.py`. An inventory refers to entity identities and
equips at most one item with the `tool` role; it neither owns those entities nor
requires them to have been instantiated by an archetype. The initial
`color-selector` is a semantic input whose action is `set-color`.
