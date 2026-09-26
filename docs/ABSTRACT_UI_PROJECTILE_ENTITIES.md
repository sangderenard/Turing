# AbstractUI projectile entities

The reference player owns a physics-ball gun in hotbar slot 7 and a stack of
64 balls in slot 8. Primary action on the equipped gun mints a projectile
identity beneath the world's projectile system and adds that identity to the
dedicated projectile organization.

Each ball carries:

- a stable mint-on-fire identity;
- player ownership and world-container placement edges;
- a spherical geometry recipe and semantic body identity;
- position, velocity, contact, lifetime, and status state;
- a receipt naming the compiled SymPy→SSA→WebAssembly world-physics program.

The browser emits a small latitude/longitude sphere mesh. Every active ball is
advanced by invoking the same compiled physics transition used for the player,
with its own position, velocity, radius, drag, and floor parameters. JavaScript
administers state rows and publishes poses; it does not implement a second
gravity integrator.

A fired ball is also inserted into `entity_mezzanine.entities` as a
`physics-ball-entity`, with the compiled projectile-physics controller and a
data-world pose. The div map projects that pose through the same traversal
chart as the player and renders a 0.45-scale entity marker, so the ball remains
locatable independently of the 3D camera.

Balls collide with the outer bounds and the current wall/solid collider table.
The active set is capped at 24. Lifetime or capacity expiry removes presentation
geometry and active-organization membership while retaining the projectile
record and its last world pose as spent identity history. Its smaller div-map
marker remains visible in a spent state until secondary action explicitly
clears spent history.

The present equation set performs compliant contact projection rather than
restitution, so these are launched, falling, colliding physics balls rather
than deliberately bouncy rubber balls. Restitution, friction, dense dynamic
identity specialization, sphere-sphere broad phase, and ammunition recovery
remain explicit future physics stages.
