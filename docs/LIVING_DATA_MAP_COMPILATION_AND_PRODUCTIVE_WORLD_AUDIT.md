# Living Data Map: compilation boundary and productive-world audit

Date: 2026-08-25

Scope: the `AbstractUI` → introspective world → Living Data Map → static HTML
path, with particular attention to (1) what is genuinely compiled versus
model-driven or bespoke and (2) what the world needs before archetypes become
productive actors and objects rather than decorative or narrowly interactive
representations.

This is a diagnostic report. It does not propose that all browser code should
be compiled. Browser effects need adapters. The important question is whether
an adapter realizes a declared portable contract or secretly owns the world
semantics.

## Executive verdict

The page is neither a handcrafted demo with decorative metadata nor a compiled
application in the strong whole-program sense. It is a hybrid with three real
strengths:

1. A substantial, typed Python model is synthesized from program introspection
   and serialized as a neutral identity graph.
2. Several bounded numerical regions are genuinely compiled to WebAssembly,
   and the Pluck fragment shader is translated through project IR before the
   browser compiles it for WebGL.
3. Timing, entity, tool, placement, world-object, and action concepts already
   have neutral records that can become compiler inputs.

Its limiting fact is equally clear: the browser behavior is a 6,156-line,
232-function handwritten JavaScript backend template. It does not merely bind
DOM/WebGL APIs. It currently owns navigation orchestration, collision policy,
tool dispatch, placement state machines, projectile lifecycle, portals,
inventory mutation, audio scheduling, persistence, and much of entity
execution. Those are portable world semantics and therefore exceed a healthy
host-adapter boundary.

The fastest path toward a working world is not to compile more arithmetic.
It is to define a small behavior/work IR and a semantic pose/attachment ABI,
then make the browser host execute or present those contracts generically.
The existing `ArchetypeLibrary`, `AbstractUIFactory`, entity mezzanine, action
edges, and state-loop planner provide most of the nouns, but they are not yet
wired into the Living Data Map runtime.

## Terms used by this audit

The report distinguishes five levels that are easy to conflate:

| Level | Meaning |
|---|---|
| Compiled kernel | Source or symbolic equations lower through an IR into an executable artifact with a published ABI. |
| Compiled/deployed control | A declared state loop or behavior graph determines ownership, timing, communication, and execution placement. |
| Model-generated | Python constructs typed neutral data that a backend consumes. This is valuable compilation work, but it is not executable behavior by itself. |
| Model-driven adapter | Handwritten host code interprets a portable contract and limits itself to host mechanics or presentation. |
| Bespoke semantic runtime | Handwritten host code decides game/world meaning through local state, name checks, operation switches, and direct mutations. |

“Generated JavaScript” is therefore not automatically compiled JavaScript. A
large fixed template emitted into every page remains bespoke even though a
Python function placed it there.

## Audit evidence and scale

For the current `AbstractUI` projection:

- `abstract_ui_div_map.py` is 7,191 lines / about 360 KB.
- The embedded JavaScript template is 6,156 lines / about 308 KB and declares
  232 top-level functions.
- The CSS template is 247 lines / about 22 KB.
- The compact JSON model is about 802 KB, dominated by embedded content and
  compiled modules.
- The resulting single-file document is about 1.14 MB.
- The model publishes 22 named schema-bearing subsystems, including entities,
  navigation, filesystem, geometry, viewer, placement, projectiles, inventory,
  control focus, persistence, audio, physics, loop deployment, scene mesh,
  world objects, and actions.
- Five content-addressed Wasm plugins total roughly 25 KB: parametric mesh
  vertex construction (240 B), world transform (268 B), software projection
  (491 B), FFT (22,062 B), and symbolic physics (2,153 B). Navigation adds a
  separate 5,485 B Zig/C-compiled Wasm kernel.

The size split is not itself a defect. It shows that the semantic model is
substantial and the numerical kernels are compact, while orchestration is the
largest manually authored executable surface.

## The actual build pipeline

```text
Python or ClassSchema-shaped subject
  │
  ├─ build_introspective_world
  │    program identities → regions/buildings/rooms/tracks/code receipts
  │
  ├─ world_div_map_model
  │    adds viewer, geometry, entities, tools, inventory, placement,
  │    projectiles, persistence, audio, physics, world registry, loops
  │
  ├─ bounded compiler paths
  │    Python numeric source → ProcessGraph/FusedProgram → Wasm
  │    SymPy equations → SSA → Wasm
  │    bounded FFT model → FusedProgram → Wasm
  │    C A* source → Zig/Clang → Wasm
  │    Pluck GLSL → project IR → WebGL GLSL
  │
  ├─ project_world_to_div_map
  │    four identified packets: HTML, CSS, JSON model, JavaScript behavior
  │
  └─ AbstractUI.document
       assembles one static browser document
```

The first introspection stage is genuinely subject-derived. The second stage
also injects a fixed game/demo environment into every projected world: the
player body, ball gun, projectile system, music room, celestial sky, placement
recipes, persistence rules, and physics runtime are backend-builder choices,
not semantics discovered in the projected subject.

## Audit 1: compiled and model-derived surfaces

### Program/world identity graph — strong model generation

`project_class_to_div_map()` calls `build_introspective_world()` and preserves
program identity, containment, source kind, member order, implied code, and
tracks. `world_div_map_model()` then serializes this graph without asking the
browser to rediscover class structure. DOM nodes, geometry spans, inspector
destinations, filesystem artifacts, and source closures share identities.

This is the page's strongest compilation property: the visual world is a
projection of a neutral program/world model rather than a separately authored
scene graph.

Limits:

- The model is assembled into plain nested dictionaries at the last mile.
- Several runtime additions mutate those dictionaries directly in JavaScript.
- There is no general patch/edit reducer shared by Python and the browser.
- The introspected subject does not declare most of the demo's game systems;
  `world_div_map_model()` installs them imperatively.

### World and mesh identity — strong contract, partial compiled realization

The world registry separates `WorldObject` authority from mesh packets and
publishes object/semantic-part span tables. Dense runtime IDs specialize the
authored string identities without replacing them. This is exactly the right
shape for tools, collision, rendering, and work actions to address the same
object.

Parametric box vertex arithmetic is compiled from Python numeric source through
the common pipeline to Wasm. World yaw transform and software projection are
also compiled bounded kernels.

Limits:

- Topology composition, openings, sphere construction, portal splats/tubes,
  normals, collider extraction, mesh batching, and revision propagation remain
  handwritten JavaScript.
- The Wasm kernels are helpers. They do not own a mesh-build transaction or
  prove that every world recipe has a backend realization.
- Geometry capabilities are strings; there is no typed operation ABI saying
  what inputs, effects, or failure modes `publish-mesh`, `collide-static`, or
  `receive-materials` entail.

### Symbolic physics — genuine kernel compilation, bespoke world solver

Gravity, drag, unilateral obstacle projection, bounds, and portal coordinate
terms originate as SymPy equations and lower to SSA/Wasm with a published
parameter ABI. The fixed-step worker invokes this Wasm for bodies and owns the
authoritative body state between snapshots.

The deployment model contributes real value: one-writer validation, host
placement, a capacity-one latest-snapshot channel, preallocated transferable
buffers, stable slots/generations, and three engine gears.

Limits:

- The worker source is a physics-specific JavaScript template, not emitted from
  the body of a general annotated state loop.
- Broad phase, collider scans, top support, ball/player and ball/ball response,
  bounce sound, sleep/wake policy, attractor fields, absorption, and lifecycle
  transitions are handwritten outside the compiled equation set.
- The provenance record calls the whole physics loop “compiler-emitted,” but
  only the scalar numerical interior and deployment envelope deserve that
  label. The contact/solver semantics in the envelope remain bespoke.

### Navigation — compiled search kernel, bespoke planning stack

The A* search itself is real C compiled by Zig to freestanding Wasm. Its ABI,
grid limits, heuristic, connectivity, and ownership are published. Per-entity
kernel assignment is data.

The browser still owns rasterization, coordinate charts, endpoint heuristics,
line-of-sight simplification, Catmull–Rom validation, quaternion orientation,
waypoint queues, presence pauses, route overlays, and worker protocol. These
are more than graphics glue; they define navigation semantics.

### Audio/FFT — executable compiled result with an important provenance caveat

The page's FFT executes as Wasm and its input/output ABI is explicit. The
bounded C source is parsed and structurally validated, and the generated
FusedProgram implements the same radix-2 butterfly.

However, the parsed C AST is not itself lowered into that FusedProgram. Python
code independently constructs the specialized 64-point graph. This is a
verified algorithm import/specialization, not yet a general C-source→IR
compiler path. The model should say that as plainly as this audit does.

AudioContext setup, file selection, playback synchronization, impact voices,
FFT window scheduling, spectrum-to-color mapping, and light response are
handwritten browser behavior.

### Shader path — mixed

The Pluck fragment shader is read from the sibling `spectral-analyzer` checkout
and translated through project GLSL IR into WebGL GLSL. That is a meaningful
compiler product, and its binding manifest is retained.

The first-person vertex adapter, default living-map shaders, sky shader,
crosshair shader, shadow pass shaders, framebuffer management, light grouping,
palette-to-material matching, and render passes are handwritten. The browser
still performs the final WebGL driver compilation, as expected.

Build reproducibility is currently environmental: if the sibling shader or
material-catalog bridge is absent, the Pluck path silently disappears. A
product build needs a declared input manifest and an explicit refusal or
fallback policy rather than path-sensitive feature discovery.

### HTML/CSS/DOM — model-informed, mostly bespoke adapter

The base `AbstractUI` packet model is sound: structure, presentation, model,
and behavior are separately identified and declare dependencies. Palette
values become CSS variables and semantic identities drive DOM selection.

The actual shell markup, 247-line CSS body, every DOM constructor, inspector,
hotbar, telemetry panel, mobile controls, source district, and event wiring are
fixed templates. They are generated artifacts but not derived from a general
HTML process graph or component/layout IR.

This is acceptable for a backend prototype. It becomes a product risk when
portable semantics (inventory mutation, behavior dispatch, work progress) are
implemented inside the same template and cannot be tested independently of
the browser projection.

### Tools, placement, portals, projectiles, and persistence — neutral nouns,
bespoke verbs

Tools declare hooks and modes; placement declares custody, gimbal, snapping,
subtractive openings, and a probabilistic portal graph; projectiles declare an
archetype and event transitions. Those are useful neutral records.

Execution is still dispatched through name and operation comparisons such as
`tool.name === "Physics-ball gun"` and the `routeActiveToolHook()` `if` chain.
Placement, portal construction/traversal, projectile creation, inventory
decrement, attraction, pickup, and world mutation are direct JavaScript
procedures.

Persistence is likewise bespoke and incomplete. It serializes dirty geometry,
portal splats/graph settings, recipe stock, tool modes, and physics parameters
to cookie/local storage. It does not persist the complete living document,
dynamic entities, work state, action history, factory heaps, behavior progress,
or authoritative inventory/world custody.

### Actions and entities — good contracts, execution not connected

The Python entity mezzanine cleanly separates archetype, controller, pose,
organization, interaction, and presentation. Its reference cycle has fixed
control/integration/interaction/presentation phases and can be placed inline or
in a worker without changing semantics.

But the Living Data Map does not execute that reference cycle as a compiled or
generic behavior engine. Its JavaScript `runEntityCycle()` is a separate,
page-specific orchestration loop.

Action edges currently provide registration, issue counts, and recency. They
are an observability table, not an action reducer. An `EntityInteraction`
remains a conceptual record; nothing resolves a capability signature,
validates preconditions, applies typed effects, or records completion/failure.

## Audit 1: findings ordered by architectural risk

### P0 — the provenance manifest materially understates bespoke authority

`remaining_bespoke_surface` lists only DOM construction, WebGL presentation,
browser input, and inspector layout. It omits navigation planning, audio host,
world collision/contact policy, tool dispatcher, placement editor, portal
runtime, projectile lifecycle, persistence, entity orchestration, and inventory
mutation.

This makes the page look more compiled than it is and prevents honest progress
measurement. Provenance should be attached per subsystem and, ideally, per
operation with one of the five levels defined above.

### P0 — portable behavior semantics live in a backend template

The central architectural debt is not JavaScript itself. It is that browser
functions own portable state transitions. A textual, native, networked, or
headless backend could consume the nouns but could not reproduce the same
world without reimplementing the verbs.

### P0 — there is no authoritative living-world edit/event reducer

Python archetype edits, factory edits, entity spawns, action rows, browser
geometry edits, projectile transitions, and inventory mutations each have
their own representations. They do not converge on one revisioned operation
stream with preconditions, effects, and replay.

### P1 — generated demo policy is mixed with projection policy

`world_div_map_model()` installs the gun, music, physics, placement stock,
player body, and sky. A projection should select adapters; a world/program
bundle should select systems. Until those are separate, every projected class
implicitly becomes the same demo game.

### P1 — browser and Python reference cycles can drift

Entity, action, and loop contracts exist in Python, while equivalent or
expanded behavior is manually recreated in JavaScript. Documentation has
already drifted in places (for example, older projectile-expiry descriptions
no longer match current marker/entity behavior). This will worsen as work
behaviors grow.

### P1 — build inputs are not hermetic

The Pluck shader and material catalog are discovered through sibling filesystem
paths and subprocess output. Navigation relies on a local Zig Python package.
The single resulting page is portable, but rebuilding it is not yet described
by a complete content/input manifest.

### P2 — compiled kernels are too fine-grained to own transactions

The numerical helpers are real and valuable, but the host still decides how a
complete operation is sequenced and committed. Compiling progressively larger
behavior/state regions will provide more architectural leverage than adding
more tiny arithmetic plugins.

## Audit 2: from decorative archetypes to productive work

### What “productive” must mean

A decorative object has identity, form, material, and inspection. An
interactive prop adds a host-routed verb. A productive object additionally
participates in reproducible transformations:

```text
typed inputs + actor/tool + location/pose + duration/conditions
    → observable state transition
    → typed outputs + evidence/events
```

A productive world therefore needs more than animations or NPC scripts. It
needs contracts that let the system answer:

- What work is requested, by whom, and against which identity?
- What capabilities and resources are required?
- Who may claim it, and how is double work prevented?
- Where must the actor and tool be posed?
- What progresses while time passes, and what interrupts it?
- What exact world edit proves completion?
- How can another backend replay or continue the work?

### Current substrate: what is already worth keeping

| Substrate | Current strength | Productive-world use |
|---|---|---|
| Stable world/program identities | Strong | Address jobs, resources, actors, tools, ports, and receipts. |
| `ArchetypeLibrary` and `LivingDocumentEdit` | Strong but isolated | Construct structural instances transactionally and publish namespace symbols. |
| `IntelliType` slots/bindings/capabilities | Promising | Infer affordances and connection points from structure. |
| `AbstractUIFactory` + generation-checked heap | Strong reference model, unwired | Dispense/destroy live instances and batch method-call records. |
| Entity mezzanine | Good nouns | Host workers, machines, projectiles, carriers, and organizations. |
| `EntityPose` derivatives | Useful kinematic base | Motion prediction and presentation. |
| Tools/modes/inventory/custody | Useful partial model | Equipment, carried inputs, placed outputs. |
| Action edges/system timer | Good observation seed | Trace requested and completed work once effect records exist. |
| State-loop deployment | Good ownership/timing seed | Deploy behavior executors and production processes off the render thread. |
| World object/semantic-part spans | Strong | Target a specific machine face, socket, bin, wall, or material patch. |
| Navigation and physics | Functional | Move actors to work sites and maintain spatial constraints. |

The problem is integration, not absence of all foundations.

### Missing contract 1 — typed capabilities and affordances

Capabilities are currently strings such as `interact`, `receive-materials`, or
`fire-projectile`. Productive matching needs a signature:

```text
capability identity + version
accepted actor/tool/resource types
target/part role
preconditions and permissions
declared reads and writes
effects and outputs
duration/clock/interruptibility
failure vocabulary
implementation/deployment reference
```

An affordance is the binding of such a capability to a particular object or
semantic part. “This machine has an input hopper” should identify the hopper
part, accepted material types, capacity, approach anchor, and `deposit`
behavior—not merely add `receive-materials` to an object-level string list.

### Missing contract 2 — behavior graphs and reducers

The first behavior vocabulary should be deliberately small:

- `move-to`
- `orient-to`
- `acquire`
- `carry`
- `deposit`
- `operate`
- `wait/hold`
- `transform/produce`
- `release`

Each behavior needs explicit states such as proposed, admitted, running,
blocked, completed, failed, cancelled, and compensated. It should emit typed
events and world edits rather than directly mutate browser maps.

The behavior graph must be the portable authority. JavaScript may execute a
behavior interpreter initially; later the compiler can lower numerical and
control regions into workers/Wasm. Starting with a typed interpreter is better
than adding another hardcoded tool-name branch.

### Missing contract 3 — semantic poses, attachments, and work anchors

`EntityPose` currently carries coordinate space, position, velocity,
acceleration, jerk, and one facing vector. This is sufficient for point-like
movement but not work.

The minimum productive pose ABI needs:

- parent/reference frame identity and revision;
- orientation (quaternion or an orthonormal frame), not only facing;
- linear and angular derivatives where physically relevant;
- named attachment frames: hand/tool socket, carry socket, eye/aim, feet/base;
- object affordance anchors: approach, operate, input, output, inspect;
- contact/support set and grounded/attached state;
- occupancy/reservation of a work anchor;
- pose tolerance, not exact-coordinate equality;
- timestamp/generation and authority;
- optional posture/animation label separated from physical pose.

This need not begin with a skeletal character. A capsule worker with one
`tool-socket` and a machine with one `operate-anchor` is enough to establish
the correct contract.

### Missing contract 4 — work orders, claims, and progress

Organizations currently corral members but do not allocate work. Add a neutral
work-order record:

```text
identity, requester, target, behavior/recipe
required capabilities and resources
dependency orders
priority and timing policy
claim/lease owner + generation + expiry
current phase and progress
blocked reason / retry policy
resulting edit identities and evidence
```

Claims must be generation-checked, like factory heap slots, so stale workers
cannot complete reassigned work. A work order should reference world identities
and semantic parts, never browser elements.

### Missing contract 5 — resources, recipes, and custody transitions

Inventory has quantities and stack keys, and placement has custody. Productive
work needs typed resource lots and transformations:

- resource/item archetype and unit;
- quantity, quality/properties, provenance;
- container capacity and accepted types;
- custody/ownership/location;
- reservation versus consumed quantity;
- recipe inputs, catalysts/tools, outputs, waste/byproducts;
- conservation checks where applicable;
- transaction identity connecting consumption to production.

The existing projectile entity→pickup→ammo transition is a useful tiny example
of a resource lifecycle, but it is implemented directly in JavaScript. It
should eventually become an ordinary recipe/transition exercised by the same
reducer as other work.

### Missing contract 6 — agency and scheduling

Controllers are string kinds with parameter maps. The reference entity cycle
implements native input and follower dynamics; it does not select or execute
jobs.

Agency should be layered:

```text
goal/policy → choose work order → claim → plan behavior graph
→ navigation/pose acquisition → execute → commit result
```

Navigation is already an interchangeable kernel and should remain a service,
not become the job planner. Organizations can provide queues, policies, and
permissions while entity controllers retain the final behavior assignment.

### Missing contract 7 — persistence, authority, and replay

Productive work cannot rely on the current cookie/local-storage override. The
authoritative state must include:

- living-document/world revisions;
- archetype/factory instances and heap generations;
- entity identities and semantic poses;
- inventories, resource reservations, and custody;
- work orders, claims, progress, and outcomes;
- behavior events and resulting edits;
- compiler/artifact versions needed to continue execution.

The minimum viable form is an append-only event/edit log plus periodic
snapshots. Browser local storage can remain one storage backend, but it should
store the neutral records rather than a projection-specific selection of
fields.

### Missing contract 8 — observability and refusal

Action-edge issue counts are not enough for work. Every behavior should expose
requested, admitted, started, progress, blocked, completed, failed, cancelled,
and compensated events with causal links.

Unknown capabilities, missing implementations, conflicting writers, stale
claims, invalid poses, and insufficient resources must refuse explicitly.
Silently falling back to a visual animation would make the world decorative
again while pretending work occurred.

## Recommended implementation order

### Phase 0 — make the compilation boundary truthful

1. Replace the four-item bespoke list with a per-subsystem provenance table.
2. Mark each operation as compiled kernel, compiled control, model-driven
   adapter, or bespoke semantic runtime.
3. Publish build-input digests for the Pluck shader/catalog and navigation
   toolchain.
4. Add a generated audit/coverage assertion so new browser semantics cannot be
   added without an authority classification.

### Phase 1 — one minimal productive vertical slice

Build exactly one worker, one worksite, one resource, and one recipe:

1. Define typed capability, affordance, semantic pose anchor, work order,
   resource lot, and behavior-event records.
2. Give the worker `move`, `carry`, and `operate` capabilities plus a
   `tool-socket` pose frame.
3. Give the worksite an approach anchor, one input port, one operate anchor,
   and one output port.
4. Express the job as `move-to → acquire → move-to → deposit → operate →
   produce → release`.
5. Execute it first with a generic deterministic behavior reducer, even if that
   reducer is JavaScript.
6. Commit every transition as a neutral edit/event and restore it after reload.

Acceptance criterion: the same event stream can be replayed headlessly to the
same world/inventory/work state without DOM, Canvas, WebGL, or pointer input.

### Phase 2 — connect the existing dormant substrates

1. Make `ArchetypeLibrary.instantiate()` edits part of the authoritative world
   edit stream and transport them in the page model.
2. Use `IntelliType` slots and bindings to publish affordances.
3. Connect `AbstractUIFactory` dispense/destroy/broadcast records to world
   instances and organizations.
4. Make entity spawns, inventory custody changes, placement, and projectile
   transitions use the same reducer.
5. Replace tool-name checks with capability/behavior resolution.

### Phase 3 — compile behavior regions

1. Represent behavior graphs as control plus state regions with declared
   reads, writes, effects, clocks, and interruption points.
2. Reuse state-loop planning for ownership and worker placement.
3. Lower numeric interiors—physics, progress laws, transforms—to existing
   FusedProgram/SSA/Wasm paths.
4. Keep irreducible browser effects behind typed host capabilities.
5. Require compiled/interpreted parity tests against the same event log.

### Phase 4 — richer pose and multi-agent work

Only after the vertical slice is stable, add multiple anchors, two-handed/tool
poses, cooperative claims, queues, dynamic factories, material networks, and
graph-backed portal/work routing. These should extend the same contracts rather
than add special game systems to `abstract_ui_div_map.py`.

## What should remain bespoke

Some host code is correct and unavoidable:

- pointer lock, file chooser, AudioContext permission, sensors, and gamepad API;
- DOM node creation and accessibility attributes;
- WebGL resource allocation, framebuffer setup, and draw submission;
- Canvas paint submission;
- cookie/local-storage access as a storage adapter;
- worker construction and transferable-buffer plumbing.

These surfaces should be small typed capability adapters. They should not
choose recipes, mutate inventory, decide work completion, own lifecycle rules,
or encode tool-specific behavior.

## Near-term design constraints

- Do not make archetypes compulsory for all world objects; direct identified
  objects remain valid.
- Do not turn capabilities into nominal class inheritance. Keep structural
  signatures and explicit bindings.
- Do not equate visual pose/animation with authoritative physical or semantic
  pose.
- Do not hide productive transitions in renderer callbacks.
- Do not compile browser authority that should instead be an explicit host
  capability.
- Do not add a second persistence path for work; converge on one edit/event
  stream.
- Do not treat an operation as compiled merely because its JavaScript template
  was emitted by Python.
- Preserve identity across archetype instance, factory allocation, entity,
  world object, semantic parts, inventory lot, work order, and output receipt.

## Bottom line

The Living Data Map already has enough compiler substrate to avoid becoming a
conventional game-engine rewrite. Its next bottleneck is semantic control, not
math throughput or graphics. The decisive move is to make work an explicit,
portable, revisioned behavior/event system and let archetypes contribute typed
parts and affordances to it. Once that exists, workers, tools, machines,
resources, portals, projectiles, DOM, WebGL, and headless execution can all be
different projections or consumers of the same living world rather than
features that only exist inside one page template.
