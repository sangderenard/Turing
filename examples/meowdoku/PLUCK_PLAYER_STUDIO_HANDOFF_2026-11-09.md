# Studio player-body reuse findings

Inspected the canonical Windows Turing and sibling spectral-analyzer sources.
This is an integration receipt, not a claim that Studio has a rigid-body player.

## Working Studio controls

`web/marble_studio.js` already offers Particle simulation and Marble perspective.
It follows particle zero in the compiled Marble Madness case. `driveGravity`
uses `cameraNudgeGravity` and `ComputeRuntime.setGravity`; it tilts the whole
native toy's gravity direction, not a force on just the chosen marble.

The new `web/marble_drive_input.js` adds focused-canvas WASD/arrow intent to that
same route. D-pad intent takes priority while held; releasing it restores any
held keyboard intent. The center button clears keyboard intent. Keys reset on
canvas/window blur, hidden page, and leaving particle mode. Inputs in the side
panel are not captured. Diagonal gain remains normalized by the existing native
gravity mapper. There is no new body integration, simulation clock, or fake pose.

## Existing Pluck body and timing

- `spectral-analyzer/player_controller.py`: `PlayerController._tick_walk` owns
  WASD, camera yaw, gravity and floor movement; `_apply_room_bounds` delegates to
  `PlayerClipEngine`. Horizontal motion currently changes position directly.
- `spectral-analyzer/player_clip_py.py`: the facade selects the native pybind,
  ctypes, or Python clipping implementation. `tick`, `read_state`, and the
  32-column state tensor are the existing interface. Native selection here was
  measured as `_spectral_kernels.PlayerClipEngine`, not inferred from a name.
- `spectral-analyzer/csrc/include/player_clip.h` and
  `csrc/kernels/player_clip.cpp`: native slab correction, clipped velocities,
  atomically published completed frame, generation counter and owned history.
  `tick` does not integrate applied force or return an equal/opposite body force.
- `spectral-analyzer/camera_software/managed_time_bridge.py`:
  `CameraManagedTimeBridge` translates `CameraSliceJob` into the existing Turing
  `TimeWindowRequest` and calls the transactional commit gate. Its
  `make_request`/`advance` seam is the real camera-owned scientific-time path.
  `exposure_timing.py` supplies camera causality identity and program generation.
  Reuse this boundary for server slice work; do not create another display-clock
  integrator. `spectral-analyzer/physics_backend.py` is explicitly a stub and is
  not evidence of an available rigid-body solver.

## Existing client mechanical-demo path

- `src/compiler/state_loop_deployment.py` emits the fixed-step physics worker.
  `upsert` installs bodies, `wrench` updates force/moment, `stepWorld` consumes
  them at substeps, and the worker publishes completed snapshots. Existing
  `impulse` messages assign a resulting velocity; `player-jump` adds vertical
  delta velocity. A `wrench` persists until changed: it is not a one-shot slice.
- `src/compiler/abstract_ui_div_map.py` registers the player snapshot/body in
  `registerPlayerPhysicsBody`. `resolveProjectilePlayerContacts` computes a
  reciprocal projectile/player impulse and synchronizes player velocity.
  These contacts are client-specific capsule/projectile handling, not Studio's
  current shared native triangle/DEM product. Copying only the UI would hide
  that distinction.

## Integration boundary still required

1. Use the native shared-scene contact product for both the marble and player;
   retain the exact shared scene geometry and declared body mass/contact shape.
   The current Studio preview's sphere DEM ABI has no general per-player wrench.
2. Consume input intent within the existing managed tick/slice boundary. Keep one
   authoritative pose/velocity state and publish it only after the slice commits.
   Rejected or superseded work must not apply the same input twice. The existing
   Pluck camera bridge provides timing, not the missing reciprocal contact law.
3. Bind the scene camera to that published player state and keep orbit mode as
   a presentation choice. Pluck's current kinematic position update cannot also
   advance a body that the native contact solver owns.
4. Validate zero-input rest, sustained wall contact, rolling/sliding course
   contact, player-marble momentum exchange, pause/resume, focus loss, repeated
   body enable/disable, and stale-generation rejection before exposing the body
   option as operative. Retain exact mesh/material/scale coordinates across the
   handoff. No new functional body toggle was added by this worker.

## Stage-sized course geometry added after the first receipt

`web/marble_studio_courses.js` now authors a gentle S route and broad return
route. `makeStudioCourse({kind, stage:{width,length,baseY}, materialId,
railMaterialId, segments})` consumes explicit stage dimensions in Studio units,
never the marble radius. The caller must choose the actual table or room stage;
there is no alternate scale. Both routes fit within the supplied footprint.
Their running surfaces descend gently, with separate raised rails and end stops.
At a 16 x 10 stage the measured maximum running grades are 1.8% and 2.2%.

The returned `vertices` are global authoritative vertex identities. Each
`parts` entry is a closed, outward-wound convex vertical triangular prism with
`id`, `bodyId`, `vertexIndices`, `triangles` using those global indices, and
matching `triangleMeta`. Metadata includes `surfaceId`, exterior/internal
boundary role, and exact adjacent part IDs for internal construction faces.
Shared internal quads use identical triangle diagonals and opposite winding.

`faces` uses the existing Studio [a,b,c,material,normals] contract, excluding
internal construction faces. `faceRecords` links every rendered face back to
its exact part and triangle. No mesh proxy or separate collider was authored.
`surfaces` explicitly certifies constructed planar/convex or planar/nonconvex
patches: the whole S running ribbon is planar but has a nonconvex footprint;
the return's straight entry/exit strips are planar and convex. Folded turn
patches remain individually identified. `surfaceGroups` retains continuous
route regions without pretending that they have a solved collision manifold.

The assembly remains `collisionReady:false`. These touching convex parts are
partitions of ONE body. Their internal faces and joined surface edges must not
produce extra forces. The native integration worker is resolving that authored
compound-union boundary; summing existing per-part contacts is not correct.
There is no course UI, production export registration, or playable course
claim yet. `spawn` and `finish` describe surface points, not body-center poses.
`studioCourseMarbleFit` is radius advice only and never resizes anything.

Ten Node geometry cases cover square/wide/narrow stages and 16..128 requested
segments: every part's closed edges, outward winding and convexity; exact
shared triangle pairs; the complete watertight exterior; part/exterior volume
equality; authored planar certificates; grade and footprint limits; and scale
independence. An isolated CPU rendering of the actual triangles was inspected
at `build/dot-studio-course-20261109/courses.png`. It is geometry QA, not a native
physics run or a live Studio screenshot.

## Measured checks

`node tests/meowdoku_marble_drive_input_check.mjs` passes keyboard aliases,
diagonals, focus/modifier isolation, repeat handling, interruption cleanup,
disposal, and the real extracted Studio gravity adapter with native mapper.
Both modified/new JavaScript files pass `node --check`; web export Python
compiles. The export list includes the new input module.

Native Pluck smoke: a sphere proposed at X=-0.1 against the X=0 wall with
radius 0.25 resolves to X=0.25, reports CLIP_X, publishes velocity X=0 and
generation 1, and records a row in its (256,32) tensor. This proves the existing
native clip interface, not dynamic player-marble contact.

The pre-existing source `meowdoku_marble_sim_preview_check.mjs` suite could not
load because `web/compute.js` imports missing generated `turing-wasm-shell.js`.
No replacement shell was invented. Browser/player movement, GPU dynamics,
production export and active live-view behavior were not exercised here.
