# AbstractUI device telemetry and dynamics space

A controllable viewport carries two neutral objects beneath its presentation
surface. Neither depends on HTML, Canvas, WebGL, or document geometry.

`DeviceMonitor` is derived in Python from the viewport's ordered control
bindings. It groups each mapped source by device while retaining source,
semantic action, label, actor, and identity. The browser backend currently
renders these as compact signal lamps. Pointer motion/buttons, W/A/S/D, gamepad
sticks, and primary/secondary mouse and gamepad buttons light from live samples; inactive signals stay
dark. Other backends can present or route the same records differently.

`DynamicsSpace` establishes two deliberately separate lanes:

```text
system timer
  ├── user-dynamics: intent → position / velocity / facing
  └── world-physics: geometry → contacts / collision / gravity
```

The current user lane is bound to the viewport actor and reports its live world
position and velocity. The world lane binds document geometry and publishes
contacts, collision, and gravity through compiled physics. It also orders selected stages for
identity specialization, static welding, broad phase, contact generation,
player resolution, and pose publication. Those labels reserve stable graph
identities and compilation destinations; they do not pretend physics exists.

The physics executable contract begins as a SymPy equation set. Stage
selection chooses the active equations, which lower through the canonical
ProcessGraph and compiler SSA into WebAssembly. WebGPU compute, WASM SIMD, and CPU execution
remain backend candidates for the same program rather than separate physics
definitions. The browser reference now binds the first direct SSA→WASM
gravity/contact/traversal program to a compiler-emitted 120 Hz worker. Segmented wall prisms
now feed one selected semantic wall-plane contact into that program. Dense
broad-phase dispatch and welded multi-contact batches remain explicitly
unbound. Presentation stays on its own animation-frame clock and consumes an
preallocated, ownership-transferred latest-only pose snapshot, so a slow frame does not slow physics
and extra physics ticks do not become a rendering backlog.

Both objects are constructed as Python AbstractUI records and serialized into
the page model. JavaScript only realizes their declared signals and channels.
This is the intended migration pattern for the demo: first move authority into
typed Python graph objects, then reduce the page script to a generic backend
which happens to emit the same interface.
