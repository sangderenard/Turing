# AbstractUI state-loop deployment

The Living Data Map now treats time-bearing state machines as deployment
regions, not as incidental calls inside its renderer. The neutral contract is
`StateLoop`: an identity and domain, a clock, read/write sets, observable
effects, a preferred isolation, and (for fixed clocks) a frequency.

The compiler enforces one writer for each state field. DOM, Canvas, WebGL,
and animation-frame effects have main-thread affinity. A fixed-step loop with
no presentation effects is eligible for a dedicated worker. Cross-host data
is a recycled, capacity-one, latest-complete-snapshot channel: simulation
never waits for rendering and rendering never accumulates stale frames.

The channel preallocates three fixed-capacity `ArrayBuffer` slabs before the
first tick. Worker and presenter exchange exclusive ownership with transferable
buffers; no mutex, spin lock, per-frame body array, or per-frame backing-store
allocation occurs. Stable numeric slots and generation counters prevent a late
snapshot from being mistaken for a newly reused projectile identity.

Control handoff uses a second monotonic generation per live body. Navigation
or manual input publishes an X/Z proposal with that generation; presentation
retains the proposed position until physics returns the same generation, then
accepts the worker's collision-corrected result. This prevents a stale worker
snapshot and a fresh control frame from alternately moving the player.

The HTML projection measures rendered element border frames with
`getBoundingClientRect()` relative to the map root. Pointer conversion uses the
same viewport-to-local transform. This avoids mixing page offsets, viewport
coordinates, nested layout offsets, and scroll state in route and entity-dot
placement.

Route samples remain authoritative in traversal coordinates. For HTML, each
world point is projected with a continuous hierarchy blend: a room or building
chart contributes most near its interior and fades to its parent at the wall.
This avoids discontinuities from selecting a different smallest container for
every polyline sample. For WebGL and the software mesh viewer, the same
certified samples become a thin, non-colliding floor ribbon in world space.
Route presentation never feeds back into pathfinding.

```text
browser events ── control.intent ──> physics worker @ 120 Hz
                                          │
                               latest body-pose snapshot
                                          │
                                          v
                               graphics @ animation frame
                               (DOM / WebGL / Canvas)
```

The physics worker source is emitted from the deployment model and wraps the
existing SymPy → SSA → WebAssembly physics program. It owns body position and
velocity, advances players and projectiles independently, and publishes copied
snapshots. The graphics frame only submits current intent, consumes the newest
snapshot, updates presentation objects, and draws. The former synchronous
WASM invocation remains a compatibility fallback when workers are unavailable.

`identify_state_loops()` recognizes explicit Python annotations such as:

```python
@state_loop(domain="world.physics", clock="fixed-step",
            frequency_hz=120, isolation="worker")
def integrate(self, dt):
    self.position = self.position + self.velocity * dt
```

Discovery parses source without importing or executing it. Isolation is
explicit rather than inferred from arbitrary `while` loops: a loop's clock,
effects, and state ownership are semantic promises that a backend can reject
when contradictory. Existing marked class-state-machine lowering can later
publish this same annotation beside its `StateMachineTick` plan.

The generated page includes `emission_provenance`, distinguishing compiler
products (physics WASM and worker), neutral AbstractUI authority (document and
palette), and remaining browser backend templates (DOM construction, graphics
adapters, input adapter, inspector layout). This is the incremental route away
from a bespoke demo without pretending all browser glue is portable already.
