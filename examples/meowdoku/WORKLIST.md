# Meowdoku work list

The single list of user-requested items still open. Update it as items land.
Rules: the server and page run the same law text; fix generators, not output;
deploy to 8766 as soon as work lands.

## In flight (workers)

- **Fountain stage A follow-ups** (stage A landed in ee0883d7):
  - dropped contacts (`n_drop`) are not 0 while grains emit and in dense piles at 4k, and reach 1.2e6 per step at 64k; the row-run walk drops the right-hand cells first (a bias); needs a complete or order-free neighbour selection (stage B) and contact stiffness scaled with pile depth;
  - the 64k pile never settles (E stays near 5e-2 at the 40 s cap), probably the same cause;
  - single grains still exceed their own ballistic ceiling through exchange between species of different mass (total energy holds to 1.7%);
  - emission is now 21% slower (60% at mix 1.0) because the jet falls back through its own nozzle: tune `EMIT_MARGIN`, the bar width and `EMIT_SPAN`.

## Autopilot follow-ups (the base landed in 727be7c2)

- An autopilot for worldwalk (not yet declared).
- A page control to pick the token net or the graph net (only the server action `pilot_arch` exists now).
- Better teachers:
  - tilequest: rushing the nearest monster scores below standing still;
  - delve: it dies in rooms and reaches the goal in about 1 of 6 seeds.
- The platformer and side scroller students are far below their teachers (overfit): more data and DAgger rounds.
- The platformer weights were trained on the old level (the climber was rebuilt in 492dc368): retrain, or let the idle trainer adapt.
- The board `pieceModel` path still leaves out card thickness (the entity games were fixed).
- Verify the live, end-to-end demo handoff for the new games on 8766.

## Fountain queue (next, in order)

1. **Run modes** (landed; browser-verified numbers in the commit message):
   - the compute free-runs (a bounded number of batches in flight, steps per batch adapted to the measured step cost) into a sliding-window recording; the display draws from it at screen rate (`web/compute.js`, `web/recording.js`);
   - Rebirth, Replay, Bounce and Continue (Continue rebuilds the frame's exact state from a keyframe plus the logged inputs);
   - the run is continuous by default; "stop at rest" (the energy `SettleDetector`) and the image replay stay as the `run` option;
   - phone orientation is the gravity vector by default (`web/gravity.js`, runtime inputs `grav_x`, `grav_y`, `lid`, `rows` of the law), with a toggle; iOS asks on the first tap.
   - **Open:** keyframes cost one state copy per second of simulation (25 MB each at 1M ids), so the 1M+ sizes keep a very short window; the image mapping needs a settled pile (run: stop at rest); a phone-driven lid and grid height are page-side inputs the server does not see.
2. **Neighbour list:** in-dispatch neighbour-of-neighbour refresh, one dispatch per step, a full rebuild only rarely.
3. **Counting sort**, with N set at runtime.
4. **Third dimension** (landed, one law change; browser numbers in the commit message):
   - the state is 3-D (`x y z`, `vx vy vz`, spin vector `wx wy wz`); the chamber is a slab between two plexiglass walls at `z = +-depth/2` (runtime `dh`; default 0.03, about one grain diameter, so the pile is the 2-D one; range 0.02-0.06);
   - the basin outline is a polyline of up to 8 vertices (uniforms `px0..py7`; presets box (the old basin), funnel, hexagon, ramp); each segment is a wall of the same contact law;
   - the inlet is a tube: runtime position, direction, cross-section and cone half-angle (`spread` 0 = colinear); presets fan fountain, colinear jet, pour from above, side tube;
   - gravity is a full 3-vector runtime input (`grav_x/y/z`); `web/gravity.js` gives all three, so a phone lying flat pulls into the slab;
   - camera: auto-rotate toggle (slow yaw orbit in the scene-camera layer, tips the view so the depth shows); points are drawn at true radius with z as depth (`PointCloud`, 20-byte points);
   - **Open:** the glass walls are not drawn; the neighbour walk covers about 2 layers (`k_run` / `n_drop` say when it does not); 10M ids need 7-slot (280 MB) bindings, 49 slots/id (was 40): packing the colours (4 slots) or serialising the diagnostics' reduction would bring it back to 6.
5. **Page controls** (landed): Grains (per species radius, density, friction, restitution, roughness, irregularity, share), Physics (stiffness, Hertz blend, cohesion, spin drag, nozzle speed and rate, neighbour cap with the dropped-contact readout), Fluid (air / water / glycerine presets plus density and viscosity), Chamber (depth, polygon, inlet, spread); all server-declared `controls_ui` (`fountain_param`, `fountain_choice`); the live ones are runtime inputs of the law, logged by the recording, the restart ones begin a new run.
6. **Fluids and crystals** (landed): buoyancy plus implicit Stokes and quadratic drag; stiffness `omega` (clamped to the CFL bound) and a Hertz blend; cohesion. **Open:** a measured crystal (a near-monodisperse stiff pile reaching a hexagonal packing) has not been demonstrated.
7. **Grain species** (landed): four species, each with radius, density, friction, restitution, rolling roughness and per-id size jitter (irregularity), and mix shares; the poly-mix slider is the macro that sets the shares.
8. **Pipe and laminar flow:**
   - large grains at a reasonable population fill the area;
   - set by spawn flow rate, or by spawning inside an enclosed vessel that empties through a tube.
9. **Rendering:**
   - every particle is handed to fragment shaders;
   - persistent textures act as memory the points stamp sprites into (per-species sprites, trails and layering over time);
   - colour covers all particles.
  - **Landed (first cut):** a page control "look" (web/particle_styles.js, render only): points (default), liquid (screen-space fluid), sand (shaded impostors + density AO), glow, heat (density and moving share: the packed points carry no velocity), stamp (a decaying 8-bit memory under the points, in board coordinates).
  - **Open:** per-species sprites, stamped layers and trails per id, the memory following the 3D law, a velocity channel in the packed points (heat by speed).

## Later desires (the user's, recorded 2026-11-06)

- **Rigid objects in the fountain**, built from the platformer's map emojis and other drawing primitives:
  - each object is free, pinned/jointed, or fixed;
  - two-way force interaction with the grains (contacts write forces; only the integrator integrates);
  - coupled through engine_toy's machine sim and its torque negotiation (its couplings, the dyno pattern), so a waterwheel in the sand toy can be driven (a motor turns it) or receptive (the sand turns it, delivering torque);
  - first study the existing coupling and collision rules: collisions are force transfer, not penalties.
- **Prebaked rendering and physics for fixed-camera scenes** (the user, verbatim: "we need to build in way to prebake animations, anything with a fixed camera, with solid objects, we could be rendering them to texture and eliminating them from various physics tasks, like meowdoku could be 100% dynamic animation capture"):
  - **static layer:** with a fixed camera, solid objects that don't move render once to colour + depth textures; each frame composites only the dynamic parts over them; the cache is invalidated on camera, light or time-of-day change;
  - **baked colliders:** fixed solids become a distance/occupancy field texture, so grains and bodies query a texture instead of per-object contacts (fixed fountain objects, level tiles, chamber walls);
  - **animation capture:** deterministic animations (Meowdoku piece placement, cat pops, tray and roster effects) are recorded once per (event, parameters) into flipbooks or atlases and replayed, so Meowdoku can run as 100% captured animation;
  - the bake runs through the existing deployment/compile path and is keyed like other cached products (the declared inputs digest), never hand-edited.
- **Fountain regions inside platform games:** a granular material that spills through a course (platformer, side scroller, delve). The same fountain law runs as a region of the level, and its grains interact with the level's tiles and the player.

## Other open items

- **Pilot:** keep trimming activations and cost; profile server autopilot smoothness.
- **Arcade and sun:** confirm the black-render fix on the phone (needs a refresh).
- **Phone fountain:** confirm particles no longer vanish (unit-0, limits and Metal fixes).
