# Meowdoku work list

The single list of user-requested items still open. Update it as items land.
Rules: the server and page run the same law text; fix generators, not output;
deploy to 8766 as soon as work lands.

## Fountain lane rules

- No fountain lane waits on the large native: law tests and parity run on the 256-id native piece only, render-only work never compiles the law, and the 4k native/wasm fallbacks build only in the background (the server builds WebGPU first, then the fallbacks).
- A dynamic-N law (sort stages, search and candidate walk as real loops, n as a runtime uniform) is being measured by the 3D job: if it compiles in minutes, it becomes the parity reference at any size.

## Next big fountain work, right after the 3D change lands (the user, 2026-11-06)

**Confinement shapes, rigid-body actors and boundary regions, paired with the detailed grain size and shape settings:**
- confinement as shapes: polygon chambers and inner obstacles as boundary regions (the 3D job brings a static polygon outline; this makes them first-class);
- rigid-body actors: shapes that are fixed, kinematic (driven: a prescribed rotation or motion) or dynamic (free or pinned, moved by the grains' forces), with two-way contact forces, through the engine_toy machine-sim coupling/torque path (the same path as the waterwheel and the teeter-totter);
- the first target is a ROTATING POLYGON container or drum that sorts grains (size and shape segregation: the rotating-drum / Brazil-nut effects), so it needs the detailed per-species size distribution, roughness and shape irregularity at the same time;
- the fixed shapes can later use the baked distance-field colliders (prebake item).

## Fountain physics follow-ups (the user, 2026-11-06)

- **Rotating-frame forces (in progress, the 3D job's next landing):** case spin and rock only rotate gravity today, so at gravity 0 they do nothing. Fix: in the case frame add the Coriolis, centrifugal and Euler terms from runtime uniforms omega, domega and the pivot; the page writes them from the spin or rock motion.
- **True 3D volume (queued after the rotating-frame forces):** fountain3d is a 2-layer slab today (depth 0.03 by default, 0.05 max, because the neighbour walk is planar). Next: z cells in the grid, depth up to a box, a visibly 3D default.
- **Port the 3D tunables to 2D (then):** temperature, stiffness, cohesion, spin drag, nozzle, k_run, fluid, size distributions, roughness and irregularity; orbeez into 3D.
- **Solvent flow (later):** a moving interstitial fluid with buoyancy and displacement (the space around grains), so real liquids and gases can be modelled; today's fluid is a still medium (drag + buoyancy only).
- **Watch the case move (in progress, a render job):** a toggle (default off: the camera rides with the case) that rotates the fountain's objects in the world by the case angle; plus the backlit slide image on the plate (positive or negative) through the plate material's texture layers.

## Chambers, plumbing and the waterwheel (the user, 2026-11-07), queued after the true 3D volume

A. **Signed-distance chambers:** primitives (box/cube, sphere/circle, capsule/ramp, torus/annulus, hourglass, polygon prism, tetrahedron, peg array) composed by union/subtract as runtime uniforms; walls are the thick sided band contact from the composed SDF. Presets: box, funnel, hexagon, ramp, toroidal shell, sphere, cube, tetrahedron, hourglass, and the oil & glycerine toy (alternating ramps, an hourglass, a plinko field).
B. **Ports, pumps and lines:** outlet regions park grains into a line (transit time from length and pump rate) and re-emit them FIFO at an inlet port, deterministically in law state; see-through glass tubing drawn with the transiting grains.
C. **Hinged rigid actors + waterwheel:** a body of SDF primitives on one hinge, two-way contact forces, angle as law state, coupled through the engine_toy machine-sim torque path; preset "waterwheel fountain" (nozzle -> wheel -> outlet -> pump line -> nozzle).

## Render priority (the user, 2026-11-06)

1. Colour modes: grain size, species breakdown; palettes (sea first); "stained glass" grains through the scene's Phong + glass pass; no opaque black backdrop behind the fountain.
2. Then the heat-map curve management, then the emoji style (image atlas + instanced meshes).

## In flight (workers)

- **Fountain stage A follow-ups** (stage A landed in ee0883d7):
  - the "rightward blast": the user's hypothesis is an honest pressure wave from the naive initial deposit (no pipes or nozzles yet), rather than the drop bias. Being checked by the 3D job on the 256 piece: drops during the blast, the momentum asymmetry over time, and a mirrored emission order (does the asymmetry flip?);
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
4b. **Glyph start (a 3D must, as soon as 3D is stable):**
   - you type a glyph sequence; its bounding box becomes a closed cube shell (all six walls);
   - inside, the glyph or glyph string is built from the run's N particles filling each glyph's ENCLOSED solid volume (not its surface). The glyph meshes are extrusions of an outline (token_mesh in web_export: outline with holes, depth_ratio), so "inside" is exact and cheap: an even-odd point-in-outline test (holes count) and |z| < depth/2, from the same outline data the exported mesh used (emoji keep only their largest component). Grain size from N·(4/3)πr³ = φ·V (φ ≈ 0.64 random, or ≈ 0.74 on a lattice for a crystalline start); with a species mix, scale the drawn radii to fit;
   - the simulation starts from that state at rest, then evolves (gravity, tilt, temperature and the rest apply as usual);
   - the starting state is part of the replay (same text, same N, same result).
4c. **Teeter-totter case (after 3D, with glyph start):**
   - the whole chamber sits on a fulcrum: one rotational degree of freedom, a pivot under (or beside) the case;
   - the grains' own mass drives it: the contact forces the grains put on the walls (plus gravity on the case itself) give the torque about the pivot, and the case angle is integrated from that over time, like the desk-toy wave rockers;
   - options: case mass and inertia, pivot position, pivot friction or damping, an optional restoring spring, start tilt;
   - the case angle is law state (exact replay). In the case frame the law sees gravity rotated by the angle, plus the frame's inertial terms (Euler and centrifugal), which is the honest version of the case spin, which only rotates gravity;
   - the pivot is the first rigid-body coupling, so build it on the engine_toy machine-sim coupling/torque path that the rigid emoji objects and the waterwheel will use, not as a one-off.
5. **Page controls** (landed): Grains (per species radius, density, friction, restitution, roughness, irregularity, share), Physics (stiffness, Hertz blend, cohesion, spin drag, nozzle speed and rate, neighbour cap with the dropped-contact readout), Fluid (air / water / glycerine presets plus density and viscosity), Chamber (depth, polygon, inlet, spread); all server-declared `controls_ui` (`fountain_param`, `fountain_choice`); the live ones are runtime inputs of the law, logged by the recording, the restart ones begin a new run.
6. **Fluids, crystals and temperature** (landed): temperature `temp` (kT) is a Langevin thermostat tied to the fluid drag (variance 2 gamma m kT / DT per component, noise a hash of id, step and component; 0 = off exactly), log slider in Physics, measured T = ke / (3 n_live) in the status line; buoyancy plus implicit Stokes and quadratic drag; stiffness `omega` (clamped to the CFL bound) and a Hertz blend; cohesion. **Open:** a measured crystal (a near-monodisperse stiff pile reaching a hexagonal packing) has not been demonstrated.
7. **Grain species** (landed): four species, each with radius, density, friction, restitution, rolling roughness and per-id size jitter (irregularity), and mix shares; the poly-mix slider is the macro that sets the shares.
2D: the same tunables landed in the 2D law (`fountain.py`, unchanged in dimension): species radius 0.1x-10x (log; restart), friction 0-2, restitution, density, grid cell as a runtime uniform; ORBEEZ species (swell rate, max ratio; the radius is state `rs`; growth ~ exposed surface x solvent; air does not swell). The 3D law is its own module and products (`fountain3d.py`, `fountain3d_build.py`, game "fountain3d"); the 2D one is the page's `fountain`.
8. **Pipe and laminar flow:**
   - large grains at a reasonable population fill the area;
   - set by spawn flow rate, or by spawning inside an enclosed vessel that empties through a tube.
9. **Rendering:**
   - every particle is handed to fragment shaders;
   - persistent textures act as memory the points stamp sprites into (per-species sprites, trails and layering over time);
   - colour covers all particles.
  - **Landed (first cut):** a page control "look" (web/particle_styles.js, render only): points (default), liquid (screen-space fluid), sand (shaded impostors + density AO), glow, heat (density and moving share: the packed points carry no velocity), stamp (a decaying 8-bit memory under the points, in board coordinates).
  - **Open:** per-species sprites, stamped layers and trails per id, the memory following the 3D law, a velocity channel in the packed points (heat by speed).
  - **Landed (noise filter):** web/noise_filter.js, render only, layered over any look (toggle, replace-with mean / spectral class, strength, threshold; local controls, remembered): the colour field is splat to a board texture and mip-mapped; the gaps between mip levels are band variances, the slope of their log against scale is beta (white 0, pink 1, brown 2, a step edge 3), and speckle (variance in the fine bands) is replaced by the region's coarse mean or by a fixed class colour (white, pink, brown, grey = luminance-only white, solid). Needs half-float render targets; off, with the reason in the control title, without them.

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
- **Self-attraction, dipoles and aggregates in the fountain** (for dynamic richness, and to calm the sim):
  - **attraction:** short-range cohesion (liquid-bridge / van der Waals style; the 3D job adds a basic cohesion uniform) and phase behaviour, e.g. a wet clump, a dry flow, a crystal;
  - **dipoles:** each grain carries a moment (a per-id orientation vector) with an orientation-dependent pair force and torque (magnetic/electric), so chains and needles form;
  - **aggregates:** grains bonded at rest fuse into a cluster that moves as one rigid body (fewer degrees of freedom, so the sim calms down and gets cheaper), and break apart when bond stress exceeds a strength; the same rigid-body machinery as the emoji objects;
  - the bonds and their breakage are law state, deterministic, and replay bit-identically;
  - **lattice-sensitive crystal shading** (grouped here by the user):
    - the law exports per-grain bond-orientational order ψ6 = mean of e^{6iθ} over neighbours, in the existing neighbour walk: crystallinity |ψ6| and lattice angle arg(ψ6)/6;
    - the scene Phong shader blends crystalline grains into a gem material: facet normals rotated by the lattice angle, so whole domains glint together; rainbow (dispersion-like) glints from the lattice-to-light/view angle; ruby, sapphire, emerald and opal presets and faceted "gem pebbles";
    - cheap: a few multiply-adds per contact plus a dot product and a hue ramp per fragment.
- **Late-stage colour from per-id data** (fountain):
  - the law hands the render per-id attributes instead of a baked rgba8: species, grain size, smoothness/roughness, irregularity, speed, age/emission order, contact count, temperature;
  - a late-stage shader composes colour from them through maps (ramps, palettes) and simple blend recipes (e.g. hue = species, value = size, saturation = speed, or a weighted mix of any channels), quick to switch, chosen for dramatic informational and aesthetic effect;
  - colour modes: species, grain size, grain smoothness, plus the existing time strata, launch angle and image;
  - in 3D: depth shadow (grains shadow those behind them in the slab) and depth alpha (fade with depth) as recipe terms;
  - this is the same per-id data the noise filter and the render styles consume.
- **Client/server load share for the fountain page:**
  - a percentage control that splits the simulation between the page's GPU and the server, by dividing the adjacency (neighbour) list: a partition of the grid cells, so each side steps its own grains;
  - the boundary grains are exchanged each step as a halo; the same law text runs on both sides;
  - the split moves with the measured step cost on each side;
  - the result stays deterministic and bit-identical to a single-side run.
- **Fountain regions inside platform games:** a granular material that spills through a course (platformer, side scroller, delve). The same fountain law runs as a region of the level, and its grains interact with the level's tiles and the player.

## Other open items

- **Pilot:** keep trimming activations and cost; profile server autopilot smoothness.
- **Arcade and sun:** confirm the black-render fix on the phone (needs a refresh).
- **Phone fountain:** confirm particles no longer vanish (unit-0, limits and Metal fixes).

## Fountain: signed-distance chambers, ports/pumps/lines, hinged actors (DESIGN, written before any code; none of it is built)

State of the walls now: the 2D law has three planes plus (non-box shapes) a polyline of up to 8 vertices; the 3D law has the polyline (thick, sided, overlapping half-spaces, penetration saturating at 6 radii) plus two plexiglass planes and a lid. Closed shapes are inscribed in the view window.

**A. Signed-distance chambers (2D and 3D).**
- A chamber is a fixed-capacity list of primitives as uniforms: kind, pose (x, y, z, angle), size (3), flags (union / subtract). The wall contact stays the existing thick sided band: distance d and normal n from the composed SDF (analytic gradient per primitive; a central difference only for the peg array's nearest-peg search), force by `_contact`.
- CONFLICT / cost: a runtime `kind` switch evaluated for every slot costs every SDF for every slot per grain (7 kinds x 16 slots x gradients) -- far beyond the polyline walls. Proposal: slots are typed, a few per kind (e.g. 4 box/prism, 4 sphere/capsule, 2 torus, 1 hourglass, 1 peg array, 1 tetra), each slot evaluating exactly one SDF; the preset fills them. Measure ms/step against the polyline before widening.
- CONFLICT: the 3D solids (sphere, cube, tetrahedron, torus) need the true 3D volume first (the neighbour walk is planar with a ~2-layer slab; the z cells of the true-volume item). Until then they are 2D shapes extruded through the slab.
- Peg array: rows x cols at a pitch, staggered, nearest-peg by index arithmetic (no loop): one primitive.
- Presets: box, funnel, hexagon, ramp (as now, polylines kept); toroidal shell, sphere, cube, tetrahedron, hourglass; "oil & glycerine toy" = alternating capsule ramps, an hourglass throat, a staggered peg field, a floor; default fluid glycerine, two species of different density and colour.
- Inlet/outlet placement per preset (the inlet is already clamped to the shape's chord).

**B. Ports, pumps and lines.**
- An outlet is a region on a wall; a grain whose centre enters it is parked into a LINE: new law state `tdue` (the time it re-emerges = now + length / speed from the pump rate), then it takes the existing ARMED path at the line's inlet port (the slot-clear test and the FIFO by due time then id are already how emission works). Counters gain `n_in_line`; the partition stays exact.
- CONFLICT: today a grain's emit time is `pid x emit_rate` (a pure function of the id) and `age` is a state shared by all ids; a recycled grain needs its own due time (a new state column, 2 slots/id more: check the 10M binding budget, it is exactly full at 42 slots with the 2D law).
- A pump: rate (grains/s) and lift are uniforms of the line (rate limits how many may leave the outlet per second: the outlet admits a grain only if the line's counter allows, deterministic in id order).
- The page draws each line as glass tubing (a path of segments through the scene material path) and the transiting grains interpolated along it from `tdue` (no per-frame readback beyond the packed points: add the due-time fraction to the packing's spare word).

**C. Hinged rigid actors + waterwheel.**
- A body = SDF primitives (hub + paddles) with one hinge DOF; the grains' wall contact is two-way: the force and torque on the body are the negated sums of the contact forces over grains, a fixed-order reduction (like `ke`), and the body's angle / angular velocity are law state integrated with inertia and hinge damping.
- CONFLICT: the law is one batched per-id program and a rank-0 state has no home (the WebGPU backend recomputes rank-0 values; a state must be a lane). The body state would live as a small replicated lane (id 0 owns it, the others read it with a gather) or as a second, one-element program stepped between batches by the page -- the second breaks exact replay unless its inputs are logged. To be decided with the engine_toy owner BEFORE coding. The coupling must go through the engine_toy machine-sim torque / coupling path (not read yet by this branch): it must be read first; no parallel torque model.
- Preset "waterwheel fountain": the nozzle falls on the wheel; the outlet at the bottom feeds a pump line back to the nozzle (needs B).

**True 3D volume (next, before the 3D solids).** z cells in the grid (cells in x, y and z; the 3x3x3 walk = three row runs per z neighbour = 9 runs), depth a runtime uniform up to a box-like depth, default several grain layers. Cost: 3x the binary searches and 3x the candidates per grain; K_RUN_MAX must grow (at depth d the dense-pack candidates per run are ~ 16 d / grain diameter); watch `n_drop`. The compile grows with it (the 3D native 256 compile already exceeds 90 minutes): parity is the NumPy lane (its product emits in ~14 minutes) until the compile is cut down.

**DONE (the 3D tunables are in the 2D law, all neutral at their defaults, one fluid for drag/buoyancy/orbeez; GPU ms/step unchanged; no new per-id column).** Was: **Ports of the 3D tunables to the 2D law:** temperature, stiffness/Hertz, cohesion, spin drag, nozzle speed and rate, k_run with its n_drop readout, fluid, size distributions, roughness/irregularity (orbeez and the polygon walls are in). Each adds uniforms and law terms to the 2D law and so re-keys the 2D products; 2D native 256 compile is ~25 minutes.
