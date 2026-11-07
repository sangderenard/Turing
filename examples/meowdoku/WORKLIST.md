# Meowdoku work list

The single list of user-requested items still open. Update it as items land.
Rules: the server and page run the same law text; fix generators, not output;
deploy to 8766 as soon as work lands.

## Fountain lane rules

- No fountain lane waits on the large native: law tests and parity run on the 256-id native piece only, render-only work never compiles the law, and the 4k native/wasm fallbacks build only in the background (the server builds WebGPU first, then the fallbacks).
- A dynamic-N law (sort stages, search and candidate walk as real loops, n as a runtime uniform) is being measured by the 3D job: if it compiles in minutes, it becomes the parity reference at any size.

## Render priority (the user, 2026-11-06)

1. Colour modes: grain size, species breakdown; palettes (sea first); "stained glass" grains through the scene's Phong + glass pass; no opaque black backdrop behind the fountain.
2. Then the heat-map curve management, then the emoji style (image atlas + instanced meshes).

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
4. **Third dimension (confirmed wanted):**
   - a plexiglass slab chamber with depth;
   - polygon chambers;
   - auto-rotate;
   - a colinear tube inlet.
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
5. **Page controls** for grain and physics details: size, density, friction, restitution, cohesion, compressibility, fluid, gravity, and so on, all set on the page and fed into the law as inputs, without recompiling.
6. **Fluids, crystals and temperature:**
   - temperature: Brownian agitation through a Langevin thermostat tied to the fluid drag (deterministic hashed noise), plus the system's measured kinetic temperature as a law output shown in the status line;
   - an interstitial fluid setting (air, water, glycerine settling toy);
   - particle compressibility (Hertz–Mindlin), so crystalline packings form.
7. **Grain species:**
   - several species, each with its own size, roughness and shape irregularity;
   - mixed populations.
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
  - the bonds and their breakage are law state, deterministic, and replay bit-identically.
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
