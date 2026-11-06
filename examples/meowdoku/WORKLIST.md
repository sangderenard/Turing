# Meowdoku work list

The single list of user-requested items still open. Update it as items land.
Rules: the server and page run the same law text; fix generators, not output;
deploy to 8766 as soon as work lands.

## In flight (workers)

- **Fountain stage A:** nozzle energy (the ymax 3.7 vs 2.6 ballistic ceiling), symmetry, particle conservation, energy-aware settle, basin fills the screen aspect. The "graph forward WASM compile failed (5)" build error must be fixed.
- **Platformers** (platformer, side scroller, delve): better gravity law and z clipping.
- **Platformer lights:** drop the rocket shield and marker lights; one intent-aimed cone spotlight (control input or acceleration, smoothed; velocity as the fallback).
- **Character picker:** a persistent emoji character chosen from a tile set on the left (stored in the profile), replacing the hardcoded frog and similar sprites in platformer-style games.
- **Platformer level generator:** an ice-climber style climb of wide floors, each reachable under the jump law (proved in a test). No impossible single-brick maps.

## Autopilot follow-ups (the base landed in 727be7c2)

- An autopilot for worldwalk (not yet declared).
- A page control to pick the token net or the graph net (only the server action `pilot_arch` exists now).
- Better teachers:
  - tilequest: rushing the nearest monster scores below standing still;
  - delve: it dies in rooms and reaches the goal in about 1 of 6 seeds.
- The platformer and side scroller students are far below their teachers (overfit): more data and DAgger rounds.
- Verify the live, end-to-end demo handoff for the new games on 8766.

## Fountain queue (next, in order)

1. **Run modes:**
   - the compute spins free in a hot loop into a buffer, and the display reads from the buffer;
   - cached recordings with rebirth, replay, bounce and continue;
   - continuous run by default (settle is optional);
   - phone orientation gives gravity by default, with a toggle (iOS asks permission on tap).
2. **Neighbour list:** in-dispatch neighbour-of-neighbour refresh, one dispatch per step, a full rebuild only rarely.
3. **Counting sort**, with N set at runtime.
4. **Third dimension (confirmed wanted):**
   - a plexiglass slab chamber with depth;
   - polygon chambers;
   - auto-rotate;
   - a colinear tube inlet.
5. **Page controls** for grain and physics details: size, density, friction, restitution, cohesion, compressibility, fluid, gravity, and so on, all set on the page and fed into the law as inputs, without recompiling.
6. **Fluids and crystals:**
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

## Other open items

- **Pilot:** keep trimming activations and cost; profile server autopilot smoothness.
- **Arcade and sun:** confirm the black-render fix on the phone (needs a refresh).
- **Phone fountain:** confirm particles no longer vanish (unit-0, limits and Metal fixes).
