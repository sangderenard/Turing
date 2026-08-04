# Columnar multifluid voxel state machine

## Primitive and coordinates

One occupied slot is one unit voxel and one unit of physics. A voxel is
spatially referenced by its current centroid. Slot numbers are packed-storage
addresses, not world identities.

The world is uniform in two dimensions and continuous in the third:

```text
x = integer column coordinate + 0.5 while column-constrained
y = integer column coordinate + 0.5 while column-constrained
z = continuous centroid coordinate
```

Tiles contain fixed `x × y` column grids, and columns contain a fixed number
of unit-voxel slots. The fixed-capacity layout is deliberate: the physics hot
loop can operate on `tile × x × y × slot` tensors without visiting voxels in
Python.

## State

`ColumnarMultifluidState` holds:

- tile coordinates and column centroids;
- four-neighbor column indices;
- occupancy and centroid/velocity per voxel;
- material fractions whose occupied rows sum to exactly one unit;
- density, pressure, and temperature channels;
- player markers and local physics-domain labels;
- player path origins for managed-time sinusoidal motion;
- column rest heights plus spring displacement and displacement velocity;
- column surface, material-mass, and mean-velocity summaries;
- six independently diffusing hue-band ink liquids;
- dense directional material-transfer proposals;
- managed time and state-machine phase.

The state has a complete tensor checkpoint/restore contract. It contains no
clock, scheduler, or private substep count.

## Managed time

`ColumnarMultifluidEngine` is an `AbstractTensorStateMachine`. It receives one
externally admitted `dt` from `dt_system`, performs exactly one transition,
and reports `Metrics`. It can be registered beside the game-world state
machine under the same parent transaction. A rejected managed window restores
the packed fluid state just like every other registered state.

The game shell should not call an internal fluid clock. It leases time to the
shared dt graph; the dt graph supplies accepted slices to the game and physics
state machines.

## Current parallel transition

The implemented transition uses AbstractTensor broadcasting and reductions
for all per-voxel numeric work:

1. move every player from the accepted managed time along its Python-defined
   sinusoidal path;
2. classify the player load over all columns and advance the damped,
   neighbor-coupled spring sheet;
3. find the nearest player voxel for every occupied voxel in one distance
   broadcast and assign local physics domains inside the capture radius;
4. apply gravity and integrate centroids in one tensor expression;
   material inside an entity's unit-voxel interior also receives a smooth
   outward rejection force;
5. constrain column-domain `x/y` to their centroids and project `z` against
   the displaced packed unit-slot supports;
6. reduce column material mass and mean velocity;
7. gather four-neighbor surfaces and produce dense material-transfer flux
   proposals weighted by material mobility.
8. inject the managed-time hue into six named liquid channels and diffuse each
   channel through the same four-neighbor stencil with its own coefficient.

Topology initialization and YoungMan's tetrahedron case table remain ordinary
Python/NumPy control. They do not perform serial per-voxel physics.

## YoungMan surface

The state machine exposes all nonempty column prisms and player unit cubes as
one batched signed-distance field. `ColumnarSurfaceField` evaluates box SDFs
for every query and primitive through AbstractTensor broadcasting, then uses a
hard or smooth union. The existing YoungMan algorithm evaluates that field at
all tetrahedral vertices and solves every active edge crossing in bulk.

`tetrahedra_from_axes` was added to YoungMan so a world that already owns
physical rectilinear coordinates does not need a second `GridDomain` wrapper.

YoungMan's crossings are exact zeros of its per-edge linear field
interpolation. Curved smooth-union geometry remains an approximation at the
chosen sampling resolution.

## Deliberately not implemented yet

- applying transfer proposals with a conservative scatter/compaction pass;
- pressure projection, viscosity, phase change, or material reactions;
- general contact forces among non-entity voxels inside a player-local domain;
- releasing/repacking freely moving voxels into different columns;
- a desktop-GLSL storage binding for the physics tensors (the browser
  presentation shader is compiled separately from Python expressions).

The current transfer tensor is an inspectable proposal, not fake completed
fluid transport. It is already arranged as a shader-friendly
`column × direction × material` surface for the next tranche.

## Demo

```powershell
python -m src.common.dt_system.fluid_mechanics.columnar_multifluid_demo
```

The demo advances through `ManagedTimeRuntime`, publishes the accepted tensors
to a `StateTable`, runs YoungMan on the same surface field, and writes a visual
report plus a JSON world-state snapshot under
`build/columnar_multifluid_demo/`.

The browser program is also authored in Python. Its literal `TURING_PAGE`
contract names both the state-machine entrypoint and a Python presentation
entrypoint, so it enters the general page-bundle compiler without a
demo-specific builder. The compiler lowers the former to Wasm and the latter
through SSA to a packed WebGL fragment shader:

```python
from pathlib import Path

from src.common.dt_system.fluid_mechanics.columnar_multifluid_web_demo import SOURCE
from src.compiler.site_bundle import build_program_bundle

build_program_bundle(
    SOURCE,
    Path("build/columnar_multifluid_web"),
    source_filename="columnar_multifluid_web_demo.py",
)
```

Its Python source enters the normal AST/ProcessGraph recompiler and is emitted
as WebAssembly. The compiled transition returns three RGB planes, the next
spring/entity state and managed time, and six persistent hue-band ink planes.
The HTML shell only feeds those explicitly named state outputs into the next
Wasm invocation. The bundled Python synthesizer produces one AbstractTensor
sample surface and analyzes it with `AbstractTensor.rfft`; the shell indexes
those features by compiled managed time. Speaker playback is resampled so its
playback rate follows managed-time advance per wall-time advance, and the
entity's compiled x position controls stereo pan. Neither mechanism alters
`dt`. The generated WebGL shader samples the named RGB output planes and
stretches them over the viewport; it contains no second physics
implementation, private time manager, or precomputed visual frames.

### Native Fortran display isolation

The same Python-authored tick can be compiled into the common native
Fortran/C-shell display without the browser bundle. This diagnostic keeps all
45,056-element terrain, state, and audio planes in caller-owned arenas rather
than baking large constants:

```powershell
# First prove the Fortran + Win32 RGB shell with a cheap gradient.
python -m src.compiler.native_fortran_display probe --frames 300

# Then compile and run the real 256 x 176 columnar tick through the same shell.
python -m src.compiler.native_fortran_display columnar --frames 300
```

Use `--compile-only` to separate compiler/antivirus activity from execution.
Generated executables find `initial-state.bin` beside themselves, so they also
work when launched outside their build directory. On the reference Windows
machine one full columnar frame takes about 0.2 seconds; a blank interval much
longer than that is not expected simulation latency.
