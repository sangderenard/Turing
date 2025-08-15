# Bath–Fluid Coupling Handshake

Roadmap section **C** sketches the handshake between the zero-dimensional `Bath` model and a spatial fluid solver. This document describes the data exchanged and the ordering of operations used in the simulator.

## Data exchanged

The coupling operates on a minimal set of fields:

| Symbol | Description |
|---|---|
| `centers` | `(N,3)` array of cell center-of-mass positions. |
| `vols` | `(N,)` array of cell volumes used to compute `dV`. |
| `dV` | Per-cell volume change over the last step. Converted to mass when using discrete SPH. |
| `dS` | Per-cell solute changes (currently zeroed by the coupler). |
| `P` | Pressure samples returned by the fluid engine for feedback. |

Inside-cell fluids default to a **voxel** (MAC grid) representation. Discrete particle clouds are enabled only when a caller explicitly requests the `"discrete"` kind.

## Step ordering

1. **Softbody update** – advance cell mechanics and obtain tentative volumes.
2. **Osmotic step** – compute osmotic fluxes and update proposed `dV`, `dS`.
3. **Coupling A** – `BathFluidCoupler.exchange` converts `dV` into fluid sources and primes the engine.
4. **Fluid step** – advance the voxel or discrete fluid solver by `dt`.
5. **Coupling B** – sample fluid pressure around each cell and blend into `Bath.pressure`.
6. **Feedback** – corrected fluxes are returned to the softbody and osmotic engines.
7. **Conservation check** – verify global volume and solute conservation across Bath and fluid.

This handshake keeps the bulk Bath model authoritative while allowing higher-fidelity fluid solvers to participate in the simulation only when desired.
