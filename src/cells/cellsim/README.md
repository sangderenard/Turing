# cellsim — Nested mechano–osmotic cell simulator (PhD-ready scaffold)

This package organizes your saline-pressure simulator into a scalable architecture with:
- **Organelles as sub-compartments** (excluded volume + inner exchange).
- **Kedem–Katchalsky transport** with solvent drag.
- **Membrane mechanics** via Laplace tension and optional anchoring.
- Clean separation for **future PhD extensions**: electrochemistry, CRN, gates, IMEX, SSA.

## Layout
- `core/` — constants, geometry, numerics.
- `data/` — dataclasses: `Cell`, `Organelle`, `Bath`; species registry.
- `transport/` — flux laws (`kedem_katchalsky` implemented).
- `mechanics/` — membrane tension model.
- `organelles/` — `inner_exchange()` fully implemented.
- `engine/` — `SalineEngine` orchestrates inner (organelle) then outer (cell↔bath).
- `chemistry/`, `membranes/` — stubs for CRN and channel/porter gating (prepared).
- `placement/` — stubs to connect to your BitBitBuffer.
- `examples/` — `demo_sim.py` small runnable demo.

## Run demo
```bash
python -m cellsim.examples.demo_sim
```
