# Continuation: viewer realization ring (lane VR)

File: `tools/view_identity_concordance.py` only.  Not committed.  Design:
`docs/CONCORDANCE_SINGLE_API_DESIGN_2026-09-30.md` section 8.

## What changed

- **Source that emits.** `--probe NAME[:annotated]` lowers one
  `probe_emission_chain` program (`lower_probe`: the probe's entry, contract
  and tensor reference, plus `resolved_process_graph_sink` for the core) and
  `emit_module` emits it through `emit_ssa_module_to_c` and/or
  `emit_ssa_function_to_llvm` (`--emit c|llvm|both`, default both).  Nothing is
  compiled, so the artifact rows are MODULE_TEXT and BUFFER_ORDER only.
  `--case NAME --emit ... --emit-root SYMBOL` does the same on an audit case
  (not run: no case root was chosen here).
- **Third node class.** Rows of the pages named by the declarations
  `EMISSION_ARTIFACT` / `EMISSION_FUNCTION` / `EMISSION_UNIT` get
  `kind = 2` (ARTIFACT); they name no identity points (unit ordinals, byte
  lengths are not ids) but keep every causal and cell_ref edge.
- **realize edge class** (kind 4): the book's DERIVED edge from a non-emission
  cell into an emission row, reclassified at extraction, never merged with the
  chain inside the artifact.  Weight 1 (REAL) in diffusion; magenta.
- **Ring placement** (`ring_placement`, saved): `art_class`, `art_backend`,
  `backends`, `art_angle`, `art_radius`.  Clockwise from 12 o'clock with an
  18 degree seam at the top (HUD); per backend (declared order): artifact
  parts (declared `ArtifactPart` order), then each function row (by emission
  time) followed by its units by ordinal; a gap between backends.  Radius
  `tanh(d/2)`, d linear in emission-clock rank from 2.6 to 5.4.
  `ensure_artifacts` defaults old npz files (no ring).
- **GL.** Ring nodes are hidden in `NODE_VERT` (meta flag 4) and drawn by
  `RING_POINT_VERT` in screen space (ellipse inscribed in the window); ring
  edges by `RING_LINE_VERT`, one end on the ring, the other the sphere cell's
  projected position each frame (dimmed on the far side); the limit ellipse
  as a line loop.  Ring edges are zeroed from the sphere edge buffer and from
  the pick lines.  Order-field: ring nodes masked out of `TimeField`.
- **Pick / focus / diffusion.** `pick_at` hits the ring first (10 px);
  C / D focus and diffuse from a ring node; a ring seed turns the sphere to
  its hottest sphere cell.  Labels place ring nodes at their ring pixels.
- **HUD / stdout**: `realization ring  <BACKEND>: N units (s sourced, u
  unsourced)  F functions  A artifacts | ... | realize edges R`;
  `causal_summary` prints `realize N`.

## Verified (PNGs read)

- `--probe bump --list emission_`: C 22 units (18/4), LLVM 21 (16/5), 2+2
  functions, 2+2 artifacts, realize 35.
- `--probe shared --save-graph shots/ring_shared.npz --snapshot
  shots/ring_shared.png`: C 365 units (41 sourced, 324 unsourced), 5
  functions, 2 artifacts; LLVM 42 (28/14), 2, 2; realize 50; core pinned
  23/23.  Ring visible, magenta realize edges reach the sphere's control
  cluster.
- `--graph shots/ring_shared.npz --diffuse --focus "#445" --settle 60`
  (C root unit 14): history 51, consequence 15, flow died at 1 unsourced;
  `shots/diffuse_445_emission_unit_scalar_native_f_Backend_C_MODULE.png`
  (hops -1 emission_function, -2 MINT cell_set root).
- Old npz: `--graph shots/mapping_pins.npz` ->
  `shots/ring_old_npz_mapping.png`, "no emission rows", everything else as
  before.

## Not done

- Interactive click picking of a ring node is unobserved (batch only).
- `--case ... --emit` unrun; `--compile` (file/command/library rows) not
  offered (a build).
- The function <- every-unit DERIVED fan dominates the chain layer; drawn at
  alpha 0.10.
