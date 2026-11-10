# Hidden first-frame Studio integration boundary

Source-only implementation. No server restart, default product rebuild,
activation, or publication was performed for this batch.

## Executable path now present

- `ComputeRuntime.preparePausedFirstFrame` requires a fresh runtime and full
  named state. It validates the manifest before device allocation, executes
  exactly one existing recorded snapshot (bounded native steps), waits for
  queue/error scopes, checks every declared diagnostic and conservation
  counter, reads complete state/pose, then remains paused. It does not pump
  extra frames or emit snapshot callbacks. `disposeAfterIdle` cancels ownership
  without destroying an in-flight load, batch, or readback.
- `studioCapturedPhysicsFrame` consumes the actually displayed Studio unit,
  containing room and subject pose. It does not read Exposure defaults and
  refuses non-marble bodies for the spherical law.
- `marble_studio_dynamic.js` preserves displayed rotation, full velocity/spin
  seed, source triangle/material/normal identity, static room and camera.
  Each output is a new immutable frame; old ray-packet arrays are unchanged.
  Non-off prescribed preview motion remains explicitly unsupported.
- `studio_scene_prepare.py` binds the real JS visible-scene capture to exact
  native-metre BVH triangles and source IDs, verifies its ray coordinates and
  material identities, and proves the complete room tiling before replacing
  it with the incumbent six inward planes. Incomplete contact admission never
  returns a partial usable packet.
- `studio_mesh_environment.py` opts the existing particle law into the native
  mesh local-feature manifold. Real seams have one contact; concave corners
  retain independent contacts. Mutated baked vertices and failed finite/sign/
  feature checks feed the existing `n_out` diagnostic. Build selection is
  explicit `contact_model: native-local-features-v1`; default laws are not
  changed by this wrapper. Product identity includes manifold implementation
  and the admitted native boundary digests. Checkpoints retain source ownership.

## Validation

- `node tests/meowdoku_compute_prepare_check.mjs`: 9 grouped lifecycle/race
  checks pass using actual ComputeRuntime and transport doubles, no GPU.
- `node tests/meowdoku_studio_dynamic_check.mjs`: displayed pose rotations,
  half-turns, scale, full-state seed, material/normal/camera preservation,
  immutable old/new packets, shear/reflection and incomplete-motion refusal.
- `pytest tests/test_meowdoku_studio_scene_prepare.py`: 5 passed, 0.77 s;
  actual JS room/table capture, every source triangle, immutable bake, exact
  native/ray frame, missing/duplicate room and compound refusal.
- `pytest tests/test_meowdoku_studio_mesh_environment.py`: 5 passed, 28.93 s;
  complete native law corner forces, seam equivalence, geometry mutation
  refusal, product identity, full existing SSA/WebGPU ABI emission. No GPU
  product execution occurred in this gate.

## Not yet a deployed feature

The active toy preview has not been replaced. A complete current-scene contact
packet, scene-specific product delivery, real hidden native GPU execution,
both-renderer staged receipts, and one final revision-checked reveal must all
be connected and tested before activation. Exact authored table compound union
is being added separately. Mixed fixture boundaries and arbitrary compound
topology must not be silently omitted. The ray reader's initial bounded dynamic
adapter uses geometric face normals and has narrower material/media/camera
support than the authoring renderer; do not call it full optical equivalence.

The running server currently retains validated default fountain products with
`--retain-fountain-products`. A physics release must deliberately validate and
publish its separate product and assets; a normal dirty-source startup is not
a safe substitute.
