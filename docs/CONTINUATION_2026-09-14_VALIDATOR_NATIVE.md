# Native compilation of the original Python validator

## Requested integration

The user accepted the current managed-tire precision provisionally and asked
to use the work in the original validator's Python version. The source entry
is `tools.run_vehicle_native_assembly._run_dually_python_profile`. Its worker
calls the real repository `run_superstep` with `_DuallyDTState` and the nested
`advance` function, which advances the coupled vehicle, fixture, and tire.
The previously verified managed-tire executable has a different material/state
contract and cannot replace this closure directly.

After the whole-program attempt below, the user explicitly chose
**"Keep Python viewer; compile simulation"**. This supersedes the historical
whole-application scope for the current integration: Python may own display,
input and stage presentation, while the coupled physics, adaptive controller,
internal substeps, state feedback and rollback belong to the native window.
Do not revive the
abandoned Pygame-to-2D-intrinsics approach. The intended presentation route is
authored world objects/geometry delivered to the generic game renderer.

## Current attempt

```powershell
python -u tools/lower_vehicle_validator_program.py --output build/validator_native_resume_20260914
```

Log: `build/validator_native_resume_20260914.log`. This attempt started at
18:11:56 local time, PID 444. The user first said to allow up to 20 minutes, then clarified
that up to an hour is normal and asked for patience. No elapsed-time cutoff
was imposed. It published a
108-unit plan and the exact reduced graph (52,145,381 bytes), including the
actual `advance`, worker, `run_superstep`, and viewer. It terminated naturally
with exit 1 after **2720.846 seconds** (45 minutes). The concrete failure is
four opaque Pygame draw-loop effects in `PythonValidatorViewer.draw`:
loops 584/589/615/640, calls 583/588/600/638 (line, line, polygon, circle).
The graph is acyclic with complete levels. Exact call diagnostics confirm
`extraction_action=reject`, rule `execution:native_extension_abi_required`.
Evidence: `receipt.json`, `failed-process-graph.pkl`, and
`draw-call-diagnostics.json` in the attempt directory. No full-validator SSA,
native artifact, or coupled numerical verdict was produced. This process loaded the compiler before the classifier
change below; do not attribute that change's behavior to this run.

## Planning cost repaired

Two read-only CPython stack samples of the owned compiler process both found
`_is_dispatch_metadata_node` inside `NetworkX.number_of_edges`. Although node
classification was cached, its cache fingerprint counted all graph edges on
every single node query. The two complete node scans now acquire a classifier
once for the stable pass. Single-node callers retain per-query validation;
new passes retain the existing node/edge-count invalidation behavior. No
classification rules, physical laws, numerical tolerances, or native compiler
optimization flags changed.

The focused classifier checks pass 3 tests. The adjacent dispatch/reference/
context-manager group passes 38 tests with one failure:
`test_tensor_method_candidate_requires_tensor_receiver_value` lacks the
expected `tensor` attribute. It reproduces with the HEAD deployment-strategy
module executed in an isolated process, without replacing the checkout.
Logs: `build/validator_dispatch_classifier_tests_20260914.log` and
`build/validator_dispatch_classifier_baseline_20260914.log`.

The fixed classifier also processed the saved real 22,376-node graph in
2.470 seconds (20,204 metadata nodes), with every result checked against the
unchanged uncached classifier. This measures classification only, not a full
build. Evidence: `build/validator_dispatch_classifier_graph_check_20260914.log`.

The source graph plan predates this compiler change. A later run must respect
the compiler fingerprint check; do not relabel the old plan as newly built.

## Python viewer / native simulation work in progress

`src/compiler/vehicle_validator_simulation.py` composes the existing canonical
dually tensor graph with the validator's named input packing, persistent state
feedback, lane-0 error metrics and the real `run_superstep`. It keeps controller
state across native calls and excludes Python presentation callbacks from the
compiled entry. `tools/build_vehicle_validator_simulation.py` builds this entry.
The existing runner has an opt-in `--native-simulation DIRECTORY` route, using
its unchanged viewer and a shared `accept_tick_result` publication method.
These changes are under validation, not a working-native claim yet.

`tests/test_vehicle_validator_simulation.py` compares the new eager entry with
the existing validator's real initialization/tick/feedback and checks snapshot
restoration. Initial test run is in `build/validator_simulation_eager_tests_20260914.log`.

The third eager run passed all four checks (280.82 seconds), including inactive
and active tire state/feedback equality at rtol/atol 1e-12, full snapshot restore,
and the existing viewer acknowledgement check. Log:
`build/validator_simulation_eager_tests_v3_20260914.log`. The first two attempts
exposed a scalar `.abs()` call in the metrics adapter (fixed with built-in
`abs`) and a test-side `float(AbstractTensor)` truncation (fixed by applying
the same `coerce_metrics` normalization used by the real DT controller).
The Python publication branch also now supplies every required `Metrics` field.

At 19:26 local time the eight-lane build was started with:

```powershell
python -u tools/build_vehicle_validator_simulation.py --output build/validator_simulation_native_20260914 --lanes 8
```

Log: `build/validator_simulation_native_20260914.log`. No time limit is imposed.
The builder retains the resolved graph and final SSA, validates structural
findings and C emission, and compiles at O0. A working artifact must have
`manifest.json` and its referenced DLL; source wiring alone is not readiness.
The optional native integration test selects the actual build with
`TURING_VALIDATOR_NATIVE_ARTIFACT` and checks two consecutive windows, including
the persistent controller. It has not run yet.

The build stopped naturally after 798.786 seconds (13 minutes), before SSA
or C emission: `_atomic_region_node_order` rejects cyclic dependencies between
contracted regions (9, 15, 11, 16, 17). `failure.json` has the full stack.
The exact resolved graph is saved; no native artifact exists yet.

The user explicitly required repairing the compiler that created the bad
regions, even if an intermediate graph could be manually repaired. A new
small regression proved a defect in `reduce_scheduled_shader_regions`: a
direct A -> B edge masked a parallel A -> coordinator -> B path. The fusion
legality check subtracted direct endpoint pairs, losing that second path's
external-boundary provenance. The current repair retains coordinator-path
pairs independently; it does not rewrite numerical operations or drop edges.
The regression failed before the change and the region-planning suite passes
all 23 tests afterwards. Logs: `validator_parallel_path_before_20260914.log`
and `validator_parallel_path_after_20260914.log` in `build/`.

The advance function alone passes the old atomic-order check. The helper
`../speaktome/AGENTS/tools/inspect_turing_dispatch_cycle.py` is inspecting all
function graphs using HEAD's fusion module in an isolated process, without
checking out files. Log: `build/validator_simulation_cycle_baseline_20260914.log`.
It reproduced the **identical full-build error** in `balloon_tire_vector_step`.
The retained `cycle.json` and `function-graph.pkl` are in
`build/validator_simulation_cycle_baseline_20260914/`. Its eight cyclic paths
are shape dependencies: e.g. `predicted` directly feeds a reshape and also
feeds `predicted.shape[2]` through a shape tuple into that same reshape.
The repaired compiler passes atomic ordering on that unchanged saved function
graph: `build/validator_simulation_cycle_fixed_20260914.log`.
This correlates the small regression with the real failure; no intermediate
graph was manually reordered or repaired.

The combined region-planning, control-dependency, and function-linking suite
passes 84 tests and fails 14. All 14 failures reproduce with the old fusion
module in an isolated process (`validator_fusion_regressions_baseline_20260914.log`,
31.24 s); no checkout files were replaced. The first repaired source rebuild
is now running:

```powershell
python -u tools/build_vehicle_validator_simulation.py --output build/validator_simulation_native_v2_20260914 --lanes 8
```

Log: `build/validator_simulation_native_v2_20260914.log`. Do not mistake the
successful function-ordering diagnostic for a completed native simulation.

The fresh rebuild passed instantiation (including the original failure) and
reached full control-graph preparation. It then stopped naturally after
1146.769 s on a **different** cycle: atomic control regions 1/3/5 in
`vehicle_tire_recurrence`, involving loop node 288 and the fine/reduced tire
conditional node 287. Its `failure.json` retains the complete message.
Generated simulation source is unchanged byte-for-byte from v1; file SHA256
`986076bff29d3c1a28aa1cee9721e7d6da3d996d7a6b5b06bad70b620bb581ea`.
The first fusion repair is therefore verified in the fresh complete build;
the new control-membership/dependency frontier remains to repair in the compiler.

The isolated `--precompile` run (`build/validator_control_cycle_20260914.log`)
stopped because its isolated shell has no deployment shells for the child
tire functions. A narrower `--control-only` diagnostic, using the actual
region-schedule, projection, nesting and overlay functions, passes on the
unspecialized recurrence (`build/validator_control_cycle_cut_20260914.log`).
Do not treat that as reproducing or fixing the specialized full-build failure.

The exact full deployment is now replaying from the saved resolved graph via
`../speaktome/AGENTS/tools/replay_turing_resolved_deployment.py`, skipping source
preparation. It wraps the real overlay only to save its exact arguments and
the caller's target graph/regions/loop plans on failure; it does not change
planning, scopes, dependencies, or control order. Log:
`build/validator_control_exact_20260914.log`; output directory:
`build/validator_control_exact_20260914`. If it reaches SSA it also retains a
pre-frame-link checkpoint. There is still no SSA/DLL or tested viewer command.

The exact replay reproduced the second cycle and saved `control-snapshot.pkl`
and `control-cycle.json` in `build/validator_control_exact_20260914/`.
Specialization's typed tuple-return leaves (nodes 311/313/315/317, marked
`authored_call_result_projection`) were incorrectly classified as numerical
tensor indexing. Their independent regions escaped the owning loop/branch.
The dispatch classifier now treats this explicit compiler marker as routing;
ordinary tensor indexing remains numerical. No edges or physics were removed.
The classification cache now includes a schema version, so saved graph replays
do not reuse classifications computed by an older compiler. The exact saved
specialized graph fails before these repairs and passes actual control
replanning afterwards (`build/validator_control_replan_v2_20260914.log`).
The combined regression group now passes 86 and has the same 14 baseline
failures (`build/validator_projection_regressions_20260914.log`).

A fresh source build with both compiler repairs is running:

```powershell
python -u ../speaktome/AGENTS/tools/replay_turing_resolved_deployment.py --repo . --fresh-validator-build --output build/validator_simulation_native_v3_20260914 --lanes 8
```

Log: `build/validator_simulation_native_v3_20260914.log`. This helper calls the
real `build_simulation` and only observes failures / saves the pre-frame-link
checkpoint; it does not alter compiler semantics. Native execution remains
unverified pending SSA emission, C compilation, and the runtime comparisons.

The fresh v3 build passed instantiation, call topology, and complete control
planning with both repairs. It lowered the individual regions, including
`run_superstep`, `pi_update`, `copy_shallow`, and `restore`, and is assembling
the complete repository SSA. The helper saved the real inputs to
`_class_surface_ssa_program` as `pre-frame-link.pkl` in the v3 directory.
Both scheduling fixes are now verified through their original failure stages
in a fresh source compile; this is still not a completed native artifact.
