# Continuation: one Python-defined dually through the complete validator, native compilation, and the in-world rig

Date: 2026-09-02

## Objective

Finish one end-to-end validator session for the commercial dually rear axle. The session must prove that these are cooperating modules rather than one bespoke demonstration:

1. the four-tire pneumatic simulation;
2. the dually vehicle/mechanical graph;
3. the validator's pillars, rollers, structural grasp, assembly sequence, and qualification stages;
4. the standalone validator host;
5. an in-world validator-rig host.

The domain behavior must be authored in Python and use `AbstractTensor` where tensor behavior is needed. The native executable must be produced by the repository's Python-source-to-repository-SSA-to-C path. Python execution and native execution must consume the same authored program and the same assembly definition. A host may add presentation, input, persistence, and scheduling adapters; it may not contain a second tire simulation, a second assembly sequence, or native-only vehicle logic.

The proof vehicle is one loaded vehicle artifact: a full-floating commercial rear axle with a structural axle casing, differential, generic rotational torque input at the pinion, two shared dual-wheel hubs, four tubeless wheel/tire assemblies, and hydraulic truck brakes. It intentionally has no suspension, steering, engine, transmission, or body. Those physics systems remain available to other vehicle definitions; they are not deleted or disabled in the validator.

## Current truth

The following repository pieces already exist and should be completed or connected, not replaced:

- `src/compiler/abstract_ui_dually_axle.py` defines `roadside_dually_axle_assembly()`. It includes the structural casing, differential detail, torque port, bearings, shafts, hubs, brakes, four wheels, two dual-wheel groups, tubeless cheap-retread pneumatic configuration, conventional rim valves, mechanical graph, validator stages, and world geometry.
- `src/compiler/mechanical_ports.py` defines bearing nodes, bearing race edges, rotating hubs, and generic rotational torque ports. A bearing is deliberately one node that may have both structural and drivetrain incident edges.
- `src/compiler/vehicle_native_assembly.py` contains structural-grasp and wheel-fixture negotiation used by the dually profile.
- `src/compiler/vehicle_validator_profiles.py` defines `dually_validator_profile()`, which loads the dually artifact, negotiates its fixtures, infers its structural grasp, and produces the graph constants and tire dimensions used by the canonical program.
- `src/compiler/vehicle_native_graph_program.py` contains `VEHICLE_NATIVE_GRAPH_VECTOR_SOURCE`; its public authored entry is `vehicle_graph_tick_vector`. `BATCH_CAPACITY` is 8. The program includes the vehicle graph and calls the balloon-tire program through the tire recurrence.
- `src/compiler/vehicle_balloon_tire_program.py` contains `BALLOON_TIRE_VECTOR_SOURCE`; its authored entry is `balloon_tire_vector_step`. The program supports `tubeless` and `tube` configurations and per-face material data.
- `src/compiler/vehicle_python_compilation.py` connects the authored Python, linked process graphs, extraction contract, repository SSA lowering, and C/Wasm emitters. It also contains the managed tire window that uses the repository `run_superstep` DT system.
- `tools/run_vehicle_native_assembly.py` has provided the visible Python dually reference run, including the real dually graph, negotiated pillars, actual tire mesh coordinates, pressure evolution, and DT telemetry. It is a prototype host around shared code, not permission to maintain a separate product.
- `tools/build_vehicle_validator_native.py` is the checked-in native bundle builder. It emits compiler-owned C sections, links the pool and DLL, builds the scientific viewer, copies shaders, and writes receipts.
- `src/compiler/abstract_ui_validator_rig.py` already defines `ValidatorRigAssembly`, `validator_rig_assembly()`, and `validator_rig_geometry_boxes()`. It gives the validator a stable world identity, material hopper, construction stage, tick ownership, qualification and release concepts, persistence, and world geometry.
- `src/compiler/abstract_ui_div_map.py` already publishes the dually and validator-rig descriptive objects into the AbstractUI world.

The important missing piece is that `ValidatorRigAssembly` is presently a descriptive world model. It does not yet own and advance the canonical validator program state. The next product feature is therefore not another renderer or tire harness: it is the runtime validator game object described below.

## The exact canonical source-to-native path

There must be one authored computation path:

```text
roadside_dually_axle_assembly()
    src/compiler/abstract_ui_dually_axle.py
        |
        v
dually_validator_profile()
    src/compiler/vehicle_validator_profiles.py
        |
        v
dually_vehicle_python_compilation_inputs()
    src/compiler/vehicle_python_compilation.py
        |
        +-- VEHICLE_NATIVE_GRAPH_VECTOR_SOURCE / vehicle_graph_tick_vector
        |      src/compiler/vehicle_native_graph_program.py
        |
        +-- BALLOON_TIRE_VECTOR_SOURCE / balloon_tire_vector_step
        |      src/compiler/vehicle_balloon_tire_program.py
        |
        +-- linked symbolic/process graphs
        |
        v
vehicle_python_extraction_contract()
    extraction_contracts/program_extraction.yaml
    extraction_contracts/vehicle_full_native_execution.yaml
        |
        v
lower_vehicle_python_graph_ssa()
    src.compiler.fortran_c_shell.lower_ast_source_to_ssa()
        |
        v
repository SSA
        |
        v
emit_vehicle_python_graph_c() / native module assembly
        |
        v
write_native_vehicle_kernels()
    src/compiler/vehicle_native_deployment.py
        |
        v
tools/build_vehicle_validator_native.py
        |
        v
vehicle_game_kernels.dll + vehicle_scientific_viewer.exe + shaders + manifest
```

`compile_vehicle_python_graph_aot()` still names the deprecated/legacy `compile_ast_aot` route. It is not the acceptance path for this work. Do not redirect the dually through it and do not write handwritten C to stand in for the Python program.

The Python reference host must build `dually_validator_profile()` and execute the same `vehicle_graph_tick_vector`/`balloon_tire_vector_step` source and the repository DT system. The native host must lower those functions through `lower_ast_source_to_ssa`. This is one source program with two execution hosts, not “Python logic” and “compiled logic.”

## Phase A status after the 2026-09-02 session (read this first)

Phase A steps 1-4 are done and step 5 has moved past the plan stage. Every item
below was proved by a ~1 minute synthetic program before the full lowering was
run; the programs are `tests/test_aggregate_call_identity.py` (5 tests, all
passing) and the full-lowering evidence is in `TEST_BASELINE_AND_HAZARDS.md`.

Root cause of `aggregate call binding for 'tire_history' ... 4 != 0`:
`_fold_callsite_structural_values` evaluated `tire_history = (hub, basis, angle,
plane)` over ProgramABI span parameters to a tuple of value FACTS and replaced
the Tuple with a Constant, severing the four leaves; the now-unconsumed span
Inputs were dropped as unused parameters, so every later ledger named dead ids
(this is also why `%106`/`%120` looked like rank `()`).  The fix keeps any
fact-bearing aggregate as dataflow (`_contains_program_abi_fact`).  The
"unjustified heuristic removal" flagged above turned out to be a MOVE of
`_propagate_callsite_tensor_specializations` after the structural fold, not a
removal; nothing was reverted.

Rules that landed behind that fix (all in the dirty working tree, uncommitted):

1. `_dependency_order` cache fingerprint includes the node identity set
   (two removed constants plus two edge-less member inputs collided).
2. A callee aggregate formal materializes one member Input per leaf
   (`aggregate_parent_binding`, `aggregate_index`), every authored projection
   `name[i]` is aliased onto its member (`_alias_projection_to_member`), and the
   formal becomes a Tuple over its members so `f(*constants)` keeps them alive.
   `_build_shell_hierarchy_plan` binds members by exact index even after the
   shapeless formal is dropped; the whole-aggregate node is never a physical
   argument.  Caller-side `history[k]` over an authored tuple aliases to leaf k.
3. PlanCall result bindings correlate by structural PATH; nested tuple returns
   publish their leaves as Ret operands in path order
   (`aggregate_return_layouts`), and the linker expands callee outputs with
   that layout so `next_tire_history[k]` becomes a typed call output.
4. Name-history repair excludes definitions reachable from the call: the
   unpack targets of `tire_state, tire_output = balloon_tire_vector_step(...)`
   sit left of the call and used to feed the call its own result.  That was
   the "carried update value 251 has no producer" shortfall; it is gone.
5. A multi-result `__plan_callsite_N__` marker publishes EVERY bound result
   through aggregate GEP/Load placeholders (`plan_callsite_marker_projection`);
   `replace_at_callsite_marker` rebinds the linked call onto them.
6. Value ids are never recycled: `src/compiler/process_graph_value_ids.py`
   watermark, used by the planner, `loop_composer`, and `topological_reducer`
   (reset at canonical renumbering).  Recycled ids had bound `rest` to a
   LoopResult port.
7. Fold node removal filters the identity table; a removed formal left in a
   name's history was seeded as an untyped argument and reported as
   `missing_caller_value`.  Port `value_source_id` follows every alias.
8. `tensor_data_descriptors` (where/stack/cat fold) is strict: one undescribed
   operand means an undescribed result; scalar literals are rank 0.  Before
   this, a mask-only broadcast produced `(8, 4, 1, ...)` where `800` belonged.

Full dually lowering (`lower_vehicle_python_graph_ssa(inputs=
dually_vehicle_python_compilation_inputs())`, ~7 min): plan and control
deployment complete; repository-SSA emission stopped on two shortfalls inside
`balloon_tire_vector_step__specialized...`: `planned_region_12::where` with
operands `((8,4,1,1,1), (8,1,1,1,2), (8,4,800,2,3))` and
`planned_region_13::add` with `((8,4,1,2), (8,4,1,3))`.  The probe
(`scratch: probe_balloon_shapes.py`) shows the `stack` building
`cylinder_normal` receives `radial[..., 0]/len` and `radial[..., 1]/len` as
`(8,4,1,2)` while `radial_length*0` is `(8,4,800,2)`: the vertex axis was
collapsed to 1 upstream in region feeds (`contact_position` `(8,4,1,2,3)`).
Rule 8 was applied in response and the full lowering was rerun: both balloon
shape shortfalls are gone and every planned region reaches repository SSA.
The run now stops one stage later, in frame linking:
`call has fewer operands than the callee signature while pruning
'vehicle_tire_recurrence__specialized...': operands=85 formals=501`.  The
probe shows 402 of the 501 formals are `linked_call_frame_storage` slots
propagated outward from the `balloon_tire_vector_step` callsite (463 of 501
are rank-0 float64 scalars; 13 carry `unbound_variant_source_id` /
`variant_column=row`).  The balloon step's loop-body scalars are being
treated as physical frame storage and pushed into every enclosing signature,
and the tick->recurrence call was linked before that growth.  This is the
activation-storage contract family already in the hazards ledger, not the
aggregate family: the `tire_history` members arrive as proper `(8,4,3)`,
`(8,4,3,3)` formals.  The next Phase A action is therefore: make loop-body
scalars of a specialized callee locals of that callee (or a declared
activation record), so its signature is its authored ABI plus proven storage
only; then the pruning arity check no longer sees 501.

Known pre-existing ABI leak, NOT caused by the aggregate work (identical in a
fully positional program): the callee loop variable `step` and loop-body
scalars appear as `linked_call_frame_storage` root arguments with no dtype.
This is the activation-storage contract family already in the hazards ledger.

Test classification: the 31 failures in the planner/linking/loop batch all
reproduce on a pre-patch copy of `glsl_deployment_strategy.py` (swap-verified
for the 15 that plausibly touched the patched paths); none is a regression.

## Current compiler frontier: verified facts, not guesses

Direct canonical lowering of `balloon_tire_vector_step` has succeeded. Therefore the authored tire Python and the repository's basic tensor lowering are not, by themselves, the demonstrated failure.

The complete dually specialization has exposed loss of identity/contracts across calls, aggregates, and loops:

- In a direct good tire SSA, formal `%106` (`position.reshape((-1, 3))`) and `%120` (`velocity.reshape((-1, 3))`) have shape `(8, 4, 144, 3)`. In the specialized full dually path, the same identities were observed with rank `()`.
- `%211` is associated with `rest` passed through `*tire_constants`, another aggregate/call boundary.
- A separate failure reported loop-carried update value `251` with no producer inside the loop body.
- The latest full dually lowering reached “instantiating complete control/operator deployment” and then failed in `_build_shell_hierarchy_plan()` with: `aggregate call binding for 'tire_history' has different caller/callee arity: 4 != 0`.

The immediate compiler task is to prove where the `tire_history` callee aggregate lost its four members and repair the identity rule at that exact construction boundary. The repair must satisfy these invariants:

1. Structural tensor arguments authored in Python, such as `dim` in reshape/reduction calls, become part of operator identity at AST ingestion. They are not reconstructed later from whatever shapes survived.
2. Every aggregate formal member has a scoped child identity. Caller-to-callee binding maps exact child identities and preserves the member contract. It does not reuse a caller-local integer ID, match by incidental name, or infer membership from shape.
3. A `LoopResult` preserves the exact source value identity, type, shape, and producer relationship. Loop composition does not manufacture a replacement formal or detach a carried update from its producer.
4. Specialization may refine a contract but may not turn a known tensor or four-member aggregate into an untyped scalar/empty aggregate.
5. Linkage is deterministic and scope-aware. There is no after-the-fact cleaning, re-derivation, or heuristic matching to make a malformed identity graph appear valid.

The focused test `tests/test_vehicle_python_graph_source.py::test_stack_region_keeps_the_authored_list_element_producers` passed during the investigation. That is useful but is not sufficient proof of full call/aggregate/loop identity preservation.

### Required audit before claiming a compiler fix

The working tree is large and contains changes from several efforts. Recent edits in at least these files are not yet proven by a complete dually lowering and native run:

- `src/transmogrifier/graph/graph_express2.py`
- `src/compiler/glsl_deployment_strategy.py`
- `src/compiler/loop_composer.py`

Literal tensor keyword capture at ingestion, source fact propagation into loop ports, aggregate-member input materialization, and exact `PlanCall` binding are plausible root-level work only if their producer identities and tests demonstrate the rule. An existing cleanup heuristic was also removed during investigation merely because heuristic behavior was under suspicion. That removal was not independently justified and must be audited and reverted unless a focused failure/proof shows it violates the identity contract.

Do not wholesale reset this dirty working tree; it contains user work and cross-cutting compiler work. Review the relevant hunks, identify which change belongs to which failing identity, preserve unrelated behavior, and add narrow regression tests before another expensive full compile.

## Runtime architecture to finish

The final system needs four explicit layers.

### 1. Python domain definitions

These are serializable, host-independent objects and graphs:

- `VehicleAssemblyDefinition`: the loaded artifact, mechanical graph, structural regions, component identities, installation relationships, and requested qualification capabilities. The existing dually mapping is the first complete instance.
- `WheelGroup`: one or more wheels at a shared axle station, including whether they share a hub, are locked hub-to-hub, require independent articulation, and can use one long roller pair or separate rollers.
- `PneumaticAssembly`: distinct hub, bearing, wheel, rim, bead, casing/sidewall/tread, optional tube, and pneumatic-port identities; material regions and thermal/oriented shell data belong here.
- `RigNegotiation`: the result of structural-grasp selection, pillar allocation, roller configuration, eventual installation transforms, and stage capabilities.

These names may become dataclasses or protocols, but the content should extend the existing models rather than duplicating them in a new hierarchy.

### 2. Canonical validator program

`ValidatorProgramState` should own the stage identity, custody of each part, negotiated fixtures, vehicle/tire state tensors, DT controller state, qualification results, and failure report. It advances through one canonical `tick(dt envelope)` entry used by every host.

`ValidatorCorePorts` should be limited to domain necessities: load/read an assembly definition, obtain material/components, command existing rig actuators, publish actual graph/mesh state, record qualification results, and transfer custody of the same built object. The program must not know whether its caller is a standalone window or the game world.

### 3. Standalone validator host

The standalone CLI/window supplies a clock, controls, native/Python execution adapter, telemetry presentation, and actual graph/mesh rendering. It owns no assembly rules. It is the laboratory proof host.

### 4. In-world validator object

Extend `ValidatorRigAssembly` with a runtime companion that owns a `ValidatorProgramState` and invokes the same canonical tick. `ValidatorGamePorts` add only game mechanics:

- world-assigned `tick`, `dt`, `subdt`, and substep envelope;
- player/operator controls and inspection UI;
- projectile/material intake and inventory ledger;
- persistence/save/load of rig state and the stable vehicle/part identities;
- spawn, custody, installation, qualification, and release hooks;
- damage, breakage, repair, and unusable-state hooks;
- audio, VFX, animation, and renderer publication;
- network/authority hooks if the game world later requires them.

Those ports drive or observe `ValidatorProgramState`. They do not fork the physics, reimplement the stages, or create a visual copy of a vehicle. The object being assembled, qualified, released, damaged, and saved is one stable graph identity.

## Canonical full-session sequence for the dually

The dually profile's current stage list should be reconciled with this host-independent state machine:

0. **Load and validate the artifact graph.** Load the axle definition, its four wheel assemblies, two shared hubs, casing, differential, pinion torque interface, bearings, shafts, and brakes. Resolve required capabilities and reject missing graph contracts before motion.
1. **Negotiate the unloaded grasp and fixtures.** Select four noninteracting grasp points on the strongest accessible square-like structural region. For this axle, the points are front/back positions around the left and right structural axle-tube saddle regions, analogous to U-bolt/accessory mounts. Allocate one gravity-parallel pillar per wheel because four wheel assemblies are being prepared, while recognizing two shared-hub wheel groups. Select long or separate roller pairs from articulation and clearance requirements, not from a hard-coded wheel count.
2. **Stage hub/bearing/wheel/rim/bead components.** Show and track each distinct component identity. Place every rim/wheel on the pillar hub at its eventual installation transform with rollers down/available. Do not collapse “hub,” “bearing,” “wheel,” and “rim” into one object.
3. **Mount and qualify each tubeless casing.** Use the existing pillar mechanics: articulate the rollers vertically for mounting, pinch the casing to the rim seat, use the cosmetic slow-turn handover only as a rendering of the real stage transition, inflate through the conventional rotating rim valve, seat the non-bolted beads, check sealing, set nominal pressure, and balance. The roller returns to its dyno orientation afterward. The cosmetic motion never substitutes for bead/contact physics.
4. **Install the two dual-wheel groups.** Transfer the four prepared wheel assemblies from pillar custody to the left and right shared hubs without changing their identities. Inner/outer wheels retain separate rim, bead, casing, and valve identities even though the pair shares a hub station.
5. **Complete axle and brake assembly.** Clamp the casing at the inferred structural grasp, install/verify differential, shafts, bearings, hubs, and hydraulic brakes according to graph dependencies. The open pinion/driveshaft flange is the generic torque-input attachment used by the validator.
6. **Qualify the complete object.** Exercise differential torque transfer, bearing constraints, wheel-group rotation, hydraulic braking, tire pressure/sealing, balance, roller dyno loads, structural load paths, DT completion, and breakage reporting. Draw the actual mechanical graph and physical mesh coordinates while it runs.
7. **Release.** Publish the same qualified axle object to the standalone result slot or game world. Never synthesize a replacement render object at release.

## Tire work that the compiler detour displaced

The dually is configured as a tubeless, non-bolted-bead, inexpensive commercial retread with one conventional rotating rim valve per wheel. It has no bearing-fed pneumatic port. This must remain the first acceptance configuration.

The tire still needs these substantive improvements after the canonical full path runs:

- Preserve an invariant shell center surface so material thickness, inside/outside classification, collision, and rendering do not redefine the mechanical reference surface.
- Make the natural/rest casing geometry express a real sidewall-to-shoulder-to-flat-tread form. A rounded torus at approximately 101 kPa is not the target.
- Store a natural-position UV/material field and rest Jacobian per face. Compare the current local Jacobian to the authored resting Jacobian in the shell solver.
- Support directional/oriented coefficients: circumferential, meridional/sidewall, shear, and bending response must be independently controllable. Refine the mesh where a material boundary or direction field cannot be represented accurately on the current triangles.
- Represent casing layers/regions rather than one homogeneous membrane: inner liner, sidewall rubber, tread/retread cap, carcass cords, oriented belts, steel and/or composite layers, bead region, and bead wire/reinforcement.
- Carry thermal properties and heat production/transfer for tire skin, cords/belts, beads, and tubes. Tube mode and tubeless casing mode have different interfaces and heat paths.
- Publish triangle visualization data from the actual mesh: interior versus exterior orientation and material/layer identity. Colors are diagnostics for real face data, not painted substitute geometry.
- Keep the conventional rim valve as a wheel-attached rotating mass so balance can account for it and remove a small amount of wheel/hub material as appropriate.

Later configurations must fit the same object model:

- tube mode, where the tube valve synchronizes with/becomes the installed rim valve;
- rim-interior pneumatic passages connected by material-graph edges through a hub/bearing rotary union;
- tractor wheels that expose both the outer conventional valve and a bearing-fed pneumatic path;
- bolted-bead wheels, which skip the cosmetic bead-mount handover where inappropriate.

## General machine requirements retained by the dually proof

- Wheel handling is N-wheel/M-axle. The dually has four tire instances and two wheel groups on one solid axle; those numbers are data, not program structure.
- The program remains batch-vectorized with capacity and active tensor extent 8. One validator may currently load one artifact, but no scalar execution lane becomes the default and no dually-specific batch ownership is embedded in the coordinator.
- A lone wheel receives a lone pillar. Ordinary independent wheel pairs receive corresponding pillars. Shared-hub groups negotiate roller width/count and articulation. A solid axle may use a cross-axle roller when articulation is irrelevant. Tracked systems use rollers before track installation and may use ground projection after track installation.
- Structural support discovery replaces the old car-specific “four corners.” It selects four strong, accessible, square-like grasp locations or uses authored structural mount regions. The four articulated graspers do not collide with each other and should minimize burden on the rest of the structure; a future solver may keep one grasper searching while the other three hold.
- Bearings remain shared force-transfer nodes between structural and drivetrain graphs. Their contracts include dimensions, freedoms/ranges, stiffness/damping, friction/heat, load limits, breakage, race ownership, and damage distribution. Bushings and other connectors should use the same transformer-style approach.
- The axle casing remains the structural pipe/housing and lubricant boundary. Its front/back attachment regions make it useful for later spring bars, U-bolts, torque arms, tow/accessory mounts, and N-axle suspension platforms without pretending this axle already has suspension.

## Proof and acceptance matrix

For the same serialized dually profile and deterministic initial state, capture the following from both the Python live host and the native bundle:

| Evidence | Python host | Native host | Acceptance |
|---|---|---|---|
| Loaded graph | Object and edge identities | Same identities/manifest receipt | No missing or invented parts |
| Tensor ABI | Shapes, dtypes, batch extent, active count | Emitted ABI/storage receipt | Exact structural match |
| Stage machine | Stage ID, part custody, fixture state | Same sequence and transitions | No host-owned stages |
| DT system | Candidate DT including rejected candidates, accepted advance, rule/error channel, rollback | Same fields | Same accept/reject decisions within numeric tolerance; no retry cap or forced acceptance |
| Tire state | Pressure, volume, bead contact/seal, positions, velocities, material regions | Same | Bounded numerical difference declared in the test |
| Vehicle state | graph forces, bearing/shaft/hub/brake/torque state | Same | Bounded numerical difference declared in the test |
| Rig state | grasp points, pillar transforms, roller articulation, actuator loads | Same | Same negotiation and custody |
| Rendering | actual graph nodes/edges and physical mesh vertices/faces | actual graph nodes/edges and physical mesh vertices/faces | No illustrative replacement geometry |
| Completion | qualification report and released object identity | Same | Same object identity from load/build through release |

An accepted frame must remain visible until replaced by the next actual state. Every attempted DT subdivision, including rejected attempts, must be observable live with the violating rule/error channel and useful error-matrix statistics. Telemetry is observation of the canonical run, never a second scheduler.

## Execution plan

### Phase A — establish a trustworthy compiler baseline

1. Audit the recent identity-related hunks and revert the unjustified heuristic removal unless independently proven.
2. Trace the exact construction of the empty callee aggregate for `tire_history`; add a regression that begins with the authored four-member aggregate, crosses the same call/specialization boundary, and asserts four exact bound child identities and contracts.
3. Add focused tests for `%106`/`%120` reshape identity/shape preservation and loop-carried value `251` producer preservation. Tests should assert provenance, not merely final inferred shapes.
4. Re-run direct balloon lowering to protect the already-working leaf path.
5. Run full `lower_vehicle_python_graph_ssa(inputs=dually_vehicle_python_compilation_inputs())`. Do not start native compilation until repository SSA completes and validates.

### Phase B — complete the standalone modular dually validator

1. Move remaining stage ownership out of the prototype host and into the shared canonical validator state machine without creating another entrypoint.
2. Connect the existing dually profile, rig negotiation, tire state, vehicle state, and qualification state through explicit core ports.
3. Make the Python live host consume this state machine and retain live actual-mesh/graph rendering and DT telemetry.
4. Run the entire sequence through qualification and release, not merely tire inflation or one rendered frame.

### Phase C — native proof

1. Lower the exact Phase B Python source through the canonical repository SSA path.
2. Build at `O0` first to reduce compilation cost while preserving the same emitted program and pool/dispatch structure.
3. Run the native scientific viewer through the full sequence and compare snapshots/telemetry against the Python reference.
4. Only after parity succeeds, build the deploy optimization and record compiler/dispatch receipts.

### Phase D — make the existing rig a live game object

1. Add the runtime companion to `ValidatorRigAssembly`; do not replace its existing world identities or geometry.
2. Attach `ValidatorGamePorts` to the world's assigned tick, material projectile/hopper ledger, player controls, persistence, damage, audio/VFX, and release registry.
3. Load the same dually assembly definition and invoke the same validator tick used standalone.
4. Prove that pause/focus/window behavior changes presentation/input only; simulation progress is determined by the assigned tick envelope.
5. Save and restore mid-stage while preserving every part, graph, controller, and custody identity.

### Phase E — generalize from the proved case

Add fixtures and tests for 0 wheels, one spare/lone wheel, independent pairs, multiple axles, mixed wheel groups, bearing-fed pneumatics, tube mode, bolted beads, and pre/post-track handling. Generalization must be expressed as new assembly data and negotiation outcomes, not branches that name specific vehicles.

## Commands once Phase A lowering is green

From `C:\dev\Powershell\turing`, run the visible Python reference using the existing bundle directory for assets/manifest:

```powershell
python tools/run_vehicle_native_assembly.py build/vehicle_validator_dually_o0 --assembly-profile dually-axle --python-material --python-viewer
```

The current prototype host still requires `--python-material` and either `--python-viewer` or `--headless-frame` for the dually. The completed architecture should remove those product-specific restrictions by giving both hosts the shared runtime interface; it should not add an alternate validator.

Build the canonical native dually at `O0`:

```powershell
python tools/build_vehicle_validator_native.py --output build/vehicle_validator_dually_o0 --assembly-profile dually-axle --contract deploy --optimization O0
```

The current viewer opens shader names relative to its working directory. Until the launcher resolves assets relative to the executable/bundle path, run it from the bundle directory:

```powershell
Push-Location build/vehicle_validator_dually_o0
.\vehicle_scientific_viewer.exe
Pop-Location
```

The permanent fix is for the generated viewer/launcher to resolve its shaders and manifest from the executable's bundle directory, so launching the absolute executable from any current directory works. This is packaging/host behavior, not tire or validator logic.

## Do not repeat these detours

- Do not make a tire-only “one-wheel validator.” The proof artifact is one dually vehicle with four tire instances.
- Do not write a pretend validator, replacement pillar, illustrative shape renderer, one-frame native sample, or handwritten native tire loop.
- Do not maintain separate Python-for-running and Python-for-compiling domain logic. Clearly isolated host-only adapters are acceptable; duplicated physics and stage logic are not.
- Do not use NumPy inside authored tensor computation. Host-side feed construction may allocate arrays; the authored program uses `AbstractTensor` operations that the repository compiler understands.
- Do not use a scalar lane by default or hard-code one graph lane to one four-wheel simulation.
- Do not load the default car and hide/overlay it with dually visuals. Load the dually assembly profile as the vehicle artifact.
- Do not remove suspension, steering, engine, or transmission physics cores merely because this particular artifact does not instantiate them.
- Do not add a retry cap, minimum DT, or forced acceptance to make the validator appear to advance. Use the repository `run_superstep` semantics. The current managed configuration is `allow_increase_mid_round=True`, `allow_unresolved=False`, `max_retries=None`, `rollback_threshold_multiplier=2.0`, and `dt_min=None`.
- Do not reconstruct identities, aggregates, or shapes after specialization with names, shape guesses, selected addresses, or cleanup heuristics. Fix the rule at the identity's source/binding boundary.
- Do not start repeated expensive native builds while repository SSA still fails.
- Do not claim parity from screenshots. Rendering is a view of actual state; parity comes from graph, ABI, state, DT, and qualification evidence.

## Definition of done

This continuation is complete when one dually artifact can be selected by the normal validator CLI, visibly proceeds from graph load through four tire preparations, shared-hub installation, complete axle/brake/torque qualification, and release, and does so in both Python and native execution from the same authored source and assembly definition. The native binary is produced by `tools/build_vehicle_validator_native.py` through `lower_vehicle_python_graph_ssa`, not a handwritten or deprecated compilation path. The standalone host and in-world rig call the same validator state machine. The in-world rig adds only game ports and preserves the same object identities. Python/native evidence demonstrates matching graph structure, tensor ABI, stage/custody transitions, DT decisions, tire/vehicle/rig state, and qualification result within declared numerical tolerances.

The next implementation action is Phase A, step 2: repair and prove the exact aggregate-identity construction that leaves the specialized `tire_history` callee with zero members. Once full dually repository SSA is green, return immediately to the full standalone validator sequence, then wrap that proven program in the existing in-world `ValidatorRigAssembly`.
