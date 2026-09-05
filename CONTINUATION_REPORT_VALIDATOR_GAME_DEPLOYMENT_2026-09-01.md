# Validator, game, and integrated-game continuation report

Date: 2026-09-01  
Repository: `C:\dev\Powershell\turing`  
Scope: compiler dispatcher/deployment repair and product-status handoff

## Executive status

No release product in this report is yet both current and fully qualified.
There are useful runnable artifacts and substantial source-level integration,
but they occupy three distinct product boundaries and must not be conflated.

| Product | Web status | Native status |
|---|---|---|
| Validator — settles and assembles the rig | Runnable O0 four-tire managed-step page exists at `build/vehicle_web_validator_o0/index.html`. It runs a 4.3 MB exact C-to-Wasm module in a worker and reports finite/change summaries, but it is a one-step tire validator page, not the complete 21-stage assembly machine, and it has no substep telemetry. | Runnable O0 scientific viewer, batch viewer, DLL, shaders, manifest, and assembly checkpoints exist at `build/vehicle_validator_managed_tire_compiler_fixed_o0`. The newest bundle has checkpoints only through stage 3. Older reports reach 18/19 and 19/21 passing stages, but every complete report has `pass: false`; stage-19 quiescence and clean release remain unresolved. The current viewer is not an optimized release and does not yet display managed-substep statistics. |
| Game — Living Data Map with the self-referential map | Source and an older generated page exist. `docs/generated/abstract_ui_object_map.html` is the last checked-in/generated surface, dated before the latest validator compiler work. `build/mechanical_creature_game_current` is currently empty, so there is no fresh current web build to hand to a player. | No native build of the self-referential Living Data Map was found. The SDL/OpenGL scientific vehicle viewer is a validator shell, not a native realization of the recursive map/game. |
| Upgraded game — game plus validator as a participating producer/consumer object | The neutral model is implemented: `abstract_ui_validator_rig.py` gives the rig stable world identity, material-ball consumption, assigned-tick participation, assembly/qualification state, and release of the same built vehicle object into the existing vehicle slot. `abstract_ui_div_map.py` includes it in the game model. This integration is source/test-level; the current game build directory is empty and the available generated page predates the latest native validator work. | No integrated native upgraded-game artifact exists. The native validator and viewer run separately; they do not yet inhabit a native self-referential world, consume in-world material projectiles, or release a newly qualified live car into that world. |

## Explicit non-change failure: native viewer dispatch

The current native validator retains the original visible failure and must not
be credited as a dispatcher improvement:

- Space only toggles the `simulate` flag on the viewer's sole UI/render thread.
- That same thread synchronously executes every `vehicle_native_graph_tick`
  in `STEPS_PER_FRAME` before geometry upload, event polling, or
  `SDL_GL_SwapWindow` can happen again.
- The viewer loads vertex and fragment GLSL only. It has no compute shader,
  `glDispatchCompute`, compute barrier, or GPU physics/readback path.
- `repository_ssa_dispatch.py` and its host pool are not connected to this
  viewer executable. The expanded program therefore provides no actual
  multi-core work to the running validator, regardless of what a standalone
  deployment manifest can describe.
- If a managed tick stalls, the pause key cannot take effect until that tick
  returns. No completed physics snapshot, geometry refresh, HUD update, or
  display swap/readback follows while it is blocked.

This is a **non-change failure from the previous product**: compiler planning
and ABI repairs have not yet changed the runtime dispatch behavior of the
runnable native viewer. A release claim requires the viewer to submit compiled
frame work to a real CPU/GPU deployment worker, keep event/render ownership
responsive, and publish only complete joined snapshots back to the display.

## Concrete artifact evidence

The current native validator directory contains:

- `vehicle_scientific_viewer.exe` and `vehicle_scientific_viewer_batch.exe`;
- `vehicle_game_kernels.dll`, PDB, GLSL vertex/fragment shaders, and a native
  manifest;
- compiler-emitted vehicle, wheel-contact, graph-tick, leveling, fixture,
  material, and balloon-tire C sections;
- native assembly checkpoints for `clamp-pan`, `engine-pan`, and `engine`.

Historical assembly reports remain valuable but are not release receipts:

- `vehicle_native_validator_20260829_final`: 18 of 19 stages pass; overall
  false at differential-wrench/quiescence.
- `vehicle_native_validator_20260829_resume2`: 19 of 21 stages pass; overall
  false, with the release path contaminated by replay/destructive-pull state.
- `vehicle_game_validator_120x3`: one attempted suspension-load-transfer stage,
  non-finite and overall false. It does not prove a working game schedule.

The web validator page explicitly labels itself O0 and warns that its exact
managed step is slow. It reports elapsed time, finite state/output counts,
changed state, maximum magnitude, and four wheel summaries. It does not run the
full assembly sequence or expose the dt controller's requested/accepted
substeps, retry depth, limiting metric, minimum accepted dt, or rejection
counts.

## Compiler and dispatcher work completed in this session

No authored vehicle Python was modified.

1. Control partitioning no longer correlates locationless AST helper nodes
   such as singleton `ast.Load` objects across unrelated branches. Structural
   conditional arms own only their predicate terminal region, leaving
   transitive ancestors in the flat dependency schedule. This cleared the
   overlapping maximal-control-block failure in `run_superstep`.
2. `repository_ssa_dispatch.py` now plans deployment from the repository SSA
   `deployment_table`, proves lane independence through deployment dataflow,
   follows complete internal function closures, selects the real host pool,
   executes prepared lanes with frame-join semantics, and compiles LLVM closure
   roots at O3 with a deployment manifest.
3. Work-contract `deploy` and `fast` presets now request `deployment=auto`;
   `develop` remains serial. This makes deployment policy one compiler-level
   source rather than a validator-specific switch.
4. Source-call linking now correlates ordered semantic outputs across caller
   and callee SSA namespaces, preserves repeated return identities, records
   stale-to-final aliases, and separates scalar semantic values from expanded
   record/tensor storage.
5. Aggregate-output legalization now removes a pass-through output only when a
   concrete projection exists, its uses are rebound, and no independent direct
   consumer still requires that call result.
6. The latest identified ABI defect was that a mixed `(bool, record)` return
   advertised semantic ids `(200, 121)` while the actual callee `Ret` contained
   only record `%121`'s fourteen physical fields. The generic output authority
   now reconstructs the physical ABI in semantic order and synchronizes the
   actual `Ret`: scalar `%200`, followed by record fields.

Focused verification after that last repair passed 32 tests, including the
previously failing completed-record-call surface test and the new independent
scalar-consumer regression. A larger function-linking run had 72 passes and 7
failures before the last repair; several failures are established dirty-tree
baseline issues, while the completed-record-call failure is now fixed. A later
four-file product run was deliberately interrupted after nine observed passes
because vehicle projection was taking several minutes; it is not a completed
suite result.

## Diagnostics added to the translation trouble tree

`tools/TRANSLATION_DEBUGGING.md` now documents three opt-in receipts:

- `TURING_DEBUG_STRUCTURAL_OUTPUTS=1` — graph/named outputs, physical `Ret`,
  and record layouts;
- `TURING_DEBUG_LINKED_CALLS=1` — final linked calls and source projections;
- `TURING_DEBUG_AGGREGATE_CALLSITE=<id>` — one call's semantic bindings,
  expanded physical bindings, callee outputs, and pre/post legalization
  attributes.

These receipts distinguish four failures which previously looked like one
freeze: control-region overlap, unresolved call linkage, incomplete physical
return ABI, and legitimate adaptive-dt work. They also prevent an absent
measurement from being described as “infinite subdivision.”

## Recent troubleshooting path

The investigation began at the visible symptom: pressing Space in the native
scientific viewer appeared to freeze. The first compiler failure was not dt at
all; maximal control partitions overlapped because branch correlation included
locationless AST singleton nodes. Once partitioning was corrected, lowering
reached all thirteen `run_superstep` regions and exposed source-call ABI
problems.

The next layer was a multi-result managed call. Ordered semantic outputs and
record layouts lived in different SSA numbering domains, repeated physical
field ids were legal, and the linker initially could not correlate them. After
positional correlation and alias receipts removed all unresolved calls, the
full-native gate still found one undefined `ok` predicate.

At first the symptom appeared to implicate aggregate-view pruning. A focused
regression proved one real bug there: a formal-storage match is not sufficient
to delete an output when an independent direct consumer exists. The full
validator still failed, so opt-in pre/post receipts were added. They showed the
predicate was absent before legalization. A second receipt at result expansion
then showed the exact contradiction: semantic and physical bindings contained
the boolean, while `callee_outputs` contained only the expanded record. The
record-only `Ret` caused the linker to classify the boolean as non-return. The
current repair makes semantic output order the shared ABI authority.

This path matters because none of the evidence supports “metrics forced
infinite subdivision.” The compiler never reached a stable native managed
frame from which subdivision counts could be measured. The apparent freeze was
dominated by slow O0 compilation/execution and compiler ABI defects. Actual dt
behavior must be assessed only after the corrected managed artifact runs and
publishes its controller telemetry.

## Natural next work

1. Rerun the complete managed lowering gate after the semantic-`Ret` repair.
   Require zero unresolved calls and zero undefined operands, then cache the
   repository SSA product.
2. Compile the cached whole closure through the generic O3 LLVM deployment
   entry and inspect `deployment.json`. Confirm that independent lanes are
   real pool work, not merely manifest entries, and keep trivial internal
   closures as one dispatchable lane callback.
3. Connect the generic compiled-program ABI to the scientific viewer. Do not
   use the bespoke managed balloon renderer. The complete validator assembly,
   fixture, telemetry, and shaders must link against the same generated
   repository-SSA deployment.
4. Move physics execution off the SDL event/render thread. Space/pause must
   cancel or stop future frame submission immediately, while the last complete
   snapshot remains drawable. A joined worker result must be generation-tagged
   so a late frame cannot overwrite a newer pause/reset state.
5. Put substep telemetry on screen: target display dt, requested and accepted
   substeps, retry/rejection count, deepest subdivision, minimum accepted dt,
   limiting metric/channel, finite-state status, and wall time. Space must
   remain event-responsive while a frame is computed.
6. Run finite one-frame equivalence, then a bounded soak, then the full 21-stage
   qualification. An optimized artifact is not a release until the report is
   finite and overall true.
7. Rebuild the Living Data Map web game from the current source and run its
   browser checks. Only after that should the validator rig be described as a
   deployed participating web object.
8. For the upgraded native game, first define the native realization of the
   self-referential world/map. Then host two independent material frames—live
   car and rig-built car—under the same world tick, with projectile material
   consumption and identity-preserving release.

## Acceptance boundary for the requested optimized release

The requested release is not “an optimized balloon DLL.” It is a runnable
native validator product containing the complete assembly/qualification rig,
compiler-generated physics closure, real multi-core dispatch where proven,
GLSL presentation, responsive UI, visible dt/substep statistics, finite soak,
and a passing qualification report. LLVM is the primary optimized CPU backend;
C remains host ABI/fallback, Fortran is a reference/verifier lane, and GLSL is
the graphics/eligible GPU channel. Backend selection must come from the shared
deployment contract, not product-specific source logic.

## Prompt history

> “DONT MODIFY THE PYTHON SOURCE”

> “FIX THE COMPILER DISPATCHING”

> “The target backend should probably be fortran or llvm ... dial in the dispatcher work in the compiler and then start that llvm+deployment compile”

> “the deployer should be such that it can tolerate trivial internal closures”

> “an actually optimized release version of the entire validator rig not just the balloon physics”

> “update documentation, translation trouble tree (add any diagnostics you created), give me status on the web and the native versions of: the validator ... the game ... AND the upgraded game”

---

## Session 2 (2026-09-01, later): dispatch made real in SSA, C, and the shell

Direction fixed by the user this session: deployment demands are carried
through SSA (firm, in-place), the Python-runtime frame executor is removed,
shells keep their double-buffer discipline, and LLVM-pool versus GPU compute
are not mutually exclusive lanes.

### Landed and verified

1. **SSA lane outlining** (`src/compiler/deployment_outlining.py`): a proven
   single-lane `independent_iterations` region (retained-loop lane template)
   is outlined into a real module function; the parent loop body becomes one
   internal call, so the serial stream stays byte-equivalent. Honest gates
   with a closed refusal vocabulary (live-outs, mixed blocks, unproven
   shared stores, order-DEPENDENT sequence mutation). A shared append of a
   lane-INVARIANT value is accepted as a guarded critical block: every
   iteration appends an identical element, so only the count is observable
   and atomicity (the new `turing_pool_effect_lock`) preserves serial
   semantics.
2. **Native pooled emission** (`ssa_c_backend.py`): at the Deploy marker of
   an outlined region the C emitter marshals the outline's live-ins into a
   stack context, emits `turing_pool_deploy_span(...)` and jumps to the join
   on success; the serial loop remains in-text as the fallback. Guarded
   blocks emit effect lock/unlock. `CModuleArtifact.pool_required` makes
   `compile()` link `turing_pool.c`. Proof: `tests/test_deployment_outlining.py`
   — a compiled pooled loop matches the serial build bit-for-bit.
3. **The real vehicle region deploys in SSA and C.** The managed program's
   sole deployment region (`step_with_dt_control_used...` region 0, the
   8-lane managed batch loop) outlines (19 live-ins, one guarded append
   block), plans launchable+parallel, and emits a real
   `turing_pool_deploy_span` plus effect locks in the module C.
4. **Python runtime dispatcher removed.** `RepositorySSAFrameExecutor` is
   gone; planning stays pure, execution is native (`turing_pool.c`).
   The planner now treats outlined iteration regions as launchable and
   names the outlining pass when it has not run.
5. **Threaded scientific viewer** (`vehicle_scientific_viewer_shell.c.in`):
   physics runs on its own thread as the only caller of the compiled
   kernels; complete frames publish through a generation-tagged double
   buffer (the DisplayDoubleBufferABI front/back+generation discipline the
   shells already carry); UI edits queue as commands applied at window
   boundaries; Space toggles at the next boundary while the last complete
   snapshot stays drawable; HUD adds `DISPATCH POOL <workers> PHYS <ms>
   GEN <n>`. Compiled and smoke-run against the existing O0 bundle DLL as
   `vehicle_scientific_viewer_threaded.exe` (ABI verified against the
   bundle before linking).
6. **Checked-in build driver** (`tools/build_vehicle_validator_native.py`):
   the previously missing seam. Emits kernels under a named work contract
   (`deploy` = O3 + deployment=auto), compiles every section plus
   `turing_pool.c` into `vehicle_game_kernels.dll`, compiles the threaded
   viewer, and writes dispatch receipts into the manifest.
   `write_native_vehicle_kernels` now ships `turing_pool.c/h` in the bundle
   and `emit_balloon_tire_managed_python_c` runs the outlining pass when
   the active contract requests `deployment=auto`.

### The one remaining wall to the pooled O3 vehicle DLL

Managed-window emission is still incomplete — `dynamic temporary %t96 has
no native activation-storage contract` plus extent-origin shortfalls in
`balloon_tire_managed_window__planned_region_2/3`, identical at 1/120 and
1/1024 windows and identical with or without outlining (so not a
deployment regression). The six failing tests in
`tests/test_ssa_c_aggregate_constants.py` are the prior session's SPEC for
this repair (invocation-local frame storage, const aggregate constants,
static extent emission, physical-dtype preservation). Finish that contract
and `tools/build_vehicle_validator_native.py --contract deploy` produces
the optimized, pool-dispatched validator in one command.

### Product notes from the user this session

- 1024 Hz outer windows are fine; the managed dt substepper inside each
  window remains authoritative for the validator.
- In the eventual game the reaction system may be simplified into a token
  set of physical responses — validator keeps the full controller.
- GPU compute and the LLVM/C pool are complementary lanes; deployment
  should find reasons for both, preferring work that stays in the GPU
  domain. The GLSL compute artifact for eligible regions remains future
  work at the shader-region seam.

---

## Session 2 addendum: the activation-storage gate is OPEN

The managed vehicle program now emits COMPLETE C on the module lane, with
the pooled deploy inside. The fixes, in `ssa_c_backend.py`:

1. **Requirement-backed dynamic temporaries.** A tensor-table temporary
   whose runtime metadata stays dynamic but whose storage requirement is a
   static bound (`dynamic=False`, concrete element count) allocates that
   bound; the requirements solver is the arena authority, semantic shape
   keeps flowing through extents.
2. **A real span-origin walk for extents.** `resolve_span_origin` walks:
   callee formal -> caller actual, planned output -> caller destination
   (the previously missing edge), the linker's `ssa_call_result_from`
   identity hops, `cast_like`'s schema operand, elementwise/broadcast
   shape transfer, and a numpy-broadcast combine for `binary_double`
   (including scalar-identity against a public origin). Termination is
   either a PUBLIC root buffer (slot in the extents array) or a STATIC
   descriptor (compile-time constants). Local producers are analyzed
   before call edges — edge-first closed a two-node projection/identity
   loop on in-place outputs. Unresolvable origins are again refusals,
   never unfillable slots.
3. **Flat reduction spellings.** `max`/`min`/`all`/`any` over one tensor
   operand (the dt controller's max-wave-speed and all-finite metrics)
   emit scan loops sized by the descriptor's SEMANTIC element count (never
   the arena bound), NaN-propagating for max/min, NaN-truthy for all/any.
   The LLVM lane records the same case as a named shortfall instead of
   crashing in str.format.

An O3 build of the complete pooled module (turing_pool.c linked) exists;
the ~27-minute compile is the monolithic-translation-unit cost.

## Automatic compute-shader selection (receipts live)

`deployment_compute_selection.select_compute_lanes` judges every outlined
deployment lane against the desktop-GLSL compute dialect and rides the
verdicts on the module and the bundle manifest (`manifest["deployment"]`).
The vehicle's lane refuses today with four exact reasons (internal planned
region calls, one conditional, Gt outside the dialect, the effect-locked
append); a synthetic straight-line arithmetic lane judges ELIGIBLE — the
determination is the compiler's, not a product's. Widening the GPU
dialect and inlining planned regions is the named path to real
`glDispatchCompute` on the tire math.

## Session 2: the pooled native vehicle program RUNS

The complete managed vehicle module (pooled deploy inside, `turing_pool.c`
linked, `extent_order` empty — every extent proven compile-time constant)
compiles at O2/O3 (~26 min, monolithic-TU cost) and EXECUTES:

- one 1/120-s managed window: ~0.78 s wall at O2 (vs the previous
  never-ran state);
- six consecutive windows: tire state stays FULLY FINITE and bounded
  (|state| <= 0.55) — against the 2026-08-30 baseline of 12,289
  non-finite state values in a single frame, the tuple/ABI/storage chain
  is now numerically sound end to end;
- one precise residue: the 24 rim force/moment output fields per lane are
  persistent NaN (0/0 pattern, all 8 lanes, all 4 wheels, stable across
  windows, state unaffected). Anonymous-cell zero feeds were empirically
  ruled out (setting every anonymous scalar nonzero changes nothing).
  Most probable: an average-contact-wrench divide with zero contacts in
  the captured free-floating feed configuration. A serial-twin build (same
  module, no outlining) was launched for the bitwise pooled-vs-serial
  discriminator.

Run-harness contract still owed by the lowering: 14 anonymous captured
closure cells (dt-system lists/scalars) surface as unnamed public buffers;
`compile_balloon_tire_managed_python_native` correctly refuses them. The
zero-fill they get in the probe harness is right for empty-list arenas and
unproven for the scalars — publishing capture names/values for closure
cells is the named lowering task that closes it.
