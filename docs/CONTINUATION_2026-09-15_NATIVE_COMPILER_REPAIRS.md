# Native compiler repairs — 2026-09-15

## What was wrong

Five compiler defects were found by compiling authored source and comparing the
executed result with eager `AbstractTensor` execution. Four of them never
raised: each produced a complete, compiling program that silently read the
wrong elements, so only value comparison could expose them. The fifth is the
single undefined operand that has been stopping the validator build.

### 1. A reshape view occurrence lost its extents

`b.reshape((-1, 1, 2))` and `b` are one storage identity with two shaped views.
Two passes keyed by integer value id collapsed them onto the allocation owner's
shape:

- `propagate_repository_ssa_call_metadata` (`tensor_ssa_lowering.py`) restamped
  every occurrence of the id with the call-edge shape when acting
  authoritatively.
- `_intern_unshadowed_formal_uses` (`ssa_call_input_adapters.py`) replaced any
  operand whose id matched a formal with the formal object itself.

The broadcast kernel then conformed the wrong axes: `broadcast_double` received
source extents `(8, 2)` for a `(8, 1, 2)` view being broadcast to `(8, 4, 2)`,
which is not even a legal broadcast, and every consumer read misaligned
elements.

The view alias now records an `ssa_storage_view` receipt naming the storage
value id, the view shape and the operation. Both passes retain an occurrence
that carries such a receipt when its element count equals the storage's, and
interning still adopts the formal's dtype, device and accounting so one storage
identity keeps exactly one physical contract.

### 2. `%` and `//` had no tensor opcode spelling

`_SHAPED_SSA_OPERATIONS` lacked `Mod`/`FloorDiv`, so a shaped modulo or floor
division fell through to the scalar instruction emitter. The generated C
computed element zero into a scalar and left the result buffer at its calloc
zeros. Both are catalogued binary kernels (`CT_OP_MOD`, `CT_OP_FLOORDIV`) and
now lower as such. Native output matches eager for all four operand-sign
combinations, so the catalogue's Python floor semantics are preserved.

### 3. A dtype spelling counted as a non-numeric data operand

`_is_dispatch_metadata_node` rejected any node with a constant operand that
`flatten_tensor_constant` could not read. The `"int64"` string in
`x.to_dtype("int64")` is operation metadata, not data, so the whole cast became
coordinator metadata, left every numerical region, and its result became an
unproduced region feed that read as zero. Dtype spellings are now exempt, the
same way basic index literals already were.

### 4. A sliced call result was treated as a call-boundary projection

`call_result_projections` walked every `Indexed` successor of a callsite and
removed it from region ownership. Only a literal integer index is a structural
path onto a call's outputs; `moment[:, :, 0:2]` is a numerical view some region
must compute. Treating it as a projection removed it from every region and left
the caller with an unnamed formal that nothing produces — the
`formal_parity` finding seen on `vehicle_close_contact_graph`,
`vehicle_rig_points_vector` and `vehicle_periodic_terrain_vector`.

### 5. A short-circuit reduction operand was never produced

`physical = bool(finite and position.abs().max() < 100.0 and ...)` in
`validator_simulation_advance` is the construct behind the build's single
undefined operand. A short-circuited `and` operand is coordinator work whose
evaluation may be skipped, so it belongs to no numerical region. Structural
recovery rebuilds such operands from their exact graph edges and already
handled comparisons, casts, constants and `isfinite` -- but had no reduction
case, so `pressure.isfinite().all()` failed with reason `operator`. Its
enclosing boolean chain failed with it (`operand:9`), and the owner was left
calling its own region with a value nothing defines.

Recovery now emits, for a single-operand `all`, `any`, `sum`, `prod`, `mean`,
`min` or `max`, exactly the instruction a region body carries for the same
source node; tensor lowering turns it into the reduction kernel either way. An
explicit axis or keepdim is left unrecovered rather than guessed.

`TURING_DEBUG_STRUCTURAL_RECOVERY` prints each recovery attempt and its
outcome, which is how the failing operand was located.

## Supporting changes

- `_DISPATCH_METADATA_CACHE_SCHEMA` moved 2 -> 3 so saved graph replays
  recompute classifications under the new rule.
- `reduce_scheduled_shader_regions` lost two superlinear scans: the
  coordinator-crossing legality test now uses a per-source index, and vertical
  fusion builds one owner map per quotient instead of scanning every direct
  edge and every region member per candidate edge. The fusion identities,
  legality rules and emitted regions are unchanged.
- A parameter whose only use is being returned keeps its ABI name. The `Ret` is
  emitted after the naming pass, so `return vehicle_input, ...` previously left
  that parameter unnamed.
- `TURING_DEBUG_VIEW_ALIAS` prints view-alias decisions; the existing
  `TURING_DEBUG_GRAPH_NODES` probe now also prints each node's tensor
  descriptor.

## Evidence

`tests/test_native_shaped_view_lowering.py` covers every repair by compiling at
`-O0` and comparing with eager execution, and pins the three open defects:
6 passed, 3 xfailed in about 18 seconds.

Three real validator components compile and match eager execution exactly,
with zero structural findings:

| function | outputs | worst absolute error |
|---|---|---|
| `vehicle_periodic_terrain_vector` | point, normal | 0.0 |
| `vehicle_rig_points_vector` | force, moment, reactions | 0.0 |
| `vehicle_close_contact_graph` | all seven, including the in-place `contact_input` | 0.0 |

Before the repairs the terrain function's own outputs were wrong by more than
1.6 absolute, and the other two carried unnamed formals.

Adjacent focused batch: `test_ssa_fusion_regions`, `test_region_kernel_dedup`,
`test_deployment_outlining`, `test_repository_ssa_dispatch`,
`test_scheduled_process_graph_dispatches`, `test_native_call_input_receipts`
and `test_tensor_ssa_call_metadata` pass 71 tests in 12.54 s.

The combined nine-file gate (the seven above plus
`test_precompile_to_ssa` and `test_ir_sequence_tables`) reports 9 failed /
194 passed. The identical nine failures reproduce in a clean `git worktree` at
`3af7d206`, so none of them belong to these repairs.

## Closure captures

A nested function's captures are formals with exact source names, but they are
not authored parameters, so every signature containing one read as though it
had grown values no caller could name -- two such groups on the balloon tire's
`_wrench_force` alone. A `closure_formals` receipt now records each one and
`check_formal_parity` accounts them.

The discriminator is the source graph, not the name: only an `Input` node whose
binding kind is `closure` or `external` is a capture. A first attempt keyed on
name history instead, and it claimed `balloon_tire_reduced_vector_step`'s own
`r0`/`z0` tuple-unpack temporaries, which are escaped locals rather than
captures. That would have masked a real defect. With the graph-backed rule
`_wrench_force` drops from 14 unnamed formals to 6 (the eight authored captures
`wrench_k`, `wrench_c`, `shoulder_r`, `roller_engaged`, `surface_kind_q`,
`cylinder_radius_q`, `plane_point_q` and `plane_normal_q`), while the balloon
step's escaped locals keep announcing themselves. The remaining six in
`_wrench_force` are anonymous expression captures -- two subscripts and two
`.shape` reads -- which have no source name at all.

Execution of nested calls that capture enclosing values remains broken and is
pinned as two xfail regressions. The call result reaches the caller as a
scalar occurrence, so only one element is ever written, and an enclosing
parameter consumed only through the closure is dropped from the enclosing
signature and replaced by one anonymous formal per callsite. Both reproduce
identically in a clean worktree at `3af7d206`.

## Open, pinned, not fixed

A declared `int64` output is published as the double working representation
whenever a tensor kernel also consumes it. Returned alone the cast writes
`int64_t` storage and the exported buffer agrees; consumed as well, its storage
settles to `double` for the kernel while the root wrapper still declares
`int64_t`, and the publication copies raw doubles. The call edge has physical
input adapters but no output adapter. Marked xfail as
`test_consumed_integer_output_is_published_in_its_declared_dtype`.

An isolated lowering of `vehicle_material_nodes_vector` pursues
`AbstractTensor.reshape` as Python source when the target shape is derived from
`.shape` on a `matmul` over a `swapaxes` view. The real builder lowers that
shell successfully, so this is probably a fidelity gap in the isolated harness
rather than a build defect; it was not chased.

## Validator build result

The fresh build with every repair (`build/validator_frontier_20260915_v13`)
closed the blocker it was aimed at. The full-native execution contract now
reports:

| gate family | v12 (repairs 1-4) | v13 (all repairs) |
|---|---|---|
| unmaterialized boundaries | 0 | 0 |
| unresolved calls | 0 | 0 |
| **undefined operands** | **1** | **0** |
| unaccounted formals | 47 | 27 |
| optional merges / structural outputs / non-native | 0 | 0 |

Unaccounted formals is the only remaining family, and it is fully enumerated:

| function | unnamed | what they are |
|---|---|---|
| `validator_simulation_advance` | 13 | shaped `(341,)` and `(1, 1, 1)` values used only as region-call feeds |
| `_wrench_force` (two specializations) | 6 each | two subscripts and two `.shape` reads over captures, plus one variant-row column |
| `step_with_dt_control_used` | 1 | value 473 |
| `vehicle_tire_recurrence` | 1 | value 6 |

`balloon_tire_reduced_vector_step`'s four `r0`/`z0` tuple-unpack temporaries
are absent from the v13 list only because that build carried the earlier
name-keyed capture receipt. Under the graph-backed receipt they return, as they
should: they are escaped locals, not captures, and closing them needs a real
repair rather than an accounting entry.

Giving closure captures their caller-side tensor descriptors was tried and
reverted. It reached the specialization (the digest changed) but did not let
`wrench_k.shape[:2]` fold, so it bought nothing and was not worth the extra
specialization variants. A minimal repro is
`def root(a, gain): def blend(query): return query * gain.reshape(gain.shape[:1] + (1,))`,
where the capture's graph node still shows no tensor descriptor.

## Reaching execution

`build_simulation` writes `manifest.json`, the compiled library, the repository
SSA and the artifact into its output directory; the viewer then runs natively
through `tools/run_vehicle_native_assembly.py --native-simulation DIRECTORY`,
which drives `NativeValidatorSimulation`. Both gates before that point are
strict: the lowering contract above, then `run_all` findings must be empty and
C emission complete.

## Validator build command

The balloon-tire validator simulation is the target: its program includes
`balloon_tire_vector_step`, `balloon_tire_reduced_vector_step`,
`balloon_tire_gas`, `balloon_tire_membrane_face`,
`balloon_tire_bead_implicit_step` and `balloon_tire_vector_initialize` through
`vehicle_python_compilation`'s import of `BALLOON_TIRE_VECTOR_SOURCE`.

```powershell
python -u tools/build_vehicle_validator_simulation.py --output build/validator_frontier_20260915_v12 --lanes 8
```

Log: `build/validator_frontier_20260915_v12.log`. A fresh build is required
rather than a checkpoint replay because repairs 3 and 4 act before SSA, so a
saved pre-frame checkpoint cannot exercise them.

The machine is shared while this runs. Watch the actual build process, not the
launching shell: a previous watch followed the wrapper pid and wrongly reported
the build as exited while it was still working.

## Perforated engine training

Paused at the user's request because that frontier has concurrent work in
progress. Nothing in this session's repairs is specific to it, and no engine
training process of this session's is running.
