# Tensorized dt spans — 2026-09-19

The full Metrics/Targets migration is implemented locally, resolving the scope
question in HANDOFF_2026-09-19_dt_publication_spans.md. Whole-program lowering
time and seven-law native correctness have **not** been measured.

## Numerical ABI

| Fields | Extent | Meaning |
| --- | --- | --- |
| Metrics.error_channels, error_present | C | Measures and publication mask |
| Targets.error_limits, error_limits_present | C | Limits and declaration mask |
| Metrics.control_values, control_present | 10 | Controller diagnostics and presence |
| pub_tau, pub_tau_present, pub_contract, pub_dt_limit, pub_dt_limit_present | P | Each participant's time contract |
| pub_values, pub_present, pub_limits, pub_limits_present | P*C | Participant channel measures and limits |

All are AbstractTensor spans. The native ABI uses float64, including numeric
presence and contract codes. Absence differs from a published zero. P follows the
supplied laws' causal order; offset is `participant*C + column`.

The shared C=15 layout is `DT_CHANNEL_NAMES` in error_channels.py:

0 energy_j; 1 power_w; 2 shadow_growth; 3 div_inf; 4 mass_err;
5 height_positivity; 6 tracer_bounds; 7 maximum_substep_displacement_m;
8 causal_dt_excess; 9 time_slip; 10 spring_causal_dt_excess;
11 world_sparse_shape; 12 columnar_material_unit_error;
13 columnar_nonfinite; 14 damping_factor.

This order is independent of the process registry. Programs may append columns
while preserving this prefix, but must declare matching extents for all producers,
consumers and extraction contracts. Artifact layout metadata publishes the names.
`channel_fields` converts configuration at setup; `channel_report` and
`control_report` resolve names at reporting boundaries. The old keyed compiler
fixture is frozen in tests/fixtures/keyed_dt_record_abi.yaml, outside the live ABI.

## Producers and consumers

The controller judges intersected presence masks with tensor operations.
Participant penalties remain per participant. The original denominator floor of
1e-30 remains. Numeric controller diagnostics have separate columns; attempt logs
carry soft and rollback channel masks. Scalar Metrics fields and the existing dt
save/restore mechanism remain in use.

llvm_dt_system.py allocates nine flat pub_* buffers on PieceState, declares their
extents, fills values/masks/contracts on every attempt, and forwards the buffers
through Metrics. Setup copies target limits to each row. Metrics without
participants carries zero-length publication spans. No Publication constructor,
state.publications dictionary, or mapping-to-span conversion remains on that step
path. Publication builders remain host-side APIs.

Metrics.pub_* references producer-owned storage for the current attempt; retain
copies when recording historical publication arrays. Engine, graph, chamber,
fluid, vehicle, shadow, damping, host and diagnostic-tool users were migrated.

## Measured validation and remaining failures

- Six span regressions pass (20.04 s), including four C native executions:
  energy presence/absence, constructor/coercion/publication storage, single-channel
  proposals and two-participant proposals. Tests exercise the final flattened
  column and actual numerical results, not just emitter COMPLETE status.
- 78 dt fast tests passed. This marker does not establish lowering correctness.
- 37 runtime/engine/scientific/rollback checks passed (6.18 s).
- 37 focused metadata and pure-call compiler checks passed (5.58 s).
- Optional-value tests and the LLVM drift-piece/Python-controller test passed in
  the preceding combined batch. The drift test is not combined-controller lowering.
- The keyed-record selection had 8 passes and 2 failures. Both failures reproduced
  at clean b74c1874: the obsolete <1e9 id assertion and expected mapping-or-default
  Select. The temporary clean baseline was removed after testing.

Energy, single-channel and participant proposal audits report zero findings
(601, 709 and 761 rows). Publication stores execute correctly but report two
alias-target-missing findings: root 23 to 16 and 27 to 24. These remain unresolved;
native values alone do not dismiss them.

tools/repro_targets_expansion.py remains red. The real coerce_metrics-to-proposal
slice lowers, but reports 9 identity findings (6 conflicting-storage, 2 duplicate
storage, 1 unaccounted formal); its reference evaluator stops on unsupported
llvm.fcmp.ord. It is not counted as validation. Earlier persistent-failure and
failed-window tests that stalled were stopped, not counted as passing or attributed
to a baseline. The full dt suite was not completed.

## Compiler changes supported by these slices

1. Specialized coercion left an unused generated receiver argument despite
   forwarding the actual span fields. Its callee contained only Ret. Existing
   pure-call cleanup now removes an unused return-only call, updates its receipt,
   and existing entry cleanup removes the unused generated receiver. Used results,
   Phi inputs, declared outputs, stores and arithmetic in/out writes are preserved.
2. Metadata propagation grouped equal local integers in unrelated functions.
   An SSAValue write trace showed energy-helper value 68's bool changing
   binary_value's unrelated double value 68. Propagation now uses the existing
   source-region owner scope; cross-owner propagation follows call edges.
3. The cached tensor reference was borrowed as mutable Function objects. The same
   trace showed fill_double shape/type writes modifying cached input. Each whole
   source compilation now owns one copy, shared across its regions, retaining ids.
   The energy regression checks that the cached module is unchanged. Energy then
   proposal now passes in one process.
4. worst_penalty now tests the declared pub_values extent before computing
   penalties. Reading the returned temporary's shape had introduced an anonymous
   zero-filled shape argument, incorrectly taking the empty branch for 30 elements.

## Seven-law measurement still to run

AGENTS.md reserves long lowering runs for the user. From the turing root, this
uses the existing b64 pieces in explicit causal order and the sanctioned whole
source entry. It compiles but does not execute the result:

```powershell
python -u -c "import sys; sys.path.insert(0, 'examples'); from llvm_dt_system import lowered_system; from src.compiler.identity_concordance import concordance_report; laws=('voxel_air_step','voxel_species_step','aerosol_step','droplet_step','salt_solution_step','surface_step','pool_step'); system=lowered_system([f'artifacts/llvm_pieces/{law}/b64/{law}.piece' for law in laws], directory='build/tensorized_dt_seven_b64'); print(concordance_report(system.module))"
```

No new pieces are needed for a 4x4x4 host. The handoff's C extent/gather issues
and combined correctness still need verification before a real render.
No compilation speedup is claimed from the small tests.

## Worktree cleanup

wtp, wtb and wt_head were removed after preserving binary patches and untracked
files with SHA256 checks under build/worktree_preservation_20260919. Contaminated
wtp was never used as a clean b8e69910 baseline. A stale missing registration was
pruned; main and gh-pages remain.

The original handoff, pre-existing RAR scheduler edit, extent-audit file and
dt_negotiation_helpers.py.bak were preserved. Changes are uncommitted.
