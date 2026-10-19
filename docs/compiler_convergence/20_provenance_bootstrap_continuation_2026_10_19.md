# Provenance bootstrap continuation — 2026-10-19

## Current checkpoint

Canonical repository: `C:/dev/Powershell/turing`, branch `main`.
This report accompanies a checkpoint of the accumulated tracked compiler,
tests, bootstrap tooling, and documentation changes since `7c560b04`.
The checkpoint is incomplete compiler work, not a successful self-host build.
Generated products and the untracked `.claude/worktrees/` checkout are excluded.

The latest evidence is the user's normal bootstrap result, read alongside its
local unit receipts:

```text
fingerprint: 9a2048331e799d30ff626164904bb2b675184fa738ecd1f0dd21994e41fd0f71
diagnostic log: build/compiler-logs/compile_erxeo2k9.log
receipt root: build/compiler-partitions/products/provenance-core/<fingerprint>
stage receipt: <receipt root>/compartment.json
unit receipts: <receipt root>/growth/product/units/unit_NNN/
admitted=28/327, pending=299, peak-workers=5
verified=0, compiled-unverified=25, partial=1, failed=2
stage 4/11: 5m09s; coordinator RSS 1.67 GB; reported peak 3.63 GB
planning-cache release: 104.062s
```

`compiled-unverified` means compilation/accounting completed; it does not
establish native execution parity. Admission counts from different fail-fast
runs are not a monotonic correctness metric because worker completion order
changes the admitted set.

Confirmed progress in this fingerprint:

- Unit 033, `IdentityPage._resolved`: `compiled-unverified`, complete accounting,
  nine source conditionals and nine lowered conditionals, no shortfalls.
- Unit 049, `MonotonicIdSource.mint`: `compiled-unverified`, complete accounting,
  output value 7, no shortfalls. Its previous unexplained receiver-field input
  blocker has cleared in the actual user run.
- Unit 035 has no unit receipt in this fingerprint. Do not infer that its
  earlier optional-tuple call-result problem has cleared.

## Immediate blocking units

### Units 041 and 065: physical call adaptation revisits a completed state

Unit 041, `_validate_row`, fails during SSA lowering with:

```text
physical call input adaptation repeated a completed state:
round 1 returned to round 0; last round reported 1 change(s) over 306 call operands
```

Unit 065, `src.transmogrifier.ssa.SSASequenceDescriptor.to_mapping`, fails at
the same gate, reporting two changes over 245 call operands.

Start with `src/compiler/ssa_call_input_adapters.py`:
`_adapt_physical_call_inputs_round`, `_adaptation_state_signature`, and
`adapt_physical_call_inputs`. Trace the exact operands, formal/resident objects,
dtype/accounting changes, and reported change count across both rounds.
The receipts prove a repeated tracked state with a positive change count.
They do not yet prove whether this is a false progress report, a mutation
missing from the signature, or two writers undoing each other. No writer has
yet been convicted. Keep the convergence refusal until that chain is traced.
The recent authoritative-resident and early-reference-feed changes are relevant
inspection points, not an established cause.

### Unit 034: lookup and return publication remain incomplete

`IdentityBook._virtual_edges_into` reaches compiler finalization as partial.
Its receipt contains all three shortfalls:

- Required source value 26 (`get`) is unresolved. Its exact parents are
  `(23, operand)` and `(25, arg:0)`.
- Public inputs 19, 13, 35, and 37 are unexplained.
- Native return paths `((0, 0), 41)` and `((0, 1), 0)` have no published outputs.

Trace the lookup's descriptor/key contract and its control-owned return path.
Do not classify these inputs as explained merely because a linked call uses
them, and do not turn the empty Ret into successful accounting.

## Changes retained in this checkpoint

The accumulated changes include declared native type expectations and unions,
source Enum and formatted-glyph contracts, sequence/keyed reference lowering,
loop-carried union tags, provisional linked-result tags, provenance-preserving
SSA writes, return/control ownership, bootstrap restart handling, and the
later independent correctness review. Earlier receipts in
`19_census_2026_10_16.md` and `TEST_BASELINE_AND_HAZARDS.md` describe those rounds;
their old frontier counts must not be substituted for the current receipt.

The latest independent reviews found and repaired these localized defects:

- `_settle_child_pool_operation_dtypes` keyed evidence by bare numeric SSA ID
  across functions. Evidence and mutations now include the owning function;
  unrelated equal-numbered values retain their own dtype.
- `_scoped_value_type_abi` could accept supplied rows absent from a lexical
  scope's authoritative page. Scoped handoffs now require the page row and
  agreement with it.
- `_publish_concordant_sequence_views` retained stale nonempty metadata when
  recomputation yielded no views. It now publishes the current snapshot,
  including the empty snapshot.
- Exact region feeds selected a resident using `feed_id` but checked and
  published evidence using another occurrence. The selected resident is now
  retained through eligibility, provenance, and rebinding.
- Dependency order used additive fingerprints and dependency levels used only
  node count. Their cache keys now include exact topology identities.
- Retained methods without an enclosing ClassDef now contribute authored
  scalar stores to the receiver field table. `MonotonicIdSource` also explicitly
  declares `_next: int`.
- Dominance checking could start at an earlier-inserted dead block. It now
  prefers the declared entry and then the block named `entry`.
- Final self-checks now validate Phi operand/label arity, unique incoming
  labels, reachable predecessor coverage, and leading-Phi placement.

The optional-result hierarchy regression was extended to a single declared
semantic output owning several payload/presence pairs. This does not resolve
the real `_resolved` return's missing aggregate ownership, described below.

## Open independent-review findings

1. **Optional tuple ABI:** frozen unit 035 from the earlier
   `9d791044663bed86b7447aa47256121594659f887eb149d5d35f609b8c01b086` run
   presents `None | tuple` as one `opaque_ref` payload in `VALUE_TYPE_ABI`,
   while `_resolved` physically returns `opaque_ref, bool, int64, bool`.
   It has two leaf-level optional layouts, no single semantic output, and
   no native union return layout connecting that frame to caller value 8.
   The reducer reports incompatible return arities `[1, 2]` for tuple-return
   versus None-return sites. The structural aggregate/presence ABI must be
   authoritative across reducer, result signature, and call binding.
2. **Hidden region effects:** `_schedule_loop_callsites` orders value
   dependencies but can move regions carrying `table_region_operations` or
   post effects across authored mutations. The purity classification also
   ignores those operations. Effects need their existing storage/control
   identities in the scheduling dependency contract.
3. **Final in-place bit pack:** an unused pursued in-place pack is attached to
   the next numerical region. If there is none, the current path can discard
   the mutation. Preserve its authored callsite/control effect before return.
4. **Finalization diagnostics:** `_unexplained_root_argument_details` already
   exists but unit partial receipts retain only IDs. Publishing those existing
   details would make the failure receipts more actionable.

These review findings were not repaired by suppressing gates or recognizing
particular unit numbers. They remain continuation work.

## Validation at the checkpoint

The pre-commit focused run completed in 23.23s with **211 passed, 9 failed**:

```powershell
python -m pytest -q tests/test_precompile_to_ssa.py tests/test_value_union_abi.py tests/test_ssa_call_input_adapters.py tests/test_ssa_definition_dominance.py tests/test_hierarchical_plan.py tests/test_concordance_audit_contracts.py
```

All nine failures are in `test_concordance_audit_contracts.py`:

- Three `test_capture_audit_reports_unresolved_binding_instead_of_crashing`
  variants fail in fixture construction at forbidden raw `name_binding.set`.
- Two `test_removed_binding_retires_its_capture_with_the_same_provenance`
  variants fail at the same raw-write restriction.
- Two `test_controller_audit_preserves_existing_restore_type_contract`
  variants and the `energy`/`toplevel` variants of
  `test_audit_has_no_undefined_values_or_unposted_minted_ids` fail because the
  optional `Metrics.dt_limit` field has no `class_field_declaration` cell.

These failures are observed on this checkpoint. They were not compared with
an untouched checkout in this packaging round, so their introduction point is
unclassified. Do not weaken raw-write or declaration authority to green them.
The earlier statement that the 32 focused checks were green was narrower than
this new run; it was not a clean result for the whole concordance-audit suite.

The additional receiver/cache/receipt and bootstrap-policy batch completed
with **28 passed, 2 skipped** in 133.73s:

```powershell
python -m pytest -q tests/test_abstract_tensor_topological_reducer.py::test_retained_method_scalar_store_declares_receiver_field tests/test_aggregate_call_identity.py::test_dependency_order_cache_keys_on_node_identity tests/test_aggregate_call_identity.py::test_dependency_order_cache_keys_on_exact_edges tests/test_memory_regulation.py::test_dependency_level_cache_invalidates_on_edge_rewrite tests/test_sequence_contract_concordance.py::test_concordant_sequence_view_snapshot_clears_a_stale_receipt tests/test_whole_program_compiler_bootstrap.py tests/test_native_provenance_bootstrap.py
```

Together, these packaging checks yielded **239 passed, 2 skipped, 9 failed**.
`git diff --check` passes. No full test suite or bootstrap was launched during
report/commit preparation.

## Continuation constraints

- Use the canonical source entry `lower_ast_source_to_ssa`; study the existing
  mechanism before editing it. Corrections must apply to every program with
  the same contract, not a named function or unit.
- The normal user command is `python -u -m tools.bootstrap_compiler_whole_program`.
  The CLI explicitly passes `stop_after_failure=True` unless the user supplies
  `--continue-after-failure`. `--jobs` controls concurrency; `--stage-units` is
  an explicit admission bound. Twelve is not an implicit clean checkpoint.
- The lower-level `bootstrap_provenance_core` API currently defaults
  `stop_after_failure=False`; this differs from the CLI's explicit argument.
  This is retained in the checkpoint and should be reviewed against the user's
  fast-fail requirement. Report preparation did not change execution policy.
- Honor the user's preference for normal runs and visible output. Do not
  substitute custom admission modes to claim a whole bootstrap succeeded.
- Native sectional inputs must carry declared ctypes-compatible storage,
  semantic type alternatives, and exact concordance provenance. Python object
  inspection is not a bootstrap ABI.
- Preserve current failure receipts while diagnosing. Cache cleanup must name
  and validate exact generated targets; never remove broad workspace trees.

## Prompt history

> can you spread out a few agents doing independent review of provenance and compiling correctness, no goal, just read the code at whatever scale an agent can be relied on doing, consolidate reports, and attempt to find problems preemptively

> if I'm not mistaken, this is progress

> can you make a continuation report and commit and push
