# DT continuation: receiver identity, 2026-09-05

Continues `ACTION_PLAN_NEXT_AGENT.md` on `codex/recursive-reduction-bridge`,
starting from pushed commit `bce4f5da`. That report's claims that its changes
were uncommitted are historical; the initial working tree was clean.

## Findings and changes

1. `tools/repro_run_superstep.py` did reproduce `opaque-state-effect` in
   78.90 s. Preserving the managed ABI bindings alone did not fix it
   (72.21 s). The new optional `--diagnose-effects` traced the actual effects
   to `ctrl.update_dt_max` at nodes 421 and 487 in
   `step_with_dt_control_used`, loop 614. These were unresolved method calls,
   not unclassified field assignments.
2. The harness copied selected function bodies into an anonymous source
   module, losing the real module namespace. It now passes the real
   `run_superstep` callable through `python_bindings`, as managed lowering
   does, so the compiler resolves its complete authored closure. The tiny
   fabricated advance kernel is unchanged. Managed ABI bindings are retained.
   `python -u tools/repro_run_superstep.py --diagnose-effects` then produced
   `LOWERED in 88.74s`, with no opaque diagnostics. No classifier weakening.
3. A four-second regression isolates the receiver defect: a function writes
   `metrics.max_vel`, returns `metrics`, and its caller reads both the result
   and the original parameter. Before the fix, the caller's record table
   assigned `max_vel` two slots, `(3,)` and `(8,)`. Plain identity return did
   not fail; mutation is necessary to reproduce this case.
4. `fortran_c_shell.py` now checks the exact argument ledger while publishing
   a record result, before allocating result storage. If the returned record
   is the formal already bound to a caller record, its descriptor and member
   storage are reused. Field schemas must agree; mismatches raise. Fresh
   constructed records still receive distinct storage.
5. Restored the original eight-field normalization in `coerce_metrics`,
   removing the twelve-field source workaround. The separate
   `last_metrics = Metrics(...)` workaround in `dt_controller.py` remains.

## Verification

Each command was run separately, with no overlapping tests or compiler runs.

- `python -m pytest tests/test_aggregate_call_identity.py::test_returned_record_parameter_keeps_its_field_formals -q --tb=short`:
  failed before the compiler fix (two storage slots); passed after (3.95 s).
- `python -m pytest tests/test_aggregate_call_identity.py::test_coerce_metrics_return_keeps_caller_storage -q --tb=short`:
  passed with real `coerce_metrics`; passed again after restoring the original
  eight-field normalization (4.23 s).
- `python -m pytest tests/test_aggregate_call_identity.py::test_fresh_record_return_has_distinct_caller_storage -q --tb=short`:
  passed (3.63 s).
- `python tools/repro_metrics_rebind.py`: `LOWERED in 2.37s` (before removing
  the twelve-field workaround).
- `python tools/repro_return_merge.py`: `OK`, no duplicate result IDs.
- `python tools/repro_return_merge_toplevel.py`: `OK`, no duplicate result IDs.
- `python tools/audit_break_in_if_trace.py` with each of `single`, `nested`,
  `param`, `carried`, `while_break`: each exited 0; exit Phis carry
  `binding='loop_result_port'`, with the expected break predecessors.
- `python -u tools/scan_managed_duplicates.py`: exited 0 after 361.4 s with
  `LOWERED OK`, `functions=169`, `duplicate functions=0`. Only the documented
  `pi_update ... [(11, 2)]` in/out redefinition remains. This run used the
  original eight-field scaler normalization and passed the managed full-native
  extraction contract. The old report's 170-function count is not the count
  of this changed source. Log: `%TEMP%/turing_managed_receiver_identity.log`.
- `git diff --check`: passed.
- Guestbook validator: `All filenames conform to pattern` (read-only run).

The checks above preceded native-build authorization. See the follow-up below
for the subsequent build work. No commit or push has been made in this continuation.

## Full native follow-up

The user requested an eager/native DT option, then explicitly chose to pursue
the full native product. They authorized stopping the stale eager validator
before timing; Windows PID 10852 was stopped. Do not restart it implicitly.

The first detached, serial build used `--assembly-profile dually-axle
--contract develop --optimization O0` and output directory
`build/vehicle_validator_dually_receiver_identity_20260905`. It failed in the
managed tire C emitter, before full vehicle graph lowering: a Const carrying
`"power_w"` reached `int(held)` and raised ValueError. Logs remain in that
directory. This is a C emission failure, not an SSA lowering failure.

`test_c_string_constants_preserve_canonical_token_identity` reproduced that
failure. The C emitter now uses `string_table.string_token` for string/bytes
Const payloads, retaining signed 64-bit identity. The focused test compiles
and executes C and checks text, equal bytes, and numeric-looking text against
the canonical tokens; it passed in 15.75 s. No floating conversion or separate
hash scheme is used.

A second detached build was launched as Windows PID 18628, output directory
`build/vehicle_validator_dually_string_tokens_20260905`, using the same flags.
It terminated with C emission shortfalls (no DLL): call arities 1493/1485
for `run_superstep`, 1001/993 for `step_with_dt_control_used`, 17/30 for
`ssa_sequence_67_append`, and 34/23 for `_apply_energy_sidechain`; unavailable
operands; unsupported module-lane `LAnd` and `item` operations. The original
string Const exception no longer occurs. Full stderr is authoritative.
The build driver now forwards compiler progress and reports stage boundaries
for future runs (this logging change was made after PID 18628 imported it).

Static inspection found a later integration gap: the writer emits
`vehicle_native_graph_tick(void **buffers, long long *extents)` from canonical
vehicle SSA, but the native viewer still expects the older batch exports.
Do not restore the disabled handwritten numeric shell as a workaround. Also,
the managed tire DT window and complete vehicle graph are separate entrypoints;
do not claim the latter invokes the former without checking the call path.
`tools/frame_parity.py` is law-level parity, not full DT window parity.

The user corrected the direction explicitly: the entire Pythonic validator,
including its outer DT/stage loop, must be compiled together and represented
by a generic shell. Adapting separately compiled pieces is not completion.
An active goal now records that requirement. Native law substitutions are
eager-only: `vehicle_python_runtime_bindings` wraps runtime law callables,
whereas graph lowering receives source and retained ProcessGraphs. Preserve
that separation when ingesting the complete program.

`tools/lower_vehicle_validator_program.py` now uses `compile_project_call`
on the actual runner file and `_run_dually_python_profile` entrypoint under
full-native policy. This uses the established authored-source realization,
source partition/import retention, and class-field contract discovery.
`--plan-only` stops before SSA. It does not execute the validator.

The earlier direct-call wrapper probe (`build/vehicle_validator_whole_program_probe`)
produced 56 units but omitted that project-entry setup. Its TypeError was not
accepted as proof of a compiler defect. The corrected project entry produced
75 units and a saved 39 MB ProcessGraph in
`build/vehicle_validator_project_entry_plan` (31.17 s total).
Full lowering via that entry (`build/vehicle_validator_project_entry_lowering`)
also failed after 38.27 s at `_tensor_descriptor`: `dict(data['tensor'])`
receives a non-mapping during `_propagate_callsite_tensor_specializations`.
Those initial runs ended. Replaying the full planner on its saved graph
identified `_node(...)` in `pneumatic_wheel_assembly`: a single dictionary
return has a nested descriptor tree, which callsite propagation incorrectly
published as `data['tensor']`. A small authored `pack(values)` returning a
dictionary reproduces the identical TypeError without the validator contract.
The publication now checks for a Mapping leaf; aggregate trees stay in
`tensor_output_descriptors`. The targeted regression failed before (4.16 s)
and passed after (2.34 s). No catch-and-ignore was added to `_tensor_descriptor`.
The next complete project-entry attempt is in
`build/vehicle_validator_project_entry_aggregate_fix`.
That attempt terminated normally with a caught `KeyError: 792` after
317.63 s (PID 8644 was already gone when a diagnostic stop was attempted).
It passed the aggregate-descriptor exception. Its first console traceback
was truncated, so a repeat was launched with periodic faulthandler traces.
That repeat (`build/vehicle_validator_project_entry_traced`, PID 20284)
crashed inside python311.dll with Windows exception 0xc0000005 at 06:44:00;
Application event 1000 names process 0x4f3c. Causality is not established.
Periodic traceback dumping was removed from the diagnostic driver. The
ordinary repeat now captures full exception tracebacks in receipt.json and
stdout/stderr to disk: `build/vehicle_validator_project_entry_failure_capture`,
PID 11808, tool session 70012. Confirm its live state/result before any run.
That ordinary repeat exited 1 after 310.95 s. Full traceback is now preserved
in its receipt and stderr: `materialize_retained_loop_ports` at loop_composer.py
1672 looks up absent materializer 792. All compiler/probe processes from this
follow-up have ended.

Read-only inspection of the saved graph located this in `PythonValidatorViewer.draw`:
loop 790 is the inner `for channel in color` generator, with result/materializer
792 (`tuple(max(0, min(255, int(channel * scale))) for channel in color)`).
Loop 794 is the outer loop over `sorted(tire_draw, key=lambda row: row[0],
reverse=True)`. Planning just the saved draw graph reproduces the exception in
seconds. Observational wrappers (no compiler semantics changed) establish that
`evaporate_unrolled_loops` selects loop 794 as `CONSTANT`, iterable `()`, and
removes 792 through `remove_nodes_from` at line 1253, while inner loop 790
survives and its immutable plan still names 792. Logs: `draw_probe.log`,
`draw_deletion.log`, `draw_owner.log` in the failure_capture directory.
Before adding cleanup for the nested loop, investigate whether the empty
iterable classification is valid: tire_draw is populated by earlier runtime
appends. Losing mutable collection state would be more serious than the
subsequent missing-node error. No loop-composer patch made yet.

The small source regression `test_populated_sequence_is_not_folded_to_its_empty_initializer`
confirms that more serious problem independently: `rows=[]`, followed by
`for value in values: rows.append(value)`, then a loop over `sorted(rows)`
was classified CONSTANT with iterable `()`. `_fold_callsite_structural_values`
already retains exact `source_sequence_mutation_records` but ignored them
when adding initializer contents to its known-value table. It now excludes
those mutated sequence identities from literal propagation, like carried Phi
initializers; ProgramABI type/storage facts remain usable. The regression
failed in 2.65 s and passed in 2.26 s. No missing-node guard was added.
The original saved draw graph is being planned again with this fix, tool
session 52813, Windows PID 2328, log `draw_mutation_fix.log` in the
failure_capture directory. The prior claim that all runs ended predates this
latest run; check this handle before starting more compilation.
The draw replay subsequently exited 0 with `PLANNED DRAW`. Complete project
compilation resumed in `build/vehicle_validator_project_entry_mutation_fix`;
read its compile-progress.json for the current process ID and its receipt/logs
for completion. No full native success or parity has been claimed.
That complete run exited 1 after 363.78 s. It passed deployment planning
and reached instantiation; `_dependency_order` raised NetworkXUnfeasible
inside `_build_shell_hierarchy_plan` while constructing a function shell.
The traceback and receipt are in the mutation_fix directory. The failing
function is not identified in that initial traceback. The diagnostic driver
now saves the innermost exception-frame ProcessGraph plus cycle edges,
source expressions, and recursion/levels coverage in the receipt. A repeat
is active in `build/vehicle_validator_project_entry_cycle_capture`, tool
session 57265. Read compile-progress.json for its PID and verify it before
starting another run. No cycle fallback or planner patch has been made.
That capture ended after 357.09 s. The innermost graph is the actual outer
`_run_dually_python_profile`, with retained while/lock control already present
in recursion_table. Its levels still name removed constants 154, 359, 363.
`_build_shell_hierarchy_plan` invokes structural folding again, and that pass
deleted constants without refreshing ordering/recursion metadata. A targeted
feedback-graph regression failed before (3.52 s), passed after (2.24 s).
Structural folding now calls the existing `_rebuild_graph_edges` only when
it changes topology. The dependency-order validator remains unchanged.
The populated-list regression also still passes (2.06 s). Replaying the
original failing root with its three isolated pre-fold constants restored
then folding produces `ORDERED REAL ROOT 268 nodes; removed [154, 359, 363]
recursion regions 1`. This reconstructs only the deletion step, not a full
build. Complete compilation resumed in
`build/vehicle_validator_project_entry_order_fix`; inspect its live process
and receipt before starting another run.

Inspection also confirms the saved worker's `python_bindings['run_superstep']`
is the real function. Thus its external_callee_ref at the planning snapshot
does not alone prove the callable is unavailable; audit its eventual source
resolution/lowering rather than substituting a new DT entrypoint.

The saved worker graph also contains run_superstep as external_callee_ref 15,
static_python_reference `run_superstep` (node 98), with no pursued DT function
in this planning receipt. Follow its later resolution; do not mistake the
75-unit plan for complete DT source closure.

The user's native-insertion requirement exposed a separate verified gap:
`bind_native_stand_ins` did not honor the standard authored-source realization
protocol. It now uses `deployed_with_authored_fallback`, retaining inspectable
source and selecting it inside the compiler context even for cached bindings.
The focused source-realization test failed before this change and passed
afterward (0.52 s), checking eager/native -> compiler/source -> eager/native
selection and `inspect.unwrap` identity. The test mocks native kernel execution;
it is boundary-protocol proof, not numerical or end-to-end compilation proof.

## Remaining work

Latest update: `vehicle_validator_project_entry_order_fix` ended after 441.46 s
with aggregate binding `geometry` caller/callee arity 8 != 0, in
`pneumatic_wheel_assembly`. It passed the earlier ordering failure. Its failed
graph is saved. The geometry dictionary is fully static at this callsite;
the specialized callee retains its formal identity as a Constant marked
`structural_specialization`. Hierarchy binding incorrectly treated that
identity as a runtime formal. A two-function dictionary-copy reproducer
failed with the same error (4 != 0). Binding now omits the exact formal when
it carries the structural-specialization marker. No arity check was relaxed.
The focused regression
`test_specialized_dictionary_argument_has_no_runtime_argument_binding`
passes (2.69 s), asserting a resolved native call, no runtime source argument
bindings, and retained result bindings. This is lowering proof only.
Complete compilation resumed in
`build/vehicle_validator_project_entry_literal_binding_fix`, tool session
17154. Check its progress/PID/receipt before starting another heavy run.

The active user goal authorizes sequential native builds and parity work;
the older approval note below predates that authorization. Full-program
compilation, full DT parity, and performance remain unproven.

The literal-binding retry ended after 658.80 s (session 17154 closed). It
passed hierarchy instantiation, then failed during `prepare_graph_precompile`
in `target.refresh_hierarchy_plan`: numerical_region_parents indexed absent
node 2127129342112. The saved innermost graph has no function_name. Traceback,
receipt and failed-process-graph.pkl are in the literal_binding_fix directory.
No native compilation is currently running. Next: trace region-node ownership
and deletion at hierarchy refresh; do not guard away a missing live operation.

The user requested durable translation diagnostics. Updated
`tools/TRANSLATION_DEBUGGING.md` and the machine-readiness decision tree with
the invocation/source-authority/pre-SSA branch and the observed invariant
discriminators. `tools/diagnose_translation.py --compilation-unit` now reads
whole-program receipt.json failures and names the saved failing graph.
`--process-graph` delegates to new `tools/diagnose_process_graph_calls.py`,
reading trusted snapshots without planning or execution. It reports exact
source spans, call references, Python binding availability, and aggregate
argument ledgers, with an explicit limit on native-closure claims.
Verified on the actual wheel call 480 and exact worker/run_superstep call 98;
JSON reports are saved in order_fix. The focused failure-receipt regression
passed in 0.82 s. The lowering driver now records invocation/contract at start
and distinguishes planned/lowered/incomplete status; lowering is not a binary
build. These driver edits occurred after the last run had loaded its code.

The missing node was exactly `range(3)` in `_adapter_source`, present in the
original module catalogue graph and removed in the failed graph. The
precompile loop folded the raw module catalogue unconditionally, although
hierarchy construction already restricts that pass to function graphs.
Precompile now uses the same function_name boundary. A minimal module with an
uncalled `helper(): return list(zip('xyz', range(3)))` and `root(value): return
value + 1` reproduced the identical missing-node traceback before the fix.
The focused regression passes in 2.70 s, including the Add in the selected
root's linked numeric region. No missing-node guard or region pruning added.
Whole-program retry: `build/vehicle_validator_project_entry_catalogue_fold_fix`.
Check that directory's progress process ID and terminal receipt before any
additional heavy run. Full native and parity remain unproven.

The catalogue-fold retry finished after 502.56 s, passing the missing-node
frontier. It raised CompilationSubdivisionRequired for outer validator loops
535 (wheel initialization, opaque material methods) and 587 (viewer loop,
opaque Condition.notify_all plus With). Session 16525 is closed; PID 19480
ended. No compiler run is live. The diagnostic driver now records subdivision
boundaries and captures prepare_graph_precompile's target.process_graph.

Saved-graph inspection found method_ref/callee_ref already present on all
three wheel initialization calls, while earlier loop_state_effects still
marked them opaque. `_resolve_grounded_method_references` now refreshes those
effect records using the same exact linked-call criterion as the lexical
reducer. Unresolved calls and collection mutations remain unchanged. The
Counter.update versus unrelated external.update regression failed before
(2.75 s) and passed after (2.28 s).

The user's With-region suggestion was checked: binary file contexts have a
shell-file region transform, readable generator contexts have source inlining,
and the Python coordinator uses ExitStack. `with status_lock` remains raw With
and is refused by native loop composition. Condition.notify_all has a native
external-reference contract but no internal method_ref; it remains opaque.
Next investigate the native context/effect path without stripping locking,
moving scientific orchestration to a custom shell, or compiling loop bodies
once outside their control owner. The user's unrelated algorithm-change
comment was explicitly retracted and is not authorization to rewrite sources.

Latest user direction: Python threading constructs must become dispatch-region
requirements. For this validator, honor the requested concurrency and let the
dispatcher own threads; do not serialize away its worker/viewer handshake.
Thread coordination belongs to a dispatcher-capable native region, not GLSL.
The user permits investigating/importing Nodus thread infrastructure.

Inspected Turing's native turing_pool.c/h and Python HostDeploymentPool:
these support independent barrier-joined frames, one frame in flight, serial
fallback and refusal of nested native deploy. They do not yet implement an
asynchronous Thread.start/join plus Condition/Event contract. Reusing that
serial fallback for communicating actors could deadlock; nested scientific
dispatch from an actor also needs explicit ownership rather than waiting for
its own enclosing frame.

Read-only Nodus inspection: include/common/thread_pool.h and
src/common/thread_pool.cpp provide enqueue/submit_batch with JobBatch completion
handles, useful as the async-submission model. include/thread_manager.h and
src/thread_manager.cpp additionally own graph/table tick scheduling and are
not a standalone drop-in runtime. Neither inspection proves starvation-free
progress for blocking interdependent jobs. No Nodus files changed or imported.
Next implementation must connect exact Python threading identities and closure
captures to dispatcher-owned task lifecycle and condition/event operations,
with concurrency admission and native execution evidence. No full build live.

Native task runtime progress: turing_pool.h/c now expose dispatcher-owned
asynchronous turing_task_start, timed turing_task_wait and joining/freeing
turing_task_destroy. These are native-entry/explicit-context tasks, separate
from independent numerical frames, with no serial fallback. They preserve the
launching thread's floating-point environment and refuse self-waits. Captured
storage must outlive task destruction; destruction requires exclusive lifetime
ownership. Condition/Event compiler operators are not implemented yet.

The compiled-C handshake fixture in test_turing_pool_runtime exercises two
native threads, a timed wait before peer release, completion/join, and numerical
deployment from the asynchronous task. First run exposed shared TLS under
Windows GCC: __declspec(thread) was not giving the intended semantics. The
platform macro now uses __thread for GNU and __declspec(thread) for MSVC.
The handshake passed in 2.86 s and existing nested-deploy refusal passed in
2.85 s. Windows GCC is verified; POSIX implementation is present but untested.
This is a backend runtime increment, not proof of threading source lowering or
full validator compilation. No compiler/build process remains running.

User explicitly directed threading conversion at AST ingestion, in the existing
special-cases file. Added lower_python_threading to python_special_cases.py and
its invocation beside lower_python_shell_file_contexts in graph_express2.py.
It uses extraction identities, emits dispatcher intrinsic Call records,
preserves Thread constructor targets/keywords, and turns proven Condition
scopes into captured-handle acquire / try-finally release. No source files of
the validator changed. The initially separate python_threading.py was removed
and its implementation moved into the user-requested special-cases file.

The AST semantic test for rebinding+exception cleanup passed (2.35 s). However,
real lower_ast_source_to_ssa plan-only ingestion of `import threading; ...
condition = threading.Condition(); with condition: condition.notify_all()`
exposed an unresolved integration defect: graph dispatch records contain all
four operations, while reduced root nodes contain create/release/notify but
not acquire. Create/notify graph nodes also carry the original stdlib use_native
extraction receipt alongside the new dispatcher intrinsic candidate. Do not
claim this conversion complete or bypass native admission. Next trace effect
retention and original-source receipt correlation in topological reduction;
preserve distinct generated operation identities without invented source spans.
No full compiler run is active. Native Condition/Event implementation and
dispatcher SSA/backend hookup remain outstanding.

Follow-up: traced the missing acquire to per-function ownership filtering in
topological_reducer.py. The call was lexically owned but its temporarily
untranslated operands were not, so the connected-member filter dropped it.
That filter now retains ordered effects; lexical normalization then restores
the actual constructor receiver edge. Receipt restoration now honors the
explicit AST dispatcher-operation provenance instead of overwriting it from
the original source-location ledger (several generated operations share the
same source location). Real plan-only ingestion regression passed in 3.41 s:
create/acquire/notify_all/release all survive with dispatcher intrinsic receipts.
A subsequent graph inspection confirmed all three receiver-taking operations
depend on the constructed condition. This does not yet verify effect ordering,
try/finally native cleanup, backend admission, or native Condition/Event lowering.

Next frontier verified with the same tiny Condition source: opportunistic
lower_ast_source_to_ssa emits only Const(1)/Ret, dropping all four synchronization
operations. The full-native overlay rejects this with two unmaterialized original
stdlib boundary receipts. Do not "fix" this by clearing that boundary ledger.
The control program currently installs use_native calls through
fortran_c_shell._install_external_reference_calls (~5777), but dispatcher
intrinsics have no corresponding control/SSA materialization. Source scope and
finally ordering must be represented, not reconstructed by sorting line numbers
(acquire and release share a source location). User suggested explicit dispatcher
SSA operators and stressed provable backend lowering or safe serial fallback.
No new SSA opcode has been added yet; this remains the next compiler task.

Implemented native default-Condition semantics in turing_pool.c/.h as the
required backend foundation: recursive ownership, wait releasing/restoring the
entire depth, individually selected waiters, notify(n)/notify_all, timed waits,
ownership errors, and exclusive lifetime destruction. It uses a separate native
monitor and per-waiter condition variables, independent of numerical pool frames.
Two native C protocol regressions passed on Windows GCC: recursive release /
restore, timeout and ownership in 3.12 s; two concurrent waiters with notify(1)
leaving the other blocked until notify_all in 3.17 s. No Python callbacks run
either protocol. POSIX code is present but unverified; its timed wait currently
uses CLOCK_REALTIME, so monotonic timeout fidelity still needs addressing there.
Native Event, Thread source-object lifetime/closure lowering, dispatcher SSA
materialization, backend admission, source cleanup, and whole-program parity
remain outstanding. No full build was started in this turn.

Dispatcher SSA increment: added Handler.Dispatch and ControlSource.DispatchBlock.
The final native-boundary control installation now materializes straight-line
dispatcher intrinsics, and precompile_to_ssa emits ordered/effectful Dispatch
instructions with explicit value ports, semantic operation name, source callsite,
keyword names, dispatcher ownership and communicating_tasks capability. Dispatch
is distinct from Deploy/Join (which the serial C lane may erase as receipts),
and is an explicit numerical-fusion barrier. Native C currently reports an
unsupported Dispatch shortfall; no runtime adapter is wired yet.

An actual Python source probe (Condition/create, explicit acquire, notify_all,
release, return 1) now emits all four ordered Dispatch instructions with no
spurious function inputs and one shared receiver. Regression passed in 3.68 s.
It initially missed acquire/release because Python installs those lock methods
on the Condition instance; the special-case transformer now uses proven local
constructor/alias receiver bindings to resolve these methods. Explicit rejected
extraction receipts still take precedence.

Crucial scope guard: the AST transformer records containing control scopes by
AST fields. Generated finally-release shares the acquire source location, so
line-number ordering is inadequate. Scoped dispatcher calls currently fail with
"dispatcher operation requires retained control scope lowering" rather than
being flattened. Try/finally regression passed in 3.74 s. This guard covers
conditions, loops, comprehensions, short-circuit expressions and nested calls;
it is a temporary explicit frontier, NOT completed scope lowering. Native C
backend-refusal regression passed in 2.37 s. Continue by representing cleanup
and dispatcher calls in their original control paths, then lower Dispatch into
the tested runtime with handle lifetime/error behavior. Preserve source boundary
accounting until actual native materialization proves each operation implemented.

Cleanup-control increment: ResourceScopeBlock now represents captured-handle
dispatcher cleanup around a control body. The SSA builder maintains lexical
resource frames and emits cleanup inside-out on normal completion, return,
break and continue. Return values are captured before cleanup; predicated loop
exits get a dedicated leaving block. Loop depth determines whether a break or
continue actually exits a resource scope, so an outer scope around an inner
loop stays held. Cleanup operations deliberately publish no result, avoiding
duplicate SSA definitions when the same cleanup is emitted on different exits.
This is generated resource cleanup, not arbitrary Python finally that can itself
return/raise and override a pending exit.

Targeted control-to-SSA proofs passed: conditional early-return versus normal
fallthrough releases nested handles in reverse order exactly once on both CFG
paths (2.56 s); break releases an inner scope but retains the outer handle until
its post-loop use and final cleanup (2.48 s). These tests instantiate the control
representation directly. AST scope installation is STILL guarded: connect the
generated Condition Try/finally body to ResourceScopeBlock while preserving
ordinary numerical regions, nested control and source identity; do not just sort
dispatcher calls by source line or support only an empty numerical body.

Native emission audit also found CModuleArtifact's generic entry returns void
and currently has no error-status channel for synchronization failure. The
runtime's negative status must not be cast to Python bool (which would turn an
error into True). A checked Dispatch lowering needs explicit error propagation
and cleanup before backend emission can be claimed correct. No native Dispatch
adapter, exception cleanup edge, or full-validator build was added this turn.

AST resource integration now starts in the existing final control-installation
pass: generated Condition Try nodes carry explicit provenance; branch ownership
from glsl_deployment_strategy._branch_compartments groups already-planned numeric
regions/control inside ResourceScopeBlock. The generated release becomes its
result-free cleanup, while acquire/other operations are inserted in their exact
retained control owner. Nested scopes wrap inner-first. A source AST ordinal
breaks equal-location ties (acquire and protected body share the original With
location). Cross-region/noncontiguous or missing enclosing-control ownership
remains a shortfall; loop/comprehension/short-circuit dispatcher sites are still
guarded pending proper insertion there. Do not claim general scoped lowering.

Real source regression passed in 3.46 s: with Condition contains y=x+1 and
notify_all, followed by return y*2. SSA preserves create/acquire/Add/notify_all/
release/Mul using the existing dual-IR numerical regions, rather than replacing
the body or making a second numerical compiler. The former test expecting a
scope rejection was updated to assert this behavior.

User flagged possible overlap with dual IR. Inspected DualIRShell (existing
numeric ControlProgram/FusedProgram pair plus map/resource identities),
ControlDeploymentRegion (independent-lane permissions with serial fallback),
and node_special_cases._ContextInliner (readable generator contexts expanded to
AST Try/finally). New DispatchBlock/ResourceScopeBlock extend that existing
ControlProgram; no new parallel IR container was created. Consolidated duplicate
AST setup/body/finally construction into node_special_cases.context_scope_statements,
used by both the existing context inliner and threading adapter. Existing nested
context/failed-inner-entry cleanup test passed in 2.20 s. Did not find an existing
ControlProgram cleanup-on-nonlocal-exit block. Deployment frame receipts still
cannot substitute for communicating-task synchronization without a serial
progress proof. Further ownership/capability/backend consolidation remains part
of the ongoing work. Native Dispatch emission/error propagation remain absent.

- Resolve the optional-record / `last_metrics=None` source workaround with a
  small reproducer before changing that source.
- Finish the zip arity and retained-loop ancestry audits (items 2.3 and 2.4
  in the prior action plan). These have been inspected, not changed here.
- Native build and frame parity only after the compiler work is verified and
  the user explicitly authorizes those runs.

Loop-placement audit (next frontier): the actual source `with c: i=0; while i<n: c.wait(timeout=0.1); i=i+1` loses its While from the final ControlProgram although the graph retains While, its predicate region and loop_carried_bindings. Separately the wait timeout literal disappeared before lexical normalization. Live ast.Constant leaves now rematerialize from their exact authored value/span, cached by AST occurrence. Direct wait(timeout=0.1) regression passed in 3.52 s: two operands, Const(0.1), no spurious root inputs.

Control installation now indexes lexical loop ownership using the same AST signature fallback as _branch_compartments when graph/source ASTs differ. It supports insertion into retained loop bodies and while conditions. Missing innermost ownership is an error; it must never fall back to an outer scope (that briefly emitted one wait outside the missing loop during development). The loop probe now rejects with "dispatcher operation's control owner is not retained". Missing dispatcher argument identities also raise instead of becoming an optional omission. Next trace why the While vanishes under generated Try, then verify loop-resident wait SSA. No full build or native Dispatch adapter this turn.

While projection fix: analyze_shader_loop_reductions did retain the source While. project_control_regions subsequently deleted it solely because its numeric body was empty. Authored while loops and carried/effect-bearing loops now survive empty-body projection when their predicate remains representable. The source resource/wait probe now emits acquire in entry, wait in while_body, release in while_exit. Also fixed duplicate predicate SSA definitions: the numeric condition projection and explicit control expression now get distinct result IDs; the expression owns the predicate backedge.

The targeted scope/unique-definition test passed (see pytest timing). This was NOT iteration parity: inspection found no Add for i=i+1 in the emitted loop body. Its graph Add existed (updated value 11, input seed 8), but no numerical body region captured it. The while builder then treated the unpublished carried update as an identity backedge because it lacked declared region outputs.

Follow-up: scalar metadata classification now treats loop-carried initial IDs as runtime values, even when their initial node is Constant. Region extraction converts those boundary captures to Inputs with detached metadata, retaining the Add and passing the current phi to it. Planned regions now publish explicit SSA Ret operands. Native execution then exposed a separate C publication bug: singleton shaped arithmetic was computed in a scalar local but retained the address of its unwritten output buffer. The scalar emission path now binds singleton results to the actual local before publication.

Verified: test_literal_seeded_counter_executes_native_iterations passed in 9.17 s, comparing compiled execution with eager Python for n=-3,0,1,7,19. Its native execution runs in a subprocess with a 20-second timeout; retain that bound. Earlier versions hung, including the version with explicit Ret alone. test_resource_wait_retains_loop_scope_and_unique_ssa_definitions passed in 2.71 s with strengthened checks for a nonidentity update and the current-phi region argument. Compare SSA argument IDs, not entire dataclasses: call occurrences legitimately add storage-accounting metadata.

The existing dual IR remains the architectural owner; no second IR container was introduced. Native Dispatch emission, error propagation and full validator parity remain unfinished. The generic identity fallback for unpublished carried updates still needs a source-proof audit; this repair removes that fallback from the tested counter path, not every possible source program. No full build or performance claim this turn.

### Whole-program replay and Event runtime (2026-09-05, 10:27)

Previous turn classified as progress (native counter parity and scope SSA
regressions). Revalidated no Python/zig processes before running
`python tools/lower_vehicle_validator_program.py --output build/vehicle_validator_project_entry_scoped_dispatch_fix`.
Session 34732 / PID 19348 is TERMINAL, exit 1. Receipt records 498.276 seconds,
75 compilation units, and CompilationSubdivisionRequired in
`src.compiler.vehicle_python_live_viewer.PythonValidatorViewer.draw`, reference
50. Failed graph is acyclic with complete levels. Five loop owners 792, 793,
798, 816, 829 retain opaque effects. Actual effect callsites 376, 758, 797,
298, 815, 827 are pygame.draw.circle/line/polygon calls; their attributes contain
only source_type=Call, no extraction identity or native binding. Do not erase
these effects or hoist their loops into a special-purpose Python viewer. Resolve
the display operation boundary through the existing shell/display mechanisms
while keeping authored calculation/control compiled. This run did not produce a
verified native product, nor prove every previously failing unit independently.

Added turing_dispatch_event create/set/clear/is_set/wait/destroy to the existing
turing_pool C/H dispatcher runtime, implemented through the existing Condition.
Wait returns its delivered notification even if another thread clears before
reacquisition, matching installed Python311 threading.py. Native regression
`test_native_event_broadcast_clear_and_future_waits` passed in 2.95 s after the
lowering process finished: two native workers, deterministic parked-wait proof,
set+clear before reacquisition, future timeout, persistent set, null rejection.
Execution is in a subprocess with timeout=20; no Python callbacks in workers.
Windows verified; POSIX remains unexecuted and Condition timed waits still use
CLOCK_REALTIME there. Native Dispatch adapters/handle lifetime ownership are
still absent. Runtime existence alone does not satisfy compilation admission.

Extended tools/diagnose_process_graph_calls.py: selected loop IDs now include
their recorded effect calls and expose missing effect nodes, with a new loops
report field. `test_selected_loop_explains_effect_calls_and_missing_nodes`
passed in 0.55 s. Updated translation troubleshooting guidance for dual IR reuse,
carried scalar publication and native failure propagation.

Error-boundary audit: CModuleExecution calls a void native entry; there is no
checked native error return yet. Existing DeploymentErrorBuffer in
glsl_deployment_strategy.py is root-owned Python exception telemetry, with
path/phase/node/handled/propagation fields, not native control flow. Preserve
that diagnostic model when adding a native failure channel. It must propagate
across compiled calls and unwind active resources; worker failures must not
leave a peer waiting forever. MachineExternalThreadSpawn is a guest virtual-core
request and does not supply this native ABI. No second error orchestration loop
or Python callback is authorized as a substitute for compiled behavior.

### Reachable-body omissions repaired (2026-09-05, 10:45)

Previous goal turn was progress. Investigating the missing drawing extraction
receipts showed the existing `_class_field_reference` and local alias resolver
already resolve PythonValidatorViewer.self.pygame to the real module and
pygame.draw.circle to the real builtin. The saved failed graph even retains
the correct self/pygame bindings, but its drawing AST calls have no receipt.
Do not add a second field resolver based on that symptom.

Found and fixed source worklist omissions in graph_express2:

- Class admission pre-registers every method in target_definitions. Previously
  a later first call queued a registered method only if argument bindings
  changed. Added activated_definitions independently of binding revisions;
  first reachable use queues the body even with unchanged bindings.
- Constructing an admitted class now queues its directly authored __new__ and
  __init__ methods. Merely traversing class-scope executable statements misses
  their dependencies. Unused sibling methods remain inactive.
- An existing reachability regression exposed source-only lexical helpers
  linked by the final pass but never queued because no live callable existed.
  The worklist now activates exact lexical source definitions. Lexical parent
  indexing follows nested scopes, skips class namespaces as method closures,
  and rejects shadowing by parameters, stores/deletes, imports or competing
  definitions. The final link fallback uses this same lookup instead of unique
  basename across unrelated scopes. Unknown/rebound calls remain unresolved.

Tests reproduced failures before each relevant fix. Latest verification:
three registered-method/constructor/module-field receipt checks passed together
in 2.55 s; existing test_pursuit_roots_exclude_unreachable_module_calls_without_bounding_closure
passed in 2.12 s; new nested-helper plus parameter-shadowing check passed in
2.27 s. All are in tests/test_ast_parent_ingestion.py. The module-field receipt
test uses the real program_extraction policy and checks math.sin's identity.
These prove source pursuit, not native drawing semantics or full closure.

No full-program replay after these source fixes yet. Last failed graph/receipt
under scoped_dispatch_fix predates them and must not be treated as the updated
frontier. Re-run the correct project entry sequentially to discover the new
frontier; expect previously skipped dependencies now to become visible. Native
dispatcher error/cleanup ABI, display lowering, complete product parity and
performance remain unfinished. No build/test processes were left running.

### Replay after reachable-body fixes (2026-09-05, 11:02)

Previous turn classified as progress. Verified no Python/zig processes, then
ran the exact full-program driver into
`build/vehicle_validator_project_entry_reachable_bodies_fix`, log beside it.
Session 33298 / PID 19188 is TERMINAL, exit 1, 543.445 seconds. Published 76
units versus 75 before: newly included authored unit is
src.compiler.vehicle_native_deployment.derive_vehicle_rig_rate_hz. The same
five draw-loop owners 792/793/798/816/829 still reject as opaque-state-effect.

Crucial difference: all six drawing calls now HAVE extraction receipts in the
failing graph. IDs 298/797 are pygame.draw.polygon, 376/827 circle, 758/815 line.
They are `use_native`, rule default:native_extension. Parameters request
existing_module/in_place, callbacks=reject, shell_io.external_references,
host_references and host_system. They do NOT declare result_dtype or native_abi.
Thus source coverage is repaired for these calls; do not keep chasing lost
receipts or add another module-field resolver. Next is real native/display
boundary semantics. The installer requires result_dtype before materialization;
the underlying Pygame extension is a CPython callable, not automatically a plain
C ABI. Do not erase opaque effects or label use_native as native completion.

Updated saved-call diagnostic to include extraction_rule and the full declared
extraction_parameters. Its regression passed in 0.55 s. Existing loop-effects.json
in the new build directory was generated BEFORE those extra fields; regenerate
through tools/diagnose_translation.py if full parameters are desired there.

After the full run stopped, expanded compiler_toolchain_fingerprint to include
node_special_cases.py, glsl_deployment_strategy.py, ssa_c_backend.py and
turing_pool.c/.h. These semantic dependencies were missing from pinned-plan
invalidation. Focused fingerprint test passed in 2.34 s. Historical ledgers
missing the newly required files now correctly fail a current-plan check; do
not bypass it or rewrite old ledgers to pretend they recorded those hashes.

Error-path audit: existing ValidationBlock lowers to turing_validation_error;
Fortran emits error stop and C has no corresponding implementation. Neither
provides the recoverable compiled worker error propagation/resource unwind
needed here. Reuse the existing control representation and root diagnostics,
but runtime status handling remains substantive unfinished work. No native
binary/parity/performance run this turn. No active build/test processes remain.

### Presentation direction: geometry instead of Pygame raster calls

The user explicitly proposed removing Pygame from the authored validator's
presentation boundary and delivering geometry to a universal shader renderer.
This authorizes a presentation-interface change; it does not authorize moving
DT, material updates, geometry preparation or worker acknowledgment logic out
of the whole compiled Python program. An unintegrated native raster prototype
and its downloaded Pygame sources were removed. Do not resume that approach.

Existing reusable implementation: src/rendering/opengl_render/renderer.py has
MeshLayer (positions, triangle indices, optional normals/colors), LineLayer,
PointLayer and generic GLSL shaders with an MVP input. However, its draw method
still imports Pygame and the existing threaded host pumps Pygame events. It is
not yet a Pygame-free shell, nor a compiled geometry ABI. shell_io.py currently
describes pixel double buffers, not geometry packets. Reuse these facilities
where appropriate, but do not claim the geometry boundary is already wired.

Implementation contract for the next step:
- Python-authored geometry preparation emits mesh/line/point buffers, camera
  transform, presentation attributes and frame identity. Preserve material
  colors, stage visibility, finite-position checks and diagnostic overlays.
- Extend the existing shell mailbox with a renderer-neutral geometry display
  contract, explicit buffer ownership and completion/error reporting. The host
  performs generic rendering and input transport, with no vehicle computation.
- Acknowledgment must identify the presented frame/attempt/revision. Queue
  acceptance is not presentation; existing threaded presented_frames counters
  alone do not establish the required handshake. Closing or renderer failure
  must release waiting peers through the compiled control/error path.
- Keep dispatcher coordination in the existing dual IR/control mechanism.
  Rendering geometry on GLSL does not make communicating thread blocks GLSL.
- Verify geometry and presentation semantics, then replay the exact whole
  project entry. Measure native compute and display-paced throughput separately.

This inspection changes the intended presentation boundary, not the last
verified compilation result: the complete native product still has not built
or passed parity. No build or tests were run for this documentation update.

### External GL host boundary (2026-09-05, 11:18)

Previous turn classified as progress: it established the user-approved geometry
direction and inspected the actual existing renderer. Implemented RendererHost
in src/rendering/opengl_render/renderer.py, exported by that package. Passing
a host uses its already-current GL context without creating a window or
importing Pygame. Presentation, elapsed milliseconds and optional text rendering
are explicit callbacks. Missing requested text support raises; presentation
exceptions propagate. The existing default Pygame host remains available.
Compatibility-only point sprite enables are restricted to hosts requesting
them; a core GL context does not receive those invalid legacy enables.

Focused test test_external_context_renders_without_pygame_and_propagates_present_failure
in tests/test_glrenderer_host.py passed in 1.19 s. It denies Pygame imports,
records GL operations, checks overlay-before-presentation and resize dimensions,
and verifies presentation failure/missing overlay support raise. It uses mocked
GL: this proves the host boundary, not real GPU output or native frame parity.

The authored vehicle viewer and shell geometry mailbox are still unmodified.
Next integrate renderer-neutral geometry publication and exact frame completion
through the existing shell IO contract, preserving compiled worker feedback.
The existing GLRenderThread still pumps Pygame and is not a suitable generic
completion adapter unchanged. No full compilation/build was run this turn.

### Frame acknowledgment race (2026-09-05, 11:22)

Previous turn classified as progress (external renderer host implemented and
tested). Inspection before geometry mailbox integration found a real eager
viewer race: after draw returned, the loop acknowledged visual_revision[0],
which could have advanced during drawing. That released the initialized-mesh
worker barrier without presenting that revision. The loop now captures the
revision and snapshot alongside current status under status_lock, then only
acknowledges that captured revision after drawing. This is a source correctness
fix, not a rewrite to evade a compiler defect. Numerical algorithms unchanged.

tests/test_vehicle_viewer_frame_ack.py executes the actual authored while loop
extracted from the source AST, with controlled publication during draw and no
expensive worker/model startup. Initial fixture lacked populated status keys
and failed with KeyError; corrected fixture then reproduced the real defect
(acknowledgments [0,1] instead of [0,0], 0.70 s). After the source fix it passed
in 0.57 s, also verifying the second frame acknowledges revision 1. This tests
the specific interleaving, not all concurrency or compiled dispatcher behavior.

Geometry mailbox integration remains the next task. No shell geometry ABI was
added, no whole-program compilation was run, and native parity/performance
remain unverified. Preserve the frame identity invariant in the new transport.

### Geometry wire format foundation (2026-09-05, 11:27)

Previous turn classified as progress (authored frame acknowledgment race fixed
with a reproducing test). Added src/compiler/geometry_display.py: TURGEOM1
little-endian packet header <8sQQQ7I carries generation/attempt/revision,
viewport and element counts. Payload is row-major MVP, xyz+normalized-RGBA
mesh vertices, u32 triangle indices, xyz+RGBA line pairs, xyz+RGBA+diameter
points, and UTF-8 overlay text. Decoder validates total spans before array
views, primitive cardinality, index range, finite values, colors and diameter.
Views are backed by immutable bytes. Producer mutations cannot overwrite an
already encoded packet. This is a foundation, not native emission support.

shell_io.py adds an explicit geometry_display capability and opt-in
ShellIOABI(geometry_display=True) mapping describing this format and exact
completion identity. Existing pixel-only shell profiles do NOT advertise it;
requesting it currently fails shell selection rather than silently claiming
support. Existing ABI mapping is unchanged when extension disabled.

Focused packet ownership/malformed-span/index test passed in 0.52 s. Explicit
geometry ABI/pixel-host rejection and existing shared-mailbox regression passed
together in 1.91 s. No real display or native execution tested.

Next integrate a generic consumer, completion/error mailbox, and authored
geometry production. The initial packet has no per-line width or render-state
batches; preserve the viewer's widths and style through an explicit extension
or authored geometric expansion before adoption. It is not yet a full viewer
replacement. Inspection also found existing GLRenderer.set_mesh allocates new
buffers per invocation without releasing old mesh allocations: repeated frame
consumption must reuse or release them. Existing draw_layers auto-camera and
retained absent layers are unsuitable unchanged for exact frame transport.
No whole-program compilation, parity or timing run this turn.

### Generic geometry consumer (2026-09-05, 11:31)

Previous turn classified as progress (wire codec and explicit shell capability
implemented/tested). Added rendering/opengl_render/geometry_consumer.py. It
decodes packets, replaces all mesh/line/point/text layers including empty ones,
transposes the supplied mathematical row-major MVP for the renderer's GL_FALSE
upload, and returns generation/attempt/revision only after draw returns. It
rejects generations already presented. Presentation failures propagate and do
not advance last_generation; the caller still must transport failed/closed
completions to the compiled program. This is a synchronous adapter for a
dedicated generic renderer, not the shared history/CUDA viewer thread.

GLRenderer.set_mesh now uses persistent buffers with geometric capacity growth,
including explicit zero attributes when normals/colors are removed. Previous
implementation leaked mesh objects on every repeated upload. Mesh shader now
uses explicit uHasColor rather than alpha>0 to distinguish missing colors;
transparent supplied colors must not become the default mesh color.

Two focused tests in tests/test_geometry_consumer.py passed together in 0.91 s:
camera/identity/empty replacement/presentation failure and buffer reuse/removal
of old color attributes. After explicit color-presence addition, mesh test
passed in 0.84 s. Mocked GL only; no GPU shader execution proved this turn.

Remaining: authored viewer geometry producer, width/style representation,
native emission and completion/error transport through shell IO, and dispatcher
native operator lowering. Existing native_process shell still does not claim
geometry support. No complete native artifact, parity or performance results.

### Shared authored part geometry (2026-09-05, 11:34)

Previous turn classified as progress (generic consumer and persistent mesh
uploads implemented/tested). Extracted the existing viewer's part construction
into part_geometry_lines in vehicle_python_live_viewer.py. It emits world-space
polylines with RGB shade, pixel width and closure flag, without Pygame calls or
camera projection. The existing viewer _draw_part_geometry now projects and
draws these same records, preserving one authored geometry implementation.
This implements the user-approved presentation separation; it is not a compiler
workaround and does not move vehicle calculations into a native shell.

Focused test tests/test_vehicle_part_geometry.py checks all seven authored
primitive types, segment/ring cardinalities, widths, shading, ring closure,
world-space radii/offsets and skipped solver-membrane geometry. Passed in
0.54 s. No real rendering or full compilation was performed. The full viewer
still uses Pygame for other shapes and presentation; this shared function is
only the part-geometry portion of the forthcoming producer. Remaining geometry
includes floor, graph edges, membrane colors/triangles, pillars, rollers,
node markers, and diagnostic text. Preserve their behavior and the revision
acknowledgment fix when replacing the presentation boundary.

### Shared authored membrane geometry (2026-09-05, 11:37)

Previous turn classified as progress (part geometry extracted and tested).
Extracted tire_geometry_triangles beside part_geometry_lines in the authored
viewer module. It returns world-space center-surface triangles, fill RGB and
outline RGB using the existing mean-depth admission, stable far-to-near order,
winding-selected interior/exterior palettes and thickness brightness clamp.
The current viewer projects and rasterizes these returned triangles. There is
one source implementation of these calculations, with no native insertion.

Focused tire regression in tests/test_vehicle_part_geometry.py passed in
0.57 s: opposing winding, material brightness upper/lower clamps, far-to-near
order, behind-camera rejection, and unchanged center-surface coordinates under
material changes. This is numerical geometry validation, not pixel parity or
native compilation. Existing mean-depth admission is preserved, not upgraded
to geometric near-plane clipping.

Remaining producer work: floor, graph edges, fixtures/rollers, node markers,
diagnostic overlay, and conversion of their widths/styles into generic geometry
transport. Current whole viewer still has Pygame calls. Do not claim that the
full authored ingestion is Pygame-free yet. No long build or parity run here.

### Explicit Abstract UI ownership (2026-09-05, 11:43)

User steering: "work conspicuously to stay inside abstract ui, making discrete
objects with an api, graphs, etc, serve us in geometric representation as much
as in any other aspect". This governs subsequent geometry work. Do not turn
anonymous render packets into a parallel scene model. Abstract UI graph objects
own geometry; backend arrays are a final projection and must retain identity
spans back to those objects. Inspected AbstractUI, EntityGeometry/AbstractUIEntity,
existing scene-mesh identity spans, and pneumatic mechanical-node contracts.

Moved part construction to src/compiler/abstract_ui_geometry.py and added
realize_part_geometry(node, position, color, alpha) -> existing AbstractUI type.
The projection retains the source node identity, declared geometry and graph
relationship fields. Its polylines have stable child identities and explicit
owner references. The existing viewer now uses this object API to draw parts.
part_geometry_lines remains re-exported in the viewer for existing callers.

Before the steering arrived, extracted fixture_geometry and graph_geometry_lines
from raster code; their existing drawing paths now use them. They retain world
positions/widths/stage shading but still return anonymous temporary records.
Bring these and tire geometry under identified Abstract UI objects next; do not
extend that temporary tuple convention as the canonical geometry model.

Three focused checks passed together in 0.70 s: Abstract UI object/child identity
and relationship preservation across pose changes; fixture poses/styles and
graph stage visibility; existing seven-part shape regression after module move.
No whole native compile, renderer execution or frame parity was run.

The TURGEOM1 transport currently lacks object identity spans and per-object
styles. It is not the authoritative UI model and is not ready to adopt unchanged.
Reuse the existing scene-mesh identity-span convention when extending it.

### Identified graph-edge geometry (2026-09-05, 11:45)

Previous turn classified as progress: part realization moved under Abstract UI
and identity preservation was tested. Added realize_edge_geometry in
abstract_ui_geometry.py. It returns the existing AbstractUI type, preserves
source edge identity, nodes/endpoints and all graph attributes, and attaches an
identified owned polyline projection. Stage-hidden edges remain objects with
visible=False and no drawable polylines, rather than disappearing from the
object representation. Existing shading, width and edge depth order retained.

Viewer graph_geometry_lines was replaced with graph_geometry_objects, and its
draw path consumes these objects. The focused fixture/edge regression passed
in 0.57 s, checking hidden/visible identity and endpoint preservation, source
nonmutation and existing pose/style behavior. No full compile or display run.
Fixtures and tires still need identified object projections; anonymous packet
identity spans, native transport, dispatcher lowering and full parity remain.

### Fixture objects and mechanical communication (2026-09-05, 11:51)

Previous turn classified as progress (identified edge API tested). User now
emphasizes real deterministic support, actuation and torque communication of
the vehicle system. Treat fixtures as causal mechanical objects with commands
and reactions; presentation must not invent their motion/physics. Deterministic
command routing and native feedback still require investigation and verification.

Traced PillarArmPlan/RollerCoveragePlan in vehicle_native_assembly.py and the
negotiated profile.fixture_plan. Viewer now receives that existing plan (the
sole constructor caller updated) and uses realize_fixture_geometry in
abstract_ui_geometry.py. Pillars/roller mounts are AbstractUI objects preserving
negotiated identities, wheel/hub links, articulation/operation and stable child
identities. Count/order mismatches fail explicitly. Existing tuple fixture
geometry was moved into Abstract UI as an internal numerical realization helper.

The coupled graph already emits fixture actuator/passive/compensation forces,
rig reactions and pillar reactions; snapshot previously omitted some channels.
Added fixture_command (input fixture wheel rows), pillar_reaction_force
(result[12]) and rig_reactions (result[8]) to visual publication. Fixture object
mechanical_state exposes the actual command/response and pillar reaction, with
source labels. No extra simulator or new fixture dynamics was introduced.
Tire reaction torque already participates in vehicle_close_contact_graph and
structural-support torque transfer; complete deterministic torque API/audit is
still pending. Do not equate telemetry publication with that work being done.

Fixture identity/mechanics/pose test and existing frame acknowledgment regression
passed together in 2.17 s. Tests use real fixture plan record classes with a small
synthetic state, not a native solver run. No full compiler replay/parity/timing.

### Mechanical feedback audit (2026-09-05, 11:54)

Previous turn classified as progress (fixture objects and existing mechanical
telemetry exposed/tested). Traced authored graph support/torque communication.
vehicle_close_contact_graph masks tire wrench by assembly presence, transfers
vertical wheel load, forms chassis contact moments, and returns negative tire
z-moment as wheel reaction torque. vehicle_graph_tick_vector maps those wheel
loads/torques through wheel_to_structural_support before vehicle physics.
This is evidence of an existing causal numerical path, not proof of conservation
or determinism. No force/moment equation was changed this turn.

Corrected the recent UI publication interpretation: result[12] is -pillar_force,
shape (batch,wheel), a vertical scalar. It is now mechanical_state.reaction_force_y_n,
not a claimed world-space vector. Updated synthetic fixture test to actual
scalar shape; passed in 1.89 s. The prior vector-shaped fixture test was too weak
to establish compatibility with the actual graph result.

Important newly found frontier: _run_dually_python_profile.advance calls
material.tick but has no vehicle_out -> vehicle_in feedback update. tick copies
vehicle output and retains tire/material/history state and carriage state, but
does not update vehicle motion inputs. The legacy main loop explicitly builds
feedback=(output_index,input_index) from declared _next publications around
line 1531 and applies it around line 2416. The dually stage code also prescribes
wheel speed/angle, which needs separating from actual actuator commands under
the user's mechanics direction. Prioritize verifying/fixing the missing eager
vehicle recurrence, including rejection restore and multi-lane behavior, before
claiming real mechanical feedback or comparing native performance. Do not just
copy the separate-component orchestration into the compiled product.

No feedback fix yet, no full compile or native execution this turn. Geometry
work remains incomplete, but this state recurrence is a correctness prerequisite
for both the same eager reference and the complete compiled validator.

### Game object scope and vehicle recurrence (2026-09-05, 11:59)

Previous turn was no progress (acknowledgment only). User clarified the validator
must be a game-hosted object with shape, texture, actions, poses and constituent
actuators making real game connections. This is the architecture, not a future
cosmetic wrapper. Inspected WorldObject/world_graph_model, existing validator rig
WorldObjects, Abstract UI archetype connections/LivingDocument and ActionEdgeTable.
WorldObject supplies persistent containment, forms/materials, capabilities and
physics metadata. ActionEdgeTable records deliveries; it is not a mechanical
command executor. Do not claim registering action rows executes a force/torque
connection. Executable connection binding to the compiled program remains work.

Fixed the missing dually eager vehicle-state recurrence: the entry builds a
mapping from declared output_names ending _next to existing input_names, the
same publication convention used by the legacy caller. Each advance now copies
those outputs back after material.tick, leaving commands and diagnostics alone.
This is inside the authored Python program, not added host orchestration.
The existing _DuallyDTState snapshots/restores that vehicle input buffer.

New tests/test_vehicle_state_feedback.py runs the actual authored advance prefix
with a small deterministic test plant. Before the fix, successive ticks both
observed zero rather than 0 then 0.2 (failure 0.69 s). After fix passed 0.59 s.
Expanded to execute the actual feedback declaration and _DuallyDTState class,
check rollback of inputs/outputs and deterministic replay; passed 0.60 s.
This isolates recurrence and rollback, not physical solver stability, full DT
accept/reject policy, multi-lane vehicle motion or native execution. Existing
stage-prescribed angular motion still needs conversion to actual commands.

No complete native compile/parity/performance run. This behavioral fix may expose
real chassis/support motion previously suppressed by the missing recurrence;
do not tune it away or claim an old eager run validates the new behavior.

### General attachment-point placement API (2026-09-05, 12:03)

Previous turn classified as progress (authored state recurrence/rollback fixed).
User further requires marking any number of points and connecting them with a
chosen structural, actuator or drivetrain part through the same API available
to a person and a machine. Investigated current placement payload/transform and
living document APIs. Extended abstract_ui_placement.py rather than introducing
a separate scene registry:
- mark_attachment_point adds an identified owner-local frame and attached-frame-of
  edge to an existing LivingDocument object.
- connect_component installs an existing PlacementPayload and creates connected-at
  edges for its declared ordered connection_ports, with named port properties.
  It preserves identity/custody semantics and requires exact binding coverage.
  Number of ports is declaration-driven; no fixed two-point assumption.
  Missing owners/points, duplicate identities and invalid port declarations fail.

Focused four-port placement test passed 0.58 s. It verifies ownership/local
frame, exact routing, insertion-order-independent port order, document revision,
placed custody without modifying the source payload, and invalid requests.
This is graph-edit support, NOT yet a working mechanical/game command pathway.
Tool hooks, validator commands, action provenance/dispatch and mechanical-law
binding still need integration; do not label connected-at records physical
constraints merely because they exist. Existing WorldObject/AbstractUI game
hosting remains authoritative. Stage-prescribed wheel motion audit was paused
for this steering and is still unfinished. No native compile/parity this turn.

### Placement actions and edit provenance (2026-09-05, 12:05)

Previous turn classified as progress (general attachment placement tested).
Extended existing IssuedAction with optional parameters, default empty, and
added apply_component_placement_action to abstract_ui_placement. It dispatches
mark-attachment-point/connect-component to the shared graph-edit APIs, checks
destination and expected_revision, rejects duplicate parameter names, and emits
the existing LivingDocumentEdit with actor, action identity, before/after revision
and added nodes/edges. It does not create a second action or document type.

Focused test executes the same mark/connect actions as player and validator,
verifies equal resulting documents, actor-specific edit receipts, unchanged input
document, and rejection of stale replay. Passed in 0.60 s. Existing action-table
registration/count regression passed in 0.52 s after the optional field addition.

Still no live game tool dispatch binding or validator invocation of these actions,
no compiled placement proof, no actual mechanical-law connection evaluator.
Input event ordering is still the dispatcher's responsibility. This action
executor is groundwork, not evidence of physical connection support. Continue
toward a real end-to-end component connection through the existing game object
and compiled mechanical graph; avoid accumulating more disconnected schemas.

### Variable-count existing rig law (2026-09-05, 12:11)

Previous turn classified as progress (shared placement action executor tested).
Audited reusable mechanical support. mechanical_ports.py already defines torque
and bearing boundary contracts; MechanicalCreature methods are descriptive,
not an executable substitute. The actual vehicle_rig_points_vector law had
three hard-coded 16-point reshapes. Replaced them with explicit batch_count and
point_count from the supplied rig tensor. No force/torque equations changed.

New eager test executes that actual authored law for 0/1/3/19 attachments and
two lanes, checking capped command force, lever-arm moment and equal/opposite
reaction sums. Before fix failed at the empty axis broadcasting against
(0,16,1), 3.54 s; afterward passed in 2.95 s. Then changed the existing declared
rig compiler regression ABI from (8,16,21) to (8,19,21): it passed lowering,
complete C emission and shared-library compilation in 9.99 s. Session 10997
terminal exit 0. The library was not executed by that compiler regression.

This removes one real evaluator limit for general placement. Existing rig input
allocation/default RIG_POINT_COUNT remains 16; assembly actions still need an
actual binding/allocation path into the law. This law is body-to-world support,
not a complete arbitrary two-body joint/actuator network. No complete native
validator, full parity, or performance result yet. Do not substitute this small
compiled support check for the required whole-program gate.

### Configurable rig allocation; verification RUNNING (2026-09-05)

Previous turn classified as progress (variable rig law and 19-point C compilation
verified). Added rig_point_count to vehicle_python_compilation_inputs, defaulting
to existing RIG_POINT_COUNT, validated as a nonnegative integer. Allocation now
uses that count. dually_vehicle_python_compilation_inputs forwards its keyword
argument to the same constructor. Native ABI declarations already derive shapes
from the resulting feeds, and eager materialization copies the same layout.

New test test_dually_program_allocates_and_declares_requested_rig_point_count in
tests/test_vehicle_python_graph_source.py prepares two lanes/19 points, checks
eager buffer independence and ABI shape, and rejects invalid counts. It has NOT
finished. Unified exec session 5053, Windows PID 7280, started 12:14:25. Last
Get-Process confirmed alive with 261.91 CPU seconds; subsequent session polls
remain running without output. Full dually preparation is expensive. Resume
THIS session with write_stdin; do not start another run merely because output
is buffered. No other test/build was launched concurrently.

Until terminal output is inspected, allocation change verification is pending.
This does not prove native execution or full validator compilation. The actual
placement-to-rig binding remains unfinished; existing callers retain default16.

### Allocation test correction; replacement run active (2026-09-05, 12:26)

Previous turn classified as progress plus verified wait. Resumed session 5053,
confirmed PID7280 active/CPU advancing, then observed terminal exit1. Test took
575.12 s and passed constructor/eager shape and independence assertions before
failing because the TEST used binding.shape. ProgramABIValueBinding stores the
shape under binding.field.shape; inspected its declaration and corrected the
assertion. This was a test defect, not a demonstrated program allocation defect.

Started the corrected test once, after the old run was terminal: session50509,
Windows PID20456, start12:24:54, latest CPU57.34s. It remains ACTIVE without
output. Resume session50509; session5053 is closed and PID7280 is no longer the
test. Do not claim complete verification until the replacement test terminates.
The full preparation is ~10min and must not be repeated for cosmetic output.
No other test/build started concurrently. Goal remains active, not blocked.

### General placement and first physical binding verified (2026-09-05, 12:34)

Corrected allocation test session50509 terminated exit0: 1 passed in106.92s.
The full dually constructor preserves 19 attachment rows across eager buffers
and ProgramABI; the earlier binding.shape failure was solely a test typo.
Neither session50509 nor session5053 remains active.

Added RigPointBinding / bind_placed_rig_point in mechanical_ports.py. This
resolves an identified placed component's named body/world attachment ports
to the existing 21-scalar vehicle_rig_points_vector input. It validates endpoint
ownership, explicit fixed world root, coordinate space, operator, control mode,
and finite gains/commands. It retains component and endpoint identities beside
the input row. Scope is one body against a fixed world target; this is not a
moving two-body joint or drivetrain executor.

tests/test_component_attachment_placement.py now exercises player/machine
action equivalence, arbitrary named port routing, and a placed support feeding
the actual authored rig law. Offset support produces 20N force, 20Nm moment,
and opposite reaction, with incorrect owner rejected. Entire file: 3 passed,
one existing dependency deprecation warning, 2.22s, terminal exit0. A prior
single-test output was lost to context truncation; no Python process remained
before this verification run, and the pytest timing ledger recorded 2.26s.

Next: connect this binding and arbitrary-count allocation to the actual
validator/game action path, retaining component identity for returned reactions.
Current default callers still use16 rows. General structural/drivetrain parts
need explicit physical operators; graph placement alone does not execute them.
Pygame-free whole-program ingestion, native dispatch adapter, full native
execution/parity/performance remain outstanding. Goal remains active.

### Authored dually stage now consumes placement bindings (2026-09-05, 12:39)

Previous turn classified as progress. Continued from live source, no active
build assumed. Added place_validator_support in abstract_ui_validator_rig.py:
it marks body/world points and installs a support through IssuedAction and
apply_component_placement_action. It accepts existing world/body identities.

_run_dually_python_profile now allocates rig rows from the negotiated structural
support count. Its grasp stage builds a LivingDocument using the loaded model's
actual identity as the body, invokes those placement actions, resolves each
component with bind_placed_rig_point, and installs the resulting row through
_PythonVehicleMaterial.install_rig_binding. Same gains, targets, forces as before;
installation now populates every lane. Material tracks slot-to-component identity
and publishes that tuple alongside rig_reactions in visual snapshots. Identity
metadata is deliberately outside numerical feeds. The older separate native
diagnostic path and its ctypes rig setup are unchanged.

New tests/test_vehicle_rig_placement_execution.py executes the actual authored
grasp AST block and actual install method with19 supports/2 lanes, checking
57 placement revisions,38 attachment points, loaded-body identity, exact command
rows, lane coverage, and no reinstallation on stage reentry. Combined with the
placement/law and persistent-state tests:5 passed in3.01s, terminal exit0. Initial
run failed on a constructor indentation mistake introduced while editing, fixed
before rerun. No test/build process remains active from this turn.

This moves placement into actual authored execution but is not a complete live
game host integration: the LivingDocument still belongs to the validator entry,
viewer does not yet realize these support components, and general joint/torque
operators remain incomplete. The newly reachable Python helpers must be ingested
by the whole-program compiler, not compiled separately or replaced by native
insertions. No newer full compile than the prior draw/Pygame frontier. Next major
work remains completing Abstract UI geometry delivery and native dispatch lowering,
then exact-entry compile, parity and performance. Goal remains active.

### Current whole-program compile active (2026-09-05, 12:44)

Previous turn classified as progress; this turn advances exact-entry evidence
and is a verified wait. With no Python/native compiler process present, started
one full diagnostic (no --plan-only):
python -u tools/lower_vehicle_validator_program.py --output
build/vehicle_validator_project_placement_integrated
Output redirected to build/vehicle_validator_project_placement_integrated.log.
Unified exec session91926, Windows PID18764, started12:39:27. Latest live process
inspection:250.30 CPU seconds,847835136 resident bytes. Session polls still say
running; no terminal receipt exists yet. Resume this exact session, do not launch
a replacement on observation timeout. No source changes made during this run.

Source closure/reduction completed and published84 compilation units (previous
run76), including _PythonVehicleMaterial.install_rig_binding; saved resolved graph
is41879122 bytes. Current stage: selecting complete control/operator deployment.
Imported placement helper names appear in compile-source.py but not as named
units in process-graph-units.json. Inspect exact call receipts before interpreting
that absence: unit count alone does not establish their complete ingestion.

No new full native success, parity or performance evidence. No tests or second
heavy process overlapped this compiler. Get-CimInstance Win32_Process is unavailable
here (Invalid class); Get-Process and unified session polling work. Goal active.

### Native-only admission and Pygame AST replacement started (2026-09-05, 12:55)

Resumed session91926/PID18764, confirmed CPU progress, observed terminal exit1
after611.818s. Latest full compile remains incomplete: draw loop owners557,562,
576,608 reject opaque-state-effect. Exact failed graph saved; call diagnostics
in build/vehicle_validator_project_placement_integrated/draw-call-diagnostics.json.
Effects are pygame.draw.line/circle callsites552,556,561,606, default use_native
with existing_module loader, no native ABI. Graph acyclic, levels complete.
No compiler process remains active. Do not resume/restart91926.

Saved resolved graph inspection also found place_validator_support and
bind_placed_rig_point in apply_stage as external refs15/16, callsites175/206,
with callable bindings present but no extraction receipt or source function-table
body. Source admission/discovery requires investigation; not runtime permission
to invoke Python. User reiterated zero Python runtime callbacks.

Hardened ExtractionContract.decide: full-native native_extension/use_native
requires nonempty non-CPython native_abi, otherwise explicit REJECT reason
native_extension_requires_non_python_abi. Existing final-link gate also rejects
Python host/CPython ABI/profile/unresolved boundaries.5 selected contract tests
pass2.82s; fixed an older test fixture lacking function.metadata and adjusted
its assertion to ignore added diagnostic fields (initial run4passed/1failed).

User explicitly requested Pygame-to-own-renderer AST special cases, using the
game Phong shader. Added exact intrinsic rules for pygame.draw line/lines/polygon/
circle and lower_python_pygame_geometry in python_special_cases.py, invoked after
thread lowering in ProcessGraph AST ingestion. Resolved identity and explicit
abstract_ui intrinsic receipt required; spelling alone/rejected calls untouched.
Calls become turing_abstract_ui_* with ordered abstract_ui_operation metadata,
geometry_display capability, original source identity/span and argument edges.
These are screen-pixel overlay operations with damage-rectangle results; do not
pretend that screen coordinates reconstruct world meshes or use Phong directly.
5 targeted ingestion tests pass2.66s using real Pygame callable identities.

NEXT REQUIRED: consume these operations in existing dual IR/backend geometry
path, preserve stroke/clipping/order/damage rectangles, translate surface/window/
event/text/present operations, connect identified world geometry/materials to game
Phong rendering and retain exact frame acknowledgment. No adapter implemented
yet; intrinsic classification is not a successful lowering receipt. No new full
compile after these changes. Earlier build fingerprints correctly stale now.
Native product/parity/performance remain outstanding. Goal active.

### User correction: real 3D machine objects, no 2D compatibility layer (13:02)

IMPORTANT: user rejected the preceding Pygame primitive intrinsic direction.
Removed ALL newly added pygame-draw-to-abstract-ui rules, AST transformer and
graph hook, and tests/test_pygame_abstract_ui_ingestion.py. The in-progress reducer
receipt preservation change for those operations was also removed. Its temporary
regression reproduced receipt loss in2.41s, but that path is abandoned, not a
remaining implementation goal. Native-extension ABI rejection remains valid.

User specifically corrected rollers: cylinders, not circles. Added reusable
cylinder_surface in abstract_ui_geometry: closed triangle mesh along local z,
separate cap/side normals, physical radius/length, positive finite validation.
RollerCoveragePlan now carries radius(default existing physical0.18m) and optional
length; per-wheel mounting negotiation derives length from section_width_m.
Missing dimensions stay None and solid realization refuses instead of inventing
pixel widths. Other fixture negotiation variants remain valid without geometry.
Articulated/shared dyno roller lengths still need proper group coverage design.

realize_fixture_geometry now emits identified cylinder solids, no marker circles,
with owner, carriage-following position, axle direction, physical form dimensions,
surface mesh and Phong material intent. Existing mechanical command/reaction data
remain attached to the parent roller object. Legacy viewer now projects the same
triangles instead of circles to avoid breaking its preview. This legacy renderer
is still Pygame and flat shaded: NOT the requested game/Phong integration.

Geometry identity/state/cylinder dimensions/unit normals/outward winding test
passed2.00s; real dually and zero/independent/tracked fixture negotiations passed
2.29s. No full compile rerun or process active. Primary remaining presentation
work is canonical WorldObject rig/parts with genuine geometry/material/pose and
mechanical connections, consumed by game Phong renderer. Do NOT resume replacing
Pygame line/circle calls with another 2D API. Compile the authored object/geometry
program end to end, retaining generic shell and zero Python callbacks.

### Authored tick publishes canonical roller objects and mesh (13:05)

Previous turn progress. Inspected actual game Abstract UI scene path:
abstract_ui_div_map.py installSceneMesh/installVehiclePresentationMesh consumes
9float vertices position.xyz/normal.xyz/color.rgb; current game realization is
largely buildExtrudedBoxMesh and its own simulation components, not the desired
single native validator. Do not confuse that existing game route with completion.

Added world_surface_mesh_packet in abstract_ui_world.py using existing
WORLD_MESH_PACKET_VERSION. Expands indexed surfaces with rigid pose, rotates
normals, preserves variable-length object/semantic-part spans, material bindings,
and existing identity specialization table. Validates shape, finite values,
rotation, unit normals, colors, indices and surface-part ownership. No separate
component compilation or renderer-owned geometry generation.

fixture_world_objects in abstract_ui_geometry promotes realized roller carriage
and cylinder children to actual WorldObject records. Parent carries actual
mechanical command/reaction state and wheel/hub/articulation links. Children
carry indexed cylinder surfaces, poses, kinematic/cylinder physics description,
semantic surface identities and Phong shading intent. Material catalogue binding
is still absent; do not call this verified Phong shading or dynamic roller torque.

_PythonVehicleMaterial accepts fixture_plan (dually entry passes profile plan).
Actual tick publishes world_objects and world_mesh_packet in both trial and
committed visual snapshots through its existing publication path. The packet
currently includes rollers only, not the entire rig/vehicle. Host rendering and
native shell transport not wired. Geometry currently rebuilt each tick; keep
eager/native workload equivalent and only optimize after correctness.

Expanded actual geometry/fixture-state test verifies object parent/link ownership,
packet vertices/normals/identity ranges and nontrivial rigid rotation. Paired with
existing vehicle feedback check:2passed2.64s; extended rotation check1passed2.28s.
No whole validator execution or native compile test performed. All processes
from this turn terminal. Goal active; next connect generic world-packet consumption
to game rendering/materials/acknowledgment while completing real rig object set.

### Game shader accepts authored world surface packets (13:09)

Previous turn progress. Added turingDecodeWorldSurfacePacket to existing
WORLD_REGISTRY_SOURCE in javascript_runtime_utilities. Validates existing world
packet schema/layout, finite f32-representable9float rows, complete contiguous
object/part triangle spans, unique identities and part-to-object ownership.
Produces Float32Array plus camel-case spans usable by existing game code.

abstract_ui_div_map.js template now has installAuthoredWorldSurfacePacket(packet):
validates before touching GL, uploads supplied geometry to its own VAO/VBO with
position/normal/color layout matching existing game/Pluck shader pipeline. Existing
drawSceneMeshes now draws this additional layer. Does not replace game vehicle
geometry, regenerate forms or run physics. Empty packets replace previous geometry.
No presentation acknowledgment on upload; completion must wait for actual render.

tests/test_world_surface_game_upload.py extracts and executes the actual game
JavaScript plus shared runtime in Node, with mocked GL: data/layout preserved,
part ownership retained, malformed frame rejected without replacing prior state,
empty frame clears/draws0vertices. Passed1.36s, subprocess15s timeout. No real GL,
shader visual verification, picking or transport test. Node available at
C:/nvm4w/nodejs/node.exe. No tests/builds still running.

Remaining: connect authored snapshot transport to installer, integrate objects
and spans with game registry/picking/material catalog, present/ack exact attempt
and revision, complete all vehicle/rig solids and full-native source compilation.
The installer is currently an available function, not a live connected validator
stream. No claim of game-hosted running machine yet. Goal active.

### Dispatcher target source admission fixed; real plan active (13:14)

Previous turn progress. Reproduced the missing worker closure: source pursuit
followed lexical calls but not Thread(target=worker), so helper calls inside
worker->stage remained undiscovered. New targeted test initially had missing Path
import (fixed), then reproduced real missing _source_helper in3.07s.

Added python_dispatch_entry_expressions in python_special_cases for resolved
threading.Thread target keyword or positional argument1 (argument0 is group).
graph_express2 source pursuit now activates a matching lexical target definition
through its existing worklist/lexical shadow rules, unless extraction rejected it.
Temporary AST node used only for lookup, not inserted or executed. Original
target argument unchanged. Does not yet resolve arbitrary indirect/attribute or
imported callable target expressions; those remain explicit work.
Keyword/positional target closure plus condition real extraction:3passed3.42s.
Earlier lexical shadowing regression passed alongside first fix2.71s. Unused
nested recursive helper remains excluded. No runtime Python callback introduced.

User emphasized dispatcher chooses native thread/Web Worker/WebGPU/serial as
semantically valid. Source admission must be independent of backend placement.
Numerical regions may choose GPU; communicating condition/frame handshake must
preserve coordination and progress, never blindly convert the whole thread to a
GPU invocation. No placement policy change made in this turn.

Started one real exact-entry --plan-only diagnostic after tests terminated:
python -u tools/lower_vehicle_validator_program.py --plan-only --output
build/vehicle_validator_thread_target_source_fix
Log:build/vehicle_validator_thread_target_source_fix.log.
Unified session47667, PID4812, start13:13:21. Confirmed running by session poll
and Get-Process; last published stage building complete ProcessGraph source
closure. Resume this session, do not restart on observation timeout. No source
changes after starting it. Plan-only is source closure evidence, not native
lowering, binary, parity or performance. Goal active.

#### Terminal update (same turn)

Session47667 terminated EXIT0, PID4812 gone. Receipt status planned,
elapsed64.782s. The immediately preceding active-process paragraph is now
historical: do not resume47667. Authoritative process-graph-units.json now
contains src.compiler.abstract_ui_validator_rig.place_validator_support,
src.compiler.mechanical_ports.bind_placed_rig_point (and nested vector), and
src.compiler.abstract_ui_placement.apply_component_placement_action. This proves
the actual worker-closure admission gap was fixed; no full lowering was run.
Next use this expanded source plan for native lowering diagnostics, alongside
the still-unfinished game rendering/native shell integration. Goal active.

### User requests eager host with one native superstep dispatch (13:18)

User explicitly requests an interim performance product while complete validator/
game integration continues: eager Python outside, one compiled superstep call,
DT and trial stepping inside, resident state. Do not present Python callbacks
per trial or a tire-only measurement as whole-vehicle performance. Original
whole-program goal remains unchanged.

Artifact inspection: old vehicle DLLs are component products; latest receiver
build failed int('power_w'), subsequent string-token build failed C emission
with call signature counts1493/1485,1001/993,17/30,34/23, undefined operands and
LAnd/item spellings. These are historical outputs, not current fixed-source proof.
CModuleArtifact already supports shared library compile/allocate; CModuleExecution
holds resident pointer/extents and run() invokes exactly one native entry.

Started fresh managed DT+tire prerequisite build (not whole vehicle result):
python -u tools/build_balloon_tire_native.py --managed-dt --batch-size8
--optimization O2 --frames0 --output build/managed_superstep_refresh_20260905
(Actual CLI used spaces: --batch-size 8 and --frames 0.)
Log build/managed_superstep_refresh_20260905.log. Session18206, PID13860,
start13:16:58. Last confirmed alive20.06CPU seconds, stage reducing source topology.
No other Python/native compiler running before launch. No source edits during
build. Resume same session; do not relaunch on timeout. --frames0 prevents
unbounded automatic native execution; subsequent native tests need subprocess
timeouts, eager/native parity before performance. User has not received a ready
benchmark yet. If emission succeeds, reuse emitted module for resident eager
native wrapper rather than per-call subprocess/serialization timing.

### PRIORITY: DT compiler first; first current C defect fixed (13:28)

User explicitly corrected priority: stop future game/API work and get DT to
compile and run; that is prerequisite across the project. Game integration is
paused. Preserve full objective, work on DT compiler defects directly.

Resumed managed refresh session18206, verified CPU/phase progress, then terminal
EXIT1 at C emission (PID13860 no longer active). Current log authoritative:
call counts1508/1500 run_superstep,1014/1006 step_with_dt_control_used,17/30 append,
0/1 planned_region10,34/23 energy_sidechain; missing operands; variadic LAnd;
zero-arg item. Build did not produce a valid native product.

Updated tools/repro_apply_energy_sidechain.py to use actual callable globals
(python_bindings), complete managed ABI records, native-only overlay, and save
module/output/export pickle plus C source/shortfalls. No copied function-body
reconstruction. Command --output build/dt_sidechain_c_frontier_20260905 reproduced
real failures in1.88s, terminal exit1. Artifacts repository-ssa.pkl,module.c,
shortfalls.json. Reuse this snapshot for C-only changes.

Actual _no_exchange_observed emits LAnd with THREE bool operands. Module C emitter
only accepted two. Fixed logical expression emission for2+ operands for LAnd/LOr.
New tests/test_c_variadic_logic.py compiles C and checks all8Boolean triples in
subprocess(timeout20s) against Python all/any; passed11.11s. Re-emission of saved
DT module confirms LAnd shortfall gone; shortfalls-after-logical-fix.json retains
only missing %t4,%t35,%t12,%t40 in _apply_energy_sidechain and zero-arg item /missing
%t22 in specialized _scalar planned_region1. No unsupported item placeholder added.

Useful saved SSA facts: _apply_energy_sidechain calls _energy_time_limit with0args
result%4 dtype none, subsequently uses%4 asfloat; same for _shadow_dt_limit result12.
Investigate optional-result/control pruning, do not emitNone as numeric0. _scalar
planned_region1 contains item with NO arguments, receipt builtins.NoneType;
this needs source/specialization/operand accounting diagnosis. Larger DT call
arity defects not yet isolated/fixed. All sessions this turn terminal. No
successful DT executable/parity/performance yet. Goal active.

### DT native control repairs and fresh managed build (13:54)

Game/API work remains paused. Fixed scalar-call placement by extending existing
marker/result dominance relocation beyond aggregate results. None literals now
retain NoneValue through string interning; C NoneValue owns an address so a
callee publishes it rather than leaving its output slot uninitialized.
Exact builtin callable over known structural constants folds without invocation;
callable(None) now removes the dead zero-arg item branch. No .item stub added.

A real sidechain native run initially returned0 instead of0.01 despite complete
C emission. Captured ControlProgram proved order0,shadow,no-exchange despite
carried aliases0->29->31->33. Added conditional state dependency ordering to
control_source overlay: topologically orders adjacent ConditionalBlocks by
explicit carried aliases; other statements are barriers; transparent Sequence
wrappers flattened. Existing nesting/overlay checks3passed2.72s.

Native regression tests/test_native_call_predicate.py: None comparison and
callable(None) passed; new sequential case with independent flag passed11.98s.
Prior combined Boolean+None run3passed29.70s. A broader variant positive(x)
after conditionally reassigning x FAILED: call uses arm result6 instead of
merge13 and is below its own predicate. This is NOT fixed. The checked-in
sequential test uses independent flag, matching actual DT sidechain structure;
retain the broader defect as next call-frame provenance frontier.

Actual callable + managed ABI sidechain build/dt_sidechain_state_order2_20260905
has no emission shortfalls, linked DLL, and native-parity.txt: dt_next0.125,
0.25,0.0 match real eager _apply_energy_sidechain with Metrics(0,0,0,0),
Targets(0.5,0.1,0.1),dt_tensor0.5. Optional energy/shadow limits disabled,
empty channels only; not general DT parity. native-artifact.pkl locally reusable.

Managed emitter now optionally retains repository-ssa.pkl (module,outputs,exports),
module.c,shortfalls.json before raising C failures; compile function supplies
its output directory. This avoids6minute re-ingestions for emitter-only work.

Fresh full managed build started13:53:35: session91655, PID17456:
python -u tools/build_balloon_tire_native.py --managed-dt --batch-size8
--optimization O2 --frames0 --output build/managed_dt_control_fixes_20260905
Actual CLI has spaces after option names. Log same path plus.log. Last CPU16.70s.
No overlapping Python/clang processes before launch. Resume session91655;
do NOT restart on observation timeout. No complete DT executable/parity yet.

#### Terminal managed result and zero-versus-None correction (14:06)

Session91655 TERMINATED EXIT1; PID17456 gone. Do NOT resume/restart it.
Fresh managed output directory now holds repository-ssa.pkl27MB (tuple
module,outputs,exports), module.c2.16MB,shortfalls.json. Six shortfalls:
- root->run_superstep actual1500/formal1492 (callsite43)
- root planned_region0 operand113 unavailable
- run_superstep->step_with_dt_control_used actual1011/formal1002 (callsite223)
- ssa_sequence_67_append actual17/formal30
- run_superstep operand450 unavailable
- step_with_dt_control_used planned_region10 actual0/formal1 (formal76float64,
  accounting sequence_arena=True, call output_ids76,332).
All prior actual sidechain shortfalls are gone. No full native executable.

Added zero-return regression; it FAILED: numeric0.0 was mistaken for None's
zero storage cell. Corrected C Eq/Ne against known semantic dtype none to use
typed absence equality for known primitive SSA types, not their numeric cell;
unknown/reference/record operands retain ordinary presence handling. Four
native tests in tests/test_native_call_predicate.py NOW PASS30.22s (session15880
terminal): None return, callable(None), sequential independent flag, numeric0.
This is after the managed build, so saved module.c predates this emitter-only
fix; saved SSA can be re-emitted without repeating ingestion.

Call-frame inspection of saved current full module:
- native linked call_table root->run has621frame_bindings vs1492formals; many
  late propagated workspace formals absent from original SSACallRecord.
- actual and formal tails both end11Metrics fields (max_vel...advanced_dt), but
  actual has8extra earlier slots for root->run and9extra forrun->step.
- actual tails immediately before11fields include error_channels.keys/values/
  length and keyed error_channels; formal ends propagated workspace then11fields.
- _propagate_record_field_demand atfortran_c_shell.py8169 blindly appends missing
  record field metadata identities, then _harmonize_call_argument_shapes uses
  field-name-only identities and skips ambiguous functions. Possible source of
  duplicate actuals; NOT yet proved/fixed. A safe future direction is preserving
  exact callee-formal IDs on native calls when constructed, updating that receipt
  atomically on appended/pruned slots; do not trim8/9operands by position.
- appendcall attrs ssa_deferred_record_row=(239,'Metrics',14), source_effect246.
  Caller record_table.records239 ABSENT. Actual final239unknown after16storage
  args; formal needs14scalar row fields. Native stepcall aggregate publishes
  outputs239,253,263 from callee1537,1538,1539. Trace missing Metrics record
  descriptor at that returned/merged record boundary; do not invent rowfields.

No live compiler/native/test processes left. Game/API remainspaused. Goalactive.

### Exact native input receipts; late record merges (14:26)

Previous goal turn was progress (compiler edits/native checks/full-build evidence).
This turn remains DT-only; no game/API work.

Implemented native Call callee_input_ids metadata at source-call construction in
fortran_c_shell.py. _complete_propagated_frame_tails now reconciles known actuals
by these callee-local IDs and appends only declared workspace needs; preserves
receipt through reordering. _propagate_record_field_demand respects already-bound
formal IDs instead of appending another value merely because actual field-name
metadata is missing. _harmonize_call_argument_shapes consumes exact receipts even
when field names are ambiguous; incomplete exact bindings never fall back to
field-name matching. Both late frame-refresh sites resetreceipt; dead-formal
pruning slicesreceipt withoperands. New tests/test_native_call_input_receipts.py
covers two distinct receivers withsame fieldname, lost actualfieldmetadata,
workspacegrowth, formalreordering, deadformalpruning.
6focusedtests (2receipt+4nativepredicate) PASS30.88s; actual sidechain emitscleanly
build/dt_sidechain_input_receipts_20260905. Existing loop-carried record-return
checkpasses. test_returned_record_fields_feed_structural_call_argument FAILS but
is explicitly documented pre-existing in TEST_BASELINE_AND_HAZARDS.md line123;
do not rebaseline or alteritsassertion. It hasunexplainedpublicinput9 andnoLOr.

Fullbuild build/managed_dt_input_receipts_20260905 (session83504,PID13564) TERMINAL
EXIT1. Do notresume/restart83504. Failed insidefinal _prune_unused_callee_formals
at receiptlengthinvariant for _propose_dt_pen. Foundlater keyed-storage removal
transaction atfortran_c_shell.py~22817 filteringcallargs butnotreceipt. Fixed it
to sliceIDs withsamepositions. Errornowreports owner/callee/lengths/uniqueness.
Added --entry proposal option totools/repro_apply_energy_sidechain.py so real
_propose_dt_pen closureusesmanagedABI/native-onlypolicy. Command--entry proposal
--output build/dt_proposal_input_receipts_20260905 emitscleanly1.08s. (No native
proposal parityrun.)

SavedpriorfullSSA revealedstep's semanticreturn1537 isreturn_merge Phi withTHREE
incoming459 record references, allsameMetrics descriptorfields1648..1661.
Recordtable has28,7,27,459,460 butNOT1537. Metadatarecord_return_layouts only460;
output_identity_aliases maps459/460->1537. Callerthereforegets239unknownscalar,
noMetricsdescriptor, append17/30. Existing materialize_record_phis ranONLYbefore
calllinking created459fields. Now rerunthat existing pass inlink fixedpoint before
latepublicRet expansion; setchangedwhennewdescriptorsregistered. No newrecord
mergealgorithm orguessedfieldlayout. Existing looprecordtest+2receipt testsPASS
2.89s afterthischange. Needactualfullbuildtoverifylateoutputsurfaces/callrefresh.
A tiny makeMetrics/choose2return/rootfield probe stillhasmissing48 andnolayout,
bothwithdefault/fullmanageddeclarations; it doesnotisolateactuallateDTrecordcase.
Do notclaim generalmulti-exitrecordparity.

Otherpriorfullsnapshotdetail: missing450 isjustSSAappendresult, downstreamof
17/30signaturefailure. Missing113 isrootplanned_region0 tensorresultfreshened
from7: functionmetadatafreshened_synthetic_value_ids=((7,113),), tensortable still
has7output, actualbinary_scalar_double writes113 andRet stillreturnsformal7.
Thisisnotjustmissingmalloc; investigatewhetherauthoredin/out aliaswasincorrectly
freshened. Storage solver113hasabsurdlybroadviews124416elements duemalformedcall
shapespropagating, sofixcallidentitiesbeforetrustingthatbound.
planned_region10 actual0/formal76(sequence_arenaTrue) isbool(reasons) path and
Ret76,332. Needsrealresidentsequence/presencebinding, notanarbitraryscalarinput.

CURRENT LIVE BUILD:
python -u build/run_managed_binding_probe.py
Wrapperrunsnormal tools/build_balloon_tire_native.py --managed-dt --batch-size8
--optimization O2 --frames0 --output build/managed_dt_record_merge_receipts_20260905
withactualspaces. It catcheserrors ONLYto saveunfinishedall_functions/tables from
traceback asfailed-link-ssa.pkl (module,{},()), metadata diagnostic_incompleteTrue,
thenre-raisesoriginalerror. NormalcompletedSSA/C artifacts useexistingmanaged
emitter. Wrapperisignoredlocalbuilddiagnostic, notproductionprogram orchestration.
Logbuild/managed_dt_record_merge_receipts_20260905.log.
Session78547,PID5848,start14:26:24,lastCPU20.50s,stage reducing source topology.
NootherPython/nativecompiler beforelaunch. Resume78547; doNOTrestartonobservation
timeout. NofullDTexecutable/parity/performanceyet. Goalactive.


## DT storage correction and isolated sequence truth failure (14:41)

The previous session 78547/PID5848 is TERMINAL exit 1. Do not resume it.
Its completed snapshot is build/managed_dt_record_merge_receipts_20260905/repository-ssa.pkl.
There are now three C emission shortfalls: root -> run_superstep 73/1513,
root planned_region_0 unavailable113, step planned_region_10 0/1.
The previous step-call and record append arities no longer fail emission.
This is emission progress only, not native correctness.

The 1440 missing root-call formals comprise frame storage plus fourteen late
record-result fields. The latter lacked compiler_frame_storage, so the existing
all-tail-storage check refused the entire tail. Late result field construction
now explicitly stamps compiler_frame_storage at its allocation site.

Extracted _intern_writable_region_outputs in fortran_c_shell.py. Besides the
existing written ProgramABI field proof, it accepts a matching tensor descriptor
with storage=output and writable=True, still requiring mutable ProgramABI storage.
This lets the telemetry region retain its exact output7 across the collision
freshener without manufacturing field-name demand. Test verifies no interning
from mutability alone, then exact producer/output operand/Ret identity when the
tensor output proof is present. Three receipt/storage tests passed in2.17s.
A small authored material.telemetry reset emits without shortfalls, but does not
create the same region as full DT and is not the full-regression evidence.

The remaining bool(reasons) failure reproduces with plain native-only source:
    def root(flag):
        reasons = []
        if flag:
            reasons.append(0.0)
        return bool(reasons)
Saved build/sequence_truth_probe.pkl, tuple(module,outputs,exports).
It emits a region0 with formal1(sequence_arena), Cast1->7, Ret1,7. Caller passes
zero args; append uses arena1 BEFORE its later aggregate Load definition.
Simply adding a dummy input or casting the first element would be wrong:
nonempty [0.0] is true. Need lexical resident length query and proper arena
allocation independent of the numerical region. Existing SequenceQueryBlock
is the right mechanism to extend: supports length/first_or_default/lookup.
_install_lexical_sequence_queries currently only recognizes len/next/sum whose
sequence has a producer loop; plain list bool is excluded. precompile_to_ssa
already removes query-result numerical instructions and lowers length-cell reads
at lexical control position. Need proper placement at the authored bool use,
AFTER prior mutations; do not place it immediately after list creation.
No sequence fix made yet. Avoid compiler edits while the live build imports it.

CURRENT LIVE BUILD (supersedes earlier live notes): session90068, PID16520,
started14:39:26. Wrapper build/run_managed_binding_probe.py now selects
build/managed_dt_output_storage_20260905; log same basename.log.
Runs one full managed build, O2, batch8, frames0. Last observed stage selecting
complete control/operator deployment. No other Python/native compiler at launch.
Resume90068, do not restart on observation timeout. It checks the two storage
corrections; the known sequence failure is still expected. No full DT native
executable/parity/performance claim. Goal stays active, all game/API work paused.


## Native sequence truth regression passes; new full build (14:50)

Session90068/PID16520 TERMINAL exit1. Full snapshot at
build/managed_dt_output_storage_20260905/repository-ssa.pkl and module.c.
ONLY remaining C emission shortfall was step planned_region10 arity0/1.
The root call-tail and telemetry113 failures are gone in this full run.

Implemented native sequence truth using existing SequenceQueryBlock:
- control_source admits operation truth and renders length>0.
- _install_lexical_sequence_queries in fortran_c_shell recognizes actual
  builtins.bool on writable List source nodes (not arbitrary objects/tensors).
  Places local list queries immediately before their exact deployment-region
  marker, recursively retaining branch/loop scope, rather than after creation.
  Existing generator queries keep their producer-loop placement.
- precompile_to_ssa removes the replaced query result and its empty-list Const
  from the numerical region. The control builder already owns local sequence
  arenas and length initialization. Query lowering emits Load length then Gt0.
  There is no sequence scalar cast, missing region input, or Python callback.

New tests/test_native_sequence_truth.py compiles actual authored
conditional append0.0 then bool(list) and executes the C DLL in subprocess20s.
First reproduced arity0/1 in3.19s. After query fix, C/DLL emitted but native
assertion failed (24.57s test): C public wrapper allocated arena+capacity as
private root storage, then left capacity zero. The append helper correctly
refused the zero capacity and length stayed zero. prepare_execution feeds
cannot configure these hidden private cells; do not fake public inputs.

ssa_c_backend public activation setup now initializes a PRIVATE sequence's
capacity cell from the minimum actual allocation count of its PRIVATE column
buffers, only when every column allocation is known there. Caller-owned
capacity is untouched. This publishes allocated capacity; it does not prove
sufficient bounds for arbitrary loops or all propagated child sequence frames.
Full DT may expose further allocation/capacity/error-handling defects at runtime.

Six tests passed11.33s: new native sequence truth (flagFalse/True/False/True,
reusing execution, nonempty[0.0] true), three receipt/output-storage checks,
two existing sequence-query scheduling checks in test_fortran_c_shell.py.
No full DT native parity yet. Do not infer full correctness from six tests.

CURRENT LIVE BUILD: session41778, PID19476, start14:49:51, lastCPU21.78s,
reducing source topology. Command python -u build/run_managed_binding_probe.py
redirected to build/managed_dt_sequence_truth_20260905.log; wrapper's output
now build/managed_dt_sequence_truth_20260905. Full managed DT, batch8, O2,
frames0. No other compiler/Python process before launch. Resume SAME session;
no restart on observation timeout. If C complete, this also builds DLL/exe,
so may run longer. Still no native run automatically. Next: inspect terminal
result, then isolated bounded native DT eager parity under same inputs before
performance. Goal active; game/future API work remains paused.


## Full DT emits C; public intermediate inputs remain (15:02)

Session41778/PID19476 TERMINAL exit1. No active full build/native compiler.
Do not resume/restart that job. build/managed_dt_sequence_truth_20260905 contains
repository-ssa.pkl (26920678 bytes), module.c (2193606 bytes), shortfalls.json=[]!
The sequence truth fix removed the last C emission shortfall. Build stopped
BEFORE native compilation at vehicle_python_compilation.py:1063:
managed standalone C material contract has unnamed public buffers: (47,57).
C emission is not native build/correctness proof. Do not fill these with zeros.

Root47 is unannotated float64 formal feeding planned_region3 bool47->50.
Root57 is unannotated float64 formal, value_names completed_window, feeding
planned_region4 float57->58 (telemetry[0] assignment). Root44advanced/45dt_next
are still ssa.aggregate from run_superstep outputs280/255. Ret38state,39output,
44advanced,45dt_next,40last_displacement,41last_velocity. Root46metrics record
projection47 likely hard_failure; needs investigation of result descriptor.

IMPORTANT source-loss evidence: root planned_region1,4..11 show indexed Loads
for authored telemetry assignments and Ret[], no Stores in these region bodies.
Root region0 telemetry reset occurs AFTER run_superstep in current SSA order;
root call order run, region3, region0, region1, region2, region4..11. C emission
success does not make this faithful. Check compiler structural store replacement
before concluding those effects are absent everywhere. Don't patch the source.

Isolated root completed_window reproduction:
    def root(advanced, duration, hard_failure):
        completed = float(advanced) >= float(duration) - 1.0e-15 and not bool(hard_failure)
        return float(completed)
Under native-only ExtractionContract, root signature erroneously has extra12,
only calls planned_region2 float12->13, Ret13. Parameter names0,1,2. Emission
shortfalls empty. Changing ONLY return to completed produces correct root calls
for boolhardfailure and comparison followed by LNot5->6, Select11,6,11->12,
Ret12, noextra formal. So internal BoolOp producer recovery fails at later
numeric call boundaries, not a DT-specific source problem.

Likely fix location: fortran_c_shell.py recover_structural_source_outputs
(~11475). It has ensure_structural_value (~11958), BoolOp selection lowering,
and private feed recovery (~12730) but private recovery loops ONLY over
pending_call_feed_ids (source-linked calls), not planned numeric Call inputs.
Claimability further requires ssa_call_result_from annotation, absent for12/57.
Structural insertions currently placed before Ret; extending roots requires
proper producer dominance before their actual numeric consumers, and may need
recovering their dependency regions/source expressions lost to earlier DCE.
Do not blindly append computation after its consumer or reclassify as workspace.
No fix implemented for these two new public-intermediate defects yet.

Added tools/managed_dt_parity.py (syntax checked only, not executed). It consumes
successful build manifest+initial-state.bin+repository-ssa.pkl, asserts fixture
matches byte-decoded native inputs, then runs executable and eager source in
SEQUENTIAL subprocesses with timeout(default180). Eager uses symbolic laws'
AbstractTensor stage (no native stand-ins) and exact managed source. It compares
all physical public buffers (exact int/bool, float rtol1e-8 atol1e-10), stores
JSON and mismatchNPZ. Native final output removed before run so stale output
cannot masquerade as success. Scope managed DT+tire only, not wholevalidator.
CLI: python tools/managed_dt_parity.py build/managed_dt_sequence_truth_20260905
WILL currently fail missing manifest because native build not reached. Wait for
successful binary build; no need to run to rediscover that. Tool's eager path
and fixture pack remain to be verified. No performance measurements yet.
Goal stays active. Previous turn progress: actual full emission nowcomplete,
new exact compiler defect repro, prepared parity gate. No external blocker.


## Boolean numeric feeds and linked result refresh (15:09)

Previous turn classified progress. No external blocker. Continued after
session41778 terminal, no process overlap.

fortran_c_shell recover_structural_source_outputs now includes exact ast.BoolOp
arguments of planned-region Call instructions among private structural roots.
Such a provisional formal can be claimed without ssa_call_result_from, while
still excluding authored parameters and ProgramABI/compiler/linked storage.
Only these graph-proven BoolOps extend admission, not arbitrary missing inputs.
Initially reconstruction removed extra12 but emitted LNot/Select AFTER float12
consumer. Now recovered instruction dependency closure is placed before its
same-block first consumer when external operand producers already precede it.
Existing calls/mutations are not reordered. This is deliberately a same-block
placement repair; cross-block dominance/source-effect issues remain to audit.
New tests/test_native_boolean_region_feed.py: exactly3publicparameters, actual
C/DLL native6 combinations advanced0/.25/.5 and hard_failure0/1, bound20s.
PASS12.65s. Source is prior completed_window reproducer returningfloat(completed).

Root missing47 traced further: root record46 DOES exist with hard_failure
field274, but274 lacks ssa_call_result_from because original root -> run call
stillpublishes3outputs44,45,46 after calleeRet expands to13physical fields.
Late projection rebind correctly refuses an unproduced resident field. So
simply wiring47to274 would read uninitialized return storage.

Native calls now record native_result_contract=(id,dtype,shape) per callee
output. Linking fixedpoint permits refresh when that exact result contract
changes, even if call record resolution is already native_call. Before replacing,
existing occurrence with same plan_callsite_id is reopened as marker; only
GEPs derived from its exact aggregate result and Loads from those addresses
are marked as placeholder projections for the EXISTING marker replacement
transaction to retire/rebind. No duplicate independent call is appended.
New native_call gets fresh result receipt. Full DT verification pending.

Selected existing test_record_return_call_refreshes_completed_physical_field_surface
(previously baseline failure, now run explicitly because working on this defect)
PASSES. Together with scalar-result-type test and new nativeBoolean test:
3passed12.79s. No native full record-return parity claim; first testchecksSSA.

CURRENT LIVE: session8623, PID2844, start15:08:09, CPU16.50s at lastpoll.
Wrapper build/run_managed_binding_probe.py output selects
build/managed_dt_result_contract_refresh_20260905; log samebasename.log.
Fullmanaged DT batch8 O2 frames0, no other heavy process at launch. Resume8623,
do not restart on timeout. Next confirm missing47/57 gone and native build
actually completes, then run tools/managed_dt_parity.py on successful directory
with bounded native/eager execution. Tool has only syntax verification so far.
Root telemetry source-order/store concerns from preceding note still need
runtime/SSA investigation, regardless of C emission success. Goal active.


## Indexed stores fixed; shared record outputs prevent type convergence (15:20)

Session8623/PID2844 TERMINAL exit1. No full build/native compiler active.
Full managed run failed at fortran_c_shell.py aggregate call-result physical
types did not reach a fixed point (~22175). Wrapper saved
build/managed_dt_result_contract_refresh_20260905/failed-link-ssa.pkl, 2827392bytes,
metadata diagnostic_incomplete. This is NOT a complete emitted module.

Found silent indexed-store loss independently while build ran. Without tensor
provider, both array[0]=value and material.telemetry[0]=value retain Store.
With actual c_backend_repository_ssa_reference used by full DT, same source
became GetElementPtr+Load and value was unused. tensor_ssa_lowering.py's scalar
dropped-axis basic-index shortcut (~1467) ran for basic_index_store too, before
store-specific code. Restricted that shortcut to operation slice/basic_index;
store now reaches existing basic_index_store lowering. New
 tests/test_native_scalar_index_store.py uses the actual provider and mutable
float64 span[2], compiles C/DLL and executes values-3.5,0,12 in subprocess20s,
asserting assigned first element and unchanged second42. PASS13.39s. This was
a real silent DT telemetry-write defect, not merely a diagnostics concern.
No full DT rebuild after this fix yet because identity cycle needs fixing first.

Captured type-cycle evidence in step_with_dt_control_used specialized function:
- coerce_metrics Call outputs1647..1660 from callee68..81.
- balloon_tire_managed_advance Call outputs456,1647..1660 from
  callee323,193,193,154,155,378,379,380,381,382,262,212,261,383,384.
- Both Calls have Loads defining THE SAME SSAValue OBJECTS1655,1656,1660 twice.
  (Not merely colliding integers; generic freshener cannot separate this.)
- Six changes repeat each type round:1655/1660 alternate float64 andnone;
  1656 alternatesint64 scalar andint64 shape(1,). Do not increase iterationbound
  or forceone dtype as a cure for duplicate definitions.
- record459 (advance metrics) and460 (coerce result) both have same fields
  1647..1660. Return Phi1537 has separate fields1740 onward.
- call records: advance site326 results((323,456),(214,459)); coerce site460
  results((3,460),). In saved while_body, coerce Call precedes advance Call,
  another dependency/order problem to investigate along with shared outputs.
- coerce_metrics callee Ret68..81 are EXACT formal IDs, forwarded input fields.
  Its body has Const17/18,GEP14,17->19,Store18,19,Call[0]->9,Ret68..81.
  Args14int64(1),0float64,68..81 typedfields,11unknown. Thus same-record return
  identity/forwarded output handling is central. Existing single-result
  aliased_return_argument_index path (~20213) does not cover multifield record
  forwarding. Need honor field identity without defining each input again as
  an aggregate Load, or otherwise maintain correct distinct producer surfaces.
  Do not merely rebind unproduced fields or hide duplicate definitions.

Attempted smaller real-coerce source make(value)->Metrics, rootmake then
coerce_metrics thenmetrics.max_vel under _base_records native contract. It
fails earlier with ValueError returned record formal lacks receiver field
'error_channels.length'; does not isolate this full-DT cycle. Don't claim a
minimal reproduction passed. No permanent source/test added for that probe.

All tiny probe sessions22979,56257,49582 terminal; native index test25955 terminal.
Current tree includes prior result-contract refresh and nativeBoolean fixes,
plus indexed store correction. No running full job to resume. Next fix the
forwarded record result identity/duplicate-def problem, then one full managed
rebuild. tools/managed_dt_parity.py remains syntax-checked only; no executable
for current program exists, no native DT/frame/performance proof. Goal active.


## Exact forwarded aggregate fields preserve the input frame (15:26)

Previous turn progress: indexed-store compiler correction and captured typecycle.
This turn confirmed all14 coerce_metrics physical output fields are exact callee
formals bound to THE SAME caller IDs1647..1660 (same SSA objects too), so the
second aggregate publication is redundant/invalid SSA, not a new record value.

Added _retain_forwarded_aggregate_storage(call,callee) in fortran_c_shell.py.
Admits only aggregate output receipt with matching selectedcallee IDs, every
callee Ret ID already a formal, complete unique input bindingreceipt, and every
selected outputcaller ID equal to its actual input slotID. Distinctcaller result
IDs or nonformalcallee output fail the guard. Recordsforwarded_output_bindings,
removes aggregate output convention/IDs/positions/slots, and sets Call.res=None.
Call itself and inputs/effects remain. Invoked immediately on new nativeCall
before aggregate unpack generation; exactforwarding skipsduplicateLoads.
This uses existing void/inout backend call ABI; no Python callback introduced.
Captured actual coerce Call satisfiesguard for all14 fields. The captured advance
call is not an all-formal forwarded-return call and retains its real outputs.

New tests/test_native_forwarded_record_fields.py constructs the exact SSA
convention with a callee that MUTATES a scalar input then returns its same2
formals. Verifies differingcaller outputIDs doNOTnormalize, exactsameIDs do,
and nativeC/DLL call still changesfirstinput3->17 and downstreamSubreturns15.
Nativeexecution subprocess20s. Together with existing record-return physical
surface test:2passed12.31s. This proves the mechanism, not full DT native parity.

Potential remaining ordering issue: earlier failed snapshot had coercebefore
advance. Suppressing false publication may prevent erroneous placement, but
actual fullresult needscheck. Source-call fallback placement uses physical
result IDs as consumer anchors when no marker is found; forwardedfields can
be shared inputs, so those must not be mistaken for newlyproduced dependencies.
Don't assumeorderingfixed from helper test. Also retainedoptional dtypeNone
storage semantics need realparity, notforced float/int type choices.

CURRENT LIVE BUILD session86224, PID15832, start15:25:42, lastCPU14.36s.
Wrapper build/run_managed_binding_probe.py output now
build/managed_dt_forwarded_fields_20260905; log samebasename.log.
Fullmanaged batch8 O2 frames0 includes prior indexed-store fix and newfield
forwarding. No other compiler/Python process at launch. ResumeSAME86224; don't
restart on observation timeout. Goalactive, no native DT executable/parity yet.


## Forwarded marker fallback corrected; C snapshot complete (15:39)

Session86224/PID15832 TERMINAL exit1. No longer live. Full lowering now passes
its result-type fixed point and saves complete repository SSA in
build/managed_dt_forwarded_fields_20260905/repository-ssa.pkl (26924324bytes).
Only C emission shortfall: coerce_metrics returns14nativeoutputs without a
matching caller result record. Inspection shows Call has correct all14
forwarded_output_bindings, no aggregate convention, BUT res was restored to
SSAValue459 dtypeaggregate. _retain_forwarded_aggregate_storage had setresNone;
replace_at_callsite_marker's scalar fallback (~18475) then restoredmarker_result
to the last Call because aggregate_sequence=False. That's the defect.

Source fix: marker-result scalar rebinding now excludes a sequence containing
explicit forwarded_output_bindings. The call still executes, without result.
Diagnostic proof: loaded the full snapshot, clearedres on the ONE forwarded
Call with incorrectly restored result, re-emitted C. shortfalls=(). This did
not rebuildfromsource and is not native proof. No modifiedsnapshot/executable
saved by that probe (session38683 terminal). Exact rootformals check against
parameter_names, ProgramABI parameters, and is_compiler_owned_storage found
no unbound candidates in same snapshot (session75054 terminal). Full actual
material-feed gate still needsfreshbuild.

Read-only source ordering probe (actual repositorytensorprovider):
array[0]=value; result=read(array); array[1]=result; returnarray retains ordered
region0 Store, source-linkedreadCall, region1Store. So don't assume a universal
store/call schedulerbug merely from earlier DTrecordstate issues. Probe38486
terminal, no nativeexecution/test added for it.
Tiny identity(metrics) -> return metrics, root identity thenmax_vel does not
exercise all-fieldforwardedmarker: nativecallres1, noforwardedreceipt; compile
probe49531 terminal. Don't claimthat coveredthisfix.

CURRENT LIVE BUILD session76288, PID16400, start15:38:52, lastCPU13.91s.
Wrapper output build/managed_dt_forwarded_marker_20260905, log samebasename.log.
Fullmanaged batch8 O2 frames0. Onlyheavyprocessatlaunch. Resume76288; don't
restartonobservationtimeout. Freshbuildmust verify Ccomplete, namedmaterial
buffers, then actualnative toolchain. Afterterminalsuccess use bounded
 tools/managed_dt_parity.py withsuccessfuldirectory; eager/nativepathnotyet
verified. Goalactive; no fullDTexecutable/parity/performanceclaim yet.

### Fresh source emits complete C; wrapper output allocation (15:55)

Session76288/PID16400 is TERMINAL exit1. Fresh full source lowering saved
repository-ssa.pkl (26923850 bytes), module.c (2176509 bytes), shortfalls.json=[]
in build/managed_dt_forwarded_marker_20260905. Wrapper rejected public44/45:
these are actual Ret outputs advanced/dt_next, not unbound input formals.
compile_balloon_tire_managed_python_native now allocates statically shaped
returned output buffers only when they are not root arguments; missing inputs
remain errors. Manifest labels output role and return index. Parity tool now
captures eager returned values as well as mutated inputs.
Native compilation in progress session44475/PID18872 via ignored
build/resume_managed_native.py. Re-emits the unchanged complete fresh SSA and
uses corrected normal compile wrapper; no modified SSA/physics or component
orchestration. Do not start another build. No native/parity success yet.
Added tests/test_managed_native_output_contract.py, not run yet.

Ordering diagnosis while toolchain runs: build/probe_managed_order.py is a
small source-lowering probe. Direct source call and Python-bound direct call
both preserve reset -> advance -> report ordering. Changing advance to take
an authored callback produces reset -> report -> specialized advance in root
SSA. Session49099 terminal; log build/probe_managed_order.log. No native run
of this probe and no ordering source fix yet. Full DT C has a different
manifestation (run_superstep before telemetry-reset region); investigate
specialization/callsite anchors and region composition, not broad sorting.
Wrapper tests 2passed3.81s. Native session44475 still live as of16:01; zigPID11188
CPU381.50s, working set~508MB. Do not restart/overlap heavybuilds.

### Switch to correctness build before optimization (16:16)

Session44475 is TERMINAL exit1 due explicit cancellation of its zig optimizer
PIDs11188/19176 (~1178CPU seconds, ~2GB). It did NOT report a compiler error
before cancellation. Changed sequence because emitted C already has known
ordering defect: debug native execution should precede expensive optimization.
No competing build overlapped. Do not interpret this as optimizer failure.
Current build session66549, ignored build/resume_managed_native.py now O0.
Same unchanged full source-lowered repository-ssa.pkl is re-emitted; saves
native-artifact.pkl for later packaging reuse. Then normal corrected wrapper
creates executable/manifest. Still no native run/parity yet.

### DT native executable built successfully (16:18)

Session66549 TERMINAL exit0. O0 complete source-lowered DT+tire program compiled
through corrected wrapper to balloon_tire_managed_native_c.exe in
build/managed_dt_forwarded_marker_20260905. Manifest now exists with output44/45
roles/return indices. Cached native-artifact.pkl is from the unchanged full SSA.
This is actual native toolchain success, not only C emission. O2 was cancelled,
so no optimized-performance claim. Full vehicle validator still not established.
Parity session64383 active: tools/managed_dt_parity.py <directory> --frames1
--timeout180. Native then eager sequential, initial fixture equality required.
No parity outcome yet; resume64383. No other build running.

### First execution timeout; smaller-window diagnosis (16:22)

Parity session64383 TERMINAL exit1: actual O0 native frame exceeded180s and was
killed by subprocess timeout. Eager worker was not reached; no parity report,
no evidence yet whether slow unoptimized stepping or incorrect control flow.
Also caught artifact.buffer_shapes inflating returned scalar44/45 to
(3,144,288). Root Ret SSA shapes are correctly(). Wrapper now allocates public
outputs using returned SSA shape, as input packaging already respects actual
feed shape over internal inferred workspace capacity. Regression updated with
oversized artifact estimate; needs rerun.
Microstep build session7130 active via cached unchanged native artifact,
output build/managed_dt_debug_microstep_20260905. Same full DT program andbatch8,
only window_duration=dt_initial=2**-20 runtime inputs; O0, corrected scalar
output buffers. This is a narrower correctness diagnostic, not full-frame
verification or performance evidence. After terminal build, parity tool there
with bounded timeout. Original full-window artifact retained.

### Eager terminates; native cap is unproduced; scalar ABI fix (16:40)

All earlier execution jobs terminal. Microstep parity14237 timed out native60s;
eager was then run separately session65372, terminal0. Identical fixture/source:
setup19.527454s, actual eager stepping0.359702s; advanced=2**-20, next~2**-20/10,
one attempt, zero rejection, completed_window1. So this configuration is not
itself nonterminating in eager. No native/eager parity or performance claim.
User reminded us of slow Python setup, possible pathological configs, and the
contract's symbolic-value policy. Setup is outside native subprocess timing;
inspect contracts/producer binding before blaming configuration or changing it.

Ignored build/trace_managed_c.py instruments a COPY of C, not a fresh compiler
product. First trace52680 timeout20s showed repeated superstep step calls.
Narrower trace34813 TERMINAL (script0, tracedexe3) showed:
LOOP total=0 cap=0; STEPdt0; ADVANCEdt0; step returnednext0 used0; then an alignment
panic: root reporting region4 loaded bool input57 asdouble. Instrumentation
changes stack layout, so do not assert exact iteration counts from these runs.
Native originals remain in separate files. Trace paths in microstep directory.

Root cause of bool width reproduced in tests/test_native_scalar_storage_dtype.py:
rankless helper formal explicit physical_dtype=float64 + indexed input causes
array-type union to absorb a scalar bool caller. Failed11.20s (true read as5e-324).
ssa_c_backend.py now unions physical types only when ACTUAL caller has authored
array contract; a helper's formal array use cannot widen scalar caller storage.
Scalar conversion then remains separate. New regression + expanded existing
boolean tests (generic/repository provider, return/store-to-span) + wrapper tests:
7passed50.01s. New Cbackendfix NOT yet re-emitted into full DT artifact!
Cached native-artifact.pkl contains OLD C; explicitly re-emit before next build.

Cap inspector build/inspect_dt_cap.py/log: run_superstep formal175 has
ssa_call_result_from=(...run_superstep...__planned_region_5,175). While Phi350
initialinput175, updated253. Caller binds it to private zeroed160 with no producer.
Region5 exists in module and computes minimum(capture48,capture53)->175, but
run_superstep contains NO call to it. Several source conditionals specialized;
likely lost selected/fallback value after conditional pruning. Active execution
contract is abstract_tensor/all_numeric; fields/input symbolic ABI is involved.
Do not constant-patch175 or change runtime configuration to hide it.
New targeted native source regression tests/test_native_pruned_cap_initialization.py
currently running session4863 (None cap branch, retained initial dt across loop).

### Symbolic cap ordering reproduction and fresh rebuild (17:01)

Prior sessions 4863, 32176, 19293 and 88450 are terminal. Scalar ABI refinement
now resolves stale same-ID operand copies through the caller's formal/definition
before physical type union and conversion. Both native scalar storage tests pass.
The managed_dt_scalar_abi_20260905 artifact predates this refinement; do not use
its generated C as evidence of the current backend.

Minimal source while reproduction showed initial cap bounds AFTER while_exit:
region0 fused initialization with the repeatedly evaluated predicate; overlay
claimed regions(0,3) and inserted the loop before intervening bounds regions1/2.
_control_partition_keys now distinguishes while predicate syntax from pre-loop
initialization using authored AST signatures and condition-node membership.
Native cap regression: four combinations (pruned None/runtime numeric bounds,
with/without while) pass; outputs explicitly exposed through watch. Runtime
numeric bounds are 0/+inf, no symbolic value was replaced by a fixture constant.
Combined bounded checks: 7 passed, 1 failed in45.84s. Failure is the existing
physical mask test's obsolete typed C signature assertion (backend uses void*).
Control source tests: 20 passed in2.04s. Scalar storage tests are included in7pass.
Generated minimal C now computes max/min before entering the while loop.
Remaining observation: nested next_cap uses initial cap rather than carried Phi;
this one-iteration regression does not prove repeated-call carried binding.

Fresh FULL SOURCE DT+tire lowering/build is ACTIVE session43311, O0, batch8,
window_duration=dt_initial=2**-20, directory
build/managed_dt_predicate_partition_20260905. Driver:
python -u build/run_managed_binding_probe.py. No other build/native job running.
This incorporates current planner and C backend; saved older SSA cannot validate
planner changes. After completion run tools/managed_dt_parity.py on this directory
with one frame and bounded timeout. Script now runs eager even after native
failure, recording both outcomes. No successful full DT native parity yet and
no optimized timing; complete validator is still the eventual goal.

### Fresh build and bounded parity result (17:10)

Full fresh source build43311 TERMINAL0, complete C shortfalls=[], O0 executable.
Parity67705 TERMINAL1: native timed out60s; eager returned0, setup15.446909s,
stepping0.276230s. Report/logs in managed_dt_predicate_partition_20260905.
No build or native execution remains live. This is NOT successful parity.
Full generated C still uses formal174 as initial cap, and computes region6
(minimum of lower-bound result48 and upper limit53) after while_exit. Predicate
partition fixed the minimal example but does not resolve this full-program
ordering defect. Do not report DT as working or benchmark this binary.

Expanded minimal runtime bounds repeated inside the while: passed10.22s,
session58263 terminal0. Source test currently parametrized repeat_bounds,
through_loop, runtime_limits; irrelevant repeated-bound combinations skip.
Predicate source signatures now exclude locationless AST helper tokens,
consistent with branch ownership logic (final small refinement not yet retested).
Known separate nested-call carried-value issue remains uncorrected. Next reduce
full cap ordering further or capture its exact planned control/region ownership.

17:12 final focused check: session49988 TERMINAL0, 25 passed / 3 intentionally
inapplicable parameter combinations skipped in39.05s. Covers five native cap
cases and20 control-source checks after filtering locationless predicate syntax.
No live jobs. Full-program cap ordering remains unresolved; goal remains active.

### Exact atomic-control ordering defect and fix (17:22)

Previous goal turn was progress (source fix, native rebuild/parity evidence).
This turn captured the full run_superstep graph and overlay before SSA via
build/capture_dt_overlay.py, session15882 terminal0. Snapshot build/dt-overlay.pkl,
logs build/dt-overlay.log and dt-overlay-inspection.log. Capture script currently
predates region_dependencies keyword; update signature before reusing it.

Runtime flat order was (0..14), while owned3,4,7..13, cap producer6 outside.
Region6 -> region7 is a real data dependency; flat order satisfies it, but the
whole while replaces first body region3 and executes before6. This is distinct
from the earlier initializer/predicate fused-region defect. Numeric min node175
is authored line42, while beginsline81. It is not a symbolic constant0.

control_source.order_control_region_dependencies now reorders SequenceBlock
siblings against the original region DAG AFTER overlay; moves prerequisites on
demand before consuming atomic controls, leaves them in their lexical scope,
keeps unrelated post-loop reports after the loop, and rejects control-group
cycles. _topological_region_schedule publishes order+edges; both deployment
paths pass these edges into _overlay_control_or_require_subdivision.
Tests test_control_region_dependencies.py (3) + existing control_source (20) +
native cap matrix (5) passed28, skipped3 inapplicable combinations,39.52s;
session24126 terminal0. No claim about full native behavior from these tests.

Fresh full source DT+tire build ACTIVE session11416, directory
build/managed_dt_atomic_control_order_20260905. O0,batch8,window=dt=2**-20.
No other heavy build/native job. Driver build/run_managed_binding_probe.py.
After completion run bounded parity on this directory, not previous binaries.

Read-only follow-up while11416 builds: likely cause of stale loop-call inputs
is final native frame argument refresh in fortran_c_shell.py (~22320 and~22790).
Both rebuild instruction.args from original record.frame_bindings via global
final_frame_value/cleaned_frame_value, overwriting marker-bound carried Phi.
Need a multi-iteration native source regression (cap starts dt/4, next_cap*2,
total accumulates until dt, expected1.75dt vs stale initial gives1.25dt), then
preserve exact callee_input_ids resident arguments through late refresh where
valid. No fix applied yet; do not claim proven cause until regression checks it.

### Cap producer fixed in full C; remaining step return defect (17:34)

Fullbuild11416 TERMINAL0, managed_dt_atomic_control_order_20260905, shortfalls[].
Generated run_superstep C now calls planned_region6 in entry BEFORE while_header;
carried cap uses produced callout175_0, not an unproduced formal. This validates
atomic control prerequisite ordering in the actual full DT source build.
Parity45565 TERMINAL1: original native timed out60s. Eager exit0, setup15.479724s,
stepping0.276441s. No successful native/eager parity or performance result.

Instrumented COPY built/run via updated build/trace_managed_c.py,48147TERMINAL0.
trace-native.log: LOOP cap=2**-20, STEP dt=2**-20, ADVANCE dt=2**-20, then
STEP RETURN next=0 used=0, traceexe exits0. Stack layout differs, so this is
provenance evidence only, not an execution fix. Original executable remains hung.
Generated full C shows real uninitialized output path:
- step if_merge_19 assigns return t1538=callout593_0 andt1539=callout330_0
  then branches function_exit (module.c about2478).
- their producer regions9/45 plus _apply_energy_sidechain and update_dt_max
  occur only in L_impl_unreachable_return_control AFTER that return (about2482).
- pi_update and _propose_dt_pen are also inside an apparently constant-false
  if_true_19, while if_false uses callout582_0. Need inspect source control/
  callsite/return placement; do not initialize locals to0 to conceal it.
This is now the next full DT correctness gate.

Separate carried-call bug was reproduced natively in new
 tests/test_native_loop_call_argument.py: expecteddt*1.75, gotdt*1.25
(session86189 failure10.34s). _resident_call_inputs now preserves exact live
formal-to-actual bindings during TWO late frame refresh passes in
fortran_c_shell.py, instead of reconstructing them from pre-loop frame IDs.
New native regression + native_call_input_receipts + native_forwarded_record_fields:
5passed17.68s,39229TERMINAL0. This source fix postdates the full snapshot/binary,
so it has NOT yet been validated in full DT. No live builds/tests/native jobs.

### Return-site diagnosis and source fixes (17:54)

Prior turn was progress (fullcapfix verified, carriedcallfix native regression).
Full step overlay capture71021 TERMINAL0, snapshotbuild/step-overlay.pkl and
step-overlay-inspection.log. Capture driver now targets step_with_dt_control_used,
accepts region_dependencies keyword, saves8tuple(G,regions,reductions,loops,
conditionals,nesting,result,kwargs). This predates current return fixes.
The overlaid full step contains unconditional return460,593,330 BEFORE regions45
and9. This confirms misplaced return already in planner control, before SSA.

loop_composer.py fixes: source_order_walk uses postorder for a Return.value
subtree (Tuple must follow its element computations). collect_loop_controls
retains tuple exits whose container was reduced away but exact return_slot_values
survive, using a surviving slot anchor. Return placement waits for all available
slot positions. precompile_to_ssa dependency_signature now exposes return slot
and guard value inputs, allowing trailing source calls needed by return to move
before the function exit. No initialization-to-zero workaround.

New tests/test_native_return_after_call.py: direct scalar/arithmetic examples
passed; tuple return with dynamic-loop fallthrough hung natively before fix
(65922timeout20). After tuple-anchor retention, C correctly refused missingt20
(64095), identifying trailing return-only call. After return dependencies,
scalar+tuple passed2 in17.56s (36455). Dynamic guard variant then failed when
branch skipped: post-if finish consumed branch output11 instead of mergedPhi15.
Test fixture needed every formal dt_max field ID fed (initial variant only fed
one duplicate); after correction still failed false arm withNaN (1624).

emit_plan_callsite now replaces non-dominating argument by a UNIQUE dominating
conditional_carried Phi explicitly consuming the same SSA object. Existing
resident bindings untouched; multiple candidates emit named shortfall rather
than guess. Four return/native conditional cases + carried-call native + loop
break-control + post-loop-break-port tests passed7 in39.92s (73613terminal0).

One additional direct-truth guard variant is now ACTIVE test session31465 is
OLD terminalfailed; current new direct-truth session is recorded in tool output
following this note. No fullbuild/nativeparity since these source fixes yet.

17:55 correction: direct-truth test20540 TERMINAL0,2passed17.61s. All tests
and capture sessions before this time are terminal. New full source build
in build/managed_dt_return_order_20260905 is ACTIVE (session in next note).
It includes prior resident-call refresh plus all return/conditional-call fixes.

17:55 full fresh build is session75251, ACTIVE, O0,batch8,window=dt=2**-20,
managed_dt_return_order_20260905. No overlapping heavy/native/test job.
Poll75251; if terminal0 inspect cap/return producers in module.c and run
python -u tools/managed_dt_parity.py build/managed_dt_return_order_20260905 --frames 1 --timeout 60
Do not run old parity binaries as proof of these source fixes. git diff --check
on the changed compiler files returned clean. Full DT correctness still unproven;
complete Python-authored vehicle validator remains the eventual goal.

2026-09-05 continuation correction after return-order build:
Build75251 is TERMINAL0. Parity6606 is TERMINAL1: native and eager both
exit0 without timeout, but 11 buffer comparisons fail. Artifact directory:
build/managed_dt_return_order_20260905. O0,batch8,window=dt=2**-20.
Eager setup16.650621s, stepping0.276684s; no optimized/native timing claim.
Native advanced/dt_next remain0; material max displacement/velocity are NaN.
Details: build/return-order-mismatches.log and artifact managed-dt-parity.json.

New check_definition_dominance in src/compiler/ssa_self_check.py has three
focused passing tests (tests/test_ssa_definition_dominance.py,3passed1.87s).
It reports structural CFG dominance, not memory initialization; constant
branches are not pruned. Thus syntactically reachable while-True exits need
separate runtime-reachability interpretation. Normal return producer ordering
findings from the older artifact disappear. Entry instruction238 in exact
step_with_dt_control_used__specialized_d8d399b621bf is coerce_metrics call460:
call-table enclosing_loop_ids=(616,), but Call is in entry and reads advance
fields1649..1660 defined in while_body. This is not a symbolic zero fixture.
Active contract confirmed program_extraction.yaml + vehicle_full_native_execution
with abstract_tensor/all_numeric and explicit runtime window_duration/dt_initial.

build/probe_forward_loop.py is an ignored exploratory reduction, NOT a passing
regression. Local normalize with explicit Metrics ABI appears after its first
consumer; real coerce_metrics with empty channels rejects missing receiver
error_channels.length, with nonempty channels rejects undefined return field25.
These reductions have not yet reproduced exact full-program entry relocation.

ACTIVE diagnostic lowering session9550, script build/trace_coerce_placement.py,
output build/coerce-placement-trace.log. sys.settrace watches marker/loop-anchor
functions for exact call460 in step. No native run or C compilation is launched.
Poll this existing session; do not restart it on observation timeout. Need inspect
MARKER RETURN/INPUTS lines to identify why lexical loop ownership was lost.
Goal remains full validator; DT parity prerequisite still FAILING. No commit/push.

Trace9550 TERMINAL0 (LOWERING COMPLETE). See build/coerce-placement-trace.log.
Contrary to earlier relocation suspicion, FIRST replace_at_callsite_marker for
coerce460 returnsFalse (no marker; last iterated block if_merge.20 is NOT its
location). insert_at_loop_anchor alsoFalse. NEXT link refresh finds the existing
Call at entry245 and replaces it there without relocation(consumers=[]).
Thus missing scheduled marker/anchor predates initial fallback insertion.
Need inspect _schedule_loop_callsites output and subsequent structural passes,
not simply disable marker relocation. No production compiler edit made from
unproven relocation theory.
Reduced root(dt,metrics)->while->normalize(metrics)->return max_vel with explicit
Metrics ABI rejects undefinedoperand6; build/probe_forward_input.py records that
separate single-field forwarding seam, not yet exact missing-marker reproduction.
New driver build/capture_step_call_schedule.py wraps only the scheduler (no global
sys.settrace overhead) and saves exact control/plan/signatures/result to
build/step-call-schedule.pkl for cheap local replay of the scheduling defect.
ACTIVE capture86652 writes build/step-call-schedule-capture.log; expected saved
build/step-call-schedule.pkl. This is full lowering only, no native/C build.
Trace9550 is terminal; no overlap. Poll86652 next.

18:45 capture86652 TERMINAL0. Exact scheduler input/output saved in
build/step-call-schedule.pkl (433754bytes), replay via
python -m build.inspect_step_call_schedule. Output in
build/step-call-schedule-inspection.log. Plan DOES contain coerce460 with
binding459->460; scheduled marker lives root/2/loop616/1. Thus marker disappears
AFTER scheduling, BEFORE initial link, not absent from hierarchy. Also advance326
is wrongly at root/2/loop616/29, after coerce. Hierarchy anchors advance on region9
which the control overlay places late. Conditional barriers prevent exact
call dependency459 from repairing this cross-region ordering.

Implemented ConditionalBlock dependency signature in precompile_to_ssa:
known arms remain atomic, publish only explicit merged aliases, expose external
inputs; unknown arm content remains a barrier. New
 tests/test_call_order_across_conditional.py reproduces this ordering seam:
failed before (producer index3,consumer0), passed after.
Focused file + test_precompile_to_ssa:76passed5failed4.84s, session25573 terminal1.
Baseline disabling ONLY new conditional signature in memory:75passed SAME5failed
2.31s; build/check_scheduler_baseline.py. Failures are old mean/Fill expectation,
index dtype, record getattr field capture, nested row alias-accounting equality,
and nested loop result alias-accounting equality. Do not call entire file green.

Full saved-plan replay with change STILL WRONG: coerce460 moves to loopindex12,
advance326 remains29. Another barrier/cycle remains. No fresh full compile or
native parity after this change. Use cheap saved-plan replay to inspect exact
barriers before next rebuild. New function needs guard coverage (unknown arms,
private branch outputs and predicate expressions) before expanding it further.
All captures/tests terminal; no live jobs at this note. Full DT parity still11
mismatches; full validator unverified. No commit/push.
