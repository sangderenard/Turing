2026-11-12 LLVM materializes the SSA exception record operators at parity with C (MEASURED, branch claude/llvm-exception-records from main 15fe1f3e):

- Before: an exception program emitted to LLVM refused every record operator.
  For `try: raise ValueError("authored failure") finally: ...`, each of
  ExceptionNone/Make/Take/Active/Raise gave "operation has no repository LLVM
  emission". Only the C lane materialized these backend-neutral operators.
- Now, `_emit_repository_call_module` handles the `Exception*` family:
  - It reads the same SOURCE_EXCEPTION_OPERATION fact as the C lane, and
    imports an accepted component's cell from its owner book, as C does.
  - It uses the same turing_validation_runtime packet ABI: an entry-frame
    alloca of `turing_error_packet`, zeroed, then `turing_error_packet_init`
    or `turing_validation_error_take_active`. `turing_error_packet_matches`
    and `turing_validation_error_raise_packet` follow.
  - ExceptionActive reads `code` only on the non-null edge.
  - Type and message texts are module constants with the C `_trace_text`
    bytes.
  - It sets `needs_validation_runtime`, so the runtime is linked.
  - It refuses the same unowned payloads with the C spellings: cause, dynamic
    message, constructor payload (field 8 when the record carries it).
  - It posts its policy on the new page `llvm_exception_operation` (same row
    and value shape as `c_exception_operation`).
  - The audit reads both pages with one check; a mismatch is reported as
    `llvm-exception-provenance`.
- Measured under tests/test_llvm_source_exceptions.py, 5 passed (a run in
  which the lowering refusal in the next entry did not occur):
  - try/finally raise (n=4): the LLVM packet is builtins.ValueError
    "authored failure", suppress 0. The C shell reports the same type and
    message.
  - Nested rethrow: 50.
  - Raise from None: 2 for n=1; for n=-1, builtins.KeyError "hidden context"
    with suppress 1.
  - Python, C and LLVM agree on every case. The concordance has 0 findings
    after both emissions.
  - A forged LLVM policy row is reported as `llvm-exception-provenance`.
- Not covered by a test: the refusal branches. Both candidate programs stop
  on main before emission:
  - The f-string message hits `physical-call-formal-unproven`
    (append_decimal_integer_glyphs, `missing_unique_span_abi`). The C test
    `test_standalone_raise_retains_dynamic_message_as_graph_operand` fails the
    same way on clean main.
  - `raise ... from error` with no return fails at "emission_artifact: no
    source cell for buffer_order", on C and LLVM alike.
  - With `return n` it hits the hazard below.
- Host reporting is unchanged and equal across lanes. The in-process hosts
  (`LLVMExecution.run`, C `ssa_c_backend.py:1305`) name a fault from its code
  through `native_faults.validation_fault`. A raised record's default code
  100 is also BROADCAST_INCOMPATIBLE_SHAPE, so in process both lanes report
  "incompatible broadcast shapes". The packet itself carries the right type
  and message.

2026-11-12 "dispatcher operation's control owner is not retained" is a structurally decided If still projected as a guard (TRACED, clean detached worktree at 15fe1f3e):

Program: the C test `test_exception_handler_does_not_catch_base_exception_itself`
(`try: raise BaseException("base") / except Exception: ... / except BaseException: ...`).
Exception ingestion writes each handler's test as a chain:

    __turing_exception_4 = False
    __turing_exception_15 = not __turing_exception_4
    if __turing_exception_15:                  # If 28: predicate is constant True
        __turing_exception_16 = __turing_exception_active(__turing_exception_2)
        __turing_exception_15 = __turing_exception_16
    if __turing_exception_15: ...              # If 43 (match)
    if __turing_exception_15: ...              # If 45 (handler body)

Chain, each link observed:
1. `_fold_callsite_structural_values` (glsl_deployment_strategy.py) folds
   If 28's predicate (node 27) to a Constant. It selects `body`.
2. It posts `source_control_specialization_concordance[('root', 28)]` =
   `{selected_role: 'body', rejected_role: 'orelse', proof: 'structural_constant_predicate'}`,
   adds 28 to `structurally_specialized_conditional_node_ids`, and removes
   the If node with its local `remove_node`.
3. `_ordinary_conditional_control_programs` skips every structurally
   specialized record, by design ("reconstructing this retained source record
   would put that producer back behind a runtime branch"). So no
   ConditionalBlock 28 is ever built; 43 and 45 are.
4. `control_activity.memberships` still projects the `exception_active`
   inside If 28's body onto owner 28. It does this through its
   retained-source-control branch (28 is not in the graph), without reading
   the specialization record. The result is `{(28, 'body'), (61, 'body')}`.
5. `_install_external_reference_calls` takes the innermost guard, 28, finds
   no block for it, correctly refuses to hoist to 61, and raises
   (fortran_c_shell.py:16757).

The two records that disagree: the specialization record (If 28 is decided,
its body is unconditional) and the activity membership (node guarded by 28's
body). The fix site is the projection in `control_activity.memberships`:
- A guard on a structurally specialized owner's selected arm is not a guard.
  Drop it; the enclosing owners are already in the node's activity fact.
- A resident node in the rejected arm is a contradiction and must refuse.
Not changed yet.

Earlier reading corrected: I first blamed the cffi accelerator build (a cold
`accelerator_backends/__pycache__` passed twice). Hash seed and repo-cache
also do not decide it. The refusal is structural and present in every
failing run. The occasional passes are not explained. The exception
operations' node numbering differs from run to run, which suggests an
identity-ordered iteration upstream, but that is not traced.

Regression check of this branch (MEASURED; same command on both trees, each in
one pytest process started with a cold cffi build):
`pytest tests/test_source_c_exceptions.py tests/test_linked_accepted_c_emission_owner.py`
- Clean main at 15fe1f3e: 34 failed, 13 passed (7 min).
- This branch: 24 failed, 23 passed (8.9 min).
- The 14 failure names captured from this branch's run all appear in main's
  list. The output was truncated, so the other 10 were not seen.
- The C exception file has no stable baseline while the projection above is
  unfixed.
