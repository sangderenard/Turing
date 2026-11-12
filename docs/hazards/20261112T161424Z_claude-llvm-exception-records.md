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
- Measured with a cold cffi accelerator build (see the next entry), under
  tests/test_llvm_source_exceptions.py, 5 passed:
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

2026-11-12 The "nondeterministic" C exception tests depend on the cffi accelerator build (MEASURED, clean detached worktree at 15fe1f3e):

- `tests/test_source_c_exceptions.py::test_exception_handler_does_not_catch_base_exception_itself`
  passes when `src/common/tensors/accelerator_backends/__pycache__` is absent,
  i.e. on the run that builds the cffi module.
- On the baseline it failed every time it loaded the prebuilt module: 14 of
  14 repeats, PYTHONHASHSEED 0-3 alike. The failure is FortranEmissionError
  "dispatcher operation's control owner is not retained"
  (fortran_c_shell.py:16757, `_install_external_reference_calls`).
- `repo-cache/` does not matter.
- So most "flaky" readings are cold-versus-warm accelerator state.
- Not explained: one pass in four warm runs on this branch's worktree.
  Nothing removed the cache before that run.
- With a warm build, the same refusal blocks `if n < 0: raise`, try/except
  ZeroDivisionError, and BaseException handler programs.
- Not traced: which accelerated function's answer differs from its Python
  form.
- To measure exception tests until then: delete that `__pycache__` and run
  one pytest process.

Regression check of this branch (MEASURED; same command on both trees, each in
one pytest process started with a cold cffi build):
`pytest tests/test_source_c_exceptions.py tests/test_linked_accepted_c_emission_owner.py`
- Clean main at 15fe1f3e: 34 failed, 13 passed (7 min).
- This branch: 24 failed, 23 passed (8.9 min).
- The 14 failure names captured from this branch's run all appear in main's
  list. The output was truncated, so the other 10 were not seen.
- One process builds the accelerator once and then reuses it, which is
  consistent with the cold/warm dependence above. The C exception file has
  no stable baseline until that is traced.
