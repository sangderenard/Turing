2026-11-12 RAISED EXCEPTION CONSTRUCTOR PAYLOAD AS EXACT GRAPH OPERANDS (MEASURED in worktree C:/dev/Powershell/turing-exception-payload, branch claude/exception-payload-operands, base main 85faf666):

Program: string_core acceptance/native_string_program.py through tools/compile_native_string_program.py (full-native
execution). Token Orchard's api.py `raise StringCoreError(status, byte_token, text_token)` stopped the whole program at
ingestion in 0.5 s: "native try: dynamic exception payload ownership is not lowered" (source_exception_ingestion.py,
raised_record). Any raise with more than one constructor actual or a keyword actual was refused before graph ingestion.

Rule now (the pattern the dynamic message already had): each constructor actual of a raised class is an exact graph
operand, positional then keyword; SOURCE_EXCEPTION_OPERATION gains field 8, one (role "arg:i"/"kw:name", operand
position, operand identity cell) per actual, derived from those cells (reducer). Native selection owns the ABI: the C
emitter refuses by name, "raised exception constructor payload requires an owned native payload ABI" (turing_error_packet
holds a type and a literal message only). `**mapping` actuals are refused at ingestion ("unpacked exception constructor
actuals are not lowered"). The identity audit checks field 8 (finding "ssa-exception-payload"). ExceptionMake's
max_inputs is None (it was 1; that limit only feeds ILP metadata). The single-argument forms are unchanged.

Measured:
- tests/test_source_c_exceptions.py new tests (3 payload forms + the unpacked refusal): 4 passed. Each payload form
  lowers, carries one ExceptionMake operand per actual, C refuses by the payload name, 0 concordance findings.
- tests/test_source_c_exceptions.py existing 46: 29 failed / 17 passed on this branch. No failing name is new against
  the base: the three listed as new against an earlier baseline (raise_from_none, zero_iteration_loop_inside_try[-2,0])
  fail identically on a clean detached worktree of 85faf666. That file is largely red on main ("dispatcher operation's
  control owner is not retained", physical-call-formal-unproven on f-string glyph appends). The main checkout passes
  those three only with another session's uncommitted graph_express2.py change.
- tests/test_exception_normal_sequence_binding.py + tests/test_guarded_intrinsic_integer_ranges.py: 16 passed.
- tests/test_process_graph_autograd.py index-adjoint raise test: passed. test_source_control_activity worker-while-try
  (asap, alap): failed, identically on the clean base (FortranEmissionError, control owner not retained).
- string program after the change: ingestion passes; reduction stops after 13 s at Token Orchard's
  `with _RUNTIME_LOCK:` in api.py: MissingResourceControlTopology "native resource scope requires concorded entry, exit and
  cleanup edges" (exception_graph.py:151), the same class as issue 67fcfe276bb9473cb4ab45d45d692215 (`with open`).
  string_core's handoff says that veneer (locks, queue) is never a compilation target: the String component boundary is
  the next step, not lowering the lock.

Not done: no native payload ABI (packet fields, ownership); LLVM has no exception lowering at all (only C does). Whole
suite not run. Hazard noted while baselining: background test runs here notify "completed" late and the task output
file stays empty; four runs overlapped before that was noticed (results above are from single foreground runs).
