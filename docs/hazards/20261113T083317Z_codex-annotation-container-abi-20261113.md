2026-11-13 Input-bounded mapping comprehension results (MEASURED, isolated worktree)

Source comprehension capacity is proved from the current canonical iterator,
materializer publication, immutable control mutation ledger and exact placement
path. One publication per input row is an upper bound, never the written length.
Conditional publication blocks are allowed only through exact single-execution
branch paths; nested/repeated loops, additional writes and unknown arena users
are refused. The result reader joins both descriptor identities and all returned
scalar table columns to that source proof.

C and LLVM wrappers publish only those proof-backed data/length/capacity ports.
The helper formal scalar receipt also retains the exact declared scalar record
field shape proof, rather than copying rank zero while retaining shape (1,).
Native tests exercise empty/no-match/all-match/duplicate keys, repeated full to
empty to full execution, input preservation and retained output arrays.

Verification: final focused candidate 48 passed / 1 failed. Same incumbent plus
native cases at HEAD b46f5210: 31 passed / 3 failed; candidate includes 15 new
source-proof tests. Both new native cases pass versus fail at HEAD; incumbent
public bytes-input test still fails at frame formal reconciliation IndexError.
Mapping audit: 28 rows, 2 functions, zero findings or unsourced facts, CLOSED.
Logs: registry_final_suite.log, registry_baseline.log, registry_mapping.log in
local developer scratch (not repository artifacts).

Actual GraphSolver compile_nodes moves past registry103 capacity refusal. It
now reaches API receipt publication and fails KeyError('uintp'). This does NOT
prove GraphSolver runnable: selected record list still lacks exact public
logical-length/owner-escape ABI. Its pointer-cell shape is not a data extent.
No service changes, deployment, push or merge. No capacity-as-length shortcut.

Helper-shape focused comparison, reserved block only: candidate combined
25 passed / 7 failed vs 24 passed / 8 failed; exact seven incumbent failures
unchanged, C native result is the sole improvement. Baseline restored and
verified. Log baseline_scalar_formal.log. Final mapping log
registry_mapping_final.log.
