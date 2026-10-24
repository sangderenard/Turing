2026-10-24 WASM CORE CARRIES PYTHON INT AS I64 — WIP, UNGATED (MEASURED, cloud Linux):
Lane X2. On main the independent Wasm core typed int formals as i32: c601 silently
returned the other branch for a = 2**31 and 2**32 + 1. Branch 18ca38b makes int an
i64, tracks each local's kind, and turns kind mismatches into named shortfalls.
Typed corpus through c,wasm (new TYPED_CHECK_BACKENDS=wasm): main 1/381 PASS;
branch 41 PASS, 335 named refusals, 5 WRONG newly exposed (were invalid modules
before): c124/c125/c126 (loop break target, got 10 want 9) and c427/c537 (crossed or
swapped fields; inferred: edge_assignments writes Phis sequentially, not as a
parallel copy). NOT GREEN: those 5 must be fixed before merge; gate not run (parked).
