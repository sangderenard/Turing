2026-10-28 WASM CORE CARRIES PYTHON INT AS I64; PHIS COPY IN PARALLEL (MEASURED, cloud Linux):
Lane X2, resumed and merged with main. On main the independent Wasm core typed int
formals as i32: c961 silently returned the other branch for a = 2**31 and 2**32 + 1.
Int is now an i64 formal, local and output cell; each local carries its kind and a
kind mismatch is a named shortfall instead of a module node rejects. Unproven int
arithmetic still refuses by name (main's python_int_unchecked_refusal, now against
int64). Loop-header Phis were written one by one, so a Phi reading another Phi on
the same edge saw its new value (c125 gave 10 for 9; c962 holds it): edge copies
are now parallel. TYPED_CHECK_BACKENDS=wasm runs the core in the typed check.
Measured: c125, c961, c962 PASS through wasm; the wasm/mobile test files show the
same 101 failure names on main and on this branch (unrelated, pre-existing). Not
gated, no full corpus (per the 2026-10-28 merge flow).
