2026-10-24 MAX MEANINGS: BUILTIN max(x, y) ON TENSORS REFUSED, emax/emin, NaN RULES (MEASURED, cloud Linux, base 8fb85e7):
the planner stamps `max_semantics` on every authored max/min from its call
form (select, scan, reduce, reduce_axis, elementwise, emax); C, LLVM,
Fortran and SPIR-V spell Python's builtin rule for select/scan/emax and
numpy's NaN propagation for reduce/elementwise. Builtin `max(x, y)` of
multi-element tensors is refused (Python raises); `x.emax(y)` is the
builtin rule per element. Also fixed: LLVM `!=` and float truthiness were
ordered compares (`nan != nan` False); `maximum`/`emax` over runtime-sized
spans returned one element. Gate vs a main worktree at 8fb85e7: same
failure names (aa 77 / ab 7 / ac 25 / rec 20). Typed corpus
C/LLVM/Fortran: same as main plus c570, c571 passing on all three (both
silently wrong on main). HAZARD, not this PR's: since 31e0fdf the default
build target is windows-amd64 and LLVM passes `-target x86_64-windows-gnu`,
so on Linux every LLVM product is a PE file ("invalid ELF header"); both
the gate and corpus above were measured with a local, uncommitted
workaround applied identically to both trees. Not green: reduce_axis
loops; `(y > x).where(y, x)` refused on C, wrong on LLVM.
