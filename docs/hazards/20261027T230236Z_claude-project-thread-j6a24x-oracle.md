2026-10-27 The C lane folds a constant-argument exp() with zig's own libm, not glibc (MEASURED, Linux cloud, ziglang cc)

The JS fuzz (seed 91, js_91_059) found native 5 ulp away from node on
`for (4 times) a *= Math.exp(1)`. The emitted C is right. The cause is the
toolchain: `exp(0x1p+0)` with a literal argument resolves differently from a
runtime-argument call.

- Runtime argument: the library imports `exp@GLIBC_2.29`; exp(1) = 0x1.5bf0a8b145769p+1 (glibc, CPython, node agree).
- Literal argument, -O0: the call binds to zig's static `compiler_rt.exp.exp`; exp(1) = 0x1.5bf0a8b14576ap+1 (1 ulp high).
- Literal argument, -O1 and up: LLVM folds the call with zig's own libm; same 1 ulp high value, no import at all.
- `-fno-builtin-exp` (or `-fno-builtin`) restores glibc's answer at every level. gcc builds agree with glibc.

So any C artifact whose transcendental has a constant argument (after the
compiler's own propagation) may differ from CPython by an ulp, on every
frontend, not just JS. The pinned JS oracle cannot see this: it pins node to
ctypes libm, and the native side is not calling libm.

Proposed fix (C backend owner; not made here): add `-fno-builtin-<fn>` for
the approximated math functions (exp, log, sin, cos, tan, pow, ...) to the
zig cc command in `ssa_c_backend.py`, or pass them from the work contract.
Repro: tests/javascript fuzz seed 91 program 59, or a one-line C file with
`return exp(1.0);` built with `python -m ziglang cc -shared -O0`.
