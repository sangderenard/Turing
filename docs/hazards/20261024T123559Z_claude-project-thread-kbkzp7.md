2026-10-24 LANE E, SOURCE >> SIGN-FILLS (MEASURED, cloud): Python's right
shift is arithmetic, but C cast Shr's left operand to unsigned and LLVM
emitted lshr, so (n - 100) >> 2 at n = 0 gave 2**62 - 7 on both (silent WRONG
in the reference backend; Fortran's shifta was right; c410). Shr now
sign-fills in C (no signed >> of a negative value is left to the compiler)
and LLVM (ashr). The lifter, the x86 binary equivalence tables and the LLVM
importer used Shr for the zero-filling machine shift; they emit a new
Handler.LShr, so Shr has one meaning everywhere (SAR rows name AShr). LShr
lowers on C, LLVM, Fortran (shiftr), SPIR-V, the reference evaluator and the
Python materializer. Measured at main ae465796 + fix (361 programs): C 360/360
(c428 WRONG on main too), LLVM 358/361 (c367, c398, c428; same on main),
Fortran 349/361 (same names as main); c410 passes on all three. Fuzz seed 69
(30): 0 WRONG / 0 TYPE. Machine, binary, x86, wasm, LLVM-import and Fortran
suites: same failure names as main (117). Gate names identical to main
ae465796 in all four parts. On fresh main 8fb85e7b every LLVM corpus program
fails to load on Linux (Windows build target default, 31e0fdf7; separate fix).
