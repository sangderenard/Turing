2026-10-24 LANE E, SOURCE >> SIGN-FILLS (MEASURED, cloud): Python's right
shift is arithmetic, but C cast Shr's left operand to unsigned and LLVM
emitted lshr, so (n - 100) >> 2 at n = 0 gave 2**62 - 7 on both (silent WRONG
in the reference backend; Fortran's shifta was right; c410). Shr now
sign-fills in C (no signed >> of a negative value is left to the compiler)
and LLVM (ashr). The lifter, the x86 binary equivalence tables and the LLVM
importer used Shr for the zero-filling machine shift; they emit a new
Handler.LShr, so Shr has one meaning everywhere (SAR rows name AShr). LShr
lowers on C, LLVM, Fortran (shiftr), SPIR-V, the reference evaluator and the
Python materializer. Measured at main 51330ad8 + fix (382 programs): C 381/382 (c428 WRONG on main
too), LLVM 379/382 (c367, c398, c428; same on main), Fortran same 18 names as
main; c410 passes on all three. Fuzz seed 71 (30): 0 WRONG / 0 TYPE. Machine,
binary, x86, wasm, LLVM-import and Fortran suites: same failure names as main.
Gate names identical to main 51330ad8 in all four parts (aa 77 / ab 7 / ac 26 /
rec 20). New x86 rows BTS/BTR r64 imm8 name LShr like the other bit tests.
