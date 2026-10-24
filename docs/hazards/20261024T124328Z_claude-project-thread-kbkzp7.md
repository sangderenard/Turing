2026-10-24 HOST BUILD TARGET DEFAULT (MEASURED, cloud): lowering stamped
WINDOWS_AMD64_NATIVE_LINKER on every module whose caller named no build
target. Since 31e0fdf7 the LLVM lane reads that receipt, so on a Linux host
every LLVM artifact was an x86_64-windows-gnu PE DLL that failed to load
("invalid ELF header"). The unnamed default is now host_build_target():
linux-amd64 (new ELF registry entry, triple x86_64-linux-gnu) on a Linux amd64
host, the Windows linker on Windows and any other host, exactly as before.
At 463e082d (381 programs): C 380/381 (c428 WRONG on main too), LLVM 378/381
(c367, c398, c428), Fortran same names as main. Fuzz seed 70 (30): 0 WRONG /
0 TYPE. Gate vs main c5af44c9: the 10 names 31e0fdf7 added (9
test_loop_carried_producers[...native_llvm], ab
test_project_unit_recognizes_exact_lowered_closure_formals) pass again; no
new names; counts back to aa 77 / ab 7 / ac 26 / rec 20.
