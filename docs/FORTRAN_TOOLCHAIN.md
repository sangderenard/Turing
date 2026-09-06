# Generated Fortran toolchain policy

Generated Fortran artifacts use the centralized policy in
`src/compiler/fortran_toolchain.py`. GNU Fortran and its C host are compiled
with `-O3`, `-march=native`, LTO, loop unrolling, and frame-pointer omission.
The final link uses LTO plus static `libgfortran` and `libgcc`; MinGW builds
also use `-static` so `libquadmath` and transitive GNU support libraries are
selected from archives.

Native C-shell executables and `compile_module()` shared libraries are
standalone by default. Callers may explicitly pass `standalone=False` when a
dynamic compiler runtime is intentional. An unknown Fortran compiler fails a
standalone request rather than silently producing an artifact with unverified
runtime dependencies.

`-ffast-math` is intentionally absent. It changes NaN, infinity, signed-zero,
and reassociation semantics and therefore is not a safe global policy for SSA
programs. Host CPU tuning also means the resulting artifact is standalone with
respect to redistributable libraries, but targets the instruction set of the
machine that compiled it rather than every older CPU.

On the reference MSYS2 GCC 16 MinGW toolchain, static `libgfortran.a` refers to
the POSIX `strndup` symbol which the Windows CRT does not export. The generic C
shell supplies that small compatibility function. PE import-table tests reject
`libgfortran`, `libquadmath`, `libgcc_s`, and `libwinpthread` regressions.
