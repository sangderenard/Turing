2026-10-24 AXIS REDUCTIONS OVER STATIC SHAPES (MEASURED, cloud Linux, re-measured on main e0e61ab):
y.sum/prod/mean/max/min(axis=k) now spelled on C, LLVM, Fortran, SPIR-V
(were refused). tests/test_axis_reductions.py 111 pass; 4 Vulkan runs pass.
Gate parts aa/ab/ac/rec vs main worktree e0e61ab: same failure names. Typed
corpus (c,llvm,fortran) vs main: one new failure, c469 (builtin min(n, 3)
arrives with dim=3 and was taken for an axis reduction); fixed by skipping
scalar sources, c469 passes again on C and LLVM. Fortran's rank>=2 storage now reads y[i][j] as t0(i+1, j+1) on main
(fixed upstream by 2026-10-27), so the Fortran spelling is plain DIM=k+1.
