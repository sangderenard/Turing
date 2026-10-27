2026-10-24 AXIS REDUCTIONS OVER STATIC SHAPES (MEASURED, cloud Linux, re-measured on main e0e61ab):
y.sum/prod/mean/max/min(axis=k) now spelled on C, LLVM, Fortran, SPIR-V
(were refused). tests/test_axis_reductions.py 105 pass; 4 Vulkan runs pass.
Gate vs main worktree 32de07f: same failure names. Full typed corpus NOT
run (stopped when parked). Fortran's rank>=2 storage now reads y[i][j] as t0(i+1, j+1) on main
(fixed upstream by 2026-10-27), so the Fortran spelling is plain DIM=k+1.
