2026-10-24 AXIS REDUCTIONS OVER STATIC SHAPES (MEASURED, cloud Linux, base 32de07f):
y.sum/prod/mean/max/min(axis=k) now spelled on C, LLVM, Fortran, SPIR-V
(were refused). tests/test_axis_reductions.py 105 pass; 4 Vulkan runs pass.
Gate vs main worktree 32de07f: same failure names. Full typed corpus NOT
run (stopped when parked). HAZARD, not this PR's: Fortran rank>=2 dummies
are declared with Python's extents over row-major storage, so y[0, 1] of a
(2, 3) span reads the wrong element on main; the axis code here reads
through a reversed-extent reshape and must change with that convention.
