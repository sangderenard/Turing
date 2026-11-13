# Existing NumPy lowerer: actual signal graph

Worktree based on 284ff7f8; no concurrent compiler changes copied. Extended the
existing lower_functions_to_numpy dialect, not a new backend or graph solver.
Repairs: resolve real AbstractTensor aliases, preserve authored lexical super
when emitting subclasses, preserve complex/dtype/device constructor semantics,
and spell broadcast/shape/clone/detach/CPU cast/stack/linear-algebra operations.
NumPy is inference-only here; training/autograd is not a claimed product.

Verification: 16 focused tests passed, 1 deselected. The deselected existing
Nodus-arena dispatch assertion also fails on canonical baseline with the same
missing observed sub call; it is not hidden as a passing result. An earlier
run without the arena library skipped that environment-dependent test.

Reproduce complete current proof (Turing root, user scratch output):
python tools/check_signal_graph_numpy.py --spectral-root <spectral-worktree>
  --output-dir <untracked-user-scratch>
Requires Spectral 7a7d8f0 optional AbstractTensor path. This compiles the actual
GraphSolver methods plus TensorEdge.apply using the existing lowerer; it does
not rewrite solver equations. Outputs are bare NumPy arrays, compared against
real interpreted AbstractTensor GraphSolver results.

Measured chain and feedback-cycle 256-frame products have max error zero.
Seven alternating batches of 50 warmed windows, five warmups, no profiling:
chain median 7.359 ms interpreted vs 0.1475 ms emitted; feedback 7.646 ms vs
0.1648 ms. This is a bounded tiny-graph CPU result, not a universal speedup or
an LLVM measurement. Includes scheduler overhead, excludes source generation
and startup. One-sample delay, direct nonlinear square and the actual materialized
camera radio graph also pass differential checks (1e-12 tolerances).

Not proved: arbitrary nonlinear modules, training, all shapes/dtypes, every
GraphSolver capability or production deployment. No live service changes,
shader rendering, dependency install, or new network requests.
