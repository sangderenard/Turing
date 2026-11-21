2026-11-14 2-D literal wash; LLVM axis reductions and tensor-op casts; nonzero as a composite of base operators; whole-list consumers unroll (MEASURED, main 33ef86d5 + 944ffddf + this change)

Continues 20261114T130606Z (broadcast law, tuple-data wash, static argmax), carried here as 944ffddf.  Branch
claude/youngman-row-wash-main.  YoungMan target: `_edge_lookup`'s `all(edge_pairs == edge_tensor, dim=1)` and
`AbstractTensor.nonzero(forward_match, as_tuple=True)[0].item()`.

1. WASH `tensor_rect_literal_as_flat_reshape` (ast_control_normalization `_wash_tensor_rectangular_literal`, run
   before the tuple-data wash on the entry and pursued definitions): `AbstractTensor.tensor(<R x C nested display of
   numeric constants>, ...)` -> `AbstractTensor.tensor(<flat list>, ...).reshape(R, C)` (any depth).  Row-major;
   checked in Python: `[(0, 1), (0, 2), (1, 2)]` int64 == `[0, 1, 0, 2, 1, 2]` int64 `.reshape(3, 2)` (values, shape,
   dtype).  Constants only, so no leaf can be a runtime sequence that changes the packing.  Before (scratch lit2d/c1,
   c2): the nested display lowered as a resident arena ("private sequence N has no proven capacity").  Recorded on
   SOURCE_IDENTITY_WASH; the audit accepts a List or Tuple literal span for this identity.  After: c1, c2 PASS LLVM.

2. LLVM (ssa_llvm_backend): a static-shape axis reduction (`all`/`any`/`sum` with a literal `dim`, `keepdim`) is a
   nested sweep -- result element o * inner + i reduces o * extent * inner + k * inner + i -- with the flat reduction's
   accumulators.  Before: only flat reductions to a scalar had emission ("operation has no repository LLVM emission").
   `to`/`to_dtype`/`astype` with one source go through the span Cast branch (they are `_CAST_OPERATIONS`).
   After (LLVM): lit2w/c5 (2-D literal, broadcast ==, all(dim=1), .to(int64), sum) PASS; lit2r c3 (broadcast ==), c4
   (all(dim=1)) PASS.

3. `AbstractTensor.nonzero` (abstraction_methods/comparison.py) is now a composite of base operators (owner
   direction: self-referential operators of base ops, no backend specifics): mask `self.reshape(-1) != 0`, positions
   `arange(numel, dtype=int64)[mask]`, per axis `(positions // stride) % extent`; as_tuple -> tuple, else
   `stack(coords, dim=1)`; rank 0 is treated as rank 1.  Registered in `operator_catalog.compositional_tensor_source_
   references` (beside emax/emin/solve), so the compiler pursues its source.  Python check: numpy (1-D bool/float,
   2-D, 3-D, empty, rank 0) and torch (5 cases) identical, int64.  Behavior change: as_tuple now returns
   AbstractTensors, not raw backend arrays; callers laplace_nd (get_tensor of the stacked form, [0][0].item(),
   zip(*tuple) indexing) and meowdoku lifecycle (.nonzero()[0]) checked to behave as before.

4. loop_composer `local_append_publication`: a whole-list consumer (`arg:0` of a declared `builtins.tuple` /
   `builtins.list`, or op `stack`/`cat`/`concat`/`concatenate`) is admitted (read with literal None).  After the
   unroll the list keeps its identity as the display of the iteration values, and a `tuple(L)`/`list(L)`
   materialization becomes the Tuple/List display of the same members (as glsl `_publish_aggregate_formal_members`
   converts an aggregate formal).  Before (scratch whole/c11, nonzero's coordinates loop): a whole consumer was an
   "escape", the static loop was retained, and the tuple of its tensors had no record.  After: whole/c11 PASS LLVM;
   whole/c10 (`stack(parts)`) lowers with audit 0 and stops at "stack has no repository LLVM emission".

Regression gates, this branch vs a baseline worktree at 33ef86d5 (Temp\wtm33): 40 self-field (LLVM + Fortran), 74
for+if (LLVM), 11 clues: identical, every line.

Open: nzc/c06 `AbstractTensor.nonzero(m, as_tuple=True)[0].item()` in a loop still refuses ("region value 42 has no
settled dtype in the region's value records"): callsite specialization reads the composite's return before shell
planning unrolls its loops, so `tuple(coordinates)` is still a materialization of an empty list at that point.
LLVM: flat `argmax` and `stack` have no emission.  Fortran: no `reshape` emission.
