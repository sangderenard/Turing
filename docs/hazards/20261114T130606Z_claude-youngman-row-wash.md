2026-11-14 LLVM broadcast read past its operand (silent wrong result); tensor tuple-data wash; static argmax (MEASURED, b510e3e2 + region-frontier fix + this change)

Branch claude/youngman-row-wash: b510e3e2 (all gates identical there) + cherry-pick of the region-frontier fix (PR 1224),
so measurements are free of main's 0e3dd724 IndexError regression.  Owner direction: wash difficult Python with exact
identities before lowering (YoungMan `_edge_lookup`); owner priority: fix the silent broadcast result first.

1. SILENT WRONG RESULT, LLVM (ssa_llvm_backend elementwise span sweep).
   Program (scratch lit2r/c2): `ep = tensor([0, 1, 0, 2, 1, 2], dtype="int64").reshape(3, 2)`; in a loop
   `t = tensor([k, 2], dtype="int64")`; `total = total * 10 + (ep + t).sum().item()`.  LLVM returned 0, Python 135
   (n=2) / 1368 (n=3); constant row (c5): 0 vs 165.  Audit clean.  SSA was right: `Add (19, int64, (3, 2)) (13,
   int64, (2,)) -> (22, int64, (3, 2))`.  Writer: the sweep loads every span operand at the RESULT's flat index; the
   (2,) row was read at 0..5, past its buffer.  The runtime-sized branch refused mixed ranks; the static branch had
   no check at all.  Rule now holding: an operand whose static shape differs from the result's is indexed by NumPy's
   broadcast law (axes right-aligned; an extent-1 or absent axis contributes 0; flat index -> coordinates by the
   result's static strides); any other mismatch, or a rank mismatch without static extents, is refused
   ("does not broadcast to the result by static extents").  Measured after (LLVM): c2, c5 PASS; same-shape c6 PASS;
   row+row PASS; column `(3, 2) * (3, 1)` (bcfix2/c7) PASS; two-sided `(1, 2) + (3, 1)` (c8) PASS.
   Not an audit gap: the SSA facts were correct; the fault was in emission.

2. WASH `tensor_tuple_data_as_list` (ast_control_normalization `_wash_tensor_tuple_data`, run beside the
   requires_grad_ filter on the entry and pursued definitions).  `AbstractTensor.tensor((a, b), ...)` ->
   `AbstractTensor.tensor([a, b], ...)`: `ensure_tensor` and every backend's `tensor_from_list_` test
   `isinstance(data, (list, tuple))` and treat both alike.  `AbstractTensor` is identified by its declared import
   (`from src.common.tensors[.abstraction] import AbstractTensor`), never rebound.  Before (tpair/c03b): the runtime
   tuple display had no SSA producer -> `root` got an invented formal ("unaccounted-formal ... dtype=int64").
   New page SOURCE_IDENTITY_WASH (site = call span) = (identity, call span, data display span), DERIVED from both;
   audit `_source_identity_wash_findings` (cells on the receipt, kinds Call/Tuple); a tampered fact is reported.
   Measured: tpair/c03, c03b PASS LLVM, audit 0.

3. Static-form reductions (glsl `_tensor_descriptor_rule`): `AbstractTensor.argmax(x)` has no receiver role; its
   tensor is `arg:0` (then not read as a positional axis).  Before (nz/c07): no descriptor -> region record missing.
4. Static-reference call deferral (glsl `_decide_value_shape`): a static-reference call whose own descriptor states a
   tensor contract is not an "unlinked callee"; before, everything computed from `argmax(...)` stayed deferred.
5. Planner: `.item()` of an argmax/argmin receiver is int64 (operator law).
   After 3-5, nz/c07 reaches emission and stops at the next gap below.

Open, measured (LLVM): 2-D tensor literals (`tensor([(0, 1), ...])`, `tensor([[0, 1], ...])`) lower as a resident
arena ("private sequence N has no proven capacity"); exact identity `tensor(rect literal) == tensor(flat).reshape(R, C)`
verified in Python, reshaped form lowers.  Broadcast `==` and `AbstractTensor.all(x, dim=1)`: "operation has no
repository LLVM emission".  Fortran: no `reshape` emission.  `nonzero(m, as_tuple=True)[0].item()`: no lowering.

Regression gates: see the commit message.
