# Regulation through the concordance — report B (2026-09-26)

Continues [`CONCORDANCE_REGULATION_REPORT.md`](CONCORDANCE_REGULATION_REPORT.md)
(commit `56255f4`, integrated here by fast-forward).  The working rule for
this round: **a privately held record that causes a bug is a place the
concordance should grow.**  Each defect below was found by running the
compiled program natively against the authored Python.  Each fix moves the
deciding fact onto the identity book, where one row answers every reader.

## Environment and baseline

The run was in a Linux container, not the Windows environment of report A.
The native toolchain (`ziglang`, `llvmlite`) and `joblib<1.6` were
installed.  Report A's focused set (the 20 test files touched by
`56255f4`) was re-baselined in a clean worktree at `56255f4`:
**47 failed / 328 passed**.  Report A's "19 failed" was measured on
Windows, so the two counts do not compare directly.

| Check | `56255f4` (this env) | Now |
|---|---|---|
| Focused set + new guards | 47 failed, 328 passed | **37 failed, 348 passed**, none new |
| Native linalg / in-place suites | 7 wrong-answer failures | 1 (structural IR-shape assertion) |
| New native guard programs | — | 6/6 pass |

## Defects fixed, and the book page that now holds each fact

### 1. Two carried bindings seeded from one value (silent wrong answer)

`second = value; third = value`, both carried.  The lowering rebound the
shared initial id to the first carried entry's header Phi only.  A direct
`if` predicate leaf read that id, so `third > 2.0` tested `second`.

- `ControlExpression.read`: each `value` leaf carries its operand position
  `(consumer, role, ordinal)`, the key of its `lexical_read_binding` row.
  This closes A6 item 1: direct predicates no longer read by value id.
- Page **`loop_carried_entry`** `(control scope, loop, binding) -> entry`.
  The resolver returns that entry's header value, or its update at the
  latch.

### 2. One captured value read as two bindings by one region (refused)

`second = value; while ...: second = second + value`.  A region had one
formal per value id, so it could not receive `second` (carried) and `value`
(invariant) at once.

- Page **`region_capture_binding`** `(control scope, region, formal) ->
  (source value, binding)`.  The region gets one formal per binding, and
  the feed reads each formal by its binding.

### 3. `while go:` never terminated

`go` is rebound in the body.  The reducer classed it as
overwritten-before-read, although a while test is read every iteration.
Projection then dropped the carried pair, and the latch re-ran the
pre-loop condition region.

- Reducer: names loaded by a `while` test are never overwritten before
  read.
- Projection keeps carried pairs whose initial the predicate reads.
- Page **`while_carried_test`** `(control scope, loop) -> binding`.  The
  latch uses the carried update and does not re-run the condition regions.

### 4. Region formals typed apart from their callers (wrong answers)

`pair_update`, baked GEMM.  `n` is an `int` loop bound, but region formals
typed it `float64` from the planner's default.  `n + k` was typed twice by
two parties: the caller loaded an `int64` slot as a double.

- Page **`control_uniform_dtype`** `(control scope, value) -> dtype`.
  Loop-bound uniforms are ints, and region formal typing reads the row.
- Page **`region_value_dtype`** `(control scope, value) -> dtype`.  The
  producing region publishes each scalar result's dtype, and consumer
  formals read it.  Integer-only arithmetic is decided integer once, at
  region build.

### 5. Carried read through an in-place store chain (refused)

In-place rotation loops (Jacobi, four-deep sweep): the operand row was
keyed by the store version, not by the arena the feed resolves to.  The
feed now resolves through the alias record to find the row.

### 6. After-loop store pulled into the inner loop (wrong answer)

`v[i,i] = 1` after an inner zeroing loop.  Placement joined the two
distinct versions of `v` through their shared arena id.

- Page **`loop_region_membership`** `(read scope, loop) -> regions`,
  written by the composer.  Placement moves a region into a loop only if
  that loop owns the region.

### 7. Second, unconnected formal for one array

An init loop followed by a nested update loop.  The carried initial (the
post-init version) was looked up without following aliases, so it became a
new `v` formal.  An initial with no binding of its own now follows the
alias record to its arena.  With 4–7, the full compiled Jacobi eigh matches
numpy natively.

## Guards added

- `tests/test_loop_binding_reads_native.py`: defects 1–3 (C backend,
  subprocess-bounded so a non-terminating loop fails, not hangs).
- `tests/test_inplace_arena_loops_native.py`: defects 6–7 (LLVM).
- Defects 4–5 are guarded by the existing `test_compiled_linalg.py` and
  `test_llvm_inplace_store_aliasing.py`, which now pass.

## Known to remain

1. `external_values` is still a private value-keyed dict (A6 item 2).
   These fixes narrow what it decides, but binding state is not yet a book
   view.
2. Region dtype publication relies on plan order (producers first).  A
   consumer typed before its producer would still read the planner guess.
   A fixed point or refusal on disagreement is the next step.
3. The in-place alias record contains a cycle (`5 -> 18 -> 5` in the
   init/update repro).  The resolvers stop on it; it should be
   normalized at the source.
4. The control scope key still embeds `id(control)`, a process address.
5. The remaining 37 focused failures.  The largest groups are record/table
   linking in `test_fortran_c_shell.py` and
   `test_process_graph_function_linking.py`, which include four
   `sequence contract concordance disagreement` refusals and two
   `max id < 1e9` assertions that predate structured ids.  Also:
   `test_loop_interchange` (4), `test_python_shell` (3), and one IR-shape
   assertion (`test_the_loaded_value_is_pinned_rather_than_re_read`).
6. Report A's items 3–6: the private linker maps, name matchers, emitted
   parameter order, and unrecorded linker decisions.
