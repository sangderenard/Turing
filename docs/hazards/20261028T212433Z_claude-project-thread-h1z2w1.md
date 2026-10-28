2026-10-28 LANE D, D5: A CALL'S SPELLING IS NOT ITS IDENTITY (MEASURED, cloud):
Ingestion renamed a bare callee Name to its tensor-catalog canonical spelling
(`sum_` -> `sum`, `T` -> `transpose`, `randint` -> `random_source`) before the
callee resolved, and marked any free call whose spelling is in the catalog as
that tensor operation. A program's own `def sum_(a, b): return a * b` then
ran as builtins.sum (SILENT WRONG: 6 for n=3, Python 9); a user class or
function `T`, or a user `randint`, became a free Input and refused.
`tensor_operation_name` now keeps a bare name's spelling (only the tensor
constructor spellings still become `tensor`), and the reducer drops the
spelled `tensor`/`tensor_candidate` from a call once its callee resolves to a
program function (a catalog source reference keeps it). c812-c815 PASS (C and
LLVM). Focused corpus subset (19 programs that define or call names matching
the catalog) is identical to main otherwise. Pytest files touching
ingestion/catalog give identical failure names on both trees.
Still open, loud: a user `def max(a, b)` shadowing the builtin loops the
callsite specialization guard on main and here (python_special_cases treats
`max(...)` as the builtin by spelling).
