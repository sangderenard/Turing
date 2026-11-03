2026-11-03 Imported LLVM kernel formal physical identity (MEASURED)

In the isolated compressor-results checkout, the imported C
`topk_int64_dim` function has seven formals. Its repository metadata formerly
recorded argument names only. The C registry retains pointee semantic dtypes
for opaque LLVM pointer arguments: input, shape, indices and values carried
`SSAValue.dtype` int64/int32. `post_repository_kernel_identity` used that
resident dtype to post `SSA_VALUE` storage=scalar, rank=0 for those pointers.
The LLVM declaration is stronger: their authored types are `ptr`, while ndim,
dim and k are `i32`. The old writer's source expression and the imported
value metadata were inspected before changing it; no full compressor run was
used for attribution.

The importer now places each exact LLVM argument spelling in its
`REPOSITORY_KERNEL_VALUE` fact (`ptr %input`, `i32 %ndim`, and so on). The
physical writer reads that fact. Pointer formals leave physical_dtype,
storage and rank unposted for caller ABI settlement. Declared scalar formals
post the dtype from the LLVM scalar layout with storage=scalar, rank=0.
No pointee dtype is used to choose physical storage. The existing
`fill_double` kernel used by zeros retains its pointer output without
physical fields, while its `double` and `i32` scalar formals post float64 and
int32 physical dtypes respectively. This is a kernel formal declaration
repair; caller output shape and storage remain separate proof obligations.

Focused gate: `python -u -m pytest -q --tb=short
tests/test_c_backend_llvm_ssa.py tests/test_repository_kernel_physical_identity.py`
finished 52 passed in 2.76 seconds. `python -u
tools/audit_identity_concordance.py mapping` finished with 28 rows across two
functions, zero findings, zero unsourced facts/identities, latch CLOSED.
`python -m py_compile` for the changed importer and `git diff --check` passed.
No full compressor lowering, native DLL, or parity claim is made here.
