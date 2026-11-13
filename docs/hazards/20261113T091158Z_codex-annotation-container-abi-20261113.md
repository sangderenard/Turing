2026-11-13 Exact sequence lifetime births before owned record escape (MEASURED, compiler source/SSA boundary):

A dynamic descriptor and a heap-owner-shaped cell do not prove that the caller may receive ownership. Parameter admission can reuse the same descriptor machinery, and selected owner addresses can refer to different predecessor arenas.

Added typed LOCAL_NEW, BORROWED and ADOPTED lifetime receipts with immutable birth references, exact canonical contract source, descriptor-member mint links and a current-descriptor join. Global concordance checks these receipts. Local birth requires an authored list display, an existing constructor snapshot receipt, or an exact comprehension publication. Adoption preserves original owner/arena identity; cycles and ambiguous originals are refused. These receipts do not authorize native ownership transfer.

Nine targeted tests passed before global integration: real mixed births, borrowed input, forged source/contract/owner/data with matching forged cause lists, original adoption, cycle refusal and current descriptor mismatch. Four real helper/alias/caller-mutation probes also passed and are being promoted to regression tests. Mapping audit: 28 rows, 2 functions, zero findings, zero unsourced identities, latch CLOSED.

Actual GraphSolver compile_nodes passes source/final concordance with this audit and still refuses LLVM emission because owned sequence 100 has no validated public ownership escape contract. Residents 106 and 104 have exact local birth receipts. No actual string-list result or merged-return native execution is claimed.

Known independent boundary: list(set-comprehension) currently has no durable constructor snapshot receipt and remains unproven. Extending that receipt requires preserving the existing ordered clear/extend copy semantics, not blessing a call by its spelling.

Global audit initially exposed two bytes-parameter regressions in the new reader. Fixed by checking the actual existing signature contracts separately: explicit ProgramABI PARAMETER_ABI_KIND is minted by PARAMETER_ABI_DECLARATION with the exact version-zero binding; annotation-only undeclared kind derives from that binding. The first combined candidate had 33 passes and 3 failures versus 25 passes and 1 incumbent failure; the two new failures are now covered by a 17-pass targeted/global/native rerun. Full incumbent rerun remains pending before commit.

Final actual-source rerun after the signature fix again reaches only the explicit sequence-100 public escape refusal. Mapping audit remains 28 rows across 2 functions, zero findings and zero unsourced identities, latch CLOSED. The independent scalar-pointer snapshot repair is committed separately as f4319f07; it does not authorize sequence escape.

Authoritative frozen-source combined run: 43 passed, 1 unchanged C bytes-concat private-root-storage failure (134.98s). The earlier provisional run was repeated because of a brief overlapping backend edit; it is not used as verification. The final comparison is 25 incumbent passes plus 18 new lifetime/selection tests, with the same one incumbent failure. Includes 17 lifetime tests (two declared-signature negative cases), one selected-lineage test, native LLVM-owned reruns, Optional sequence iteration and nested descriptor alias tests. No new incumbent failure remains.
