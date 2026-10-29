# 2026-10-29 Native finalizer entry joins source exception states

The reducer now posts each Call/Raise lexical entry state immediately to
`SOURCE_EXCEPTION_SITE_STATE`. Canonical relabel derives canonical state
cells from those ingestion cells, and exception topology reads the page to
post transfers. The raw `source_exception_site_bindings` dictionary is gone.

For `try: z = int(x); except ...: raise TypeError; finally: z = z + 1`,
the reducer creates a source-owned Phi for the finalbody read: the success
arm supplies the Call result and exceptional arms supply the pre-call `z`.
`SOURCE_EXCEPTION_FINALIZER_PHI` records the exact Try, source edges, and
operand cells. SSA branches to one finalizer entry, merges those values and
the pending status, executes finalbody, and propagates a nonzero status.
The concordance audit reads both new pages and checks their causes.

Native C and LLVM compilation and execution both succeeded for that
program. x=2.4 with n=0,1,2 returned 0,3,3; x=NaN with n=1 raised the
handler's NativeTypeError. For a direct `raise ValueError` caught by a
handler that increments `z`, then a finalbody that increments it again,
both backends returned 0,3,6 for n=0,1,2. When finalbody itself raised
KeyError, both backends raised NativeKeyError for both successful conversion
and NaN, overriding the pending exception. These were exact native artifact
runs from the public source compiler, not interpreter parity checks.

Still incomplete: handler-entry Phis for differing predecessor states,
finalbody return/break/continue/nested control, nested Try outward dispatch,
and the real codec's dynamic Slice native view. The broad view audit's
prior selected-return refusal was not addressed. The user asked to skip
the test gates for this step; only exact native artifact runs were made.
