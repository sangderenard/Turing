# 2026-10-29 Handler entry Phi from source exception arms

The source reducer creates a handler-entry Phi when the handler reads a
name whose value differs among exceptional entry states. Each arm is the
value recorded at its exact protected Call or Raise site. Canonical relabel
preserves the handler, site, and Phi identities. Exception topology posts
`SOURCE_EXCEPTION_HANDLER_PHI` with the transfer and value cells, and the
concordance audit checks those causes and the handler entry receipt.

SSA captures each arm at the actual failure branch, emits the handler Phi
at the handler block entry, and binds the graph Phi through
`EXCEPTION_HANDLER_MERGE`. This fixes the prior formal value refusal for
the handler read. The branch records the exact source transfer cell; no
post-return exception report or Python callout is involved.

Native C and LLVM artifacts from the public source compiler both compiled
and ran the same loop with `z=int(x); z=z+1; raise ValueError` and a
`ValueError` handler that adds 2. For x=2.4 and n=0,1,2 the outputs were
0,5,5; for x=NaN and n=1,2 they were 2,4. With a `finally` that adds 3,
both backends returned 0,8,8 and 5,10 respectively. These are native
artifact results, not Python interpreter comparisons. The user asked to
skip test gates, so no pytest gate was run for this change.

The remaining goal is not complete: nested Try outward dispatch,
return/break/continue in finalbody, and the codec's dynamic Slice and
materialization still need native lowering. The broad concordance view's
prior selected-return authority refusal was not addressed here.
