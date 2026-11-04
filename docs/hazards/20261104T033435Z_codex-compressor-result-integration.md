2026-11-04 numerical constructor dimension audit (MEASURED, repair worktree):

- A bounded public `root(like: AbstractTensor, count: int)` returning `AbstractTensor.zeros((count,), cls=type(like))` lowered through the actual rank-one constructor region. The numerical dimension receipt names current forked member 0, the authored tuple layout, and the exact constructor operand snapshot. `REGION_SIGNATURE` feeds member 0 and cites that receipt directly.
- The layout's member cell remains in its original source scope while the current graph member is in the read-scope fork. Both have the same unique `INGESTION_VALUE` origin. The audit requires that exact origin equivalence and both direct causes; numeric ID agreement alone is insufficient.
- A forged receipt substituting the tuple container for the member produces a finding. Removing the receipt-to-REGION_SIGNATURE edge also produces a finding. `python -m pytest -q tests/test_numerical_constructor_dimension_audit.py`: 1 passed in 5.45 s. Evidence_inventory separately ran the positive full-source test and mapping audit.
- No full compressor run was made for this audit slice.

Root five-case frozen gate passed21.84 seconds, mapping28/0,CLOSED. Both dimension member substitution and missing signature receipt are rejected; the real root Call takes authored count0 rather than unexecuted tuple2.
