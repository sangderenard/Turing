2026-11-03 octet_lz._header break predicate binding (MEASURED)

The frozen parallel catalogue product at
`C:/dev/Powershell/turing/build/compressor_unit_scan_27/product/units/unit_013/failure.json`
reports `_header` site 12: break-edge value 6 for pre-loop identity 0 does not
dominate. This was reproduced in this isolated worktree through the public
`lower_ast_source_to_ssa` entry using the actual `inspect.getsource(_header)`
and a `root(x:int)` caller under `program_extraction.yaml` (about 15 seconds).
The result was the same `break-edge-value` shortfall; it is not caused by the
catalogue unit cut.

The reducer's `SOURCE_LOOP_SITE_EDGE` records `(initial 0, updated 6) -> 6` at
break site 12. The control builder emits the rebound integer as `Load -> 6`
before the conditional. Its conditional reader previously requested
`external_value(6, dtype='bool')`; that writer posted a `REFINED` binding for a
new bool SSA object with the same ID, replacing the integer object's read view.
Instrumented `_bind` observed the int64 LOOP_SEED followed by the bool REFINED
write. The bool clone has no producer in the break predecessor, so the
dominance check correctly refused it.

The conditional now reads the resident value without refining its identity.
For a scalar numeric value it mints a separate bool `Ne(value, zero)` predicate
from the resident SSA cell and the exact conditional construct cell. The
authored integer binding remains available to the break edge. A focused test
captures the actual `_header` control function before later finalization and
checks the separate predicate, the int64 source, the binding row and absence of
`break-edge-value`; a bool-conditional test checks the existing bool path.
`python -u -m pytest -q --tb=short tests/test_break_predicate_binding.py`:
2 passed in 12.66 seconds. A bounded nested-loop scalar source reached
SSA-COMPLETE in 8.3 seconds.

The actual `_header` public lowering now passes this break edge and reaches a
later final concordance refusal: `loop_body: Load int64 from column 1 whose
exact native element is int32`. A separate count-return native slice reaches
SSA-COMPLETE but its C emitter refuses `missing_physical_dtype` on a root loop
Phi. Neither later boundary is repaired or claimed as native parity here.
`python -u tools/audit_identity_concordance.py mapping` reports 28 rows, zero
findings, zero unsourced facts and CLOSED latch. One selected older composer
test failed an assertion in its own planner stage, while the selected terminal
return activity test passed; that composer result was not compared with an
untouched checkout and is not attributed to this change.
