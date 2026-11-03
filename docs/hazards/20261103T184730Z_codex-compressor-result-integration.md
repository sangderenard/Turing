2026-11-03 Native output paths read the existing output identity ledger (MEASURED)

The frozen compressor scan at
`C:/dev/Powershell/turing/build/compressor_unit_scan_27` marked unit 0
`_read_header` partial. Its authored tuple paths were `(0,) -> 4` and
`(1,) -> 13`, while the accepted SSA outputs were 4 and 43. The exact
function's OUTPUT_IDENTITY page already recorded `13 -> 43`; its durable
`output_identity_aliases` agreed, while its `value_aliases` view was empty.
The unit publisher consulted only the latter view and consequently published
the obsolete source spelling 13 as a physical output.

`_accepted_physical_output_paths` now reads the incumbent checked
`_concordant_function_aliases(include_output_identities=True)` reader in the
module's owned book. The earlier, weaker value-alias-only rewrite was removed
from the unit publisher. No alias is inferred or added. Reading the same
frozen function and book before the repair returned paths 4 and 13; after the
repair it returns 4 and 43. The pinned old plan has not been recompiled with
the changed compiler, and no native verification of `_read_header` is claimed.

The focused return-layout checks, including an exact output identity and a
forged durable snapshot that must refuse, passed: 5 passed in 1.64 seconds.
Command: `python -u -m pytest -q --tb=short tests/test_project_compilation_product.py -k "record_return_layout or optional_tuple_return_layout or native_output_path"`.
