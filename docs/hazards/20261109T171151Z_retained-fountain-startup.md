2026-11-09 RETAINED FOUNTAIN STARTUP (MEASURED, canonical Windows;
source-only startup option, no production restart):

server.py now accepts --retain-fountain-products RECEIPT. This replaces only
the fountain preparation lane with read-only retained artifact validation.
The default startup path, other preparation lanes, shell installation, and
original source digests are unchanged. The receipt is checked before any
shell copy, session creation or socket binding, and rechecked when the lane
starts. A missing/corrupt artifact refuses without any build fallback.

retained_products.py inventories the existing eight fountain manifests and
every WGSL/CPU file they advertise. It checks schema, required cost model,
size/count and argument/state/output/slot contracts, arena sizes, compute
entry names, CPU headers/ABI references, safe local filenames, and SHA-256
identity against an explicit operator-held receipt. The helper imports no
source compiler or physics builders. CPU readiness names only the retained
fallback sizes; a family with none is skipped rather than ready, even when
the old source build would have offered a fallback.

The receipt writer is create-only and refuses a destination inside the served
directory. Inventorying is not a claim of current-source parity or new physics
validation. WGSL is not compiled or executed by this validator. A malformed
previously accepted kernel is not made correct by its checksum. Do not refresh
the receipt to conceal a changed file; restore the accepted release or perform
the deliberate new-release validation instead.

Production read-only inspection passed for 8 families and 74 referenced files:
3.659 s inventory, 5.051 s pinned verification. All 83 existing fountain*
files (including unreferenced legacy artifacts and the page JS/HTML), totaling
270132287 bytes, are SHA-256-identical before and after this work.

Receipt: build/dot-retained-fountain-20261109/fountain-receipt.json
SHA-256: 7f3f6c2177b322099d3d0df02c2d64fae33b638fddefe6244efbe3b8b04cb93f
Additional receipts in that directory: product-hashes-before.json,
product-hashes-after.json, validation-receipt.json.

Focused tests: 30 passed in 1.55 s (test_meowdoku_retained_products.py),
then 6 passed in 1.48 s (existing test_meowdoku_startup.py). The latter starts
only its isolated temporary localhost test server. Negative coverage includes
missing/modified manifests, WGSL, CPU binaries and ABIs; path traversal and
absolute/URL references; malformed counts/slots/metadata; incomplete CPU
references; stale receipt recheck; refusal before startup side effects; exact
default build order; truthful CPU readiness; no builder calls/imports and no
product writes in the retained lane. Both runs used installed Python311 and
PYTHONIOENCODING=utf-8. git diff --check passed for the server changes.

No native physics edit, compile, GPU job, export, production service action,
exposure action, commit or push. Other workers' source changes are preserved.
