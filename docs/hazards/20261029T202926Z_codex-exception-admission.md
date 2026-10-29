2026-10-29 SOURCE EXCEPTION FACTS (MEASURED, Windows worktree)

The bounded public-source tests in `tests/test_source_exception_admission.py`
initially failed 6/7 after source-site and transfer records were added. A
call's `SOURCE_EXCEPTION_SITE_STATE` row is in the reducer's ingestion read
scope, while its exact `INGESTION_VALUE` cell is in the graph build's scope.
The transfer writer wrongly looked up the source under the reducer row scope
and refused. A direct book inspection printed the site row, its source Ref,
the absent same-row ingestion Ref, and the exact enclosing Try Ref. The
transfer now reads the stored source Ref and verifies that cell directly.

The two-Try source also exposed a prior reducer error: after merging the
first Try's success and handler values, it bound `y` to the Try syntax node
without a `NAME_BINDING` version. A later protected call could not cite that
value. The authored-Try merge now mints a Phi derived from exact continuing
arm binding cells and posts `SOURCE_EXCEPTION_CONTINUATION_PHI`. Generated
resource scopes retain their existing path. A nested handler read of the
static `TypeError` declaration had been treated as a runtime name demand;
the site writer now uses the Name resolver's static-binding sources to
exclude compiler-static names.

After these writer corrections, the seven focused admission/source-topology
tests passed (2.87 s). A separate focused test checked that all new source
exception concordance audit methods returned no disagreement on the failed
compilation's book (1 passed, 1.64 s). The authored-Try admission remains
Unresolved and compilation still refuses. `tools/audit_identity_concordance.py
view` remained red before and after at the existing `selected return has no
exact source call reachability authority` refusal. Native C/LLVM exception
execution, handler matching, finalizer state, and pending abrupt outcomes
are not verified or claimed.
