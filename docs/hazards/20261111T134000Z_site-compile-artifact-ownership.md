2026-11-11 SITE COMPILER ARTIFACT OWNERSHIP (MEASURED Windows; rebased onto d2a7f022):

Existing signed-in divider HTTP/API now wires to one Nodus bridge worker via
explicit startup configuration. The authenticated divider job's account maps
atomically to the existing workspace ledger, preserving host contract layout.
External user-space output, charged reservations/measurement, retained lifetime
review dates and page status replace unowned bridge output. No deletion occurs.
Timeout/unknown remote state stays unfinished/charged; review dates never free
bytes. Context-free bridge CLI dispatch now refuses with a site-only message.

Paired Nodus adapter commits b3e14222 and 3cd6a330 validate site artifact context,
redirect cache/temp/spill destinations, capture stdout/stderr and inherited child
streams, retain lifecycle/traceback receipts and restore process settings.
Turing root selectors additionally fix five repository-derived output defaults,
including graph_work's explicit compiler_foundations directory found in review.
Review also found/fixed API host-path leakage from scan exceptions and configured
backend override silently relabeling older queued attempts.

Final Turing focused gate: 116 passed, 12 subtests, one inherited cffi imp
deprecation warning, 58.83 seconds, using SpeakToMe REST v2 commit 86721dd.
Covers compile artifacts, workspace ledger,
real HTTP/auth/intake with fake Nodus contract endpoint, bridge statuses and
path-only compiler selectors, shared schema gate and REST dependency checks.
Node divider syntax passed. An earlier repeat exposed the existing queue's
concurrent first-schema declaration race; SiteCompiler now initializes that
queue synchronously before installing the hook or starting its worker. The
final authenticated HTTP run passes after this repair. Another intermediate
collection failure was the other lane's REST v2 transition; rebased onto its
Turing counterpart, preserving both adjacent startup blocks, then reran.
Nodus: 28 passed, 1 skipped for Windows symlink privilege, 3 real-compiler cases
explicitly deselected. Plain Python print child verified diagnostic inheritance.
No actual compiler, native build, production schema/config, ACL/network change,
file migration/deletion, live restart, push or full-suite run.

The old srv/state jobs root is under the outer Git checkout and intentionally
fails strict external-root validation. Production needs an explicit external
jobs root, capacity/quota/reservation/review policy, matching Nodus host root,
paired source deployment and coordinated lifecycle restart. The committed
REST/Studio changes are preserved in this branch; later uncommitted canonical
Apache changes must still not be overwritten.
See docs/SITE_COMPILER_ARTIFACTS_2026-11-11.md for uncovered legacy/source-copy,
adopted-library and SQL overhead accounting, and the unchanged OS isolation gate.
