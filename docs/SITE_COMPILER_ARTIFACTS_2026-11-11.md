# Site compiler ownership and artifact lifetime

Source implementation, not a live deployment. This extends the existing Source
divider page/API, Nodus bridge and workspace accounting. No new identity or
compiler queue is introduced.

## Identity and request route

`divider.js` submits with the existing session cookie to `/api/divider/jobs`.
`server.py` requires `compile_submit`, resolves the logged-in account through
`_job_identity`, and stores `account:<users.id>` in `divider_jobs.owner_key`.
The Nodus worker checks that owner and the current worker/generation/deadline in
the same SQL transaction that reserves its workspace bytes. A client-supplied
owner field, header, Nodus participant id or filesystem username cannot select
the owner. Anonymous/browser-owned historical jobs are refused by this worker.

The configured `SiteCompiler` is the existing `sessions.divider_hook`. The hook
wakes one background worker and returns promptly. That worker drains the
existing source-divider lease queue through `divider_nodus_bridge`; it does not
run source_divider's direct subprocess compiler. Uploads use the configured
Nodus backend. An older attempt with another backend is refused, never silently
compiled as a different backend. The bridge's old direct dispatch CLI refuses
with a site-only message. Library helpers and legacy development adapters are
not an OS security boundary and have not been globally removed.

## Storage and the existing ledger

New schema module `compile_artifacts` v1 owns only `compile_artifact_runs`.
Its workspace_id references `workspace_leases`, which retains the authoritative
account owner, reservations, measured bytes and release generation. Reservation
and run binding are atomic. All quota/headroom admission uses the existing
workspace ledger, including truthful overruns. It is not an OS disk quota.

The native host convention is unchanged:

- `<jobs_dir>/<contract_id>.divider.json`, `.request.json`, extraction contract
  and any interrupted atomic-write temporary files;
- `<jobs_dir>/<contract_id>/` contains the retained product, compiler work,
  caches, spill data, diagnostics and lifecycle receipt.

The `site_artifact_context` sent to Nodus contains the authenticated `owner_id`,
existing `workspace_id`, `job_id`, and exact `job_directory`. Nodus validates
this directory against output_dir, confines composition and environment-driven
cache/temp paths, and retains success/failure lifecycle metadata. The Turing
`TURING_COMPILER_ARTIFACT_ROOT` selector relocates otherwise repository-fixed
compiler foundation, native shell and Eigen output defaults. Existing defaults
outside site jobs are preserved.

The jobs root must be genuinely outside every Git worktree, without junctions
or symlink redirection. The old `C:/dev/Powershell/srv/state` path is below the
outer repository's `.git`, so it is not an acceptable new compile output root.
Use an explicitly provisioned user-space directory, such as a directory below
the operator's LocalAppData. `workspace_accounting.for_state` now accepts that
explicit configured root while binding the SAME Accounts database. Its existing
one-root-per-filesystem rule still applies: it will not silently migrate a
pre-existing ledger or change its capacity policy.

## Monitoring and cleanup boundaries

Files are measured after preparation and outcomes, approximately every five
seconds while waiting for Nodus, and every minute while the site worker is
running for retained/interrupted work. Recursive measurement refuses reparse
points, symlinks and multiply-linked/nonregular files. A failed measurement
retains the previous charge and records an attention flag rather than zero.
An observation of mutable files clears the sealed revision: it cannot become a
review snapshot just because it was measured.

The existing job detail API/page shows measured bytes, charged reservation,
status, measurement problems and retention-review dates. Contract ids are
strings in the page API to preserve all 63 bits. Cross-account access remains
the existing 404. Host paths and detailed scan exceptions are not returned.

Review deadlines do not delete files or release charges. A timed-out request is
`unfinished`, not proof the remote compiler stopped. Release requests remain
charged and continue being measured; a new measurement invalidates stale removal
receipts. Actual removal still requires the existing trusted cleanup boundary
to establish no writers, retained/adopted products unloaded or otherwise no
longer needed, and absence before `confirm_removed`. No new deletion endpoint,
automatic expiry deletion, legacy migration or file removal is included.

Not covered by this per-attempt measurement: existing uploaded source/intake
copies in the site's existing jobs tree, old unregistered compiler directories,
shared SQL/WAL/backup overhead, and copies adopted into a separate module library.
These must not be misrepresented as charged by the compile workspace. The
configured compile host should remain non-adopting for this storage scope until
adoption copy accounting and release are integrated.

## Configure, then deploy together

Server startup accepts `--site-compiler-config <operator-owned JSON file>`.
Required keys:

- `jobs_dir`: absolute external user-space root, also the Nodus host jobs_dir.
- `service_id`: this bridge's engine participant id, not the logged-in owner.
- `backend`: `c`, `llvm` or `fortran`.
- `capacity_bytes`, `user_quota_bytes`, `free_floor_bytes`: existing ledger policy.
- `reservation_bytes`: positive per-attempt lifetime reservation.
- `review_after_s`: positive time until retention review is due.

Optional `port` uses `--federation-port` when omitted; the host is always
loopback. Optional `nodus_tools`, `turing_root` and `timeout_s` (at most one hour)
select installed sources and wait timeout. No quota, account, port, retention
period or production path has been guessed or installed. Missing/mismatched
configuration refuses before dispatch. An instance using an unrelated
`--accounts-db` requires a correctly bound state/ledger, not an implicit fallback.

Deploy the paired Nodus adapter and Turing selectors together; the old compiler
will ignore the new root variable. Update the host jobs_dir to the same external
root, preserve existing products in place, and coordinate a lifecycle-safe
Apache/Nodus restart with the service owner. This source change does not authorize
or perform that restart, change ACLs/firewall, or run an actual compile.

This is trusted-process path routing and accounting, not filesystem/process
isolation against arbitrary user Python or concurrent path replacement. The
existing isolation gates still apply before exposing untrusted execution.

## Verification

Tests use disposable state roots, real Accounts/HTTP intake and a fake Nodus
endpoint speaking the real contract frames. They verify signed-in account
ownership end to end, guest denial, other-account 404, forged ownership ignored,
backend selection, quota rollback, schema version refusal, stale leases,
measurement errors, truthful overruns and retained charges. Compiler selector
tests inspect/evaluate path expressions only. No compiler or native build is
executed. Full-suite, real Nodus compile and production rollout are not claimed.

Final gate after rebasing onto Turing d2a7f022 and using SpeakToMe 86721dd's REST
v2: 116 passed and 12 subtests. The fresh-state queue schema is initialized
before the hook/worker starts, preventing the first-request declaration race
found during a repeat. Nodus fake/path/stream tests: 28 passed, one Windows
symlink privilege skip, three real-compiler tests deliberately deselected.
