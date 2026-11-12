# Living Data and site controls source handoff (2026-11-11)

Source checkpoint for the next contributor, based on main d6b0853c. No deployment,
server restart, production database migration, or generated live asset update was
performed. Earlier documents saying "not published" describe their implementation
checkpoint; this handoff publishes source only, not an operational rollout.

## Included

- Living Data bounded graph query, account-owned graph cursors, and static DM REST
  commands through the existing account/capability checks.
- StorybookEnvironment and canonical static map production, existing Phong host,
  realized native mesh capture, OBJ/MTL export and provenance.
- Explicit standalone Studio scene/camera runner and Living Data composition
  adapter using the captured Studio rig. Composition is not appearance-faithful
  native-map replay; existing lighting limits remain documented.
- Account-owned SQL issue tickets and UI, shared site banner, and reusable control
  suite registration/telemetry using existing page controls.
- Focused tests and implementation/verification documents for these features.

## Deliberately unfinished

Physical actor-only/full-world stepping and mechanical actor service are not
implemented. Static DM operations do not advance simulation. Direct standalone
Studio scene-and-camera REST is not implemented; the standalone API is JavaScript.
Generated captures and browser output in ignored build directories are not source
artifacts and were not committed. Browser tests that require those captures must
materialize them through the documented staging/capture workflow.

## Other work preserved locally, not included

Concurrent repository-world mapping: repo_world_scan.py and its tests; portal_record
and repo_world_refs_*; worldmap server/UI/export integration; compile_repo_world_map;
and their tests and documents. Shared server.py and web_export.py were staged with
only this checkpoint's hunks. Their working copies retain all worldmap additions.
.claude worktrees and canvas_workspace.txt were untouched and untracked.

## Fresh publication verification

An isolated worktree containing exactly the staged source, without concurrent
worldmap additions, passed 67 focused Python tests and 10 subtests. Three tests
skipped because captured build fixtures are not present in a fresh checkout.
The initial isolation exposed tests importing an uncommitted worldmap test helper;
four imports now reuse the equivalent committed profile-dashboard HTTP fixtures.
No production behavior was changed to address that test dependency.

Control-suite and standalone camera Node checks passed. Existing Studio subject
renderer (21 negative contracts) and scene asset (18 negative contracts) checks
passed. Composition validation passed against the canonical local Studio fixture;
its clean-checkout run explicitly skipped the missing fixture. No new browser/GPU
run was performed during this publication pass. Earlier browser results are in
feature receipts and are not represented as freshly rerun.

A wider initial dirty-tree run had 209 passes and 10 subtests; four excluded
compile_repo_world_map tests failed when sandboxed MSYS git clone could not create
its Windows named object. No full repository suite or compiler-wide gate was run.
Staged diff whitespace check and credential-pattern scan passed. Remote CI must
be checked separately for the resulting commit; this is not a deployment receipt.
