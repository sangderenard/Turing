# Transient build output only

This directory is for reproducible, disposable output from compiler runs.
It is not persistent storage.

Material formerly kept here was moved on 2026-09-24 because a broad build
cleanup must be safe:

- hand-authored probe and repair scripts moved to `tools/compiler_probes/`;
- documentation-linked logs and images moved to
  `artifacts/compiler_evidence/`;
- resumable compiler checkpoints moved to `artifacts/compiler_checkpoints/`;
- preserved worktree patches and source copies moved to
  `artifacts/worktree_preservation/`.

New compiler runs may write generated sources, binaries, caches, traces, and
checkpoints here while they are in progress. Promote anything that must survive
a cleanup to the appropriate location above, update its documentation links,
and leave only reproducible output in `build/`.

