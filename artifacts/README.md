# Retained local artifacts

This ignored tree holds local material that is intentionally retained across
build cleanups. It is separate from `build/`, whose contents are disposable.

- `compiler_checkpoints/` contains expensive resumable compiler state.
- `compiler_evidence/` contains logs and rendered evidence cited by tracked
  documentation.
- `worktree_preservation/` contains explicitly preserved patches or source
  copies that have not yet been retired.
- `identity_logs/` contains compiler identity audits. Most are disposable;
  only cited or current-run logs should be promoted to `compiler_evidence/`.
- `llvm_pieces/` and `llvm_pieces_internal/` are generated but retained because
  they are live inputs to the LLVM dt-system examples.

Files here remain ignored unless explicitly promoted into a tracked document.
The directory names are the retention contract; do not use this tree as a
second general-purpose build directory.

