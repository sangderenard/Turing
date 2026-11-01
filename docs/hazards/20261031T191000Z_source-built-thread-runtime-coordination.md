# 2026-10-31 Source-built threading runtime coordination (SOURCE-INSPECTED, DESIGN)

User clarified the required architecture: a unified application threading runtime
built from source, with recognized Python redirected to real SSA procedures;
backend-specific compromises occur downstream. Inspected current source-pursuit
and Token Orchard source admission tests, glyph formatter private source admission,
Python synchronization identity declarations, source synchronization receipts and
SSALockBox attachment. Current declarations are primitive markers, not source
runtime implementations. Existing table is descriptors, not allocated runtime data.

Recorded exact reuse path, application runtime/child-table ownership, source
procedure versus primitive boundary, identity-based ingestion mapping and first
vertical slice in docs/compiler_convergence/source_built_thread_runtime_2026-10-31.md.
No compiler code changed. No new runtime procedures, native behavior or test pass
is claimed. Documentation checked with git diff --check; previous runtime test
counts were not rerun. Next implementation: actual source module/table procedures,
worker-body SSA retention and token representation, connected through current
pursuit/concordance to existing native task/mutex leaves; then atomics and deques.
