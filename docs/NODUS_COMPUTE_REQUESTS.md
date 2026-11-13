# General Nodus compute admission

The existing StudioPictures contract queue is now NodusComputeRequests (the old name
remains an alias). It has one FIFO, four admitted requests maximum, 64 process-local
receipts, and the existing native Nodus contract transport. It is not a new renderer.

POST /api/compute/requests with application/json:

```json
{"request_id":"scene-1","operation":"studio.scene","arguments":{}}
```

Arguments for Studio operations are settings and/or photo overrides. Empty arguments
require configured baked defaults. studio.scene constructs the actual Studio scene;
studio.ray-photo constructs the existing ray payload and commits it into PhotoJobs.
Both currently require photo_submit. Receipt polling uses GET
/api/compute/requests/<compute_id>. A committed scene receipt names its owned
/api/compute/requests/<compute_id>/product JSON envelope; it contains the existing
Studio product blob, not a rendering. Photo results name the existing photo job URL.

Only trusted host code registers ComputeOperation(capability, prepare, build, commit).
Clients cannot select executables, modules, transport addresses or credentials.
Admission snapshots finite JSON, uses account-scoped idempotency and bounded queue
limits. Both receipt and product reads enforce account ownership and operation rights.
Running requests participate in service_jobs restart visibility. Product files are
account-partitioned beneath the untracked state jobs store and removed with receipt
eviction; orphan products expire after 30 minutes. Receipts do not survive restart.
This does not yet merge compile/math queues or offer durable scheduling/cancellation.

Deploy the corresponding SpeakToMe routes/catalog and Turing host together. No service
was restarted for this change.

## Verification, 2026-11-13

Real Studio scene built through an isolated native Nodus canvas/contract port, with
actual default Studio assets/settings: 28,492,804 decoded bytes, committed receipt,
matching idempotent replay, cross-account product read refused. No photo worker/GPU
render started. This proves the real queue/contract/product path, not live authenticated
HTTP end to end. Existing Studio regressions: 18 passed, 1 optional native test skipped.
Shared REST/catalog pytest was blocked by its environment bootstrap trying PyPI,
which network policy denies. No network workaround attempted.
