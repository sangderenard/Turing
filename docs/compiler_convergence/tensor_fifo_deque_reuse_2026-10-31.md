# Tensor FIFOs and faithful Python → SSA → C sequence storage

Scope correction: the PremixRing change/test described below were subsequently
removed at the user’s direction. Spectral histories are not queue dependencies.
The general queue direction is now opaque byte records; see
`byte_queue_c_runtime_and_ssa_tests_2026-10-31.md`. The source call-frame repair
remains independently applicable.

## Measured compiler repair

The source program `rows = make(n); alias = rows; change(alias, 17)` refused
before C emission. `make` returned an ordinary list; `change` wrote its last
item. The actual source graph and call records retained the exact list binding.
At frame ownership selection, the factory's result binding `(5, 1)` was omitted:
only argument bindings proved membership. Its length, capacity, status and heap
owner were consequently classified as private leases. At the next call,
`change`'s argument binding proved those cells belonged to caller sequence 1,
but the incumbent factory leases won and new cells were allocated. Native heap
projection correctly refused the now-contradictory owners. No C printer change
could repair that identity loss.

The existing ownership selector now considers unique exact result bindings as
well as arguments. An input remains the anchor for a returned argument; multiple
result views require additional proof rather than selecting by position. Both
sides must claim the same descriptor member role in the authoritative sequence
member page. Retention goes through `exact_sequence_member_binding` in the
existing frame transformation ledger, derived from the call record and both
membership cells. Distinct results and unrelated arenas remain distinct.

Seven authored programs compile, emit C, compile to native code, execute eight
inputs including repeated invocations, and match Python exactly. They cover
branching helper/loop control, wrapping indexed writes, argument aliases,
append/pop/negative indexing, returned-list mutation, separate returned lists,
and returned-list growth/length publication. Concordance checks run before and
after emission/execution. The returned-alias probe measures 160 rows across eight
functions, zero findings and zero unsourced facts/identities. This uses normal
source ingestion, not fabricated SSA or a container mutation exemption.

## Existing FIFO/ring implementations

| Implementation | Observed storage and operations | Reusable property / limitation |
| --- | --- | --- |
| GLChunkFIFOLane / GLChunkFIFOArena | Typed resident GPU arena views; write/read sequences; empty_output → publish and input → consume; full/empty refusals | Best resident-slot/lease model. SPSC orchestration, not a thread-safe MPMC implementation |
| GLSLFusedProgramNetwork | Lane per producer/consumer edge; fanout gets separate consumer lanes | Preserve exact tensor value identity and each consumer's lifetime; residency across computation |
| RingBuffer in fs_harness | Tensor clone + scatter_row, monotonic idx/count, returns evicted tensor row | Retains differentiable history and evicted values; overwrite policy rather than backpressure |
| PremixRing | Tensor clone + scatter_row; chronological spectral window consumer | Logical write count must remain monotonic; fixed wraparound bug described below |
| ParamRing / ParamWheel | Parameter snapshots, tick/lineage and rotating slots | A physical slot is not a parameter version or logical time; lifetime/reuse proof matters |
| TensorRingBuffer in demo_spectral_routing | Named tensor columns, cloned entries, chronological stacked snapshots, flush hook on eviction | Columnar records and explicit copy/snapshot/flush semantics; Python list storage here |
| SlotBackpropQueue | Slots selected from tick/row; residual accumulation and job lists pinned to parameter versions | Queue identity includes row/version, not merely modulo slot; preserve gradients and source lineage |
| OpQueue / QueueResultSink | Condition-protected deque, bounded producer waiting, batch consumption, close/notification | Existing blocking/MPMC/close policies; these hold tensor-bearing jobs/results, not resident tensor arenas |
| YoungMan streaming spline engines | Immutable copied tensor/array batches queued through queue.Queue; drain a current prefix and publish a model generation | Batch consistency and publication generations; not zero-copy resident FIFO storage |
| DeploymentFIFO | Bounded Python single-reader helper with publish failure on full and availability/value consume | Explicit full/empty behavior, distinct from deque(maxlen) eviction |
| RingBufferABI in shell_io | Shared-memory SPSC header: capacity/read/write/dropped; fixed record layout | ABI organization; header declarations alone do not establish atomic publication |

The native `turing_stream_buffer` is an accumulating text-publication sink, not
a general tensor deque. Deque3D is a three-double ctypes record. Neither should
be mistaken for the resident FIFO mechanisms above.

## Measured tensor-ring repair

For capacity 3, pushing 1,2,3 gives spectral window [1,2,3]. Pushing 4 previously
reset PremixRing.idx to 1, and gather_recent_windows returned no full window,
instead of [2,3,4]. The unchanged baseline reproduces this. PremixRing now keeps
idx monotonic and applies modulo only when selecting the physical slot. Tensor
clone/scatter and the autograd path are unchanged. New tests use actual
RingHarness/gather_recent_windows across several wraps at capacities 1,3,5.

## Deque direction (proposal)

Reuse SSASequenceDescriptor's typed column arenas, row shapes, length, capacity,
status and heap owner, with exact call-frame storage binding. Extend the logical
sequence state to name its front and occupied extent. A logical row maps to a
wrapped physical slot; a deque's left/right operations update its logical bounds.
Do not confuse occupancy with total publications or a wrapped physical index.

Preserve these as separate policy dimensions:

- Row schema: scalar or shaped tensor payload, dtype/alignment and column layout.
- Endpoint operation: left/right push/pop, nonblocking try result, batch operation.
- Capacity: grow, reject/block when full, or opposite-end eviction as required by
  authored collections.deque(maxlen). Empty versus a legitimate None is explicit.
- Ownership: borrowed resident view, transferred lease, copied snapshot, or
  differentiable tensor history. A slot cannot be reused while a required view
  remains live. Autograd history cannot silently become destructive mutation.
- Synchronization: serial, proved SPSC, or mutex-protected shared state. Reservation
  is distinct from publishing initialized payload; consume/retire precedes reuse.
  Concurrent reservations need committed/readiness state, not just atomic cursor
  increments. Use the lock box for an initial correctness implementation.
- Progress/lifecycle: waiting, timeout, close, cancellation and destruction, where
  the source API specifies them. A backend needs enough progress to service waits.

An authored deque and a dispatcher work-stealing deque share storage mechanics,
but owner/thief rules and last-item races are an additional contract. Reuse the
storage substrate first; do not infer work-stealing guarantees from SPSC lanes.

The compiled source tests establish ordinary sequence fidelity, not a new deque
implementation. Next boundary: describe deque endpoints/leases in the existing
sequence model, bind exact collections.deque identities, then lower storage and
mutex/atomic transitions in C with source-driven empty/full/wrap/alias/lifetime
and producer/consumer tests. Hardware-specific specialization remains downstream.

## Remaining observed limits

A small authored Ring class with a list field still refuses an Indexed value's
missing source_value_class/planned dtype. Unannotated mapping audit still refuses
Add 7's unsettled dtype. These reproduce in current source; the Ring case also
reproduces in the untouched baseline. Neither is hidden with a type default.
GPU FIFO integration tests pass with PYOPENGL_PLATFORM=egl in this environment;
default platform selection instead fails glProgramParameteri lookup. Another
existing graph-built FIFO shell test refuses missing ingestion provenance on both
worktrees. These are separate from successful resident-lane runtime checks.
