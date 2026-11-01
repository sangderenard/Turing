# Byte deque: native C contract and threading tests

## Implemented C foundation

`turing_byte_queue.c/.h` implements a bounded deque of copied byte records.
It reuses the existing native condition/lock ABI in turing_pool. It does not
create a service thread. The queue owns storage, bounds, a closed flag and wait
state. Calling tasks perform operations under the private monitor; waiting
releases it atomically and reacquires it before checking readiness again.

Creation specifies slot count and maximum bytes per record. Every record carries
its actual length, an opaque type tag, an opaque operation-contract tag, and the
publishing logical worker ID. Tags are convenience/provenance data, not payload
interpreters. Zero-length records are valid and distinct from empty. Binary data
may contain every byte value. No tensor schema, dtype or tensor subsystem is used.

An endpoint is `(queue handle, left/right, logical worker ID)`, held by the caller.
It is not an exclusive reservation of that end and does not bind a physical
thread. Multiple workers can anchor the same end; endpoints can be used by a
native task or on one physical thread servicing several logical contexts. The
queue must outlive all endpoint uses. Worker identity is carried as metadata;
the runtime does not authenticate it against OS-thread identity.

Push/pop on either end are linearized by the monitor. Push copies bytes before
publication; pop copies into caller-owned storage before retiring the slot.
No ring storage view escapes. Operations are constant-time in queue length,
plus payload copy and wakeup work. No throughput or fairness claim is made.

| Status | Meaning |
| --- | --- |
| 1 | Operation succeeded |
| 0 | Empty/full try operation, or bounded wait expired without readiness |
| -1 | Invalid operand, invalid timeout, or synchronization error |
| -2 | Closed push, or closed and drained pop |
| -3 | Pop destination too small; required length/metadata returned, nothing consumed |
| -4 | Payload exceeds maximum record size; nothing published |

Input validation precedes state checks. Timeout 0 tries once; -1 waits;
positive milliseconds use a monotonic total budget across wakeups; lower negative
values are invalid. Closing is idempotent, wakes producers and consumers, rejects
valid future pushes and permits draining. Size/waiter counts are snapshots, not
reservations. Retry after BUFFER_SMALL is not guaranteed to see the same record
when another consumer is active. Destruction requires exclusive lifetime ownership:
close, complete/join participants, then destroy. No cancellation/reopen is defined.

## Executable tests

`tests/test_byte_queue_runtime.py` compiles the actual queue and existing native
task runtime. Ordinary API tests inspect bytes/status/metadata; threaded probes
execute compiled C task entries, not Python callbacks or an emulated queue.
Concurrent probes run in bounded subprocesses to contain a runtime deadlock.
The test library's direct ABI calls are runtime tests, not a source compiler path.

| Scenario | Executed assertion |
| --- | --- |
| Right push / left pop | FIFO order, binary data and tags |
| Left push / right pop | Reverse-oriented FIFO |
| Same-end push/pop, both ends | Stack order |
| Mixed ends over 120 operations | Wraparound, exact size and reference deque order |
| Endpoint aliases / arbitrary worker IDs | Same queue storage, independent logical contexts |
| Full / oversized push | No publication or overwrite on failure |
| Undersized pop buffer | Required length/tags, no consumption, successful retry |
| Empty pop | Caller bytes and metadata remain untouched |
| Copy isolation | Mutating source and reusing queue slots do not alter copied results |
| Zero-byte slots/records | Real record and metadata despite no payload bytes |
| Close/drain | Pending records preserved; drained queue reports closed |
| Timeout/invalid arguments | Defined status and intact queue state |
| SPSC, capacity 1 and reverse endpoints | Strict sequential delivery across hundreds of waits |
| MPSC / SPMC / MPMC | Each message delivered exactly once; payload checksum and tags intact |
| Same-end and mixed-end concurrent traffic | No loss/duplication under competing endpoint operations |
| Close while producer/consumer is blocked | Registered waiter wakes and reports closed |

The concurrent suite has nine producer/consumer/end configurations, up to four
producers/four consumers, 600 records per producer. The parent closes only after
producers complete and consumers drain before joining. SPSC asserts strict FIFO
where applicable; MPMC asserts legal conservation without demanding a race winner
or assuming completion order equals operation linearization order. Close tests
observe pending waits under the actual queue monitor before closing; no timing
sleep substitutes for wait registration. Tests do not establish lock freedom or
prove every possible interleaving.

## Python → SSA → C test design: next boundary, not implemented

Ordinary `collections.deque` must be recognized by exact constructor/receiver
identity, including import aliases and captures. Same-spelled unrelated methods
remain ordinary calls. The byte queue is not yet wired into that ingestion path,
SSA operator emission or C artifact linking. No hand-constructed SSA, xfail or
runtime-only test is offered as proof that those stages work.

The coordinated source-driven scenarios for that connection are:

| Authored source scenario | Required graph/SSA assertion | Required native assertion |
| --- | --- | --- |
| deque binding captured by producer and consumer | One exact queue identity; distinct logical contexts; exact capture edges | Payload publication and consumption preserve bytes |
| Alias passed through a helper | Endpoint/queue storage survives call and return unchanged | Same queue observed through both aliases |
| append/appendleft/pop/popleft combinations | Explicit endpoint operand and result/effect projections | Four endpoint orderings match Python |
| Binary and empty bytes | Payload pointer/length separate from success/empty status | Embedded NULs and zero-length records round-trip |
| Byte record with type/contract metadata | Tags carried as operands/results without type-based queue allocation | Tags/length preserved by every backend lowering |
| Two producers/two consumers | Context-internal order and lifecycle edges retained; no invented comparative worker edge | No loss/duplication; joins precede destruction |
| Single worker serving several contexts | Logical IDs remain distinct; waits expose suspension/readiness | Another runnable context can satisfy a wait |
| Consumer begins before producer | Blocking control projection and memory acquire derive from same source operation | No lost wakeup; observing published payload |
| Close with pending work | Close, drain, completion and destruction are distinct facts | Waiters wake and accepted records survive |
| Returned queue/endpoints | Exact object identity and ownership across SSA call frames | No duplicated storage/owner on later operations |
| Unsupported payload/API/scope | Explicit source-located capability refusal | No silent object conversion or incorrect C fallback |

For each scenario: ingest actual Python through lower_ast_source_to_ssa, inspect
the actual dual graphs, check concordance, verify actual SSA calls/operators,
emit/link C with the queue ABI, compile and execute, and compare observable source
semantics. Test ASAP and ALAP on the same graph: each respects required edges,
neither makes queue contention an authored cross-worker ordering guarantee.

Queue storage and endpoint handles need declared SSA representations and exact
concordance identity. A blocking operation has control readiness and memory-effect
projections from one source/operation contract. Successful publication orders
payload before availability; successful consumption observes it before copying;
full/empty attempts do not fabricate a synchronization edge. The native monitor
realizes these constraints without assigning hardware lanes in the source graph.

Standard deque semantics must remain exact. Default deque grows; deque(maxlen=N)
evicts at the opposite end; pop on empty raises IndexError. The current bounded
try/wait ABI does not implement those policies automatically. Ingestion must select
an explicitly implemented policy or refuse it; it cannot silently substitute
WOULD_BLOCK for append, or an empty status for an authored exception. Growth,
evicting append transactions and error/result adaptation are next implementation
work alongside byte-span ingestion and worker-body SSA retention.

## Scope correction

The unrelated PremixRing change and its added test were removed. The queue uses
no spectral/autograd code. Earlier tensor-ring observations remain historical,
not dependencies or prerequisites of this queue. The independent returned-sequence
call-frame repair and source-to-C fidelity tests remain in the worktree.
