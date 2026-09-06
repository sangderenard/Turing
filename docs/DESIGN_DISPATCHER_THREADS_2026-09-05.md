# Design note: Thread, Condition and Event as dispatcher-owned constructs

Section 5 of `ACTION_PLAN_2026-09-05_EVENING.md`. Written before code, as
that plan requires. Scope: the whole validator
(`tools.run_vehicle_native_assembly._run_dually_python_profile`) compiled
with the dispatcher owning its threads and the worker/viewer handshake kept.

## What the authored program actually does

`_run_dually_python_profile` (tools/run_vehicle_native_assembly.py, lines
781-1200) uses exactly these constructs:

| Construct | Authored use |
|---|---|
| `stop = threading.Event()` | `stop.set()` in the `finally`; `stop.is_set()` polled by both tasks |
| `status_lock = threading.Condition()` | `with status_lock:` around every read/write of the shared `live` dict and the revision counters; `status_lock.wait(timeout=0.1)` in bounded loops; `status_lock.notify_all()` after every publication |
| `threading.Thread(target=worker, name=..., daemon=True)` | `.start()` once; `.join(timeout=10.0)` in the `finally` |
| `threading.Lock()` | `PythonValidatorViewer._visual_lock` (viewer-internal) |
| shared state | `live` (a dict of scalars and strings), `displayed_attempt[0]`, `visual_revision[0]`, `displayed_visual_revision[0]`, `accepted_clock[0]` |

The handshake: the worker publishes a visual revision under the condition
and waits until the viewer has displayed it (`displayed_visual_revision`);
the viewer draws, then acknowledges under the same condition. Both waits
are bounded by `timeout=0.1` and re-check `stop`. Nothing here needs a
serial-progress proof; it needs two tasks and one condition.

## What already exists

- Control IR: `DispatchBlock` (semantic operation, SSA handle/value ports,
  `deployment_owner=dispatcher`, `required_capability=communicating_tasks`,
  ordered effect) and `ResourceScopeBlock` (`with` body plus cleanup that
  also runs on `return`/`break`/`continue`, tests
  `test_resource_scope_unwinds_nested_handles_on_conditional_return`,
  `test_resource_scope_break_releases_only_scopes_exited_by_loop`).
- Lowering: `precompile_to_ssa` emits `Handler.Dispatch` for both; the
  resource-wait regression proves a `with c: while i < n: c.wait(...)` loop
  keeps its scope and unique SSA definitions.
- Runtime (turing_pool.c/h): `turing_dispatch_condition_{create,acquire,
  release,wait(timeout_ms),notify(count),notify_all,destroy}` and
  `turing_dispatch_event_{create,set,clear,is_set,wait(timeout_ms),destroy}`
  (event implemented over the condition, Python 3.11 semantics for a wait
  that observes a set-then-clear). Frames: `turing_pool_start/deploy/
  deploy_span/stop`, barrier-joined, plus `turing_pool_effect_lock`.
- Nodus `ThreadPool` (`include/common/thread_pool.h`): `enqueue(Job)`,
  `submit_batch(jobs, count) -> JobBatch`, `JobBatch.wait()`. This is the
  async submission model; a job is `(fn, context)` on a worker id.

## What is missing: the Thread contract

Three dispatcher operations, spelled like the condition/event ones:

| Operation | Ports | Semantics |
|---|---|---|
| `thread_create(target, frame) -> handle` | `target` = the id of a dispatcher-owned region (the compiled `worker` closure); `frame` = the captured values (arena handles, condition/event handles, scalars) | allocates a job record; nothing runs |
| `thread_start(handle)` | handle | submits the job once to the pool (`ThreadPool.enqueue` / a one-lane `turing_pool_deploy` that is NOT barrier-joined); a second start is an error, as in Python |
| `thread_join(handle, timeout_ms) -> joined` | handle, timeout | blocks the caller until the job finishes or the timeout elapses (`JobBatch.wait` with a deadline); returns whether it finished |
| `thread_is_alive(handle) -> bool` | handle | job submitted and not finished |
| `thread_destroy(handle)` | handle | frees the record; joining is the caller's responsibility, `daemon=True` means process exit does not wait |

The `target` is a region, not a Python callable: the loop composer already
outlines a region per function; `worker` is one such region whose formals
are the captured names. Captured mutable state (`live`, the revision
counters) is arena storage shared by identity, exactly as a record
parameter is shared today; the only synchronization is the authored
condition, never `turing_pool_effect_lock`.

## Rules

1. One owner: a thread's body is a dispatcher-owned region; the shell never
   runs it inline "once with no effect" (the `CompilationSubdivisionRequired`
   refusal stays for anything the dispatcher cannot admit).
2. Every wait is bounded exactly as authored (`timeout_ms` from the literal
   `0.1`), and every `with` lowers through `ResourceScopeBlock` so a
   `return` inside the scope releases the condition.
3. Effects inside the thread body keep their authored position (the
   placement rules landed 2026-09-05: loop-owned effects installed
   lexically, conditionals nested by span, arm-owned callsites stamped).
4. No host-side band-aid: the viewer's `pygame.draw.*` calls are a
   separate display boundary (continuation, viewer `draw` refusal) and are
   NOT threading work; they go through the geometry wire format / GL
   consumer, not through this contract.

## Regressions to write before the whole-program run (seconds each)

1. `with lock:` + `notify_all` on a Condition, with a `return` inside the
   scope (extends the resource-wait test with the release-on-return path).
2. `t = threading.Thread(target=f); t.start(); t.join()` where `f` writes
   one arena cell: lowered SSA holds `thread_create/start/join`, the target
   region has the captured formal, and the C emission compiles and runs
   (native subprocess with a 20 s bound, like the counter test).
3. `stop = threading.Event()` with `set/is_set/wait(timeout=...)` across
   two tasks: the waiter returns when set, and `is_set` is a runtime
   value, never folded.
4. The handshake itself as a two-task program: worker publishes a revision
   under the condition and waits for the acknowledgement; main acknowledges
   in a bounded loop; both exit on `stop`.

Then `python tools/lower_vehicle_validator_program.py` (about 10 minutes,
one run, detached, receipt checked before any further heavy job).
