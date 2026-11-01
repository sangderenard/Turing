# Thread management table: implemented source and SSA records

Current correction (2026-10-31): source/SSA/native thread table acceptance
still passes (8 table cases and 10 standalone C cases). Process operation,
operand, callable, capture-layout, formal-order and C call-contract readers
have been moved to concordance. The former AST exception path and synthesized
run-to-join region have been removed. These successes do not establish native
try/except or the missing paired barrier region. The remaining reader/writer
audit is recorded in the newest hazards receipt.

`IRModule.thread_table` is an `SSAThreadTable`, backed by the compilation's
identity book. It records definitions separately from activations: two authored
Thread constructors can reference one worker body and the same captured storage,
while retaining different create tokens and handles. A constructor in a loop is
one factory site, not one statically allocated native worker.

## Implemented representation

| View | Key | Recorded facts |
| --- | --- | --- |
| `bodies` | Exact FunctionTable address cell | Function reference, every ingested body node identity, exact closure capture cells |
| `activations` | Exact create occurrence cell | Body cell, parent function reference, source operands and their roles, direct start/join cells, authored control scopes |
| `ssa_bodies` | Body cell and lowered function symbol | Source function reference, closure formals, entry block; provenance includes actual SSA signature cells |
| `ssa_activations` | Parent SSA function and create site | Source activation, runtime handle value, operand values, exact lowered body cells |
| `contexts` | Source activation, capture, caller and worker | Exact parent storage and worker input IDs, physical member claims from the existing sequence/record tables, live or unused signature status |

`source_body(body_cell)` reads the actual ProcessGraph from FunctionTable.
`worker_functions(body_cell)` reads actual Function objects from the SSA module.
These are references to existing representations, not copied bodies or a Python
execution facility. Body selection excludes ordinary callsite specializations.

The declared per-handle state columns are worker_token, body, arguments,
closure_context, lifecycle, completion, waiters and result. This is the state
contract; native allocation and state transitions are not implemented here.

## Ingestion and planning

After source topology reduction, each recognized ThreadCreate joins its exact
target operand to the FunctionTable address and retained body. Captures read
SOURCE_CLOSURE_CAPTURE and NAME_BINDING; matching variable spelling is not a
binding operation. Explicit capture edges are added through the existing operand
writer, under the registered thread_context_binding transformation. Capture-use
facts record whether the worker may mutate the referenced storage.

The planner traverses these body targets as worker activations, separately from
ordinary call edges and public entrypoints. The existing numerical region,
control, sequence and call-linking paths lower those workers. The target address
stays in the table rather than becoming a host callable input to ThreadCreate.

Captured numeric values retain exact source class receipts. Captured sequences
retain their storage and receive their element class through the parent binding;
the existing writer-checked type fixed point supplies the physical column type.
A retained collection cannot be replaced by its initializer at a worker boundary.
This also preserves the parent's post-join load after a worker mutation. No
cross-worker comparative ordering, lock or synchronization is inferred.

At SSA publication, context member claims come from the existing descriptor
tables. If a lowered worker has no physical formal for a capture (for example a
zero-field receiver), that unused context is recorded explicitly and contributes
no fabricated SSA operand. Live contexts keep the actual storage operand.

## Verified scope and remaining boundary

Tests compile authored Python through lower_ast_source_to_ssa and inspect actual
module bodies, capture and lifecycle records, sequence members and audits. Cases
include shared mutation by two threads, read-only sequence/scalar captures,
a worker with an argument and branch, a create site inside a loop, an unrelated
wait method, and a worker also reached through an ordinary specialized call.
The original missing-worker-body strict xfail now passes and was removed.

This change does not implement native task launch/join, context marshalling or
lifetime servicing in the C emitter. Constructor args/kwargs are retained as
source/SSA operands; general argument-pack-to-worker ABI lowering remains next.
Direct handle edges supply the lifecycle facts currently catalogued; general
runtime handle-container routing needs its existing identity flow extended.
A capture whose storage belongs to a different enclosing activation currently
refuses explicitly rather than guessing a context. Python Barrier policy and
native barrier/fence servicing remain separate work; table construction does not
establish those execution semantics.

## Subsequent measured C slice

The earlier boundary above records the ingestion milestone. C lowering now
consumes this table for native task create/start/join/is_alive, using the
existing task ABI. Captured bounded sequence views retain their actual storage;
scalar captures and authored tuple arguments are copied per activation. The
exact tuple member/formal relation is posted on SOURCE_THREAD_ARGUMENT before
SSA pack expansion. A create in a loop copies the current induction value.
C_THREAD_ACTIVATION records the worker, physical argument bindings, and the
proof that normal owner return follows join. Escaping captured activations
refuse. A created but unstarted task executes no worker body.

The runtime support lives in turing_thread_table.c/h and uses turing_pool.c/h's
existing native task implementation. Worker failures retain native records;
join completes normally and publishes the record on the shell reporting
channel. Thread-local handling can consume an operator error before it reaches
that channel. This is not a complete queue-based error bus.

Tests/test_source_c_threads.py compiles authored source to standalone C shells
and runs separate executables. It covers captured writes, two disjoint workers,
scalar arguments, create without start, repeated joins, invalid lifecycle,
worker-origin errors, and a loop factory. Tests/test_source_c_exceptions.py adds
worker-local handlers and try/except/else/finally inside a worker loop. General
kwargs, timeouts, dynamic context ownership, unjoined escaping task lifetimes,
and the full queue benchmark remain outside this verified C subset.
