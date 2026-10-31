# Thread simultaneity as an ingestion graph region

The follow-on
`thread_independent_scheduling_and_sync_operators_2026-10-31.md` settles the
schedule contract: the user gives up a known comparative schedule between
threads. The region keeps both paths independent; ordinary overlapping
operators do not induce a cross-path ordering edge. Synchronization is an
explicit graph-operator concern.

This is a measured design demonstration, not an implemented compiler pass. It
uses the ordinary Python program in
`tests/test_python_two_threads_shared_location.py` and the sanctioned
`lower_ast_source_to_ssa` source entry. The proposed operation is on the
ProcessGraph/control dual IR **before** structural value folding and SSA
lowering. It does not schedule native execution.

## Present graph, measured on 2026-10-31

The source graph has two function references: `root` is 0 and `worker` is 1.
The following are graph-local node IDs, not globally interchangeable IDs.

| Scope | Node | Present meaning |
| --- | ---: | --- |
| root | 4 | the authored `[0]` allocation bound to `shared` |
| root | 6 | the post-join `shared[0]` read |
| root | 7 | `StaticReference` to worker function reference 1 |
| root | 8, 9 | `thread_create` calls, each with edge `7 -> create` at `kw:target` |
| root | 10, 11 | `thread_start` calls, each with `create -> start` at `arg:0` |
| root | 12, 13 | `thread_join` calls, each with `create -> join` at `arg:0` |
| worker | 0 | `Input(shared, binding_kind=closure)` |
| worker | 3, 5, 6 | `Indexed` read, `Add`, `IndexedStore`; both accesses use input 0 as base |

`NAME_BINDING` in the root lexical read scope records `shared`, version 0,
with canonical value 4. `NAME_BINDING` in the worker lexical read scope
records its own `shared`, version 0, with canonical value 0. The
`NAME_BINDING_CAPTURE` page has no row connecting them. A shared spelling is
not an identity proof. The present worker graph therefore has no exact edge
from its closure input to root node 4. The target function reference is
concorded later as `SOURCE_DISPATCH_TARGET`; that is too late for the current
structural fold. These are observations from the current checkout, not from
an older log.

The ProcessGraph edges above are value/argument edges. The ordered dispatch
operations also exist in control IR. The run-to-join boundary must be found
using both halves of dual IR; a range of source lines or directed reachability
over ProcessGraph value edges alone is insufficient.

## The graph object to construct

Let a region contain typed node identities `(function reference, node ID)`
and typed edges for values, control, activation, capture, and storage. A
thread activation identity is the exact `thread_create` node, so this example
has activations 8 and 9 of the same worker definition 1. An activation is not
a cloned source function or a separately allocated copy of `shared`.

The first required transformation is a concorded capture relation:

```text
(worker ref 1, closure input 0)
    -> exact root binding cell for shared at the worker's closure creation
    -> root storage identity created by node 4
```

The relation must be obtained while the enclosing lexical environment and
its binding version are known. Looking up `shared` by its name after the fact
would be wrong if it were rebound, shadowed, or captured by two different
closure instances. The source binding cell, worker input cell, function
address, and closure creation occurrence must all be provenance of the new
concordance fact. A later compiler stage must consume that fact rather than
reconstruct the relationship from a spelling.

For this program the candidate region is bounded by starts 10/11 and joins
12/13, matched by the exact handle edges from creates 8/9. The region holds
both activation paths, their common worker body, the parent continuation,
their capture connections, and the relevant storage identity. The paths
coexist as graph structure. No serial ordering between the worker operations
is inferred from the order of the `start` source statements.

```text
root storage 4 ----------------------------- root read 6
     |                                            ^
     +-- capture --> worker input 0              | after joins
                       |                          |
                    read 3 -> add 5 -> store 6   |
                       ^                 ^        |
                    activation 8      activation 9
                    start 10          start 11
                    join 12           join 13
```

The picture abbreviates two dynamic activations of one worker body. A graph
representation may share the body and distinguish activation identities; it
must not claim that the single drawn store happens only once.

## Ingestion operation and its unresolved policy

1. Resolve each start and join through the same exact thread handle.
2. Resolve each create's target through its exact function-address fact.
3. Construct the region over both activation paths and the parent path,
   retaining every value, control, capture, and storage edge crossing its
   boundary. Record region membership and each new edge in the concordance.
4. Before any fold, make the worker's indexed store a possible effect on the
   storage reached by root node 4. Root node 6 must remain a read of that
   storage. The current fold aliases node 6 to the literal 0 because it sees
   neither a root-local store nor an escaped argument.
5. Inspect operator pairs belonging to distinct activation paths only after
   exact operand/storage identities and graph dependencies have been
   established. Preserve both operators and their provenance. A detected
   pair does not itself create a comparative schedule. Only an explicit
   synchronization operator or declared policy may add a cross-path
   dependency, and that decision must be concorded before SSA consumes it.

The term **double operator** and the policy entries still need an exact
contract. This document does not equate it silently with any two operations
that share a name or any two accesses to a location. A policy entry must say
which typed operator pair it recognizes, which identity relation it requires,
which rewrites are legal, and whether it produces any graph edges. An
unhandled pair cannot be silently ordered or dropped; it remains independent
unless an explicit contract requires a source-located refusal.

## Observable acceptance boundary

For this source program, a successful ingestion demonstration must show:

* one concorded connection from each worker closure instance to the exact
  parent binding/storage identity, with no name-only lookup;
* one region containing both start-to-join activation paths and the worker
  read/add/store body, with all graph edges retained;
* recorded double-operator candidates with no invented comparative order,
  and concorded graph constraints for any explicit synchronization;
* a post-join root read that has not been folded to the initializer leaf;
* SSA with one worker function body and two distinct activation identities
  whose captured storage identity is the same.

The current test measures 2 passes and 1 expected failure: target identity
survives into SSA, while the worker body and shared access do not. The current
`mapping` concordance audit refuses an unrelated unsettled dtype in its own
input (`region value 7`, `Add in function root`); it does not establish a
clean audit for this program. No native scheduling or output value is claimed
by this demonstration.
