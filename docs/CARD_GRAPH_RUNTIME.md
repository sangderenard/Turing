# External card graph and lazy read-head runtime

Implemented 2026-08-04.

## Contract

`card_graph.build_card_graph` projects the program map and any lowered
shared-memory class deployment into `turing.card-graph.v1`. The graph contains:

- file functions, class methods, and deployed WebAssembly cards;
- their typed input/output ports and loadable resources;
- class-member navigation links;
- every type-compatible potential callable connection known to Map IR;
- exact compiler-authored resident-memory links between lowered cards; and
- named paths, including the existing topological `linear` path.

Potential connections are deliberately not schedules. A read head chooses a
path or edge; the graph preserves the other legal routes so an OOP environment
can be loaded and traversed incrementally.

## External-link policy

The default `address_policy` is:

```text
arena      = outer-coordinator
cache      = compiled-card
execution  = read-head
rebind     = every-traversal
inputs     = alias
outputs    = alias
```

No card graph contains a process-local pointer. A compiled `WebAssembly.Module`
may be cached across program instances because it owns no arena address. A
`WebAssembly.Instance` may only be cached inside one `ClassGraphRunner`, because
its imported memory permanently binds it to that runner's arena.

Immediately before every card invocation, `rebindCardAliases` unconditionally
rewrites all of that card's input and output slots in the resident field table,
then passes the rewritten addresses to the card. This applies on the first
load, a cache hit, and a later traversal through a different graph edge. Stale
addresses are therefore not part of the cache contract.

The deployed manifest may override `external_link_policy.execution` with
`wasm-coordinator` when it needs the translated autonomous coordinator. The
same outer arena and slot-table ABI remain valid in both modes.

## Browser surface

Every generated HTML shell now carries the projected card graph at
`MAP_IR.card_graph` and exposes `window.TuringCardGraph`. Its
`createReadHead()` method returns a path-traversal read head. Shared-memory
class execution uses the same policy for lazy module compilation,
arena-local instantiation, and alias rebinding.

The existing sequential punch-card inventory is thus one path through the
graph, not the definition of the program's complete reachable environment.

