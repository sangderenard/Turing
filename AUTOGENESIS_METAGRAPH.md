# Compiler autogenesis metagraph

The evolution metagraph is an optional, append-only provenance ledger beside
the compiler. It is not an IR and owns no compiler semantics. With no active
`record_evolution` context, its hot-loop hooks are no-ops.

## Event model

Each compiler representation opens a graph identified by stage and label.
Components have only a graph ID, a local ID, display metadata, and optional
references to components they consumed. The ledger publishes graph-open,
graph-close, component-spawn, component-update, component-link, and
component-handoff events under a small reentrant lock. Subscribers receive
immutable events after the lock is released.

`bind_artifact` and `bind_component` retain only Python object identities, not
the objects. They let the next real lowering hot loop recover the source graph
or source component without adding provenance fields to every IR dataclass.

## Current transformation path

The default demo runs the real compiler path:

```text
Python AST ingestion
  -> semantic ProcessGraph
  -> numeric FusedProgram precompile + control IR
  -> repository SSA
  -> IR package
  -> SSA/WebGL adapter
  -> GLSL ES WebGL artifact
```

ProcessGraph nodes and edges are observed where they are created. Numeric
ProcessGraph, precompile, SSA, WebGL-adapter, and GLSL nodes share retained
value identities, so those handoffs have `granularity=exact-value`. Control
dataclass blocks are bound to components and become the explicit sources of
the SSA instructions emitted while each block is active.

Packaging currently exposes functions but no record/line correspondence, so
SSA-to-package is deliberately marked `granularity=whole-function`. Registered
machine targets other than WebGL still consume `FusedProgram`; their fallback
handoff is honestly named `precompile-to-<target>` and marked
`granularity=whole-artifact`. They must not be presented as SSA consumers until
they gain a real SSA adapter.

## Visualization and concurrency

Compilation publishes at full speed into the thread-safe ledger. The renderer
buffers those events in a deque and releases them at `--release-hz`, allowing
the compiler and visualization graphs to run concurrently without slowing the
compiler hot loop. A handoff asks `MultiNetworkFluxSpring` to birth the target
at its source component's current physical position. FluxSpring owns the
growth envelope, network placement, DEC geometry, and live edge
control/transport activation. The compiler path hands FluxSpring's live
`SpringRepulsorSystem` directly to `LiveVizGLPoints`; it does not translate the
display through the generic renderer.

`LiveVizGLPoints` owns the window, shaders, GL buffers, colormaps, camera,
autoscale, point packing, edge packing, event handling, and draw loop. Its
colorful circular nodes receive live network/control state from
`MultiNetworkFluxSpring`.

Run the complete path with:

```powershell
python -m src.rendering.precompiled_graph_demo --release-hz 4
```

## Backend extension rule

A backend adapter should attach observation at the loop that performs the real
translation. Reuse existing SSA handlers and backend translation tables. Do
not introduce provenance-only operators, preserve unsupported source merely for
the visualization, or infer a line-level correspondence an emitter does not
provide. Unsupported constructs continue through the repository's existing
shortfall/error reporting.

Desktop GLSL remains special at the memory boundary. Its SSBO binding/channel
limits require the dedicated GLSL arena memory-handler specialization already
marked in `glsl_source_ingestion.py`; arena mechanics must lower through
existing SSA memory operations rather than becoming numerical opcodes.
