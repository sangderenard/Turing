# Translation boundary namespaces

This directory is the default home for project-local translation boundaries.
Run a source file through the full compiler and reveal every compiler mutation
in order while FluxSpring continues stepping:

```powershell
py -3.11 -m src.rendering.precompiled_graph_demo `
  --source examples/live_compile/spectral_route.py `
  --entrypoint spectral_route `
  --source-language python `
  --release-hz 30 `
  --event-trace
```

Each `[translation-event]` line is applied to the visible graph at that moment:
`component-spawn` adds one node and `component-link`/`component-handoff` adds
the corresponding edge. The queue never coalesces revisions. If its bounded
backlog fills, the compiler thread pauses while the spring physics keeps
playing, then resumes as events are displayed.

When the emergency growth clamp trips, the failed attempt remains visible and
the compiler writes `.growth_flags/*.flag.json`. Saving a precise
`*.node.json` boundary rule changes the namespace fingerprint and restarts the
translation into the same live event ledger; the existing physical graph is
not cleared between attempts.

Resolution starts in the language directory and follows the authored lexical
OOP scope. Missing directories are skipped rather than treated as errors. For
this source:

```python
class Model:
    def forward(self, value): ...
```

the resolver checks, in order:

```text
boundary_namespaces/
└── python/
    ├── *.node.json
    └── Model/                 # optional
        ├── *.node.json
        └── forward/           # optional
            └── *.node.json
```

If `Model/` is absent but `python/forward/` exists, `forward/` is still
checked. This permits sparse boundaries without placeholder directories.

Only declarative `*.node.json` files are loaded. Python modules, shared
libraries, and executable hooks are never imported from this tree.

## Schema record

```json
{
  "version": 1,
  "id": "python.Model.CustomNode",
  "action": "schema",
  "node_type": "CustomNode",
  "role_schema": {
    "up": {"left": 1, "right": 1},
    "down": {}
  }
}
```

The role schema is selected for that node and then interpreted by the normal
`ProcessGraph` walker. It does not create a separate traversal.

## Spoof record

```json
{
  "version": 1,
  "id": "python.Model.forward.external_runtime",
  "action": "spoof",
  "node_type": "Call",
  "match": {"func.id": "external_runtime"},
  "graph_match": {"class_definitions": ["Model"]},
  "result": {
    "type": "opaque_external_boundary",
    "attributes": {"reason": "trusted boundary"},
    "attributes_from_node": {"callee": "func.id"}
  }
}
```

`match` reads exact attribute paths from the input node. `graph_match` reads a
frozen view containing `language`, `class_definitions`, and `map_ir`. A match
returns the existing ingestion `SpecialCase` contract, collapsing that node at
the same seam used for bulk spans, casts, and publications.

## Precise exclusion

```json
{
  "version": 1,
  "id": "python.Model.no_default_runtime_spoof",
  "action": "exclude",
  "target": "python.default.external_runtime"
}
```

An exclusion removes only the exact inherited rule ID. A deeper record may
reintroduce the same ID deliberately. Node-type wildcards and directory code
execution are not supported.

Every applied schema/spoof writes a receipt into
`ProcessGraph.G.graph["boundary_namespace_receipts"]`. The live top-K growth
report includes matched rule IDs or prints `boundary=unmatched`, making the
next divergence point visible without changing FluxSpring physics.

Program implementation provenance is controlled separately by
`../extraction_contracts/program_extraction.yaml`. Use that exhaustive sheet to
decide whether resolved Python, builtin, extension, or DLL callables are
ingested, intrinsic, retained in place, explicitly decompiled, or rejected.
