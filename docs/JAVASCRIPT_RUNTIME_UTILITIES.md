# JavaScript emitter runtime utilities

Generated JavaScript may request named runtime utilities from
`javascript_runtime_utilities.py`. A utility has stable semantic identity,
content identity, dependencies, capability, callable exports, and a performance
label. Dependency closure follows deterministic request order and emits each
utility once.

The first shared utilities are:

- `turing.wasm.registry`: content-addressed WebAssembly module registration and
  lazy, promise-deduplicated instantiation;
- `turing.world.registry`: world-object, containment, ancestry, and semantic
  part lookup;
- `turing.revision.channel`: monotonic edit publication with subscribers.

`emit_ssa_module_to_javascript(..., runtime_utilities=(...))` embeds the
requested closure and exports it as `RUNTIME_UTILITIES`. The emitted API records
the utility identity, content key, exports, and performance facts.

## Performance labels

Every emitted SSA function receives an inspectable performance label. Explicit
function metadata may provide:

```python
metadata={"performance": {
    "inline": "prefer",
    "hot_path": True,
    "frequency": "per-frame",
    "allocation_risk": "none",
}}
```

Inline policy is one of `prefer`, `neutral`, `avoid`, or `forbid`. JavaScript
labels are advisory because the engine owns its JIT. Other backends can map the
same intent to their own inlining controls. Labels also retain structural
evidence: instruction, block, call, and branch counts; async boundaries; and
allocation risk. When no authored label exists, the JavaScript emitter records
that its choice is a structural estimate.

In AbstractUI, performance labels become derived `performance-observation`
objects. They are loosely placed inside the corresponding function or method
domain. They are not rooms, members, or source authority: their containment
only says which program domain the observation describes. This permits visual
hot-path and inline-awareness without corrupting the program hierarchy.
