# Ingestion scope audit

**Date:** 2026-08-06
**Repository:** `C:\dev\Powershell\turing`
**Trigger:** `KeyError('type')` building a page from
`examples/x86_read_head_page.py`, traced to module-level table construction
having no scope that owns it.

## Why this audit exists

The compiler models *some* program scopes as real containers and others not
at all. The failure below is not a bug in a backend; it is what happens when
a construct is ingested into a graph that has no owner for it. Fixing the
symptom without knowing which scopes exist would just move the crash.

The scope ladder, innermost to outermost:

    expression -> function -> nested function -> class -> file -> module -> package

## What exists today, measured

| Scope | Real container? | Where | Notes |
| --- | --- | --- | --- |
| **expression** | yes | graph nodes in `graph.G` | the working level |
| **function / lambda** | **yes, fully** | `FunctionTable` entry: integer `address`, own `graph`, own `python_bindings`, parameters, return values | the only scope with a real address |
| **nested function** | **yes** | `lexical_parent_by_function`, qualified as `parent.<locals>.name` | genuine lexical nesting |
| **class** | **partial** | `method_owners` (a *string* prefix `Class.name`); `graph.G.graph["class_table"]` with `methods` -> address, `fields`, `field_defaults`; `ClassNavigationTable` with per-member `slot` | methods are addressable; the class itself is **not** an addressable callable entity |
| **file** | **NO** | — | nothing anywhere models a file |
| **module** | **NO (only a string, and only sometimes)** | `_python_source_identity` -> `(module_name, python_qualified)`, used solely to *spell* a qualified name; `"module"` in `_shell_profile_name` is a fallback **label**, not a container | see below |
| **package** | **NO** | — | — |

### The module level, precisely

Module scope exists in exactly two degenerate forms:

1. `lexical_qualified_name` (topological_reducer.py:2399) reads
   `_python_source_identity` off a definition and joins
   `module_name.python_qualified` into a string. This is a *naming* input,
   available only for objects already imported into Python. It creates no
   container, owns no statements, and has no address.
2. `_shell_profile_name` (glsl_deployment_strategy.py:719) returns the
   literal `"module"` when a graph has neither `function_name` nor
   `program_name` -- a display fallback.

**Module-level statements have no owner at all.** They land in the root
graph implicitly, as "whatever was not captured into a function subgraph."
The root graph has no identity, no `FunctionTable` entry, no address, and --
critically -- **no defined execution order semantics**. There is no object
that says "these statements run, in this order, once, before anything else."

## The failure this produced

`examples/x86_read_head_page.py` has, at module level:

```python
CONFIG = X86ReadHeadConfig.from_rows((...))
HEAD = X86TensorReadHead(CONFIG)
```

Whole-program tracing follows `from_rows` into
`x86_tensor_read_head.py:189-264`, which allocates the decoder's lookup
tables:

```python
opcode_token = [[-1] * 256 for _ in range(map_count)]
prefix_bit   = [0] * 256
```

All 16 phantom nodes are exactly these, and nothing else. Two distinct
defects, now separable:

**1. Semantic (root cause).** `[0] * 256` is **Python list replication, not
arithmetic**. The AST says `BinOp(op=Mult)`, so the canonicalization pass
(topological_reducer.py:2881) rewrites it into a `Mul` dataflow node -- an
arithmetic operation in the compiled program that was never arithmetic. Its
operands (`ast.List`, `ast.Constant`) were never ingested as graph nodes
(they sit inside a list comprehension, which the reducer scopes specially),
so `_replace_inputs` receives two ids for nodes that do not exist. Measured:
all 16 have **both** operands missing, never one.

**2. Defensive.** `_replace_inputs` (topological_reducer.py:401) calls
`graph.G.add_edge(predecessor, node_id)`. NetworkX **creates** an absent
endpoint rather than raising, so the missing operand becomes a node whose
only key is `children` (from the `setdefault` on the next line). It has no
`type`, `expr_obj`, `label`, or `domain_node`, and detonates ~4000 nodes
later in `graph_express2.finalize_graph_with_outputs`, which reads
`node_data['type']` unconditionally. The C `BinaryOp` branch was guarded
against this earlier the same day; Python's `ast.BinOp` branch is
structurally identical and unguarded.

**Not a backend issue.** The traceback contains no WASM/WGSL frames. It is
ProcessGraph deployment, before any emitter. The read head's own
`transition` compiles clean (`control_shortfalls: ()`, 14s) precisely
*because* it receives the finished config as a static binding -- only the
page route, which ingests the module top-to-bottom, meets the setup code.

## The generalization

This is not one construct's problem. Module-level code exists to *build
things*: constant tables, registries, configured singletons. It is
compile-time setup whose **result** is a constant the program then uses. With
no module scope to own it, that code has nowhere to be evaluated, so its
expressions get ingested as runtime dataflow -- which is wrong even when it
does not crash. Constant-folding `[0] * 256` into a 256-element constant
would fix these 16 nodes and still leave the general defect: nothing decides
that module-level statements are *setup* rather than *program*.

This is the same conclusion the Dream document work reached from the other
direction (`READ_HEAD_STATE_MACHINE_PLAN.md`, F3): the blocks there declare
`entry = main` with no `main` in the body, their bodies being module-level
statements ending in `result = ...`. Both roads arrive at: **run module-level
statements in order as compile-time setup, accept definitions, then resolve
an entry point.**

## Recommended order

1. **Defensive guard first** (small, isolated, clearly correct): make
   `_replace_inputs` refuse to invent a node. It must not silently drop the
   operand either -- an operation missing an operand is a wrong program, not
   a smaller one. Fail loudly, naming the operation and the absent operand,
   so this class of defect surfaces at its origin instead of thousands of
   nodes downstream. This will make the current failure *earlier and
   legible*, not make it pass.
2. **Constant-fold static list replication**, so `[0] * 256` becomes constant
   data rather than a `Mul`. Fixes these 16 honestly.
3. **Introduce module scope as a real container** (Phase 3 of
   `READ_HEAD_STATE_MACHINE_PLAN.md`): module-level statements run in
   document order at compile time, definitions are accepted, then the entry
   point is resolved -- explicit if named, `main` by default, otherwise the
   module-level result. This is what actually removes the class of bug.
4. **File scope** only when a real multi-file unit needs it. Nothing today
   distinguishes file from module; inventing the distinction before a
   consumer exists would be speculative.

Steps 1-2 are safe now. Step 3 is the real fix and belongs with the
optional/default-entry work, not bolted onto the canonicalization pass.
