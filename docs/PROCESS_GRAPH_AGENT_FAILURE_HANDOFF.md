# ProcessGraph Agent Failure Handoff

## Purpose

This document records a failed investigation and the damage it caused to the
ProcessGraph/AbstractTensor translation effort.  It is written so the next
person working here does not repeat the same failure: treating a large,
unfamiliar codebase as incomplete before locating and understanding the paths
that already exist.

## What the agent failed to do

The agent was repeatedly told that the repository already contained the
translation material and that it needed to be found and used.  Instead of
following that instruction, it repeatedly inferred missing architecture from
incomplete local understanding.

Specifically, the agent:

- Assumed a raw Python AST graph needed a new semantic replacement before
  thoroughly tracing the existing ProcessGraph compiler path.
- Claimed that a backend could not execute the existing `BinOp` plus operator
  child representation without proving that claim from the repository.
- Treated the distinction between a structural AST graph and executable graph
  as evidence of absence, rather than first locating the existing lowering and
  backend-table machinery.
- Failed to center `src/transmogrifier/operator_defs.py` even after being told
  to find the code where `BinOp` is disambiguated and tensor functions are
  listed.
- Incorrectly described `operator_defs` as merely adjacent to AbstractTensor
  and ProcessGraph, then had to be corrected: it is the ProcessGraph
  operation/signature/backend-table contract.
- Failed to recognize that `GraphDeepCompiler` already accepts the operation
  and signature tables supplied by `operator_defs`, including its NumPy and
  Torch variants.
- Claimed the AbstractTensor table had been found, then contradicted that claim
  by calling the corresponding bridge "missing." The correct observation was
  that the existing AbstractTensor method-binding surface and canonical
  capability catalog should be exposed through the existing ProcessGraph table
  pattern.
- Searched deleted/history material after being told to inspect live code,
  rather than limiting the investigation to the working tree.

These were not harmless wording mistakes. They redirected attention toward
inventing parallel translation systems, removing code, and debating
non-problems instead of reading and using the system already present.

## Translation path that was obscured

The existing intended path is materially more complete than the agent
represented:

```text
Python AST / SymPy
    -> ProcessGraph structural and control wiring
    -> operator_defs role schemas and operation signatures
    -> backend operation/signature table
    -> GraphDeepCompiler or other ProcessGraph compiler
    -> execution on the selected backend
```

`operator_defs.py` already contains the central connection:

- `role_schemas` preserves graph structure and control relationships;
- `operator_signatures` defines operation contracts;
- `default_funcs`, `numpy_funcs`, `numpy_sigs`, `torch_funcs`, and
  `torch_sigs` provide backend-facing execution tables.

`GraphDeepCompiler` already accepts an operation table and signature table.
That means the ProcessGraph architecture was designed for backend selection.
The appropriate continuation is to make the existing AbstractTensor surface
available in that same table shape, so ProcessGraph execution can retain the
selected AbstractTensor backend: NumPy, Torch, C, GLSL, or a future Nodus
target. The control graph does not need to be recreated.

The existing AbstractTensor material that should have been used includes:

- dynamic high-level method binding in `common/tensors/abstraction.py`;
- canonical fused-operation vocabulary in `common/tensors/fused_ir.py`;
- native capability/canonical-operation auditing in
  `common/tensors/backend_capability_audit.py`;
- C and GLSL backend primitive dispatch tables.

## Time lost and project risk

Hours were lost on a false premise that the repository lacked fundamental
translation layers. This introduced substantial project risk:

- Existing translation paths were obscured by discussion of replacement IRs
  and replacement semantic front ends.
- Work was performed against assumptions rather than verified extension points.
- The working tree contains changes made during that mistaken direction,
  including removal of semantic-import-related files and modifications around
  ProcessGraph/SSA handling. Those changes must not be treated as evidence that
  the original architecture was inadequate.
- An unintegrated AST-to-SSA equivalence table was added despite the agent not
  first proving that this was the correct live integration seam.

This is a risk of architectural corruption: not because the original project
was empty or broken, but because unnecessary parallel paths and destructive
edits can conceal, bypass, or sever the already-existing route from source
graph to backend execution.

## Required discipline for any continuation

Before writing, deleting, or replacing translation code:

1. Start from `operator_defs.py` and trace every existing consumer of its
   schema, signature, and backend function tables.
2. Trace `ProcessGraph` into `GraphDeepCompiler` using the existing NumPy and
   Torch tables.
3. Treat AbstractTensor as another backend table implementation unless direct
   inspection proves otherwise.
4. Reuse the existing graph structure and control wiring; do not create a
   separate AST semantic subsystem merely because a code path has not yet been
   understood.
5. Do not infer absence from unfamiliarity. Search the live repository first,
   follow the call chain, and state only what was verified.
6. Do not use deleted files, repository history, or speculative architecture as
   substitutes for reading the active path the user identified.

## Responsibility

The agent made the assumptions, failed to perform the requested investigation
first, and repeatedly communicated conclusions beyond the available evidence.
This handoff exists to make those failures explicit and prevent them from being
repeated.
