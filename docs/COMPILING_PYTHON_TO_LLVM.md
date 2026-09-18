# Compiling Python to LLVM

A starting point for someone who wants to take Python source in this repo and
get a compiled native artifact out of it.

## The route

Three stages, three files.

| # | Stage | Where |
| - | ----- | ----- |
| 1 | Python source → repository SSA | `src/compiler/fortran_c_shell.py` → `lower_ast_source_to_ssa` |
| 2 | SSA → LLVM IR | `src/compiler/ssa_llvm_backend.py` → `emit_ssa_function_to_llvm` |
| 3 | LLVM IR → compiled artifact | `src/compiler/ssa_llvm_backend.py` → `compile_artifact` |

```python
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm

module, outputs, exports = lower_ast_source_to_ssa(
    source, entry, name="my_thing", extraction_contract=policy,
)
compiled = compile_artifact(
    emit_ssa_function_to_llvm(module, exports[0]),
    directory=build_dir, optimization="O2",
)
```

There is also a C lane (`src/compiler/ssa_c_backend.py` → `emit_ssa_module_to_c`)
which is what most of the tree actually uses, because it can link LLVM pieces
into the emitted module. Stage 1 is identical either way.

## Read this example first

**[`examples/llvm_dt_system.py`](../examples/llvm_dt_system.py)**, function
`lowered_system` (around line 228). It is the whole route in about fifty
readable lines, with all three backends side by side, and it is a real working
program rather than a toy.

If you only read one thing, read that function.

## The extraction contract is mandatory

This is the thing that stops people cold. `lower_ast_source_to_ssa` raises
immediately without an `extraction_contract`:

> `lower_ast_source_to_ssa requires an extraction_contract: pass one, or set it
> on the active work contract.`

The contract declares what crosses the native boundary: which parameters are
spans and of what dtype/rank/shape, which records exist and what fields they
have, and which parameter binds to which record.

Two reference contracts, smallest first:

- **`native_law_kernels.batch_contract(entry, argument_names, batch)`**
  (`src/compiler/native_law_kernels.py:78`) — the minimal real one. No records,
  no bindings, just one `float64` batch span per named argument. Start here.

- **`vehicle_python_compilation.balloon_tire_managed_extraction_contract`** —
  one that declares a whole state record's fields.

- **`llvm_dt_system.dt_system_contract`**
  ([`examples/llvm_dt_system.py:190`](../examples/llvm_dt_system.py:190)) — a
  good middle example: retains some records from the YAML sheet, declares one
  new record with a span per field, and binds a parameter to it. Easy to copy
  and cut down.

The base sheets live in `extraction_contracts/` (`program_extraction.yaml`,
`vehicle_full_native_execution.yaml`).

### Do not pass `extraction_contract=None` to get moving

It is tempting when you are trying to see *something* compile. Don't.

It disables the machine-decompilation gate, so every record crossing the
boundary becomes undeclared and every unresolved receiver reports as
`opaque-state-effect`. Every diagnosis you then get is a description of the
missing contract, not of your program, and you will spend the afternoon
chasing an error that does not exist. Write the contract first, even a wrong
one — a wrong contract gives you a real error message.

## Things that will bite

- **Declare extents on tensor parameters.** A tensor op that "cannot compile"
  is usually a parameter with no declared extents. See the notes on the tensor
  compile pipeline before concluding an op is unsupported.
- **Lowering is not fast.** A small function is seconds. A real program is
  minutes — the two-piece `llvm_dt_system` lowering is around seven minutes and
  that is normal, not a hang. Pass `progress=print` to
  `lower_ast_source_to_ssa` so you can see which stage you are in.
- **Value ids are not stable across runs.** Don't hard-code them.
- **The final gate is strict.** After linking, a full-native execution contract
  check rejects the module for unmaterialized boundaries, unresolved calls,
  undefined operands or unaccounted formals. "Unaccounted formals" means the
  emitted signature has a parameter no caller can name — usually an identity
  that was dropped somewhere upstream, not something wrong at the call.

## Debugging

Every compile writes an identity log to `artifacts/identity_logs/` —
`<name>.<timestamp>.<ok|failed>.log` — on success and on failure both. It is a
correlation table: one row per (function, value id, kind of fact), one column
per round or callsite. When two passes disagree about one value, that log is
where the disagreement is visible; no single pass can see it.

`python tools/audit_identity_concordance.py` runs the same checks over a few
small cases in seconds. Run it before and after any compiler change.
