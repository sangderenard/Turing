# Read-head state machine: end-to-end completion plan

**Date:** 2026-08-06
**Repository:** `C:\dev\Powershell\turing`
**Status:** plan, researched against the real tree; no code written yet

## Goal, narrowed deliberately

Compile the **bidirectional x86 read head** — its ingestion process and its
outputs — into a real state machine that runs through the ordinary compiler
to a browser-executable artifact.

This is deliberately *not* the full
`CMD_BINARY_EXECUTOR_COMPILATION_HANDOFF.md` arc (resident PE loader, system
ports, capability gating, `cmd.exe`). That document remains the long-range
contract. This plan is the tractable slice that proves the machinery, and
whose pieces are all prerequisites of that larger goal anyway.

## Why the read head is the right target (researched, not assumed)

`src/compiler/x86_tensor_read_head.py`:

- `X86ReadHeadState` is **20 named `AbstractTensor` int64 registers**, one
  scalar per parallel lane. No Python objects, no dataclass reflection at
  runtime, no host bindings.
- `transition(batch, state) -> state` is **pure** — no mutation, no side
  effects, returns a new state.
- It is **entirely branchless**. Every update is `_select(mask, a, b)` ->
  `AbstractTensor.where`. There is no control flow to structure.
- `register_tensor()` is already documented as "the stable observation ABI...
  deliberately remains an AbstractTensor operation so a captured read-head
  graph can publish register state without a host round trip."
- `ReadHeadDirection.FORWARD/BACKWARD` — bidirectionality is already in the
  type system, not bolted on.
- `_add_mask_bit` avoids `__or__` on purpose ("without relying on logical
  `__or__`"), i.e. someone already curated this for operator coverage.

Contrast with `BinaryMachineProgram.tick()`, which we compiled to
`control_shortfalls: ()` this session: `tick()` drags in the whole object
graph, host bindings, and capability ports. The read head does not. It is
the single best-shaped compilation target in the repository for this goal.

## Findings that define the work

### F1. Dream Python sections are declared non-executable, not found so

`DREAM_LANGUAGE_TRANSLATIONS` (dream_document.py:46) hardcodes
`executable=False` for `python`, `sympy`, and `cpp`. The shortfall
`"process-graph-aot has no executable Dream shell artifact"` is emitted
*purely because that literal is False* — it is not a compile result.

### F2. The Python route stops at a description, not an artifact

`compile_sections()` for `route == "ast"` really does
`build_from_ast` + `reduce_abstract_tensor_topology` — the full reducer —
then calls `_process_graph_mapping()`, which emits **JSON nodes/edges/roots
plus the function table**. Real graph, real reduction, but the output is a
description. It never reaches AOT, deployment, or emission.

Notably `_process_graph_mapping` already publishes
`entry.reference.address` per function — the token is already in the
mapping.

### F3. The Dream blocks are sequential-procedural, and their declared entries do not exist

From `examples/reversible_chip_simulator.dream`:

- `chip-setup` declares `entry = main`. **There is no `main` in its body.**
  The body is module-level code: imports, `machine_program = None`, three
  `def`s, a `machine_controls` dict, then `result = machine_controls`.
- `head-step` declares `entry = step`. **There is no `step` either.** Its
  entire body is three module-level statements ending in `result = ...`.

This is exactly the "run top-to-bottom, accept definitions, execute
operations, then look for a default entry" model — and it confirms the
optional/default-entry design is required, not optional. The `result = ...`
convention is the de-facto block value today.

### F4. Blocks reference each other's definitions; isolation breaks them

`head-step`'s body calls `tick_machine(elapsed)` — defined in **`chip-setup`**,
a different block. `compile_sections()` compiles every block in its own
`ProcessGraph`, so `tick_machine` is an unbound external name. This is the
concrete, empirical proof that sequential graph fusion is required.

### F5. (SUPERSEDED BY MEASUREMENT — see "Phase 0/1/2 results" below)

`_table_1d`/`_table_2d` (the opcode/prefix/group table lookups — the core of
the decoder) are `AbstractTensor.gather`. Checked both backends:

- `ssa_webgpu_backend.py`: `where`/`Select` **is** handled (line 273).
  `gather` is **not** in `_BINARY`, `_UNARY`, or elsewhere.
- `fused_program_wasm_backend.py`: no `gather` either.

Both are easy in principle (WGSL: a storage-buffer index; WASM: a load at a
computed offset) but neither exists. **This is the single hardest hard
requirement in the plan**, because without it the read head cannot emit.

### F6. External link / import / export already converges on one rule

Three independent subsystems already require cross-boundary calls to become
explicit typed SSA, never hidden host callbacks:

- `shader_component_abi.py` — `LinkScope`, `LinkTransport`,
  `ComponentAssemblyPlan.lower_to_ssa()` emits a real SSA function per link
  (`__turing_external_component_link__`). Most mature.
- `compiled_program_api.py` — per-function calling contract (roles, dtypes,
  C/ctypes spellings) emitted from the same `Function` objects the codegen
  used, "not a second source of truth that could disagree."
- `machine_system_ports.py` — typed request -> completion handlers.
- `card_graph.py` — `external_link_policy` as data.

Extend these; do not invent a fourth.

### F7. The card graph is already the addressable OOP module graph

`build_card_graph()` already emits one card per function **and per method**
(owner-tagged), `class_navigation` membership edges, and a
`compatible-memory` closure over **every** type-compatible card pair. The
sequential order is stored as **one named path** (`paths.linear`) over that
same graph — its own comment: "may name one preferred schedule... without
erasing the other routes." Sequential and addressable are already two views
of one artifact.

### F8. `build_program_bundle()` has no Dream dispatch

Signature accepts `source: str` (Python) only. Handoff item 7 is genuinely
unstarted.

### F9. Tokens already exist for functions, not for classes

`FunctionReference(address: int)` — "Opaque address of one entry" — is
already the integer token, already used as `function_ref`/`callee_ref`.
`FunctionEntry.metadata` is free-form. What does **not** exist: a class-level
default accessor, or a compiled id -> dispatch map.

## Phase 0/1/2 results (measured, 2026-08-06) — DONE

**The read head is now a real, validated WebAssembly state machine.**

What the measurement corrected about the plan above:

- **F5 was wrong about `gather` being the blocker at SSA level.** The
  lowered SSA module contains no `gather` at all -- table lookups become
  `GetElementPtr` (x945) + `Load` (x927), the ordinary addressing pair.
  `gather` does survive in the *FusedProgram*, which is what the WASM target
  consumes, so it mattered -- just one layer down from where F5 claimed.
- **The real blocker was that the WASM backend was float-only.** `_TYPES`
  held f32/f64 and nothing else, so an all-integer program -- and the read
  head is literally a register file -- had no working type and raised
  `WasmEmissionError` before emitting anything. Integers existed solely as
  separate memory buffers to convert in and out of (`_MEMORY_DTYPE_OPS`).
- `transition` compiles in **14 seconds** with `control_shortfalls: ()`,
  versus 330-370s for `tick()`. The premise that it is a far better target
  held up.

What was built (commit: integer working types):

1. **Integer working types (i32/i64) in the WASM backend.** The instruction
   tables were already suffix-generic (`"add"`, `"lt"` prefixed by the value
   type), so the shape was tractable. What genuinely differs for integers,
   and is now handled: explicit signedness (`div_s`, `lt_s`, `rem_s`);
   no integer `min`/`max`/`abs`/`neg` (composed from compare+`select` and
   `0 - x`); float rounding is the identity; LEB128 constants rather than
   IEEE payloads; `i64.eqz`-based boolean connectives (a truth-value `&&`
   is not a bitwise `&`); and infinite fold identities saturating to the
   type's extreme. Done in **both** emitters -- the WAT text and the binary
   assembler are independent, and an integer type in only one would emit
   readable text that refuses to assemble.
2. **`where` (ternary select).** 80 of the 88 remaining shortfalls. This is
   what a branchless state machine is *made of* -- every update predicated
   by a mask. WebAssembly's `select` is exactly it.
3. **`gather` (table lookup).** A read at a computed index rather than the
   loop cursor -- an opcode byte selecting an encoding-table row. The last 8.
4. **`mod`/`floordiv` unlocked** as a side effect, exactly as
   `_NO_WASM_INSTRUCTION`'s own comment anticipated ("await an integer-
   remainder lowering"). The read head needs both (`_add_mask_bit`).

Verification:

- `wasm: complete=True shortfalls=0`
- 22,879-byte binary, correct `\0asm` magic and version
- **`WebAssembly.validate: true` in Node/V8; compiles AND instantiates**,
  exporting `memory` and `run`
- `tests/test_read_head_wasm_state_machine.py` (new, 3 tests) locks this in
  through the ordinary entry points, not a hand-built program
- WASM suites: 142 passed, 3 pre-existing failures (confirmed against a
  stashed tree)

**Still open for the read head:** WGSL/WebGPU remains blocked by 642
`unsupported WGSL dtype 'int64'` shortfalls. WebGPU core genuinely has no
64-bit integers, so this is a real spec constraint, not a missing feature --
it needs either i32 narrowing (most read-head registers fit; displacement/
immediate/relative_target carry x86-64 addresses and do not) or 64-bit
emulation. WASM was the correct first target and is done.

## Plan

Ordered by dependency. Each phase has a verification gate that must pass
before the next begins. The non-negotiable global gate remains: the real
`binary_machine_tick.tick()` compile must still return
`control_shortfalls: ()` after any change to the reducer or lowering.

### Phase 0 — Baseline harness (small)

Compile `X86ReadHead.transition` through `compile_ast_aot` exactly as
`compile_probe2.py` does for `tick()`, `precompile_only=True`. Record the
shortfalls verbatim.

*Purpose:* replace every assumption in this document with a measured list.
Expect `gather` to surface; expect dataclass-of-tensors handling questions.
Do not fix anything yet — just get the true list.

**Gate:** a recorded, reproducible shortfall list.

### Phase 1 — `gather` lowering (F5), the hard requirement

Add `gather` to the WebGPU backend (indexed storage-buffer read) and to the
WASM backend (load at computed offset). Bounds behavior must be explicit and
fail-closed, matching `fused_program_wasm_backend.py`'s existing discipline
of naming shortfalls rather than silently approximating.

**Gate:** a real AST-generated program using `gather` emits valid WGSL and
WASM, tested the way `test_webgpu_ssa_backend.py` already tests loops
(compile through `compile_ast_aot`, lower, emit, assert on source).

### Phase 2 — Read head emits (the first real milestone)

With `gather` present, drive `transition` to a complete WGSL artifact and a
complete WASM artifact.

**Gate:** `artifact.complete is True`, `shortfalls == ()`, and the emitted
module has the expected register-count-shaped IO. This is the first point at
which "the read head is a compiled state machine" is literally true.

### Phase 3 — Optional/default entry resolution

Implement the model F3 proves is needed:

1. Run module-level statements in document order (definitions accepted,
   operations executed).
2. If an explicit entrypoint is named and exists, call it after.
3. Otherwise default to `main` if present.
4. Otherwise the module-level result is the value (today's `result =`
   convention, made real rather than incidental).

This is where `compile_ast_aot`'s entrypoint parameter becomes optional.
Reuse `FunctionTable.reference_by_source_node` (added this session) so
resolution stays identity-based, never bare-name.

**Gate:** a Dream-shaped block with no matching entry function compiles and
produces its module-level value; `tick()` still `control_shortfalls: ()`.

### Phase 4 — Sequential Dream block fusion (F4)

One `ProcessGraph` per document rather than per block. Compile blocks in
document order into a shared graph/function table so `head-step` resolves
`tick_machine` from `chip-setup` as a real `FunctionReference` rather than an
unbound name.

Same-language first (the two Python blocks). Cross-language blocks
(glsl/js) join as **external link boundaries** per F6, not as inlined nodes —
the shader blocks keep their own emission path and appear in the fused graph
as typed link calls.

**Gate:** `head-step`'s call to `tick_machine` resolves to a
`FunctionReference` in the fused table; both Python blocks report a real
artifact.

### Phase 5 — Flip `executable` from a literal to a computed result (F1/F2)

Replace the hardcoded `executable=False` with the real question: did this
section produce an artifact? Add an artifact-bearing field to
`DreamSectionCompilation` (kind, bytes/source, entry symbol, digest,
shortfalls) as the handoff's item 1 requires.

**Gate:** `chip-setup` and `head-step` report `executable=True` with a real
digest-bound artifact, and a tampered artifact fails before instantiation.

### Phase 6 — Frame ABI and the class accessor (the design just agreed)

Represent a call frame as a uniform slot table:

- `Alloca` a fixed-size pointer/offset array, index with `GetElementPtr`,
  count carried by the existing `Parameter.role == "extent"` convention.
  **Do not teach SSA what a struct is** — heterogeneity already lives in each
  `SSAValue.dtype`; backends that want a literal C struct synthesize it at
  emission time.
- Shaders: pack into one buffer with u32 offsets rather than one binding per
  slot (user decision). A packed buffer + offsets is isomorphic to the
  pointer array, so this is the same IR, not a special case.
- First parameter is `self`; absent -> class-level.
- Validate `self`-presence **against** the declared static/instance keyword.
  A null `self` for a non-static method is a hard error, never silently
  reinterpreted as a class call.
- The `self` vs callee-owner check is a **subtype/MRO** check, not exact-class
  equality — single inheritance is already in the C++ shell's scope.
- Reference-by-default with a copy-blacklist for value types.

Then the accessor: when a unit contains only a class and no entry, export the
`.` resolver as a second **entry kind** in `compiled_program_api.py`
(`"accessor"` alongside `"function"`), dispatching on
`FunctionReference.address` (F9 — already an integer token) plus an instance
address (`ClassNavigationMember.slot`, added this session).

**Gate:** a class-only unit compiles with an accessor entry; class vs
instance dispatch is an explicit compiled fork; a static/instance mismatch
fails closed.

### Phase 7 — Bundle route (F8)

Extend `build_program_bundle()`'s source dispatch to accept a Dream document,
per handoff item 7. Do not add a publisher; do not add another HTML shell.

**Gate:** one normal build command consumes the `.dream` and produces a
bundle whose manifest does **not** say `prebuilt-program-interior`.

## Sequencing notes

- **Phases 0-2 are the critical path** and are independently valuable: they
  end with a genuinely compiled read-head state machine, regardless of
  whether the Dream/bundle work ever lands.
- Phases 3-5 are the Dream document track. Phase 3 is a prerequisite for 4;
  4 for 5.
- Phase 6 is largely independent of 3-5 and could run in parallel; it is
  required before any complex class-bearing program.
- Phase 7 is last by necessity.

If time forces a cut, **stop after Phase 2 with a real artifact** rather than
spreading half-finished work across phases.

## Standing verification discipline

- The real `tick()` compile (`compile_probe2.py`, ~300-370s,
  `control_shortfalls: ()`) after every reducer/lowering change.
- Targeted suite, then the broad `-k "reducer or graph or ssa or process"`
  selection compared **against a stashed baseline** — that suite has 18
  pre-existing failures, so an absolute count means nothing without the
  comparison.
- Do not build test harnesses that duplicate real ingestion. Drive the real
  entry points (`load_dream_document`, `compile_ast_aot`,
  `build_program_bundle`) and read their real outputs.
