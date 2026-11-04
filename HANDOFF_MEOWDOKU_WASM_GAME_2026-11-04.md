# Meowdoku as a dt-system game in one HTML file: compiler handoff

Date: 2026-11-04
Repository: `C:\dev\Powershell\turing` (branch `codex/live-string-core-demo`, uncommitted)
Author of the request: the game lane. Owner of the open item: the compiler lane.

## Status in one line

The game is written as one dt-system piece and plays through `dt_system`
in Python. Its step law does not lower to SSA: one refusal, `bitor`, stops
the whole-object lowering before any WebAssembly emission is attempted.
Nothing in `src/compiler` was changed by this lane.

## What exists and is green

| File | What it is | Test |
|---|---|---|
| `examples/meowdoku/puzzle.py` | Setup, not a law. Unique cat arrangement, regions grown from the cats, static columns `row`, `col`, `region`, `solution`. | `tests/test_meowdoku_program.py` |
| `examples/meowdoku/program.py` | `meowdoku_step`, the law: masked span arithmetic, no branch, read-head form. `MeowdokuPiece` wraps it with `llvm_dt_system.PIECE_API`. | same |
| `examples/meowdoku/session.py` | The game through `dt_system` with one persistent state. One move is one round and one law call. Undo and reset are `copy_shallow` and `restore`. | `tests/test_meowdoku_dt_session.py` |
| `tools/probe_meowdoku_ssa.py` | The repro below. Report only. | – |

```
python -m pytest tests/test_meowdoku_program.py tests/test_meowdoku_dt_session.py -q
7 passed in 4.54s
```

Rules the law implements (the published game): one cat per region, row and
column; no two cats touch, diagonals included; a placement that is not on the
hidden solution costs one of three hearts and lands nothing. One move is
one round. The law publishes `residual` (cats still to place) and
`dt_limit = 1.0`, no energy and no power, so the dt system reads it as a
holding participant and never subdivides a move.

## The refusal

Repro, about nine seconds, from the repo root under the system Python
(`turing/.venv` has neither `llvmlite` nor `yaml`):

```
python tools/probe_meowdoku_ssa.py 25
```

The probe calls `lower_ast_source_to_ssa` on the text of `meowdoku_step`
with `python_bindings` for `AbstractTensor` and the two move constants and
`extraction_contract=native_law_kernels.batch_contract(...)`, which declares
all nine arguments as rank-1 float64 spans of `batch` cells. That is the
law lane's own call shape (`native_law_kernels._lower_law`), minus the LLVM
tensor reference. Output:

```
SSALoweringIncompleteError: whole-object SSA lowering is incomplete:
meowdoku_step_batched__meowdoku_step::meowdoku_step_batched__meowdoku_step__planned_region_8:entry::bitor
(mixed binary has no exact dynamic/static metadata binding and terminal guard topology)
```

Where in the log it falls:

```
[ast-parent] unresolved call 'reshape' line=4..9,16 reason=dynamic_or_primitive
ssa-section ...__meowdoku_step: control lowering complete; blocks=1 shortfalls=0
ssa-section ...__meowdoku_step: tensor repository linking complete; shortfalls=1
ssa-section ...__meowdoku_step: complete; functions=35 shortfalls=1
```

Identity log of the failed compile:
`artifacts/identity_logs/meowdoku_step_batched.20261104T121549.failed.log.xz`
(numberings in it are valid for that run only).

The source that produces the refused `bitor`, verbatim from the law:

```python
cells = int(row.shape[0])
r1 = row.reshape((cells, 1))
r2 = row.reshape((1, cells))
c1 = col.reshape((cells, 1))
c2 = col.reshape((1, cells))
g1 = region.reshape((cells, 1))
g2 = region.reshape((1, cells))
touching = ((r1 - r2).abs() <= 1.0) & ((c1 - c2).abs() <= 1.0)
conflict = (r1 == r2) | (c1 == c2) | (g1 == g2) | touching
load = (conflict.to_dtype("float") * cat.reshape((1, cells))).sum(dim=1)
```

Reading of the message by this lane, unverified: the four masks are rank-2
`(cells, cells)` values produced by broadcasting compares of reshaped
declared spans. Every `reshape` is logged as an unresolved primitive, so
the operands of `|` reach repository linking without an exact static or
dynamic shape binding and the binary is classed as mixed. The `&` in
`touching` is the same shape and is not named, so either it links or it is
hidden behind the first refusal.

## What is asked of the compiler lane

1. Make the law as written lower, or state the sanctioned spelling of a
   pairwise outer compare over one declared span. The game lane will not
   rewrite the law around the compiler (valid Python must compile).
2. Say what is behind this refusal once it clears. The law also uses
   `.abs()`, `<=`, `==`, `&`, `~`, `to_dtype("float")`, `.sum(dim=1)`,
   `.sum()` to rank 0, `AbstractTensor.where` with scalar branches, and
   rank-0 broadcast back onto a column (`x + cat_next * 0.0`). None of
   these has been reached.
3. The emission target is the direct lane: `ssa_wasm_backend.emit_ssa_module_to_wasm_core`,
   as `examples/mobile_reference_app/build_mobile.py` uses it. Not the
   fused backend, which declares itself off-limits and is what the
   registered `wasm` machine target currently routes through. The probe
   also tries `emit_ssa_function_to_wasm` and prints both verdicts.

## Acceptance contract

The probe prints, for batch 25 and batch 64:

```
[probe] lowered in <seconds>
[probe] wasm core: complete=True
```

with zero shortfall lines, under the law's text unchanged. On that, the game
lane continues alone:

- a `wasm` backend in `examples/llvm_dt_system.lowered_system` through the
  direct core, beside `c`, `llvm` and `fortran`;
- the one-file page through `wasm_html_shell` (binary inline as base64),
  with intent columns written from clicks and `copy_shallow`/`restore`
  behind Undo and Reset;
- a `wasm_fidelity`-style parity check of the shipped binary against the
  Python piece over a scripted game, as `tests/test_meowdoku_dt_session.py`
  plays it.

## Not claimed

No wasm has been emitted. No dt round has been lowered for this game. The
dt system itself lowering to SSA in the current compiler is not assumed;
if `dt_system_over` over this piece refuses, that is a second report.

## Compiler lane reply (2026-11-04, 13:15)

Source: `C:\Users\alber\.codex\worktrees\compressor-validation\turing\HANDOFF_MEOWDOKU_COMPILER_WASM_2026-11-04.md`,
a validation checkout at `dbd3f7c9` with uncommitted compiler changes. None of
it is in `C:\dev\Powershell\turing`; the probe there still stops on `bitor`.

- Ask 1 is met there. The unchanged law lowers to complete SSA at batch 25
  and 64 with the sanctioned C tensor reference: 35 functions, entry
  `meowdoku_step_batched__meowdoku_step`. The fix was a float alias Call
  adapter change (`tests/test_meowdoku_float_call_adapter.py`). Saved
  modules: `%TEMP%\meowdoku_validation_25_module.pkl` and `_64_`.
- Ask 2 is answered: no further non-wasm refusal on those two lowers.
- Ask 3 is open. The direct core emits `complete=False` on both. Batch 25:
  124 shortfalls (Call 48, GetElementPtr 21, Load 21, Ret 5, unavailable
  operand 29). The host ABI is classified as eight input buffers and five
  outputs; compiler-owned frame cells are internal. `pow` has no direct wasm
  spelling. `ssa_wasm_backend.py` carries a 696-line WIP diff owned by two
  agents; it is not accepted and must not be transplanted whole.
- The acceptance contract above is unchanged and unmet. The wasm lane is
  frozen until a new instruction. The game lane does nothing until it
  resumes, then reruns `tools/probe_meowdoku_ssa.py` against the merged
  compiler.

## Finding 2: the saved SSA through the C and LLVM lanes (2026-11-04, afternoon)

Both saved modules (`%TEMP%\meowdoku_validation_25_module.pkl`, `_64_`) were
emitted through the more developed lanes from the validation checkout, with
no compiler change.

| Lane | Verdict |
|---|---|
| C module lane (`emit_ssa_module_to_c`, the canonical route) | incomplete, **1 shortfall**, 13 buffers: `Eq: elementwise Eq broadcasts (25, 1) to (25, 25); C spells only equal extents in ...planned_region_8`. The lane refuses by name the same bare `equal` the LLVM lane miscompiles below. |
| C scalar adapter (`emit_ssa_function_to_c`) | not applicable: legacy straight-line `(const double *in, double *out)` ABI; it refuses GetElementPtr/Load/Store by design, so its 103 shortfalls say nothing about this program. (Corrected: an earlier revision reported these as the C verdict.) |
| LLVM (`emit_ssa_function_to_llvm`) | complete, 0 shortfalls; compiles in 0.5 s; 8 input buffers + 5 outputs, 4 compiler-owned frame cells internal. |

The LLVM artifact plays the scripted game **wrong**, at batch 25 and 64 alike:
it matches Python exactly for moves 0 to 5 and diverges at move 6, the first
legal placement with a cat already on the board. From then on native judges
every placement wrong. Native `load` is nonzero at every cell whenever any cat
exists, for cats at cells 0, 4, 12 and 24, and even under identity geometry
where `conflict` must be the identity matrix. The Codex lane's pairwise slice
is exact natively (sum of `|r1 - r2|` = 1000.0), so the subtraction path is
sound.

**Located miscompile.** In `planned_region_8` the row compare `r1 == r2` is
the only rank-2 operation in the module left as a bare SSA `equal` on the
`(25, 1)` and `(1, 25)` views; the column and region compares beside it are
linked as `broadcast_double` + `binary_double(CT_OP_EQ)`. The LLVM backend
emits the bare `equal` as an elementwise loop of 625 `fcmp oeq` over
`%arg.3` and `%arg.4`, which the entry fills with the same raw 25-element
`row` column (`%5` passed twice). That reads 600 doubles past each buffer
and compares every element with itself, so `%88` is all ones, `conflict` is
all ones, and `load` equals the cat count at every cell. Everything observed
follows from that one instruction, including the clean first six moves (no
cat, or only wrong placements).

```
%88(25, 25):int64 = equal(%68(25, 1), %70(1, 25))     ; region 8, source_type Compare
  ew.body.51.88: %ew.op.51.88.0 = load double, ptr %arg.3 + i   ; i < 625
                 %ew.op.51.88.1 = load double, ptr %arg.4 + i
                 %ew.val.51.88  = fcmp oeq double ...
```

Reproduce from the validation checkout, each in seconds:

```
set PYTHONPATH=%CD%
python C:\dev\Powershell\turing\tools\probe_meowdoku_llvm_parity.py 25      # first divergence at move 6
python C:\dev\Powershell\turing\tools\probe_meowdoku_load_map.py 25 12 identity
python C:\dev\Powershell\turing\tools\dump_meowdoku_region_ir.py 25         # bare ops + region 8 IR
python C:\dev\Powershell\turing\tools\probe_meowdoku_pairwise_slice.py      # 1000.0, sound
```

Ask added for the compiler lane: the bare `equal` on broadcast views must be
linked like its siblings, or refused by name as the C module lane already
does (`probe_meowdoku_c_module.py 25`, one shortfall). It must not emit as an
elementwise loop over aliased, undersized operands. This is the kind of
silent wrong answer the acceptance contract's parity check exists to catch,
so parity against the Python piece stays part of acceptance for every lane.

## Finding 3: the direct Wasm core modelled on the LLVM and C module lanes (2026-11-04, evening)

The wasm lane resumed on the user's instruction after the compiler checkpoint
merge (`cd7cf06d`). Canonical now lowers the law itself: batch 25 and 64 each
in ~10 s to 35 complete functions (`tools/probe_meowdoku_ssa.py [batch]`
saves `%TEMP%\meowdoku_canonical_{batch}_module.pkl`). The validation
checkout is no longer needed for this lane.

**What changed (wasm lane files only).**

- `src/compiler/wasm_binary.py`: `build_module` assembles further functions
  (`WasmFunction`, with value results and optional export) and value-returning
  imports (`WasmImport.result_types`). Backward compatible; every existing
  caller is unchanged. `tests/test_wasm_binary_functions.py` runs both in node.
- `src/compiler/ssa_wasm_backend.py`, `emit_ssa_module_to_wasm_core`: every
  reachable SSA function is now its own wasm function with the LLVM
  repository-module convention (pointer formals as i32 offsets, scalars by
  value, `Ret` values as wasm results, multi-value where a function returns
  several). A `Call` is a `call`; a callee with several `Ret` blocks
  (`binary_value`, 23 arms) returns from each. The exported entry keeps the
  host ABI the shells speak. Scalar externs: exact Wasm opcodes for
  `llvm.fabs/ceil/floor/trunc/sqrt.f64`, an exact spelling of
  `llvm.round.f64` (half away from zero; `f64.nearest` is half to even),
  `llvm.fcmp.ord/uno` by self-equality, and `math` imports for `pow`, `exp`,
  `log` and the trig family, which the node worker binds from `Math`.
  Shaped `Cast` (the compiler's `physical_region_input_conversion`) is the
  element loop into a fresh static arena, as the C module lane and the LLVM
  lane emit it. A function's own `output` span is allocated in its static
  frame and returned by pointer (LLVM receives it as a result pointer).
  `float32` is a value kind (`cast_double_to_float_values` narrows through it).
- Gate: `test_wasm_core_int64`, `test_mobile_reference_app`,
  `test_wasm_multiblock_callee`, `test_wasm_binary_functions`: 14 passed.
  The lane's other test files: 230 passed; the 2 failures and 3 errors
  (`test_wasm_class_modules` x2, `test_read_head_wasm_state_machine` x3) are
  compiler-stage refusals (`_decide_value_shape` dtype settlement,
  `source_call_actual_formal` concordance disagreement, fused-program dtype
  record) reached before any wasm emission; none touches the changed files.

**Proof on repository kernels.** The Codex lane's pairwise slice module
(`build/meowdoku_pairwise25_ssa.pkl` in the validation checkout: 10 functions,
`broadcast_double`, `binary_double`, `binary_value`, `unary_double`,
`sum_double` ...) emits complete through the new core (9 further functions,
5.8 kB), runs in node and returns exactly 1000.0, the Python value.

**Meowdoku through the core: 124 shortfalls -> 9**, identical at batch 25
and 64 (`PYTHONPATH=. python tools/probe_meowdoku_wasm_core.py 25 --canonical`).
All nine are two compiler-side items and their cascades:

1. `planned_region_8`: the bare `equal` over `(25, 1)` and `(1, 25)` (Finding 2).
   The core refuses it by name, as the C module lane does; its result `%t88`
   and the region's output cascade.
2. `planned_region_18`: the storage requirements for the two
   `cast_double_to_float_values` values `%t2305843010213694258/259` are
   `(625, (25, 25))` while the values, the region formal `%t105` and its
   output `%t110` are all `(25,)`. The core refuses the output arena because
   the booked obligation disagrees with the value's declared shape
   (`output ... shaped (25,) lacks a booked static arena (requirement (625,
   (25, 25)) ...)`), and `planned_region_19` cascades. `planned_region_9` has
   the same kernel at `(25, 25)` and books `(625, (25, 25))` correctly, so this
   looks like the kernel's obligation being shared across its call sites
   rather than per value.

**Asks for the compiler lane (both exact, both seconds-long to reproduce):**
link or refuse the bare `equal` (Finding 2), and post per-value storage
requirements for the region-18 cast temporaries. With those two, the core
has no remaining refusal on this program.

**Not claimed.** No Meowdoku wasm parity yet (the module does not emit while
the two items stand). `lowered_system(backend="wasm")` and the one-file page
through `wasm_html_shell` are the game lane's next steps once the module
emits; the shell's loader will need the same `math` import object the node
worker supplies. Host `Math.*` is not bit-identical to libm in the last ulp;
the Meowdoku law never reaches those imports at run time (dead opcode arms),
so its parity target stays exact.
