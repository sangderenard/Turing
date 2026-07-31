# Backend performance and shell architecture — handoff

Status: current as of commit `1e227d3`. Working tree clean.

This covers the LLVM/Fortran lowering work, the two launch shells, the native
compile path, and the torture harness changes. Every number here was measured
on this machine (AMD `bdver2` / Piledriver, NVIDIA RTX 3060, Windows) and is
reproducible from the commands given.

---

## 1. What landed

| Commit | Subject |
|---|---|
| `1a7f328` | JIT backend suite, SSA/Fortran lowering, torture harness |
| `cb52985` | C++ dispatch decision shell; deepened C launch shell |
| `e78ae83` | Native C compile/load; Fortran accelerated control target |
| `d4c7bef` | Scaled torture operator set; `--llvm-profile` |
| `1e227d3` | Fixed stale torture cache identity; fused-chain case |

---

## 2. The LLVM lowering was never optimized

`llvm_jit_backend.py` built its target machine with
`create_target_machine(opt=0)` against the default triple and ran **no pass
pipeline at all**. That is scalar, generic code — LLVM was never asked to
optimize anything.

It now runs `tuned_host_profile()` from `src/compiler/llvm_optimizing_pipeline.py`:
`-O3` via `PassBuilder`, host CPU named, `noalias` asserted on kernel buffers,
and a host-selected vector width.

### The vector width is the whole story on this CPU

gfortran beat LLVM by 2.4× at 64K elements. The cause was counter-intuitive:

| | vector width | FMA |
|---|---|---|
| LLVM `-O3` | ymm (256-bit) | 0 |
| gfortran `-O3 -march=native` | **xmm (128-bit)** | 13 |

Bulldozer-family cores (`bdver1-4`, `btver*`, `znver1`) split a 256-bit AVX
operation into two 128-bit halves internally, so wider vectors buy nothing and
cost the split plus `vzeroupper`. GCC's `-march=native` encodes that tuning;
LLVM's target machine does not.

Setting `prefer_vector_width=128` closed the gap entirely — LLVM now matches or
beats gfortran at every size.

**This is CPU-specific.** `host_preferred_vector_width()` is a deliberate
allow-list; on Zen 2+ and modern Intel it returns `None` (let LLVM decide),
because forcing 128 there would give up half the throughput. Do not hardcode it.

---

## 3. Measured numbers

### Backend selection by size (float64, elementwise)

Three regimes, from a 15-operator × 9-size × 4-backend sweep:

- **≤ 4K elements** — NumPy wins. Call overhead dominates; compiling loses.
- **4K–128K** — Fortran wins, up to 2.1×; tuned LLVM reaches 1.56×.
- **≥ 256K** — tie. Memory-bound; the choice stops mattering.
- **Transcendentals** (`exp`/`log`/`sin`/`cos`/`tan`) — NumPy wins at *every*
  size. All compiled backends bottom out in the same scalar libm; NumPy
  vectorizes them.

`opt=0` never beat NumPy at any size. It was a pessimization throughout.

### Launch boundary

```
pure C shell, empty kernel        2.185 ns/call
via cffi python callback       1,430     ns/call   (654x)
```

The shell is not what makes a dispatch expensive. Anything that re-enters
Python costs ~650× the boundary itself. Design against that number.

### Native compile vs cffi

```
native gcc -> dll + ctypes      548 ms
cffi.verify (extension module) 5,591 ms   (10.2x)
```

### Fused chain, 15 operations, 1,048,576 elements

```
raw_numpy    237.6 ms    eager, unfused, 117 MB of intermediates materialised
numpy       8740.5 ms    AbstractTensor eager  (37x slower than raw NumPy)
torch       9605.3 ms    *** CPU TENSORS -- see section 6 ***
c_jit        243.5 ms    fused, one pass
llvm_jit     302.2 ms    fused, one pass
glsl_jit     264.8 ms    fused, one pass
```

AbstractTensor eager vs raw NumPy is **37×–106×**, and the ratio is *worse* at
small sizes (106× at 64K) because the cost is per-operation Python dispatch — a
fixed charge that dominates when arrays are small.

Compiling recovers that: **36×** against the AbstractTensor path. It does *not*
beat raw NumPy on this chain, because the chain is compute-bound on libm
transcendentals — eliminating 117 MB of memory traffic buys nothing when the CPU
is waiting on `sin`, not on memory.

NumPy is not autograd-capable, so parity with it while remaining differentiable
is an acceptable outcome, not a failure.

---

## 4. The two shells

They are deliberately separate types.

### `profiled_c_shell.py` — the launch boundary

Must stay a ~2 ns boundary; casual callers pay nothing. Now carries statistics
accumulated **inside C** (calls, min, max, totals, failures), an optional logger
hook, a `language` tag so a hot swap is observable, and boundary calibration.

> **Trap:** calibration must time a *batch* and divide. An empty launch costs far
> less than one tick of `QueryPerformanceCounter` (~100 ns), so per-call samples
> all floor to zero. `turing_measure_launch_overhead_ps` returns picoseconds for
> this reason.

### `dispatch_shell.py` + `dispatch_shell.cpp` — the decision surface

A cube of **modules × backends × reduction-stages**. Every launch routed through
a decision records itself into the cell that decision named, so the selection
surface is measured rather than assumed.

Accessors return exactly what a GNN consumes:

```
features      float32[modules, feature_dim]        node features
edge_index    int32[2, edges]                      COO, PyTorch Geometric
observations  float64[modules, backends, stages]   NaN where unmeasured
decisions     int32[modules, 2]                    (backend, stage)
```

Policy lives entirely in Python; `select_best()` is only a measured baseline for
a learned policy to beat. Unmeasured cells read **NaN**, never 0, so an
unexplored option cannot masquerade as a free one.

> **Trap:** the implementation is a separate translation unit, not embedded
> source, because cffi's ABI-mode wrapper performs implicit `void*` → `T*`
> assignment — valid C, rejected by C++. The whole module cannot compile as C++.

---

## 5. Traps discovered the hard way

Each of these cost real time this session.

1. **Torture cache identity.** `semantic_record()` hashed name/tier/operations/
   inputs but nothing derived from the program body. Editing a case left its
   digest unchanged and served an artifact compiled for the *previous* program —
   presenting as a wrong-shape failure next to `cache=hit`. Fixed by adding the
   reference output signature and bumping the schema to v2. **Any change to a
   case body must change the digest.**

2. **`gfortran` fails silently.** Invoked by absolute path without its own `bin`
   on `PATH`, `f951` cannot load its support DLLs and the driver exits non-zero
   with *no diagnostic at all*. `compile_module` and `compile_shared_library`
   both prepend the toolchain directory to the child's `PATH`.

3. **MCJIT owns the code.** Dropping an `OptimizingJITProgram` frees the
   executable memory while function pointers still reference it — the next call
   is an access violation. Callers must hold the program.

4. **`noalias` does not gate vectorization.** LLVM versions the loop behind a
   runtime alias check and vectorizes either way. `noalias` removes the check
   (95 → 85 asm lines), it does not enable SIMD. An earlier claim to the contrary
   in this session was wrong.

5. **FMA contraction breaks bit-exactness.** `gfortran -march=native` and
   `gcc -march=native` contract `a*2.5 + b` into one FMA — one rounding instead
   of two. More accurate, but not bit-identical to NumPy or to LLVM without
   `contract`. Cross-backend comparison needs a 1-ulp tolerance or
   `-ffp-contract=off`.

6. **Don't derive what states meaning.** Recomputing the reshape target from the
   shape silently changed the operation being measured. Case parameters that
   define *what is measured* should be stated and validated, not inferred.

7. **Measuring pointers.** Building `ctypes` pointers inside a timed region
   charges compiled paths for marshalling NumPy never pays. Hoist them out.

8. **Element-wise ULP is meaningless across zero.** `a*2.5+b` on normal data
   crosses zero, `np.spacing` underflows, and any difference looks enormous.
   Measure against the data scale.

---

## 6. Open work

### Done since the numbers above were measured

**`torch_cuda` backend.** `BACKENDS` in `backend_torture_runner.py` now has a
`torch_cuda` entry that threads `device="cuda"` through to
`PyTorchTensorOperations` (`eager_backend_result` gained a `device` kwarg); it
raises rather than silently falling back if CUDA is unavailable. The plain
`torch` row is unchanged (still CPU) so the two are directly comparable. The
matrix in section 3 above still reflects the old torch-CPU-only numbers — a
GPU-conclusion rerun of the fused chain and the 15×9×4 sweep with `torch_cuda`
included is still open.

**`fortran` torture backend row.** `fortran_jit_backend.py` bridges a captured
tape straight into an SSA `Function`
(`ssa_fortran_backend.py`'s `emit_module`/`compile_module` machinery), links a
generated C shim into the same `turing_compute_closure` ABI the C/LLVM
backends use, and loads the result with `ctypes.CDLL` — not cffi, on purpose:
`ffi.verify` is the 10x-slower path this doc's own numbers (section 3, "native
compile vs cffi") rule out. Only elementwise-shaped tapes are covered: the
Fortran emitter declares every array dummy over one shared extent, so
anything shape-changing (matmul, permute, cumsum along an axis, stack, cat,
dim-scoped reductions) reports a `FortranJITShortfall` instead of being
guessed at. All of the isolated-tier elementwise cases, `where`, `flat_sum`,
`operator_grab_bag`, and — the one that matters for section 3's headline
number — `fused_chain` all pass. Not yet run against the large tier or folded
into a fresh fused-chain comparison table.

### Immediate, small, and changes existing numbers

### Threading in the C++ shell

Recommended design, **two-tier**:

- **Inside kernels: static or guided block partitioning, not work stealing.**
  Elementwise work is uniform per element over contiguous memory — balanced by
  construction. Static spans give sequential prefetch and no false sharing;
  stealing would add per-chunk atomics and cross-core traffic to fix an
  imbalance that does not exist.
- **Across the module DAG: work stealing.** That graph (already in
  `dispatch_shell`'s `edge_index`) is genuinely irregular — heterogeneous
  modules, different backends, different costs, real dependencies. That is what
  a stealing deque is for.

Note: at 1M elements the arithmetic chain is **memory-bandwidth-bound** and
saturates DRAM at 2–4 threads. Threading helps the compute-bound transcendental
chain far more. Key the thread-count policy off the cube's measured work/byte
ratio.

### OpenGL workgroup deployment

Wanted as a **distinct category** from non-workgroup dispatch, so the two are
compared rather than conflated.

### SSA → SPIR-V

**First cut landed.** `src/compiler/ssa_spirv_backend.py` mirrors
`ssa_fortran_backend.py`'s structure — `emit_function`/`emit_module`,
non-raising `SPIRVShortfall` collection, an optional `compile_module` that
shells out to `spirv-as` when present — but consumes stage 4's SSA
`IRModule` (`precompile_to_ssa.py`) rather than a tape, so control and
numeric are already unified by the time this backend sees them (see
`docs/PIPELINE_STAGE_DISAMBIGUATION.md`). Scoped the same way the Fortran
backend was on its first pass: straight-line, single-block, scalar-only
functions. Array operands/outputs, multi-block control flow (`Br`/`CondBr`/
`Phi`), structural ops, and region calls all report an honest
`SPIRVShortfall` rather than being guessed at — none of those are silently
wrong, they are just not reached yet. `tests/test_ssa_spirv_backend.py`
covers each shortfall path plus one emitted module actually assembled and
validated through the real `spirv-as`/`spirv-val` toolchain (present on this
machine via the Vulkan SDK; the test skips cleanly where it isn't).

Two things worth knowing before extending this:

- **float64 breaks GLSL.std.450's trig/exp/log/pow subset.** `spirv-val`
  rejects `Sin`/`Cos`/`Tan`/`Exp`/`Log`/`Pow`/`Tanh`/etc. on a `double`
  operand outright — that extended-instruction subset is specified for
  16/32-bit float only, unlike the "common" subset (`Sqrt`, `FAbs`, `FSign`,
  `Floor`, `Ceil`, `Trunc`, `RoundEven`, `FMin`, `FMax`), which does allow
  double. Since float64 is this repo's default dtype everywhere else, this
  is a real, expected shortfall (`_EXTINST_FLOAT16_32_ONLY` in the backend),
  not an edge case — any transcendental-heavy float64 chain will report it
  until there's a deliberate narrowing/promotion story.
- **`CondBr` attribute mismatch, discovered while grounding this work.**
  `precompile_to_ssa.py`'s `_ControlSSABuilder` emits
  `attributes={"true_target": ..., "false_target": ...}` on `CondBr`, but
  `ssa_fortran_backend.py` (and the hand-built test fixtures) only read
  `"true"`/`"false"`. So control-flow SSA coming out of the real
  `lower_precompile_and_control_to_ssa` pipeline does not have its branches
  correctly resolved by the *existing* Fortran backend either — this
  predates the SPIR-V work and needs a decision (fix on one canonical key
  pair, or read both) before either backend's multi-block support is real.

The buffer/descriptor binding ABI (SSBOs, descriptor sets, workgroup
dispatch) needed to turn one of these functions into an actual runnable
GLCompute/Vulkan kernel is still separate, later work — this backend proves
the op-level translation, not the deployment story. `nodus_graph_ir.py`
(`process_graph_to_nodus_graph_ir(graph) -> str`) remains a distinct, older
text emitter straight off `ProcessGraph` (stage 1), not this pipeline; worth
not confusing the two just because both eventually target Nodus/SPIR-V-style
consumers.

### Smaller items

- **Arithmetic-only fused chain.** The existing chain is transcendental-heavy,
  where fusion cannot win. `llvm_jit` already beats raw NumPy on single-op `add`
  (4.7 ms vs 15.2 ms at 1M), so a 15-op arithmetic chain should show fusion's
  real case. Add it beside the current one.
- **Structural ops in the Fortran emitter.** `reshape`, `cat`, `stack`,
  `permute`, `cumsum` (and anything mixing array extents, e.g. `matmul`,
  `dim_sum`) are reported as `FortranJITShortfall`s by `fortran_jit_backend.py`,
  not silently wrong. The emitter declares every array over one shared
  extent; these need real shape modelling before they can be more than a
  shortfall.
- **Full-matrix crash.** All backends × all three default tiers in one process
  exits 255. Each backend alone and each tier alone pass. Looks like cumulative
  resource exhaustion across GPU contexts, JIT allocations and cffi modules.

---

## 7. Reproducing

```powershell
# Backend matrix, one backend at a time (avoids the full-matrix crash)
python -m src.common.tensors.accelerator_backends.backend_torture_runner `
    --backend llvm_jit --tier large --llvm-profile tuned --json report.json

# A/B the LLVM tuning
python -m src.common.tensors.accelerator_backends.backend_torture_runner `
    --backend llvm_jit --tier large --llvm-profile reference

# Focused suites
python -m pytest tests/test_ssa_fortran_and_optimizing_llvm.py `
    tests/test_dispatch_shell.py tests/test_native_library.py `
    tests/test_fortran_control_target.py tests/test_tensor_torture.py -q
```

`--llvm-profile` matters: over the isolated tier tuned measures **0.70×** (at 24
elements the vectorized prologue costs more than it saves), over the large tier
**2.12×**. Quoting either alone is misleading.

Toolchains in use: gcc/gfortran 16.1.0 from MSYS2 (`C:\msys64\mingw64\bin`),
llvmlite 0.47 / LLVM 20.1.8, MSVC for cffi only.
