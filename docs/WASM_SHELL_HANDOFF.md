# WebAssembly shell and math handoff

Written 2026-08-01. State of the demo page, what is finished, what is not,
and the one design rule that keeps being the thing worth protecting.

## The rule

**Everything is one ingested program.** Source → AST → `ProcessGraph` → AOT →
backend. If part of the algorithm ends up running in JavaScript, or in a
second hand-built module, the pitch is gone: the page stops being *the
compiler's output* and becomes a page that happens to contain some output.

Two specific ways this has already nearly gone wrong:

- the camera trajectory was written as JavaScript feed expressions, so the
  page was computing the tour and the module only shaded it;
- building a `FusedProgram` by hand is always the wrong move. `FusedProgram`
  is a downstream leaf, not the object. The AOT compilation is the object.

## What is finished

**The compiler path.** `build_homepage.py` compiles one kernel through
`compile_ast_aot` and emits it through eight backends — SSA, Fortran, SPIR-V,
GLSL, WebAssembly, NumPy, PyTorch, AbstractTensor — all from that single
compilation (`src/compiler/backend_sources.py`). 8/8 currently serve it.

**The assembler.** `src/compiler/wasm_binary.py` writes real `.wasm` — LEB128,
section ordering, data segments. No `wat2wasm` anywhere. Verified by running
the modules in a browser, not by inspecting bytes.

**The math module.** `src/compiler/wasm_math_tables.py` plus the cache in
`src/compiler/math_cache/` (15 tables, 20.2 MB, flat little-endian f64, one
manifest). Sizing is `max|f''| * h^2 / 8` with curvature measured, not
derived. Rebuild with `python -m src.compiler.build_math_cache`.

Read **`achieved`** from the manifest, not `bound`. Four tables — `acos`,
`asin`, `acosh`, `atanh` — miss their predicted bound because curvature is
singular at an endpoint and a sampled maximum is optimistic there. `acosh`
predicts 3.15e-07 and delivers 7.84e-07. Every entry carries both plus
`bound_met`.

`tan` is deliberately absent: poles inside any useful interval, so no bounded
table describes it. `log`/`log2` start at 1/4 for the same class of reason —
curvature is 1/x², and reaching towards zero wanted 64 MB while still being
worst where it was needed. Scale by a power of two and add `k*ln2`.

Series are the alternative, epsilon-driven by the same alternating-remainder
argument `llvm_signal_math._continuous_terms` already uses (sin 8 terms, cos
9, exp 15, atan 15). `exp` uses the Lagrange remainder instead, its terms
being all positive.

**In-emitter LUTs.** All fifteen catalogue functions are wired into
`fused_program_wasm_backend`. `_LUT_OPS` is taken *from* the catalogue, so
adding a function to `wasm_math_tables` makes it emittable without a second
edit. `plan_tables` lays several tables end to end, each op addresses its own
base, and `reserved_bytes` in the API descriptor tells a caller where its own
arrays may start. Tables load from the cache and fall back to computing.
Browser-verified: sin/cos/tanh to ~2.9e-07 out to |x| = 2000 (periodic range
reduction holds for an ever-growing frame counter), and exp2 4.93e-07, exp
2.28e-07, log 3.34e-07, asin 1.26e-07, atan 3.09e-07, sinh 2.07e-07, acosh
4.96e-08.

**Feeds are named after their source parameters.** `def compute(a, b)` yields
the contract `['count', 'a', 'b', 'out0']`, in the descriptor and in the WAT.
Three things had to line up: the name is kept against the resident range as
well as the object (a supplied array is rewrapped before use, so object
identity does not survive); the helper is handed into the generated
deployment class's namespace, which is an explicit whitelist; and the
descriptor loop actually uses the labels. Unnamed, non-identifier and
colliding names fall back to something bindable.

**The camera is in the ingested program.** `render` computes centre, span,
family blend and Julia constant from `t` using baked `sin`/`cos`/`exp`. The
page supplies only the grid, the clock and the routed interest. The network
drives the clock: `advanceFeedback` scores three future candidates, the best
sets a dwell speed, and that advances `travel`, which is the `t` feed via
the manifest's `travel_feed`.

**The shell.** `wasm_html_shell.py`. Telemetry, progress, per-kind log
filters, the process graph, both sources, the editable API descriptor, output
tabs, gaussian and expression feeds, continuous-by-default repeats.

## What is not finished

**1. The page is ~12 MB, uncommitted.** The parametric kernel at 160
unrolled iterations compiles fine, but the *source tabs* embed 133,920 lines
of SPIR-V verbatim. The published page is still the older 1.5 MB one. Options:
truncate each pane to a few hundred lines with a count (recommended — nobody
reads 133k lines in a browser); drop to ~64 iterations (loses the deep dive);
or fetch sources as files (page stops being self-contained).

Most of the weight is now two things: 133k lines of SPIR-V in the source
tabs, and `exp`'s 4 MB table embedded in the module (`reserved_bytes` reads
4325408). The table is the easier win -- see item 2.

**2. Tables are still embedded, not fetched.** The intended design is that
the shell fetches from `math_cache/` and writes into the reserved region, so
a page needing `sin`/`cos` pulls 64 KB instead of carrying it. `reserved_bytes`
and the manifest are the two halves that make this straightforward.

**3. Fortran Mandelbrot numeric check never completed.** Compiles and links;
its output was never compared against NumPy. Unverified, not known-bad.

## Things that cost hours, so they are written down

- **A shell script that fails to parse renders perfectly and does nothing.**
  The controls are static HTML. The diagnostics bootstrap is a *separate*
  `<script>` for this reason — a handler inside the failing script cannot
  catch its own parse error. It has since caught a duplicated `const elapsed`
  immediately, with a line number.
- **`_JS` must stay a raw string.** Twice, a JS escape written into a non-raw
  Python string became a real newline inside a string literal and killed the
  page. A test pins it.
- **`preserveDrawingBuffer` is false, so the WebGL draw must be synchronous
  inside the loop and must wait.** `drawArrays` only queues; without
  `gl.finish()` the loop starts the next compute while the frame is still in
  the queue. Deferring the draw into a frame callback takes it *out* of the
  loop and is worse.
- **Honour `reserved_bytes`.** The interest network bakes its `tanh` table at
  offset 0; the shell was laying arrays over it, destroying the activation of
  the network it was about to ask, after which scores are garbage and the
  trajectory stops — presenting as a frozen page at full frame rate.
- **Feed order is first use, not sorted id.** A value id is an allocation
  address; sorting by it made parameter order arbitrary, and getting it wrong
  computes a wrong answer from correctly-shaped inputs rather than failing.
  On an 11-input network sorted order was the permutation
  `[10,9,7,0,2,6,5,3,8,1,4]` of the source's own order.
- **`unroll_limit` was capped three ways**: hardcoded on
  `LoopBackendCapabilities`; not inherited by the callee's graph (a function's
  graph is built independently, and the callee is where loops live); and
  `max_nodes_per_dispatch` at 256, a GLSL constraint, split larger programs
  into regions so a caller asking for a flat unrolled loop silently got a
  retained one.
- **No simultaneous tuple assignment for a loop-carried value.** It binds to a
  tuple temporary and lowering fails with "carried update value N has no
  producer inside the loop body", which reads as unrelated. See
  `aot_compile.py`'s docstring.

## Rebuild

```
python -m src.compiler.build_math_cache      # only when epsilon/domains change
python build_homepage.py                     # writes index.html
```

`index.html` is served from the `nogodsnomasters` repository root. That repo's
branch is `nogodsnomasters`, not `main` — Pages must point at it.
