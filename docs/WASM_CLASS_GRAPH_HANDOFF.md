# Segmented WebAssembly homepage handoff

Corrected and verified 2026-08-01.

## Product boundary

The published Mandelbrot homepage is **one compiled program object** named
`render`. It has one authoritative `CompiledProgramAPI`, one logical
ProcessGraph, four public inputs (`t`, `unit_x`, `unit_y`, `interest`), and
three public outputs (`red`, `green`, `blue`).

The eight WebAssembly regions are private deployment artifacts. They are not
eight public objects, do not create a second API, and do not reinterpret the
source as object-oriented classes. The Graph tab shows the logical
ProcessGraph first and a clearly labelled deployment overlay underneath it.

This work compiles only the reduced Mandelbrot/Julia math-and-colour toy in
`build_homepage.py`. It does **not** attempt to compile AVI/JPEG encoding.
Codec experiments and the separate `wasm-gallery` scratch pages are outside
the published homepage and are not its build source.

## Published execution model

`build_homepage.py` compiles the existing kernel to a reduced `FusedProgram`,
selects its RGB outputs, and calls
`wasm_class_modules.partition_reduced_program`. The resulting regions are
emitted independently and written to `site-wasm/*.wasm`.

`index.html` contains the logical API, graph/schedule description, controls,
and source views. It does not contain the region binaries as base64. On Run,
`wasm_html_shell.ClassGraphRunner` follows compiler-derived edges, fetches a
region only when execution reaches it, instantiates it once, and caches the
instance for later frames. Arrays cross private module memories through the
host runner. Animation, feedback, rendering, timing, and named outputs remain
features of the one logical program.

`build_embedded_class_graph(..., embed_binaries=True)` remains available for
callers that genuinely need a self-contained file. The homepage deliberately
uses `embed_binaries=False` and relative `site-wasm/` URLs.

## Important correctness fixes

### Tensor constants belong to the IR

`tensor_from_list` is now understood at the shared `FusedProgram`/
ProcessGraph boundary. `fused_ir.flatten_tensor_constant` defines nested
numeric-list flattening, while `uniform_tensor_constant` identifies captured
broadcast scalars such as `[2, 2, 2, 2]` without discarding genuine varying
tensors.

ProcessGraph nodes preserve full tensor constant payloads and metadata.
Round-tripping a dispatch region recreates `tensor_from_list`; it does not
import a private WebAssembly helper or reject arrays. When both operands of a
binary graph operation are uniform constants, one remains a constructor and
the other uses `right_scalar`, preserving operand order without a
backend-specific constant fold.

The WebAssembly backend packs genuine varying constants into the module data
segment after lookup tables and loads element `i` from linear memory. Uniform
constants remain scalar immediates. The runner starts input/output allocation
after each module's declared `reserved_bytes`, so it cannot overwrite baked
tables or constants.

### Region manifests follow the emitted ABI

A region's boundary `input_ids` order is not necessarily its emitted function
parameter order. `emit_wasm_module` orders feeds by first use. The manifest
now pairs input names with `program_feed_order(spec.program)`, the exact same
ordering used by the emitter. Pairing names with the boundary-set order
silently routed correctly shaped arrays to the wrong operands and produced a
flat image without throwing; a dedicated subtraction regression test now
guards this case.

### Full-domain execution

The segmented runner operates on the requested `count`, copies full typed
arrays, broadcasts only length-one inputs, returns full output arrays, and
supports the same repeat/continuous animation and feedback loop as the former
single-module path. Logical outputs are resolved by explicit
`logical_outputs` bindings, not by assuming every result belongs to a chosen
root chunk.

## Verification performed

Focused compiler/runtime suite:

```text
python -m pytest tests/test_process_graph_fusion.py tests/test_wasm_binary.py \
  tests/test_wasm_class_modules.py tests/test_wasm_html_shell.py -q
75 passed
```

Real homepage build:

```text
python build_homepage.py
8/8 backend source views emitted
8 lazy WASM regions written to site-wasm/
```

The generated directory was served over HTTP and loaded in isolated headless
Edge. A real 32x32 run fetched all eight relative `.wasm` files and reported:

```text
ran 1024 elements in 117.400 ms (segmented WASM)
red:   1024 finite values, 111 distinct
green: 1024 finite values, 110 distinct
blue:  1024 finite values, 110 distinct
```

This browser check caught and then verified the ABI-order fix; a successful
WebAssembly instantiation alone is not considered sufficient.

## Rebuild and publish

From the Turing repository root:

```text
python build_homepage.py
```

Commit `index.html`, `site-wasm/*.wasm`, and the compiler/runtime sources and
tests that produced them. GitHub Pages serves the repository root, so relative
module URLs resolve beneath the project page without a special loader or a
second site.

## Deliberately unresolved

`emit_class_modules(link_calls=True)` can describe real imports, but the
current deployed design intentionally uses independently instantiated modules
and host-carried arrays. Adding direct cross-module calls would require a
separate shared-memory ABI design and is not needed by the homepage.

`WASM_REGION_STEPS = 400` remains an explicit deployment choice rather than an
automatic cost heuristic. Changing it must not change the public program API.
