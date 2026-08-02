# Shared-memory WebAssembly homepage handoff

Corrected and verified 2026-08-01.

> Historical v2 record. The host-scheduled deployment described below has
> been superseded by the translated v3 coordinator and class-memory inventory
> in [`WASM_CLASS_COORDINATOR.md`](WASM_CLASS_COORDINATOR.md). Keep this file
> for the v2 ABI archaeology; do not treat its "deliberately unresolved"
> coordinator note as current status.

## Product boundary

The published Mandelbrot homepage is **one compiled program object** named
`render`. It has one authoritative `CompiledProgramAPI`, one logical
ProcessGraph, four public inputs (`t`, `unit_x`, `unit_y`, `interest`), and
three public outputs (`red`, `green`, `blue`).

The seven live WebAssembly regions are private deployment artifacts. They are not
seven public objects, do not create a second API, and do not reinterpret the
source as object-oriented classes. The Graph tab shows the logical
ProcessGraph and the reduced ProcessGraph, followed by the live deployment
schedule. A formerly emitted eighth region was dead work after the RGB outputs;
partitioning now prunes that tail before forming regions.

This work compiles only the reduced Mandelbrot/Julia math-and-colour toy in
`build_homepage.py`. It does **not** attempt to compile AVI/JPEG encoding.
Codec experiments and the separate `wasm-gallery` scratch pages are outside
the published homepage and are not its build source.

## Published execution model

`build_homepage.py` compiles the existing kernel to a reduced `FusedProgram`,
selects its RGB outputs, and calls
`wasm_class_modules.partition_reduced_program`. The resulting regions are
emitted as shared-memory punch cards and written to `site/v2/wasm/*.wasm`.
The same reduced program is also emitted once as
`site/v2/wasm/render_contiguous.wasm`.

`index.html` contains the logical API, graph/schedule descriptions, controls,
and source-download metadata. It contains neither WASM binaries nor language
source bodies. On staged Run, `wasm_html_shell.ClassGraphRunner` creates one
`WebAssembly.Memory`, follows compiler-derived edges, fetches a region only
when execution reaches it, instantiates it once against `env.memory`, and
caches the instance. The coordinator writes each public feed once, assigns one
global offset to every boundary value, and passes offsets through the punch
cards. It copies no tensor payload through JavaScript at a seam and reads back
only the three public RGB outputs.

The Execution shape buttons retain the same object and API while switching to
the contiguous module. That artifact is not fetched until its first selected
run. It provides the useful comparison between modular code delivery and one
fully fused function without rebuilding the page or changing its inputs.

`build_embedded_class_graph(..., embed_binaries=True)` remains available for
callers that genuinely need a self-contained file. The homepage deliberately
uses `embed_binaries=False` and relative `site/v2/wasm/` URLs. The versioned
directory makes future deployment-ABI changes additive instead of silently
replacing the files used by an older generated page.

## Shared static data and global slots

Every region imports exactly one memory. Lookup tables and varying tensor
constants are assigned disjoint absolute ranges at emission time; their active
data segments initialize those ranges when the module is instantiated. The
manifest's `shared_static_bytes` is the first safe byte for dynamic arrays.

At each domain extent the runner lays out one slot per logical input and one
slot per produced boundary value. An edge binds a consumer parameter directly
to its producer's slot. Fan-out therefore reuses an address rather than
duplicating the tensor for each consumer. Memory may grow between extents, but
the numeric ABI remains `(count, input offsets..., output offsets...)`.

## ProcessGraph hue and profiling view

The shell carries all 557 original nodes and all 3,099 output-reachable reduced
nodes. Every node retains a schedule level and group. The user can click between
the original semantic/AST ProcessGraph and the reduced ProcessGraph.

Top-level source functions seed conceptual hue identities. Reduction regions
and public feeds seed procedural/data identities. Those identities propagate
through graph dependencies, so a downstream node lists every contributing
identity. The browser mixes their hue values at redraw time.

Runtime profiling drives the same view. Every completed punch card deposits
measured-duration energy into its reduced nodes. `phosphorColor(node, now)`
integrates every deposit in a rolling window and applies exponential decay.
Calls that finish several times between display refreshes accumulate brightness
instead of disappearing. The decay slider changes the view function, not the
recorded provenance or timings.

## Source publication

All available language sources live under
`site/v2/source/render/<language>/`. The page embeds only filename, byte/line
metadata, and URL. Selecting a language tab does not fetch it. A fetch exists
only inside that language's Download button click handler, which then creates
the browser download from the returned blob.

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

The segmented runner operates on the requested `count`, writes public typed
arrays once, broadcasts only length-one inputs, returns the public output arrays, and
supports the same repeat/continuous animation and feedback loop as the former
single-module path. Logical outputs are resolved by explicit
`logical_outputs` bindings, not by assuming every result belongs to a chosen
root chunk.

## Verification performed

Focused compiler/runtime suite:

```text
python -m pytest tests/test_process_graph_fusion.py tests/test_wasm_binary.py \
  tests/test_wasm_class_modules.py tests/test_wasm_html_shell.py -q
81 passed
```

Real homepage build:

```text
python build_homepage.py
8/8 backend source views emitted
7 shared-memory WASM regions and one contiguous module written to site/v2/wasm/
```

The generated directory was served over HTTP and loaded in isolated headless
Edge. Initial load fetched no source and no WASM. A real 256x256 staged run
fetched exactly seven region files, illuminated 2,695 reduced operation nodes,
and reported 118/117/117 distinct RGB values. Selecting contiguous then fetched
only `render_contiguous.wasm` and produced the same distinct counts. Clicking
Fortran Download fetched only `render.f90` (161,301 bytes).

An independent Node/WebAssembly parity runner executed both shapes at 32x32:

```text
7 staged modules, each importing only env.memory
maximum staged-versus-contiguous RGB difference: 0
red/green/blue distinct values: 111 / 110 / 110
```

This browser check caught and then verified the ABI-order fix; a successful
WebAssembly instantiation alone is not considered sufficient.

## Rebuild and publish

From the Turing repository root:

```text
python build_homepage.py
```

Commit `../index.html` and `../site/` in the parent `nogodsnomasters`
repository. Commit the compiler/runtime sources and tests that produced them
in Turing. GitHub Pages serves the parent repository root, so relative module
URLs resolve without a special loader or a duplicated site inside Turing.

## Deliberately unresolved

`emit_class_modules(link_calls=True)` remains a separate experimental function-
import mode. The deployed staged runtime intentionally uses host scheduling
over one shared memory because this keeps pieces independently lazy-loadable
while making seams offset-only. A future root WASM orchestrator could call the
same punch cards directly, but it is no longer required to eliminate tensor
transfers.

`WASM_REGION_STEPS = 400` remains an explicit deployment choice rather than an
automatic cost heuristic. Changing it must not change the public program API.
