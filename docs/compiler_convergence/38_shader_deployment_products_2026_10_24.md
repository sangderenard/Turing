# 38. Shader deployment products (2026-10-24)

Answers `docs/NODUS_SHADER_DEPLOYMENT_REQUEST.md`. This is packaging and
deployment metadata only. No SSA lowering and no planner code changed.

## What a product directory holds

Source products (WGSL, desktop GLSL compute, WebGL fragment, extracted host
shaders), written by `WGSLModule.write`, `GLSLComputeComponent.write`,
`WebGLFragmentModule.write` and `shader_deployment.publish_extracted_shader`:

| File | Content |
| --- | --- |
| `<name>.wgsl` / `.comp` / `.frag` / ... | the emitted source, byte for byte |
| `<name>.vert` | WebGL only: the companion full-screen vertex stage |
| `<name>.api.yaml` | the existing `turing-compiled-program-api-v1` sidecar, plus `metadata.shader_deployment` |
| `<name>.shader.json` | `{"schema": "nodus-turing-shader-artifact-v1", "source", "api", "targets"}`, exactly the four keys Nodus accepts |
| `SHA256SUMS` | `sha256sum` format over every file above |

SPIR-V products, written by `shader_deployment.publish_spirv_product`:
`<name>.spvasm`, `<name>.spv` (when `spirv-as` exists), `<name>.spirv.json`
(`turing.spirv-product.v1` with `product_kind`), `SHA256SUMS`, and for a
deployable shader also `<name>.api.yaml`.

## The deployment record

`metadata.shader_deployment` (`turing.shader-deployment.v1`,
`src/compiler/shader_deployment.py`) refers to `metadata.component_abi` by
`component_id` and never copies its ports. It adds:

- `source`: file and SHA-256;
- `language`, `version`, `profile` (read from the `#version` directive),
  `stage`, `entrypoint`;
- `target`: `client`, `environment`, `spirv_version`;
- `resources`: per physical resource, `set`, `binding`, `kind`, `access`,
  `layout`, `dtype`, `element_bytes`, `byte_offset`, `byte_extent`,
  `extensions`, `capabilities`, `specialization_constants`, and `realizes`:
  one view per logical port slot with its byte offset and extent, `rank`,
  `shape` (null when runtime-sized) and `extents_from`;
- `launch`: `local_size`, `group_count` (a rule, or exact groups),
  `element_count` (where the count comes from), `extents.order`, and
  `controls`, each with one derivation per element;
- `provenance`, `complete`, `shortfalls` (stable `code` plus detail).

A byte quantity supplied at launch is `{"control", "index", "scale"}`: for
example `{"control": "u_slot", "index": 2, "scale": 4}` is `4 * u_slot[2]`.
Runtime extent spellings are the ones `turing-runtime-abi-v1` uses
(`src/compiler/runtime_abi.py`, rank-known tensor lowering thread): `rank`
always present, `shape` or null, `extents_from` is `"fed"` or the index of
the input whose extents an output takes, `extents.order` entries are
`{slot, buffer, kind, axis}` with `kind` an open string.

## Per emitter (observed from each emitter's own code)

- **Desktop GLSL compute** (`glsl_component_backend`). The std430 `Arena`
  of `uint` words at set 0, binding 0, plus two read-only std430 control
  blocks: `TuringLaunchSlots { uint u_slot[]; }` at binding 1 and
  `TuringLaunchExtents { uint u_count; uint u_extent[]; }` at binding 2
  (section "Vulkan-ready arena controls" below). Every port is a view at
  `4 * u_slot[slot]` bytes, extent `4 * u_extent[0]` unless a static shape
  was given. Controls: `u_count` and `u_extent[0]` are the element count,
  `u_slot[i]` the arena word offset of port `i` (`_bind_launch_controls`).
  Group count: any grid covering the element count (`turing_linear_gid`
  folds x/y/z; the `u_extent[0]` guard stops the rest). The logical
  `backend_binding` numbers are not descriptor bindings here.
- **WGSL** (`ssa_webgpu_backend`). One storage buffer per port at
  `@group(0) @binding(backend_binding)`, extent `4 * count` (times the member
  count when packed), exact groups from the launch plan, count baked as a
  literal.
- **WebGL fragment**. Sampled textures (by uniform name) and draw buffers
  (by location), `turing_resolution` when sampling is normalized.
- **Extracted host shader**. Exact source plus filename, line, source
  symbol, compiler call and digest, and a component ABI read from the
  source's own global declarations (section "Extracted shader interface").
- **Deployable SPIR-V** (`ssa_spirv_backend.emit_deployable_compute_shader`).
  `main` is a GLCompute entry with `LocalSize`; it reads one element of each
  feed at `GlobalInvocationId.x`, calls the library translation of the SSA
  function, writes one element of the output, and guards on the
  `element_count` word of a control storage buffer. Feeds, output and
  control are storage buffers in set 0 at contiguous bindings. No `Linkage`
  (Vulkan forbids it). SPIR-V 1.3, `vulkan1.1`. The library product
  (`emit_function`/`emit_module`) is unchanged and is published as
  `linkable-spirv-library` with no entry point and no API.

## Shortfalls published (each checked, not assumed)

- `vulkan-default-uniform-controls` (GLSL compute), as first published:
  `u_count`, `u_extent` and `u_slot` were default-block uniforms, which
  `glslangValidator -V` refuses ("non-opaque uniforms outside a block").
  Closed by the follow-up below; the record no longer carries it.
- `graphics-stage` (WebGL): a raster pipeline with the companion vertex
  stage, not a compute dispatch.
- `no-component-abi` (extracted shaders): only when the declared
  interface cannot be read with certainty, with the reader's reason.
- `link-assigned` (extracted shaders): a resource whose binding or
  location the source leaves to the host program or the linker.
- `host-launch` (extracted compute shaders): the host's dispatch call picks
  the group count.
- Deployable SPIR-V refuses by name: a runtime-sized span (rank stated, no
  static shape), a dtype with no storage element (bool), or no result.
  Array-shaped values and several results are carried (see "Instances").

## Fault found on the way

`emit_glsl_compute_component` labelled every port `f32`, while the source it
emits reads each value through its metadata dtype (`int(arena[...])` for
int32). A consumer trusting the logical ABI would feed float bits to an
integer port. Ports now carry the dtype the source reads, through the same
`_normalize_dtype` rule. Test:
`test_glsl_compute_ports_carry_the_dtype_the_source_reads`.

## Evidence

`tests/test_shader_deployment.py` (13 tests). Where the tools exist it
compiles each GLSL product with `glslangValidator`, assembles and validates
the deployable SPIR-V for `vulkan1.1`, and runs it on a Vulkan device through
`tools/vulkan_compute_probe.c`, driving bindings, the control word and the
group count only from the published record. On lavapipe (CPU) float32 add,
int32 mul and int32 sub match numpy bit for bit over 1000 elements, and the
words past the element count are untouched with an extra group dispatched.

## Open

- The installed control shell (`compose_control_shader`) compiles for
  Vulkan now (section "Vulkan-ready control shell"), but it is not a
  published product yet.
- The C lane drops a lowered pack core's Precision-typed formals from its
  public buffer ABI (its run corrupts the heap), so the precision lane's two
  C-comparison tests fail loudly by name. The GPU result is checked against
  a reference evaluation of the lowered module instead (section "Precision
  lane reads today's lowering").
- A lowered pack core loads one double per element with a 0.0 low limb and
  stores only the leading limb. Albert decided this is a bug for limbed
  `Precision[k]` inputs: they must load and store every limb. It is the
  next follow-up.
- Deployable SPIR-V for a matmul, an axis reduction, or any function that
  mixes a reduction with other work over runtime-sized spans. Elementwise
  functions deploy in element mode, and a lone whole-span reduction deploys
  in reduction mode (see "Reduction mode").

## Vulkan-ready control shell (follow-up, 2026-10-24)

Wrong record: `compose_control_shader` declared `uniform uint u_count`,
`uniform uint u_debug_capacity` (debug builds) and `uniform int
u_dispatch_iteration` (C-planned dispatch loops) in the default block.
`glslangValidator -V --target-env vulkan1.1` refused every emitted shell at
main cbe2263 ("non-opaque uniforms outside a block").

Rule: the three scalars ride in one std140 uniform block,
`TuringShellControls { uint u_count; uint u_debug_capacity; int
u_dispatch_iteration; }`, at binding 8 (`SHELL_CONTROL_BINDING`). Every field
is always declared, so the layout does not change with debug or loop
options. It is a uniform block rather than a ninth storage block because the
shell's storage bindings 0..7 already reach the GL 4.3 minimum of eight
storage blocks per compute shader; uniform blocks have their own GL binding
space, and in a Vulkan set binding 8 follows the storage buffers. The GL
host (`_ShellControlBuffer`, called from `_dispatch_composed_control`)
uploads `[count, debug capacity or 0, dispatch iteration or 0]` per dispatch,
re-specifying the store like the arena controls. The refusal "C-planned GLSL
closure lacks u_dispatch_iteration" used to come from an inactive uniform
location; it now comes from installing a C-planned artifact whose phase
source never reads the field.

Evidence (cloud: llvmpipe OpenGL 4.5):

- `test_control_shell_scalars_ride_in_a_vulkan_ready_block` (with and
  without instrumentation) checks the block and compiles the shell with
  `-V --target-env vulkan1.1`; `test_c_dispatch_shell_without_the_iteration_read_is_refused`
  keeps the refusal;
- the installed-shell tests that read `u_count`, the debug capacity and the
  dispatch iteration on a GL device pass (one-dispatch loop with debug
  records and no overflow, stream publications, the C-planned loop's three
  frames);
- the 12-file GPU test set fails the same 27 names and errors the same 4 on
  this branch and on main cbe2263.


## Element mode (runtime-sized spans)

Program: `(x - y) * (x + y) * 0.5` with `x` and `y` declared `storage: span,
rank: 2` and no shape. Its lowered region is elementwise over runtime-sized
spans. Wrong record: the deployable entry refused it, because instances need
static extents.

Rule: when every span is runtime-sized with one rank R, and every operation
over a span is elementwise with a span result, invocation `i` computes
element `i` of every span. Rank-0 feeds broadcast: each invocation reads
their element 0, and their `byte_extent` is one element. `element_count` is
the product of the host's extents (`launch.element_count.rank` = R,
`elements_of` = the first span feed's slot). Span views state `rank` R,
`shape` null, and `extents_from` (`"fed"`, or the first span feed's slot for
an output). Mixed ranks, a non-elementwise operation, or a scalar output
beside span outputs are refused by name (a lone reduction runs in reduction
mode instead).

Evidence: `test_runtime_sized_elementwise_deploys_one_element_per_invocation`,
`test_runtime_sized_deployable_spirv_matches_python_on_vulkan` (lavapipe, a
(17, 13) feed the shader never saw, matches numpy; the words past the count
are untouched), and
`tests/test_runtime_sized_spans.py::test_rank_only_region_deploys_as_elementwise_spirv`
(the sanctioned entry's region deploys and passes spirv-val).

## Reductions (static shapes)

Program: `x.mean() - y.prod()` with `x` and `y` of static shape (5,). Each
reduction lowers to its own region, and the SPIR-V library refused every one
("reductions over arrays are not yet supported").

Rule: a whole-array `sum`, `prod`, `max`, `min`, `mean`, `all` or `any` of
a static array unrolls. Elements are loaded row-major and folded left to
right. `sum(xs, start)` starts from `start`. `max` and `min` let a NaN in
and never displace one, which matches numpy and the C backend. `all` and
`any` count NaN as true. `mean` divides the sum by the element count and
needs a float result. An axis reduction, a bool sum, and `max`, `min` or
`mean` of no elements are refused by name. Over runtime-sized spans see
"Reduction mode".

Evidence: `tests/test_ssa_spirv_backend.py::test_whole_array_reductions_assemble_and_validate`
(float32, float64, int32 pass spirv-val),
`tests/test_shader_deployment.py::test_reductions_over_instances_match_python_on_vulkan`
(lavapipe, 200 (2, 3) instances with one NaN, matches numpy), and
`tests/test_runtime_sized_spans.py::test_reduction_regions_deploy_static_and_runtime_sized`
(the sanctioned entry's reduction regions deploy and pass spirv-val).

## Reduction mode (runtime-sized spans)

Program: `x.mean() - y.prod()` with `x` and `y` declared `storage: span,
rank: 1` and no shape. Each reduction region was refused: the library has
no count to unroll, and element mode writes one element per invocation.

Rule: a function whose only instruction is a whole-span `sum`, `prod`,
`max`, `min`, `mean`, `all` or `any` of a runtime-sized span feed (`sum`
may take a scalar start feed) deploys as ONE looping invocation.
Invocation 0 runs a structured loop (`OpLoopMerge`) over
`[0, element_count)` and folds left to right with the unrolled form's rules.
Then it writes element 0 of `output_0`. Every other invocation does nothing.
The record says:

- `launch.group_count` is `{rule: fixed, groups: 1}`.
- `launch.element_count` names `reduction` and `minimum`. The minimum is 1
  for `max`, `min` and `mean`: with no elements they have no value, and the
  shader writes nothing.
- The span feed is sized by `element_count`. The output and any start feed
  are one element.

An axis, a bool result, or a reduction beside other work is refused by name.

Evidence: `tests/test_shader_deployment.py::test_runtime_sized_reduction_deploys_as_one_looping_invocation`
(the record), and `test_runtime_sized_reduction_matches_python_on_vulkan`.
That second test runs on lavapipe and covers:

- every op over float32, and `sum`/`max` over int32;
- (7, 5) extents the shader never saw;
- a NaN for `max`/`min`;
- an empty span, where `max`/`min`/`mean` leave the sentinel untouched;
- two groups launched, with no effect beyond the first invocation.

## Extracted shader interface (follow-up, 2026-10-24)

Wrong record: every extracted host shader was published with no component
ABI and the `no-component-abi` shortfall, so a consumer had to parse the
source to find what to bind.

Rule: `glsl_interface.read_glsl_interface` reads the global declarations,
and only those: stage inputs and outputs (`attribute`/`varying` by stage),
default-block uniforms, opaque uniforms, uniform/buffer/IO blocks and their
members, `layout` qualifiers, and a compute `local_size`. Function bodies,
comments, `const`/`shared` globals and builtin redeclarations are not
interface. `extracted_shader_component` turns each declaration into one
`turing.shader-component.v1` port (feeds, then outputs, then uniforms, each
in declaration order) and one physical resource:

| Declaration | role | transport / kind | access |
| --- | --- | --- | --- |
| vertex `in` | feed | vertex-attribute | read |
| other `in` | feed | stage-input | read |
| fragment `out` | output | draw-buffer | write |
| other `out` | output | stage-output | write |
| `buffer` block | feed if `readonly`, else output | storage-buffer | read / write / read_write |
| `uniform` block | uniform | uniform-buffer | read |
| sampler/texture | feed | texture / sampled-texture | read |
| image | feed if `readonly`, else output | storage-image | as declared |
| plain `uniform` | uniform | default-uniform (by name) | read |

Dtypes: `f32`/`i32`/`u32`/`bool`, vectors `f32x4`, matrices `f32x4x4`
(columns first), opaque types by GLSL name, and `block` for a block of
several members, whose members are listed on the resource. A runtime-sized
array is `shape: null`, `extents_from: "fed"`. `backend_binding` is the
declared binding, else the declared location, else `-1`. A sampler, image or
block without a `binding`, a vertex attribute without a `location`, or one
of several fragment outputs without one, is named in a `link-assigned`
shortfall. Graphics stages keep `graphics-stage`. A compute shader states
its local size and carries `host-launch`.

Refused by name (the product keeps `no-component-abi` with the reason):
conditional compilation (`#if`, `#ifdef`, ...), a macro in a global
declaration, a struct-typed interface variable or block member, and a
non-literal array size or layout value.

Evidence:

- `tests/test_extracted_shader_interface.py`, with `glslangValidator`'s
  reflection database (`-l -q --reflect-separate-buffers`) as the oracle:
  for all 24 shaders the extractor finds in `src` (12 vertex, 12 fragment),
  every uniform, block, buffer variable, pipeline input and output glslang
  reports appears in the record with the same type, binding and runtime-array
  status. The same check passes on a compute shader with every resource kind;
- on llvmpipe, an extracted compute shader runs from its record alone
  (storage buffers bound at the record's bindings, uniforms set by the
  record's names, groups from its local size) and matches numpy over 1000
  elements, with the words past the count left untouched.

## Precision compute product (follow-up, 2026-10-24)

Wrong records found on the way:

- **Baked trip count read every coefficient one slot off.** With the count
  baked, `emit_ssa_module_to_glsl_compute` dropped `element_count` from the
  scalars block but kept `uint _pad0`, so `scalars[]` sat at byte 8 while
  `prepare_execution` packs the doubles from byte 0. On llvmpipe,
  `y = fma(x, 3, 0.25)` computed `x * 0.25`. The block now has no header when
  the count is baked (glslang reflection: `ScalarFeed.scalars` at offset 0).
- **GLSL `fma()` is not single-rounding on Mesa.** A `precise double`
  `fma()` on llvmpipe and on lavapipe rounds twice: 45 of 777 elements equal
  `x*c + d` and differ from the once-rounded value. GLSL allows this (it
  fixes only that every fma() consumed by a precise variable rounds the same
  way). The record publishes an `fma-rounding` shortfall whenever the source
  calls `fma()`.

Rule: `GLSLComputeArtifact.component_abi`, `.deployment()`
(`shader_deployment.precision_compute_deployment`) and `.write()` publish
the layout the emitter already documented:

- ports: read buffers, then each coefficient (a feed carried in the scalars
  block), then written buffers; all `f64`, `value_id` the wrapper's SSA id;
- resources: `Buffer<slot>` at set 0, binding `slot` (`read`, or
  `read_write` when stored into), runtime-sized (`extents_from: "fed"`); and
  `ScalarFeed` at binding `n` with one view per coefficient at byte
  `8 + 8k` (or `8k` baked), its members listed with their offsets;
- launch: local size 64; the `element_count` control (u32 at byte 0 of
  `ScalarFeed`, derived from the count's value id) and a `cover` group rule
  on x only, or, baked, `exact` groups and no control;
- targets OpenGL 4.3 and Vulkan 1.1 (`Float64`), with the source's
  `#extension` lines on each resource.

Evidence (`tests/test_precision_compute_product.py`, kernels built by hand
in the wrapper-plus-region shape, run and baked count each):

- every block's binding and `ScalarFeed`'s member offsets match glslang
  reflection, and the source compiles for OpenGL and `-V --target-env
  vulkan1.1`;
- driven only from the record (buffer bytes from views and control offsets,
  groups from the rule), the shader matches the fma reference on llvmpipe
  and on lavapipe over 777 elements, with words past the count untouched;
  `prepare_execution` now feeds a baked kernel the coefficients it reads.

Also: `tools/benchmark_microprecision_matrix._prepare_core` lowers with an
extraction contract that declares no records (a pack core passes only
limb arrays, a count and coefficients), which `lower_ast_source_to_ssa`
requires. The lane's own tests still fail after that, on the wrapper shape
above.

## Precision lane reads today's lowering (follow-up, 2026-10-24)

The precision-core GLSL lane compiles the lowered microprecision pack cores
again (sin, width 2, the `fma` and `split` flavours), with no shortfalls.

Rule:

- the trip count is the loop bound followed through `Cast`s, so a wrapper
  whose bound is `Cast(n)` still bakes or feeds `n`;
- a loop body that calls several planned regions is fused into one body
  (`_fuse_loop_body`): each call is inlined in body order with formals
  replaced by actuals, a call's result maps to its region's `Ret` operands,
  and `GetElementPtr(result, Const k)` + `Load` reads operand `k`. A callee
  called twice, a multi-block region, or any other use of a call result is
  refused by name;
- `tools/benchmark_precision_cores._roles` picks the count as the int formal
  that is the loop bound; other int formals (limb extents) are extras.

Evidence (`tests/test_ssa_glsl_compute_backend.py`):

- both flavours emit and compute sin on llvmpipe;
- `test_gpu_matches_reference_evaluation_of_the_lowered_core` runs a Python
  interpreter of the original (unfused) module over the same feeds: the
  split flavour is bit-identical; the fma flavour equals the once-rounded
  or twice-rounded reference per element (Mesa, see above); both are within
  1e-13 of `np.sin`;
- the two C-comparison tests now fail by name before running (every wrapper
  formal must be in the C artifact's `buffer_order`), instead of aborting
  the pytest process with `free(): invalid next size`.
