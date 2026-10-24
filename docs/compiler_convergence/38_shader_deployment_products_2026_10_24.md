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
  symbol, compiler call and digest. No component ABI is derived yet.
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
- `no-component-abi` (extracted shaders).
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
- A component ABI for extracted host shaders.
- The precision-core compute lane (`ssa_glsl_compute_backend`) has no
  component ABI, so it has no product yet.
- Deployable SPIR-V for a NON-elementwise function over runtime-sized spans
  (a reduction, a matmul). That needs a structured loop in the library
  function, because static shapes are unrolled. Elementwise ones deploy in
  element mode (see "Element mode").

## Vulkan-ready arena controls (follow-up, 2026-10-24)

`compose_shader`, the one place a composed elementwise/structural shader is
finished, now declares its controls as read-only std430 storage blocks at
set 0, bindings 1 and 2, the same mechanism the control shell already used
for its tables (bindings 4 and 6). The two tables are separate so a wrong
extent count can never shift the slot offsets. Writers changed with it:

- the GL host (`_bind_launch_controls`, called from both `_dispatch_many`
  paths) uploads `u_slot[]` and `[u_count, *u_extent]` into two small
  buffers, re-specifying each store per dispatch so a queued dispatch keeps
  its own words. `_bind_arena` binds only the arena, so the control shell's
  own bindings are untouched. The dispatch now unbinds 0..2;
- the standalone BLAS C shell (`glsl_blas_deployment`) binds the same two
  buffers instead of calling `glUniform*`.

Evidence (cloud: llvmpipe OpenGL 4.5, lavapipe Vulkan):

- the GPU test set (`test_glsl_backend`, `test_glsl_fused_network`,
  `test_glsl_jit_backend`, BLAS, shader tests and others) fails the same 28
  names and errors the same 4 on this branch and on main; the operator tests
  that compare against numpy pass on both;
- the published GLSL compute product compiles with `glslangValidator` for
  OpenGL and with `-V --target-env vulkan1.1`, and run on Vulkan from its
  record alone (resources, control derivations, byte offsets, group rule) it
  matches numpy bit for bit for float32 and int32 add over 777 elements, with
  arena words past the output left untouched;
- the BLAS shell compiles for x86_64-windows-gnu with `-Wall -Werror`. It
  has not been run here; it needs Windows.

## Instances (follow-up, rank-known tensor lowering thread)

Program: a deployable entry over `(a + b) * s` and `a - b`, with `a` and `b`
of shape (2, 3) and `s` a scalar. Wrong record: emission refused both the
shape and the second result, so array-shaped products could not be
deployed.

Rule: invocation `i` runs one instance of the function. A value of static
shape `S` holds `element_count` instances of `prod(S)` elements each, row
major, back to back; a scalar's instance is one element. Each output is its
own buffer, in order. The library function returns a struct when there are
several outputs, and `main` extracts each member. Each `realizes` view
states `instance_shape` (`[]` for a scalar), `rank` = 1 + its length,
`shape` null (the leading extent is the instance count), and `byte_extent`
scaled by `element_bytes * prod(instance_shape)`. A runtime-sized span is
refused by name in both the library and the deployable product. Before
this, the library read one as a scalar.

Evidence: `test_deployable_spirv_states_instance_shapes_and_every_output`,
and `test_instanced_deployable_spirv_matches_python_on_vulkan`. On lavapipe,
over 300 instances, both outputs match numpy and the words past the
instance count are untouched.

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
an output). Mixed ranks, a reduction, or a scalar output beside span outputs
are refused by name.

Evidence: `test_runtime_sized_elementwise_deploys_one_element_per_invocation`,
`test_runtime_sized_deployable_spirv_matches_python_on_vulkan` (lavapipe, a
(17, 13) feed the shader never saw, matches numpy; the words past the count
are untouched), and
`tests/test_runtime_sized_spans.py::test_rank_only_region_deploys_as_elementwise_spirv`
(the sanctioned entry's region deploys and passes spirv-val).
