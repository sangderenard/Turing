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

- **Desktop GLSL compute** (`glsl_component_backend`). One resource: the
  std430 `Arena` of `uint` words at set 0, binding 0. Every port is a view
  at `4 * u_slot[slot]` bytes, extent `4 * u_extent[0]` unless a static shape
  was given. Controls: `u_count` and `u_extent[0]` are the element count,
  `u_slot[i]` the arena word offset of port `i` (`_bind_arena`,
  `_dispatch_many`). Group count: any grid covering the element count
  (`turing_linear_gid` folds x/y/z; the `u_extent[0]` guard stops the rest).
  The logical `backend_binding` numbers are not descriptor bindings here.
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

- `vulkan-default-uniform-controls` (GLSL compute): `u_count`, `u_extent`
  and `u_slot` are default-block uniforms. `glslangValidator` accepts the
  source for OpenGL and refuses it with `-V`: "non-opaque uniforms outside a
  block: not allowed when using GLSL for Vulkan". Making the arena
  Vulkan-ready needs the controls in a uniform block or push constants,
  which changes the GL runtime's binding code; it is not done here.
- `graphics-stage` (WebGL): a raster pipeline with the companion vertex
  stage, not a compute dispatch.
- `no-component-abi` (extracted shaders).
- Deployable SPIR-V refuses by name: an array-shaped value, a dtype with no
  storage element (bool), or more than one result.

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

- A Vulkan-ready spelling of the GLSL arena controls.
- A component ABI for extracted host shaders.
- The precision-core compute lane (`ssa_glsl_compute_backend`) has no
  component ABI, so it has no product yet.
- Deployable SPIR-V for array-shaped SSA values and multi-result functions.
