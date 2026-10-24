# Nodus shader deployment handoff

## Status

Nodus can now retain and translate a Turing-emitted shader when it receives
the shader source beside the existing `turing-compiled-program-api-v1`
sidecar containing `turing.shader-component.v1` metadata. This is sufficient
for source conversion and SPIR-V validation. It is not yet sufficient for
general runtime dispatch.

This request is packaging and deployment metadata work. It does not ask for a
new shader compiler path, a second logical ABI, or changes to Turing's SSA
lowering.

## Immediate publish contract

Each deployable shader product should publish, in one directory:

- the exact emitted source file;
- its existing compiled-program API sidecar;
- a SHA-256 for each file; and
- one small request manifest naming those two files and the desired derived
  source targets.

Nodus currently accepts this manifest:

```json
{
  "schema": "nodus-turing-shader-artifact-v1",
  "source": "kernel.comp",
  "api": "kernel.api.yaml",
  "targets": ["glsl", "essl", "hlsl", "msl"]
}
```

The paths are relative to the manifest. Nodus reads language, stage,
entrypoint, component identity, logical ports, bindings, and sentinels from
the existing API/component ABI. It retains the source and API byte-for-byte,
compiles GLSL or HLSL to SPIR-V, validates the module, and records hashes,
tool diagnostics, and derived sources. WGSL is explicitly refused until a
WGSL frontend is available or Turing publishes a validated SPIR-V or GLSL
sibling produced from the same component ABI.

The closest existing Turing writer is `WGSLModule.write()`. Desktop GLSL
compute, WebGL/graphics members, and extracted host shaders need the same
source-plus-sidecar publication behavior. `ExtractedShader` provenance
(filename, line, source symbol, compiler call, and source digest) should be
carried into the product receipt rather than discarded.

## Runtime deployment contract still required

Source conversion needs only the existing logical component ABI. A runnable
Nodus row needs the physical launch contract as a derived deployment view of
that same ABI. For each physical resource, publish:

- descriptor set and binding;
- resource kind and read/write access;
- memory layout, dtype, byte offset, and byte extent;
- the canonical logical component port it realizes; and
- any required extension, capability, or specialization constant.

For compute shaders also publish:

- local workgroup size;
- the group-count rule or exact group counts;
- the logical element-count source; and
- every control value and its derivation, including the current GLSL arena
  controls `u_count`, `u_extent`, and `u_slot`.

This data cannot be inferred from logical port order. The present Nodus Vulkan
runner accepts only a compute entrypoint with descriptor set 0 and contiguous
storage-buffer bindings. Uniforms, images, textures, samplers, nonzero sets,
and graphics stages must remain explicit shortfalls until that runner gains
their declared binding behavior.

## SPIR-V products

`ssa_spirv_backend` currently emits a linkage-exported SPIR-V library function.
It deliberately has no `OpEntryPoint`, execution mode, or descriptor/interface
ABI. Preserve that product as a library.

If Turing publishes SPIR-V for deployment, publish a separate product kind
whose module contains the entrypoint, execution mode, interface variables,
physical resource ABI, target environment, and launch contract. A manifest
must distinguish at least:

- `linkable-spirv-library`; and
- `deployable-shader`.

Nodus must not reinterpret the current library module as an executable shader.

## Completion criteria

The handoff is complete when a produced directory can be consumed without
Nodus joining unrelated in-memory objects or guessing declarations from shader
text, and when these facts agree under one component identity:

1. exact source and digest;
2. language, version/profile, stage, and entrypoint;
3. logical component ABI;
4. physical resources and launch controls;
5. target client/environment and intended SPIR-V version;
6. authored and compiler/SSA provenance; and
7. structured completeness and shortfalls.

The logical `turing.shader-component.v1` record remains authoritative. The
deployment record should reference or derive from it, not duplicate it as a
competing ABI.
