# Living Data geometry export

The static native host exposes `livingDataStaticHost.meshSnapshot()` after
`await livingDataStaticHost.ready`. It copies the actual renderer's interleaved
Float32 triangle stream, native material resources, object/semantic spans,
explicit route/portal overlay ranges and the complete world registry.
It never rebuilds geometry from layout boxes. Separate dynamic/solver draw
buffers are refused rather than silently omitted. Changes to the captured
static source or realized mesh latch a refusal until a new host is loaded.

Export a captured snapshot:

```powershell
python tools/export_living_data_obj.py build/dot-storybook-static-phong-20261110/realized-mesh.json build/dot-storybook-obj-20261111 --stem storybook-living-data
```

The output contains:

- `.obj`: all original triangles, world-space positions and normals, object
  identities as `o`, semantic identities as `g`, face material assignments.
- `.mtl`: native albedo, ambient/specular/shininess, emission, IOR and opacity;
  optional `Pr`/`Pm` extensions carry roughness and metallic values.
- `.manifest.json`: the complete lossless input snapshot, original names,
  authored transforms, graph and semantic records, material arrays, face
  material mapping, reversible exported-name mapping and file hashes.

Keep OBJ and MTL in the same directory. OBJ names use reversible UTF-8 percent
encoding, so spaces, Unicode and line breaks cannot inject OBJ directives.
Object/overlay ranges must partition every triangle. Semantic ranges must be
inside the matching registry object. Unknown owners and overlay kinds fail
closed. The source hash is a host-supplied provenance receipt, not independent
attestation of the original source.

## Format limits

OBJ preserves this static geometry, not executable Living Data behavior.
World transforms are baked into the vertices and normals, without recentering,
axis conversion or scale changes. OBJ does not declare units. Full original
transforms remain in the manifest.

MTL is a portable approximation, not the native shader. Partial transmission is
only indicated qualitatively by `illum 7`; opacity does not reproduce the native
optical interpretation. PBR extensions depend on the importing application.
Enamel, texture-stack and thermal semantics, original vertex colours and all
native material arrays remain in the manifest. No UVs, lights or cameras are
invented. Graph relations and physics state remain data, not executable OBJ
features.

## Measured Storybook export

Actual native StorybookEnvironment capture: 33 registry objects, 33 object
spans, 199 semantic spans, 7,164 vertices, 2,388 triangles, 16 material rows,
zero overlay triangles at this static state, zero browser Workers.

`tests/living_data_mesh_browser_check.py` captures the genuine native browser
mesh and verifies repeated snapshot stability and permanent refusal after a
mesh mutation. `tests/test_living_data_obj.py` verifies exact OBJ position/normal
and identity round trips, material selection, unsupported/provenance refusal,
and parsing with the existing Spectral OBJ importer. That importer deliberately
normalizes geometry to a unit radius; its compatibility test verifies that
documented normalization, not preservation of world coordinates inside Studio.
