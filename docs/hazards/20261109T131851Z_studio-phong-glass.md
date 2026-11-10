# Studio Phong dielectric preview alignment

User request: "can you tweak the phong to be more like the ray tracing"

Implemented as opt-in PLUCK_PREMULTIPLIED_GLASS plus PLUCK_TRANSMISSION in the
existing Spectral base_material.frag.glsl and Studio host. Existing other
shader users do not enable the new mode. Ray shaders/material tables are not
modified by this follow-up.

The preview keeps the Schlick Fresnel surface signal when opacity is zero.
Opacity adds the ray branch's forced-reflection mixture rather than deleting
all surface lighting. Fully transmissive dielectrics no longer receive opaque
diffuse illumination. Reflected and direct-transmitted radiance are combined
before display conversion and composited with ONE, ONE_MINUS_SRC_ALPHA, so
coverage does not multiply Fresnel a second time. Non-transmissive surfaces
keep their existing lighting; opaque output pixels were bit-identical in the
real shader comparison.

Studio separates opaque object faces from transparent ones. Transmissive dye
and the shell now share a back-to-front centroid-sorted indexed draw with
depth writes disabled. Sorting updates only for changed camera/object depth
axes; geometry and saved recipes do not change. Default 72,960 transparent
triangles sorted in about 16.7 ms on the current Windows host (10 changes).
This is approximate triangle sorting, not order-independent transparency.

Limits: RGB Phong, no ray-bent refraction, no per-channel framebuffer absorption,
no spectral/nested-medium correctness claim. The unchanged direct transmission
lobe uses the existing authored albedo/thickness model. Arbitrary background
transmission remains straight-through with scalar coverage. Intersecting
triangles and multiple objects can still have approximate compositing order.

Verification on the RTX 3060:
- Actual native fragment programs compiled and rendered to a hidden 1x1 RGBA32F
  framebuffer, without opening/restarting the live app or touching an exposure.
- Normal-incidence clear glass: old 0.200000 background-only; new 0.395528,
  expected 0.395528. Grazing cosine .2: new 0.864081, expected 0.864081.
- Opacity .4 forced-reflection mixture matches the numerical expectation.
- Blue direct transmitted light remains coloured at opacity zero. IOR 1 gives
  no reflection and preserves the background. Opaque pixels match bit-for-bit.
- tests/test_base_material_webgl_compilation.py and test_studio_phong_glass.py:
  3 passed. The real GPU probe is tests/studio_phong_glass_probe.py.
- Compiled legacy, transmission, and Studio WebGL shadow variants through
  glslangValidator: all passed. Project IR export passed.
- Node transparency ordering/partition and axial/default/saved-design checks
  passed. Studio syntax and scoped diff checks passed.
- No whole-application browser image or many-particle performance validation.

Publication handoff (no live export or restart performed by this worker):
- Copy current marble_studio.js/html and new marble_glass_preview.js alongside
  the existing modules. web_export.py's copy list includes the new module.
- Use repo-cache/phong-preview-20261109/base_material.webgl.shadow.frag.glsl,
  SHA256 33D83F06F25A49CA1C5ECA45EF222D9D1ADA9FFBED3465BD290C4D5272CC320A.
- Generated from the canonical Spectral shader using the existing
  examples.meowdoku.web_export.export_shadow_fragment(Path(...)) entry.
- No default generated Spectral shaders or live build files were overwritten.
