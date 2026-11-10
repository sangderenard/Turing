# Parametric cookie and filter slice

Source-only. No production reader, UI kind, static publication or process restart.

## Implemented and measured

- Spectral `camera_designer/curve_cookie.py` implements the existing ParametricSurface CPU/GLSL predicate using original line, quadratic and cubic font controls from `scene_orders._font_path`. Even-odd fill retains holes and disconnected contours. Cubic derivative roots partition monotone intervals; ray crossings use the polynomial, not an outline polygon or bitmap. Finite float64 CPU / float32 GPU precision is explicit.
- `MEOW` has 54 source curves and 56 monotone intervals. A hidden native OpenGL compute dispatch on RTX 3060 tested 2,048 deterministic rays: 902 blocked, 1,146 open/outside, zero CPU/GPU hit mismatches. This tests the real predicate, not a complete scene render.
- Spectral `curve_filter_probe.py` uses the exact contour region plus existing `validate_optical_media`, `optical_extinction` and `OpticalMaterial.transmittance`. ND 1 transmits 0.1 internally over 1 mm at 400–700 nm. Constant absorption, not constant imaginary IOR, makes it neutral. Thickness and concentration multiply optical depth; interpolated spectral dye follows the shared absorption interpolation. No RGB opacity enters this path.
- The absorption probe accepts normal incidence through an index-matched layer only. It refuses oblique rays. It is not a finite-volume scene registration, interface/refraction solver, or diffraction model. Varying density on an oblique ray requires the actual in-material path integral and side-exit handling.
- `studio_text_assets.py` writes the exact curve cookie and an explicitly tessellated glyph via the existing `phong_tiles.token_mesh` / `scene_orders._glyph_triangles` builder. Actual MEOW output: 566 triangles, approximately 92.759 × 20 × 4 mm. The mesh is not advertised as curve-exact or collision-certified.

## Compiled density gate, not bypassed

`studio_filter_density.py` authors a tensor-product cubic Bernstein relative-concentration lambda and Beer–Lambert outputs. It calls the existing symbolic compiler, then existing SSA C/LLVM/WGSL backends. No alternative compiler or Python runtime fallback was added.

Actual execution currently stops before SSA emission: `post_derived_node_identity` raises `canonical graph node ... has no lexical read scope for its identity row`. Both constant and bicubic inputs fail. The untouched baseline `test_named_sympy_equations_share_one_process_graph_and_ssa_function` fails at the same guard. This is not a successful native density build. `build/studio-filter-density-probe/probe-receipt.json` preserves the exact failed stack and `fallback_used:false`.

The native test command is `python tests/probe_studio_filter_density.py`. It exits nonzero while blocked. Once the original compiler gate is repaired, it compiles the real C product and checks 16 gradient-density/ND samples. LLVM/WGSL emission alone must not be represented as execution.

## Tests and artifacts

- Spectral: `python -m pytest tests/test_curve_cookie.py tests/test_curve_filter_probe.py -q`: 16 passed.
- Spectral: `python tests/curve_cookie_gpu_probe.py`: actual GPU predicate passed.
- Turing: `python -m pytest tests/test_studio_filter_density_authoring.py tests/test_studio_text_assets.py -q`: 5 passed. Authoring tests are not native compiler tests.
- `python -m examples.meowdoku.studio_text_assets MEOW --glyph --output build/studio-curve-cookie-meow` wrote `curve-cookie.json` and `glyph-mesh.npz`.

## Remaining integration

The production mixed-scene native reader must dispatch the exact cookie/filter event before accepting a proxy intersection. A plane or tessellated glyph is not that event. The light-tool registry must not enable `curve_cookie` until this consumer exists. Absorbing open sheets must not be relabeled as finite closed volumes to pass material routing. Keep actual per-wavelength material identity and physical path; no display-mask substitution.

## Existing shutter seam

The original `thick_lens_focus_lab.py` camera loop already advances each CameraTimeline slice via `on_dt_request(slice.dt)`, submits separately weighted flash/natural packets, waits the sensor stage, then drains and joins T5. Its current loop starts continuously; SPACE saves the live accumulation rather than triggers admission. A requested shutter mode should gate this existing frame admission, preserve its scheduler/barrier, and gate ceiling emission at source, not apply a post-render flash.
