# Grid reflector material separation

User intent: the active upper and underlight grids should form polished silver mirror tunnels, not matte baffles. Interior cell-facing walls need a selectable material separate from front/back caps, perimeter exterior faces and sealed apron. Polished gold should also be selectable as a warmer photographic reflector.

Implementation in progress: the shared closed-box triangles retain their positions and winding. Material IDs differ only on the two longitudinal cell-facing sides of interior bars, or the single inward side of perimeter bars. Caps and exterior/apron remain the separately selected outer material. All IDs resolve through the shared catalogue and the same preview/photo scene packer; no extra emission or RGB tint shortcut is introduced. Existing records without material leaves keep housing materials. New default upper and underlight interiors request polished_silver.

Validation: light-tools face-count/orientation assignment and legacy migration tests, full Studio profile/UI tests, gobo shared scene/photo isolation and stand photo bridge tests pass. Tests use explicit catalogue IDs; presets must be appended and exported before live release. Current saved user profiles are not rewritten. Both currently active grids require an explicit UI material selection after publication.

Bounce limitation: marble_live.js forwards photoBounceCount and photoBounceMode from the exposure controls. marble_photo.js validates1–200 and describes the separate1000-bounce Auto pilot; marble_photo.py validates the same1–200 render input. No bounce setting was changed for this material update. Sparse Auto probing is not a guarantee that a long mirror tunnel has converged; existing diagnostics report surviving paths and render-cap limits.

Publication: not yet released at this note creation. This note is a design/verification receipt, not a measured silver/gold optical calibration claim.


Published2026-11-10T03:45:39Z. Five exact HTTPS asset hashes verified,265 unrelated files unchanged, no restart. Receipt: build/dot-grid-reflectors-20261110/publication.json. Shared polished_silver39 and polished_gold40 append after all39 prior rows; both roughness0.005, metallic1, zero grain, no transmission/emission. Colours are artistic estimates. Current importer sends these fields into MaterialDatabase; legacy ray derivative uses reflectance0.85/diffuse0 for metallic1, and generic spectral material derives imaginary IOR as metallic×3. This is real colored reflective transport but not measured conductor Fresnel calibration. No bounce settings changed. Existing currently loaded grid choices require UI application, requested from camera/browser owner.
