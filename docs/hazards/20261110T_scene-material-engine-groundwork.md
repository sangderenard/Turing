# Scene material engine groundwork, 2026-11-10

User requests three independently selected engines: subject, product stage and lighting, remaining room. All share material authority. No lighting or authored material values should change implicitly.

Implemented, not released:
- `scene_material_engines.py` validates two non-subject groups, complete triangle coverage, closed outward volume ranges, conservative non-overlapping bounds, explicit reference-thickness spectra. Unsupported transmissive faces reject instead of falling back.
- Photo decode retains validated `fixtures.material_engines`.
- `studio_scene_engines.py` projects each explicit component to its own native material identity, preserving source PBR and optical samples. Adapter opacity/transmission flags select the existing Fresnel/Beer interface, not new material values.
- `studio_photo_geometry.py` passes boundary and spectral records for fixtures through the existing shared append pipeline.
- Standalone positive-boundary region handling from interrupted worker recovered in `studio_region_gpu.py`; CPU validation covers distinct external media alongside subject host.

Verification: three Turing contract/projection tests passed. Eighteen Spectral region tests passed. GPU and full-scene image not run. No source assets published or server restart.

Remaining release gates: actual three-selector UI/capture/profile persistence, body metadata for every transmissive scene object, full decode/replay integration, combined GPU forward/camera image gate including origins inside table and touching slab/legs, independent review. Current explicit admission only handles disjoint bounds; overlapping/touching transitions need careful verification. Canonical document field projection and native parametric lens attachments remain separate unfinished integrations. Artistic milk glass is not calibrated volumetric scattering and no replacement coefficients were invented.

Continuation verification:
- Full forward/camera region shaders linked on RTX3060.
-36existing region GPU cases passed across1m/25mm/1mm and thin-bubble scales.
-4new separate subject/room GPU cases passed (entry,exit,origin medium,Beer attenuation), tests/studio_scene_volume_gpu_check.py.
-69Spectral CPU checks passed,1explicit GPU pytest skipped (separate real GPU gate ran above).
-5Turing admission/projection checks passed.
-Actual retained scene config141b4777... CPU-packed with an in-memory illustrative spectrum only:5table solids plus3subject regions. No config or image edits. Mixed opaque/optical scene exposed KeyError in scene_builder hashing; fixed partial-spectrum lookup and added regression.
-UI integration delegated to wire_scene_material_controls. No publication/restart yet. Full rendered image comparison remains unrun.
