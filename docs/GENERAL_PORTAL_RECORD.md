# General portal record

Status: record module and tests landed; no consumer reads it yet. Everything
under OBSERVED was read from the cited lines on 2026-11-10. Everything under
PROPOSED is a suggestion for where a reader would attach, not a change.

Module: `src/compiler/portal_record.py` (stdlib only, schema `portal-record.v1`).
Tests: `tests/test_portal_record.py`.

## 1. One engine, two modes

The Living Data Map already works. A portal there is an edge of a graph whose
two ends are attached to the boundary of bounded things. The record is that
shape, lifted out of the browser template. Nothing is invented beside it.

- **Class mode.** `abstract_ui_introspection` builds a world from a class:
  buildings are classes, rooms are members, tracks join buildings
  (`docs/REPO_WORLD_MAP_DESIGN.md:49-80`). A reference between two bounded
  things is a directed graph edge; `PortalPair` is built from it with
  `from_graph_edge(source_id, target_id, kind=..., provenance=...)`.
- **File mode.** `repo_world_scan.py` builds regions and rooms from a clone;
  a resolved import is a directed edge between two file rooms
  (`repo_world_scan.py:136-146`, `:463-466`). `from_repo_scan_pair(pair)`
  builds the same `PortalPair`.

Both modes need only: nodes (owner identities), containment (which node owns
the boundary), and directed reference edges. The record depends on no graph
library; owners are plain node identity strings. The same record is also what
the browser's placed splats and tube edges (section 2) project onto.

## 2. OBSERVED: what a portal is today

### 2.1 Living Data Map (the source)

- Policy, Python: `PlacementPolicy.portal_contract`
  (`src/compiler/abstract_ui_placement.py:206-237`): `set_size` many-to-many,
  `directionality: in-to-out`, `mapping: local-manifold-frame`, `backing:
  probabilistic-tube-graph`, `distribution: normalized-spatial-gaussian`,
  `path_model: relaxed-quaternion-cubic`, `modes.standard` (aperture_class
  person, scale 1.0) and `modes.mega` (vehicle, 4.0). Recipe `opening_kind=
  "portal"` at `abstract_ui_placement.py:268`; `SUBTRACTIVE_KINDS` at `:14`.
- Minting, browser only: `placePortalSplat`
  (`src/compiler/abstract_ui_div_map.py:5864`) builds the end ("splat") at
  `:5893-5910`. `portalRuntime` (`:3133`) holds splats, the graph (sigma 4.0,
  traversalSpeed 3.2), transits and cooldowns. `rebuildPortalGraph` (`:4325`)
  makes one edge from every `in` splat to every same-aperture-class `out` splat
  with a Gaussian-in-distance weight normalised per source (`:4347-4352`).
  Persistence: `portal_splats` saved at `:4417`, restored at `:4523`.
- Traversal: `traversePortalBody` (`:8192`) enters at `in` splats and exits at
  `out` splats along the chosen edge. `portalOpeningTransform` (`:8097`) maps a
  wall opening's side/offset to a centre and normal.
- Physics law, Python/WASM: `abstract_ui_physics.py:197-224`: `p_to =
  portal_target + R(portal_cos, portal_sin)(p - portal_source)`, `v_to = R v`,
  blended by `portal_active`; parameters at `:123-131`; stated at
  `docs/ABSTRACT_UI_SYMBOLIC_WORLD_PHYSICS.md:50-67`.

| Living-map field (file:line) | Record field | New? |
|---|---|---|
| splat `identity` (`div_map:5893`) | `PortalEnd.identity` | no |
| `port_role` in/out (`:5893`) | `PortalEnd.role` | no |
| `object_identity`, `part_identity` (`:5893-5910`) | `PortalEnd.owner` (the object; part is not carried) | no |
| `center`, `normal`, `tangent`, `bitangent`, `radius` (`:5893-5910`) | `PortalPlacement` same names | no |
| (wall `side` + `offset`, `:8097-8105`) | `PortalPlacement.face` (+ `extent`) | `face`, `extent` are new names |
| `aperture_class`, `aperture_scale` (`:5893-5910`) | `PortalEnd.aperture_class/_scale` | no |
| edge `identity` (`:4347`) | `PortalPair.identity` | no |
| edge `weight` (`:4349`) | `PortalPair.weight` | no |
| edge `path_model` (`:4351`) | `PortalPair.path_model` | no |
| `portal_source_*`, `portal_target_*`, `portal_cos/sin` (`physics:123-131`) | `PortalFrameMap` (as a 3x3 rotation; `from_yaw` takes cos/sin) | record is new, law is not |
| `source`/`target` of the edge, split by `in`/`out` | `PortalPair.source/target` | no |
| (none) | `PortalEnd.pair` | **new** |
| (none) | `PortalProvenance` kind/specifier/source_line/origin | **new** (scan) |
| (none) | `PortalTransmission` | **new** (declared slot) |
| `control_points`, `distance`, `length`, tube radius, tangent frames | not carried | realisation, stays in the page |

### 2.2 Python records near it

- `RoomOpening` (`abstract_ui_house_builder.py:40-60`, `:278`): door/gate/window
  only, `connects_to` + `seal_model`, emits `portal-adjacency` between air
  volumes for the atmosphere engine. Not a portal; it is the sealed-wall case.
- `pluck_placed_object` kind portal (`abstract_ui_world.py:376-379`): one-ended
  (`target_region`), import adapter only.
- `document_geometry` mints south-wall doors/gates only
  (`REPO_WORLD_MAP_DESIGN.md:49-57`).
- Meowdoku: `examples/meowdoku/marble_world_house.py:29` records that the
  captured front wall has "no admitted doorway or portal" as pending.

### 2.3 Scan side

`RepoPortalEnd` (`repo_world_scan.py:105-132`: identity, room, role, pair,
`transmission: Mapping | None`) and `RepoPortalPair` (`:136-157`) are produced
at `:463-466`. `transmission` is never written by the scan. Design:
`docs/REPO_WORLD_MAP_DESIGN.md:156-175`, gap C at `:204-208`.

### 2.4 What the particle sim and the optical stack consume as surfaces today

- Particle sim: `examples/meowdoku/fountain.py` (grid, walls and contacts in
  its module docstring and `:58`, `:802`); no portal or pair concept, no
  state transfer across a surface.
- Optics, meowdoku: per-material `transmission`, `ior`, `opacity`
  (`examples/meowdoku/material_catalog.py:11-16`, glass at `:54-58`); the
  ray-shader treats opacity as a surface-reflection mix, not volume
  (`:54-57`). `examples/meowdoku/studio_filter_density.py:1-6,154` is
  absorption only: "no interface Fresnel, refraction, scattering or
  diffraction"; no registration with a ray reader.
- Optics, spectral-analyzer (read-only): `camera_designer/scene_builder.py:
  1836-1884` takes `region_media` and validates region ranges;
  `studio_region_gpu.py:136` (`with_region_media`);
  `optical_material.py:102` Beer-Lambert `transmittance(path_m,
  wavelength_um)`; `scene_builder.py:1330-1360` per-triangle
  `transmittance_bands`; `demo_pluck_gl.py:5503-5506` wires region media into
  the GPU ray-field shaders. Acoustic: `acoustic_amr.py:137-170` has an
  aperture class and a TRANSMISSION border condition for nested simulations.
  None of these reads an end that continues somewhere else.

## 3. The record

`PortalPair(identity, source: PortalEnd, target: PortalEnd, provenance,
direction="source-to-target", frame_map | None, weight | None, path_model |
None)`. `PortalEnd(identity, owner, role, pair, placement | None,
transmission | None, aperture_class, aperture_scale)`. Validated in
`__post_init__`: roles in/out and opposite, distinct owners, a frame map only
once both ends are placed, fractions in [0,1], wavelengths positive and
strictly increasing, rotation proper orthonormal.

`PortalTransmission` is declared, every field optional: `spectral`
((wavelength_m, fraction) samples) or `scalar` (exclusive); `refractive_index`
(n_before, n_after); `pass_probability`; `maps_velocity`; `thermal` and
`acoustic` named and left None.

Builders: `from_graph_edge`, `from_repo_scan_pair` (duck-typed), `ends_by_owner`
(sorted, input-order independent). Round trip: `to_data()` / `from_data()`.
No geometry policy, no backend name appears in the record.

## 4. PROPOSED: which existing function would read it

- World builder fills `placement` and `frame_map`: where `document_geometry`
  (`abstract_ui_viewports.py:286`) mints openings today, one opening per end,
  using the wall rule of `portalOpeningTransform` (`div_map:8097`).
- Browser minting: `placePortalSplat` (`div_map:5864`) would emit a record
  end instead of an ad-hoc dict; `rebuildPortalGraph` (`:4325`) would emit
  `PortalPair.weight`/`path_model`. The tube geometry stays page-local.
- Traversal: `abstract_ui_physics.py:197` takes its anchors and rotation from
  `frame_map`.
- Particle sim: a contact step in `fountain.py` (near `:802`) would read
  `transmission.pass_probability` and `frame_map` for particles reaching an
  end's placement.
- Ray tracer / shaders: `scene_builder.py:1880` (`region_media` intake) and
  `studio_region_gpu.py:136` would accept an end as a region boundary reading
  `transmission`; `optical_material.transmittance` supplies the volume term.

## 5. What an optical ray tracer needs

- **Frame-map continuity.** A ray hitting the `out` end at point p with
  direction d continues from `frame_map(p)` with direction `R d`. Position,
  direction and polarisation basis must all go through the same rigid map,
  so a ray leaving the pair is continuous in the sense of the map. A
  proper rotation preserves handedness, so no mirror case arises.
- **Radiance through the pair.** Radiance is carried unchanged along the
  continued ray, then multiplied by the end's spectral transmittance
  (and Fresnel if the refractive-index pair differs). The two ends'
  indices are a pair, so the interface law at each face is explicit.
- **Recursion bound.** A ray may see another portal through a portal.
  The tracer needs a maximum portal depth per ray and a rule for a
  throughput below a floor; the record does not say what they are.
- **Spectral.** Samples are in metres; the existing stack's
  `transmittance(path_m, wavelength_um)` (`optical_material.py:102`) is in
  micrometres, so a conversion is the consumer's job.

## 6. What the particle simulation needs

- **State transfer.** Position through `frame_map`; velocity through its
  rotation if `maps_velocity` is True; angular velocity likewise, since a
  proper rotation preserves it as an axial vector; scalar fields unchanged.
- **Conservation statement.** The map is rigid, so mass, speed, kinetic
  energy and angular momentum about the particle's own centre are preserved
  by the transfer. With `pass_probability` p, each particle is transferred
  with probability p and otherwise handled by ordinary contact, so expected
  flux across the pair equals p times the incident flux. Momentum in the
  world frame is not conserved across a rotation, and a transfer
  removes a particle from one owner and adds it to another; the
  owner-level ledger is the consumer's to book.

## 7. Explicitly not decided

- **Role naming.** The living map enters at `in` and exits at `out`
  (`div_map:8192`; `abstract_ui_placement.py:206-211`). The scan labels the
  referencing room's end `out` (`repo_world_scan.py:109`). The record keeps
  the scan's labels and requires only that the two roles differ; which end a
  traveller enters is unresolved.
- Whether `part_identity` is part of the owner or a separate field.
- Per-pair vs per-end transmission when the faces differ (this record is per-end).
- Whether `weight` normalises per source end as in the living map or is free.
- Reciprocity (whether the reverse transit is the same pair).
- Thermal and acoustic fields, and the units for them.
- Portal depth limit and throughput floor for optics.
- Mapping the record back into the page (the page keeps its own splat format
  until a consumer is chosen).
- Placement for the scanned pairs: which face, in what order, left to the
  world builder.
