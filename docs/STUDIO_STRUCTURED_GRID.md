# Studio structured grids and graph-paper reference

The existing gobo and backdrop builders accept optional structured data. The
Studio page, detached agent product and MRP1 photo path consume the same faces,
material names and ownership. There is no new renderer, light or material table.

## Perforated cell grid

`gobo.grid` or `backdropLayer.grid` is:

```json
{"version":1,"rows":2,"columns":3,"gap":0.05,
 "cells":["white_plaster",null,"","black_felt","white_plaster",null]}
```

Rows run top to bottom (+Y to -Y), columns left to right, looking at the +Z
front. `null` omits geometry, `""` inherits the parent material, and a named
cell uses that existing material. Gobo inheritance preserves the independent
front/back materials. A named gobo cell uses its material on both sides/rim.

Each axis is 1..64, with at most 1024 cells. Gap is 0.01..0.9 of cell pitch,
so closed gobo cells never share coincident surfaces. Cells are real closed
gobo solids or planar backdrop sheets, not alpha cutouts. Backdrop sweep uses
the same rows/columns over its existing floor footprint. Ellipses are refused.
Legacy profiles without `grid` keep their prior geometry and serialized shape.

The Advanced editors expose grid enable/disable, dimensions, gap and explicit
JSON cell data. Resize preserves overlapping coordinates. Bad data retains the
last accepted grid. The default 16x16 gobo fits the existing full-scene MRP1
2 MiB metadata limit; denser scenes remain subject to that unchanged limit.

## Reflective graph-paper card

`studioGraphPaperArtwork()` generates real rectangular surface markings and
the explicit reference metadata. `studioGraphPaperGobo({unitMm,center,yaw,id})`
places them on the incumbent closed rectangular flag. The UI's **Add graph-paper
reference** uses the current physical Studio unit and positions it behind the
marble for the usual +Z-side Studio view. Rotate it for another camera; the
built-in ray photo camera looks the opposite direction.

The reference is 64x64 mm at creation. Its central squares are 2 mm, with major
lines every 8 mm and a segmented 2 mm scale at the bottom. Resizing changes
those physical dimensions; the QR payloads are static labels, not a measuring
instrument that updates itself. The flag retains existing 0.12 Studio-unit
thickness. Markings sit 0.001 units in front of its surface as real geometry.

The card is reflective. Its paper material is `white_plaster`; its dark marks
use `black_matte_paint`. Swatches reuse `heart`, `marble_cat_eye_green`,
`marble_cat_eye_blue`, `marble_cat_eye_amber`, `metal_copper`,
`metal_oxidized_copper`, `black_matte_paint`, and `white_plaster`. These are
existing material swatches, not a certified color chart. Every chosen source
PBR row has zero emission. The similarly named `calibration_*` catalog entries
are emitters and are deliberately not used by this reflective preset.

Both QR codes use the existing `tipjar.js` encoder and `qr_wall.js` four-module
quiet zone. They encode only `STUDIO;2mm;L` and `STUDIO;2mm;R`. No account or
personal data is encoded. Their physical module geometry is real; a small,
blurred, poorly lit or heavily downsampled output may still be unreadable.

## Structured artwork API

`gobo.artwork` and `backdropLayer.artwork` contain `version:1` and up to 2048
`rectangles`. Each rectangle has a unique string `id`, canonical `material`,
and finite `x`, `y`, `width`, `height`. Coordinates are normalized to the parent
panel, centered at zero, with positive dimensions entirely inside [-0.5,0.5].
The caller owns region overlap and source meaning. The graph-paper preset uses
disjoint markings. Grid and artwork cannot be combined on the same parent;
artwork requires a solid rectangle. Backdrop artwork marks the wall sheet;
the existing optional floor sweep remains its parent material.

An external data producer can supply either record directly through the normal
resolved lighting profile. No dataset is inferred or fetched. Existing Living
Data mesh import/composition remains unchanged; this does not claim a new live
Living Data source binding.

The reflective card is not a projected light pattern. For an actual transmissive
or perforated gobo, use the explicit cell materials/holes and existing ray lights.

## Verification

`node tests/meowdoku_studio_grid_check.mjs` checks bounds, cells/holes, normals,
closed solids, material sides, transforms, backdrop/sweep ownership, atomic UI,
cloned persistence, graph-paper scale, non-emissive materials, and actual
settings-to-MRP1 packing. `tests/test_meowdoku_studio_grid.py` invokes that test
and decodes its real MRP1 with the existing server decoder. No GPU is needed.

The actual requested acceptance is a 256x256 output from one 1024x1024 sensor
tile (4x oversampling), with the default saved marble/room and a new card.
Receipts and image artifacts are under `build/dot-graph-paper-20261111`.
The first test used the wrong emissive calibration materials; it is retained
as a failed-material test, not reflective-card acceptance.
The two corrected non-emissive tests remain too dark for the grid and QR
details to be readable, including a matched camera/card reversal. Their
default rig has all directed lights disabled and only the overhead shop lights
enabled. Geometry/material tests pass, but readable photographic acceptance
and independent QR decoding remain unverified. No light or display-gain
workaround has been applied to claim success.
