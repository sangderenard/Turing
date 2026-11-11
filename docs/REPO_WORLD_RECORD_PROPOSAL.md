# Repository world: feeding the existing Living Data Map engine (proposal)

Status: proposal, read-only analysis. Nothing in `src/` was edited.
Closes gaps A, B, C and D of `docs/REPO_WORLD_MAP_DESIGN.md` section 5.

Marking: **OBSERVED** = read in code (file:line). **PROPOSED** = not in the
tree. Line numbers are at the working tree read on 2026-11-10; the compiler
numbers nothing here, these are source lines only.

Owner framing (received mid-task, applied throughout): class mode and file
mode are two modes of one engine. Portals come from the living map. The
engine's input must be a generic graph. This document therefore does NOT add
a parallel record family as the main route; it names the graph the engine
already consumes, says what a file scan must emit to be a second producer of
it, and lists the places where the engine reads class vocabulary instead of
the graph.

---------------------------------------------------------------------------

## 0. Summary

1. OBSERVED. There is no graph class. The "neutral identity graph" the audit
   names (`docs/LIVING_DATA_MAP_COMPILATION_AND_PRODUCTIVE_WORLD_AUDIT.md:22-23`,
   `:119-137`) is three layered, identity-keyed record sets, each carrying
   containment as a parent reference:
   - (G1) the model dictionary from `world_div_map_model`, nodes with
     `identity`, `kind`, `dependencies: [{relationship, target}]`;
   - (G2) `document_geometry` boxes, nodes with `identity`, `kind`,
     `parent_identity`, `hierarchy_depth`, `openings`;
   - (G3) the `WorldObject` registry, nodes with `identity`, `kind`, `parent`
     (`abstract_ui_world.py:100`, `:212`; "parent-is-authority", `:605`).
2. OBSERVED. G2 and G3 are already generic: `document_world_objects` reads
   boxes only through `identity`, `kind`, `parent_identity`, `center`,
   `half_extent`, `openings` (`abstract_ui_world.py:197-296`). Class
   vocabulary enters only where G1 is turned into G2 (`document_geometry`) and
   where G1 is built from dataclasses (`world_div_map_model`), because both
   walk the fixed nesting `regions -> buildings -> rooms` instead of
   following the `contained-by` dependency that G1 already carries.
3. OBSERVED. The living map's portal today is a placed splat with
   `port_role in|out`, a host `object_identity` and a `part_identity`, plus a
   graph whose edges are derived (all IN x all OUT, same aperture class,
   Gaussian-weighted by distance), not declared pairs
   (`abstract_ui_div_map.py:4325-4364`). A Python-minted opening with
   `kind: "portal"` and `port_role` is already converted to a splat by
   `migrateLegacyPortalOpenings` (`:4460-4490`, called `:4533`). That is the
   existing entry for gap C.
4. PROPOSED. Four small, default-neutral changes: (A) region `parent`;
   (B) region-level `rooms`; (C) an optional `openings` list on a room that
   `document_geometry` copies onto the room box; (D) `build_repository_world`
   composing them. Explicit pair edges (the scan's "A imports B") are not
   expressible by the current derived portal graph; that is open question 1.

---------------------------------------------------------------------------

## 1. The generic graph the class-mode path already passes

### 1.1 Producer: class introspection fills the dataclasses

OBSERVED. Chain: `project_class_to_div_map` (`abstract_ui_div_map.py:11325`)
-> `build_introspective_world` (`abstract_ui_introspection.py:527-635`) ->
`AbstractUIWorld` -> `project_world_to_div_map` (`:11249`) ->
`world_div_map_model` (`:314`).

- Python class -> `_ClassDescription` per class, grouped by module
  (`abstract_ui_introspection.py:566-568`); one `AbstractUIRegion` per module
  (`:594-604`), identity `region:{module}`; one `AbstractUIBuilding` per class
  (`:427-478`); one `AbstractUIRoom` per member (`:446-458`), identity
  `{class_identity}/member:{kind}:{name}` (`:439-441`).
- Relationships between classes become `AbstractUITrack(source, target,
  relationship, label, intention)` (`:606-621`), kept only when both
  endpoints are buildings (`:608`).
- Grid positions come from `_grid_positions(count, aspect)` (`:211-224`).

### 1.2 The serialised graph (G1)

OBSERVED. `world_div_map_model` (`abstract_ui_div_map.py:314-367`) emits:

| node | identity | kind | containment dependency | other keys |
|---|---|---|---|---|
| world | `world.identity` | `world` | none | `regions`, `tracks` (`:352-367`) |
| region | `region.identity` | `region` | `contained-by -> world.identity` (`:346-349`) | `name`, `position`, `buildings` |
| building | `building.identity` | `building` | `contained-by -> region.identity` (`:333-336`) | `module`, `source_kind`, `metaphor`, `position`, `rooms` |
| room | `room.identity` | `room` | `contained-by -> building.identity` (`_room_model`, `:253-278`) | `member_kind`, `metaphor`, `position`, `type_name`, `parameters`, `interaction`, `implied_code` |
| track | none | none | `source`/`target` buildings (`:358-366`) | `relationship`, `label` |

OBSERVED. Containment is carried TWICE: as the nesting of Python lists and as
the `contained-by` dependency. Consumers read the nesting. A search of the
engine files found no reader of `dependencies` for layout: the only reads
are the inspector property list (`abstract_ui_div_map.py:9273-9278`) and
serialisation (`abstract_ui_filesystem.py:121-143`,
`abstract_ui_viewports.py:236-256`, `abstract_ui_world.py:100`).

### 1.3 G2 and G3 (already generic)

- OBSERVED. `document_geometry(model)` (`abstract_ui_viewports.py:286-457`)
  turns G1 into boxes: envelope (`:391-411`), one `courtyard` per region
  (`:355-371`), one `building` per building (`:306-319`), one `room` per room
  (`:328-342`). Every box carries `parent_identity` and `hierarchy_depth`
  (constants 0, 1, 2, 3 at `:359`, `:309`, `:332`, `:394`).
- OBSERVED. `document_world_objects` promotes boxes to `WorldObject`s using
  `parent_identity` (`abstract_ui_world.py:212`) and `openings`
  (`:265-270`, `:288`). It has no class vocabulary.
- OBSERVED. `world_graph_model` assembles the registry and
  `identity_specialization_table` (`abstract_ui_world.py:444-482`,
  `:535-621`); the dense ids are by appearance order, so they depend on the
  box order produced by `document_geometry`.

### 1.4 Does the scan's `to_data()` already match G1?

OBSERVED (`src/compiler/repo_world_scan.py:76-155`, `:190-201`). Partial
match. Matches as-is: `identity`, `kind` (`region`, `room`),
`name`, `dependencies: [{relationship: "contained-by", target}]`. Mismatches,
all of them:

| # | Scan record | Engine G1 | Mismatch |
|---|---|---|---|
| m1 | regions listed flat with `parent` and `depth` | regions each own `buildings`; no region parent | gap A |
| m2 | root region has `dependencies: []` (`:80-83`) | region depends on world (`div_map:346-349`) | root must depend on `repo:<H>`; scan root has no dependency |
| m3 | room `contained-by -> region` (`:100`) | room `contained-by -> building` | gap B |
| m4 | room has `language`, `path`, no `member_kind` | room requires `member_kind` (`abstract_ui_introspection.py:83`) | scan `language` must map to an open `member_kind` |
| m5 | no `position` on region/room | `GridPosition` is a required field of `AbstractUIRoom` and `AbstractUIRegion` (`:85`, `:114`); `document_geometry` reads `room["position"]["column"|"row"]` (`viewports:300-301`, `:321-323`) | the builder must assign positions with the existing `_grid_positions` |
| m6 | no `metaphor`, `intentions`, `interaction`, `implied_code` | present on every room record | builder supplies them from the existing palette and `words` helpers |
| m7 | `portal-pair` (explicit directed pair) with two `portal-end`s | no pair; openings of kind `portal` with `port_role` become splats; edges derived | gap C and open question 1 |
| m8 | identities `repo:<H>/dir:<path>`, `.../file:<path>` | identities are authored strings; any string is accepted | none; `/surface:` or `/opening:` suffixes must not collide (checked by the duplicate-identity raise, `abstract_ui_world.py:209-211`) |
| m9 | `RepoUnresolvedReference` has no identity | no counterpart | stays in the scan; not fed (as `REPO_WORLD_MAP_DESIGN.md` already states) |

Conclusion: the scan's record shape is the same SHAPE as G1 (identity, kind,
dependencies), so it is a second producer of the same graph. It differs from
G1 in three structural places only: region parent, room-in-region, and
explicit portal pairs.

---------------------------------------------------------------------------

## 2. Gap A: region parent

### 2.1 Definition today (OBSERVED)

`abstract_ui_introspection.py:110-116`:

```
@dataclass(frozen=True, slots=True)
class AbstractUIRegion:
    identity: str
    name: str
    position: GridPosition
    buildings: tuple[AbstractUIBuilding, ...]
    intention: UIIntention
```

### 2.2 Constructor call sites (OBSERVED)

`git grep "AbstractUIRegion("` over the whole tree: exactly ONE,
`abstract_ui_introspection.py:594`, positional (five arguments). No test,
example or tool constructs it; tests only `isinstance` it
(`tests/test_abstract_ui_introspection.py:32`, `:154-158`).

### 2.3 Consumers that iterate regions (OBSERVED, `git grep`)

| # | Site | What it does |
|---|---|---|
| 1 | `abstract_ui_introspection.py:136-141` `AbstractUIWorld.buildings()` | flattens `region.buildings` |
| 2 | `abstract_ui_introspection.py:149-167` `objects()` | yields region, then building and room per building; a region-level child or room would be missed |
| 3 | `abstract_ui_div_map.py:322-351` `world_div_map_model` | builds region dicts, all with `contained-by -> world.identity` |
| 4 | `abstract_ui_div_map.py:288-310` `_introspective_performance_labels` | iterates region -> building -> room (model dict) |
| 5 | `abstract_ui_filesystem.py:258-266` `filesystem_graph_for_world` | takes `buildings[0]` as owner; raises `ValueError` if no buildings (`:263-264`) |
| 6 | `abstract_ui_viewports.py:294` `document_geometry` | lays regions left to right |
| 7 | `abstract_ui_div_map.py:11067-11068`, `:10384-10393` `renderRegion` | browser: every entry of `model.regions` rendered at top level; `region.buildings.map(...)` (a missing `buildings` key throws, an empty list is fine) |
| 8 | `abstract_ui_div_map.py:11114-11133` `semanticSourceClosures` | browser: every region gets `parent: model.identity` |
| 9 | `examples/meowdoku/marble_world_house.py:195-204` | takes `introspection.regions[0]`, `.buildings[0]`, `existing_ids` from `original["regions"]` |
| 10 | `abstract_ui_div_map.py:474-485` | first `courtyard` box: spawn point and gate; relies on box ORDER |

`git grep` also hit unrelated `.regions` of other compiler structures
(`virtual_memory`, `binary_structure_graph`, `program_order`, ...); they are
not Living Data Map consumers and are excluded.

### 2.4 Smallest field addition (PROPOSED)

```
parent: str | None = None      # appended LAST, after `intention`
```

Appended last with a default so the single positional constructor at
`:594` is byte-for-byte unchanged (a `slots=True` frozen dataclass accepts a
trailing defaulted field). `None` means "child of the world", which is
exactly today's `contained-by -> world.identity`.

Neutral by construction (PROPOSED, each reads the field and falls through
when it is `None`): `world_div_map_model` writes
`contained-by -> region.parent or world.identity`, so every existing model
dictionary is unchanged. Regions stay a FLAT tuple (child regions are not
nested in the dataclass); the graph carries containment in the dependency,
as owner direction requires.

Required companion (PROPOSED): `objects()` already yields all regions of the
flat tuple, so it needs no change for A. `renderRegion` needs none either
(each region is a card). Browser item 8 (`:11116`) must read the region's
`contained-by` target instead of `model.identity` to report the right parent.

### 2.5 What `document_geometry` must do with a child region

OBSERVED existing rule (`abstract_ui_viewports.py:286-374`). Constants
`room_pitch = 3.2`, `building_clearance = 3.5` (`:289-290`). Per building,
`columns` and `rows` from room grid positions (`:300-301`),

- `width = columns*3.2 + 1.4`, `depth = rows*3.2 + 1.4` (`:302-303`);
- the building is placed at `cursor_x + width/2` (`:304`);
- after it, `cursor_x += width + 3.5 + max(width, depth)^0.72` (`:345`);
- `region_depth = max(4.0, depth + 3.5*2)` (`:346`);
- after the last building:
  `occupied_width = max(3.0, cursor_x - region_start - 3.5)` (`:347`),
  `courtyard_margin = 4.0 + sqrt(occupied_width*region_depth)*0.45` (`:348`),
  `region_width = occupied_width + 2*margin`,
  `region_total_depth = region_depth + 2*margin` (`:349-350`);
- courtyard centre `[region_start + occupied_width/2,
  total_depth/2 - margin/4]` (`:351-354`);
- next region starts at
  `region_start + region_width + 6.0 + region_width^0.78` (`:373`).

So today regions sit side by side on the x axis and a region is a courtyard
of blocks laid in a row.

PROPOSED one new rule: a child region is a BLOCK of its parent, exactly as a
building is a block of a region. Process the child with the same function,
starting at the parent's `cursor_x`; its result is `W_c = region_width`,
`D_c = region_total_depth`. Then, in the parent:

- `cursor_x += W_c + 3.5 + max(W_c, D_c)^0.72` (the sibling rule at `:345`,
  not the region rule `:373`; a child is inside the parent);
- `region_depth = max(region_depth, D_c + 3.5*2)` (the rule at `:346`);
- the child box is emitted AFTER its parent box (OBSERVED dependency: first
  `courtyard` is read as the spawn courtyard at `div_map:474-476`), with
  `parent_identity = parent region identity` and
  `hierarchy_depth = parent.hierarchy_depth + 1` (today a constant; unchanged
  for parentless regions);
- `contents` of the parent become child courtyards and their contents, then
  own buildings or rooms. The world `relationships` list (`:424-427`) gains
  `courtyard-contains-courtyard` ONLY when some region has a parent, so the
  current test (`tests/test_abstract_ui_viewports.py:63-67`, exact list
  equality) still holds.

Worked arithmetic (PROPOSED rule; existing constants):
a leaf region holding 3 rooms directly (gap B, below) uses the building block
rule, `_grid_positions(3)` = 3 columns, 1 row (`ceil(sqrt(3*1.35))=3`):
`width = 3*3.2 + 1.4 = 11.0`, `depth = 1*3.2 + 1.4 = 4.6`.
`cursor_x` after the block `= 11.0 + 3.5 + 11.0^0.72 = 14.5 + 5.62 = 20.12`;
`occupied_width = 20.12 - 3.5 = 16.62`;
`region_depth = max(4.0, 4.6 + 7.0) = 11.6`;
`margin = 4 + 0.45*sqrt(16.62*11.6) = 4 + 0.45*13.89 = 10.25`;
`W_c = 16.62 + 20.50 = 37.12`, `D_c = 11.6 + 20.50 = 32.10`.
A parent with only this child (no own rooms): `cursor_x += 37.12 + 3.5 +
37.12^0.72 = 40.62 + 13.5 = 54.1`; `occupied_width = 54.1 - 3.5 = 50.6`;
`region_depth = max(4.0, 32.10 + 7.0) = 39.1`;
`margin = 4 + 0.45*sqrt(50.6*39.1) = 4 + 20.0 = 24.0`;
parent `W = 50.6 + 48.0 = 98.6`, `D = 39.1 + 48.0 = 87.1`.
(Powers computed by hand to two decimals; the implementer must verify with
the repo's own arithmetic, not these figures.) Nothing about the child rule
is a new formula; it reuses `:345`, `:346`, `:347-350`.

OBSERVED limit carried forward: spacing grows super-linearly
(`REPO_WORLD_MAP_DESIGN.md` gap E: `^0.78`, `^1.18`); with nesting the
growth compounds per level. Not solved here.

---------------------------------------------------------------------------

## 3. Gap B: room directly in a region, open member kind

### 3.1 Definition today (OBSERVED)

`abstract_ui_introspection.py:79-89`: `AbstractUIRoom(identity, name,
member_kind, metaphor, position, type_name=None, parameters=(),
intentions=(), implied_code=())`. `member_kind` is a plain `str`, not an
enum. Rooms exist only as `AbstractUIBuilding.rooms` (`:100`).

### 3.2 Call sites (OBSERVED)

`git grep "AbstractUIRoom("`: ONE, `abstract_ui_introspection.py:446-458`
(inside `_building_from_description`, positional). Rooms are read, never
constructed, in tests (`tests/test_abstract_ui_introspection.py:108`,
`:158`). `git grep "AbstractUIBuilding("`: one, `:469`. Total call sites
found: regions 1, rooms 1 (plus building 1, track 1).

### 3.3 Consumers keyed on `member_kind`, with observed behaviour for an unknown kind

| Site | Observed behaviour for an unknown kind such as `"file"` |
|---|---|
| `abstract_ui_viewports.py:324-327` room height | `{"method":1.35,"property":1.05,"field":0.82,"nested-class":1.7}.get(kind, 0.95)`: unknown gets the default 0.95. Works unchanged. Note the key is `"nested-class"` but introspection emits `"nested_class"` (`introspection:392`); observed mismatch, not ours to fix |
| `abstract_ui_viewports.py:331` | copies `member_kind` into the box verbatim |
| `abstract_ui_div_map.py:258` `_room_model` | copies verbatim |
| `abstract_ui_div_map.py:291` `_introspective_performance_labels` | `if room.get("member_kind") != "method": continue`: unknown is skipped silently. File rooms produce no performance observations; correct for files. It also only visits rooms inside buildings (`:289-290`), so region-level rooms are invisible to it today |
| `abstract_ui_introspection.py:387-401` `_member_intentions` | `method`/`constructor` -> invoke; `nested_class` -> portal/navigate; ELSE inspect. Unknown falls to inspect. Class mode only; a files producer builds its own intentions |
| `abstract_ui_introspection.py:442` | `room_palette.get(kind) or room_palette.get("field")`: unknown kind falls back to "field" metaphors. Class mode only |
| browser `:10353`, `:11128`, inspector `:9273` | label text; `member_kind || "member"` |

Conclusion (OBSERVED): the engine already tolerates an open `member_kind`;
gap B is a CONTAINMENT gap (a room needs a home that is a region), not a
vocabulary gap. No consumer needs a rule for `"file"`; the existing default
(0.95 height, no performance label, inspect) is the behaviour.

### 3.4 Region-level room consumers (OBSERVED, every place that walks buildings to reach rooms)

`introspection:136-141`, `:149-167`; `div_map:324-340`, `:288-290`;
`viewports:298-343`; `filesystem:258-266`; browser `:10371-10381`,
`:11119-11133`; `marble_world_house:195-204`.

### 3.5 Smallest change (PROPOSED)

```
rooms: tuple[AbstractUIRoom, ...] = ()    # on AbstractUIRegion, after `parent`
```

Default `()` leaves the call at `:594` unchanged. Then, each a few lines,
each neutral for an empty `rooms`:

1. `AbstractUIWorld.objects()` yields `*region.rooms` after the region and
   before its buildings (containment order, `:150`).
2. `world_div_map_model` emits `"rooms": [_room_model(r, region.identity) ...]`
   on the region dict. `_room_model`'s second parameter is only the
   `contained-by` target (`:263-266`), so it already serves a region
   container unchanged. Region dicts ALWAYS carry `"buildings"` (browser
   `:10388-10390` requires it) and now also `"rooms"`.
3. `document_geometry` lays a region's own rooms as ONE block using the
   building arithmetic of section 2.5 but emits NO building box; room boxes
   get `parent_identity = region identity`, `hierarchy_depth = region depth + 1`
   (instead of the constant 3). The block precedes buildings and child
   regions.
4. `_introspective_performance_labels` need not change (files are not
   methods); the observed skip is the rule.
5. `filesystem_graph_for_world` currently raises with no buildings
   (`:263-264`): see open question 4; it needs a defined owner for a
   buildingless world.
6. Browser `renderRegion` (`:10384-10393`) must also render `region.rooms`
   (it renders only buildings today); `semanticSourceClosures`
   (`:11112-11133`) must also close region rooms. Both are template edits to
   a 6,000+ line handwritten JS string; they cannot be avoided if rooms are
   to appear in the 2-D map pane, but the 3-D world is built from G2/G3 and
   needs neither.

`member_kind` for a file: PROPOSED value `"file"` (open kind); the scan's
`language` carried as `type_name` (an existing optional room field,
`:86`, serialised at `div_map:261`), not as a new field.

---------------------------------------------------------------------------

## 4. Gap C: portals and openings

### 4.1 Where openings and portals are minted today (OBSERVED)

Python:
- `document_geometry` mints exactly three kinds of opening, all by
  geometry: building `opening:entry`, kind `door`, south, offset 0, 0.72 x
  0.72 (`viewports:314-318`); room `opening:door`, kind `door`, south,
  0.42 x min(0.58, 0.78 h) (`:337-341`); courtyard `opening:gate`, kind
  `gate`, 1.1 x 0.82 (`:365-369`). None is a portal.
- `abstract_ui_placement.py:14` lists `"portal"` among `SUBTRACTIVE_KINDS`;
  `:206-225` publishes `portal_contract` (many-to-many, in-to-out,
  `backing: probabilistic-tube-graph`, `distribution:
  normalized-spatial-gaussian`, `path_model: relaxed-quaternion-cubic`,
  per-tool-mode apertures); `:268-269` the "Portal" recipe with
  `opening_kind="portal"`. This is the PLAYER's placement policy.
- `AbstractUITrack` (`introspection:119-125`) is building-to-building,
  lineless, dropped when an endpoint is missing (`:608`). It is not a portal
  and is not drawn as one (OBSERVED: only serialised at `div_map:358-366`).

Browser:
- `placePortalSplat(role)` (`div_map:5864-5928`) mints splats by raycast onto
  a mesh triangle, identity `{graph}/nodes/{role}-{n}` (`:5892`).
- `migrateLegacyPortalOpenings` (`:4460-4490`, called at `:4533`) converts
  any host-box opening with `kind === "portal"` into a splat, using
  `opening.port_role` (or alternating in/out if absent, `:4468`), host
  `object_identity = host.identity` and `part_identity = opening.identity`
  (`:4476`), position from `portalOpeningTransform` (`:8097-8120`) which
  needs `opening.side`, `offset`, `width`, `height` (and optional `bottom`).
- Wall rendering draws a portal-coloured surface for `kind === "portal"`
  openings by `port_role` (`:4085-4097`).
- `rebuildPortalGraph` (`:4325-4364`): edges are every IN splat to every OUT
  splat of the same aperture class, weight `exp(-d^2/(2 sigma^2))` normalised
  per source, `sigma = 4.0` (`:3137`). Edge identity
  `{graph}/edges/{srcTail}-to-{dstTail}` (`:4347`).

OBSERVED fact that matters: the world registry already carries Python-side
openings through to the browser: `document_world_objects` copies
`box["openings"]` into the object form (`abstract_ui_world.py:288`) and as
semantic parts (`:265-270`). So a Python-minted portal opening needs no new
transport; it needs to exist in `box["openings"]`.

### 4.2 The existing seam, reused (PROPOSED)

Each scan portal END becomes ONE opening dict on its room's box:

| opening key | value | source |
|---|---|---|
| `identity` | the scan end identity `repo:<H>/portal:<src>:<line>-><dst>/end:out` or `/end:in` | scan |
| `kind` | `"portal"` (already in `SUBTRACTIVE_KINDS`, placement `:14`) | fixed |
| `port_role` | `"out"` on the importing room, `"in"` on the imported room | scan end `role` |
| `side`, `offset`, `width`, `height` | wall realisation; the builder chooses, never the scan | realisation |
| `pair` | the scan pair identity | scan end `pair` (new optional key) |
| `reference` | `{source_line, reference_kind, specifier}` provenance | scan pair |
| `transmission` | `None` (EMPTY declared slot; never written by the scanner nor the builder) | scan end `transmission` |

Identity rule: the opening identity IS the scan end identity (stable for a
commit; globally unique because the room identity embeds the path and the
pair identity embeds source, line and destination). It must not equal any
`/opening:door` etc.; the duplicate-identity raise guards it
(`abstract_ui_world.py:209-211`).

The two ends and direction: the pair is a FACT with two faces: `out` on the
source room, `in` on the target room, direction source-to-target. The existing
splat vocabulary already has exactly two roles `in|out` and "directionality:
in-to-out" (`placement:208-210`). Mapping: portal-end -> splat
(`port_role` = end role, `object_identity` = room, `part_identity` = opening
identity).

Which existing function receives it: `document_geometry` (the ONLY place
that builds room boxes; `viewports:337-341`). PROPOSED: `AbstractUIRoom`
gains `openings: tuple[Mapping[str, Any], ...] = ()` (appended last; the
single call at `introspection:446` is unchanged); `_room_model` serialises
it as `"openings"` (omitted or `[]` when empty); `document_geometry` appends
the room's openings AFTER the existing `opening:door` (the engine documents
`opening_order: "document-order"`, `viewports:454`), which leaves the first
opening, and therefore every existing consumer of `openings[0]`, unchanged.
The seam for tests is `viewports:337-341`.

`portal_contract` remains a REALISATION POLICY (OBSERVED, never a record
field): apertures, tube graph, Gaussian splats are what the browser does
when it draws an end (`div_map:4319-4322`, `:4325-4364`). Nothing in the
record names a tube, a sigma or an aperture class; the optional
`aperture_class` default is read from the active tool mode by the browser
(`:4318-4323`), and `migrateLegacyPortalOpenings` creates the splat without
one (`:4469-4478`), so `portalApertureClass` returns `"person"` (`:4314`).

### 4.3 What this does not give, stated plainly (open question 1)

OBSERVED: `rebuildPortalGraph` derives IN->OUT edges from distance
and weights; it cannot say "this IN is the end of that OUT". A file scan
yields 15,119 declared pairs on this repository
(`REPO_WORLD_MAP_DESIGN.md:229-233`); the derived graph would instead join
every IN to every OUT. A declared pair has to be honoured by the portal graph
for the portal to mean an import. Smallest PROPOSED change, in the browser
template only: when a splat carries `pair`, `rebuildPortalGraph` links the
`out` and `in` that share `pair` with one edge of weight 1 and skips the
Gaussian normalisation for those splats. Splats without `pair` behave as
today. Not designed further here: it is a template change and needs the
owner's decision (question 1). Until then the opening/end records are
correct and the pair identity survives in the model, but tubes would be
distance-derived.

---------------------------------------------------------------------------

## 5. Gap D: `build_repository_world(scan) -> AbstractUIWorld`

PROPOSED signature (home: next to `build_introspective_world`, or a sibling
module that imports the same dataclasses; either way a second PRODUCER, not a
second format):

```
def build_repository_world(
    scan: RepoWorldScan,
    *,
    seed: str = "abstract-ui",
    room_palette=None, region_palette=None,
) -> AbstractUIWorld
```

Steps, each in terms of sections 2-4 and existing functions only:

1. World identity: `scan.to_data()["identity"]` = `repo:<commit>`
   (`repo_world_scan.py:193`), used as `AbstractUIWorld.identity`. The world
   record is the same dataclass (`introspection:128-134`).
2. Regions: for each `RepoRegion` in scan order (parents precede children;
   scan `depth` is the order key), one `AbstractUIRegion(identity=scan
   identity, name=region.name, position=..., buildings=(), intention=...,
   parent=region.parent, rooms=...)`.
3. Region positions: `_grid_positions(n_children, aspect=1.6)`
   (`introspection:570`) among SIBLINGS, computed per parent (today it is
   computed once over modules).
4. Rooms: each `RepoRoom` becomes `AbstractUIRoom(identity, name,
   member_kind="file", metaphor, position=_grid_positions(...) within its
   region, type_name=language, intentions, implied_code=(), openings=...)`;
   metaphor from `_stable_choice` (`:199-208`, the existing deterministic
   choice); intentions from the existing `words` constructors
   (`words.world_role`, `words.identity`, `words.describe`,
   `words.inspect_`), as `_member_intentions` does (`:380-403`).
5. Portals: for each `RepoPortalPair`, attach `out_end` as an opening on the
   source room and `in_end` as an opening on the target room (section 4.2).
   `AbstractUITrack` is NOT used for imports.
6. Intentions: regions `words.region(...)`, world `words.world(...)`, exactly
   as `introspection:599-632`.
7. Return the dataclass world; the caller then uses the unchanged
   `project_world_to_div_map(world)` (`div_map:11249`), which gives
   `world_div_map_model`, `document_geometry`, `world_graph_model`,
   `identity_specialization_table` for free.

What it must NOT do:
- no building per folder to hold files (stand-in; gap B exists so that a
  room can sit in a region);
- no second world format or serialiser (the output is `AbstractUIWorld`);
- no change to the scan's identities or its refusal rules (clean checkout
  only);
- no `AbstractUITrack` for imports, no tube/aperture/sigma in any record,
  no write to `transmission`;
- no layout arithmetic in the builder (positions are grid indices only;
  coordinates are `document_geometry`'s, section 2.5).

---------------------------------------------------------------------------

## 6. Change table

Existing tests that cover the neighbourhood (from `git grep`):
`tests/test_abstract_ui_introspection.py` (world objects, determinism),
`tests/test_abstract_ui_viewports.py:49-67` (exact box kinds and the exact
`relationships` list), `tests/test_abstract_ui_world.py` (box -> object
promotion, opening parts), `tests/test_abstract_ui_filesystem.py`
(`filesystem_graph_for_world`, `artifact_geometry_boxes`),
`tests/test_abstract_ui_div_map.py` (end-to-end model/geometry; two call
sites of the project functions), `tests/test_abstract_ui_vehicles.py`,
`tests/test_marble_world_house.py:156`, `tests/test_repo_world_scan.py`
(scan only), and the `.mjs` browser checks under `tests/` that read
`model.document_geometry.boxes`.

| change | files touched | covering tests | risk |
|---|---|---|---|
| A1 `AbstractUIRegion.parent=None` | `abstract_ui_introspection.py` | introspection tests | low: one constructor, trailing default |
| A2 model emits `contained-by -> parent or world` | `abstract_ui_div_map.py:346-349` | div_map tests | low: identical when `None` |
| A3 `document_geometry` child-region recursion | `abstract_ui_viewports.py:286-457` | `test_abstract_ui_viewports.py:49`, div_map, `.mjs` box checks | HIGH: a refactor of the one layout function; parentless output must be byte-identical and box ORDER preserved (first courtyard is the spawn, `div_map:474`; dense ids by appearance order) |
| A4 browser parent for source closures | `abstract_ui_div_map.py:11116` (template) | `.mjs` checks, div_map | low-medium: handwritten template |
| B1 `AbstractUIRegion.rooms=()` | `abstract_ui_introspection.py` (field, `objects()`) | `test_world_reports_exact_...` | low |
| B2 region dict `rooms` | `abstract_ui_div_map.py:341-351` | div_map | low |
| B3 region-level room block | `abstract_ui_viewports.py` | viewports test | medium: new branch, same arithmetic |
| B4 `renderRegion` and closures render rooms | `abstract_ui_div_map.py:10384`, `:11112` (template) | `.mjs` | medium: not required for the 3-D world |
| B5 `filesystem_graph_for_world` owner for buildingless world | `abstract_ui_filesystem.py:258-266` | `test_abstract_ui_filesystem.py` | medium: question 4 |
| C1 `AbstractUIRoom.openings=()` and `_room_model` | `abstract_ui_introspection.py:79`, `div_map:253` | introspection, div_map | low |
| C2 `document_geometry` appends room openings after the door | `abstract_ui_viewports.py:337-341` | viewports, world | low: after existing opening, `openings[0]` unchanged |
| C3 declared pair honoured by portal graph | `abstract_ui_div_map.py:4325` (template) | `.mjs` | medium-high: question 1; shared with the player's portal placement |
| D `build_repository_world` | new function next to `build_introspective_world` | `test_repo_world_scan.py` fixture | low: pure composition; a new test belongs with it |

No change touches `repo_world_scan.py`, the compiler, or any build.

---------------------------------------------------------------------------

## 7. Open questions for the owner (answerable in a sentence each)

1. **Declared pair vs derived portal graph.** Must an import portal be a
   fixed OUT->IN tube (the portal graph honours `pair`), or is the
   distance-weighted all-to-all graph acceptable for file mode?
2. **Nesting shape in the model.** May child regions stay a flat
   `model.regions` list with `contained-by -> parent`, or must the model
   nest them under their parent region as buildings are nested?
3. **Child spacing rule.** May a child region reuse the sibling-block rule
   (`width + 3.5 + max(w,d)^0.72`) inside its parent, rather than the
   region rule (`6 + w^0.78`)?
4. **Reference filesystem for a buildingless world.** Should file mode skip
   `filesystem_graph_for_world` entirely (gap F), or name a root region as
   owner of the five reference artifacts?
5. **Room kind for files.** Is `member_kind="file"` with the language in
   `type_name` the intended open kind, or one kind per language?
6. **Browser template scope.** Are edits to the 2-D map template
   (`renderRegion`, `semanticSourceClosures`) in scope now, or is the first
   deliverable the 3-D world only (G2/G3), leaving the 2-D pane for later?
