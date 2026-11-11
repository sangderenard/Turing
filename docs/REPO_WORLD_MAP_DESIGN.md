# Repository world map: design note

Date: 2026-11-10. Branch `claude/repo-world-map`. Scanner:
`src/compiler/repo_world_scan.py`; test: `tests/test_repo_world_scan.py`.

The owner wants the Living Data Map adapted to a filesystem: a world compiled
occasionally from a **fresh clone** of the GitHub repository, folders as
regions and subregions, files as rooms, and portals on a room's walls for the
calls out of that file and the calls into it. Portals are to be lifted out of
the one game page into a general record that the particle simulation and the
studio's shaders and ray tracers can later consume. That later consumption is
out of scope; this note scopes the records and starts the producer.

Everything below marked *observed* was read in the current checkout at
`a74b3bf5`. Nothing was run against the live page.

## 1. What the records are today

### Regions, rooms, tracks: `abstract_ui_introspection.py` (observed)

| Record | Fields | Who fills it |
|---|---|---|
| `AbstractUIRegion` | `identity` (`region:<module>`), `name`, `position: GridPosition`, `buildings: tuple[AbstractUIBuilding]`, `intention` | `build_introspective_world`, one per Python module seen |
| `AbstractUIBuilding` | `identity` (`python:<module>.<qualname>` or `ssa:<id>`), `name`, `module`, `source_kind`, `metaphor`, `position`, `rooms`, `intention` | one per class description |
| `AbstractUIRoom` | `identity` (`<class identity>/member:<kind>:<name>`), `name`, `member_kind`, `metaphor`, `position`, `type_name`, `parameters`, `intentions`, `implied_code` | one per class member |
| `AbstractUITrack` | `source`, `target` (building identities), `relationship` (`inherits`, `contains-type`), `label`, `intention` | from `_python_descriptions`; dropped when either building is absent |
| `AbstractUIWorld` | `identity`, `regions`, `tracks`, `intention`, `schema_version` | the container |

The producer is `build_introspective_world(subject, depth_up, depth_down, seed, …)`.
Its input is a Python class or an SSA `ClassEmission`. It groups class
descriptions by module into regions, lays regions, buildings and rooms on
row-major grids (`_grid_positions`), and mints tracks only between buildings.
`world_div_map_model` (`abstract_ui_div_map.py:313`) serialises exactly this
shape into the page's JSON model, adding `kind`, `dependencies:
[contained-by]` and `interaction: inspect` per node.

Facts that matter for the filesystem case:

- A region has **no parent**. `AbstractUIRegion` is flat and
  `AbstractUIWorld.regions` is a flat tuple. There is no subregion.
- A room lives **only inside a building**. There is no room directly in a
  region, and the room identity format is built from the class identity.
- `member_kind` is a closed vocabulary: `field`, `property`, `method`,
  `constructor`, `nested_class`. `document_geometry` keys room height on it
  and `_introspective_performance_labels` filters on `method`.
- A track joins two **buildings**, has no source line, no direction beyond
  `source`/`target`, and is silently filtered when an endpoint is missing.

### Geometry, openings and portals (observed)

`document_geometry` (`abstract_ui_viewports.py:286`) turns the serialised
regions/buildings/rooms into boxes. It mints every opening itself: one
`door` on the south wall of every building (`/opening:entry`) and every room
(`/opening:door`), one `gate` on every region courtyard (`/opening:gate`).
Openings are `{identity, kind, side, offset, width, height}` dicts inside
`box["openings"]`; no Python record type exists for them.

**There is no Python portal record.** Openings with `kind: "portal"` and
`port_role: in|out` are created only in the browser by `placePortalSplat`
and `portalRuntime` inside `DIV_MAP_JAVASCRIPT` (`abstract_ui_div_map.py`
around lines 3123, 4007–4054, 4422–4446, 5825–5884). Python publishes only the
policy: `PlacementPolicy.portal_contract` (`abstract_ui_placement.py:206`,
many-to-many, in-to-out, probabilistic tube graph, person/vehicle apertures)
and a `PlacementRecipe` with `opening_kind="portal"`. The game page owns the
portal instances, their persistence (cookie/local storage) and their
realisation (splats, quaternion tubes). The 2026-08-25 audit already names
this as bespoke semantic runtime living in the backend template.

Two other records come close and are not portals:

- `abstract_ui_house_builder.RoomOpening` has `connects_to` and
  `seal_model` and produces a `portal-adjacency` edge between two *air
  volumes*. Its `kind` is limited to `door|gate|window`, its `side` to the
  four walls, and it connects volumes for the atmosphere engine, not rooms
  for traversal.
- `abstract_ui_world.pluck_placed_object` adapts a Pluck `portal` object
  with `form.target_region` and capability `traverse`. It is one-ended (a
  target, no source) and exists only as an import adapter.

### Filesystem: `abstract_ui_filesystem.py` (observed)

`FileSystemNode(identity, path, kind: directory|file, parent, artifact)` and
`FileSystemGraph(identity, root, nodes, artifacts)` already model a directory
tree with `contained-by` dependencies and canonical POSIX paths. But:

- `filesystem_graph_for_world` **fabricates** the tree from the first
  building's module name: five synthetic `HumanArtifact`s (source, test,
  readme, annotation, scratch). It never reads a filesystem.
- `FileSystemGraph.__post_init__` requires every file node to have exactly
  one `HumanArtifact`, and `HumanArtifact` requires an `owner`, a
  `placed_in` container and an `ArtifactAttachment` (weld physics). A real
  scan has thousands of files and no owner or placement for them.
- Nodes are not regions or rooms; artifacts are loose tokens placed in the
  courtyard by `artifact_geometry_boxes`.

So the filesystem graph is a reference fixture for the game's loose-token
mechanic, not a producer the scan can fill as it stands.

## 2. How a filesystem scan feeds the same records

The scan is a second **producer** standing where `build_introspective_world`
stands. Its output is a neutral record set (`RepoWorldScan`) that a world
builder then lifts into `AbstractUIWorld`, so `world_div_map_model`,
`document_geometry`, the identity specialisation table and the page consume
it unchanged. The mapping, record by record:

| Filesystem fact | Scan record | Target record | Status |
|---|---|---|---|
| repository at commit `H` | `RepoWorldScan.commit`; identity `repo:H` | `AbstractUIWorld.identity` | fits |
| directory | `RepoRegion(identity, path, name, parent, depth)` | `AbstractUIRegion` | **gap A**: region has no `parent`; subregions cannot be expressed |
| file | `RepoRoom(identity, path, name, region, language)` | `AbstractUIRoom` | **gap B**: a room must sit in a building and carry a class `member_kind` |
| import A → B | `RepoPortalPair` with an `out` end on A's room and an `in` end on B's room, plus `source_line`, `reference_kind`, `specifier` | a portal record | **gap C**: no Python portal record exists; `AbstractUITrack` is building-level and lineless; openings are minted by geometry |
| import of an untracked target | `RepoUnresolvedReference` | none | kept so nothing disappears; not a portal because a portal needs two rooms |

Identities are keyed by the commit: `repo:<H>/dir:<path>`,
`repo:<H>/file:<path>`, `repo:<H>/portal:<src>:<line>-><dst>`, with ends
`…/end:out` and `…/end:in`. Two scans of one commit produce equal records
(asserted by the test); two commits never collide. The root directory is
`dir:.`.

The source is GitHub, not the owner's local filesystem. The local scratch
tree is messy and holds things that must not be exposed, so the scanner's
public entry is `scan_github_repository("owner/repo", ref=…)`: it makes a
fresh shallow clone of `https://github.com/<owner>/<repo>.git` in a
temporary directory, scans it, and deletes the clone. Only what GitHub
serves can reach the record set. The compile step is one command:

```bash
python -m src.compiler.repo_world_scan sangderenard/turing --ref main --json
```

The record set names the commit the clone was at. Underneath,
`scan_repository(path)` still checks that the path is a repository top level
and that `git status --porcelain --untracked-files=all` is empty, refusing
with `DirtyCloneError` otherwise; that guard is what makes the fresh clone
the only thing the scan can describe. The `--path` form exists for fixture
tests, not for the owner's checkout. Git LFS smudge is disabled during the
clone so no large binaries are fetched; the scan reads text sources only.

### The whole workspace, in its on-disk layout

The owner's workspace is one root repository (`nogodsnomasters`) whose child
directories are independent repositories. `src/compiler/repo_world_workspace.py`
clones all of it from GitHub in that layout:

1. `discover` reads only git metadata of the local tree (which top-level
   directories are repositories, each one's remote URL normalised to GitHub
   HTTPS, its current branch) and writes `docs/repo_world_workspace.json`.
   No file contents are read. Git worktrees (a `.git` file) are skipped as
   views of another entry; a repository with no remote is skipped with a
   note. The committed manifest is the owner's to edit; its `ref` per entry
   is what GitHub is asked for.
2. `clone` recreates the layout under an empty destination with shallow
   clones, root first, children inside it. Nothing is copied from disk.
3. `scan` clones into a temporary directory, runs `scan_repository` on every
   clone and emits one workspace record: each repository's scan plus a
   `mounts` table saying which root-repository region each child
   repository's root region (`dir:.`) is contained by.

```bash
python -m src.compiler.repo_world_workspace scan docs/repo_world_workspace.json --json
```

Manifest state at 2026-11-11: 15 repositories, 51 worktrees skipped. The
root entry's ref was set to `nogodsnomasters` (the branch the root AGENTS.md
names) rather than the session branch the checkout was on; `speaktome` is
recorded on its session branch `codex/direct-x86-vocabulary-report`, which
may not be the branch GitHub should serve. `pcb` points at the
`electrofluid` repository, which may be private; a failed clone is a note,
not a crash, unless `strict` is set.

### Reference extraction

- **Python**: `ast.parse` per tracked `.py`; `Import` and `ImportFrom`
  (relative levels resolved against the file's directory). A module name
  resolves when `a/b/c.py` or `a/b/c/__init__.py` is tracked. For
  `from pkg import name`, `pkg/name.py` wins when tracked, otherwise the
  reference is to `pkg` itself. A file that does not parse is recorded in
  `notes`, not skipped silently.
- **JavaScript**: static ES-module forms only (`import … from`, bare
  `import "…"`, `export … from`), one statement per line, relative
  specifiers resolved with `""`, `.js`, `.mjs`, `/index.js`, `/index.mjs`.
  Bare specifiers are unresolved by rule. Dynamic `import()`, CommonJS
  `require`, and multi-line import statements are not read.

Both extractors are declared; nothing else is inferred. Calls other than
imports (a function in A calling a function in B) are not references in
this version; the owner's "calls out of / into a file" is served at module
granularity first.

## 3. The general portal record

A portal must not be owned by the game page. The scan's
`RepoPortalPair`/`RepoPortalEnd` is the first instance of the general shape:

| Need | Field | Filled by |
|---|---|---|
| identity, stable for a commit | `pair.identity` = `repo:<H>/portal:<src>:<line>-><dst>` | scanner |
| the two endpoints | `source_room`, `target_room`; each `RepoPortalEnd.room` | scanner |
| direction | `direction: source-to-target`; ends carry `role: out` (on the source room) and `role: in` (on the target room) | scanner |
| provenance | `source_line`, `reference_kind`, `specifier` | scanner |
| which wall, where on it | *not a field of the record* | world builder (realisation, like `document_geometry` picks `south` today) |
| transmission / optical slot | `RepoPortalEnd.transmission: Mapping | None`, always `None` from the scanner | particle simulation, shaders, ray tracers, later |

The two ends are separate records with their own identities because the
particle simulation and the renderers address a *surface on one room's
boundary*, not the pair. The pair is the fact; the ends are its two faces.
`portal_contract` (apertures, tube graph, Gaussian splats) remains what it is
today: a realisation policy of one backend, consulted when the page draws an
end, never written into the record.

## 4. Reused unchanged

- `AbstractUIWorld` as the container, `objects()`, `world_div_map_model`
  serialisation, `document_geometry` layout, `_grid_positions`,
  `identity_specialization_table`, `project_world_to_div_map`.
- The identity discipline: authored string identities, `contained-by`
  dependencies, dense runtime ids minted only at a host boundary.
- The vocabulary: `containment` already declares `filesystem directory
  file_node region room`, `navigation` declares `portal entrance exit`,
  `world` declares `portal_role` and `track_role`.
- `tools/hazards_log.py` and the test layout.

Nothing in `abstract_ui_*.py` was edited on this branch.

## 5. Missing, named rather than worked around

- **A. Region parent.** `AbstractUIRegion` needs a `parent: str | None`
  (or `AbstractUIWorld.regions` needs nesting) before a folder tree can be
  expressed. `document_geometry` lays regions side by side along one axis
  and would need to lay a child region inside its parent's courtyard.
- **B. Room directly in a region, with an open kind.** `AbstractUIRoom`
  needs a home other than a building and a `member_kind` outside the class
  vocabulary (`file` with a `language`), and the two consumers that key on
  `member_kind` (`document_geometry` heights,
  `_introspective_performance_labels`) need a rule for it. Inventing one
  building per folder to hold files as rooms would be a stand-in and is not
  proposed.
- **C. A Python portal record and a producer-owned opening.** There is no
  record type for an opening or a portal on the Python side; `document_geometry`
  mints the openings and the browser mints the portals. The world builder
  must be able to receive the scan's portal ends and place one opening per
  end on the room box, and `AbstractUITrack` cannot stand in for the pair
  (building-level, lineless, filtered when an endpoint is missing).
- **D. A world builder from scan records.** `build_introspective_world` only
  accepts a class or a `ClassEmission`. A second entry that takes
  `RepoWorldScan` and produces `AbstractUIWorld` is the next step and
  depends on A–C.
- **E. Scale.** `document_geometry` grows region spacing super-linearly
  (`pow(region_width, 0.78)`, `pow(occupied_span, 1.18)`) and the page model
  of a one-class world is already ~800 KB. A repository of thousands of
  files is not laid out or paid for by the current geometry; this is a
  measurement to make before feeding, not a reason to skip rooms.
- **F. `FileSystemGraph` coupling.** Its file nodes require a
  `HumanArtifact` with owner, placement and weld physics. Either the
  constraint relaxes to allow artifact-less file nodes, or the scan's
  regions/rooms stay the filesystem authority and the fixture graph is left
  for the loose-token mechanic. The scan does not fill it.
- **G. Calls inside files.** Function-level calls across files are not
  extracted; only module imports are. Adding them needs the compiler's own
  process graph for Python, not a second resolver.
- **H. Import roots other than the repository root.** Python names resolve
  only from the repository root. Measured on this repository's own clean
  commit (`afc9b4bb`, 47 s, 243 regions, 5,500 rooms, 15,119 portal pairs,
  16,648 unresolved, JSON 39.7 MB): 998 unresolved references are
  `meowdoku.*`, because `examples/meowdoku` is run with `examples/` on
  `sys.path`. The rest of the unresolved set is standard library and
  third-party (`typing`, `numpy`, `pytest`, `OpenGL`, …) as expected. The
  scanner needs the repository's declared import roots as an input; it must
  not guess them from directory names.

## 7a. Measurement of the whole workspace from GitHub (observed, 2026-11-11)

`scan_workspace(docs/repo_world_workspace.json)`: all 15 repositories cloned
and scanned from github.com, no failed clones (including `pcb` → electrofluid
and `speaktome` on its session branch), 14 mounts, 235 s end to end.

| repository | commit | regions | rooms | portal pairs | unresolved |
|---|---|---|---|---|---|
| `.` (nogodsnomasters) | 937de5dd | 867 | 2,677 | 35 | 5,124 |
| amp | 2b61525d | 84 | 262 | 74 | 637 |
| fftfree | cb884533 | 7 | 44 | 0 | 111 |
| geometry | 97128724 | 14 | 172 | 0 | 0 |
| lfsavoider | c7412218 | 2 | 28 | 0 | 0 |
| meta-nn | 93fc1dc9 | 13 | 191 | 411 | 1,035 |
| nodus | fe678e9e | 43 | 486 | 40 | 600 |
| pcb | b6c64e6e | 5 | 48 | 73 | 125 |
| pigeon | c8f4854c | 1 | 13 | 17 | 95 |
| socratic | 6054198b | 11 | 142 | 217 | 1,018 |
| speaktome | 86721ddb | 210 | 2,388 | 712 | 4,666 |
| spectral-analyzer | edddb335 | 103 | 1,029 | 1,557 | 6,175 |
| tensor-calculator | 34f9fec6 | 4 | 6 | 0 | 0 |
| turing | d2a7f022 | 245 | 5,772 | 15,331 | 17,583 |
| wheelhouse | 0c4c7a37 | 2 | 12 | 0 | 0 |
| **total** | | **1,611** | **13,270** | **18,467** | **37,169** |

Workspace JSON: 56.7 MB. The root repository alone holds 867 regions and
2,677 rooms on GitHub's `nogodsnomasters` branch, far more than the
coordination files the root AGENTS.md describes; what it tracks there is
worth reading before the world builder lays it out. C++ repositories
(geometry, fftfree, nodus) show few or no portals because only Python and
ES-module imports are extracted (gap G).

## 7. Measurement on this repository (observed, 2026-11-10)

| quantity | value |
|---|---|
| scan time, clean commit `afc9b4bb` | 47 s |
| regions / rooms | 243 / 5,500 |
| portal pairs (python / es-module) | 15,119 (14,545 / 574) |
| unresolved references | 16,648 (8 self-references) |
| `to_data()` as JSON | 39.7 MB |

The current page embeds its whole model in one document (~800 KB for one
class), so gap E is confirmed at this size: the world builder cannot embed
this record set; it has to select a window or serve it separately.

## 6. What feeding the records into the world builder requires

1. Resolve A and B in `abstract_ui_introspection.py` (another agent holds
   uncommitted edits in `abstract_ui_div_map.py` and
   `abstract_ui_introspection.py` today; coordinate before touching them).
2. Add the portal record (C) where openings are declared on the Python side,
   and make `document_geometry` consume declared openings instead of minting
   only south doors.
3. Write `build_repository_world(scan: RepoWorldScan) -> AbstractUIWorld`
   (D) beside `build_introspective_world`, assigning `GridPosition`s with
   `_grid_positions` as today.
4. Measure E on this repository's own clone before rendering any HTML.
