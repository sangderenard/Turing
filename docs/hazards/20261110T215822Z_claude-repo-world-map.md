2026-11-10 REPOSITORY WORLD MAP SCANNER (MEASURED, worktree of main a74b3bf5 at C:/Users/alber/AppData/Local/Temp/wt_repomap, branch claude/repo-world-map):

New producer `src/compiler/repo_world_scan.py`: scans a clean git clone into
neutral records (regions = directories, rooms = tracked files, portal pairs =
resolved Python imports via `ast` and JavaScript static ES-module imports,
with source line), keyed by the commit hash. Refuses a tree with uncommitted
or untracked changes (`DirtyCloneError`). Design and the gaps it exposes in
the Living Data Map records are in `docs/REPO_WORLD_MAP_DESIGN.md`.

Green: `python -m pytest tests/test_repo_world_scan.py -q` — 2 passed in
1.5 s on a fixture repository built in `tmp_path` (9 files, 4 regions,
8 portal pairs, 3 unresolved references, 1 unparsable Python file noted).

Not run: the broader abstract_ui suites and any HTML generation. No
`abstract_ui_*.py` file was edited; nothing in the live page changed.

Hazards and limits observed while writing:
- The existing records cannot hold the filesystem case without changes:
  `AbstractUIRegion` has no parent (no subregion), `AbstractUIRoom` lives
  only inside a building with a class `member_kind`, and there is no Python
  portal record at all (openings are minted by `document_geometry`, portals
  only by the browser). These are named in the design note, not worked
  around; the scan output is not yet fed into `AbstractUIWorld`.
- `FileSystemGraph` requires a `HumanArtifact` per file node, so the scan
  does not fill it.
- JavaScript extraction is line-based: multi-line import statements,
  dynamic `import()` and CommonJS `require` are not read. Bare specifiers
  are recorded as unresolved by rule.
- Another agent holds uncommitted edits in `src/compiler/abstract_ui_div_map.py`
  and `abstract_ui_introspection.py` in the main checkout; this branch was
  made in a separate worktree so those edits were not touched.
- `python tools/hazards_log.py` (show) currently aborts on main: ten entry
  files committed by other branches break the name pattern (no time or no
  `Z`: `20261109T_material-library-paused-checkpoint.md`,
  `20261110T0527_shop-emitter-diagnostic.md`, `20261110T0648_…`, and seven
  `20261110T_*.md`). `new` still works. Not fixed here; those files belong
  to their branches.
