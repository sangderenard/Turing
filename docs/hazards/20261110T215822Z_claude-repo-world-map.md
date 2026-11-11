2026-11-10 REPOSITORY WORLD MAP SCANNER (MEASURED, worktree of main a74b3bf5 at C:/Users/alber/AppData/Local/Temp/wt_repomap, branch claude/repo-world-map):

New producer `src/compiler/repo_world_scan.py`: scans a clean git clone into
neutral records (regions = directories, rooms = tracked files, portal pairs =
resolved Python imports via `ast` and JavaScript static ES-module imports,
with source line), keyed by the commit hash. Refuses a tree with uncommitted
or untracked changes (`DirtyCloneError`). Design and the gaps it exposes in
the Living Data Map records are in `docs/REPO_WORLD_MAP_DESIGN.md`.

Green: `python -m pytest tests/test_repo_world_scan.py -q` — 3 passed in
2.1 s on a fixture repository built in `tmp_path` (9 files, 4 regions,
8 portal pairs, 3 unresolved references, 1 unparsable Python file noted).
The third test clones the fixture fresh through `scan_cloned_repository`
after dirtying the working tree, and shows the scan sees only the committed
tree. The GitHub entry (`scan_github_repository`, 2026-11-11) was not run
against github.com here: it needs network and the owner's choice of repo.

Also measured: `scan_repository('.')` on this clean commit afc9b4bb took 47 s
and produced 243 regions, 5,500 rooms, 15,119 portal pairs (14,545 Python,
574 ES-module), 16,648 unresolved references, 39.7 MB of JSON. 998 of the
unresolved are `meowdoku.*`: Python resolution is repository-root only and
`examples/meowdoku` is imported with `examples/` on sys.path (design note,
gap H). The rest are stdlib and third-party names.

2026-11-11: `src/compiler/repo_world_workspace.py` clones the whole workspace
(root repository plus every child repository, in the on-disk layout) from
GitHub into a temporary directory and scans all of it; the manifest is
`docs/repo_world_workspace.json` (15 repositories, 51 worktrees skipped,
discovered from git metadata only). `tests/test_repo_world_workspace.py` —
2 passed in 2.5 s on a fixture workspace with bare "origin" repositories.
The clone against github.com itself was not run here (network, and two refs
need the owner's confirmation: root set to `nogodsnomasters` per AGENTS.md,
speaktome left on `codex/direct-x86-vocabulary-report`).

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
