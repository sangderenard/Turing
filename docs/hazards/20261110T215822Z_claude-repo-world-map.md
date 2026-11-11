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
Run against github.com on the owner's go-ahead (2026-11-11): all 15
repositories cloned and scanned, no failed clones, 235 s; totals 1,611
regions, 13,270 rooms, 18,467 portal pairs, 37,169 unresolved references,
56.7 MB JSON, 14 mounts. Per-repository table in the design note §7a. The
root repository's `nogodsnomasters` branch on GitHub has 867 regions and
2,677 rooms, far more than the root AGENTS.md says it tracks.

2026-11-11, later: scan languages now come from the compiler's
`DREAM_LANGUAGE_TRANSLATIONS`; each scan publishes a `languages` table with
the route and linking state per language (python, javascript linked; cpp,
glsl, wgsl, sympy known but no reader, so no portals). Fixture gained two C++
files. `tests/test_repo_world_scan.py` + `tests/test_repo_world_workspace.py`:
5 passed in 5.3 s.

2026-11-11, dream interpreter (owner: "you touched it, it's yours"): measured
every language route of `src/compiler/dream_document.py` block by block;
report in `docs/DREAM_INTERPRETER_LIMITS.md`, pinned by
`tests/test_dream_interpreter_limits.py` (8 passed). Baseline before edits:
`tests/test_dream_document.py` 16 passed, `tests/test_dream_cpp_pipeline.py`
FAILED on this autocrlf checkout with pycparser `Illegal character '\r'`;
with the committed LF bytes it fails instead at
`topological_reducer._set_operands:3697` "no operand-position scope" (gate
from 8af44b07, 2026-10-17). That C-route failure is on main for every C
block that parses and is NOT fixed here (operand-position-writer migration).
Repaired: (1) `compile_sections` opens an identity book per block — a `with`
refusal poisoned every later Python block in the process with a
`native_exception_control_gap disagreement`; (2) desugarer drops `#include`,
refuses other directives, rewrites `C c(3);` to `C c = C__new(3);`, and no
longer emits `None__method` for an unknown receiver; (3) JavaScript route
sees `export`ed and destructured declarations; (4) `.gitattributes`:
`*.dream text eol=lf`. After edits: 31 passed, 1 failed (the C-route gate).
Real Python-route limits: `raise`, `try`, `with` refused by the declared
exception/resource gap; everything else probed admits.

2026-11-11, stand-ins (owner: "Yes, start it"): `prose` declared in
`DREAM_LANGUAGE_TRANSLATIONS` (route `stand-in`, target `work-table`);
`satisfies`/`author` sentinel keys; `DreamDocument.work_table()` with
`DreamStandIn`/`DreamContribution`/`DreamWorkTable` (state open/satisfied,
owner = first code contributor else describer, refinement children keep the
parent open, cycles and unauthored contributions refused at parse time);
world scan mints `RepoRoomClaim` (`claimed-room`) per stand-in of a tracked
`.dream` file (`DOCUMENT_SUFFIXES`, language `dream-document`). Decision
taken without the owner: ownership passes to the code author on
satisfaction, the describer holds it before. `tests/test_dream_work_table.py`
5 passed; with the scanner, workspace, limits and dream-document suites: 34
passed in 9.2 s. Nothing in `abstract_ui_*` consumes the claim (gap D).

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
