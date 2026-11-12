# Repository world map: handover report, 2026-11-12

Branch `claude/repo-world-map` on the turing repository, based on main
`7813a6a3` (main merged in at `64b0e8c1`). Worktree
`C:/Users/alber/AppData/Local/Temp/wt_repomap`. Nothing pushed, nothing
merged into main. Owner's standing constraints were kept: existing mechanisms
only, no stand-ins for systems already here, no broad test sweeps, no builds,
search through git, no edits to `abstract_ui_*`, no HTML world generated, no
game pages or shaders touched.

## 1. What was asked, in order

1. Parent session: scope and start the repository world map (folder = region,
   file = room, cross-file reference = portal pair with a general portal
   record); design note, scanner, targeted test, hazards entry, commit on a
   branch, do not push or merge.
2. Owner: map from GitHub, not the local filesystem (the local tree is messy
   and holds things not to expose).
3. Owner: a Python that clones the whole workspace in its on-disk structure
   (root `nogodsnomasters` plus every subrepo) and run it.
4. Owner: C++ repositories are the machine sim; the known-languages table
   exists, record that C++ had no linker.
5. Owner: test and develop the dream interpreter so its limits can be stated.
6. Owner: natural-language stand-ins; open work table; satisfied by an
   authored contribution; document becomes a tree; a room claimed but
   unaltered gives ownership to the author. "Yes, start it."
7. Owner: these are personal assets and transient intermediaries; store in
   user space or clean up per the call; avoid disk when the records only
   pass through an operational set on source. "Yes, wire it."
8. Owner: recheck everything after the app's worktree removal.
9. Owner: decide, with care, whether to merge main's divergent scanner.
10. Owner: thorough report, return the branch.

## 2. Commits on the branch (oldest first)

| Commit | Content |
|---|---|
| 62c77070 | Scan from GitHub: fresh shallow clone in a temp directory (later superseded by the stream lanes). |
| 8c8d30ec | `repo_world_workspace.py`: discover the on-disk layout from git metadata, clone it in that layout, scan each repository. |
| 49ed8697 | First whole-workspace scan recorded in the design note §7a. |
| 0e56e142 | Languages come from `DREAM_LANGUAGE_TRANSLATIONS`; `LINKING_READERS` declares per language whether a reference reader exists. |
| 89936d1a | Dream interpreter limits measured; repairs to `cpp_shell_desugar.py` and `dream_document.py` that the measurement found. |
| dd90b61b | Prose stand-ins: work table, satisfaction tree, claimed rooms. |
| 6557f9fd | Stream lanes: sources go through the scan as a tar stream; the committed workspace manifest removed as a personal asset. |
| 3df18e29 | `repo_world_sink.py`: scan summaries, claims and work tables sealed into account workspaces through the service's store. |
| 64b0e8c1 | Merge of main: v1 extractor-plugin scanner joined with the stream lanes. |
| 72a7ca10 | Hazards correction: the worldmap page test passes with `SPEAKTOME_REPO` set. |

Diff against main before this report: 15 files, +2563 / -177. New modules:
`src/compiler/repo_world_workspace.py`, `src/compiler/repo_world_sink.py`.
Changed: `src/compiler/repo_world_scan.py`, `src/compiler/dream_document.py`,
`src/compiler/cpp_shell_desugar.py`, `.gitattributes`. Docs:
`REPO_WORLD_MAP_DESIGN.md`, `DREAM_INTERPRETER_LIMITS.md` (new),
`DREAM_DOCUMENT.md`, the hazards entry. Tests: five new files, one extended.

## 3. What the branch delivers

### 3a. Scanner (`repo_world_scan.py`)

Records: `RepoRegion` (directory), `RepoRoom` (file, with `language`),
`RepoPortalPair` (two `RepoPortalEnd`s, out and in, `transmission=None`
until the portal record carries one), `RepoUnresolvedReference`,
`RepoRoomClaim`, `RepoWorldScan` (schema `repo-world-scan-v1`, fields
`languages`, `claims`, `extractor_languages`, `import_roots`, `notes`).
Identities: `repo:<commit>/dir:<path>`, `/file:<path>`,
`/portal:<src>:<line>-><dst>`, `/claim:<doc>#<block>`.

Lanes, all over one core `scan_files(commit, origin, files, read, notes,
*, extractors, import_roots)`:

- `scan_github_repository(owner/name, ref=)`: GitHub API gives the commit
  sha, the tarball is streamed (`mode="r|gz"`) and only the bytes of files
  the readers need are kept. No clone, no temp directory.
- `scan_remote_repository(url, ref=)`: `git ls-remote` plus
  `git archive --remote` for non-GitHub remotes (also `file://` fixtures).
- `scan_repository(path)`: an on-disk checkout, refused when dirty
  (`DirtyCloneError`). This is main's compile job's lane and keeps main's
  explicit `import_roots=()` contract.

From main, joined in the merge: `EXTRACTOR_MODULES` plugin registry (C
includes, CMake, HTML, dynamic JavaScript, PowerShell, shell; a plugin that
fails or returns an untracked target is a scan error, never skipped);
declared Python import roots (`import_roots=`, `--import-root`, or the
tree's own `pyproject.toml` / `repo_world.toml` / `pytest.ini` /
`setup.cfg` / `tox.ini`, read through the same reader so the stream lanes
honour them); pair identity (source, line, target).

Join decisions taken by me: a room's language is the compiler's name when a
known suffix claims it (`.cpp` is `cpp`, not the plugin's `c`), else the
extractor's `LANGUAGE`; the `languages` table reports `linking_readers` per
compiler language and adds extractor-only rows; stream lanes default to
declared roots (`import_roots=None`).

CLI: `python -m src.compiler.repo_world_scan <owner/repo | url | --path
checkout> [--ref] [--import-root] [--json | --out <file>]`. Records leave
memory only where the call says.

### 3b. Workspace (`repo_world_workspace.py`)

`discover(root)` reads git metadata only (remote, branch, head, relative
path) for the root and every nested repository; `scan_workspace_root(root)`
discovers and streams each repository from its remote with no manifest on
disk; `scan_workspace(manifest, strict=False)` for an explicit manifest;
`clone` exists only for explicit on-disk requests. The committed
`docs/repo_world_workspace.json` was removed: it is the owner's personal
layout.

### 3c. Languages

Scan languages come from `DREAM_LANGUAGE_TRANSLATIONS`, the existing
known-languages table. A startup check refuses a suffix table naming a
language the compiler table lacks. Before the merge the table recorded that
C++ had no linker (so geometry, fftfree and nodus showed no portals, a
declared absence, not a property of the code). After the merge the C
include plugin links C++ rooms; the §7a zero columns are historical and
marked so.

### 3d. Dream interpreter limits and stand-ins

`docs/DREAM_INTERPRETER_LIMITS.md` states what the interpreter can and
cannot do, measured by `tests/test_dream_interpreter_limits.py` (8 tests).
Repairs made along the way: CRLF `.dream` sources (`.gitattributes
*.dream text eol=lf`), a `with`-refusal that poisoned later blocks (per-block
identity book), desugarer fixes for `#include`, constructors and
`None__method`, JavaScript `export` and destructuring.

Stand-ins: `language = prose` is a declared language with route `stand-in`
and target `work-table`, not executable. `satisfies = <id>` and `author =`
on a code block close the stand-in; `DreamDocument.work_table()` gives a
`DreamWorkTable` of `DreamStandIn` records (state open/satisfied, owner =
first code contributor else the describer) with `DreamContribution` edges
(the document as a tree). Parse-time refusals: non-prose target, self
reference, missing author, cycle. Compiling a document with an open
stand-in reports the shortfall by name. In the scan, a `.dream` room with a
satisfied stand-in yields a `RepoRoomClaim` (`claimed-room`) naming the
owner: the room is claimed but unaltered.

### 3e. User-space sink (`repo_world_sink.py`)

`scan_bundle(scan, full=False)` (summary.json, claims.json, scan.json only
on request), `work_table_bundle(document)`, `submit(store, actor_id,
bundle)` through `examples/meowdoku/workspace_accounting.Store` (reserve a
lease sized from the bundle, `seal_and_submit`, `get_ticket`),
`read_bundle`. The actor is a `users.id` row and the store authenticates it;
the sink decides nothing about identity, writes no file, opens no database.
Store cap 1 MiB / 128 files, so the full record set is stored only when
asked and when it fits. The store module, `schema_version.py`, its test and
foundation note were brought from main verbatim, so the merge saw identical
files.

## 4. Measurements

- turing main `7813a6a3` streamed from GitHub, joined scanner: 39 s,
  nothing on disk; 245 regions, 5,881 rooms, 16,509 portal pairs across 13
  reference kinds, 16,850 unresolved, 0 claimed rooms. Declared import root
  `examples` read from the stream; `meowdoku.*` unresolved went 998 to 0.
- Whole-workspace scan from GitHub (pre-merge lane) recorded in the design
  note §7a per repository.
- Tests at 64b0e8c1: 193 passed in 24 s across scan, workspace, sink,
  work-table, six plugin, compile-job, portal-record, dream-document, render
  and interpreter-limits suites. `tests/test_meowdoku_worldmap.py`: 15
  passed with `SPEAKTOME_REPO=C:/dev/Powershell/speaktome` (the server
  defaults the REST package to the checkout's sibling folder, which a Temp
  worktree does not have; my earlier "environment" wording was wrong and is
  corrected in the hazards entry). Commits after 64b0e8c1 change only docs.
- `tests/test_workspace_accounting.py`: 37 passed on this base.
- Not run: broader `abstract_ui` suites, any HTML generation,
  `tests/test_dream_cpp_pipeline.py` (red on main independently: the C
  route's operand-position gate).

## 5. Named, not fixed

- `tools/compile_repo_world_map.py` (main's) still clones to a `--work`
  directory and writes map files to `--out`. It is the one remaining
  on-disk transient; its contract and tests are main's, so I left it.
- Nothing in `abstract_ui_*` consumes scan records or room claims yet
  (gaps A-H in the design note). Another agent held uncommitted edits
  there, so I did not touch those files.
- `author` on a block is free text, not joined to a `users.id` row; that
  join is the service's identity rule.
- A stand-in can only be satisfied from within its own document.
- `hazards_log show` is broken by misnamed files from other branches.
- The service has no workspace HTTP route, so the sink has no route either.

## 6. Returning the branch

The branch is ready for review and landing by whoever owns that step. To
land it: review `git diff main...claude/repo-world-map` in the turing
repository, run the suites named in §4 (set `SPEAKTOME_REPO` for the
worldmap page test), then merge or rebase as the branch policy says. I did
not push. The worktree at `C:/Users/alber/AppData/Local/Temp/wt_repomap` can
be removed with `git worktree remove` once the branch is landed; the
branch ref lives in the turing repository, not the worktree.
