# Repo world map compile job

`tools/compile_repo_world_map.py` takes a fresh shallow clone of each listed
GitHub repository (never local scratch), runs `scan_repository`, and writes a
compact `repo-world-map.v1` file for the map page.

## Usage

    python tools/compile_repo_world_map.py --out DIR [--repo URL[@ref]]... \
        [--repos-file list.json] [--work SHORTDIR] [--force] [--keep] [--allow-local]

- Only `https://github.com/<owner>/<repo>(.git)` URLs are accepted; local paths
  only with `--allow-local` (tests).
- `--repos-file` is a JSON list of the same `URL[@ref]` strings.
- Clone: `git clone --depth 1 --no-tags --single-branch`, prompts disabled, into
  a new directory under `--work` (default: a short temp dir); deleted afterwards
  unless `--keep`.
- An existing `<commit>/` directory is reused unless `--force`.
- One line per repo: repo, commit[:10], regions, rooms, portals, unresolved,
  seconds, map.json bytes. Exit code is non-zero if any repo failed (all are attempted).

## Output layout

    <out>/index.json                       list of repos with their latest
    <out>/<owner>__<name>/latest.json      {"commit","generated","map","summary"}
    <out>/<owner>__<name>/<commit>/map.json
    <out>/<owner>__<name>/<commit>/world.json.gz   full scan to_data()
    <out>/<owner>__<name>/<commit>/summary.json    counts, seconds, sizes, top 20 rooms
                                                   by in/out, top 20 region links, languages

All writes are temp file + `os.replace`.

## map.json schema "repo-world-map.v1"

    {"schema":"repo-world-map.v1","repo":"<owner>/<name>","origin":"<url>","commit":"<40 hex>","generated":<unix seconds float>,"scan_version":"<REPO_WORLD_SCAN_VERSION>",
     "regions":[{"id":<int>,"path":"<repo-relative posix dir, '' for root>","parent":<int|null>,"files":<direct file count>,"total_files":<recursive>,"bytes":<recursive bytes>}],   // id == index; root is id 0; parents precede children; sorted by path
     "rooms":[{"id":<int>,"region":<region id>,"name":"<basename>","language":"<scanner language>","bytes":<int>,"lines":<int, 0 for binary>,"out":<portal pairs leaving>,"in":<portal pairs entering>}],   // id == index; sorted by full path
     "kinds":["<reference kind>", ...],   // sorted
     "portals":[[<source room id>,<target room id>,<source line>,<index into kinds>], ...],   // sorted
     "region_links":[[<region id a>,<region id b>,<count>], ...],   // portal pairs aggregated by the two rooms' DIRECT regions, a != b, a -> b directed, sorted
     "unresolved_count":<int>,"unresolved_top":[["<specifier head>",<count>], ... up to 30],"notes":[...]}

Written compact, deterministic apart from `generated`. A "specifier head" is the
first dotted segment (python, leading dots kept) or first path segment
(`@scope/pkg` keeps two) for ES modules.

## Suggested schedule (not installed)

Weekly, e.g. Sunday 03:00, from a scheduler of your choice:

    python tools/compile_repo_world_map.py --out <served dir> --repos-file repos.json --work C:\w

Unchanged commits are reused, so a nightly run is cheap.

## Does not

- Install a scheduler, serve or draw the map, or push anywhere.
- Scan local working trees (the clone is the only input) or follow submodules.
- Resolve anything beyond what `repo_world_scan` resolves (python and ES imports).
- Prune old commit directories.
