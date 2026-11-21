#!/usr/bin/env bash
# publish.sh -- Update-LiveCheckouts.ps1 for Linux: move each live checkout (branch_policy.json's repositories, under
# SPEAKTOME_ROOT) to what landed on the remote, then re-export the site's pages when turing or spectral-analyzer moved.
#
# A checkout moves only when it is on its default branch with no tracked changes, and only by fast-forward: ahead and
# diverged are reported and left. A shallow clone -- what bootstrap_linux.sh makes -- is deepened first until git can
# prove whether one tip descends from the other. Nothing is pushed, reset or deleted. A restart loads moved code.
#   publish.sh [--what-if] [--export-web]
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

WHATIF=0
EXPORT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --what-if) WHATIF=1; shift ;;
    --export-web) EXPORT=1; shift ;;
    *) refuse "unknown argument $1" ;;
  esac
done
gate "$(basename "$0")"

POLICY="$DEPLOY_DIR/branch_policy.json"
REMOTE="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["remote"])' "$POLICY")"
mapfile -t ENTRIES < <("$PYTHON" -c '
import json, sys
for name, entry in json.load(open(sys.argv[1]))["repositories"].items():
    print(name, entry["path"], entry["default"])' "$POLICY")

# Deepen a shallow clone (200 commits a round, at most 25, then the whole history) until either tip is an ancestor of
# the other or the clone is complete; a full clone returns at once. Non-zero only when a fetch fails.
deepen_until_related() {
  local path="$1" remote="$2" base="$3" here="$4" there="$5" round
  for round in $(seq 1 26); do
    [ "$(git -C "$path" rev-parse --is-shallow-repository)" = true ] || return 0
    if git -C "$path" merge-base --is-ancestor "$here" "$there" 2>/dev/null ||
       git -C "$path" merge-base --is-ancestor "$there" "$here" 2>/dev/null; then return 0; fi
    if [ "$round" -lt 26 ]; then
      git -C "$path" fetch --quiet --deepen=200 "$remote" "$base" || return 1
    else
      git -C "$path" fetch --quiet --unshallow "$remote" "$base" || return 1
    fi
  done
}

MOVED=()
printf '%-18s %-11s %s\n' REPOSITORY STATE DETAIL
for entry in "${ENTRIES[@]}"; do
  read -r name winpath base <<<"$entry"
  path="$(policy_path "$winpath")"
  row() { printf '%-18s %-11s %s\n' "$name" "$1" "$2"; }
  if [ ! -e "$path/.git" ]; then row skipped "no checkout at $path"; continue; fi
  branch="$(git -C "$path" rev-parse --abbrev-ref HEAD)"
  if [ "$branch" != "$base" ]; then row skipped "on $branch, not $base"; continue; fi
  if [ -n "$(git -C "$path" status --porcelain --untracked-files=no)" ]; then row skipped "tracked changes"; continue; fi
  shallow="$(git -C "$path" rev-parse --is-shallow-repository)"
  here="$(git -C "$path" rev-parse HEAD)"
  depth=(); [ "$shallow" != true ] || depth=(--depth 1)
  if ! git -C "$path" fetch --quiet "${depth[@]}" "$REMOTE" "$base"; then row failed "fetch failed"; continue; fi
  there="$(git -C "$path" rev-parse "$REMOTE/$base")"
  if [ "$here" = "$there" ]; then row current "at ${here:0:8}"; continue; fi
  # A shallow clone (bootstrap_linux.sh makes them) proves nothing until its history reaches back far enough: deepen
  # until git can say whether one tip descends from the other, or the clone is complete. Fetching moves no checkout,
  # so --what-if stays repeatable; the move below is the same fast-forward as for a full clone (review 100f9c6d).
  if ! deepen_until_related "$path" "$REMOTE" "$base" "$here" "$there"; then row failed "fetch failed while deepening"; continue; fi
  if git -C "$path" merge-base --is-ancestor "$there" "$here"; then row ahead "commits not on $REMOTE"; continue; fi
  if ! git -C "$path" merge-base --is-ancestor "$here" "$there"; then row diverged "both have commits of their own"; continue; fi
  if [ "$WHATIF" -eq 1 ]; then row "would move" "${here:0:8} -> ${there:0:8}"; continue; fi
  if git -C "$path" merge --ff-only --quiet "$REMOTE/$base"; then row moved "${here:0:8} -> ${there:0:8}"; MOVED+=("$name")
  else row blocked "untracked files in the way"; fi
done
[ "${#MOVED[@]}" -eq 0 ] || say "Moved checkouts take effect at the next restart."

# The published pages: web_export.py copies them from turing and the shaders from spectral-analyzer, so a move of either
# makes the served copy stale. --skip-boards keeps the board database (boards are baked on a GPU host and copied in).
pages_moved=0
for name in "${MOVED[@]:-}"; do case "$name" in turing|spectral-analyzer) pages_moved=1 ;; esac; done
if [ "$EXPORT" -eq 1 ] || [ "$pages_moved" -eq 1 ]; then
  LIVE_TURING="$SPEAKTOME_ROOT/turing"
  for source in turing spectral-analyzer; do
    on="$(git -C "$SPEAKTOME_ROOT/$source" rev-parse --abbrev-ref HEAD)"
    want="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["repositories"][sys.argv[2]]["default"])' "$POLICY" "$source")"
    [ "$on" = "$want" ] || refuse "to publish the pages: $source is on $on, not $want."
  done
  COMPILER_PYTHON="${TURING_PYTHON:-$LIVE_TURING/.venv/bin/python}"   # web_export compiles a shader: the compiler's env
  mapfile -t WEB_DIRS < <("$PYTHON" - "$DEPLOY_DIR/instances.json" "$LIVE_TURING" <<'PY'
import json, sys
from pathlib import Path
dirs = {str(Path(sys.argv[2]) / "build" / "meowdoku" / "web")}
for instance in json.load(open(sys.argv[1])).get("instances", []):
    args = list(instance.get("args") or [])
    if "--web-dir" in args and args.index("--web-dir") + 1 < len(args):
        dirs.add(args[args.index("--web-dir") + 1])
print("\n".join(sorted(dirs)))
PY
)
  for dir in "${WEB_DIRS[@]}"; do
    dir="$(policy_path "$dir")"
    if [ ! -f "$dir/boards.json" ]; then say "skipped $dir : no boards.json there (not a published web folder)"; continue; fi
    say "cmd: (cd $LIVE_TURING && $COMPILER_PYTHON examples/meowdoku/web_export.py --skip-boards --out $dir)"
    [ "$WHATIF" -eq 1 ] || (cd "$LIVE_TURING" && "$COMPILER_PYTHON" examples/meowdoku/web_export.py --skip-boards --out "$dir")
  done
fi
