#!/usr/bin/env bash
# Regression gate: four pytest parts, each compared with its committed
# baseline of known failures.  Prints only NEW failures per part.
#
#   tools/gate/run_gate.sh <worktree> <logdir>
#
# Run it on a clean worktree of the commit under test (never a stash or a
# checkout of a path; see TEST_BASELINE_AND_HAZARDS.md).  Collect part ab
# once first: it builds the cffi module the other parts import, and building
# it in parallel races.  Part aa takes about 20 minutes; the others 2-4.
set -u
tree=$(cd "$1" && pwd); logs=$(mkdir -p "$2" && cd "$2" && pwd)
here=$(cd "$(dirname "$0")" && pwd)
# Windows parity receipt (2026-10-24): Git Bash exposes the repository's
# Python 3 interpreter as `python`; `python3` was absent, so every gate part
# stopped in `timeout` before pytest collection.  Use the same sanctioned
# executable spelling as the repository's documented commands.
(cd "$tree" && python -m pytest $(cat "$here/part_ab.txt") -q --collect-only -p no:cacheprovider >/dev/null 2>&1)
for part in aa ab ac rec; do
  (cd "$tree" && timeout 5400 python -m pytest $(cat "$here/part_$part.txt") \
     -q --tb=line -p no:cacheprovider --timeout 900 -rf > "$logs/part_$part.log" 2>&1) &
done
wait
for part in aa ab ac rec; do
  echo "== $part: $(grep -E '[0-9]+ (passed|failed)' "$logs/part_$part.log" | tail -1)"
  comm -23 <(grep '^FAILED' "$logs/part_$part.log" | sed 's/ - .*//;s/^FAILED //' | sort) \
           <(sort "$here/baseline_$part.txt") | sed 's/^/NEW /'
done
