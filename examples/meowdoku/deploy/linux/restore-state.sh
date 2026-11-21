#!/usr/bin/env bash
# restore-state.sh -- Restore-State.ps1 for Linux: restore a snapshot into an empty / sql-free state root
# (state_backup.py restore). Never overwrites existing state. Refuses while 8766 / 8768 / 47611 listen. Dry run unless
# --apply. This is also the accounts-database move: snapshot at home with --include-secrets, copy it here over ssh,
# restore it --into-empty.
#   restore-state.sh --snapshot DIR [--apply] [--into-empty] [--what-if]
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

SNAPSHOT=""
APPLY=0
INTO_EMPTY=()
WHATIF=0
while [ $# -gt 0 ]; do
  case "$1" in
    --snapshot) SNAPSHOT="$2"; shift 2 ;;
    --apply) APPLY=1; shift ;;
    --into-empty) INTO_EMPTY=(--into-empty); shift ;;
    --what-if) WHATIF=1; shift ;;
    *) refuse "unknown argument $1" ;;
  esac
done
[ -n "$SNAPSHOT" ] || refuse "pass --snapshot DIR."

gate "$(basename "$0")"
busy="$(listening 8766 8768 47611)"
[ -z "$busy" ] || refuse "servers are listening on $busy. Stop them first."
CMD=("$PYTHON" "$TURING/examples/meowdoku/state_backup.py" restore --snapshot "$SNAPSHOT" --state-dir "$STATE_DIR"
     "${INTO_EMPTY[@]}")
{ [ "$APPLY" -eq 1 ] && [ "$WHATIF" -eq 0 ]; } || CMD+=(--dry-run)
say "cmd: ${CMD[*]}"
[ "$WHATIF" -eq 0 ] || exit 0
"${CMD[@]}"
[ "$APPLY" -eq 1 ] || say "Dry run only; pass --apply to restore."
