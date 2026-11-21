#!/usr/bin/env bash
# backup-state.sh -- Backup-State.ps1 for Linux: a snapshot of the state root (state_backup.py snapshot). Safe while the
# servers are live: databases go through SQLite's backup API, then PRAGMA integrity_check. Secrets only with
# --include-secrets.
#   backup-state.sh [--out DIR] [--keep N] [--include-secrets] [--include-logs] [--what-if]
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

OUT="${SPEAKTOME_BACKUPS:-$SPEAKTOME_ROOT/srv/backups}"
KEEP=14
EXTRA=()
WHATIF=0
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --keep) KEEP="$2"; shift 2 ;;
    --include-secrets) EXTRA+=(--include-secrets); shift ;;
    --include-logs) EXTRA+=(--include-logs); shift ;;
    --what-if) WHATIF=1; shift ;;
    *) refuse "unknown argument $1" ;;
  esac
done

gate "$(basename "$0")"
[ -d "$STATE_DIR" ] || refuse "no state dir at $STATE_DIR: set MEOWDOKU_STATE_DIR."
CMD=("$PYTHON" "$TURING/examples/meowdoku/state_backup.py" snapshot --state-dir "$STATE_DIR" --out "$OUT" --keep "$KEEP" "${EXTRA[@]}")
say "cmd: ${CMD[*]}"
[ "$WHATIF" -eq 0 ] || exit 0
mkdir -p "$OUT"
"${CMD[@]}"
