# lib.sh -- what every Linux deploy script shares: the workspace layout, the service Python, and the operator gate.
# Sourced, never run. The bash scripts beside it mirror deploy/*.ps1 and call the same Python tools; only the shell
# around them differs. Owner, 2026-11-21: "Our deploy scripts will need bash versions too".
#
#   SPEAKTOME_ROOT       the workspace (bootstrap_linux.sh's --root), default /srv/powershell
#   MEOWDOKU_STATE_DIR   the state root (sql/, secrets/, content/, ...), default $SPEAKTOME_ROOT/srv/state
#   SPEAKTOME_PYTHON     the service interpreter, default $SPEAKTOME_ROOT/srv/service-venv/bin/python
#   MEOWDOKU_OPERATOR    the operator account the gate checks; its password file defaults to
#                        $MEOWDOKU_STATE_DIR/secrets/operator_credentials/<name>.secret
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TURING="$(cd "$DEPLOY_DIR/../../.." && pwd)"
SPEAKTOME_ROOT="${SPEAKTOME_ROOT:-/srv/powershell}"
STATE_DIR="${MEOWDOKU_STATE_DIR:-$SPEAKTOME_ROOT/srv/state}"
PYTHON="${SPEAKTOME_PYTHON:-$SPEAKTOME_ROOT/srv/service-venv/bin/python}"
export PYTHONUTF8=1
export GIT_TERMINAL_PROMPT=0        # a sign-in prompt nobody sees must fail, not wait

say() { printf '%s\n' "$*"; }
refuse() { printf 'REFUSED %s\n' "$*" >&2; exit 2; }

# The operator gate, exactly as deploy/*.ps1 run it: authz.py script-auth (the account must verify, be in script_access
# and hold deploy_scripts; every attempt is a script_audit row). Not skippable; --what-if is gated too.
gate() {
  local script="$1" user="${MEOWDOKU_OPERATOR:-}" password_file="${MEOWDOKU_PASSWORD_FILE:-}"
  local db="${MEOWDOKU_ACCOUNTS_DB:-$STATE_DIR/sql/accounts.db}"
  [ -n "$user" ] || refuse "no operator: set MEOWDOKU_OPERATOR (and MEOWDOKU_PASSWORD_FILE)."
  if [ -z "$password_file" ] && [ -f "$STATE_DIR/secrets/operator_credentials/$user.secret" ]; then
    password_file="$STATE_DIR/secrets/operator_credentials/$user.secret"
  fi
  local args=("$TURING/examples/meowdoku/authz.py" script-auth "$script" --user "$user" --db "$db")
  [ -z "$password_file" ] || args+=(--password-file "$password_file")
  "$PYTHON" "${args[@]}"
}

# A branch_policy.json path names the Windows workspace; on Linux the same repository sits under SPEAKTOME_ROOT.
policy_path() {
  local windows="$1"
  case "$windows" in
    C:/dev/Powershell) printf '%s\n' "$SPEAKTOME_ROOT" ;;
    C:/dev/Powershell/*) printf '%s/%s\n' "$SPEAKTOME_ROOT" "${windows#C:/dev/Powershell/}" ;;
    *) printf '%s\n' "$windows" ;;
  esac
}

# Is anything listening on these TCP ports? Prints the busy ones.
listening() {
  local busy=() port
  for port in "$@"; do
    if ss -ltnH "sport = :$port" | grep -q .; then busy+=("$port"); fi
  done
  printf '%s\n' "${busy[*]:-}"
}
