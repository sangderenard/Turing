#!/usr/bin/env bash
# status.sh -- Status.ps1 for Linux: listeners on the front and service ports with owning PID and process. Read-only.
#   status.sh [--what-if]
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

[ "${1:-}" != "--what-if" ] || { gate "$(basename "$0")"; say "cmd: ss -ltnp (80 443 8766 8767 8768 47611 47616)"; exit 0; }
gate "$(basename "$0")"
printf '%-7s %-24s %s\n' PORT ADDRESS PROCESS
for port in 80 443 8766 8767 8768 47611 47616; do
  # ss needs root to name another user's process; sudo -n keeps it non-interactive and falls back to the plain view.
  line="$( (sudo -n ss -ltnpH "sport = :$port" 2>/dev/null || ss -ltnH "sport = :$port") | head -1)"
  if [ -z "$line" ]; then
    printf '%-7s %s\n' "$port" "(not listening)"
  else
    address="$(awk '{print $4}' <<<"$line")"
    process="$(grep -o 'users:((.*))' <<<"$line" || true)"
    printf '%-7s %-24s %s\n' "$port" "$address" "${process:-?}"
  fi
done
