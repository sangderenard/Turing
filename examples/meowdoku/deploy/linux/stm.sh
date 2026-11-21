#!/usr/bin/env bash
# stm.sh -- the operator script (stm.ps1) for a Linux host: one entry for the deploy verbs that exist here.
#   stm.sh status            listeners on the front and service ports
#   stm.sh backup [...]      snapshot the state root (backup-state.sh; safe while live)
#   stm.sh publish [...]     move live checkouts to what landed, re-export the pages if they moved (publish.sh)
#   stm.sh restore [...]     restore a snapshot into empty state (restore-state.sh; dry run unless --apply)
#   stm.sh front             reload Apache after checking its configuration (oracle-front.conf). A graceful reload
#                            also restarts Apache's piped log programs, so the stack's instances restart with it.
#   stm.sh restart|stop|start  the whole stack: Apache owns it (deploy/apache/oracle-stack.conf), so this is the
#                            apache2 service; a stop closes the supervisor's stdin and it stops every instance gracefully
# Every verb runs the operator gate (MEOWDOKU_OPERATOR).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
verb="${1:-help}"
[ $# -eq 0 ] || shift
case "$verb" in
  status)  exec bash "$HERE/status.sh" "$@" ;;
  backup)  exec bash "$HERE/backup-state.sh" "$@" ;;
  publish) exec bash "$HERE/publish.sh" "$@" ;;
  restore) exec bash "$HERE/restore-state.sh" "$@" ;;
  front)
    source "$HERE/lib.sh"
    gate "stm.sh front"
    sudo apache2ctl configtest && sudo systemctl reload apache2 && say "Apache reloaded."
    ;;
  restart|stop|start)
    source "$HERE/lib.sh"
    gate "stm.sh $verb"
    [ "$verb" = stop ] || sudo apache2ctl configtest
    sudo systemctl "$verb" apache2 && say "stack $verb: apache2 $(systemctl is-active apache2)"
    ;;
  help|*)  sed -n '2,11p' "$0"; [ "$verb" = help ] || exit 2 ;;
esac
