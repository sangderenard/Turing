#!/usr/bin/env bash
# stm.sh -- the operator script (stm.ps1) for a Linux host: one entry for the deploy verbs that exist here.
#   stm.sh status            listeners on the front and service ports
#   stm.sh backup [...]      snapshot the state root (backup-state.sh; safe while live)
#   stm.sh publish [...]     move live checkouts to what landed, re-export the pages if they moved (publish.sh)
#   stm.sh restore [...]     restore a snapshot into empty state (restore-state.sh; dry run unless --apply)
#   stm.sh front             install this checkout's oracle-front.conf if it changed, check, reload Apache. A graceful
#                            reload also restarts Apache's piped log programs, so the stack's instances restart with it.
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
    # Install this checkout's deploy/apache/oracle-front.conf as the served front when it differs, keep the old one
    # beside it (.previous), and put it back if the configuration test fails, so a bad front never reaches the reload.
    # oracle-stack.conf is NOT installed here: it names the state root, and moving that is the owner's step.
    source "$HERE/lib.sh"
    gate "stm.sh front"
    src="$DEPLOY_DIR/apache/oracle-front.conf"
    dst=/etc/apache2/sites-available/oracle-front.conf
    if ! sudo cmp -s "$src" "$dst"; then
      sudo cp "$dst" "$dst.previous"
      sudo cp "$src" "$dst"
      say "installed $src"
      if ! sudo apache2ctl configtest; then
        sudo cp "$dst.previous" "$dst"; refuse "the new front fails configtest; the previous one is back, Apache untouched."
      fi
    fi
    sudo apache2ctl configtest && sudo systemctl reload apache2 && say "Apache reloaded (the stack's instances restart with it)."
    ;;
  restart|stop|start)
    source "$HERE/lib.sh"
    gate "stm.sh $verb"
    [ "$verb" = stop ] || sudo apache2ctl configtest
    sudo systemctl "$verb" apache2 && say "stack $verb: apache2 $(systemctl is-active apache2)"
    ;;
  help|*)  sed -n '2,11p' "$0"; [ "$verb" = help ] || exit 2 ;;
esac
