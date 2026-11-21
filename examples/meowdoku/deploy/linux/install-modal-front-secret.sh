#!/usr/bin/env bash
# install-modal-front-secret.sh -- write the Modal front token into Apache's root-only definitions, from stdin.
# Run on the Linux front host as root. The token is generated once at home (<state>/secrets/modal/front.token) and the
# same value is the Modal secret speaktome-front (SPEAKTOME_FRONT_TOKEN). From home:
#   tr -d '\r\n' < <state>/secrets/modal/front.token | ssh <host> 'sudo bash <turing>/examples/meowdoku/deploy/linux/install-modal-front-secret.sh'
# The token is read here, by this shell, so nothing expands it early (review 1c522e0f); it never lands in a temp file.
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "REFUSED: run as root" >&2; exit 2; }
CONF=/etc/apache2/conf-available/speaktome-secrets.conf
IFS= read -r token || true
case "$token" in
  ""|*[!A-Za-z0-9_-]*) echo "REFUSED: no token on stdin, or not a url-safe token" >&2; exit 2 ;;
esac
umask 077
tmp="$(mktemp "$CONF.XXXXXX")"
printf '# Secrets for oracle-front.conf; root-only. Written by install-modal-front-secret.sh.\nDefine SPEAKTOME_MODAL_FRONT_TOKEN "%s"\n' "$token" > "$tmp"
chmod 600 "$tmp"
mv "$tmp" "$CONF"
a2enconf -q speaktome-secrets
apache2ctl configtest
systemctl reload apache2
echo "installed: $CONF (${#token} characters); apache2 $(systemctl is-active apache2)"
