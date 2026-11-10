# Apache front for Meowdoku (2026-11-10)

**Nothing here was applied.** No change was made to C:/Apache24, the Windows firewall, any router, or any TLS setting. These are
files in a worktree and steps for the operator.

Files: `deploy/apache/meowdoku-proxy.conf` (vhost, balancer variant and single-instance variant), `proxy_trust.py` (client origin
from a trusted proxy), `tests/test_meowdoku_proxy_trust.py`.

## Operator steps
1. In C:/Apache24/conf/httpd.conf uncomment the LoadModule lines listed at the top of the conf (proxy, proxy_http, proxy_wstunnel,
   proxy_balancer, lbmethod_byrequests, slotmem_shm, ssl, headers, rewrite). All the .so files exist in C:/Apache24/modules
   (Apache 2.4.58, which supports `ProxyPass ... upgrade=websocket`). Add `Listen 443`.
2. Add the one line `Include "C:/dev/Powershell/turing/examples/meowdoku/deploy/apache/meowdoku-proxy.conf"`.
3. Edit ServerName and point SSLCertificateFile/KeyFile at the public certificate (Apache now terminates TLS; the instance's
   self-signed cert is only the backend hop).
4. Firewall/routers: open 443 to Apache instead of 8766 to python (the 8766 rule and forwards are no longer needed publicly). Do this
   by the documented manual procedure in HOW_TO_START.md; do not change it silently.
5. `httpd -t`, then start httpd.

## Instances (loopback only behind the proxy)
```
python examples\meowdoku\server.py --port 8766 --bind 127.0.0.1 --tls-cert build\meowdoku\tls\cert.pem --tls-key build\meowdoku\tls\key.pem
python examples\meowdoku\server.py --port 8768 --bind 127.0.0.1 --tls-cert build\meowdoku\tls\cert.pem --tls-key build\meowdoku\tls\key.pem
```
(plus the existing flags such as `--retain-fountain-products`). Uncomment the 8768 BalancerMember to use two. A game table lives in
one process, hence sticky routing (`MEOWDOKU_ROUTE|meowdoku_route`, route a/b).

## What changes in the server (pending in the live tree)
- `access.classify` receives `client_ip` from `proxy_trust.client_origin` instead of the socket peer when `via_proxy`.
  Without this every proxied visitor looks like loopback, i.e. tier `lan` -> `edit`. **Do not enable the proxy before this lands.**
- The Secure-cookie decision in server.py uses the `scheme` from `client_origin` (forwarded proto honoured only from a trusted peer).
- Optionally instances set `route_cookie(route)` themselves; `build/meowdoku/proxy_trust.json` may list more trusted proxies.

## What stays
Admin stays loopback-direct and NOT through Apache: `lifecycle.admin_authorize` and the loading-library owner path keep refusing any
request carrying forwarding headers; the vhost also excludes `/api/admin` from ProxyPass. Use `https://127.0.0.1:<port>` for admin.

## Verification checklist
- `curl -kI https://<host>/` returns 200 and the page's cookies carry `Secure`; instance log shows the real client IP, not 127.0.0.1.
- Spoof test: `curl -k -H "X-Forwarded-For: 192.168.1.5" https://<host>/api/whoami` still reports the true remote origin (Apache appends
  the real address; the rightmost untrusted hop wins).
- Direct to the instance from a non-loopback address with a spoofed header is ignored (instance bound to 127.0.0.1 anyway).
- WebSocket: `/ws` upgrade through Apache succeeds (101), e.g. via the browser game page or `websocat wss://<host>/ws`; `/ws/spectate` likewise.
- Sticky: response carries `Set-Cookie: MEOWDOKU_ROUTE=.a` and repeated requests with it reach the same instance.
- `curl -k https://<host>/api/admin/lifecycle` is not served via Apache.
