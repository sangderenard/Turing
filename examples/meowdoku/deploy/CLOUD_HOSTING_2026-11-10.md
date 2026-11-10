# Cloud hosting: SQL, assets and user content (2026-11-10)

Question: how to prep the SQL and user content for cloud hosting without cross-domain, or how to use publicly hosted assets.
Read-only analysis; nothing was changed. Paths are relative to `examples/meowdoku/` unless prefixed `build/` (= `C:/dev/Powershell/turing/build/meowdoku`).

## 1. Inventory

### SQLite (single local writer, WAL)
| File | Tables | Source |
|---|---|---|
| `build/meowdoku/accounts.db` (53 KB now) | `users`, `sessions`, `owned_assets` | accounts.py:74-93 (`PRAGMA journal_mode=WAL` at :72) |
| same file | `account_roles`, `script_access`, `script_audit` | authz.py:97-113 |
| same file | `lowering_cores`, `lowerings`, `lowering_blobs` (product BYTES), `lowering_leases` | lowering_library.py:34-45 (db opened :75) |
| same file | `service_instances`, `instance_leases`, `account_instances` | deploy/apache_supervisor.py:55-60 (docstring :68 says accounts.db) |
| `build/meowdoku/marble_photos/job-library.sqlite` | `job_owners (digest, owners JSON, created)` | job_library.py:13-14 |

`lowering_blobs.bytes` (lowering_library.py:41-42, insert :185) holds every compiled product file whole; row size equals the .wasm/.wgsl size. It is the only large-blob table. `lowering_leases` uses `INTEGER PRIMARY KEY AUTOINCREMENT` (:43).

### JSON files
- `build/meowdoku/profiles.json`: `{"profiles": {name: {created, games:{plays,best...}}}}`, rewritten whole via tmp+replace (profiles.py:62-76; path server.py:86). Also `profiles.dev.json`.
- `build/meowdoku/access.json` (tier policy, server.py:360), `tipjar.json`, `loading_anim.json/.mla` (server.py:352), `authz_policy.json` (authz.py:75), `deploy/instances.json` (apache_supervisor.py:457), `build/meowdoku/launch*.json`.
- `web/boards.json`, read at startup (server.py:102, :348).

### Blob/file content
- Served static tree `build/meowdoku/web` (317 MB total; 90 .wasm/.wgsl files, about 12 MB of .wasm). The handler is `SimpleHTTPRequestHandler(directory=web_dir)` (server.py:393, :410). It contains .js/.html, `*.wasm`, `*.abi.json`, `*.manifest.json`, WGSL, boards.json, and generated products (server.py:1826-1849 write policy.wasm, law.wasm, pilot.wasm plus abi/weights json).
- User content, directories beside web_dir (server.py:345-367): `marble_objects/<digest>.json` (marble_objects.py:178), `marble_photos/<digest>/{config.json,status.json,images,hdr,orbit.avi}` (marble_photo.py:685-708; 24 MB now), `obj_assets/` (uploaded OBJ/MTL, obj_assets.py:25), `shares/`, `tickets/`, `sessions/`, `canvas/messenger.canvas` (instances.json:27; Nodus canvas with contract port).
- Cache headers today: the loading animation `?v=<digest>` is `Cache-Control: public, max-age=31536000, immutable` (server.py:1119); API answers are `no-store` (server.py:624, :648, :871).

### Per process
Game tables: `self.tables[token] = [Table, ts]` (server.py:384); cookie `meowdoku_session` is per process, hence balancer stickiness (meowdoku-proxy.conf:20-22, :63-71). Also `service_instances.pid/port/state` (host-local).

## 2. Same-origin strategy (no cross-domain)

Facts: the page fetches only relative URLs (`fetch("/api/action")` app.js:4339, `TuringWasmProgram.load("motion.wasm","motion.abi.json")` app.js:7823, `fetch("fountain_purrline.manifest.json")` app.js:6405, `new URL(file, base)` studio_triangle_compute.js:6, `fetchComputeText(rec.wgsl)` compute.js:74-76, :174). No `crossorigin` attribute, no `Access-Control-*`, no `Cross-Origin-*` header and no SharedArrayBuffer use exists in server.py or web/ (the grep hits are comments only: mechanical_creature_studio_bridge.js:193, mechanical_creature_studio_return.js:35, and an origin refusal in client_ray_relay.py:249). A path-mounted asset host is therefore invisible to the page. /api/whoami classifies by origin tier (lan/ip/dns), not by CORS (server.py:628-634, access.py:13-35); assets must not change that origin classification, which Apache preserves via X-Forwarded-For (LAUNCH_2026-11-10.md).

Cookies: `meowdoku_account` is `Path=/; HttpOnly; SameSite=Lax` (+`Secure` over https) (server.py:333, :472-474); the route cookie is `Secure; HttpOnly` (meowdoku-proxy.conf:68). With assets under the same hostname the browser sees first-party requests: no preflight, no credentials-mode problem. The bucket never receives the cookie because Apache makes its own upstream request (add `RequestHeader unset Cookie` for that location anyway).

Apache fragment (inside the existing `<VirtualHost *:443>`, before `ProxyPass /`; more specific paths first). Present in `C:/Apache24/modules` (checked by `ls`): mod_proxy, mod_proxy_http, mod_ssl, mod_headers, mod_expires, mod_cache, mod_cache_disk, mod_socache_shmcb. mod_expires, mod_cache and mod_cache_disk are not yet LoadModule'd (LAUNCH_2026-11-10.md lists only proxy/ssl/headers/rewrite etc.).

```apache
LoadModule expires_module    modules/mod_expires.so
LoadModule cache_module      modules/mod_cache.so
LoadModule cache_disk_module modules/mod_cache_disk.so

# Public, content-addressed products only (see section 3).
ProxyPass        /assets/ https://BUCKET.s3.REGION.amazonaws.com/meowdoku/ retry=0 timeout=30
ProxyPassReverse /assets/ https://BUCKET.s3.REGION.amazonaws.com/meowdoku/
<Location "/assets/">
    ProxyPreserveHost Off        # send the bucket's Host, not 173.195.26.104 (vhost default is On, meowdoku-proxy.conf:51)
    SSLProxyEngine On            # already On vhost-wide (conf:54)
    RequestHeader unset Cookie
    RequestHeader unset Authorization
    Header unset Set-Cookie
    AddType application/wasm .wasm
    ExpiresActive On
    ExpiresDefault "access plus 1 year"
    Header set Cache-Control "public, max-age=31536000, immutable"
    CacheEnable disk
    CacheRoot "C:/dev/Powershell/turing/build/meowdoku/apache_cache"
    CacheDefaultExpire 86400
</Location>
```
Caveat: the vhost pins the backend CA and sets `SSLProxyVerify require` (meowdoku-proxy.conf:54-58). That pin would reject a public bucket certificate. Give the bucket its own verification context (a separate vhost or a CDN origin on a public CA, with system trust) rather than loosening the loopback pin. Not tested here; run `httpd -t` first.
The page then references `/assets/<sha256>.wasm`. The browser must receive `application/wasm` for `.wasm` (stored on the object, or forced by the `AddType` above).

Cost: a cache miss moves bytes bucket -> home Apache -> client, so first fetches cross the home uplink twice (once in from the bucket, once out to the client). Mitigation: `mod_cache_disk` above (digest-named URLs are immutable, so the cache is trivially correct and costs one miss per product per host), `immutable` headers so browsers never revalidate, and optionally a CDN in front of the bucket with Apache as its only client.

## 3. Public assets strategy (cross-origin done right)

If assets are served directly from `https://assets.example.com` instead:
- Bucket/CDN CORS: `Access-Control-Allow-Origin: https://<our origin>` (exact origin; never `*` for anything involving credentials), `Access-Control-Allow-Methods: GET, HEAD`, `Access-Control-Max-Age: 86400`, `Vary: Origin`. Our origin today is `https://173.195.26.104`, whose certificate does not list that IP as a SAN (LAUNCH_2026-11-10.md); use a DNS name first.
- Page changes: relative URLs become absolute via one `ASSET_BASE`; `fetch(url, {mode:"cors", credentials:"omit"})`. `<script type="module" src=...>` is always fetched in CORS mode; a classic `<script>` needs `crossorigin="anonymous"`. web/*.html currently sets none (index.html:350-354 are same-origin modules).
- `WebAssembly.instantiateStreaming` requires `Content-Type: application/wasm` and a CORS-readable response. The Studio loaders use `WebAssembly.instantiate(await verifiedComputeBytes(...))` on an ArrayBuffer (studio_slab_compute.js:8, studio_triangle_compute.js:22), which needs CORS but not the MIME type. `TuringWasmProgram.load` lives in `turing-wasm-shell.js` (app.js:32); check its fetch before moving files it loads.
- Integrity is already in the pipeline: `verifiedComputeBytes` checks sha256 (studio_triangle_compute.js:6) and `fetchComputeText(url, sha256)` rejects wrong bytes (compute.js:74-83). A public bucket cannot silently swap those loads; extend the same check to the remaining `TuringWasmProgram.load` paths before trusting a third-party host.
- WGSL is plain text; `text/plain` works with `response.text()` (compute.js:74-76).
- COOP/COEP: no SharedArrayBuffer or wasm threads are used and no COOP/COEP headers are sent. If ever enabled, every cross-origin asset also needs `Cross-Origin-Resource-Policy: cross-origin` (or CORS plus `crossorigin` attributes).
- Cookies never travel to a different-site asset host (and the account cookie is HttpOnly/Lax). This is the property to keep.

Publishable (public, immutable, no secrets): compiled products `*.wasm`, `*.abi.json`, `*.manifest.json`, WGSL, `policy.weights.json`/`pilot.weights.json`, static .js/.html/css, `boards.json` (game data, server.py:102), fountain manifests, shader/lens catalogs. Prefer digest-named copies (`<sha256>.<ext>`), the same key as `lowering_blobs.product_sha256`.

Never publish: marble photos/HDR/orbit.avi and marble objects/obj_assets (owned per account or browser; photos route through `_job_authorize`, server.py:447-451), `profiles.json`, `accounts.db`/`job-library.sqlite`, and anything under `build/meowdoku` that carries tokens or paths (`admin_token*.txt`, `lichess_token.txt`, `access.json`, `tls/key.pem`, `operator_credentials/`, `launch*.json`, logs, `canvas/*.canvas`, `instances.json`, `sessions/`). A wholesale sync of `build/meowdoku/` to a bucket would leak these; sync an explicit allow-list only.

## 4. SQL in the cloud

Constraints in this schema: one writer process per file (accounts.py:71 `check_same_thread=False` behind a lock; the supervisor, authz CLI and server all open the same `accounts.db`); `BEGIN IMMEDIATE` in job_library.py claim; WAL needs a local filesystem; `service_instances` (pid, port, supervisor_pid) is meaningless off the host and must stay local.

Ranked:
1. **Keep SQLite authoritative at home and replicate with Litestream** to object storage (continuous WAL shipping, point-in-time restore, optional read-only replica elsewhere). No schema or code change; fits the single-writer model; a cloud host would be restore-only. Caveats: test Litestream on Windows; run it per file (`accounts.db`, `job-library.sqlite`). It protects the data; it does not give multi-host writes.
2. **libSQL/Turso or Cloudflare D1**: same SQLite dialect; the DDL uses only TEXT/INTEGER/REAL/BLOB, `CHECK`, `REFERENCES`, so it ports as is. Costs: every query is an HTTP/WebSocket round trip (session validation `user_for_token` runs on each request, so cache it); the supervisor and authz CLI would need the client library; row-size limits make `lowering_blobs.bytes` unsuitable. Only after blobs leave the DB.
3. **Postgres**: scales to multiple writers/app hosts. Translation: `INTEGER PRIMARY KEY AUTOINCREMENT` -> `BIGINT GENERATED ALWAYS AS IDENTITY` (lowering_leases.lease_id; users.id is a plain `INTEGER PRIMARY KEY`); `BLOB` -> `bytea` or (preferred) an object key; `REAL` timestamps -> `DOUBLE PRECISION`; `INSERT OR IGNORE` (lowering_library.py:185) -> `ON CONFLICT DO NOTHING`; `BEGIN IMMEDIATE`/PRAGMA -> transactions plus `SELECT ... FOR UPDATE`; `?` -> `%s`. Largest code change: accounts, authz, lowering_library, job_library and the supervisor each call sqlite3 directly.

Blobs: `lowering_blobs` is content-addressed (PK = sha256) and leases reference it (`lowering_leases.product_sha256`, lowering_library.py:43-44). The DB should hold `product_sha256, kind, byte_count, key/url`, with bytes at `/assets/<sha256>.<kind>` (sections 2-3). `blob_bytes()` (lowering_library.py:190-193) becomes fetch-by-key plus sha256 verification. That shrinks the DB to kilobytes and makes options 2 and 3 practical.

## 5. User content (marble photos, objects, profiles)

Ownership today is `account:<id>` or `browser:<sha256(token)>` (server.py:441-444), recorded in `job_owners` and `owned_assets(asset_type, asset_key, user_id)` (accounts.py:87-93); content is addressed by digest. Two patterns (not choosing):
- **A. Same-origin proxy with server authorization.** Bytes live in a private bucket with no public path. The existing handlers authorize (`_job_authorize` for photos, `for_account(...)` for objects, account checks for obj_assets; server.py:928-966) and then stream the bytes using a server-held bucket credential. Cookies stay first-party, no shareable URL exists, and every byte transits the home host. Closest to today's code.
- **B. Signed URLs.** The handler authorizes, then returns a short-lived presigned URL (or 302). Bytes bypass the home link, but this is cross-origin: bucket CORS (`Allow-Origin` = our origin), `credentials:"omit"`, URL expiry, `Cache-Control: private`, and a leaked URL works until expiry. Anonymous browser-hash ownership has no durable identity, so for those items the URL is only as strong as the device token.

Profiles (`profiles.json`) are small mutable JSON, rewritten whole on every save (profiles.py:76): keep local or move to a DB row; do not put them in an eventually-consistent bucket.

## 6. Plan and non-recommendations

1. Name the origin: a DNS name with a certificate whose SAN matches (the public IP is not in the current SANs).
2. Define the publish allow-list (section 3) and push digest-named copies to a bucket; leave user content and `build/meowdoku` secrets out.
3. Add the `/assets/` ProxyPass plus mod_cache_disk (section 2) to the existing vhost; point a few relative fetches (`.wasm`, WGSL) at `/assets/<sha>` behind a config flag. Same origin, so no CORS work.
4. Move `lowering_blobs.bytes` to the bucket (DB keeps sha256 and key); `blob_bytes` verifies the digest.
5. Turn on Litestream for `accounts.db` and `job-library.sqlite`; perform one restore drill.
6. Decide user-content pattern A or B for marble photos/objects and implement it behind the existing authorization functions.
7. Only then consider libSQL/D1 or Postgres, and only if a second writer host is actually required.

Not recommended:
- Publishing the DB file, or syncing `build/meowdoku/` wholesale (tokens, TLS key, operator secrets, user photos).
- Cross-origin cookies (`SameSite=None`) to reach a separate app/API host; the account cookie is HttpOnly/Lax by design.
- `Access-Control-Allow-Origin: *` (or reflecting Origin with credentials) on anything authenticated or user-owned.
- Putting `service_instances`/`account_instances` (host-local pids/ports) in a shared cloud DB.
- Running two writers against one SQLite file over a network share or an object-store mount.
