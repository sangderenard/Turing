2026-11-11 APACHE SHARED SITE NAVIGATION (MEASURED isolated Windows Apache; NOT ACTIVATED):

The canonical examples/meowdoku/deploy/apache/meowdoku-proxy.conf now injects
an opt-in tag for the EXISTING /site-nav.js ES module into text/html documents.
Standard mod_filter/mod_substitute are guarded-loaded by the fragment. Same-URL
module identity preserves the authored page imports and avoids a second module
execution when an authored opt-in script is already present. No replacement
header, auth flow, ticket system, backend, TLS or route mapping was created.

Measured: 12 tests in tests/test_meowdoku_apache_site_nav.py passed in 1.67 s
against a separately launched Apache 2.4 instance on a temporary loopback port.
HTML and generated-page paths receive the tag; API/WS, Wasm, JSON, encoded HTML,
download HTML, 403 and 206 bodies remain unchanged. Cookie and Authorization
headers pass through. Full installed httpd.conf with only the fragment include
redirected to the repository copy returns Syntax OK (existing global ServerName
warning remains). No production service, configuration or session was changed.

Limits: documents require a closing body tag. Current backend HTML is not
compressed; future compression must occur after substitution. Browser module
identity is standard behavior, while actual one-header rendering still requires
live-browser verification after coordinated activation. Direct-backend pages
continue using their existing authored tags. The source header and deployed
header were different versions when parent began; asset publication belongs to
the parent task. Coordinate before Apache lifecycle because its piped logger
owns the backend supervisor and other workers have shared backend edits.
Deployment notes: examples/meowdoku/deploy/apache/SITE_NAV.md.
