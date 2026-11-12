# Apache-loaded shared site controls

The HTTPS virtual host in `meowdoku-proxy.conf` inserts the existing
`/site-nav.js` ES module, with `data-site-nav="allow"`, immediately before the
closing body tag of ordinary HTML documents. It does not implement another
header or another login/ticket system. The existing module owns those controls.

The fragment loads Apache's installed `mod_filter` and `mod_substitute` only
when not already loaded. No edit to `httpd.conf` is necessary. The filter does
not change the proxy mapping, admin exclusions, TLS settings, account authority,
route cookies, or authentication headers.

## Scope and identity

- Response media type must be `text/html`, with optional charset parameters.
- `/api`, `/api/...`, `/ws`, and `/ws/...` are excluded regardless of media type.
- Encoded responses and responses with Content-Disposition are unchanged.
- Apache's filter harness runs this filter for HTTP 200, not errors or partial
  responses. The changed-body protocol removes the original content length.
- Current backend HTML is uncompressed. If HTML compression is introduced,
  arrange it after substitution; do not substitute inside compressed bytes.
- Documents need a closing body tag. This does not manufacture wrappers for
  fragments or alter their content structure.
- Authored tags remain for direct-backend access. An injected tag uses the same
  exact URL, with no query string or wrapper: native ES module loading evaluates
  it once per document even when multiple tags request it. This preserves
  imports of `registerControlSuite` from that same module instance.

## Publish and activate

1. Publish the current existing `site-nav.js` and its requested ticket assets
   using the established asset deployment workflow. Confirm the SQL issue API
   is available through the existing backend; this filter creates no API.
2. Coordinate with other active workers before touching live Apache. Its piped
   logger is the application supervisor, so an Apache lifecycle operation can
   also load shared backend changes. Follow `../../HOW_TO_START.md` and the
   established supervisor lifecycle procedure. Check active exposures first.
3. Back up the current `C:/Apache24/conf/meowdoku-proxy.conf`, then deploy this
   repository's fragment to that path. Do not change the primary httpd.conf,
   firewall, certificates, or backend routing as part of this task.
4. Run `C:/Apache24/bin/httpd.exe -t`. Activate only in the coordinated lifecycle
   window, not by independently stopping, killing, or launching a live process.
5. Verify a trusted HTTPS HTML response through the actual Apache frontend
   contains the tag. In an existing signed-in browser, confirm one shared
   control panel, the retained account/server, and the existing Tickets/Report
   issue controls. Verify Studio, login, and a generated `/site/...` document.
   Source and isolated checks do not establish live activation or public reach.

## Reproducible isolated validation

From the repository root in PowerShell:

```powershell
$env:MEOWDOKU_TEST_HTTPD='C:/Apache24/bin/httpd.exe'
python -m pytest tests/test_meowdoku_apache_site_nav.py -q --tb=short
```

The tests extract the exact filter block from the repository fragment and run
an isolated Apache against a temporary loopback HTTP fixture. They do not read
sessions, call the live backend, run the production supervisor, or modify live
configuration. The fixture verifies HTML injection, identical existing module
URLs, unchanged excluded responses, and retained cookie/authorization headers.
The duplicate-tag check verifies identity, not browser rendering; check the
single mounted header in the coordinated live browser verification.

On 2026-11-11 the 12 fixture cases passed, and `httpd -t` using the full installed
configuration with only its fragment include redirected to the repository copy
reported Syntax OK. The staged validation config is
`build/dot-apache-header-20261111/httpd-staged.conf`. Live configuration and the
running service were not changed by that validation.
