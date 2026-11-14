# Per-user Suno MCP forwarding

The Connected services UI uses `/api/connections/suno/`. The authenticated site
account owns its connection; no body field can select another owner. GET status
and tools, POST connect/call/disconnect. Mutations use the existing Keys
same-origin plus `X-Meowdoku-Secret: 1` JSON boundary. Call accepts
`{"name":"<actual remote tool name>","arguments":{}}`; the remote catalogue
and MCP results retain their full schema and content, not a fabricated menu.

## Operator deployment (not performed by this change)

1. Install SpeakToMe's `remote-mcp` optional dependency group using repository
   setup into the service interpreter. This is separate from its stdlib incoming
   MCP adapter. The host lazily imports it so absent dependencies do not break
   other routes.
2. Deploy the tracked Apache path-only `meowdoku_safe` access-log format. Ensure
   all upstream proxies also omit callback queries and Referer. Never enable
   OAuth while a proxy records callback query strings. Do not enable HTTP debug
   logging. The application suppresses access logs for connection routes.
3. Create operator-owned `<StatePaths.run_dir>/mcp_connections.json`:

   ```json
   {"suno":{"callback_url":"https://YOUR_PUBLIC_HOST/api/connections/suno/callback",
             "allowed_origins":["https://api.suno.com"]}}
   ```

   Use the actual public HTTPS origin including its port. This is configuration,
   not a credential. Add only verified issuer origins discovered from the
   provider; never arbitrary browser-supplied URLs. The SDK discovers OAuth
   metadata and registration. An unapproved issuer is refused rather than guessed.
4. Retain existing Apache session stickiness. Pending authorization lives in one
   worker; restart or switching workers invalidates it and requires Connect again.
5. Each user clicks Connect and signs in at Suno themselves. The callback requires
   the same site account, exact single-use state, then redirects to a clean URL.

Tokens/client records use the existing restricted StatePaths secrets tree:
`user_credentials/<account id>/mcp-suno.json`, atomic replace, no static route,
no account list/download API. Existing Windows secrets-directory ACL inheritance
must remain restricted. POSIX newly created records are mode 0600. Disconnect
cancels pending work and erases this site's stored credentials; provider-side
revocation remains in Suno's account settings.

The transport checks exact configured HTTPS origins, resolves public-only IPs,
pins its chosen address and keeps original TLS hostname verification. No proxies
or environment-provided redirects. Stored authorization is not a health claim.
Missing dependencies/configuration returns 503; provider failure returns a generic
502 without OAuth secrets. Provider tools may incur credit costs; invocation is an
explicit user action, never an automatic radio-generation job.

## Verification and limits

Seven host tests cover owner isolation/guest refusal, CSRF, body owner injection,
lossless call result, callback clean redirect/refusal, route/method rejection,
atomic isolated vault storage and source routing/logging integration. SDK lifecycle
and transport are tested in SpeakToMe. No live Suno login, actual returned menu,
service restart, live Apache modification or radio library import is claimed.

## Windows candidate receipt

The existing `Provision-ServicePython.ps1` supports a separate `-Root` and
`-Requirements`. `requirements-mcp-runtime-validation.txt` is the pinned minimal
SDK-only validation manifest. It was provisioned into
`srv/candidates/mcp-sdk-20261114/python` with Python 3.11.18; `pip check` and the
nine SpeakToMe SDK tests passed. The SDK itself supports Python >=3.10; the
SpeakToMe project's >=3.12 requirement is not an SDK requirement.

Eight host tests now also pass under that candidate, including the real host
factory, SDK, disk vault, browser-code callback and full tool forwarding with
only the HTTP transport mocked. No public provider interaction is claimed.
The minimal candidate is NOT a full service interpreter replacement: scientific
service dependencies are absent. Never switch the live service to it as-is.
