# Nodus Radio explicit server-speaker control

## Scope

The existing header's Nodus Radio spectrum remains an inert viewer. Administrators
(`account_manage`, with an actual signed-in account) additionally see **Play on
server speakers** and **Stop my server playback**. This is PA playback on the
server's configured device, NOT browser playback. Opening/closing the panel,
loading the site, or reading status never starts a device. Closing the panel does
not stop an explicitly started subscription; Stop or source EOF does.

GET `/api/radio/playback` returns configured/status/owner/job/error/can_stop.
POST with `Content-Type: application/json` and `X-Meowdoku-Radio: 1` accepts only
`{"action":"start"}` or `{"action":"stop"}`. Existing session/capability checks,
same-origin checks and bounded body parsing apply. Only the starting account can
stop its active stream through this endpoint. No credentials, peer addresses,
paths, devices or executable names are accepted from request bodies.

The manager uses the existing service_jobs Registry with kind `radio_pa` and
owner `account:<id>`; failed registration prevents starting. A worker owns one
existing Subscription and one native Playback process. Shutdown cancels that
worker. Source BYE drains naturally. No producer is started by these buttons.

## Operator configuration, disabled by default

The existing StatePaths `radio_json` file (run/nodus_radio.json, NOT a repo file)
must already select `enabled: true` and `source: "file"`. Add a `playback` object
only when intentionally configuring server PA:

- `enabled`: boolean, false until the operator chooses to enable this control.
- `router_port`: existing private native router control port.
- `listen_port`: dedicated loopback UDP subscription port reserved by the operator.
- `peer_host`: currently must be `127.0.0.1`.
- `peer_port`: actual native producer's pinned source port.
- `ssrc`: actual producer RTP stream ID, unsigned 32-bit.
- `device_id`: explicitly selected native SDL output index (0 is default).
- `executable`: absolute path to the built nodus_pcm_playback executable.

Do not reuse a port owned by the SQL relay daemon: it already owns those
subscriptions. This patch does not pretend to allocate SQL relay addresses or
create a second registry. It consumes the existing configured native router and
private dedicated port. No new public listener, firewall change, automatic source,
automatic sound, or browser TCP handoff is introduced.

## Verification and limits

24 Python radio tests pass, including inert GET/construction, ownership conflict,
cross-account stop refusal, registration failure, cancellation during device start,
invalid/remote configuration, authentication/CSRF/body gates and cleanup failures.
Existing member header lifecycle test passes; admin mode adds inert opening,
explicit start/stop, correct control header and ownership-sensitive controls.
server.py parses with Python AST; git diff --check passes.

Earlier native SDL-dummy pipeline tests remain the evidence for actual PCM,
bounded slew/queue pacing and RTCP EOF. This increment's control tests use fake
subscription/device boundaries: they do not prove real speakers or a deployed
HTTP session. No audible tests, live configuration changes, server restart, push
or deployment performed. Prior explicit network-policy denial still prevents the
previous live service/Git workflow; do not bypass it. Canonical server has concurrent
work and must be reconciled through the authorized coordination path before release.
