# Nodus Radio menu and relay view

The shared menu now offers Nodus Radio to view_site accounts. It lazily polls
GET /api/radio at 10 Hz only while open, displays NRAD v1 bands/RMS and marks
stale signals. Collapse, Escape, outside click, hidden document and unload stop
polling. Late responses cannot restart a dismissed panel. No browser FFT or
fake audio player is added: NRAD v1 is spectral telemetry, not PCM.

The endpoint reads the existing native UDP router's bounded relay inbox using
a server-controlled station configuration at StatePaths.radio_json, never an
arbitrary browser-selected relay address. The owner/open address and exact
sender are validated, and only enabled source=file stations are exposed.
No system-output broadcast is enabled. No new listener, registry or SQL table.

Deployment configuration, in the instance's untracked run/nodus_radio.json:
{ "enabled": true, "source": "file", "address": <allocated relay address>,
  "owner_id": <actual owning account>, "peer": "<native sender IP:port>",
  "title": "<authored selection>" }
Create the address through the authenticated existing relay API; never insert
account ownership by hand. Native sender must target that allocated address.
Configure the exact sender endpoint from its launch record. The config is not
accepted from an unauthenticated caller. UDP peer matching is not encryption
or authentication; use the controlled local network only.

Verified: 3 unittest cases for exact packet layout and 64-bit frame precision,
invalid/truncated/nonfinite/out-of-range packets, bounded existing relay
conglomerates, missing config, stale signal, sender exclusion, system-output
exclusion and owner mismatch. Real header offline lifecycle check passed for
Radio, plus existing site-control suite and Node syntax checks. No live producer
or HTTP round-trip verified here; native file-source work is separate.

The approved service login/review-checkout attempt was blocked by execution
network policy before an HTTP response. No bypass, SQL write, live deployment,
restart, audio recording or optical rendering was performed. Source changes
remain proposed until the existing review/publish and coordinated restart.
