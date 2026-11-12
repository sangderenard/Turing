2026-11-11 PER-CHAT IMAGE OPT-IN (Windows, source-only, Turing and SpeakToMe):

The existing messaging writer stores bounded PNG attachments atomically with
messages and sender idempotency receipts. Both participants must explicitly
enable images for the pair before metadata includes a URL or image_bytes returns
data. Authenticated image fetch rechecks membership, mutual contact, consent and
removal. No public/static asset path exists. Blocking/removing/recreating a
relationship resets consent. Ordinary text message shapes are unchanged.

Images: 2 MiB, 4096 per dimension, 16 megapixels, fully decoded single-frame PNG.
Stored quotas: 32 MiB per sender and 256 MiB total. Quotas refuse, not delete.
The same existing message is refreshed after unlock, including older loaded
pages. Clients preserve image bytes and operation identity through retry.

Measured on final source:
- Messenger + image + client + dependency + authz/proxy/server/public-login
  focused gate: 93 passed, one existing imp deprecation warning, 54.95 s.
- SpeakToMe shared-route unit gate: 12 passed.
- Real Chrome disposable HTTPS fixture: 10 passed. Locked message triggers no
  image request; same message reveals decoded PNG after opt-in; disable hides;
  actual file chooser uploads PNG; delayed Back/Close and reopen are covered.
- Scoped diff checks pass; SpeakToMe guestbook filename validator passes.

The existing REST binding is now API_VERSION 3. Turing rest_api.py/messaging.py
and sibling SpeakToMe routes.py must deploy together. Concurrent admin-control
and Studio changes were preserved. This was not a full repository test suite.

NOT LIVE: two authenticated-read commands were denied by execution network
policy before running, first against the unspecified front-door port, then once
at verified https://stichka.duckdns.org:8766 after exact authorization evidence.
No alternate route was attempted. No live restart, image enable or image send
occurred. Claude's current consent and receipt are unknown. The requested PNG
was verified locally at build/dot-phong-capture-20261111/framed.png (478039 bytes).
