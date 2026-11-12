2026-11-11 MESSENGER IMAGE CLIENT (source only, Windows):

Extended existing messenger widget and agent_messenger CLI for per-chat image
opt-in and bounded PNG attachments. Preserved existing retry handles and
idempotency keys, now including image bytes. Widget refreshes already-loaded
older messages in place when either opt-in changes. Plain text bodies remain
plain text; only exact same-origin server image URLs are rendered.

Measured: python -m pytest tests/test_meowdoku_messenger_client.py -q:
24 passed in 0.55 seconds, including Node isolated send/API/image/reveal tests.
Node module syntax check and scoped git diff --check passed. Python compilation
passed. Tests use fake transport and isolated UI functions, not a full browser.
No deployment, backend route validation, or production image delivery is claimed
by this client gate. Pillow is required only for CLI PNG validation.

Follow-up actual Chrome test: tests/meowdoku_widget_images_browser_check.py passed
10/10 checks using a disposable HTTPS server and two disposable local accounts.
Verified no image request while locked, original-id reveal with decoded PNG,
opt-out hide, actual chooser upload, delayed Back/Close response suppression, and
reopen. Local certificate trusted only by fixture Python client and browser SPKI
pin, without modifying installed trust or production network settings. Screenshot
build/messenger-image-browser.png was visually inspected. Browser process and
fixture state were cleaned up. No public-host calls or production delivery.
