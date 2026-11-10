# Local Studio object and live PNG downloads

Added only `web/marble_downloads.js` and its focused Node test. Integration with
Studio snapshot/autosave controls and accepted live-frame retention belongs to
the parent change. No shared Studio or live module was edited by this worker.

The JSON export serializes the complete existing versioned recipe, including
unknown future fields, and rejects values JSON would silently lose or change.
The PNG export uses the supplied retained `image/png` Blob unchanged, validates
its MIME and eight-byte signature, and never reads a canvas or creates a Phong
substitute. Filenames include a safe bounded object name, UTC timestamp, and
optional hexadecimal digest. Each browser download owns a temporary object URL
which remains alive for one second after click and is revoked on success or
failure. Errors reject to the caller's visible status handler. A successful
call means the browser download was requested, not confirmed disk delivery.

Verification on canonical Windows source: all 8 tests in
`node tests/meowdoku_marble_downloads_check.mjs` pass. They cover exact UTF-8 JSON
bytes/full field preservation, safe filenames, unsupported/lossy inputs, exact
PNG bytes and Blob identity, missing/wrong-MIME/corrupt-signature rejection,
anchor/URL cleanup and failures, concurrent downloads and snapshot capture.
`node --check examples/meowdoku/web/marble_downloads.js` passes.

No browser was opened, no real download was triggered, and no GPU, server,
production worker, upload, or publication was touched. Browser integration and
real download UI behavior have not been exercised by this worker.
