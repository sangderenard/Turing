2026-11-09 STUDIO FILM SCIENCE AND FINISHED-PHOTO RETOUCH (LOCAL SOURCE):

The completed-photo client now retains its finished digest for postprocessing,
serializes retouch requests, keeps the newest edit, and surfaces failures.
The backend treats a terminal status as finished even while the worker process
is exiting, so that saved HDR is available for immediate retouch. Browser QA
also found two integration defects: initial Studio synthetic control inputs
and its first geometry rebuild invalidated a resumed photo; photo-panel CSS
also overrode the hidden attribute. Initial setup now does not count as a user
edit, while subsequent design changes still invalidate the matching photo.

The bounded /api/marble_photo/film_preview endpoint launches the existing
Spectral CPU profile CLI, with strict finite bounds, fixed eight patches,
a 15-second timeout, serialized requests and revision-keyed bounded cache.
It does not start a GPU job or modify an exposure. Input/output panels and
36 circular 10-degree encoded-sRGB HSV bins use returned production pixels.
Dark and low-chroma patches are explicitly excluded; linear range excursions,
conditioning and fallback warnings are shown. This is not wavelength analysis
or a perceptually uniform hue metric.

Postprocessing shows the actual saved sensor conversion and display recipe.
Raw linear HDR remains separate and unchanged. New normalization and tone
controls use the same backend recipe, not a second client implementation.
The server-owned output profile version enters new fixed3 job digests so
pre-profile completed jobs cannot alias new requests. Legacy HDR is labeled
and retains its former transform. Finished-image and direct panel recipes
are tested separately from ray transport.

Validation and publication are coordinated in build/dot-studio-publication-20261109.
This receipt is not a claim that all local changes are already live. No commit,
push, full rebuild, unrelated process termination or unrequested exposure
cancellation was performed. The dedicated QA helper uses generated test HDR
under build/dot-film-science-20261109 only. Existing preview transparency is
approximate and intersecting-medium transport remains unresolved.

PUBLICATION VERIFIED: graceful restart boot d56633da1f5e, PID80624; all17
selected asset hashes match HTTPS service. Actual public Studio browser
shows new defaults and functioning direct CPU panels. Existing session restored.
Current real worker smoke passed128px16epochs with raw/profiled HDR and exact
manual-linear retouch parity. Final receipts and served snapshot are in
build/dot-studio-publication-20261109/studio-combined-release.zip.
