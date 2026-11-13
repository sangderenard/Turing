# Server integration of the jukebox PA follow path

The actual server handler now connects the existing PA Manager to the same
jukebox Runner when the operator configuration includes `jukebox.native`.
Both route entry orders reuse one runner and store. Neither construction, GET,
nor opening the menu starts native work or sound. Explicit jukebox Resume starts
the source; explicit Play on server speakers attaches the existing native sink
to the runner's bounded PCM tap. Fixed-source PA remains unchanged for stations
without a jukebox native configuration. Station mode is chosen when the PA
manager is first initialized; change operator config before startup, not during
an active stream.

44 focused Python radio tests pass, including three tests executing the actual
server handler methods (PA first, queue first, fixed station), both member/admin
header lifecycle tests pass, and diff checks pass. The source/PA native proof
recorded in the native receipt supersedes the earlier unverified build and
fixed-peer playback limits: two silent SDL-dummy tracks delivered 96014 frames
through one sink with their changing peers/SSRCs.

This completes local handler wiring, not live deployment. Publication, operator
catalogue/native path/port configuration, and an authenticated live endpoint
trial remain outstanding. No live restart/configuration change, external push,
network-policy bypass, or audible playback was performed. Browser playback is
not implemented; the current explicit playback target is server PA.
