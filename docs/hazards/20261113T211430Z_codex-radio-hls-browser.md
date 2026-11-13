# Radio HLS browser route (2026-11-13)

The owner asked to continue the Apache/Radio audio work and corrected the Nodus
handoff: "counter offer: make that conversion a row and use the stack". Nodus
PR #73 now provides that stack conversion and its real native test: FFmpeg
published HLS before stdin closed; FFprobe found 48 kHz stereo AAC; 938 RTP
packets continued on the original exit with zero native PCM drops.

Meowdoku's opt-in `jukebox.native.hls` operator configuration takes an absolute
FFmpeg executable and a nonempty list of absolute `core_roots`. Explicit admin
resume writes the fixed `exec.radio.hls` declaration under `StatePaths.run_dir`, using
`StatePaths.content_dir/radio/hls` for output. Each owned track clears only that row's
generated playlist/segments and launches the new native HLS CLI mode. GET,
status polling and menu opening do not start a source or an encoder. The
existing private RTP subscription and PA tap remain in use.

Principals with `view_site` can read only `radio.m3u8` and numbered `.ts`
segments under `/api/radio/hls/`; requests pass through the existing Apache
proxy and Meowdoku session check. No Apache static alias bypasses account
authorization. Responses are private/no-store. The site menu adds a native HLS
audio control that starts only on a browser Play action; closing it stops local
playback. Browser support is limited to browsers with native HLS capability at
this stage.

Focused radio tests: 26 passed. Additional site/radio tests: 12 passed, 12
skipped. A real local HTTP route test passed for HLS MIME, bounded names and
the existing access gate. Node syntax check and diff whitespace check passed. The broader first
test command lacked `examples/` on `PYTHONPATH`, so collection stopped; the
same selected tests passed after setting that path. The hazards log command
itself currently rejects an older nonconforming hazard filename before
printing receipts; the newest two receipts were read directly. The HTTP route
test initially exposed a legacy-layout assumption; the implementation now
uses `StatePaths` for both modern and legacy state locations.

No operator Radio configuration exists in the inspected stable/development
state directories. No live source, Apache restart, public stream, browser
playback, or licensed catalogue content was enabled by this change. Deployment
requires the new Nodus binary and an explicit operator configuration; browser
delivery remains unverified until that happens.
