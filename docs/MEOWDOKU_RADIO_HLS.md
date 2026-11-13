# Meowdoku Radio HLS

The Nodus Radio Free table can feed its stack-converted PCM tensor to the
declared `exec.radio.hls` row. FFmpeg encodes one AAC HLS stream, and Meowdoku
serves its playlist and completed segments through Apache's existing HTTPS
proxy to principals with `view_site`.

This is an opt-in addition to the operator-owned
`<state>/run/nodus_radio.json` file. Keep the existing `enabled: true`,
`source: "file"`, relay address, owner, catalogue and `jukebox.native` paths
and ports. Add this object under `jukebox.native`:

```json
"hls": {
  "enabled": true,
  "ffmpeg_executable": "C:/Windows/System32/ffmpeg.exe",
  "core_roots": ["C:/dev/Powershell/turing", "C:/dev/Powershell/nodus"]
}
```

Use a `nodus_radio.exe` built from Nodus PR #73 or later. The two core roots
are examples for this deployment; list the actual protected source roots for
the host. The generated HLS directory is fixed by `StatePaths.content_dir` at
`<state>/content/radio/hls` in the current layout, outside those roots. The service account needs
write access there. The fixed row declaration is generated at
`StatePaths.run_dir/nodus_radio_hls_rows.txt` on explicit jukebox resume and read by
the owned native process at image load. HTTP clients cannot supply program
paths, arguments, source paths or HLS directory names.

The browser's Radio panel shows an audio control when an owned file-source
track is active. Playback starts on the viewer's Play action. The player uses
native HLS support; browsers without native HLS support need a later player
integration. Closing the panel stops that browser's playback. The existing
server-speaker PA control remains separate.

The capability-gated route `/api/radio/hls/radio.m3u8` and its numbered `.ts`
segments run through the current Apache proxy and Meowdoku session check. A
static Apache alias would bypass that session check, so this version does not
use one. The encoder is one producer; each browser is a reader of completed
HLS files.

Nodus's isolated native test is `tools/test_radio_hls_stack.py`. Meowdoku's
focused tests cover declaration generation, safe file reads, runner arguments
and environment, and the existing jukebox/Radio behavior. Operator source and
licensing choices remain in the private Radio configuration.
