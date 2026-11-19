# Direct FLAC uploads to Nodus Radio

The Radio panel accepts a signed-in account's lossless FLAC master through
`POST /api/radio/uploads?title=<URL-encoded title>`. The request body is the
FLAC bytes, with `Content-Type: audio/flac`, one `Content-Length`, the
`X-Meowdoku-Radio: 1` action header, and a same-origin `Origin`. Preview
accounts are read-only through the server's existing POST gate. The browser
control handles this request; the binary body is not an MCP tool argument.

The server streams at most 256 MiB into a staged file. It checks FLAC
STREAMINFO and the 900-second codec ceiling, then uses the host-declared
Nodus audio codec row to decode the complete file. The codec must already be
configured in `run/nodus_radio.json`; without it, the upload fails and the
stage is removed. After validation, the server retains the original FLAC once
under `users/account/<owner>/assets/radio_upload/<id>/audio.flac` and records
owner, title, bytes, SHA-256 and creation time in SQLite. This path satisfies
the existing codec work-root constraint. These assets do not use the private
file or Suno per-owner storage quotas. Available Radio library disk space is
the aggregate capacity; playback WAVs are temporary.

`GET /api/radio/uploads` lists the caller's uploads. The shared jukebox
catalogue includes their completed records with server-generated IDs. An
upload does not request the song, resume the station, start server speakers,
or expose the original as a private inventory file. Existing jukebox request,
rating and operator playback controls still govern use.

## Deployment and verification

The inspected live server had no `run/nodus_radio.json`; `GET /api/radio`
returned `unavailable`. An operator must install the existing file-station
and codec configuration, verify its declared binaries, ports and HLS settings,
and start it through the existing control. Then barnacleboy4 can upload the
two FLACs in the Radio panel and compare each SHA-256 receipt to the ledger.
If upload succeeds but audio is dark, check station status and resume
separately. No upload endpoint supplies a path, executable, or configuration.
