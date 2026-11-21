# Radio browser player

The shared server station keeps the Nodus Radio file-source canvas, FFT/NRAD
relay, PCM row, and `exec.radio.hls` FFmpeg row. Apache proxies the authenticated
`/api/radio/hls/radio.m3u8` playlist and numbered segments. The browser uses
native HLS when available and otherwise loads the local `radio-hls.min.js`
decoder after the listener presses **Listen in this browser**.

The vendored module is `hls.js` 1.7.4, `dist/hls.min.mjs` from the npm package
with integrity `sha512-3GlxGdPvT2YHvQYOt+yOzT/DEior/qJVNIylHA77kgeoBTdQQqi70GwefNEUmHQeNGyjC8U62+bruqW/Tpm+tg==`.
Its local SHA-256 is
`5501f1d9091b6f749e0ef61aff4b9ea3ba13fd5fa5e2aefb6fc0ff871847a531`.
The Apache 2.0 license is shipped as `RADIO_HLS_LICENSE.txt`.

New direct FLAC and Suno imports run the same Nodus `radio.file_output` →
`nodus.fftfree` → `radio.fft_bands` → `radio.udp_broadcast` table in offline
analysis mode against the decoded source before catalogue publication. Its
NRAD v1 records are stored as `levels.nrad` beside the owned media. Each record
has version, five-band count, 48 kHz sample rate, source-frame index, full
window RMS, and five normalized powers. The account track ID is the directory
and API key. The source uses 1024-frame FFT windows and 256-frame hops.
Admission rejects a new upload/import if analysis fails; neither media nor
levels becomes ready without both files. Derived files are bounded to 16 MiB
each and 1 GiB across the shared Radio catalogue, separately from the 10 GiB
direct-FLAC library ceiling and existing Suno owner-media quota.

`/api/radio/levels?id=<track>` sends the bounded binary analysis to signed-in
listeners, with explicit track/version/rate/hop/band headers. The browser
loads it once per playing track and selects the indexed frame by
`frame / 48000 <= audio.currentTime`; it does not poll `/api/radio/history` for
meter display. Existing direct and Suno entries with no analysis are backfilled
from the WAV already decoded when they are next claimed for playback. A failed
backfill leaves audio playable and shows that stored levels are unavailable.
Operator-declared tracks have the same no-meter fallback until an account
asset analysis path is defined. Missing files never produce fabricated values.

Source and HLS clocks share the track start, subject to AAC encoder delay and
any dropped leading PCM. Live playback must be measured before claiming
sample-accurate sync.
The FFmpeg row sets `-muxdelay 0 -muxpreload 0`: a five-second 48 kHz PCM
fixture produced a first MPEG-TS segment with `start_time=0.000000`. Without
those flags the same local FFmpeg build started at 1.4 seconds, which would
shift the FFT visualization against audible playback.

The player is a compact, boxy station control with a searchable track index.
The current route represents one shared server station. Later station IDs must
identify separate authorized sources, HLS output directories, and FFT histories.
Personal uploads and private listening belong to each account's collection and
playback route; they must not inherit the shared station's catalogue or the
admin-only server speaker control. The existing `/api/radio/playback` action
remains gated by `account_manage` and is rendered only for that capability.

Offline analysis needs the reviewed Nodus `nodus_radio.exe` build installed
before the Turing upload change is published. Isolated verification on
2026-11-19 produced 375 records for a two-second source and 45,000 records
(2,160,000 bytes) for a four-minute source in 12 seconds. The direct FLAC and
Suno FLAC/MP3 admission paths passed against that build with a temporary
database and asset directory. These checks do not establish browser timing.
Check the playlist and segment route through
Apache HTTPS with a viewer account, verify denial without `view_site`, play
through native HLS and a Media Source Extensions browser, and compare audible
onsets with the displayed FFT frame indices across a song transition.
