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

`/api/radio/history` reads only the current, operator-selected file station's
NRAD packets, with the same `view_site` gate as the audio route. It returns up
to 512 frames from the latest ten seconds, each with the captured stereo-frame
index, sample rate, FFT powers, RMS, and relay arrival time. The browser selects
an FFT frame by `frame / sample_rate <= audio.currentTime`; arrival time is not
used for display synchronization because HLS adds buffering. This provides
source-timeline alignment, subject to AAC encoder delay and any dropped leading
PCM. Live playback must be measured before claiming sample-accurate sync.
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

Live acceptance still needs an operator configured file catalogue and the
matching `nodus_radio.exe` build. Check the playlist and segment route through
Apache HTTPS with a viewer account, verify denial without `view_site`, play
through native HLS and a Media Source Extensions browser, and compare audible
onsets with the displayed FFT frame indices across a song transition.
