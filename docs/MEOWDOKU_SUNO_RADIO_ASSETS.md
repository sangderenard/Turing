# Suno songs in the Nodus Radio jukebox

The signed-in account's existing Suno connection supplies `audio-retrieve`
and `audio-download`. A selected completed song is checked before a download
credit is spent. If Suno returns `requires_credit`, the connected-services
page shows a separate explicit credit button. The import route never accepts
a local path or a provider credential from the browser.

Imported audio is durable only under
`<state>/users/account/<owner>/assets/suno_radio/<song-id>/audio.mp3` by
default. The optional lossless choice downloads WAV and encodes FLAC at level
12. SQLite records the owner, song ID, title, storage format, byte count and
digest. The shared station catalogue derives its paths from those records;
other accounts cannot list the importing account's assets. Choosing import
offers the track to the shared station. It does not start playback.

The operator's private `<state>/run/nodus_radio.json` still controls Radio's
native source, ports, HLS and executable paths. The new
`jukebox.native.codec` object names the Nodus audio codec executable, an
FFmpeg executable, protected core roots and optional DLL directories:

```json
"codec": {
  "executable": "C:/operator/nodus/nodus_audio_codec.exe",
  "ffmpeg_executable": "C:/operator/ffmpeg/ffmpeg.exe",
  "core_roots": ["C:/dev/Powershell/nodus", "C:/dev/Powershell/turing"],
  "dll_dirs": ["C:/operator/nodus/bin"]
}
```

These paths are illustrative, not installed defaults. The codec creates a
host-declared Nodus exec row with FFmpeg as its protected program. Playback
decodes the durable MP3 or FLAC to a temporary WAV in the same user's asset
directory, feeds that WAV to the existing native file-source Radio table,
and removes it after the track ends. It does not store a second durable PCM
copy. The file source retains a finite 256 MiB PCM bound; the codec's byte
span is also capped at 256 MiB. Longer songs need a streaming source path.

This source change does not configure an operator's live Radio instance or
start the service. Use the existing guarded service lifecycle and explicit
Radio admin resume after the review lands.
