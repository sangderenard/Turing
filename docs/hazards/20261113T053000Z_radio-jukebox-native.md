# Jukebox native source lifecycle

Runner construction and HTTP reads are inert. An administrator explicitly resumes
rotation. It claims the existing SQL queue token, registers a service_jobs owner,
starts the existing nodus_radio executable for the catalogue's local file, and
finishes only that token after native exit plus received RTCP EOF. Failure stops
the rotation rather than consuming all requests. Pause lets the current song end;
Skip cancels the owned source. Shutdown marks its claim interrupted. A stale claim
from a prior service is refused, not silently replayed or called complete.

Operator-only run/nodus_radio.json configuration adds jukebox.native with absolute
executable, scope_dll and fft_dll paths and dedicated router_port, pcm_port and
spectrum_port integers. Each catalogue song has an absolute path. No HTTP request
supplies those paths, executable, ports or credentials. The source is SDL dummy:
it emits real PCM/NRAD without opening speakers. Existing separate PA playback
still pins a fixed producer and DOES NOT yet follow jukebox songs.

The native CLI's optional final wait-for-start=1 holds source advancement, announces
its actual NRAD sender and PCM sender/SSRC, and waits for START on owned stdin.
The runner establishes its pinned receiver first, then releases source advancement.
The gate requires the updated native build; the old binary rejects the extra arg.
The standard 64 MiB source/conversion bound remains, not unlimited track duration.

Runner.station() gives the trusted current NRAD peer/title to the existing snapshot
reader, without changing persisted operator configuration. SQL audio storage is
not added. Native FIFO, RTP/RTCP and existing Subscription remain unchanged.

Verification: four runner tests passed (inert construction, completion, fail-stop,
skip, prior-current refusal across four cases). Native source rebuild was started
but has not completed beyond CMake Checking File Globs at this receipt; native
handshake and full-frame end-to-end proof are NOT yet verified. The bounded silent
proof is tools/test_nodus_radio_jukebox_native.py. No live configuration, audible
playback, restart, public exposure or network-denial bypass was performed.
