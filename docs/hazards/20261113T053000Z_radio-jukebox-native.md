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

## Follow-song PA update

The explicit PA Manager can now attach to Runner.subscribe_pcm(): a bounded
64-block tap of the same already-paced PCM readback. This does not re-decode,
open a second UDP port or compete for source FIFO tokens. It feeds the existing
native Playback sink only after the user's explicit server-speaker Start action.
The tap follows subsequent songs; Stop releases it. Overflow fails visibly rather
than hiding gaps or accumulating unbounded memory. No source or speaker starts
from GET, menu opening, or runner construction. The fixed-source standalone mode
is preserved when no jukebox provider is configured. This supersedes the earlier
fixed-peer PA limitation in this receipt. Browser audio remains a separate task.

14 focused Python control/runner tests pass after the update, including inert
status, two sequential PCM blocks across the shared feed, explicit cancellation,
no duplicate receiver and bounded overflow. These use fake native sink boundaries.
Standalone CMake VerifyGlobs completed immediately. MSBuild /m:1 /nr:false now
progresses through native compilation; prior hung worker launches were not evidence
of a source compile failure. Native full proof remains pending until build finishes.

## Final measured native proof

Release nodus_radio/nodus_headless build completed successfully with MSBuild
/m:1 /nr:false. The earlier build-worker hang is resolved for this invocation;
no shader/compiler implementation was changed to resolve it.

Real native source -> canvas FIFO -> RTP/RTCP -> private native router -> pinned
Subscription delivered all 48007 frames in188 packets, including the seven-frame
tail. EOF was verified, with zero lost/late/rejected/resync counts. The optional
native PA proof then ran two tracks (fresh sender/SSRC each) through one bounded
tap into the existing native Playback sink:96014 frames accepted including both
short tails. Both source and sink explicitly used SDL dummy. No audible output.

41 focused radio Python tests passed; git diff --check passed. Native proof uses
disposable fixture/temp directory and isolated owned processes, all closed after
the test. Real speakers, public HTTP deployment and browser audio are unverified.
