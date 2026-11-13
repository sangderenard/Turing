# Explicit PA ownership and native SDL sink

Existing spectral-analyzer `_play_output_bus` replaces a GUI preview buffer;
it is not an append-stream backend. Existing Nodus Radio SDL playback belongs
to the file source. The new sink reuses SDL's stream/resampler and the canonical
Nodus canvas FIFO/tool-row APIs rather than misusing either as a live consumer.

The headless API documents process-global canvas dispatch. Accordingly the
`nodus_pcm_playback` executable owns a separate process/graph/FIFO/device.
`nodus_radio_playback.Playback` opens nothing on construction/import.
An authorized service worker explicitly calls start(), then consume(sub,cancel).
Natural RTCP EOF drains; a cancellation stops only this object's child. The
caller retains ownership of the Subscription and must close its router port.
No network listener or new account privilege was created. Service authorization
and the existing UI's user/device choices must gate start(); this is a sink
interface, not a deployed public PA route or an authorization implementation.

Native transport is raw stereo signed16/48k stdin, not a new network protocol.
The FIFO refuses overflow; SDL's bounded queue and existing resampler handle
hardware-clock slew. Telemetry includes accepted frames, queue maximum, rate,
backpressure, empty-stream transitions, drain and failure. Empty-stream events
are not claimed to be actual audible hardware underruns. Invalid device/start
failure terminates only the attempted child; no fallback grabs another device.

Verification (all explicitly SDL dummy, no speakers): Release sink build;
`tools/test_nodus_radio_playback.py` verifies explicit start,5127frames including
a seven-frame final block, bounded queue, cancellation, and eager96000frame
backpressure without loss. `tools/test_nodus_radio_pcm_end.py` accepts the sink
executable as an optional fifth argument and proves the full native source ->
canvas -> RTP/RTCP -> existing router -> actual Subscription -> owned native
canvas/SDL sink path. It drained93959 subscribed frames with zero receiver
loss and the correct28byte final block. Eight startup packets preceded the
subscription; not claimed as full-file delivery. All processes/temp files owned
by the tests are cleaned. No service restart, live deployment, audible test,
network-denial bypass or push attempted.

Final native sink run reported zero empty-stream transitions in the full
pipeline and eager-source tests; the short5127frame fixture recorded one
empty-stream transition before its explicit drain. This counter is retained,
not hidden or called an audible glitch. Real hardware remains unqualified.
An invalid device ID was refused without switching devices or leaving a child.
