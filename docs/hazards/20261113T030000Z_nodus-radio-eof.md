# Native finite-source end and final partial PCM

Follow-up to the adaptive readback receipt: the existing native Radio now
publishes a terminal token through its canonical canvas PCM FIFO and emits
standard compound RTCP SR/SDES/BYE on the existing router path. No new broker,
HTTP service, public port or permission was created. The receiver pins the
existing producer endpoint and SSRC, validates the negotiated terminal
compound, and maps its exclusive RTP sample endpoint onto the buffer sequence.

RTP L16 blocks can now contain1..256 stereo frames, including a final partial
block. Known missing final frames remain counted silence of the exact declared
length. A quiet UDP stream is never inferred to be EOF; the producer's terminal
notification is necessary. Repeated identical endpoints are idempotent and
conflicting or out-of-window endpoints are rejected. RTCP is best effort, not
an authenticated/acknowledged transport. Native repeats are bounded to three
attempts while the row remains alive. Forced process stop is not natural EOF.

Observed during implementation: the source had queued its terminal token, but
then set ctx.succeeded=false. ThreadManager breaks following rows on refusal,
so the PCM consumer never drained that token. The repair uses a terminal
internal control value through the source/spectrum rows; it does not fake a
spectrum frame or change the scheduler's refusal semantics.

Verification: sixteen Radio unit tests pass (including timestamp/sequence
wrap, short final block, genuine lost short tail, EOF overtaking final data,
malformed compounds and wrong SSRC). Native Release build passed. The retained
silent test tools/test_nodus_radio_pcm_end.py created a temporary2s+7-frame WAV,
selected SDL dummy, and ran isolated native source/router and real Subscription:
368 accepted and read, final28bytes, zero lost/late/rejected/resync, ended=true.
Native sent376; eight startup packets preceded subscription. This is not a
claim of lossless whole-file startup delivery. Temp files/processes cleaned.

Remaining PA integration: the canonical spectral-analyzer system_audio GUI
output exists, but analytic_driver._play_output_bus replaces its entire preview
buffer rather than accepting a live append stream. Native Radio's current SDL
output is part of its file source, not a backend PA subscription sink. Neither
is silently treated as a ready sink. No PA device or browser sound was started;
no site deployment, restart, authorization bypass or network-denial retry.
