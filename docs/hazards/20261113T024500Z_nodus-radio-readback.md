# Bounded PCM readback slew and momentum

The isolated Radio receiver previously advanced exactly one block per caller
tick and substituted silence whenever empty. It had no rate feedback or EOF.
That is the observed source at 5d8597c7, not a claim about the live service.

`Subscription.take_due()` now uses the existing JitterBuffer with a
`time.perf_counter` clock, queue-depth feedback, a 0.5-second momentum filter,
maximum rate offset +/-0.5%, and maximum rate change 0.2% per second. The
prefill default is now eight blocks (~43ms), within the unchanged 32-block
capacity. A measured native arrival gap was 23.16ms; the old three-block
prefill represented only16ms. Settings remain constructor parameters on the
buffer. `status()` exposes read rate, next deadline, block count, rebuffer
count and end state as well as the existing packet metrics.

Consumers call `take_due()` on their high-resolution service loop. None means
wait (or ended, inspect status); bytes are unmodified PCM. `take()` remains
available for an externally clocked audio sink; do not mix two consumer clocks.
This is pacing, NOT sample-rate conversion: a fixed-rate DAC requires its own
resampler. No PA device, browser playback or TCP audio handoff is added here.

`buffer.finish(last_sequence)` is an explicit trusted control-plane end marker,
called after the final batch is delivered. It drains up to that sequence and
then returns None without counting endless tail silence as packet loss. It
does not guess EOF from quiet UDP, and the native producer still has no EOF
control message. That integration remains outstanding. Actual missing blocks
before the declared end still become counted silence; no fabricated recovery.

Verification (2026-11-13, Windows, silent):

- Baseline five focused unit tests passed before edits. All eleven now pass.
- Two deterministic sixty-second producers with +/-2000ppm clock skew and
  bounded4ms jitter delivered all11,250 blocks each in order without loss,
  late packets or resync; sequence/timestamp wrap included.
- Tests cover rate/slew bounds, momentum, explicit EOF including missing final
  packet, loss counters, malformed EOF, high-resolution clock and stalled
  consumer rebuffering. Existing real TCP fragmentation fixture now calls the
  adaptive Subscription interface.
- An eight-second real native WAV/canvas/RTP/router/BATCH/Subscription run,
  using `SDL_AUDIODRIVER=dummy`, received1504 blocks, read1495 nonzero blocks,
  left9 queued, and recorded zero lost/late/rejected/resync/invalid/foreign.
  Final read rate1.00095651, maximum observed arrival gap23.155ms. Producer
  telemetry reported zero drops. Test intentionally ended midstream, so this
  does not establish native EOF behavior or full-file delivery.

No speakers used, live service restarted, deployment performed or public
endpoint changed. Full compiler suite not run (unrelated and known unbounded).
