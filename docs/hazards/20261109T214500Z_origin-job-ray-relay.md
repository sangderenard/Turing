# Origin-job client intersection relay

## Scope

`examples/meowdoku/client_ray_relay.py` provides a bounded mailbox for the real isolated Spectral `FirstHitLoan`, not a second renderer. The engine owns registration and its GL context. There is no HTTP registration, public work pool, account credit or automatic production participation. No Studio production route was enabled and no assets deployed in this increment.

The owning GL thread calls `register(job, owner_session)` and retains the ephemeral returned credential for its originating browser. `publish(job, issued, query_kernel, seconds)` accepts the actual FirstHitLoan issue record. `kernel` identifies the composed sensor continuation; `query_kernel` identifies the exact compiled browser intersection kernel. These different hashes must both be bound, not incorrectly equated. `geometry_hash`, scene `revision`, `sample_prefix` and per-lease `token` also travel unchanged.

HTTP threads can only poll, submit, or cancel using both the originating session identity and job bearer. `handle_relay_post` checks custom header, same origin when present, JSON size, and TLS except loopback isolated tests. Hosting code must supply the existing authenticated HttpOnly session and existing access/viewer gate. Tokens never enter URLs, saved files or log messages. No registration endpoint exists.

Results transition pending -> submitted -> validating -> committed/rejected/uncertain. Submitted is explicitly not committed. Only the GL owner can take a result once, run authoritative geometry validation and `FirstHitLoan.complete`, then finish the relay. The mailbox does bounds checks, not nearest-hit correctness. Engine validators remain mandatory. Expired work stays retained; the owner may issue a new token for the same pending strip. Expiry does not close the engine loan, run fallback, advance sampling or drop rays. Uncertain GPU deposition must not retry.

Caps: one job per session, eight registered jobs total, one outstanding batch per job, 4096 rays and triangles, 60-second maximum lease, 2MiB request body, at most ten polls per second. Explicit engine close releases registration. Unsupported/empty triangle scenes reject. Numeric NaN/infinity and out-of-range hit ownership/intervals reject.

## Browser controls

`web/client_ray_participation.js` takes a job, ephemeral credential and exact compiled backend factory. Off by default; batch cap16–4096, poll rate0.1–10Hz, measured wall-time duty1–50%. Batch caps reject oversized complete strips rather than silently slicing/dropping them. Requests time out after15seconds. Hidden pages discard in-flight query output and stop polling. No result POST is retried after uncertain delivery; status is polled instead. Results are not advertised as completed or server savings before engine confirmation. Closing UI does not silently cancel the original pending strip.

## Tests and limits

19 Python relay tests passed including real loopback HTTP/session/origin/auth handling, stale/duplicate identity, expiry-before-consumption, result validation, caps, and rate limits. Node client tests passed for opt-in, kernel identity, exact-once submit, hidden discard, expiry, batch caps and engine acknowledgement. Engine/GPU browser end-to-end proof is owned by the FirstHitLoan integration worker; these tests alone do not prove actual GPU savings or full Studio transport support. Existing Live/Photo production loops remain unchanged.


Independent review found and fixed late asynchronous backend creation leaking after close, and uncertain result delivery incorrectly stopping rather than confirming. Regressions prove the late resource closes and a lost POST response causes status polling without resubmitting. Huge JSON integers now reject without numeric-overflow exceptions. Optional query cancellation signal is forwarded to the compiled backend.

Engine worker reports actual `spectral-analyzer/tests/studio_first_hit_loan_gpu_check.py --relay` passed: exact GPU ray export -> origin-authenticated localhost HTTP -> original-law WASM -> queued result -> GL-owner authoritative validation/commit yielded bitwise image parity. Receipt `spectral-analyzer/build/dot-first-hit-loan-20261109/gpu-receipt.json` identifies external_result_path `--relay`. This is isolated proof, not production delivery or speedup evidence.
