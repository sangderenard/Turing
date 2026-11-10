# Shared polling and retained publication, 2026-11-10

Observed live f4b8388571 failed publishing strip 39 after 38 committed batches (155648 rays). Browser poll setting was 60 Hz; server rejects intervals below .09 seconds, while client formerly stopped on every poll error. Worker traceback ended in publish HTTP409; response body was not retained. Isolated exact relay classes reproduce 429 followed by stale-readiness409. This reproduces the plausible chain, not an observed live response body.

Source changes align UI/settings with existing 10Hz server contract, back off a rejected poll without disabling participation, and preserve engine HTTP error detail. Publication waits on existing readiness and retries only the explicit no-admission409 with the SAME lease/token and original deadline. Other HTTP errors and lost responses are not retried. Expired unpublished leases reenter existing GL-owner retained-strip reissue. Burst capacity, physics, compiler kernels and relay admission rules unchanged.

Validation: JS participation suite including rate-limit continuation/exactly-once; 24 relay/process HTTP tests; 18 spectral cooperative tests including readiness race, deadline and uncertain-outcome no-retry. No live files copied or server restarted by this worker. Parent coordinates release with shop changes. Full ray completion on client is not implemented by this fix.
