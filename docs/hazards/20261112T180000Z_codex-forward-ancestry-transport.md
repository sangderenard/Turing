2026-11-12 QUEUED ANCESTRY TRANSPORT V3, CPU ONLY:

Based on receiver2d605524. Explicit queue option and strict128-word forward-v2
contract use existing transport/lease paths. No default activation. V3 mandates
256-word camera pair ABI, rejects mixed versions and retains exact native bits.
Spectral producer owns shader/queue lifecycle; this change only admits and
retains its actual declared provenance. No GPU/live activity or deployment.

Measured: 9 new Turing request/artifact tests, 18 existing camera-pair route tests,
actual Python-store-to-JS v3 roundtrip, existing v1/v2 response/lifecycle programs
and18 existing index cases pass. Spectral store/camera/ancestry suite71 passed, including coexistence regression. Reproduction in
STUDIO_FORWARD_ANCESTRY_TRANSPORT_V3.md. CPU Nodus fallback/deprecation warnings
are not native execution evidence. Matching producer contract is a dependency.
