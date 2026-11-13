# General compute queue extension (2026-11-13)

Request: "work on the photo job becoming a general nodus compute request kind of thing, so expand the boundaries of that queue"

Extended the existing Studio contract queue, not a new scheduler or renderer.
See docs/NODUS_COMPUTE_REQUESTS.md for endpoints, guarantees and limits.
Actual isolated native Nodus canvas built the real default Studio scene (28,492,804
bytes), with replay and cross-account refusal checked; no ray/GPU job launched.
Studio regression suite: 18 passed, 1 skipped. Shared SpeakToMe pytest bootstrap
attempted PyPI and was denied by network policy; tests there remain unverified.
No live restart, deployment, database edit or remote publication performed.
