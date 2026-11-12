2026-11-12 CAMERA QUERY-ONLY INDEX (CPU MEASURED):

Owned receiver branch c71d892b plus clean cherry-pick b99d7f9d of existing
8a75596b index, then explicit opt-in camera support4 through the same bounded
walker. Native records and outer attachment unchanged. No canonical edits.

17 unchanged existing index checks pass. Extended camera-pair checks pass for
both opt-ins, versions, miss/NaN/infinite/zero geometry, nonunit direction,
first-hit trap, memory limit, statuses and all tested media including unknown.
Existing scene-response and light-field lifecycle Node programs pass unchanged.
Actual Spectral SceneResponseStore serialization roundtrip: two pairs, 512 exact
words, four query references, zero GPU bytes. No pytest/CFFI startup required.

Query geometry is derived from copied origin/direction/parameter. It proves no
surface evaluation, path coverage, valid medium or deposited energy. Physical
source coverage, usable shading contract and hardware resource validation are
still open. No shader/GPU compile, link, render, live job or deployment occurred.
Reproduction is in docs/STUDIO_CAMERA_PAIR_RECEIVER_V2.md.

Follow-up conservative clipping correction: reproduced a genuine finite crossing
(-1e30 origin, 2e30 parameter) whose ROI entry/exit collapse to equal float64
values. It previously reported empty support. Shared walker now reports equality
as unavailable (-1), without tolerance or guessed cells; strict disjoint remains
empty (0). Camera and copied-segment regression fixtures preserve native bits.
All 18 index tests, camera-pair checks, existing response/lifecycle checks, and
actual-store serialization roundtrip pass. No GPU/live activity.
