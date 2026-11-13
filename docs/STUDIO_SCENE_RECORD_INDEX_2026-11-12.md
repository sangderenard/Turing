# Bounded lookup of delivered scene observations

This is a CPU lookup addition to the existing response decoder, not a new
producer, renderer, deposition estimator, shader splitter, or blend rule.
Canonical source, Pluck, the live service and Nodus were not modified.

## Provenance and isolation

Worktree: `C:/Users/alber/turing-vector-index-20261112`.
Branch: `codex/vector-record-index-20261112`.
Repository base: `0485ded4cc251f8d8175459a5877d2e1426230fb`.

Two existing, shared, untracked canonical files were copied as the input
baseline. They were not authored by this change:

- `examples/meowdoku/web/marble_scene_responses.js`:
  SHA256 `8adb98050c49a1d6feb7e4bac5c769a95dc62f5d8d44b2b02ea2f9af640f938e`.
- `tests/meowdoku_scene_responses_check.mjs`:
  SHA256 `415346d97fdd53abb9ddc47661eecee79d13de831ac5274f95550497664a9ef0`.

Exact baseline copies remain under
`C:/Users/alber/AppData/Local/Nodus/UserFiles/vector-index-baseline-20261112`.
Review the index change against these copies, not as if the complete shared
decoder were new work. Nothing has been copied back, pushed or deployed.

## Existing entry and explicit allocation

`receiveSceneResponses` retains the existing information opt-in, exact payload
hash, schema, context, family and section validation. The new `recordIndexBytes`
option defaults to zero. A positive explicit budget builds `result.recordIndex`
after the existing decoder succeeds; ordinary requests allocate no index.

The budget covers eight bytes per cell/record reference plus a 64 KiB bounded
scratch reserve. Before allocating the reference array, a counting pass refuses
overflow. No truncated partial index is published. In-place heapsort avoids
per-reference objects and a sort-sized temporary array. The reported budget is
for this additional index, not the decoder's payload or the entire JS process.

`recordIndex.records([x,y,z], filters)` lazily yields `{row,support}`. `row`
references the unchanged 20-word `scene_index`; that index remains authoritative
for native section, payload offset, family, context, recipient and exact vectors.
Filters can specify family, semantic, context, recipientNamespace,
recipientOrdinal, directionSemantics and directionBin. Spatial search is binary
search over the sorted reference array; exact filters then inspect that bucket.
This is not a measured claim of optimal GPU performance or an importance rank.
The decoded arrays and their derived lookup are one snapshot and must not be
mutated independently. There is no cross-publication record-identity shortcut.

## Point and interval support

Support bit 1 means the copied endpoint lies in the cell. Bit 2 means a finite
copied forward/reactive segment has positive-length support in the cell. One
record with both kinds of support yields one reference with mask 3. A segment
referenced by several cells is still one observation, not several energy samples.

Only the established forward/reactive 128-word codec is used for segments:
entry origin at words 117-119, entry direction at 114-116, actual hit at 21-23.
The native family, semantic, hit presence, known primitive and endpoint/direction
attachment must agree with the outer index; entry direction must be finite and
nonzero. Unknown primitive identity stays unknown. The packet, region words,
medium and all other payload bits stay untouched. Camera totals and RHS rows
retain point support only; this change invents no camera interval.

The interval is clipped to the existing declared grid, then classified by its
grid-plane crossings. Both endpoints may lie outside the grid. Lower boundaries
are included and upper boundaries excluded; corner-only contact has no interval
support. The actual endpoint separately retains the producer's half-open point
ownership. No epsilon, optical continuation, extra ray, path reconstruction,
energy integration or direction averaging is introduced. Unrepresentable grid
spans refuse the index. Numerically unresolved or unsupported intervals stay
raw with a segment-unavailable count rather than a fabricated interval.

Direction filters refer to the original indexed direction, which is outgoing
for forward/reactive rows and camera propagation for completed camera rows.
They do not pretend that the outgoing angular bin describes the entry segment.
Its exact entry direction remains in the original payload for a later declared
directional-support contract.

## What this does not complete

- The producer currently excludes some rows by endpoint location before
  publication. A receiver cannot recover a crossing that was not delivered.
- Missing camera intervals, non-executed reactive work, and unobserved interiors
  remain missing. Declared family codecs do not prove complete production.
- Packet throughput, response factors, scalar deposit columns, camera summands
  and completed camera samples keep their different meanings. This index cannot
  establish energy deposited in a region or complete non-opaque-volume coverage.
- Zero references means no delivered indexed information, never physical black.
  Known zero, availability and loss/eviction evidence are untouched.
- The current RGB camera application and shader blending are unchanged and
  unready. Addition risks counting existing direct Phong lighting twice;
  averaging needs comparable observations and valid weights. Record count or
  brightness is not a new coverage/importance/energy weight. No new blend mode
  or interpretation of the user's shadow-pass correction is imposed here.
- No CPU index success admits GPU execution, production installation or rendering.

## CPU verification

- `node tests/meowdoku_scene_responses_check.mjs`: both existing check groups pass.
- `node --test tests/meowdoku_scene_record_index_check.mjs`: 17 tests pass,
  including 150 deterministic segments compared with independent cell-by-cell
  slab classification, outside-to-outside crossings, negative travel, boundaries,
  mixed families/contexts, native-only interiors, unavailable intervals, explicit
  opt-in/budgets and byte-for-byte unchanged payloads.
- `node --check examples/meowdoku/web/marble_scene_responses.js`: passes.

Only CPU fixtures ran. No compiler, GL context, GPU link, dispatch, render,
service restart or publication ran. Full repository tests were not run.

## Prompt history

User: "We should bucket them in because what are our last phong layers going to do, we want them to as fast as possible apply them where it's going to make the most difference instead of bucketing deposition we deliver vectors and the phong compensates where it gets new useful information blended with the phong before it"

User: "Right phong does a shadow pass if something is not already black it won't become it should be average added or some other blend mode"

User: "We're looking to classify a grid and make statements about energy deposited ina region under conditions we will use to texture, animate, or otherwise state change based on a March the includes all non opaque volumes and we want to retain just enough directional information to paint speculars as texture over geometry"
