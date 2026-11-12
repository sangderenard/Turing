# Explicit camera-pair receiver v2

This extends the existing scene-response transport. It adds no renderer,
optical calculation, shader, new endpoint or readiness claim.

## Compatibility and admission

Ordinary requests and completed-sample snapshots remain v1. A request may add
`light_field_camera_records: "interval-responses-v2"` only alongside the existing
explicit `light_field_information: "evaluated-responses-v1"` contract. The
existing internal-v6 field requirements still apply. The new choice is not
automatically added by the UI or backend.
The existing request packer accepts the corresponding explicit
`lightFieldCameraRecords` option; it does not silently discard that selection.

The artifact reader requires that choice before serving a v2 receipt. It uses
the same immutable file, ownership checks, bounded payload validation and
transfer lease as v1. Receipt and manifest schemas must agree.

`studio-scene-responses-v2` uses the existing four sections; only the family 3
codec/stride changes to `native-camera-response-pairs-v2`, 1024 bytes. The first
128 uint32 words are the actual interval; the next 128 are its aligned response.
There is no fabricated third row. `camera_pair_contract` identifies both
component schemas, offsets, exact sizes and same-dispatch-row-ordinal alignment.
The metadata is checked exactly. V1's opaque 1536-byte family remains supported
as before, but cannot carry the v2 contract.

## Identity and limits

Family 3 v2 has no guessed surface recipient or spatial point. Its outer index
is native-only/unlocated, retains the actual current primitive and exact entry
direction, and leaves cells unavailable. Entry/current/first-hit geometry and
all native response words stay in the pair unchanged. The decoder validates
outer-to-inner attachment and the actual active-band identity. NaN response
attributes, signed zero, evaluated presence and source-state words are not
converted to RGB, coverage or calibrated regional deposited energy.

`applicationReady` remains false. No pair replaces ordinary Phong. This receiver
does not establish complete producer coverage, trace queued reactive work,
spatially index camera segments, activate a UI mode, or remove the worker's
hardware execution hold. The separate isolated record-index branch was not
merged into this current-canonical receiver branch.

## CPU verification

Base: `7813a6a37f4916a1c9d095d3203571f1e45a4b43`.

- `node tests/meowdoku_camera_pairs_check.mjs`: v2 ABI/attachment/version/receipt
  and opt-in negative cases passed.
- `node tests/meowdoku_scene_responses_check.mjs`: both existing v1 groups passed.
- `node tests/meowdoku_studio_light_field_check.mjs`: existing receiver lifecycle
  checks passed, including unavailable data preserving Phong and no GPU uploads.
- `tests/test_meowdoku_camera_pair_delivery.py`: 18 passed using the matching
  real Spectral store and publisher, including the existing leased artifact route.
- Existing `tests/test_meowdoku_forward_field.py -k "response and not http"`:
  21 passed, 183 deselected. Only direct CPU/API contract cases were selected.

For an isolated checkout, set `SPEAKTOME_REPO` to the existing SpeakToMe checkout
and `STUDIO_SPECTRAL_TEST_ROOT` to the matching Spectral v2 checkout. Put that
Spectral checkout on process-local `PYTHONPATH`. The existing regression module's
default Spectral path was rebound only inside the test process to that checkout.
No installed package or production setting changed. Initial collection without
the sibling SpeakToMe path failed; using the existing override resolved it.
Test startup built its existing CPU CFFI helper; no shader/GPU compilation ran.

The Node check also accepts a real producer snapshot on standard input:

```
node tests/meowdoku_camera_pairs_check.mjs --stdin
```

Input is JSON with `manifest`, base64 `payload`, and optional `cameraRhsWords`
containing expected exact uint32 words. It checks the actual receiver without
HTTP, a browser, a GL context or rendering.
