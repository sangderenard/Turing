# Static Living Data DM REST

Implemented locally, not deployed. This slice exposes the registered Living
Data map as analytic data and requests images from its existing Pluck Phong
renderer. It does not start a game engine or simulation. The current commands
are `inspect` and `render-phong`, not arbitrary DM edits or tensor writes.

## Authentication and ownership

All routes require the existing account cookie and dev_ui capability. Share
view links are refused. Each registered world belongs to one account. Other
accounts receive 404, even if they know its ID. No new DM role, broad admin
permission, MCP implementation or cross-account access is introduced.

The browser host explicitly registers the static map and its source identity.
Registration returns a private, random host_token used only for host work
polling/completion. Do not include it in user-facing status or files. A normal
GET never returns this token. All host calls still require the owner's account
session; the token alone grants no access. This remains an opt-in adapter, not
an automatic publication of every open browser world.

## Routes and commands

Base: `/api/living-data/dm/worlds`. JSON requests only.

- POST base: `{map, source_sha256, renderer_identity, request_id}`. Returns 201,
  world_id and host_token. The map uses the existing abstract-ui-div-map-v0
  shape. Both world and layout projections are retained separately. No uploaded
  script, shader, worker or WASM is executed by the REST service.
- GET `/{world_id}`: public attachment identity, capabilities and provenance.
- POST `/{world_id}/commands`: submit `inspect` or `render-phong`.
- GET `/{world_id}/commands/{request_id}`: retrieve its result.
- POST `/{world_id}/host`: owner plus host_token polls or completes a render.
- POST `/{world_id}/close` with `{}`: detach and discard retained data/frames.

Inspect example:

```json
{"request_id":"look-1","operation":"inspect","at":"exact-world-object-identity","radius":1}
```

Inspect defaults to the physical `world` layer. Select `layer:"layout"` for
semantic layout. The result contains the exact registered object record,
including its authored transform/form/material/semantic properties, plus the
existing graph query and a short text description. Radius/node/edge pagination
parameters are the same as the graph query endpoint. No missing geometry or
world state is reconstructed from narration.

Render example:

```json
{"request_id":"view-1","operation":"render-phong","position":[32,4,-12],"facing":[0,-0.15,1],"width":512,"height":384}
```

Render returns 202 queued. A registered browser host polls with
`{host_token,action:"poll"}`, receives request_id and camera, invokes the actual
static Phong host, and completes with:

```json
{"host_token":"private attachment token","action":"complete","request_id":"view-1","result":{"png_base64":"actual PNG bytes encoded in base64","width":512,"height":384,"position":[32,4,-12],"facing":[0,-0.15,1],"source_sha256":"registered source hash","renderer_identity":"registered Pluck Phong identity","actual_camera":{"position":[32,4,-12],"facing":[0,-0.14834045293024462,0.9889363528682975],"width":512,"height":384}}}
```

The receipt retains requested position/facing plus actual_camera from the
renderer, whose facing is normalized. Source/renderer identities and camera
position/dimensions must match exactly; normalized direction is checked within
1e-12 floating-point transport precision. This is not a physics tolerance. The service
checks PNG size, dimensions, single-frame status and full pixel decoding, then records png_sha256.
Host failures may instead complete with `result:{error:"reason"}`. A missing
host times out as failed; no placeholder frame is produced.

A DM client polls the command receipt until complete/failed and decodes the
returned png_base64. This is an actual raster image, not an SVG or generated
illustration. Camera requests do not move an actor or change simulation time.

## Existing static renderer adapter

`tools/add_living_data_static_host.py` (see the actual staged receipt) creates a
separate reversible static candidate from a verified original compiled page.
The adapter keeps original map/shader bytes and uses the existing registry,
geometry builders and updateShaderViewer/Pluck Phong path. It suppresses physics,
vehicle, navigation, audio and entity-cycle startup rather than replacing them.
The static browser API is `livingDataStaticHost.ready`, `.snapshot()` and
`.render({position,facing,width,height})`.

The original source hash is a host declaration, not independent server
attestation. The service additionally computes map_sha256 over the registered
JSON and png_sha256 over accepted bytes. The isolated browser/REST test verifies
which original source and real rendering path supplied the data. Do not call a
random uploaded map or PNG an independently verified live game state.

## Limits

One world per account, four total, 15-minute idle lifetime, 32 retained command
receipts per world. One render in flight, 60-second render deadline. Request IDs
are 1Ã¢â‚¬â€œ64 ASCII letters/digits/underscore/hyphen. Same ID and exact command replays
the receipt; changed content gets 409. At capacity, close/reattach explicitly.
Restart/close/expiry discards state and ends the retry window.

Registration and host completion bodies are at most 4 MiB; commands 4 KiB.
Images are at most 2 MiB decoded and 64Ã¢â‚¬â€œ1024 pixels per dimension. Analytic
responses are capped at 2 MiB. Graph node/edge/horizon limits remain in force.
Coordinates are finite, within +/-1,000,000; facing must be nonzero. Limits are
transport admission, not new physical laws. Source scene lighting is preserved.

No persistent saving, world mutation, reveal-to-another-player permission,
physics advance, game-engine host or MCP endpoint is claimed by this slice.


## Verified local run

Run `python -u tests/living_data_static_rest_browser_check.py` against the
materialized static stage. It starts the real Handler on an ephemeral loopback
port with temporary account/state roots, logs in, registers the actual browser
snapshot, inspects an object, requests two viewpoints, renders through native
Phong, submits the actual PNGs and retrieves them over REST. It also mutates
the test registry and verifies that the immutable static host refuses further
inspection/rendering rather than silently re-pinning. No production or denied
public endpoint is contacted.

The Storybook stage uses the actual authored StorybookEnvironment, through
`build_introspective_world` and `project_world_to_div_map(static_only=True)`.
It exposes 33 native objects from five class identities. Actual REST inspection
retains authored room metaphors; two native Phong renders target LanternCottage
and ClockworkTower geometry. Browser instrumentation observed zero Worker
instances and the server created no game Tables. Static projection stops before
game/vehicle/audio builders and its standalone JavaScript suppresses game
startup. Existing shader and geometry implementations are retained. Native solar
lighting is dark; these are class-layout geometry renders, not illustrated art.

The source SHA identifies the original native JavaScript packet before the
static adapter; the complete standalone HTML has a separate output SHA. See
`build/dot-storybook-static-phong-20261110/receipt.json`. The previous mechanical
stage remains a separate fixture, not a renamed Storybook scene.

Static host map, renderer geometry and shader choice are pinned on first use.
Checks run before and after drawing; later changes permanently invalidate the
host. A fresh verified attachment is required after editing the source.

Final focused regression: 62 tests and 10 subtests passed across static DM,
static adapter/producer, Storybook, graph cursors/query, existing worldmap and issue ticket suites.
Python/JavaScript syntax checks and independent service security review passed.
Real browser REST receipt and returned PNGs are in
build/dot-storybook-static-phong-20261110. Full repository tests and deployment were
not performed.
