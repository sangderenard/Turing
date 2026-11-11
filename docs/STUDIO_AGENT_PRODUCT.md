# Marble Studio agent short circuit

The product is **settings -> existing Studio scene construction -> blob**.
There is no page, DOM, browser, Phong program or replacement renderer in this
execution path. The Studio page and the agent share `studio_scene_builder.js`.

## Inputs and outputs

`web/marble_studio_agent.js` exports `buildStudioAgentProduct(request, host)`.
The request schema is `turing.studio-agent-request.v1`:

- `operation`: `scene` or `ray-payload`.
- `settings.object`: captured Studio object authoring state. The existing
  `prepareObjectSnapshot` verifies/hashes its exact geometry.
- `settings.lighting`: the complete **resolved Exposure** profile. This is not
  the Studio-only Room override profile, nor an invented set of light defaults.
- `settings.camera`: the existing photographic camera record in metres,
  including eye, target, field of view, aperture and focus distance.
- `photo`: optional existing photo-packer options for `ray-payload` only.
  Resolution, width/height, tiled planning and oversampling are forwarded to
  `packMarblePhoto`; that existing packer/backend own their supported bounds.
- `objAssets`: supplied existing OBJ asset records, if the object refers to one.
  An unresolved asset is refused; no arbitrary URL is fetched.

The host supplies its existing generated material catalog as `assets`. It may
supply a baked `defaults` settings record instead of requiring request settings.
An explicit settings record replaces that record; there is no ambiguous merge
or separate hand-maintained copy of the page's defaults.

The existing Exposure-profile validator runs before object preparation and again
at the shared scene boundary. `studio_lighting_fields.js` contains declarative
control metadata only; its types, bounds, choices and defaults are checked against
the actual page HTML and material-select initialization. No pretend document or
control objects are constructed to execute the page.

`scene` returns JSON containing the actual fixtures, subject geometry, prepared
object, physical center and camera. `ray-payload` returns the existing MRP1
binary upload. The scene function does not create a job.

## Plain Node entry

`tools/studio_agent_product.mjs` accepts one JSON request on stdin and writes one
JSON result on stdout. Trusted startup selects `--assets FILE` and optionally
`--defaults FILE`. No HTTP listener is started.

The result is `{content_type, encoding: "base64", sha256, blob}`. It preserves
the scene or MRP1 bytes; the digest is integrity evidence, not authorization.
Requests and decoded results are limited to 32 MiB. Result transport needs room
for base64 overhead: configure the existing Nodus result `PayloadStore` with
48 MiB when returning the full scene. The default 32 MiB store is too small for
the tested 28.46 MB scene's JSON/base64 envelope.

`examples/meowdoku/studio_agent.py` supplies the bounded subprocess invocation,
checks the returned blob, and provides `commit_studio_ray`. That commit requires
a trusted authorization callback and existing job-owner scope, then calls the
existing `PhotoJobs.start` with the unchanged MRP1 bytes. Its result is the
actual job status, not a claim that an exposure finished or was queued.

For example, with a trusted catalog and a complete request prepared by the host:

```python
import json
from pathlib import Path
from meowdoku.studio_agent import StudioAgentProducts, product_bytes

worker = StudioAgentProducts(Path("assets.json"))
product = worker.build(json.loads(Path("request.json").read_text()))
Path("product.bin").write_bytes(product_bytes(product))
```

This writes the prepared bytes only. It does not call `commit_studio_ray` or
consume a GPU. A configured `defaults=Path("settings.json")` file contains the
settings record, not an entire request, and permits a request containing just
the schema and operation (plus optional photo options).

## Existing Nodus delivery

`studio_product_nodus.py` adapts the existing `EnginePeer`, `ServicePeer` and
`PayloadStore`. The owner chooses an already-authorized connection/prefix.
`Contract.engine` remains the producer identity, not a destination or permission.

The service dispatch callback only prepares a blob. A ray job is created in the
producer's authorized commit callback after the native result is validated.
Repeated terminal routes retain the decision. An exception during commit is
held as uncertain and is not automatically retried. FIFO receipt is not producer
commit; payload release is explicit after reconciliation.

No browser participation credentials, account privileges or new service grants
are included in payloads. Authorization must come from the existing authenticated
HTTP/peer owner and be rechecked before ray-job birth.

## Limits of this change

This is an invokable product and a messenger integration seam, not a new public
endpoint or deployed pool. Binding an HTTPS address to an authorized Nodus
prefix/client, account-claim handling and live rollout remain host integration
work. There is no silent direct-execution fallback around that binding.
The earlier canvas-only `standalone_studio.js` entry remains separate and is not
imported or used by this product. No new shader choice is implied by this path.

The existing photo service still owns single-GPU busy refusal and job lifecycle.
This change does not add an exposure queue. It does not load live Studio controls
to infer defaults, modify the game engine, or claim that native CPU and GL
renderers are interchangeable.

## Verification

- `node tests/studio_agent_product_check.mjs`: actual captured Studio settings,
  absent DOM/WebGL globals, forbidden network, identical explicit/baked settings,
  exact scene/MRP1 fixture agreement and rejected unsupported fields.
- `python -m unittest discover -s tests -p test_meowdoku_studio_agent.py -v`:
  byte integrity, authorization, owner scope, exact existing job call and no
  automatic retry after an uncertain job error; actual Python-to-Node subprocess
  invocation and existing MRP1 packing from a portable authoring fixture.
- `tests/test_meowdoku_studio_product_nodus.py`: retained identities, native peer
  framing, authorization-before-work and producer commit behavior.
- Shared-builder parity checks compare the extracted output to the original
  scene-construction functions; see the accompanying hazard receipt.
- The optional native FIFO test uses `NODUS_STUDIO_NATIVE_DLL` to select the
  owner's existing native DLL. The verified local build was
  `nodus/build-bridging/Release/nodus_contract_service.dll`; an older local
  `nodus/build/Release` DLL stalled after result delivery. Neither was rebuilt
  or installed by this change.

## Explicit camera sequences

For `ray-payload`, `photo.cameraPoses` is an ordered list of 1..32 camera
**overrides**, packed as `camera_poses` in the existing MRP1 request. Each entry
inherits the common `settings.camera`; omitted controls keep that camera's
values. Allowed override fields are `eye_m`, `target_m`, `fov_degrees`,
`aperture_radius_m`, `focus_distance_m`, `aperture_blades`,
`aperture_rotation_radians`, `lens_selection`, `camera_back` and
`back_reference_fov_degrees`. Positions are scene metres. Eye and target must
remain distinct in the sensor's float32 look-at basis.

The existing API resolves every entry through the same numeric bounds,
source-authenticated analytical lens selection and physical-back crop used for
a still. A selected lens derives FOV and pupil; clear `lens_selection` explicitly
to use direct FOV/aperture controls. A supplied FOV replaces the inherited back
reference unless an explicit reference is supplied. Resolution and exposure
seconds are common to every frame. No projection, per-frame resolution or tile
window can be supplied as a pose.

Sequences cannot be combined with orbit, tiles/oversampling, live view or device
intersections. Existing `orbitFrames` 8/16/32 remains the parametric orbit mode.
All frames run serially inside the same owned Spectral worker, using its existing
native camera housing update, sensor and snapshot path. `status.total_frames`
is the requested pose count, and completed `frames` retain each camera and image
(and RGB HDR) receipt. Multi-frame sequences use the existing `orbit.avi` product
name and AVI encoder. An AVI failure retains completed frame files and receipts.
There is no pose editor or second queue/renderer in this change.

Checks: `tests/test_meowdoku_camera_pose_sequences.py`,
`node tests/camera_pose_sequence_check.mjs`, and Spectral's
`tests/test_marble_camera_pose_sequences.py` (CPU doubles, not GPU evidence).

## Host integration: pictures through the shared REST API (2026-11-11)

The binding this document left open now exists in source. It is not live until the service restarts with a canvas
that declares the Studio prefix; see `docs/hazards/20261111T133801Z_main.md`.

- Route: `POST /api/show/pictures {request_id, settings?, photo?}` answers 202 with a receipt;
  `GET /api/show/pictures/{id}` reads it. The handler bodies are `picture_route` in the SpeakToMe repository's
  `speaktome_api/routes.py` (API_VERSION 2). Capability `photo_submit`, and a signed-in account.
- Host: `examples/meowdoku/studio_pictures.py`. The server is the host peer of a dedicated prefix (`studio.product`;
  its dispatch is `StudioAgentProducts.build`) and, per request, the engine peer whose producer commit calls
  `commit_studio_ray` with the existing `PhotoJobs`, the account's saved objects and owner keys, then the existing
  account claim. The request is always `ray-payload`; `scene` is not offered over HTTP.
- Authority: the account, owner keys and a recheck come from the authenticated request. The recheck runs before
  preparation, in the dispatch authorization, and again inside `commit_studio_ray`. Nothing in the body is identity.
  The product's SHA-256 is integrity evidence only.
- Settings: a supplied `settings` record must be complete (object, the fully resolved Exposure lighting profile,
  camera in metres) and replaces the baked defaults whole. `--studio-defaults` names the baked record.
- Receipt states: queued, preparing, then committed (with the real job record), refused, failed or uncertain. An
  error raised by the commit is recorded as uncertain and is never retried.
- Not carried: the client work-sharing bridge options of the HTTP photo path.
- Configuration: `--studio-nodus-port`, `--studio-nodus-prefix`, `--studio-assets`, `--studio-defaults`; the
  supervisor supplies the port from the instance's `studio_prefix`. The canvas file must declare the prefix
  (nodus `tools/add_contract_port.py`); the result store is 48 MiB.
- Checks: `tests/test_meowdoku_studio_pictures.py` (the real Nodus round trip is opt-in with
  NODUS_HEADLESS_TEST_DLL). No GPU exposure has been made through this path yet.
