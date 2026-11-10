# Hidden native startup and renderer staging

This is tested source groundwork, not an enabled Studio mode. Production assets
and the active user's scene were left unchanged.

## Existing scene capture

`captureCurrentStudioScene()` is exported by `web/marble_studio.js`. It returns
`{revision,state}` where state has schema `turing.studio-visible-scene.v1` and
explicit `channel: "studio"`. It captures the displayed affine subject pose,
view, actual lens FOV, complete geometry and material data, exact authored
environment faces, stable component IDs and room bounds. The ray coordinate
conversion is `rayOriginM + StudioVertex * unitM`, preserving Y-up axes.

Room/table/backdrop/gobo builders accept optional source callbacks. Source
part IDs are `room/<plane>`, `table/slab`, `table/leg-i-j`, `gobo/<id>`,
`backdrop/layer-i-wall`, `backdrop/layer-i-sweep`, and `fixture/<role>`. A shared
body ID is retained for compound construction. Metadata is not a declaration
that a compound can be summed as separate native contacts. Native admission
must check exact topology, numeric conversion and manifold ownership.

The current Studio room is independent of the saved Exposure profile.
`sceneForPhoto` is not reused or changed. Hidden room produces `room:null`.
Debug glyphs and the old compiled toy course are rejected. Non-off preview
motion has null native velocity fields until a supported velocity transfer
exists. Static/rolling mode uses the displayed motion state and the incumbent
rolling law's no-slip velocity. No model/camera/lighting settings are changed.

## Native and dynamic packet

The native worker owns `ComputeRuntime.preparePausedFirstFrame`,
`studioCapturedPhysicsFrame`, `studioDisplayedStart`, and
`studioDynamicGeometry`. The first runs one existing recorded snapshot on a
fresh detached real native runtime and remains paused after exact readback.
Use `disposeAfterIdle` for native cleanup. Product identity, scene digest,
complete state and topology admission must succeed before any presentation.
The dynamic helper emits one immutable packet with the same transformed
subject for Phong and ray geometry; it does not emit a ready receipt.

## Real Phong staging

`stageCurrentStudioPhong(capture,dynamic,{signal})` uses the actual Studio
WebGL2 context and incumbent mesh uploader, material partition, bubble filter
and transparent sorting. It stages invisible world-space meshes, fences GPU
uploads, reads back every attribute and glass index, and restores prior GL
bindings. Its receipt contains `ready`, `revision`, `step`, `bundle`,
`geometry`, `release`, `coordinateSpace:"studio-world"`, `visible:false`.
`ready` becomes false after release or context loss. No draw or installation
occurs. Material textures remain the current Studio's, requiring a final exact
scene check. Furniture glass is included in the staged world-space glass sort.

The renderer owner must use world-space view transforms for these buffers,
avoid drawing the former subject/table glass twice, and retire them only after
their actual reader ownership ends. Do not apply the subject model twice.

## One unchanged-scene boundary

`createStudioPhysicsTransition` requires `captureScene`, `prepareCandidate`,
`stagePresentation`, and `releaseCandidate`. Both Phong and ray must return
actual ready receipts for the same capture revision and paused native step.
It compares exact detached data after every asynchronous step and immediately
before its one-use synchronous `takeReady()`. Released/lost renderer readiness
is checked again there. Cancel/supersede waits for in-flight work before
releasing unshown candidates. Adapters own cleanup if they throw before
returning partially allocated resources. A transferred visible bundle is no
longer owned by the coordinator.

The host still must install both consumers at one frame boundary before any
old rolling update, preserve the existing camera, and continue the real native
runtime only after that commit. There is deliberately no UI hook or fake ray
acknowledgment in this change. The ray worker's new sweep-boundary adapter must
be reached through a real transport, initialized with this exact Studio scene,
not the separate Exposure worker. Its unsupported optics/media/static changes
must remain refusals. Reuse the existing thick-lens `SceneCameraCoordinator`
and `on_dt_request` seam for temporal slices; no second scheduler was added.

## Evidence and limits

Twelve focused Node suites and seventeen native mesh/certificate tests passed.
The actual shared-builder scene test retains all 1,384 source triangles.
Real Chrome RTX3060 staging uploaded 672 actual marble triangles into thirteen
buffers, verified exact readback and unchanged framebuffer/bindings, and safely
retired a canceled candidate. `build/dot-studio-hidden-20261109/browser-result.json`
records that bounded result. It is not proof of complete dynamic rendering,
whole-scene contact support or continuous ray-traced motion.

Asset export/no-store names are registered in source only. Do not publish the
shared bootstrap until native and ray integration, current source dependencies
and final both-consumer reveal have passed their own end-to-end gate.
