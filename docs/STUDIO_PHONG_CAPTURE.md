# Phong photos and video

The Studio's **Phong photos & video** controls save its existing displayed raster output without starting a ray job. PNG captures the next successfully completed frame. Video copies completed frames to a fixed-size canvas and uses the browser's MediaRecorder encoder. No shader, scene builder, renderer, microphone or screen capture API is added.

Capture runs after the normal primary, secondary material passes and depth-of-field composition. It captures only `#view`, so ray overlays, photo panels and controls are excluded. The same path records visual preview animation and the existing compiled particle view when those are active. A baked field already present may contribute to the canvas; capture itself never requests one.

## Controls and limits

- PNG and video retain the original drawing-buffer dimensions, including device pixel ratio. They do not rerender at a requested resolution.
- Select up to 24, 30 or 60 fps. Actual frame rate depends on the existing renderer, browser and encoder; no frames are synthesized.
- WebM VP9/VP8 is preferred; MP4 is used when it is the browser's available supported format. No audio is recorded.
- Stop recording finalizes the file. Resize, page hiding or context loss stops the current clip instead of stretching or silently switching dimensions. Valid partial clips from a lost context can be downloaded with a reason shown.
- A clip is limited to two minutes, 128 MiB, and a 16-million-pixel source. Encoding or admission failures release controls so the user can retry. An over-budget clip is discarded explicitly.
- Completed PNG and video links remain until replaced or the page closes. Downloads are local browser downloads, not server library entries. Back/forward-cache navigation stops recording but does not permanently dispose the capture controls.
- Browser support differs. Unsupported video leaves PNG enabled. No claim of iPhone/Safari recording support has been measured yet.

## Validation on 2026-11-11

Node controller tests cover PNG/repeated-click behavior, frame cadence, repeat recording, stop-before-first-frame, resize, time/memory limits, missing codecs, context loss, failed encoders and stalled PNG encoders. Focused Python capture, export-closure, existing agent, host and Nodus tests: 22 passed, two optional tests skipped. Separate explicit actual Chrome host tests passed as documented in `STUDIO_PHONG_HOST.md`.

Real Chrome, isolated authenticated Studio fixture, current source and canonical exported shaders:

- PNG downloaded and decoded at 1062 x 799. Framed subject image visually inspected.
- Two VP9 WebM clips downloaded and decoded. The rotation clip contains 47 decoded changing frames, with packet timestamps through 40.582 seconds; it is not a still-image placeholder.
- This instrumented fixture performs full-frame readbacks and many GPU probes, and delivered about one frame per second. This is not a sustained-30-fps performance result.
- With an actual delivered field active, browser validation observed primary plus passes 0/1/2/3, gain 1, no GL errors and DOF enabled, then generated a 1062 x 855 PNG. This verifies capture coexistence, not a visible field improvement; the sparse field itself had zero measured on/off pixel difference.
- Repeat record/stop and replacing PNG/video links worked. No ray job was launched for capture. The pre-existing extension checksum warning is unrelated to capture; no application capture/encoder error was observed.
- Receipt artifacts are under `build/dot-phong-capture-20261111/`. No live service restart or publication was performed.

The internal captured-scene self-hosting seam is separate and documented in `STUDIO_PHONG_HOST.md`. Its current resource limits do not restrict browser capture of the live final canvas.
