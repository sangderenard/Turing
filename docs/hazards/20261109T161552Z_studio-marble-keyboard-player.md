2026-11-09 TITLE (MEASURED, where): 
# Focused Studio marble keyboard controls and Pluck reuse receipt

Added focused-canvas WASD/arrows to the existing Studio D-pad/native gravity
route. This is whole-toy gravity tilt with Marble perspective, not a newly
implemented rigid-body player. New marble_drive_input.js and its Node test;
narrow edits to marble_studio.js input/drive wiring, one HTML help sentence,
and the web_export.py asset list. No object/room transform, native numerical
law, physics snapshot/handoff, live ray lifecycle or server endpoint changed.

The Node input suite passes aliases, diagonal normalization, key repeat,
focus/form isolation, modifier/composition handling, hidden-page/window/canvas
blur, disabled mode, cleanup, pointer precedence and the actual Studio
driveGravity adapter with the existing cameraNudgeGravity function. Both JS
syntax checks, Python export compilation and scoped git diff whitespace check
pass. Existing sim-preview Node suite is blocked on a missing generated
web/turing-wasm-shell.js import; no synthetic shell added and no full pass
claimed. No browser, GPU, production export, server restart, commit or push.

Pluck's installed native _spectral_kernels.PlayerClipEngine was exercised:
wall correction, velocity clipping, frame generation and tensor history pass.
Its player controller is kinematic and the clip engine is not reciprocal body
dynamics. The real server timing seam is CameraManagedTimeBridge; the client
mechanical demo's fixed-step worker has wrench/impulse routes. Their exact
reuse boundaries and remaining native contact/timing requirements are in
examples/meowdoku/PLUCK_PLAYER_STUDIO_HANDOFF_2026-11-09.md. No misleading
functional body toggle was exposed before those requirements are satisfied.
