2026-11-11 STUDIO PHYSICAL CAMERA BACKS (MEASURED, Windows, base 61a32c63)

Physical back presets/custom dimensions resolve to a centered optical gate
through existing FOV/axial-focus camera parameters. Pixel sizes, tiling, source
lens provenance and focus/pupil are independent. Still uses full output aspect;
Live retains its existing viewport-aspect shader contract. No Spectral or
compiler law change, live server change or deployment.

155 Python tests pass: test_meowdoku_camera_back.py, test_meowdoku_lens_contract.py,
test_meowdoku_marble_live.py, test_meowdoku_marble_photo.py. Node camera-back,
lens, photo-camera and camera-binding checks pass. Legacy photo-room Node
check fails shopEmissionProfiles undefined on this tree and clean detached
61a32c63 at C:/dev/Powershell/turing-back-baseline-20261111.

Initial test collection lacked the sibling SpeakToMe dependency at a short
worktree outside Powershell; moved only this owned worktree under Powershell
and used SPEAKTOME_REPO=C:/dev/Powershell/speaktome. No source change to obtain
dependencies. hazards_log.py still rejects pre-existing malformed filename;
read latest ten dated receipts and legacy baseline directly. Full suite and
new GPU exposure not run. Real Chrome fixture with actual panel HTML/CSS/module verified all presets,
custom invalid/recovered edits, reference reset and collapse/reopen. Found
binary-float APS-C label; fixed display rounding and added regression test.
Inspected screenshot build/camera-back-controls.png. Fixture uses a 2:1 camera
aspect; it does not exercise integrated WebGL/GPU alignment. Closed owned
loopback server/browser tab after the check.
