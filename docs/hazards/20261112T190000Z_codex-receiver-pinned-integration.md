2026-11-12 PINNED CANONICAL INTEGRATION, CPU ONLY:

Owned receiver2103aa04 merged pinned canonical
 d030e18d7224df39f3499645bde850f79c44ed77 without conflict as
 a4102c1940493e3a5edcc2ac6d9ef1667f3004f9.
Committed API overlap adds trigger generator validation; packer overlap adds
trigger generator forwarding and photo preview controls. Receiver explicit
queue/camera modes and v1/v2/v3 validation remain intact. No live activity.

Passing CPU verification:
- tests/test_meowdoku_forward_ancestry.py,
  tests/test_meowdoku_camera_pair_delivery.py,
  tests/test_meowdoku_marble_photo.py:54 passed with installed Python311 and
  SPEAKTOME_REPO/PYTHONPATH/STUDIO_SPECTRAL_TEST_ROOT overrides, including
  process-local SPECTRAL_ROOT/decode_upload default override as prior receipts.
- meowdoku_camera_pairs_check.mjs, meowdoku_scene_record_index_check.mjs(18),
  meowdoku_scene_responses_check.mjs, meowdoku_studio_light_field_check.mjs,
  meowdoku_gem_cutter_check.mjs.
- actual producer serialization: studio_forward_ancestry_roundtrip.py and
  studio_camera_query_roundtrip.py.

Initial Turing.venv backend run:51 passed,3 failed because yaml was absent.
Installed Python311 already has PyYAML; rerunning all54 passed. No installation.
Existing Nodus-not-connected and CFFI deprecation warnings are not hardware proof.

Known remaining committed-tree test gap:
meowdoku_marble_photo_retouch_check.mjs fails at photo-panel.classList.toggle:
its old mock omits classList required by newly committed preview controls.
Read-only canonical dirty test includes that mock adaptation and additional
preview tests. These dirty edits were NOT copied or changed here.

Canonical dirty Stage Programmer/UI/lighting/geometry/export/test work remains
separate. It adds an untracked marble_stage_programmer.js import/export dependency;
its complete source/fixture bundle must be committed together by its owner.
No observed dirty changes to response codecs or queue/camera admission options.
This is source compatibility review only, not browser/render/energy validation.
No canonical edit, GPU link/render, server restart, push or deployment occurred.
