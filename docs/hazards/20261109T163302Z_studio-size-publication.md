# Studio size, placement, display and live workflow publication

Published 2026-11-09 at the existing authorized Studio endpoint after a fresh
process check found no photo/live worker. The guarded restart checked again
immediately before requesting lifecycle restart. No exposure or live history
was canceled. Old PID 60912 / boot 39f828156e81 exited. New PID 36180 / boot
712c18d8b10f serves 34 verified frozen static assets; 18 backend source hashes
remain unchanged. One prior session was restored. User/development tabs and
other servers were not changed.

URL: https://173.195.26.104:8766/marble_studio.html?v=size-20261109

## Delivered

- Adjustable percentile normalization and specular-shot clipping, in the shared
  CPU display recipe, completed-image retouch, Film Science and live controls.
  Specular starts at percentile 95 and forces linear luminance clipping. Raw
  HDR remains unchanged; old unprofiled HDR retains its documented legacy path.
- Table X width, Z length and floor-to-top height; independent stage-left/right
  arc/radius shortcuts; relative top-fill height; fresh-only retreat defaults
  for kickers/background. Existing authored poses are preserved.
- Physical marble-only radius, fixed Studio units, table contact, exact/bubble
  object-center transforms and explicit Frame marble. Old lighting profiles
  preserve their previous physical photo conversion and are labeled legacy.
  Materials and normalized meshes are not rewritten when radius changes.
- Immutable object autosave with failure/retry handling, JSON recipe download,
  original RGBA ray-image download, transparent unsampled/pending ray overlay,
  and actual total forward-ray batch control from 128 to 65,536 per sweep.
- Focus-safe WASD/arrow input for the existing marble-perspective native toy.
  This is not a shared-room rigid-body player implementation.

The existing published compute.js was deliberately retained at SHA256
 ebc1196da03676585ca201c360e2b1a72c65b98aff1288dabe14dbc2114a56c7.
The separate inactive compute_state/native-contact groundwork is not in this
visual release. Existing assets.json geometry/material IDs are byte-identical.

## Verification

Combined current-source gates: 130 Turing Python tests passed, 171 Spectral
Python tests passed with two optional skips. 32 Node suites passed: 29 direct
Marble suites, two Film Science/response suites, and the existing simulation
preview suite with its imports pointed at the retained published compute.js
and actual generated WASM shell. The direct source invocation cannot import
that generated shell because it is not a canonical source file; no stand-in
loader or simulated physics was substituted.

Actual isolated Chrome geometry checks verified table dimensions, stand
footprint warnings, independent polar placement and Advanced pose agreement,
5 mm marble shrinking without moving the camera/table, 0.1 mm subject-relative
fill height reaching Y=-0.99, oversize warnings, explicit framing, and legacy
conversion freezing the current unit. Browser QA found an uncaught mass-display
error during blank radius input; the actual event path was corrected and
retested. Invalid input now shows validation while preview holds its last valid
size, and valid input resumes mass and visual updates. The test tab was closed
and its autosave preference restored. No real job was started in this fixture.

Autosave/export/alpha browser receipt:
 docs/hazards/20261109T162300Z_studio-autosave-live-overlay.md
Scale actual MRP1/Spectral preparation receipt:
 docs/hazards/20261109T162153Z_independent-marble-scale.md
Percentile actual saved-HDR/browser recipe parity:
 docs/hazards/20261109T155300Z_percentile-browser-verification.md

After restart, TLS HTTP responses matched all 34 static hashes. All new modules
returned no-store; retained compute.js matched its deliberate hash. The actual
CPU Film Science endpoint accepted specular percentile 50 and reported linear
clipping. Actual production Chrome boot showed all new controls, defaults and
no QA banner, rendered the Phong view, and completed both colour panels/hue
histograms through the restarted endpoint. No object was saved and no live or
photo render was started by publication QA. The QA tab was closed.

The new live batch/coverage path has focused CPU loop/output coverage; no new
GPU exposure was run for this batch. Earlier exact/media/history/Auto GPU gates
remain separately recorded, not represented as fresh tests of every new path.

## Remaining limits

The generic object test still has its separately reported d6 proxy<low
assertion mismatch and was excluded from this focused green count. An exported
Chrome object exactly matches its saved bytes, loads in the same Chrome, and
passes real server validate_object/decode_upload admission. Its full/proxy
hashes match Node, but medium/low LOD regeneration differs in Node. The origin
has not been measured against clean pre-feature source. Cross-runtime loading
therefore remains unverified; no geometry tolerance or speculative fix was
added. There is currently no local JSON-file import UI.

Phong remains approximate and does not show air bubbles/nested ray refraction.
Exact spheres are limited to authenticated eligible sphere/bubble designs.
General intersecting-material unions, full shared-room physics, player body
and gentle stage courses are not delivered by this visual batch. A very large
marble can overlap the set; warnings do not claim complete collision checking.
No full-repository CI pass, new broad GPU performance guarantee, commit or push
is claimed.

Manifests, test logs, before/after lifecycle identity, backups and HTTP receipts:
 build/dot-studio-size-release-20261109/
