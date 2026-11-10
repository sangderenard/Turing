# Combined Studio analytic release, 2026-11-09

Published the user-requested next batch to the existing production Studio on
port 8766 after the active 180-second photograph completed normally. No user
exposure was cancelled or interrupted. The separate development server was
not touched. Lifecycle shutdown stopped PID 66316, boot 8e845898b767; the
recorded existing launch configuration started PID 60912, boot 39f828156e81.
The server restored one of one sessions.

Verified URL:
https://173.195.26.104:8766/marble_studio.html?v=analytic-20261109

## Included scope

- Exact analytic spherical boundaries in forward/camera transport, declared
  air-bubble media, saved geometry authentication and hard triangle admission.
  Fresh pages select Exact; omitted legacy requests retain mesh. Unsupported
  painted/coated/non-spherical designs are rejected, never silently replaced.
- Auto bounce pilot with separate 1000 probe-depth and 200 render-depth caps,
  persistent-tail warnings, preserved Manual control and corrected explicit
  medium continuation at all three scattering sites.
- Air bubbles and Advanced two-sided gobos, preserving old saved designs.
- Expanded live history, checkpoint sums/counts, tint and normalization,
  resolution-dependent 2 GiB budget and red warning above 512 MiB. Safe live
  replacement preserves the old scene on rejected geometry.
- Explicit underlight height, fresh default -1.60, preserving saved poses.
- Opt-in non-destructive isolated-highlight filter with threshold/strength,
  finished-HDR retouch, pipeline metadata and genuine-highlight warning.
- All prior table, ten-light, room, material, Phong and film controls retained.

## Verification

Combined focused checks: 191 Turing Python tests passed; 138 Spectral Python
tests passed with two optional GPU skips. All 21 applicable Node suites pass
after updating the older film-response fixture to the real Auto/geometry DOM
and explicit cancellation contract. That fixture correction did not alter
production behavior. Syntax and scoped tracked whitespace checks passed.

Separate actual GPU gates are recorded by each owner: analytic intersections,
media transitions/TIR, fixed1/fixed3 and mesh worker photographs; ring and
checkpoint numeric/allocation tests; and the final Auto pilot. Auto tested
diffuse (chosen depth 8), slow mirror (189), trapped TIR and a glass/air bubble
(both cap 200 with explicit persistent-energy warnings). The pilot preserved
production field bytes and the production depth uniform. No GPU workers
remained before lifecycle restart.

29 frozen static files matched TLS-validated production responses. All 18
recorded backend source files retained their final tested SHA256. The assets
JSON, including existing geometry and material IDs, is byte-identical to the
previous release. New modules return no-store to prevent stale mixed clients.
The restarted Film Science CPU endpoint accepted the exact filter fields.

Actual Chrome production boot loaded all material options and dynamic
lighting/table/gobo modules, showed Exact/250000, Auto, 30 seconds, bubbles
0..64, checkpoint on and underlight -1.6. Film Science rendered both panels
and both hue histograms, reporting the preview current. No fake QA banner was
present, no design was saved, and no production photo/live job was started by
this publication test. A browser-extension checksum error was unrelated to
the app; no application alert was present.

## Limits

Exact applies to authenticated uniform nonemissive uncoated spherical shells
and declared bubbles, not arbitrary parametric solids. Older objects need
resaving for analytic metadata; otherwise explicitly select mesh. General
intersecting-material unions and mechanical mass remain limited. Phong does
not show air bubbles or trace nested refraction. Auto is a sparse transport
pilot, not an image-error/convergence guarantee. Full 2 GiB stress and broad
200-bounce watchdog/performance are not claimed. The filter can suppress a
genuine isolated glint; raw HDR remains unchanged. Old unprofiled HDR keeps
its legacy output recipe. No full-repository CI claim or git push is made.

## Receipts

Publication scripts, before/after launch identity, source manifests, backups,
TLS hashes and endpoint results are in
build/dot-studio-analytic-release-20261109/.
Analytic GPU receipts: spectral-analyzer/repo-cache/exact-spheres-20261109/.
Auto GPU receipt: spectral-analyzer/build/dot-auto-bounces-20261109/gpu-result.json.
Live checkpoint receipt: spectral-analyzer/build/dot-live-history-20261109/gpu-check.json.
Bubble receipt: build/dot-air-bubbles-20261109/FINAL_RECEIPT.md.
