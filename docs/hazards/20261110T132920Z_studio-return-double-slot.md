# Original game frames returned to Studio, source only

Measured on connected Windows in C:/dev/Powershell/turing. New isolated
mechanical_creature_studio_return.js consumes the actual bridge's copied
mechanical-creature-studio-state-v1 packets. Complete validated frames publish
through two same-thread retained slots; no SharedArrayBuffer, interpolation,
physics step or additional clock is introduced. IDs, metre radii, tags and
lifecycle remain original-game data. Registered rigid placement is inverted
and metre coordinates divided by the authored Studio unitM for display.

Epoch, original RAF time, native worker sequence and world geometry revision
guard stale frames. Environment request sequence orders changing capture IDs;
authored source revision may repeat. A staged mapping preserves the previous
body frame until the exact registered environment receipt appears. Old receipts
are refused afterward. A collider receipt means posted to the original worker,
not applied by its solver; pending receipts are not upgraded by observation.

Existing hydrateStudioGameAsset and createStudioSubjectRenderer own asset
validation and GPU rendering. The return draw uses identity lightTransform:
captured ammunition Phong illumination stays in Studio coordinates rather than
following each body. Cross-object transparent sorting is approximate and is
not merged with Studio's furniture triangle sort. Original projectile physics
publishes no quaternion; fixed authored orientation is explicit.

Passed final focused checks in 0.91 seconds total:
- Module syntax check.
- meowdoku_mechanical_creature_studio_return_check.mjs: ten actual original
  lifecycle functions and the actual bridge publisher; copied immutable frames,
  stale/epoch gates, same-revision environment change, inverse pose mapping,
  sleep/wake/pickup/removal and identity lighting. GPU boundary explicitly mocked.
- mechanical_creature_studio_bridge_check.mjs against BOTH matched and published
  aa6cb6fb897d24e6bf769ca2dbfde3a0d3d0dbcdd64ddb87fd30173bc0f0d5e2 pages:
  original TuringWorldRegistry, native mesh/collider builder, actual launcher,
  actual queued table edit and native collider post, actual publisher to return
  consumer, acceptance of the new receipt and refusal of its stale predecessor.

No browser validation, native physics integration claim, live publication or
full repository suite was performed by this slice.
