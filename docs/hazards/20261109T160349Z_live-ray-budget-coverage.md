2026-11-09 LIVE RAY BUDGET AND SAMPLE COVERAGE (CPU-VERIFIED, Windows source)

Live settings accept ray_batch_size as a strict integer in 128..65536,
default 8192. It is the total primary forward rays submitted per full sensor
sweep, not per light/source and not the GPU dispatch chunk (still 512 rays).
The bounded cap limits queued work; no new GPU responsiveness benchmark was
run. Existing shader dispatch rejects overhanging workgroup invocations.

The live worker coalesces transport revisions until the next sweep boundary,
then resets both ring and checkpoint before pumping the new budget. Display
edits alone preserve history; returning to the active transport settings can
cancel a queued reset. Status reports the effective budget and exact cumulative
submitted live primary-ray jobs. Discarded preparation warmup is excluded;
ray jobs are not a count of unique baked directions, bounces or physical photons.

Existing forward voxel energy is additive and the sensor uses fixed _rfs/65535,
not a division by the ray count. Larger batches can change field brightness,
not only variance; direct-light/visible-emitter terms are separate. No new
photon normalization or other physics convention was introduced.

Live fixed:3 previews now use RGBA. Alpha comes from valid-sample alpha in all
three real sensor-layer readbacks, never luminance: sampled black is opaque;
untouched or camera-reset pixels are transparent. RGB bytes still come from
the unchanged shared display pipeline. Partial first-sweep snapshots show
coverage growth. sampled_pixels and coverage_fraction describe this coverage.
Live snapshots leave status publication to their owner, so each published
image/status has its effective live_revision rather than a transient base
photograph status. Still previews stay RGB with unchanged default publication.

Verification:
- Turing .venv Python: test_meowdoku_live_ray_batch.py plus
  test_meowdoku_marble_live.py: 40 passed in 0.44 s.
- Spectral system Python: test_marble_live_ray_batch.py,
  test_marble_live_coverage.py, test_sensor_history.py,
  test_marble_preview_worker_status.py, test_marble_color_profile.py and
  test_marble_film_response.py: 137 passed, 1 opt-in GPU test skipped in 6.09 s.
- CPU mocks execute the actual live loop and actual pump function body. Tests
  cover strict bounds, exact partial dispatch counts, per-sweep fields, seed
  continuity, latest queued revision, cancellation, ring/checkpoint resets,
  display-only retention, partial revisions, sampled-black alpha, all-layer
  coverage, reset coverage, still RGB and RGB display-byte equality.
- Python compilation and scoped tracked-file whitespace checks passed.

The Turing venv cannot import Spectral's neural_assembly because Torch is absent;
the speaktome venv has unavailable NumPy C extensions. Neither environment was
changed. Spectral's installed system Python completed the checks above.
No GPU, production request, active worker interruption, restart, export,
commit or publication occurred. UI/browser work belongs to the coordinator.
