# Collated photo jobs

The existing authenticated `GET /api/marble_photo/jobs` response groups related
work under its original PhotoJobs digest. A child is a member of that job, not
another GPU owner or a queued job. Existing Nodus producer commit and PhotoJobs
busy/ownership rules are unchanged.

Admission writes `collation.json` beside the payload and config before starting
the worker. Version 1 records the parent digest, canonical config hash, typed
children and immutable membership:

- `tiles`: windows from Spectral's existing `tile_plan`/`tile_windows` contract,
  with internal/output dimensions and oversampling.
- `camera_pose_sequence`, source `orbit`: the existing orbit frame indices and
  frame count. The worker still owns the actual orbit camera calculation.
- `camera_pose_sequence`, source `explicit`: the normalized `camera_poses`
  records admitted by PhotoJobs, from 1 to 32 camera poses. Request entries
  override the common camera, then use the existing lens/back validation.
  Explicit poses cannot be combined with an orbit, tiling/oversampling, live
  view or cooperative intersections. This relies on the paired camera-pose execution
  support; the manifest alone cannot make an unsupported request executable.

There is no second mutable progress ledger. The Jobs response projects the
worker's persisted `status.json` onto these members. Legacy jobs derive the same
description from their saved config without writing a new manifest or adopting
ownership. An inconsistent manifest is reported, not silently rebuilt.

Tile completion means a completed sensor contribution, not an independently
downloadable tile. The current worker deletes temporary tile snapshots after
copying their means into its assembly storage. The completed parent supplies the
final PNG/HDR. A parent claiming completion without all contributions and final
files is displayed as an error.

Camera frames expose only existing, recorded image/HDR files. Unfinished groups
are interrupted after server restart; completed frame files remain downloadable
through the original ownership-checked image/HDR routes. Canceling still stops
only the active parent and keeps finished camera frames. A last progressive
preview is not promoted into a completed child. Missing child files cannot
produce a completed group.

The Jobs selector shows completed/total children. Its expandable group survives
polling, lists child state and exposes real frame links. Assembly remains
cancellable. There is no automatic restart, child retry, paging execution,
cross-job queue, new broker, or change to the renderer's scheduling.

Focused checks are `test_meowdoku_job_collation.py`,
`test_meowdoku_job_library.py`, `test_meowdoku_marble_photo.py`,
`test_meowdoku_marble_tiles.py`, and `node tests/meowdoku_jobs_check.mjs`.
