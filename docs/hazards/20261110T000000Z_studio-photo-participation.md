# Studio exposure client participation (staged)

Source exposure opt-in sends client_intersections only for RGB fixed:3, one still,
resolution <=256. Unsupported requests fail validation before upload. Checkbox
is initially off and unavailable for other modes. A returned client_join must
match the created exposure digest. Explicit opt-in activates the existing
ClientRayParticipation controls, using StudioBvhCompute with compiled WASM;
plain-bvh and region-bvh select separate staged kernel directories. Unknown
variants fail rather than substituting a law. No sphere arithmetic was moved.

Participant closes on explicit cancellation or terminal photo status; edits do
not cancel it. No private queue or scheduler introduced. Source HTML/module,
photo payload hook and exporter changed; nothing published or restarted here.

Passed: meowdoku_photo_participation_check.mjs, test_client_ray_participation.mjs,
meowdoku_jobs_check.mjs, marble photo camera check, 50 photo-retouch cases.
No production end-to-end exposure test or browser visual check yet. Publication
requires backend bridge and exact compiled assets plus a safe restart slot.

Rejoin UI added: active client-intersection jobs expose an explicit owner-cookie
POST /api/client_rays/join action. Returned identity must match the selected job.
Listing alone never joins. Participant persists across list refreshes, and is
closed on terminal/lost ownership entries or page disposal. Jobs UI regression
now verifies explicit-only joining, headers/body, refresh retention and cleanup.
Publish participation dependencies before the jobs page. Still not published.
