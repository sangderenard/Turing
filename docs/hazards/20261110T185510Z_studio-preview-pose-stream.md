2026-11-10 DISPLAYED STUDIO PREVIEW POSE STREAM (MEASURED, offline Windows).

The existing displayed-frame hook forwards lastStudioFrame.model, not another
animation clock or a full scene capture. The controller binds it to the admitted
asset/revision/environment sequence and negotiated epoch. The host rejects old
sessions. Room admission validates the finite positive uniform affine transform,
keeps only two reusable 16-value slots, and latches the latest sequence only at
the incumbent identities/render boundary. Both opaque and transparent passes
use that pose; arrival between passes waits for the next boundary. Replacing
the environment or stopping discards pending poses. No new physics or dt owner.

The original subject renderer retains its shared transparent VAO and existing
centroid sort. Only subject position/normal subranges are uploaded with
bufferSubData; static furniture, material IDs, group IDs and lighting remain.
Changing a pose invalidates the existing sort even with an unmoving camera.
Opaque subject model changes without geometry allocation. Bounds are computed
lazily on diagnostic reads for the current pose. The original immutable asset
continues to describe its captured base pose. poseSequence/appliedPoseSequence
in room.readStatus distinguish accepted and latched preview state, not solver
execution or completed GPU work.

Passed after final edits: 10 focused pytest cases (patch, negotiation, host)
in 1.19s; actual room lifecycle Node test; subject validation (21 negatives).
The new meowdoku_studio_pose_stream_check.mjs runs actual scene/subject code
with instrumented GL commands: streamed attributes/model/index order match a
fresh native draw plan, including fixed-camera resort across furniture, only
two subrange uploads, no new mesh allocation, preserved material/group IDs,
current bounds and unchanged-pose no-op. Room tests cover old revision/asset/
sequence, two-slot overwrite, input detachment, mid-pass arrival, rejected
nonuniform/reflected/singular transforms, composed Float32 overflow and
replacement/disposal. The room invokes the renderer's authoritative
validateSubjectPose before publishing a slot; a 1e100 translation cannot
poison the next render boundary. Host checks
epoch and controller checks no recapture or forwarding to an old revision.

The existing browser GPU harness now compares three streamed poses against
fresh native renderers and split phases. Its syntax passed; pixel tests were
NOT executed by this worker. Parent owns browser/live QA and restaging. This
worker touched no staging/live/server/auth/compiler/runtime utility files,
started no server and made no publication, commit or push. Existing shared
source edits were retained. Full repository tests were not run. Hazards helper
show was blocked by the existing malformed 20261109T_material-library-paused-
checkpoint.md name; current receipts and baseline were read directly.

Frozen production source SHA256 values:
examples/meowdoku/web/marble_studio_game_layer.js: 3aade9cbb0ad6f48b0c970c74b21691a40fb8d427c5ad85406f83d1d47bc7e0c
examples/meowdoku/web/mechanical_creature_studio_layer_host.js: 54c6c7a0ea95a04d5f039ebcc32ca4afea5d5e15e4361c6a22ec7070788f206a
examples/meowdoku/web/mechanical_creature_studio_room.js: 5837ec70c057a643837c2c9e2576f06dfae332701bd9320bf10488065fd71f7d
examples/meowdoku/web/marble_studio_scene_asset.js: 87550a5129eb99f5e7fbcc5db806c3781e938705bb1832a95896506472cedf4e
examples/meowdoku/web/marble_studio_subject_renderer.js: 1793f79b94b9931aefd550889388e484279da9d775025878a49c59f8b0ab1a4f
tools/add_studio_game_layer.py: eab5d96ecdc3e19ad0dea29bc3256016ea6d3ec638b876e000e9de3a25e671f7
