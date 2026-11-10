# Wide light supports, backdrop holder and native camera mounting

Requested increment: floor-mounted wide softboxes should have supports on both
sides; the backdrop should have two stands and a crossbar; the physical camera
should have a stand without changing its selected pose or inventing a body.

## Light and backdrop support slice

The dual support threshold is an explicit physical width of four feet, derived
from the fixture width and Studio unit conversion. This is a visual support
policy, not a mechanical load rating. Two opposite sides use the existing closed
pipe construction. Ceiling-mounted upper fill remains on its existing cables.
The two tripod assemblies total 480 triangles. A room-fit failure rejects the
whole assembly with the light identity and reason, rather than hiding one side.

The backdrop holder derives its extent from actual visible backdrop vertices.
Two posts, six feet, two short arms and one crossbar produce 528 triangles.
The crossbar's contact with the top edge uses the actual polygonal pipe surface,
including the near-wall placement case. No backdrop layer is moved.

Both features share the existing support catalogue material. The new
lightStands.wideSupports and lightStands.backdropSupport leaves default true for
new rigs and false when absent from saved rigs. Ordinary Room B leaf overrides
remain independent of Exposure A. No saved profile is rewritten on load.
Both Phong and native photo packing use the same faces. Backdrop frame parts
belong to the room group. Fixture IDs and original emitter rows are unchanged.

Source and staged tests pass for real default fixture/layer geometry, all ten
light placements, room bounds, outward normals, float32 component closure,
material mapping, profile migration, full UI roundtrips and byte-identical
Exposure photo geometry when only Room support overrides change.

The five-asset release is narrowly staged over the published main in
build/dot-wide-backdrop-20261110. It includes marble_studio.js,
marble_light_stands.js, marble_light_stand_settings.js,
marble_wide_light_supports.js and marble_backdrop_supports.js. Publication is
complete after rebasing over the camera worker's DOF release. All five TLS asset
hashes were verified using the local Home CA with hostname verification; 269
unrelated files remained byte-identical. No restart occurred. The publication
receipt is build/dot-wide-backdrop-20261110/publication.json. Published main:
eb937fb85ba5409523b2c221b2ed35ba9022d2e1f9760936612cc92a97104567.

## Native camera mount and transport

marble_camera_mount.py exports the actual canonical thin_lens_housing_preset,
thin_lens_housing_frame and build_gpu_scene float32 vertices, with a geometry
hash, version and normalized pose/pupil settings. No camera mesh is guessed.
The nominal housing has 45,088 triangles and an approximately 8.2 MB JSON record.

marble_camera_mount.js consumes that exact record, explicit Studio metre scale,
origin, room bounds and shared material ID. A mounting plate, column and three
feet form five closed components and 240 triangles. The support plane touches
the real lowest native housing vertices. Missing or stale geometry and a floor
or room that cannot fit the hardware return an explicit unattached result.
The camera pose is never edited. This is support-plane contact, not certification
of a mechanical mounting thread, stability, or general prop collision freedom.

A dedicated source-only POST /api/marble_camera/housing route accepts one
explicit camera object. It requires the existing edit tier and rejects viewer
access. It permits at most 4 KiB input, the existing ten-metre coordinate and
pupil domains, at most 100,000 housing triangles and a 16 MiB JSON reply. It uses
a two-entry, 24 MiB serialized cache and admits only one native build at a time.
Invalid input, unavailable source, oversized results and busy state produce
explicit errors. It does not launch, stop, cancel or mutate an exposure.

The response explicitly permits mount-envelope use only. Native material and
face ownership are not exported, so the vertices must not be presented as a
truthfully shaded replacement housing. Native camera geometry is already in
the photo backend and must not be appended a second time as room fixtures.

Fifteen focused Python tests pass, including byte equality to an independently
built native camera, the JavaScript mount contract, real ephemeral HTTP/native
response, invalid input rejection, concurrency/size/cache limits and edit-tier
guarding, oversized integer rejection and authoritative lens-pupil selection.
No production restart or endpoint publication was performed. Camera
pose/render integration remains owned by the camera worker. A coordinated
retained-products backend release is required when no exposure is interrupted.
