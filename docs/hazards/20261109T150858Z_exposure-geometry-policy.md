2026-11-09 EXPOSURE GEOMETRY POLICY (SOURCE ONLY, Windows checkout)

Exposure now starts with a prominent triangle budget and sphere coverage
selector shared by photographs and live rays. Budget is an integer from
1000 to 1000000, default 250000. Exact analytic spheres was initially disabled.
After the renderer owner explicitly confirmed both-direction GPU/root/media
and real-photo gates, the parent authorized promotion: Exact is enabled and
explicitly selected on fresh pages. Saved triangle mesh remains available.
The wire packer and Python boundary carry only triangle_budget and
sphere_coverage; omitted coverage remains mesh. The MRP1 position, normal,
material-ID bytes and saved mesh SHA verification remain unchanged.
The UI states that older designs must be saved again for exact metadata and
that coatings, painted bands, non-spherical shapes and emissive shells are not
supported by Exact. Rejection never silently selects or retries mesh.

Worker geometry_stats is displayed independently from current inputs:
sphere_coverage, triangle_budget, input_triangles, effective_triangles,
analytic_spheres. No input-derived estimate is presented as an accepted count.
Legacy exposures explicitly report that counts were not recorded. The exact
scene cap and authenticated analytic metadata admission belong to the renderer
worker's marble_spheres/studio_photo_geometry implementation.

Invalid budget/policy edits do not upload or cancel a photograph. Valid edits
also leave an active still photograph unchanged. Live input validation preserves
its current scene during incomplete/invalid edits and cancels a pending valid
debounce when a later invalid edit arrives. The budget is rechecked immediately
before replacement. Display-only tint/normalization, checkpoint/history state,
Auto/manual bounce controls, firefly retouch, and underlight UI are preserved.

Same-session live scene replacement no longer pre-cancels the old worker before
POST. The server's canonical final-scene preflight can reject a syntactically
valid but insufficient budget without losing the accepted scene. Only an
explicit HTTP400 in the still-current enabled generation recovers the prior
digest/pose/revision; it refetches the old image and keeps the error visible.
It does not resubmit every 250ms. A new relevant edit can retry. Network/timeout
errors remain uncertain and use the existing stop/cleanup path. Off cannot be
undone by late success or rejection. Obsolete newly created jobs are canceled,
while reuse of the same active digest preserves its camera revision.
While the error is visible, heartbeat controls and image checks use the accepted
camera/live snapshot. A rejected resize to lower resolution therefore cannot
send that resolution's larger history capacity into the old larger sensor.
The final lifecycle gate proves the old scene keeps running and displaying,
then the pending resolution/history/tint values apply after a valid retry.

Measured: 46 policy Python cases cover strict types/ranges, photo/live defaults,
saved-hash rejection, unsupported legacy Exact, initial worker diagnostics, and
preflight rejection without stopping the old photo/live process. The final
combined boundary/Auto/live/mesh decode gate passes 71 tests (one existing cffi
deprecation warning). The new CPU preflight is mocked only at its own narrow
method in lifecycle tests; actual geometry preparation is the renderer gate.

Node checks pass geometry validation/wire/markup/diagnostics, 37 production photo
lifecycle cases, 21 live cases plus history boundary matrix, Auto/manual bounce,
all 200 numeric bounces, camera/room packing, ten-light strengths, profiles,
table and gobos. Python/JavaScript syntax and scoped tracked whitespace pass.
No browser, actual HTTP/API request, GPU work, export, restart, saved-state edit,
commit or publication was performed by this worker. New helper is registered
in the normal asset copy list and Studio no-store list; no export was run.

EXACT_SPHERES_READY is now true after the explicit readiness report, not from
an optimistic browser feature claim. The renderer owner separately verified
actual fixed3+Auto and fixed1 photos with nine exact spheres plus mesh
compatibility. This worker reran the final policy, still-photo, live replacement
and Python boundary gates after promotion. Production publication and combined
browser checks remain with the parent; this task did not perform them.
