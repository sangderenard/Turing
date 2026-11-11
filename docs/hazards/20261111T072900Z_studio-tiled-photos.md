2026-11-11 STUDIO TILED AND OVERSAMPLED PHOTOS (MEASURED, local Windows, base 61a32c63):

Custom/large dimensions and integer 1x..4x linear oversampling use the existing
PhotoJobs owner and Spectral worker, not a second renderer or job scheduler.
Resolution admission now permits these modes rather than the old closed preset
list. Existing ownership/active-job/live refusals are preserved. Final PNG/HDR
routes stream their verified file handles instead of allocating all bytes.

Turing focused photo/ownership/tile tests: 34 passed. Node tile payload/participation,
camera, and all 50 existing retouch lifecycle cases passed. Spectral has independent
coordinate, disk assembly and serial worker tests; exact artifact identities and
real GPU render evidence will be recorded with the requested render.

Not green: older participation test expects live+client-intersections rejection,
which its current source does not implement; same failure observed in canonical
checkout. Room check fails in unchanged Studio source because its fixture omits
shopEmissionProfiles. No claim of full suite, real GPU, deployment or public-route
verification in this receipt. No running server was touched by this implementation.
