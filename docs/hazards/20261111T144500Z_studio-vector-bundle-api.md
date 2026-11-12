# Studio directional vector bundle API (source only, not deployed)

The existing owned `/api/marble_photo/field` route accepts
`studio.forward-field.v2` alongside v1. No server route, capability or ownership
exception was introduced. The scalar RGB prefix remains unchanged. V2 appends
the producer's six-bin RGB/direction array, original packed triangle identity
and classification records, baked source records, and native PBR16 materials.

`PhotoJobs.open_field` requires contiguous exact section spans, bounded counts
and dimensions, a total size at most 256 MiB, the current live revision, and
the immutable job's geometry hash. It validates the scene hash's syntax;
the scene hash names the producer's assembled scene and is not an independent
reconstruction of that scene at the API. It hashes the opened artifact and
checks finite float sections, sequential triangle identities, material index
bounds and known class bits before rewinding that same stream for HTTP.
V1 compatibility retains its existing metadata contract.

The browser must still prove its visible triangle/material correspondence.
Server packed triangle IDs are not browser mesh IDs. The source records do
not imply per-source atlas ancestry. This endpoint does not certify calibrated
radiance, a reflection solution, or equivalence of differently assembled scenes.

Tests cover both versions through real local HTTP, exact bytes and lengths,
owner/capability isolation, stale revisions, malformed section metadata,
invalid triangle/material/classification references, payload corruption and
nonfinite data with a matching checksum. The pre-change focused baseline was
27 tests passing. No live service restart or deployment was performed.
