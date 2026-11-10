# OBJ library backend, source only

Implemented in obj_assets.py, with focused integrations in server.py,
marble_objects.py and marble_photo.py. No publication, production database write,
owner guess, cookie inspection, server restart or commit was performed.

Assets retain exact submitted OBJ and optional MTL text. Parsing calls Spectral's
existing camera_designer.obj_loader.load_obj, with text inputs only. mtllib never
causes filesystem or network reads. Normalized face order, source material IDs,
PBR16 and material names are retained. Owner ID plus name and exact sources form
the immutable digest; identical private uploads by different accounts are separate
assets. The existing Accounts claim/owner_of registry is authoritative. Uploads
are private; the HTTP API has no mutation, deletion or publication operation.

GET /api/obj_assets lists public and current-account assets. GET
/api/obj_assets/{digest} supplies geometry; /source and /mtl return authorized raw
text. POST /api/obj_assets requires an authenticated account and edit tier, takes
JSON {name,obj,mtl?}, and defaults to private. Limits: 16 MiB request, 500,000
triangles, 128 assets per owner. POST cannot set owner or visibility.

Imported snapshots use meowdoku-imported-object-v1 and base-shape=obj. Derived
mechanics must explicitly be unavailable. Saves, reads, lists and inline photo
snapshots check source access. Photo admission compares the entire ordered
position/flat-normal/material packet to the referenced source, not just a
client-supplied hash. The existing Studio shell override supplies all face IDs;
source materials remain in the asset. No new rendering path or solid repair.

## Public bunny seed procedure (not performed)

1. Verify the real owner through the existing signed-in /api/account/me response.
   Do not guess the account ID or read authentication secrets. Verify the desired
   deployment's accounts.db and obj_assets directory, and the reviewed backend
   source used by that deployment. Deployment/restart is a separate operation.
2. Use that real account and existing UI/API to POST the supplied OBJ and MTL,
   name "Stanford Bunny". Verify the returned private asset's source_sha256 is
   1eb35d1e21ce99e5ce911353b6be278990713448dd9e8f5c9387f9de39b32205.
   Source files are in C:/Users/alber/scratch/stanford-bunny-trial/:
   stanford-bunny.obj, crystal.mtl, README.txt. Read the README's exact credit and
   terms before publication. Keep the source unchanged.
3. In an explicitly authorized operator context, instantiate Accounts with that
   deployment's accounts.db, and obj_assets.Store with its obj_assets directory,
   Accounts and C:/dev/Powershell/spectral-analyzer. Call:
   store.publish(returned_digest, verified_account_id, attribution=credit_text)
   Credit text must include Stanford Computer Graphics Laboratory / Stanford
   3D Scanning Repository, the source URL and the research/redistribution terms
   from README.txt. Commercial use requires permission; do not imply otherwise.
   publish rechecks actual registered ownership and rejects an owner mismatch.
4. Verify anonymous list, geometry and source GET succeed with visibility public,
   owned false, readOnly true; verify unrelated private assets remain hidden.
   The source SHA must still match. The source has 69,451 triangles and known
   boundary holes, not certified solid geometry or mass.

## Measured validation

- tests/test_meowdoku_obj_assets.py: 9 passed, including isolated real HTTP account
  sessions, anonymous/cross-account hiding, private source downloads, public
  publication, bounded uploads, source/hash forgery rejection, null mechanics,
  and JavaScript/Python packet equality.
- Existing marble objects, photo and exposure snapshot tests: 35 passed.
- Actual bunny: all 69,451 triangles retained. Python and JS packet SHA with shell
  ID 0 both e9088539ec2c21223f11a62525064fb8b89d52bce1bb2594f333348d4e6904d8.
- py_compile and focused git diff whitespace checks pass.

Full rendering, browser UI and production seed are not claimed by these backend
checks. Existing dirty edits were retained. Do not publish whole dirty source
files. The hazards helper refuses an existing malformed filename; newest entries
and the baseline were read directly before testing.

## Additional anonymous-consumer check

Isolated explicit public fixture passes source validation and both retained-object
and inline-snapshot photo decoding with account_id=None. The same two paths reject
the private source with 404. Anonymous HTTP metadata/geometry/source GET succeed;
anonymous upload remains 401, operator publication with no verified owner fails
403, and all source storage bytes remain unchanged. These are account-anonymous
consumers within the existing permitted network tier. The shader-tier API gate
and photo edit-tier requirement are unchanged. No live permission activation,
public seed, account provisioning or rendering was performed.
