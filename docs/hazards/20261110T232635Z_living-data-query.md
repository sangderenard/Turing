# Living Data graph query REST slice (2026-11-10, Windows)

Added only POST /api/living-data/query, examples/meowdoku/living_data_query.py,
focused tests and docs/LIVING_DATA_QUERY_API.md. Existing issue/worldmap and
other concurrent changes were retained. No compiler, worker, physics, account,
network, live process or published asset was changed.

Validation: python -m pytest tests/test_living_data_query.py
tests/test_meowdoku_worldmap.py tests/test_issue_tickets.py -q:
45 passed, 10 subtests passed, one existing cffi deprecation warning.
Python syntax compilation passed. Full repository suite not run.

An in-process query of the actual embedded model in
build/dot-oriented-arena-candidate-20261110/index.html returned 5 layout nodes
and 101 world nodes at radius four from its world identity. Semantic and
physical records reuse identities with different containment; the query requires
an explicit layer and does not collapse them. Layout positions remain layout
coordinates, never XYZ. Full compiled input is approximately 14.7 MB and exceeds
the 4 MiB HTTP limit: callers must supply the compact canonical graph fields.

Actor-only and full-world advance, owned tensor updates, headless worker hosting,
stateful actor sessions and optional frame capture are NOT implemented.
The original worker still owns its setInterval/fixed-dt whole-world loop.
No substitute game or numerical update rule was introduced.

hazards_log.py refused the pre-existing malformed
20261109T_material-library-paused-checkpoint.md filename. Recent named receipts
and the legacy baseline opening were read directly before focused tests.

Independent review found cross-node-page edge loss and large-integer float
coercion. Both were fixed at the query projection, with direct/HTTP regression
coverage and a clean independent static recheck. Final aggregate rerun above
includes those fixes. Edges now paginate over the full selected horizon.


## Follow-on: actual graph cursor navigation

User requested basic graph narration/movement and an actual REST walk through
the Karate Club test network. Added account-owned, revision-checked graph
cursors and GET status / POST create, move and close routes. The existing
GraphView identity and edge projection remains the source; there is no physics
or physical actor state. Requests reuse account auth, reject viewer access,
isolate ownership, bound retained graphs/receipts and expire idle cursors.

Actual isolated HTTP calls against the real Handler exercised NetworkX's
canonical Karate Club source already used by filtered_poisson_karate_demo:
34 nodes, 78 undirected edges, no synthetic fallback. Create at 0 returned 201;
0 -> 8 -> 33 -> 8 -> 0 returned 200 at revisions 1 through 4. Direct 0 -> 33
returned 409. Repeated requests did not move twice. Every exit set matched the
canonical neighbors; explicit counts exclude duplicate narration. Accounts,
state roots and port were temporary. No production/public network call occurred.

Final combined cursor/query/worldmap/issue regression: 48 passed, 10 subtests
passed, 1 cffi deprecation warning. Python syntax and server diff checks pass.
Independent security review found no blocking ownership/concurrency issue;
the symmetric fixture's duplicate exits were corrected to one edge per
friendship with returned incoming/outgoing direction. Cache/expiry/retry
semantics are documented in docs/LIVING_DATA_QUERY_API.md.

This finishes the basic graph-walk demonstration, NOT physical actor-only or
full-world advancement, tensor input integration or graphics-on-request.
