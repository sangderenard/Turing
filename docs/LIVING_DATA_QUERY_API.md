# Living Data graph REST and narration

Status: implemented locally, not published. This supports graph inspection and account-owned
graph-cursor navigation. It is not a mechanical actor service or a substitute
world. Graph movement does not advance physical simulation.

## Route

POST `/api/living-data/query`, Content-Type `application/json`, with the existing
account-session cookie and `dev_ui` capability. Share-view links are refused.
There is no new authentication system, database, background clock, URL fetch,
source execution, compilation, or renderer.

Request fields:

- `map`: the existing `abstract-ui-div-map-v0` object or its compact graph projection.
- `at`: an exact known identity in that map, not a guessed coordinate.
- `layer`: `layout` (default) or `world`.
- `radius`: graph hops, 0Ã¢â‚¬â€œ4; default 1. Both incoming and outgoing declarations count.
- `limit`: node page size, 1Ã¢â‚¬â€œ256; default 64.
- `offset`: node page offset; default 0.
- `edge_offset`: edge page offset over the full selected horizon; default 0.

Keep request bytes at or below 4 MiB. Maximum 4096 known nodes and 16384 distinct
edges. Full compiled pages carry large binaries and normally exceed the body
limit. Send only identity/kind/name/type metadata, declared dependencies, layout
positions, nested regions/buildings/rooms, tracks and world.objects. This is a
projection of the existing model, not another map schema. Nothing is retained
between calls; the caller owns graph version consistency across pages.

Example request:

```json
{"map":{"schema":"abstract-ui-div-map-v0","identity":"world:example","kind":"world","regions":[{"identity":"region:example","kind":"region","dependencies":[{"relationship":"contained-by","target":"world:example"}],"buildings":[]}],"tracks":[]},"at":"world:example","radius":1}
```

The response includes the selected node, paginated reachable nodes with distance,
declared edge source/target/relationship, next_offset and next_edge_offset.
Unknown edge targets remain `known: false`; they are not expanded. Node and edge pages are independent. Follow next_offset to collect nodes and
next_edge_offset to collect all edges, holding the input, at, layer and radius
fixed. An edge endpoint can refer to a node on another page.

The two model layers are deliberately separate. A compiled page can declare the
same identity as a semantic region and a physical courtyard with different
parents. `layout` reads regions/buildings/rooms plus tracks; `world` reads the
world object records. Each layer keeps its own declared containment. No hierarchy
is invented from nesting, labels, grid coordinates, or physical proximity.

Contained-by points outward; its reverse points inward. This is semantic
containment, not permission to enter a building. Physical up/down, walkability,
portal crossing, visibility, collision and reachability are unknown. Layout
column/row/width/height are returned as layout_position, not physical XYZ.

## Full participation is not implemented by this endpoint

The graph query is not an actor's perception, state, or narration engine. Its
knowledge is only what the caller supplied. It returns advance=false and
render=false. It never advances the catalog game or the compiled Living Data
worker, and is safe to call without requesting a frame.

The actual Living Data worker is emitted by `src/compiler/state_loop_deployment.py`.
Its current `setEngineStage` arms setInterval; `tick` loops over all bodies and
executes the original resident physics. The REST server's catalog Table is a
different runtime, so connecting this graph to Table.session would be a false
integration. No arbitrary state-tensor writes or pose teleports were added.

Next authoritative integration must expose that same instance's validated input
and tensor identities, ownership and revision checks. Actor-only advancement
must select the actual actor update while leaving vehicle and other body state
unchanged; full-world advancement must retain the existing shared-world laws.
The current worker has no verified actor-only command. A full participation
control-translation layer, blind-accessible controls, physical actor sessions,
request deduplication for advancement and optional frames remain unfinished.

## Verification

`python -m pytest tests/test_living_data_cursor.py tests/test_living_data_query.py tests/test_meowdoku_worldmap.py tests/test_issue_tickets.py -q`

Final aggregate: 48 passed and 10 subtests passed. Python syntax checks passed.
No full repository suite or public service deployment was performed.

The graph projector was also exercised directly on the actual compiled model in
build/dot-oriented-arena-candidate-20261110/index.html: radius 4 from the world
identity returned 5 semantic nodes and 101 physical nodes (paginated). This was
an in-process read, not HTTP publication or simulation verification.

## Stateful graph narration and movement

These routes use the same authenticated `dev_ui` admission as the query route:

- POST `/api/living-data/sessions`: `{map, at, layer?, request_id}` creates a cursor.
- GET `/api/living-data/sessions/{session_id}`: read text status and available exits.
  Optional `?offset=64` pages the exit list (64 per page).
- POST `/api/living-data/sessions/{session_id}/move`: take a declared graph edge.
- POST `/api/living-data/sessions/{session_id}/close`: `{expected_revision}`.

An exit is already a movement command: `to`, `relationship`, and `direction`.
Copy the chosen exit, add `expected_revision` from status and a unique
`request_id`, and POST it to move. No free-form language parser is required.
The response includes human-readable `text`, exact identity, revision and exits.
Outgoing follows source to target; incoming follows target to source. This is
explicit graph browsing, not permission to cross a physical portal. Unknown
reference endpoints are available through query but are not enterable nodes.

For example, an exit can be:

```json
{"to":"karate:8","relationship":"friendship","direction":"outgoing"}
```

Its corresponding move is:

```json
{"to":"karate:8","relationship":"friendship","direction":"outgoing","expected_revision":0,"request_id":"walk-1"}
```

The cursor is account-owned and server-process-local. One cursor per account,
16 active cursors total, expiry after one hour without successful status/move.
Close the current cursor before creating another. Restart loses cursors; this
is not durable game saving. Uploaded programs/binaries are discarded, retaining
only the validated graph summaries and declared edge index. No actor or game
Table is created and no timer runs.

Same request ID + same body replays the original creation/move receipt without
moving again. A replay receipt can describe an earlier revision; GET status for
the current position. Different content under a retained ID gives 409. The last
64 successful move receipts are retained. Older duplicates cannot reapply their
stale expected revision. Concurrent moves from one revision admit only one.
Ownership failures return 404; stale revision or nonexistent exit returns 409.
Expiry/close ends the retry window. Creation retry does not refresh expiry.

Responses label `mode: graph-cursor`, `simulation_advanced: false`, `frame: null`.
There is no fake elapsed time and no claim that a mechanical body moved.

## Actual Karate Club REST exercise

`tests/test_living_data_cursor.py::test_karate_club_real_rest_walk` uses the
NetworkX canonical Karate Club source already used by
`src/demos/filtered_poisson_karate_demo.py`. It refuses the demo's synthetic
fallback. The 34 nodes and 78 undirected edges are projected to existing
Living Data identity records; incoming/outgoing traversal exposes both ends.

Actual HTTP calls to the production Handler class on an ephemeral loopback
server with temporary account/state roots performed:

1. Create at member 0: HTTP 201, revision 0.
2. Reject nonexistent direct edge 0 â†’ 33: HTTP 409, no movement.
3. Move 0 â†’ 8 â†’ 33 â†’ 8 â†’ 0: HTTP 200, revisions 1â€“4.
4. Retry every move: HTTP 200 replay receipt, no duplicate movement.
5. Read current position: member 0, revision 4; close, then read returns 404.

Every position's returned neighbors is checked against the actual Karate Club
graph. Other tests verify cross-account refusal, admission, input immutability,
capacity, expiry, conflicting retries and racing stale moves. No live endpoint
or previously denied public network route was used.
