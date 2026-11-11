# Nodus web service (2026-11-10)

A READ-ONLY web view of the graph of a running Nodus canvas host. It shows structure (digest, modules, rows, edges, tools, activity),
never the canvas: no pixels, no layout, and no action that can change a canvas.

Code: `examples/meowdoku/nodus_service.py`, server block `[NODUS]` in `server.py`, page `web/nodus.html` + `web/nodus.js`.

## How the host is found
The Apache supervisor records each nodus instance in `service_instances` (accounts.db: `state`, `console_port`) and writes a per-run token
file at `apache_supervisor.console_secret_path(paths, name)` (imported, not re-derived). `nodus_service.locate` reads those two and nothing
else. An instance is "console available" when it is `running`, has a `console_port`, and the token file exists. The token is read for one
`HELLO nodus-console/1 <token>`, then dropped; it and its path are never in a response, exception or log line.

## Endpoints (JSON, `Cache-Control: no-store`, capability `dev_ui`)
| Request | Result |
|---|---|
| `GET /api/nodus/instances` | `{"instances":[{"name","state","console"}]}` |
| `GET /api/nodus/digest?instance=` | the host's DIGEST JSON + `"instance"` |
| `GET /api/nodus/graph?instance=` | GRAPH JSON + `"instance"` |
| `GET /api/nodus/tools?instance=` | TOOLS JSON + `"instance"` (`complete:false` / `missing` passed through) |
| `GET /api/nodus/activity?instance=` | ACTIVITY JSON + `"instance"` |
| `GET /api/nodus/abstract?instance=` | `{"instance","revision","nodes":[...]}`: the generic identity graph |
| `POST /api/nodus/share` `{instance,to,note}` | needs `dev_ui` AND `message`; one note message to a mutual contact |

`instance` may be omitted when exactly one nodus instance has a console. Status codes: 404 unknown instance, 400 ambiguous or bad input,
501 the host predates the verb ("not supported by this host"), 503 console not available or unreachable, 502 malformed console reply.
Each request opens, uses and closes one console connection (3 s timeout; a graph/tools request after an expired digest uses two).

Cache (per process): digest and activity 1 s per instance; graph and tools 5 s keyed by (instance, revision from the digest).

## Share
Sends exactly one message through `messaging.send` (mutual contacts only, its limits and refusals apply): type `note`, subject
`{"kind":"topic","ref":"nodus:<instance>@<revision>:<first 12 of sha256>"}`, plain-text body = the note (at most 500 characters) plus one
digest summary line. No links, nothing else is sent.

## Generic graph mapping
`as_generic_graph` is a second producer of the generic identity graph (nodes with `identity`, `kind`, `name`,
`dependencies:[{relationship,target}]`), alongside the class-mode and file-mode producers and the world model.

| Canvas element | kind | identity | dependencies |
|---|---|---|---|
| canvas | `canvas` | `nodus:<instance>/<canvas>` | none |
| module | `module` | `<canvas id>/m<i>` | `contained-by` canvas |
| row | `row` | `<module id>/r<j>` | `contained-by` module |
| contact | `contact` | `<module id>/c<j>` | `contained-by` module (a contact only an edge names is declared too) |
| edge | `edge` | `<canvas id>/e<i>` | `contained-by` canvas, `from-contact`, `to-contact` |
| contract port | `contract-port` | `<canvas id>/cp<i>` | `contained-by` canvas, `bound-to` its edge when it names one |

`contained-by` is the world model's vocabulary (abstract_ui_div_map.py, abstract_ui_world.py); `from-contact`, `to-contact` and `bound-to`
are new names for directed relations. Scalar host fields (exec mode, place, token bytes, slots, capacity...) ride in `attributes`.

## How Marble Studio or an agent should poll
1. `GET /api/nodus/digest` every 1-2 s. Compare `revision` (structure) and `tick` (activity).
2. Only when `revision` changes fetch `graph`, `tools` and (if wanted) `abstract`.
3. Fetch `activity` when `tick` moves and fill/counters matter.
Polling faster than the cache TTL costs nothing extra at the console.

## Not provided
No actions (nothing here can change a canvas, start, stop or edit anything), no pixels or layout, no token or token path, no links.
Hosts without the v2 verbs answer 501; `instances` still works.
