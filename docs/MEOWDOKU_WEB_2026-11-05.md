# Meowdoku on the web: 3D tiles, prebaked boards, Python state machine

Date: 2026-11-05.  `python examples/meowdoku/web_export.py` then
`python examples/meowdoku/server.py --port 8766` and open
http://127.0.0.1:8766/.  Verified in the browser: hearts turn, an X falls
into its cell, cats jump, spin and settle; a wrong cell costs a heart and
the heart shrinks away; four correct cells placed through the page kept
three hearts.

## Pieces

- **Shader.** Pluck's base-material Phong shader, as Turing's
  `glsl_project_ir` compiled it for WebGL 2 (`spectral-analyzer/csrc/
  shaders/generated/base_material.webgl.*.glsl`).  Its SSBOs became
  `turing_ssbo_load_*` reads from R32F textures; the page uploads the
  material tables exactly as `MaterialDatabase.build_tensors` lays them
  out, plus the identity UV stacks and an empty field volume, and sets
  every uniform the compile left without a default.
- **Models.** The same glyph extrusions the scene orders describe
  (`scene_orders._glyph_triangles`), exported as flat positions and face
  normals with a material index per role: cat 4,188 triangles, X 1,736,
  heart 3,160.
- **Boards.** `boards.json`: per size (5 to 10) and style (free, relaxed)
  ten region maps, each with the leaf the game scores against (the sampled
  walk's for free boards, the constructed cat map for relaxed), its
  ring-relaxed hues (`tiles.ring_hues`) and its taxonomy record and level
  key (`tree.classify`).  The level picker is a query over that database.
- **State machine.** `server.py` runs one `MeowdokuSession` (the dt-system
  law) behind `/api/new`, `/api/move`, `/api/undo`, `/api/reset`;
  `MeowdokuSession(puzzle=...)` starts from a prebaked board.  The page
  posts clicks (unprojected onto the board plane) and animates the
  difference between states; it never decides anything.

## Light and shadow

The exporter compiles a second fragment with `#define PLUCK_SHADOW_MAP`,
the shader's own opt-in (`export_shadow_fragment`).  Each frame the page
renders a depth pre-pass from the key light (orthographic, covering the
board) into a 2048x2048 depth texture array and sets `uShadowMatrices[0]
= lightViewProj * model` per draw, since the shader shadows from
`vPosObj`.  The key light is a world-space point orbiting the board, drawn
as a bulb; `Light (L)` makes it follow the pointer.  Buttons are truncated
pyramids lit by the same program with one material per region.  Two
uniforms the newer source adds without ES defaults, `uExposure` and the
thermal table, are set explicitly (0.42, off); unset they render black.

## Animation

Per piece a model matrix from its age: cat `y = 1.6 sin(pi t)` over 1.1 s
with a full turn about y from 25% on and a squash at landing, then a
breathing scale; X falls 3 units with an ease-in and a small bounce over
0.55 s; hearts turn at 0.8 rad/s with a slow bob; removals shrink out over
0.3 s.  All of it is in `web/app.js`; the simpler engine that could compile
to wasm would own exactly these curves and the state diff.

## Not here

- Only one game per server process (a global session).
- Free boards above 7x7 take 40 s each to catalog at export (the full
  tree per board); `--boards` and `--sizes` bound it.
- No sound, no touch gestures beyond tap and long press.

## Four games, one protocol (later on 2026-11-05)

`games.py` states: `cells` (palette name per button), `marks` (X marks),
`hints` (legal destinations), `selected`, `pieces` (stable ids; `cell` may
be `hud:k` for hearts), `palette`, `rack` (Wordle's letters), `input`
(`meowdoku`, `select`, `text`).  Verbs: `cell(r, c, button)` with primary /
secondary / paint / double, `text`, `submit`, `backspace`, `undo`, `reset`.
Chess and checkers light a selected piece's legal destinations before the
move is committed; Wordle colours its rack from the guesses.  The page
builds one Phong material per palette colour plus `@hint` and `@sel`
variants, draws pieces by mesh family (letters and chess glyphs stand,
discs lie flat), slides moved pieces, and runs every request through one
queue so replies apply in order.  Controls: game selector, tilt (straight
down by default), saturation (rewrites the material table), Light (orbit or
follow the pointer).  Meowdoku: tap or swipe for X, double tap for a cat.
Verified in the browser on a second server instance; the LAN server holds
one table, so a second player (or a test) fights the first.

## The chess network (2026-11-05, late)

`chess_net.py`: encoding 12 x 64 piece planes + side to move; `ValueNet`
(tanh hidden layer, tanh value head, White's outcome in [-1, 1]) with the
forward in AbstractTensor on numpy and the gradient from
`AbstractTensor.autograd.grad`; `SelfPlay` picks moves by valuing every
legal successor position in one batch (softmax at a temperature while
learning), trains after each finished game on discounted outcomes plus a
quarter-weight material term, and saves weights every ten games.  Measured
from scratch: 300 games in under two minutes; against a random mover 6W 8L
2D before and 5W 3L 8D after.  The page's `chess_arena` plays network
against network non-stop (a tick per 350 ms while on screen; record, loss
and the last result in the status); `chess_vs_net` answers your moves.

## Full rules, batched arena, pre-training (2026-11-05, later still)

`games.Chess` now filters every move by the mover's king's safety (no
moving into check, forced evasion), ends on checkmate or stalemate, and
has castling, en passant and promotion.  `chess_net.BatchEngine` steps
sixteen games per forward pass; `ChessArena` runs it in a thread under the
table lock and shows game 0, and the page polls continuously, pacing its
slide to the measured move interval.  Measured live: 22 to 29 plies per
second per game with sixteen in flight, 49 games in the first 20 s.  Under
the full rules almost every early game is a 160-ply draw; the material
term is what teaches at this stage.

`pretrain.py` replays master games (pgnmentor.com Kasparov and Carlsen,
about 2.3 MB zipped, cached in `build/meowdoku/pgn/`) through a SAN reader
on `games.Chess` and trains the value net on them before self-play; the
arena calls `ensure` once at its first start and writes
`chess_net.pretrained.json`.  The download itself was not run from this
session (the sandbox refused the fetch); the first arena start on the LAN
server performs it, and `MEOWDOKU_NO_PRETRAIN=1` opts out.

## Training and search after the live Lichess stall

The live server kept serving Lichess after its background learner stopped.
Lichess inference and the learner share one `ConvValueNet`, while
`AbstractTensor.autograd.no_grad()` is process-wide. A training step inside
that scope was measured to fail with `No gradient found for input`. The
network now serializes evaluation, training and checkpoint access. A learner
failure is shown in the page status and printed with a traceback. The old
server was stopped at the user's request; the revised server has not been
started on port 8766.

The network has a third, residual 3x3 convolution. Its weights and bias start
at zero so existing eight-tensor `chess_conv.npz` checkpoints retain exactly
their old predictions. Both the old and new checkpoint layouts load; the new
layer receives gradients. Lichess and play-the-network search evaluate all
legal replies, then add a third ply for the three best candidates from the
two-ply pass. The deeper leaves are valued in one batch. Chess no longer
permits a king capture as a legal move.

The existing ProcessGraph/LLVM path compiled a small version of the network's
forward graph, but reverse compilation stopped in the shared backward-rule
registry at an unadmitted `Raise` in `index_adjoint`, before reaching the chess
network. Routing the same reverse graph toward C stopped at that same
backend-independent boundary, before C emission. Compiled training is therefore
not installed. The local NumPy training
step for 256 positions took 0.237 seconds on this machine after the added
convolution; a standalone learner advanced the cached corpus from epoch 0.02
to 0.12 in 25 seconds with its worker alive.

The registry's missing exception-conversion handoff was repaired afterward:
`index_adjoint` keeps its guard as a concorded native exception operation.
The small chess reverse graph now passes the registry and stops at a later
training-motion lowering refusal, `conditional ownership requires ingested
control activity`. Neither C nor LLVM training has been emitted yet. See
`docs/hazards/20261105T090043Z_main.md`.

## Lichess idle control and later reverse frontiers

The page now has **Stop Lichess after game** and **Resume Lichess** buttons in
Lichess mode. A stop request lets the current game finish, declines new
incoming challenges, and cancels an outgoing challenge. It stops seeking
once idle; the local chess learner continues. The server was restarted on
port 8766 with these controls, and both actions were checked through HTTP.
After restart the live status showed pre-training epoch 0.75 and 50 self-play
games, so the learner was still advancing.

The generated training motion now declares its empty control activity with
motion-root evidence. The linker no longer writes a redundant raw
polymorphism trace, and `detach` retains its input tensor type and shape in
region planning. A small native reverse probe passed all three former
refusals, then stopped at an untyped host scalar `ts` in `unbroadcast`.
Compiled training remains unavailable; the running server trains with the
existing Python/NumPy path. See `docs/hazards/20261105T092023Z_main.md`.

The `unbroadcast` class handoff now reads the source evidence for the nested
`enumerate(zip(...))` loop. The helper's `tuple[int, ...]` annotation covers
the empty target shape, which has no element to inspect. The small native
reverse probe passed that refusal and reached a separate fused-motion
activity error: its forward `unfold2d` node has no ingested evaluation point.
Compiled training is still unavailable. See
`docs/hazards/20261105T101433Z_main.md`.
## Lowering library (2026-11-10)

`examples/meowdoku/lowering_library.py` catalogues the lowerings (wasm, wgsl, c, llvm, fortran, spirv, cpython) of one
computational core in the same SQLite file the server already opens (`accounts.db`). Two tables: `lowering_cores`
(core_id, family, name, source_function, logical_abi_digest, source_digest, extraction_contract_sha256, first_seen) and
`lowerings` (core_id, backend, contract, entry, api_path, abi_path, product_path, product_sha256, abi_schema, arch,
verdict, scanned). Rows come only from each product's declared `turing-compiled-program-api-v1` YAML plus the JSON the
writer pairs with it (`<stem>.abi.json`, or the `*.manifest.json` record naming the `.wgsl`); verdict is `built` or
`described`, never `verified`. Scan: `python examples/meowdoku/lowering_library.py scan <web_dir> --db <accounts.db>`;
read: `... list --db <accounts.db>`. Server integration (scan at startup into accounts.db; `/api/lowerings`) is pending
in the live tree; server.py is not in this branch.


### Blobs, versions, selection, leases

`scan` also stores each built product's bytes in `lowering_blobs` (content-addressed by `product_sha256`, `kind` = file
suffix; `store_blobs=False` skips it), so a lowering row always resolves to bytes via `blob_bytes`. `versions(core_id,
backend)` lists lowerings newest first; each distinct sha256 is one version. `select_lowering(purpose, capabilities,
prefer=, contract=)` takes `purpose = (family, name)` (a core's computational purpose is its declared family+name) and
the set of backends the caller can execute, and returns the newest built lowering by `prefer` order then the fixed order
(c, llvm, fortran, wasm, wgsl, spirv, cpython), or None; it never falls back to a non-executable backend. Role/tier
rights are not interpreted: server authz passes only what the account may execute. `checkout` issues a `lowering_leases`
row under `BEGIN IMMEDIATE` with a per-blob generation and deadline; `checkin` succeeds once, on matching account and
generation; `expired` lists overdue unreturned leases (never deleted); `outstanding` counts open leases per blob (the
contention signal). CLI: `store`, `versions`, `checkout`, `checkin`, `leases`. The HTTP surface
(`/api/lowerings/checkout`, `/checkin`) is pending in the live tree.
