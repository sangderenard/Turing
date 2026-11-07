# Meowdoku layout audit

Organisation only. No file's logic was edited; the moves, the path expressions that count directories, the import
lines and the shims are the whole change. Mover: `tools/meowdoku_layout_move.py` (idempotent, staged).

## 1. Classification (79 python files, 27 page files)

| Group | Package on disk | Files | What |
|---|---|---|---|
| Generic law substrate | `s1_engine/` | 7 | entities, entity_motion, sphere_motion, sweep, walkers, world_view, roads |
| Bespoke game / content | `s2_titles/<title>/` | 34 | classic (games, puzzle, tree, words, program, shader, page, session, play), chess (chess_net, mcts, search, native_forward, lichess, pretrain), common (characters), scroller, platformer, sidescroller, invaders, delve (delve, delve_rules, delve_world, delve_plan), tilequest, worldwalk, boardwalk, purrline, mobroad, fountain (fountain, fountain3d, fountain_game, glyph_start), tipjar (+ `tipjar_fixture.json`) |
| Training | `s3_train/` | 7 | autopilot, idle_trainer, pilot_decl, pilot_law, pilot_teachers, pilot_train, pilot_variants |
| Generation / compile | `s4_generate/` | 5 | graph_wasm, policy_wasm, split_lowering, fountain_build, fountain3d_build |
| Export / assets | `s5_export/` | 4 | web_export, tiles, phong_tiles, bake_tiles |
| Serving / runtime | `s6_serve/` | 8 | server, startup, loading_anim, lifecycle, share, access, profiles, realtime_socket |
| Probes / parity / one-offs | `s7_tools/` | 14 | `*_probe` (6), `*_parity` (4), split_probe_source, profile_matmul_lanes, gallery, aesthetics |
| Page runtime (pure, no game) | `web/` (not moved, see 5) | 21 | access, bodies, compute, game_clock, gravity, lifecycle, loading_anim, loading_panel, loading_record, noise_filter, octet_codec, particle_styles, pilot_clock, pilot_timeline, recording, replay_layers, share, wheel_blur, world_curve, world_zoom `.js`, split-runtime `.mjs` |
| Page game modules | `web/` | 3 | mobroad.js, tipjar.js, wordle_layout.js |
| Page shell | `web/` | 3 | index.html, app.js (7078 lines, every game's page code), shader.html |

Judgement calls that differ from the brief's list: `program.py`, `shader.py`, `page.py` are the original Meowdoku board's
own law/shader/page (board-specific constants) so they sit in `classic`, not in generate. `native_forward.py` compiles the CHESS
net's evaluation (imports chess_net, games), so it sits in `chess`. `pilot_decl.py` is a registry of per-game declarations
(imports delve, platformer, sidescroller, tilequest), so it is train, not engine.

Mixed files that cannot be split without editing logic (listed, not touched): `fountain.py` / `fountain_build.py` hold three
products (default+wheel, purrline, mobroad) between `# <purrline>` / `# <mobroad>` markers; `games.py` holds the `Game`
base class AND every classic game; `scroller.py` holds the law, the enemy brains and content; `app.js` holds every game.

## 2. Pipeline order (disk order = `s1` ... `s7`)

```
s1_engine ──┐
            ├─> s3_train ─> s4_generate ─> s5_export ─> s6_serve ─> web/ runtime ─> web/ game modules
s2_titles ──┘   (weights)   (products,      (build/       (HTTP,       (page runtime)   (page games)
 (source laws)               manifests,      meowdoku/     sessions,
                             digests)        web, flat)    sockets)         s7_tools: probes/parity, depend on anything, nothing depends on them
```

## 3. Import graph between groups (module imports, lazy ones included)

```
s1_engine   -> s1 5   s2 4   s3 2
s2_titles   -> s1 32  s2 47  s3 2
s3_train    -> s1 1   s2 12  s3 13  s7 2
s4_generate -> s2 3   s3 2   s4 2
s5_export   -> s2 12  s5 4
s6_serve    -> s1 2   s2 16  s3 3   s4 4   s6 7
s7_tools    -> s1 4   s2 21  s3 1   s4 1   s5 3   s7 1
```

### Violations (an earlier stage importing a later one, or a module importing a probe)

Generic -> bespoke / upward (10):
- `entities` (engine) -> `games` (the `Game` base class), `session`, `characters`, `autopilot`, `pilot_decl`
- `sphere_motion` (engine) -> `scroller` (title)
- `scroller` (title) -> `autopilot`, `pilot_law` (train)
- `pilot_decl`, `pilot_teachers` (train) -> `delve_probe` (a probe: the probe holds code the pilot needs)

Cross-title (17): everything outside `classic` imports `classic.games` (base class `Game`); `classic.games` imports the whole
chess stack; `mobroad` -> `fountain`; `tipjar` -> `fountain3d`, `fountain_game`, `glyph_start`; `worldwalk` -> `scroller`.

Composition roots (downward by design, but they name every title): `server` (29 module imports, every title), `web_export` (13),
`pilot_decl`, `autopilot` (scroller, invaders), `graph_wasm` (scroller), `lifecycle` (fountain, games).

Page side: `compute.js` (runtime) imports `mobroad.js` (a game module); `app.js` imports every game module (shell, fine).

Bespoke reaching into internals: `glyph_start` -> `fountain3d._f32`, `fountain_game._pack` (private names);
`fountain_build` / `fountain3d_build` read the title's module-level tables (`F.SIZES`, `F.ARGUMENTS`, `F.STATE` ...) directly; `LawPiece.source_modules` names modules by dotted string
(`"meowdoku.walkers"`), so a rename must go through a shim (it does: the string imports the shim, which returns the moved module).

Root cause of most of them: the `Game` base class lives in `games.py` next to the classic games, and `EntityGame` (engine)
subclasses it. Splitting `Game` out is a logic move; left for the owner (see 6).

## 4. Dead / unused (git grep, not guesses)

No file is dead. Entry points reached only by command line: server, web_export, play, gallery, aesthetics, bake_tiles,
pretrain, pilot_train, lifecycle, the probes and parity scripts. Referenced by NOTHING (no importer, no test, no doc, no tool)
but their own docstring:
- `pilot_variants.py`, `aesthetics.py`, `bake_tiles.py`, `worldwalk_parity.py` (CLI one-offs)
- `split_lowering.py`, `split_probe_source.py`, `web/split-runtime.mjs`, `split_check.mjs`: a self-contained set, not imported by server/export, not exported by `web_export`, no test. Not deleted.
- `profile_matmul_lanes.py`: cited by one hazards entry only.
Held by tests only (not by any pipeline code): the `*_probe` / `*_parity` scripts, `pilot_train`.

## 5. Layout, and what was NOT moved

```
examples/meowdoku/
  s1_engine/  s2_titles/{classic,chess,common,scroller,platformer,sidescroller,invaders,delve,tilequest,worldwalk,boardwalk,purrline,mobroad,fountain,tipjar}/
  s3_train/  s4_generate/  s5_export/  s6_serve/  s7_tools/
  web/            page source (flat, see below)
  <old>.py        79 shims (import -> the moved module itself; script / python -m -> runs it as __main__)
```
Shims: `import meowdoku.games` returns the very module object of `s2_titles.classic.games` (monkeypatch, `inspect.getsource`,
`unittest.mock.patch("meowdoku.x.y")`, pickles of old class paths all keep working); `python examples\meowdoku\server.py ...`
and `python -m meowdoku.lifecycle supervise` run the moved module as `__main__`, so launch.json argv from the supervisor is unchanged.
A `__getattr__` forwards for a reference taken while an import cycle was in flight.

`web/` stays flat. Every page module imports its neighbours as `./x.js`, and the export is flat; the tests copy source files
flat into temp dirs. Moving them into `web/runtime/` and `web/games/` needs the export to rewrite import specifiers (a new
mechanism) or the source to carry `../` specifiers the flat export cannot resolve. That is a decision, not an organisation
step: left. The exported dir (`build/meowdoku/web`) and its URLs are untouched; `web_export` copies from `HERE.parent / "web"`.

## 6. Digests

Product digests hash SOURCE TEXT (`inspect.getsource` of modules / functions), never file paths or module names. A pure move
changes nothing. Text had to change only where a path expression counted directories:

| Stage | Files | Digest effect (measured: `fountain_build.digest/purrline/mobroad`, `fountain3d_build.digest`, every `LawPiece._captured_sources`) |
|---|---|---|
| 1 (67 modules) | none of the hashed files edited | all 53 digests IDENTICAL |
| 2 | entities, entity_motion, scroller (`parents[N]`) | the 10 LawPiece digests change (native/numpy law products rebuild); fountain digests identical |
| 3 | fountain_build, fountain3d_build (`HERE.parents[1]`) | all 40 fountain product digests change (one rebuild) |

Hashed set (their text must not be edited by a move): entities, entity_motion, walkers, sweep, sphere_motion, roads, delve_rules,
pilot_law, scroller, fountain, fountain3d, fountain_build, fountain3d_build and the law modules (platformer, sidescroller,
invaders, delve, tilequest, worldwalk, boardwalk, purrline, mobroad). Their `from meowdoku import x` lines were therefore NOT
rewritten; the shims serve them. Rewrite them in the same window as stage 2/3.

## 7. Tests run on the moved tree (`MEOWDOKU_LAW=numpy` except cannon)

Passed: access 34, boardwalk 11, cannon 6 (native, 4.5 min), chess_clock 6, dt_session 3, float_call_adapter 2, game_speed_js 1,
hints 3, lifecycle 12 (spawns the server through the old `server.py` shim), loading_anim 9, pilot_decl 20, replay_layers_js 1,
roster 2, shader 4, share 14, share_js 1, startup 6, tipjar 28, vertical_law 4, wheel_blur_js 1, wordle_layout_js 1, wordle_overtime 7.
Failed, unrelated to the moves (JS untouched / environment; NOT baselined on a clean checkout): emoji_style and world_curve_js
(the tests' own node scripts: `world_curve.js` not copied / `rl` undefined), pilot_clock (float spread of a wall-clock value),
tiles (`render_scene_order` subprocess: `MemoryError: bad allocation` in spectral-analyzer).
Skipped (hung / long native compiles, no per-test timeout): delve, fountain, fountain3d, game_speed, invaders, mobroad, page_laws,
platformer, play, program, purrline, tilequest, worldwalk, vertical-law siblings. `test_meowdoku_mobroad.py` pins the
default/wheel digests: run it in the digest window.
Server: `python examples\meowdoku\server.py --port 8940 --web-dir <scratch copy of build/meowdoku/web>` came up through the shim,
`/api/new` answered for meowdoku, chess, checkers, wordle, space_invaders, bullet_hell, platformer, side_scroller, tilequest,
boardwalk, worldwalk, fountain, fountain3d, tipjar, purrline, mobroad; delve did not answer in 160 s (not baselined); all
page files 200. The fountain lanes started rebuilding (stage 3 changed their digests). The browser pane refused localhost, so
no page was rendered. Killed by Windows PID.

## 8. Merge plan

1. Land NOW (independent of in-flight files): stage 1 (`c535e5c8`). 67 modules, no hashed text edited, all digests identical,
   shims everywhere. Rename conflicts only if someone edits one of those 67 old paths meanwhile: take their version, `git mv`
   it to the new path, keep the shim.
2. Digest window (one law-product rebuild; schedule with any pending law edit): stage 2 (`9156fe5b`) entities, entity_motion, scroller.
3. After mobroad/fountain, loading-studio and bug-hunt workers land: stage 3 (`0300bf66`) = server, startup, loading_anim,
   web_export, mobroad, fountain, fountain3d, fountain_build, fountain3d_build (one fountain rebuild). If their edits conflict,
   do not merge the commit: `git merge` theirs first, then run `python tools/meowdoku_layout_move.py --stage 3` (idempotent,
   it moves whatever is still at the old path and re-points the path expressions).
4. Then, optional: rewrite `from meowdoku import x` in the hashed/hot files to the new names (same window as 2/3), move the tests'
   imports, and decide the `web/` split (section 5) and the `Game` base-class split (section 3).
