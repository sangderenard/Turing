# Claude branch continuation — 2026-11-07

## Current integration point

This checkout is `codex/meowdoku-claude-integration-20261107`, based on main `66e24497`. It combines the three layout stages with the committed log backend and page work. It has **not** been merged into main or deployed. The live 8766 process was not restarted.

2026-11-07 update: main now carries the log backend and browser/server log page. Live 8766 was restarted from its recorded Python command. The Studio all-size selector, visual-only capture, material catalogue, and cat-eye marble designer reached main and live 8766 separately. This branch retains the layout move and 3-D toy product; scratch WebGPU products for `small` and `4k` built successfully, with no Wasm compile. The opening paragraph and original branch inventory remain as the integration receipt at the time they were written.

| Integration commit | Source | Result |
|---|---|---|
| `457b7fbf` | `c535e5c8` | Stage 1 module moves; `s6_serve/share.py` contains current main's newer live-view implementation. |
| `9915c4fd` | `9156fe5b` | Stage 2 law module moves. |
| `59847930` | `2c40a2ae` | Committed server log ring, dev `/api/log`, and log page. |
| `df3ba7e7` | stage 3 mover, rerun after the log commit | Nine hot modules moved, including the updated server and web exporter. |
| `04f04eaf` | `0e7a19a7` | Original layout audit; its digest and test receipts describe the source branch at the time, not this exact combined tree. |

The stage 3 mover (`python tools/meowdoku_layout_move.py --stage 3 --dry`) now reports 0 modules. Its run also rewrote an import in `mobroad_pather_sim.py` and one in `s6_serve/share.py` in addition to the original stage 3 paths. The original audit's old merge plan is historical; use this file for the present status.

## Verification on this combined checkout

- Stage 1: `test_meowdoku_share.py`, `test_meowdoku_access.py`, `test_meowdoku_startup.py`, `test_meowdoku_roster.py`: 56 passed.
- Stage 2: `test_meowdoku_vertical_law.py`, `test_meowdoku_pilot_decl.py` with `MEOWDOKU_LAW=numpy`: 24 passed.
- Log commit: `test_meowdoku_logring.py`: 5 passed; Node module syntax checks for `log.js` and `log_view.js` and `tests/meowdoku_log_view_check.mjs` passed.
- After stage 3: compileall on the moved title/generator/export/serve packages passed; access, startup, share, logring and loading animation tests: 68 passed. `git -c core.whitespace=cr-at-eol diff --check` was clean. No compiler rebuild or live site test was run.
- The two stale Node failures listed in `LAYOUT_AUDIT.md` are repaired in the test setup: the world-curve check now loads the real replay-layer tint table, and the emoji-style check copies `world_curve.js` and its `world_zoom.js` dependency beside the module under test. Both checks pass (2 passed); no page source was changed.
- Main's November 5 uncommitted chess `search.py` was copied byte-for-byte into the moved `s2_titles/chess/search.py`, along with its untracked `test_meowdoku_search.py`, in this isolated branch. The three chess-search tests pass here. Main's files were not changed by this copy; compare again before landing if the owner's work advances.

The law and fountain product digest changes described in `LAYOUT_AUDIT.md` require rebuild when this branch lands. Do not treat the test counts as a full product or browser gate.

## 3-D toy product prepared after the integration receipt

The 3-D `sdf_toy` now consumes `toy_design.toy_parts`, extrudes its split ramps and dribble holes, shares the pool/peg/hourglass dimensions, and places a free hinged wheel from the same design. The `toy` builder variant combines SDF walls and body state and routes to `fountain3d_toy.manifest.json`; readiness has its own family. Host geometry, mesh, startup and routing checks passed. After approval for WebGPU, `small` and `4k` compiled into this checkout's scratch `build/meowdoku/web_toy_dev`; both are in its manifest. No Wasm product was compiled. Startup skips toy Wasm, but an actual on-demand or idle Wasm path remains to be implemented. The toy product has not been deployed to live 8766. The 2-D `PEG_FILTER` flag is still false because the current size-filter configuration clogs; do not describe that filtering behavior as finished.

## Live branches and remaining work

- `worktree-agent-abce6c236a4a86060` still has uncommitted `web/log.html`, `web/log_view.js`, and new `web/local_log.js` as of this note. Its committed backend is included here. Let that worker finish, then bring its final commit or exact files into this checkout, run the JS checks and log tests, and resolve any changes against this tree.
- `worktree-agent-acae7257c8e4ec853` is the original layout branch. Its three stages and audit are represented here. Keep it until integration is reviewed.
- The recent toy, QR wall, Mobroad, Wordle, loading studio, and live-view branches already reached main. The older unmerged compiler branches in `git branch --no-merged main` are separate long-running work and were not folded into a Meowdoku layout change.
- Main has an uncommitted `examples/meowdoku/search.py` edit, which is a stage 1 moved path. Its content has been carried into this branch, but the dirty old path will still stop a merge into main. Main also has unrelated compiler edits, `docs/MEOWDOKU_WEB_2026-11-05.md`, tests, hazards, and other untracked files. Preserve all of them. Stage only explicit paths; do not reset or stash the owner's work indiscriminately.

## Next safe landing sequence

1. Wait for the log worker's final commit, or coordinate a precise snapshot of its three dirty page files. Incorporate it here and verify the page module checks.
2. Compare main's still-dirty `search.py` and test against the copies in this branch. If unchanged, arrange the main checkout's old-path edit so Git can install the shim during merge; retain the implementation under `s2_titles/chess/search.py`. Check whether main has advanced and merge any new commits into this isolated branch first.
3. Verify the resulting diff and focused server/page tests. Then merge to main with the owner's local edits preserved. Schedule the law and fountain product rebuild in the approved digest window; follow the scratch `--web-dir` and 8766/8767 policy for any server gate.

