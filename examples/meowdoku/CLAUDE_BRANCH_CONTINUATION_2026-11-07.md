# Claude branch continuation — 2026-11-07

## Current integration point

This checkout is `codex/meowdoku-claude-integration-20261107`, based on main `66e24497`. It combines the three layout stages with the committed log backend and page work. It has **not** been merged into main or deployed. The live 8766 process was not restarted.

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

The law and fountain product digest changes described in `LAYOUT_AUDIT.md` require rebuild when this branch lands. Do not treat the test counts as a full product or browser gate.

## Live branches and remaining work

- `worktree-agent-abce6c236a4a86060` still has uncommitted `web/log.html`, `web/log_view.js`, and new `web/local_log.js` as of this note. Its committed backend is included here. Let that worker finish, then bring its final commit or exact files into this checkout, run the JS checks and log tests, and resolve any changes against this tree.
- `worktree-agent-acae7257c8e4ec853` is the original layout branch. Its three stages and audit are represented here. Keep it until integration is reviewed.
- The recent toy, QR wall, Mobroad, Wordle, loading studio, and live-view branches already reached main. The older unmerged compiler branches in `git branch --no-merged main` are separate long-running work and were not folded into a Meowdoku layout change.
- Main has an uncommitted `examples/meowdoku/search.py` edit, which is a stage 1 moved path. Main also has unrelated compiler edits, `docs/MEOWDOKU_WEB_2026-11-05.md`, tests, hazards, and other untracked files. Preserve all of them. Resolve the `search.py` content into `s2_titles/chess/search.py` while keeping the old-path shim before merging this branch. Stage only explicit paths; do not reset or stash the owner's work indiscriminately.

## Next safe landing sequence

1. Wait for the log worker's final commit, or coordinate a precise snapshot of its three dirty page files. Incorporate it here and verify the page module checks.
2. Reconcile main's `search.py` work with the moved implementation in this branch. The old path must remain a shim. Check whether main has advanced and merge any new commits into this isolated branch first.
3. Verify the resulting diff and focused server/page tests. Then merge to main with the owner's local edits preserved. Schedule the law and fountain product rebuild in the approved digest window; follow the scratch `--web-dir` and 8766/8767 policy for any server gate.

