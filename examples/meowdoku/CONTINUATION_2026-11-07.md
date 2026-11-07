# Meowdoku continuation report, 2026-11-07

Written by a helper session at the coordinator's request (shutdown imminent). Main HEAD when written: c51fc5ab.

## 1. State of live

- Server on 8766 (TLS, public 173.195.26.104:8766), started in a Terminal tab:
  `cd C:\dev\Powershell\turing`, env `TURING_EIGEN_INCLUDE=...\build\third_party\eigen-master`,
  `NODUS_TENSOR_CORE=C:\dev\Powershell\nodus\build\Release\nodus_tensor_core.dll`,
  `MEOWDOKU_PILOT_ARCH=t16.h32.q1.g32.ahard`, `MEOWDOKU_FOUNTAIN_WASM=0`, then
  `python examples\meowdoku\server.py --port 8766 --bind 0.0.0.0 --tls-cert build\meowdoku\tls\cert.pem --tls-key build\meowdoku\tls\key.pem`.
- Deploy = `python examples/meowdoku/web_export.py --skip-boards` (a restart does NOT export page files) plus a restart for server-side changes.
- Server binds first (startup.py lanes, /api/ready). Lifecycle supervisor (`python -m meowdoku.lifecycle supervise`) exists but 8766 is NOT running under it.
- Gitignored config: build/meowdoku/tipjar.json (address set), access.json (public_ip pinned; defaults lan=edit, ip=dev, dns=shader), loading_anim.mla (active loading animation), admin_token.txt (never print).
- [client-error] lines in the 8766 log carry phone errors.
- DEPLOYED (coordinator, 2026-11-07, after merge c06690b9): everything on main through the emission-queue fix is exported and 8766 restarted (tab "meowdoku live 8766"); at shutdown the server was rebuilding all fountain sets in the background (re-keyed by the fix). Check /api/ready: every fountain family should be ready.

## 2. Merged on main and DEPLOYED (verify the fountain rebuild finished)

- c06690b9 / merge c51fc5ab: EMISSION QUEUE FIX. Root cause of the "toy weird forces" and overwhelmed mobroad mobs: armed ids piled at the inlet beyond the k_run walk and were emitted into each other. Fix: QUEUE_STEPS gate in frame() of fountain.py and fountain3d.py.
  - It RE-KEYS every fountain product. After deploy the server rebuilds all fountain sets in the background (minutes).
  - 3D toy unverified; mobroad variant sharing the gate unverified; native-compile emission tests timed out and were not run.
- cc013afb access tiers (lan/ip/dns -> edit/dev/play/shader, server-enforced): needs page export + restart; check access.json.
- Also possibly undeployed, check page vs git: mobroad first playable (77b69dac), loading/bind-first (8affb81a, 64189526), ghosts (f3c6acd8), world curve (53aa32f5), wordle glass (a2c15f9f), purrline sand (aaa70e32).

## 3. What landed today (git log, newest first, grouped)

- Emission queue fix: c06690b9.
- Access tiers: cc013afb.
- Mobroad: 682c19fd WIP, 77b69dac first playable (road-defence mob shooter; variant product, server law, page module).
- Startup / loading panel: 8affb81a (bind first, background lanes, per-product readiness, loading panel, reduced loading animation); 64189526 (12-bit positions at 12 fps, interpolation, recorder at max speed).
- Ghosts: f3c6acd8 (replay layers, behind or composed, divergence colouring, background ghost runs).
- World curve: 53aa32f5 (R log slider, flat, orthographic dolly zoom, linked to display curve); cc329f17 restored decodeIdStart return lost in the purrline merge (Safari "Unexpected keyword export" at compute.js:475).
- Wordle: a2c15f9f (obsidian glass tiles, scene entry bar, hidden keyboard input, overtime +n badge, give-up); c3cf2d62, a59449f3 (rack sizing, Enter/Backspace keys).
- Purrline: aaa70e32 (sand C as product variant; default/wheel digests unchanged), 9e8bb765 (occupancy grid + moving-box SDF slots), 86663b28 (finite sky stock, auto shuffle).
- Tip jar: f1d32ae9 (watch-only Bitcoin piggybank on 3D fountain).
- Share / live view: 438b51f6 (share links for a configured view, view-only live stream), 63e0f376 (close connection after refused unread body), f2959009 (/api/clientlog page errors to server log).
- Lifecycle restarts: 4a2fb1dd (announced restarts, session save/restore, supervisor probe).
- Clock / zoom: baa91b23 (game clock 0.05x-2x, default 0.4x; eased world zoom); 7a3edaa3 (page never zooms on mobile, Reset zoom pill).
- Fountain 2D/3D: 901831f0 (wheel motion blur), 2642150f (3D glyph start), 1c36a038 (2D ramp test), 383341bc (3D oil toy phone-tall), 2cfd79fa (emoji style), 7b92a009 and b3eccf3e (hinged waterwheel 3D / 2D), e237111e (orbeez, SDF presets), 72e2c097 (2D SDF presets keep proportions), 113321a1 and c7c95ed5 (glass ribbons), 77ae1aac (flat 3D product), 57a0846a (case angle, Turn over).

## 4. Unmerged worker branches and next steps

Unmerged into main (git branch --no-merged): ONE.

- `worktree-agent-ae0bbae7011c9a8ab` a90c338c "Meowdoku loading studio: loading screen, re-recording controls, recording library with owner-token activate/delete" (/studio.html, library, owner-token activation; finished).
  - NOT merged. The coordinator's merge attempt hit conflicts (server.py, web/loading_panel.js, web_export.py: the access-tier gating and the startup lanes) and was ABORTED cleanly (no UU files, the index is clean; the branch is intact).
  - Next steps: in a fresh worktree (or main), `git merge worktree-agent-ae0bbae7011c9a8ab`, resolve the three conflicted files (keep BOTH sides: main's startup.py lanes/prepare registries in server.py and the studio routes; web_export.py file lists must include studio.html/studio.js/loading_params.js/loading_view.html plus main's new pages; loading_panel.js: main has the 64189526 12 fps/interpolation changes, studio has re-recording controls), `git add` only those, run `node --experimental-default-type=module --check` on every page JS, run tests/test_meowdoku_loading_anim.py and test_meowdoku_loading_library.py and node tests/meowdoku_loading_params_check.mjs, commit by explicit paths (NOT -a; other people's WIP is in the tree), gate the studio's activate/delete endpoints with access.require_tier(..., "edit") in addition to the token, then deploy.
  - Alternative if it is a mess: `git restore --staged` the studio paths deliberately and redo `git merge worktree-agent-ae0bbae7011c9a8ab` in a clean worktree.
- Branches worktree-agent-a3ae074d41d2ddfb9 (emission fix), a95dd7f5fddaaa81f (wordle), acae7257c8e4ec853 and aeacc527314408423 (both at cc013afb = access tiers): already merged; the worktrees under turing/.claude/worktrees are locked and can be removed/pruned.
- Workers named by the coordinator but with NO branch of their own visible: mobroad top-down gravity + neural pather (variant only) and the code layout audit (bespoke vs generic, tested move proposal + merge plan: merge LAST). If they are still running they will appear as new worktree-agent-* branches; check `git branch --list "worktree-agent-*" --no-merged main`. Likely conflict hot-spots: web_export.py file lists, server.py prepare lanes / startup.py registries, web/app.js, web/compute.js. The layout audit moves files, so merge it after everything else and regenerate its move list against the then-current tree.

## 5. Not from the workers: other people's WIP on main (do NOT commit)

Unstaged / untracked: examples/meowdoku/search.py, docs/MEOWDOKU_WEB_2026-11-05.md (modified), tests/test_meowdoku_lichess_control.py, test_meowdoku_search.py, test_meowdoku_training.py, src/common/tensors/{backward_registry,signal_math,topological_reducer}.py, src/compiler/{fortran_c_shell,glsl_deployment_strategy,process_graph_autograd}.py, tests/test_process_graph_autograd.py, docs/hazards/*, run_windows_battery.ps1, stray files "NUL" and "e HEAD", .claude/. Stage by explicit path only.

## 6. Verification debts

- Oil toy 3D "sdf badly wrong": unverified after the emission fix.
- Mobroad sharing the QUEUE_STEPS gate: unverified. Native-compile emission tests (timed out): not run.
- Purrline sand page render: unchecked. Wordle letters ragged: not fixed.
- Post-deploy fountain rebuild: watch the log for background-lane completion and client errors.
- Phone checks from WORKLIST "Other open items": arcade/sun black-render fix (needs refresh), phone fountain particles not vanishing.
- Open WORKLIST sections: fountain stage A follow-ups (rightward blast, n_drop contacts, 64k pile never settles, emission 21% slower), fountain queue, later desires, SDF chambers/ports/pumps design, purrline sand, render priority; loading section "LANDED, see gaps".
- Other open user requests: prebaked rendering; lifecycle supervisor switch-over of 8766; the DNS/Apache pause.

## 7. Rules learned (memory in C:\Users\alber\.claude\projects\C--dev-Powershell\memory\)

- Test servers ALWAYS use --web-dir with a scratch copy (never the live page dir).
- Syntax-check page JS: `node --experimental-default-type=module --check`.
- Never `git add -A` on main (other person's WIP).
- Spawn workers from the turing cwd, or the worktree lands in the root repo.
- See feedback-meowdoku-two-port-policy.md (8766 live, 8767 dev), feedback-no-builds-without-approval.md, feedback-never-run-interpreter-lane.md, feedback-never-patch-generated-content.md, feedback-no-file-dumped-logs.md.

## 8. The user's standing preferences

Terse, no padding (fact, cause, gap, next). Sonnet workers, one narrow task each, in worktrees. Deploy as soon as work lands. Restart 8766 freely. One law text for server and page. Never patch generated output; fix the generator.
