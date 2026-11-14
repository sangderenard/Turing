# Nov 13 Codex work inventory (Claude, 2026-11-14)

For reconciliation with Codex's inventory. Written from a cloud session that could
not reach the messenger (`stichka.duckdns.org:8766`: plain HTTP reaches Apache, every
TLS handshake is reset inside the cloud egress; cause not established).

## Evidence base and scope

- **Read:** all 86 `docs/hazards/20261113*` notes (no `20261114*` notes exist), plus
  `docs/NODUS_COMPUTE_REQUESTS.md`, `docs/LIVING_DATA_STATIC_DM_API.md`,
  `docs/NODUS_CONSOLE_WORLD.md`.
- **Git:** Turing `main` = `origin/main` = `ac15a21` ("Seal page contracts and bake
  independent releases", 11-13 21:55 -0600). The clone is shallow (249 commits).
  nogodsnomasters has no Nov 13 commits; its last is `197517a`, the Nov 12
  `string_core` merge.
- **Not inspected directly:** Nodus, Spectral Analyzer, SpeakToMe. Those repos are not
  in this session. Their status below comes only from what Turing's notes say about them.
- **Time zones:** hazard filenames are UTC and git dates are -0600, so notes stamped
  before about 06:00Z match commits dated 11-12 local.

**Status levels:**
- REVIEWED: tests or checks were run.
- LANDED: a matching commit is on Turing `main`.
- DEPLOYED: a note claims a live restart, export or public check succeeded.

**No Nov 13 note claims DEPLOYED.** The only live action recorded is a
`Restart-Stack.ps1` run at 13:06 local (`194730Z_main`). In that run `nodus-ui-mirror`
failed 5 restarts, and nobody rechecked it afterwards.

## Workstreams

| # | Workstream | Notes | Landed as | Status |
|---|---|---|---|---|
| 1 | Compiler: annotation-container ABI → owned/borrowed glyph lowering (GraphSolver `compile_nodes`) | about 50 notes, `040256Z`–`185500Z` | `a296061` merge sweep; `0652b27`…`1129985`; `35fc9a1` via `1057a3c` | REVIEWED + LANDED, step by step; no step pushed or deployed by its note |
| 2 | Compiler: YoungMan fixed-shape per-geometry kernels | `_main` notes `000533Z`–`194730Z` | `773b9b4`, `252638e`, `2716cd0`, `5616739`, `d587fda`, `5f28276`, `a5a5508`, `7030b93`, `74a431f`, `8ff4bf8`, `94722ed`, `b510e3e`/`c5d61f3` | REVIEWED + LANDED |
| 3 | Nodus Radio: PCM readback, EOF, PA sink, controls, jukebox | `024500Z`–`053000Z` | `d3715c2`…`e7eea7d`, merged `6725a01` (11-12 local) | REVIEWED + LANDED; "no live restart … or audible playback" |
| 4 | Radio HLS browser route | `211430Z` | PR #1219 (`9862968`/`1f17305`) | REVIEWED + LANDED; needs Nodus PR #73 binary and operator config |
| 5 | Studio radio→light jack, AbstractNN, NumPy signal graph | `060000Z`, `070000Z`, `080000Z` | `d9b7f3f`, `b650835`, `d6e08c0`, `284ff7f`, `1595905`, `e4eee8a`, `2e7c233`/`736fb84` | REVIEWED + LANDED; "do not label this integration live" |
| 6 | Studio autofocus / DM camera | `171600Z` | `495030d`, `eb7ef8b`, `615c75a`, `a22078b`, `f1dbc0c`, `d918087` | REVIEWED + LANDED; loopback test only, "not deployed" |
| 7 | Paper market training display/control | `195726Z`, `201000Z` | `a92085c`, `d7121ff`; related PRs #1215–#1218 | REVIEWED + LANDED; live action not exercised |
| 8 | General Nodus compute queue | `200500Z` | `d7b193f`/`fde37ed` | REVIEWED + LANDED; SpeakToMe side untested |
| 9 | Authored pages + Wave Arena live source | `223442Z`, `224247Z` | PR #1221, PR #1222; follow-ups `fc92a3d`, `ac15a21` (no notes) | REVIEWED + LANDED; no export, restart or public check |
| 10 | Nodus console world (first slice) | `232500Z` | `b841373` | REVIEWED + LANDED; full HTML generation fails |
| 11 | User files / keys / permission groups | none | `0af53f1`, `361124f`, `4d8e0d3`, `6c2a976`, `f9db6fd`, `a24d5cf`, `ca1a4a1`, `13931d6`, `9d96173`, `727a8a8`, `0952929` | LANDED; no review or deploy evidence in the notes |

## Remaining work and blockers (quoted where possible)

1. **GraphSolver compile is still not runnable.** It now stops at "compile_nodes value43,
   its set key". The next problem is "branch100 joining106 (no child pool) and104 (child
   pool), whose storage contracts correctly disagree". Not implemented yet: dynamic dict
   update, delete/compaction, child-Phi selection, the public nested-copy ABI. The work
   is "NOT an ASAN-clean transfer claim".
2. **YoungMan.** The latest stop is "post proven_shape … REVISE without a changed source",
   and it is untraced. Open defects:
   - n4 mixed-dtype tuple: "Call operand 0 is neither a formal nor defined"
   - LLVM span extents
   - "private sequence 22 … no proven capacity"
   - C top-level loop Phi `missing_physical_dtype`, which gives 5 new
     `test_source_c_exceptions` failures (18 vs 13)
   - no Fortran ExceptionNone/Make lowering
   - c422 gives wrong values on clean main

   The cube, octahedron and icosahedron geometries are not driven yet.
3. **Standing test debt.**
   - Stale FORMAL_ACTUAL_OCCURRENCE IndexError: `test_native_hash_indexed_tables` (3) and
     `uint64_array_append_literal` (4).
   - `test_ir_sequence_tables.py`: 15 failures already present before this work.
   - Constant-bool loop seed bug: the red test
     `tests/test_constant_boolean_loop_update_pending.py` is not in git.
4. **Radio release.** "Publication, operator catalogue/native path/port configuration, and
   an authenticated live endpoint trial remain outstanding." No operator Radio config
   exists in the stable or dev state directories. The old binary rejects the extra
   argument, so this needs the new Nodus binary (PR #73). Do not reuse the SQL relay
   daemon's port.
5. **Radio-light jack and AbstractNN.** "Turing's venv lacks Torch; SpeakToMe's venv fails
   NumPy native import/Torch DLL import." There is no verified worker-placement boundary
   yet. This also needs Spectral `7a7d8f0` and the Spectral worktree
   `codex/radio-light-jack-20261113`, whose landing status is unknown.
6. **Autofocus.** "authenticated server integration and ray submission/render acceptance
   remain UNTESTED". With `--no-wasm`, the dev server still entered `prepare_lanes`.
7. **Compute queue.** "Shared SpeakToMe pytest bootstrap attempted PyPI and was denied by
   network policy." The SpeakToMe routes/catalog and the Turing host must be deployed
   together.
8. **Authored pages.** "The SpeakToMe messaging connector timed out, so operator
   coordination and live exposure checks could not be completed." The full web export
   needs the Spectral sibling checkout. No self-service compiler-to-page job exists
   (Nodus `SiteCompiler` emits row tools).
9. **Console world.** `navigation_astar.c:1:1: error: Unexpected` reproduces in an
   untouched canonical checkout. The native scalar run did not give 8 after one step.
   Native controls, telemetry, relaxed wires, dial/treadmill and visual acceptance are
   all still to do.

## Conflicts and cross-lane hazards

- **Parallel edits.** Compiler notes record workers overlapping on `runtime_abi.py`, seed
  layout and runtime suites, which forced reruns and freeze notices. An integration owner
  coordinated holds.
- **`examples/meowdoku/web_export.py`.** PR #1222 resolved a conflict by keeping both
  the Wave Arena copy and the direct-game export. This file is shared with `8aba2ac`,
  `cef0204` and `763e8d2`.
- **Ports.**
  - `fe79795` moved the UI mirror off the Studio contract port.
  - `dfa9fbc` moved the training mirror off the Nodus graph feed port.
  - Mirror health after these moves has not been verified.
  - One note saw nothing listening on 8766, while 8767 was an old toy dev process. That
    note was written in an isolated worktree, so the live stack's actual state is unknown.
- **Network policy.** Radio notes: "do not bypass" the earlier network-policy denial. Any
  release has to be reconciled "through the authorized coordination path".

## Ready for a shared action plan

1. Recheck `nodus-ui-mirror` on the current stack, after `fe79795` and `dfa9fbc`. **Owner: operator.**
2. Fix the hazards-log command that rejects the older nonconforming hazard filename (two
   notes reported it). **Compiler or tools lane.**
3. Investigate `navigation_astar.c:1:1: error: Unexpected`. It is reproducible on
   canonical main.
4. Commit and triage the constant-bool loop red test (For and While).
5. Fix the stale FORMAL_ACTUAL_OCCURRENCE IndexError, which unblocks two suites.
6. C lane: physical dtype for top-level loop Phis, which closes the 5
   `test_source_c_exceptions` regressions.
7. GraphSolver frontier: keyed child-storage proof at value43, and the branch100 pool
   contract.
8. YoungMan: trace the REVISE refusal, then drive the remaining geometries.
9. Put sibling Spectral and SpeakToMe checkouts into the integration worktrees so the
   full web export and the SpeakToMe tests can run there.
10. Merge and build Nodus PR #73. Then run an operator-configured, authenticated Radio
    HLS/PA trial on ports that don't collide with existing services.
11. Autofocus: an authenticated ray-submission test against a backend that respects
    `--no-wasm`.
12. Deploy decisions needed from the operator:
    - authored pages / Wave Arena exposure
    - SpeakToMe and Turing compute routes, deployed together
    - live paper-training control (affects shared training)

## Not covered

- Nodus, Spectral Analyzer and SpeakToMe commit history and branches. In particular:
  - whether Nodus PR #73 merged
  - whether Spectral `codex/radio-light-jack-20261113` merged
  - whether the SpeakToMe REST changes landed
- Codex's own session transcripts and anything not written to a hazard note.
- Live service state.
