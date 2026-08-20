# Frontier task pack — start here

You were probably asked: *"can you review the current status and where you
can help."* This directory is the answer. Pick ONE task file, follow it
exactly, and stay inside its scope. Each task is real frontier work that was
deliberately carved to be finishable in one focused session.

## Status in three sentences (2026-08-19)

A root compiler defect was found and fixed: every graph node is born with
`constant=None`, a discriminator trusted the key's presence, and as a result
**every dynamic `if` in every compiled program was silently flattened to its
else-arm**. With that fixed, the translation scorecard is 17/19 (an authored
`if` compiles correctly for the first time), and the two flagship programs
(Shoal, the fluid model; `re.compile`) moved off fabricated ground onto
honest, precisely-named walls. The full narrative is
`../HANDOFF_SHOAL_AND_RE_TARGETS.md` (§6e–6g are current) — you do NOT need
to read all of it; each task file quotes what it needs.

## The tasks

| file | task | shape | difficulty | status |
|---|---|---|---|---|
| `TASK_A_constant_presence_sweep.md` | audit 3 remaining sites of the constant-presence trap | measure, then mechanical fix | low | **done** — 1 real fix (`literal_value`), 3 sites confirmed safe |
| `TASK_B_scorecard_level14_rebound_name.md` | level 14: call argument binds a later version of a rebound name | diagnose to a named seam; fix if small | medium | **done (findings only)** — seam named exactly; a fix attempt regressed levels 8/9 and was reverted; two concrete directions recorded for the real fix |
| `TASK_C_scorecard_level18_any_generator.md` | level 18: `any()` over a generator never renders | diagnose to a named seam; fix if small | medium | **done (findings only)** — task's own hypothesis disproven: the generator loop already decomposes correctly; the real gap is the materializer having zero sequence/tensor support, not a compiler defect |
| `TASK_D_overlay_embed_scope_refusal.md` | make the control-overlay's cross-scope failure name itself | unit-level, pure data structures | low | **done** — named refusal + 4 tests; confirmed against re.compile |
| `TASK_E_shoal_dt_next_wiring.md` | the frame's dt_next output reads 0.0 — find the missing binding link | diagnosis-first, expensive builds budgeted | medium-high | open |

Scorecard is 17/19 (levels 14 and 16 both closed this arc; 18 and the
non-scorecard walls in the main handoff remain).

Do not start work outside these files. The remaining frontier (re's
`Raise` loop-blockers, the Shoal out-of-bounds sequence read, the reversible
machine's CRT import surface) needs design decisions and is not in this pack.

## Non-negotiable ground rules (each one has destroyed work before)

1. **Read `../TEST_BASELINE_AND_HAZARDS.md` before running any test.**
   Its manifest says which failures are pre-existing. A failure listed
   there is NOT yours — do not re-derive it.
2. **NEVER run the whole pytest suite** (`pytest tests`). It does not
   finish. Run single files with an external timeout:
   `timeout 120 python -m pytest tests/<one_file>.py -q --tb=short`
3. **The regression gate** (~40 s, run after every compiler change):
   ```
   timeout 300 python -m pytest tests/test_precompile_to_ssa.py tests/test_symbolic_fluid_native_runtime.py tests/test_symbolic_fluid_direct_backends.py tests/test_abstract_tensor_indexing.py tests/test_ssa_fusion_regions.py tests/test_region_kernel_dedup.py tests/test_translation_scorecard.py -q --tb=short
   ```
   Expected today: **68 passed**. The scorecard tool itself
   (`timeout 120 python tools/translation_scorecard.py`) prints 17/19.
4. **Never `git stash` and never `git checkout -- <path>`** to baseline.
   Both have destroyed uncommitted work in this tree. To compare against a
   commit, use `git worktree add`.
5. All commands run from the repo root (`turing/`). Ignore the pygame
   banner on stderr — it is noise from an import.
6. **Do not delete or gate authored code to make a wall go away.** The
   owner's rule: "we're not trying to deadcode anything that isn't dead."
   An honest, loudly-named refusal is an acceptable outcome; a silently
   wrong program is never one.
7. Commit style: small measured steps, narrative first line (look at
   `git log --oneline -15` for the voice). Only commit files your task
   names. If `git status` shows changes you did not make, leave them
   alone — another session owns them.
8. **If you get stuck or your measurement contradicts the task file:
   stop.** Append what you measured (commands + verbatim output) under a
   `## FINDINGS <date>` heading at the bottom of your task file, commit
   that, and end. A recorded dead end is a valuable result; thrashing is
   not.
