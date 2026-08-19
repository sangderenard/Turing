# Test baseline and hazards — read before running any test

This document exists because the same expensive mistake kept being repeated:
an agent sees a failing test, assumes it broke it, and spends two full suite
runs plus a stash/pop to find out the failure was already there. That costs
hours and has damaged working state in this tree. **The baseline below is the
answer to "is this mine?" — consult it before running anything.**

## The rule

1. **A failure in the manifest below is NOT yours.** Do not re-derive it. Do
   not run a second time to confirm. Say "pre-existing, see
   TEST_BASELINE_AND_HAZARDS.md" and move on.
2. **Never baseline with `git stash` or `git checkout -- <path>`.** Both have
   destroyed uncommitted work in this tree. If you genuinely must compare
   against a commit, use a worktree, which touches nothing you have:
   `git worktree add /c/dev/turing-head <commit>` — and remove it when done
   with `git worktree remove /c/dev/turing-head`.
3. **Never run a test purely to collect names for a report.** If you already
   have a pass/fail count or a progress string that answers your question,
   that IS the measurement. Re-running for cosmetics is not free here.
4. **Update this file** when you legitimately run something and learn a name
   or a count the manifest is missing. Filling a gap as a side effect of work
   you were doing anyway is right; running to fill it is not.

## Hazards, measured

| hazard | detail |
|---|---|
| `pytest tests` (whole suite) | **Does not finish.** Observed 2026-08-19: 5 h wall, **zero bytes** of output, then decayed to ~3 CPU-seconds per 5 minutes and had to be killed. Do not run it. |
| Output is buffered | pytest writes nothing until it exits when its stdout is a pipe. `-u` / `PYTHONUNBUFFERED` do not help — pytest buffers its own progress. A silent run is not a hung run, and you cannot tell them apart by watching. |
| `pytest-timeout` is **not installed** | There is no per-test timeout. One hanging test eats the whole run. Bound it from outside instead: `timeout 100 python -m pytest tests/<one_file>.py -q --tb=no`, one file at a time. Exit code 124 means it hung. |
| Killing a run | Several tests spawn native toolchains (gfortran, zig cc). Killing the pytest process can leave children; check for stray `python`/`zig` processes afterward. |
| Cached artifacts | A stale `control_repository_ssa.pkl` presents a program built by a compiler that no longer exists, so your change appears to do nothing. `_cache_is_stale` guards the fluid path; other paths may not. |

## The manifest — known-failing at `af00599`

Verified 2026-08-19 by running each file in the working tree AND in a clean
`git worktree` at `af00599`. Identical results in both, so these are
pre-existing and independent of the namespace/indexing fixes on top.

| file | result | notes |
|---|---|---|
| `tests/test_ast_indexing_aot.py` | **13 failed, 10 passed** | progress string `FF.F.FFFF.F.F.....F.FFF`, byte-identical in both trees. Individual test names NOT captured — collecting them costs a full run of an expensive file, which rule 3 forbids doing for its own sake. If you run this file for your own reasons, paste the `-rf` names here. |
| `tests/test_index_set_scatter.py` | **2 failed, 7 passed** | `test_index_set_emits_a_complete_scatter_module`, `test_index_set_scatter_runs_correctly` |

## The manifest — known-good at `af00599` plus the current working tree

These passed on 2026-08-19 and are the cheap, high-signal set. Prefer them.

| check | result | cost |
|---|---|---|
| `tools/translation_scorecard.py` | 10/10 journeys equivalent | ~9 s |
| `tests/test_precompile_to_ssa.py` | 34 passed | ~4 s |
| `tests/test_symbolic_fluid_native_runtime.py` | 1 passed | ~17 s |
| `tests/test_abstract_tensor_indexing.py` | 2 passed | ~1 s |
| `tests/test_ssa_fusion_regions.py` | 1 passed | ~3 s |
| `tests/test_region_kernel_dedup.py` | 2 passed | ~3 s |

**That table is the recommended regression gate for compiler changes.** It is
~40 seconds total and it caught nothing false in this session. Reach past it
only when your change plausibly touches something it does not cover, and then
reach for single files with an external `timeout`, never the whole tree.

## Marking expected failures in code

The manifest is the cheap fix. The better fix is `@pytest.mark.xfail(reason=
"pre-existing at af00599, see TEST_BASELINE_AND_HAZARDS.md", strict=False)` on
the known-bad tests, so a green run means green and nobody has to cross-check
a document. That needs the 13 names from `test_ast_indexing_aot.py`, which
per rule 3 should be collected the next time someone runs that file for a
real reason — not by a run commissioned for this purpose.

`strict=False` matters: these should announce themselves as XPASS when
somebody finally fixes them, rather than failing the suite for being fixed.
