# Lanes plan

Ranked work queues for lanes A (bootstrap families), C (language frontier)
and D (defaults and fallbacks). Source: `34_2026_10_22_triage.md`. Rules:
`00_COMPILER_TAKEOVER_PROMPT.md` sections 6 to 10.

Updated 2026-10-23 at `main` `3ffb64d6` (planner run 2). The daily planner
routine rewrites this file. A worker takes the top unclaimed item of its
lane, writes its branch beside the claim, and does one item per run.

## Ranking this round

Rank is blast radius x depth x whether the group can answer wrongly with a
clean audit. Silent-capable groups come first whatever their size. Merged
since run 1: #786 (A0, A3, list returns, call order, branch-merged lists),
#787 (D1), #788 (S1 = A1, W2, C12, C13, C14), #790 (D3).

| order | item | lane | silent? | why here |
|---|---|---|---|---|
| 1 | #791 indexed stores in the call order (c280) | silent lane | yes, **observed** WRONG and SEGV on `main` | PR open; waits on Albert's local gate |
| 2 | #792 = C15 = W4, `sort()` pruned beside an unrolled loop (c244) | C | yes, **observed** WRONG on `main` (fuzz tier 1 seeds 0 and 11, tier 3 seed 28) | PR open; it is the only thing between the A2 branch and a clean seed 0 |
| 3 | A2 optional return surfaces + W1 (branch `h4aybe`, no PR yet) | A | yes, **observed** (seven silent wrongs fixed on the branch) | waits on Albert's answer on the coverage-march card; unblocked once #792 merges |
| 4 | D3a stale `proven_shape` after a refused withdrawal | D | **inferred** yes | 4 corpus programs reach it; nobody has shown a reader of the stale shape |
| 5 | A2 remainder: false acceptance in accepted-unit accounting | A | yes, **observed** once (unit 027) | blast radius is every bootstrap unit with a tuple or optional result |
| 6 | D2 float64 region-input defaults (tire T2) | D | **inferred** yes (a wrong dtype is a TYPE result) | unblocked: #788 merged; depth is the planner, so it is early |
| 7 | A2b tuple-or-`None` returns | A | no: loud by name since the A2 branch | bootstrap units with `fork = None` patterns |
| 8 | A5 bootstrap measurement | A | measurement | A0 merged; the ask is Albert's (see A5) |
| 9 | C16, C1 remainder, W3, then the rest of C | C, A | no, loud | fuzz frequency: C16 is 4/120 at tier 1 seed 11 |

#789 (Lane A, a set merged from branch displays, c274) is loud-to-answer
and measured clean; it holds `_list_literal_mutations`, so C16 waits for
it. Lane E work (LLVM parity, thread `kbkzp7`) is committed on that branch
behind #789 and claims no function in this file's lanes.

## How claims work

A claim names the functions an item will edit. Two lanes never hold claims
on the same function at once. `fortran_c_shell.py` and
`glsl_deployment_strategy.py` are large, so claim the nested function, not
the file. When an item lands on `main`, delete its row and add a line to
"Done" with the commit.

Every item ends with: a corpus program (`tests/native_semantics_corpus/`),
the full typed corpus check (0 non-PASS), a fuzz seed in the lane's tier
(0 WRONG, 0 TYPE), and the gate (no new names in all four parts).

## Corpus number reservations

Corpus programs are `tests/native_semantics_corpus/cNNN_<name>.py`. Two
lanes adding programs at once collided on c218 and c240 to c243 on
2026-10-22. Before adding a program, claim a range here (on your branch, in
the same commit as the claim of your item) and number inside it. Take the
next free block of 10 above the highest row; never reuse a number another
branch has used, merged or not.

| range | lane / item | branch | used |
|---|---|---|---|
| c211-c218 | Lane A: A0, A3 and follow-ups | `claude/project-thread-kbkzp7` | c211, c212, c214-c218 |
| c230-c239 | Lane A: A2 optional returns | `claude/project-thread-h4aybe` | c230-c234 |
| c240-c249 | Lane C: C1 constant-list comprehensions, C15 | `claude/project-thread-2nw77p` | c240-c244 |
| c260-c269 | Lane A: S1 tuple birth, W2 concatenations; Lane C: C12, C13, C14 | `claude/project-thread-cz18s3` | c260-c269 (full) |
| c270-c279 | Lane A: A0 thread follow-ups (calls in source order, branch-merged displays) | `claude/project-thread-kbkzp7` | c270, c271 |
| c280-c289 | Silent lane follow-ups (indexed stores in the call order) | `claude/project-thread-cz18s3` | c280 |
| c290-c299 | next free | | |

## Lane A: bootstrap families (and the tire, where it shares a group)

| rank | item | group (doc 34) | claimed functions | branch | status |
|---|---|---|---|---|---|
| A2 | Keep a return surface whole: the remaining pieces. `emit_outputs` must not fall back to `(output_id,)` when no `Ret` exists; accepted-unit accounting must not let one value stand for a tuple/optional (`project_compilation_product`, `len(paths) == 1 and len(published_ids) == 1`); add the doc 34 §1.4 false-acceptance check to the bootstrap status tool. The optional-return part is done (doc 34 §6, §7). | B1, T3 | `fortran_c_shell._class_surface_ssa_program.emit_outputs`; accepted-unit result accounting | `claude/project-thread-h4aybe` | claimed 2026-10-22; the optional-return part and W1 are done on the branch (no PR yet: waits on #792 and Albert's card) |
| A2b | Lower a returned tuple-or-`None` (`if c: return (a, b)` / `return None` in a bootstrap unit; `fork = (a, b)` / `fork = None; return fork` anywhere). It refuses by name now (`its newest binding ... was not materialized`), was silent before. Observed by the silent-wrong-answers thread: the planner builds the aggregate Phi (presence member 15, member Phis 16, 17), but the root's return reads the handle (10) instead of that layout, and the tuple's region is then pruned as dead. | B1 | the control finish's output resolution in `precompile_to_ssa`; find the pruner with a trap | | open: loud |
| A4 | Tire: the result-storage lease mints a record span member's `SSA_VALUE` row as a scalar. Revise it from the record storage contract. `_ssa_structure` reads the registered record storage before calling a shape-empty fact a scalar. Repro idea: lift `Metrics` with its `error_present` span (15,) into a scratch program that returns the record through one call edge. Suspected pass (**inferred**): the lease writer, which reads the fact's empty shape before the record storage. | T1 | `structural_concordance._ssa_structure`; the result-storage lease writer in `fortran_c_shell` | | open: A0 merged, re-observe at `main` first (needs a tire run, see A5) |
| A5 | Measure the core-chain cascade at `main` after #791 and #792: one full bootstrap and one managed tire run. Fill doc 34 §1.2 and re-file units into groups. Seven silent-wrong fixes and the S1 birth have landed since the 23/330 measurement, so the triage groups are stale (**inferred**). | B6, B7, T1 | none (measurement) | | open: asked Albert 2026-10-23 |
| A6 | Variadic returns: the sequence-lane lowering for `return ()`/`(x,)`/`[]`/`[x]` (units 034, 162, 212). | B2 | reducer VARIADIC consumer in `precompile_to_ssa`; `ir_sequence_tables` | `claude/project-thread-kbkzp7` | claimed 2026-10-22; list returns fixed or refused (doc 35), tuple variadic and two-edge list lowering open |
| A7 | `Any`-typed receivers and generic `str()`/`!r` of facts: runtime type dispatch over the fact classes. | B3 | to be scoped | | open: needs a design note first |
| A8 | `'|'.join(tokens)` glyph recipe in f-strings (unit 6). | B5 | f-string glyph recipe builder | | open |
| W3 | A caller's `is None` test on a callee returning `x if c else None` (a conditional value union) does not compile: the call escapes whole as a `double *` table over `int64_t` slots. Loud (C compile error) on `main`. Doc 34 §7. | the union-call consumer in `ssa_c_backend` / the optional-call rewrite in `ssa_optional_values` (find which owns it with a trap) | | | open |

## Lane C: language frontier

Each item turns a LOUD cluster into a lowering, using existing mechanisms,
and adds a corpus program and a fuzz snippet. The order is by how often the
fuzzer and bootstrap meet the cluster (**inferred** from doc 33), with
silent-capable items first.

| rank | item | where the record is missing (doc 33) | claimed functions | branch | status |
|---|---|---|---|---|---|
| C1 | Reductions over a structural aggregate: `any((1 > n, 2 > n))`, `sum(x * n for x in xs)` with `xs` literal | the region takes the aggregate as an unproduced input, and no layer assembles it from its unrolled members | `resident_sequence_materialization._list_literal_mutations`; `glsl_deployment_strategy._tensor_descriptor_rule` (loop-target branch) | `claude/project-thread-2nw77p` | list/set comprehension part done (e7f4a46d, landed with PR #788; c240-c243). Still loud by name: generator over a folded list (`any(x > n for x in xs)`: the GeneratorExp materializer is not a declared sequence) and tuple displays |
| C2 | A display rebound inside a loop: `xs = [i + n, i]` | `descriptor-member-shared`; unaccounted formal (loud in the product at `main`) | descriptor birth in `precompile_to_ssa` | | open |
| C3 | Dict `del d[k]` / `d.pop(k)` | placed by graph-id order into a neighbouring region (a cycle); `pop` has no lowering. It should be one lexical mapping effect. | `_control_block_position` (read only); mapping effect lowering | | open |
| C4 | `d.get(k, 0)` in a loop | control order cycle (c202) | `_sequence_effect_read_edges` | | open |
| C5 | Standalone slice `ys = xs[1:n]` | no lowering; only `r += xs[a:b]` is recognized | slice recognizer; `ir_sequence_tables` | | open |
| C6 | Nested list stores and `xs[i % 4] += i` in a loop | no source cell for the control block; a carried read with no lexical binding row | indexed-store chain in `glsl_deployment_strategy` | | open |
| C7 | Early `return` from a loop over a list parameter | (c216) | loop exit lowering | | open |
| C8 | Membership in a folded literal inside a loop (`i in ts`) | structural recovery builds at entry, where the loop target does not exist; the folded literal has no resident sequence | membership survey | | open |
| C9 | Multi-element replication `[a, b] * k` | no operation writes its rows; fill repeats one value | `_sequence_concat_ops` | | open |
| C10 | Strings as values (`s = "a" if c else "b"`, `len(s)`, `==`, `startswith`) | dispatch metadata; minted as a root input (c114, c217) | to be scoped | | open: large |
| C15 | An in-place `sort()` beside an unrolled literal loop or comprehension: `for x in [3, -2, 0]: t += x * n` then `xs = [6, 7, n]; xs.sort(); xs[0] - xs[2]` | loop evaporation's ancestry pruning deleted the `sort` call (no retained record, no consumer); the fold read the display order (silent WRONG at `main` 3ffb64d6; fuzzer p11_29, p11_116, p23_3) | `loop_composer.evaporate_unrolled_loops` (liveness); `resident_sequence_materialization._method_mutation_edges` (new, the edges behind `_method_mutated_receiver_ids`) | `claude/project-thread-2nw77p` (PR #792) | fixed on the branch (c244). Also W4 from the A2 thread (p0_58, p0_85, p0_176 pass on this branch, observed by that thread) |
| C16 | A comprehension over a folded list beside membership in another folded list: `ys = [x - n for x in [4, 4, 8]]` with `n in [5, 0, 2]` | loud at `main` 3ffb64d6: `List # lacks one exact element producer` (from the list birth in `_list_literal_mutations`); fuzzer tier 1 seed 11 p11_0, p11_24, p11_78, p11_104. Repro idea: the two lines above in one function of `n`. Suspected (**inferred**): the repeated member `4, 4` gives two members one producer, as `[m, m, 9]` does (doc 36 §1), so the birth refuses; the fix is a per-member birth, not a dedupe | `resident_sequence_materialization._list_literal_mutations` (after #789 merges; #789 edits it) | | next for Lane C |

## Lane D: defaults and fallbacks

Inventory at `3faa6e8`, by grep over `src/compiler/*.py` (**observed**
counts). The first job of the lane is to replace these greps with a
committed inventory script and classify every hit.

| category | count | where most hits are |
|---|---|---|
| `except Exception:` followed by `pass` | 23 | `tensor_ssa_lowering` 4, `shell_telemetry` 4, `identity_concordance` 3, `glsl_deployment_strategy` 2, `fortran_c_shell` 1 |
| `or "float64"` / `else "float64"` dtype fallbacks | 89 | `glsl_deployment_strategy` 17, `precompile_to_ssa` 13, `fortran_c_shell` 13, `tensor_ssa_lowering` 9 |
| `.endswith(` (name/suffix matching; many will be legitimate) | 106 | all modules |
| `[0]` on argument lists in the two large modules | 108 | `glsl_deployment_strategy`, `fortran_c_shell` |
| `"int"` comparisons / spellings in the four core modules | 26 | `fortran_c_shell`, `glsl_deployment_strategy`, `precompile_to_ssa`, `ssa_c_backend` |
| `binding_name ==` | 0 | (the prompt's lead is stale; `binding_name` appears 104× in `glsl_deployment_strategy`, 89× in `fortran_c_shell`) |

| rank | item | claimed functions | branch | status |
|---|---|---|---|---|
| D3a | A shape withdrawal the book refuses, swallowed. In c02, c57, c58 and c62, `invalidate_shape_transformation` posts `shape_transformation_state` for the f-string glyph helpers (`append_decimal_integer_glyphs`, `hyp`). The book refuses it: "REVISE without a changed source; every Derived cell is stamped at or before the row's". Because the refusal is raised before the `proven_shape` invalidation, the stale proven shape stays on the book (**observed**). Whether any reader then uses it is **unknown**; it is **inferred** silent-capable. Next step: trap the post, find why the cause cell is not newer than the row, and supply that record. Then remove the try at the `invalidate_proven_shape` call. Repro idea: c02 alone, with a book-post trap on `shape_transformation_state` for `append_decimal_integer_glyphs`, printing the cause cell's stamp and the row's. | the `invalidate_proven_shape` call site in `glsl_deployment_strategy` (descriptor invalidation cone); `identity_concordance.invalidate_shape_transformation` | | next for Lane D |
| D2 | Planner region inputs default to float64. Count the fallbacks across the corpus again (~3,000 measured once), then supply the proofs. This is also the tire's T2 (`bool` actual vs float64 region formal). | `_planned_scalar_dtypes` (read), region-input typing in `glsl_deployment_strategy` (`or "float64"` sites near lines 21277 to 21917) | | open, after D3a. Repro idea: count with a one-function hook on the `or "float64"` sites over c01-c62 first (not the full corpus); pick the site with most hits that reach a result. Suspected (**inferred**): inputs whose producer is a call result with a settled linked-ABI class that the region typing does not read |
| D4 | `"int"` vs `"int64"` spellings reaching physical checks (rule 7). First job: a committed inventory script (`tools/` scratch is not enough) that lists each of the 26 sites and whether a physical check reads it. Suspected writers (**inferred**): provisional class spellings published before `_planned_scalar_dtypes` settles them | the inventory script only (new file); no compiler function until it lands | | open, after D2 |
| D5 | Name, suffix and position matching (`.endswith(`, `[0]` on argument lists) in decision paths | to be listed by the inventory | | open: inventory first |

## Done

- A1 / S1 (tuple displays are born and bounded), W2, C12, C13, C14: PR #788, merge `ca9668a4`, receipts in doc 36.
- D1 (private root storage proven or refused): PR #787, merge `cc6e6f36`.
- D3 (no silent except around the binding-kind lookup or the planned-shell walk): PR #790, merge `3ffb64d6`. D3a stays open.
- A0 (chained comparison with a builtin conversion middle): PR #786, merge `57709b54`, receipts in doc 35.
- A3 (`dict.update()` from a display is a table mutation): PR #786, merge `57709b54`, receipts in doc 35.
