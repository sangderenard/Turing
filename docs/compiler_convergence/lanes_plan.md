# Lanes plan

Ranked work queues for lanes A (bootstrap families), C (language frontier)
and D (defaults and fallbacks). Source: `34_2026_10_22_triage.md`. Rules:
`00_COMPILER_TAKEOVER_PROMPT.md` sections 6 to 10.

Updated 2026-10-22 at `main` `3faa6e8`. The daily planner routine rewrites
this file. A worker takes the top unclaimed item of its lane, writes its
branch beside the claim, and does one item per run.

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
| c240-c249 | Lane C: C1 constant-list comprehensions | `claude/project-thread-2nw77p` | c240-c243 |
| c260-c269 | Lane A: S1 tuple birth, W2 concatenations; Lane C: C12, C13, C14 | `claude/project-thread-cz18s3` | c260-c269 (full) |
| c270-c279 | Lane A: A0 thread follow-ups (calls in source order, branch-merged displays) | `claude/project-thread-kbkzp7` | c270, c271 |
| c280-c289 | next free | | |

## Lane A: bootstrap families (and the tire, where it shares a group)

| rank | item | group (doc 34) | claimed functions | branch | status |
|---|---|---|---|---|---|
| A1 | Give tuple displays a birth. Place the predicate region that produces the members before the effects that consume them, then birth tuples as lists are born. | S1 | `resident_sequence_materialization._list_literal_mutations`, `resident_sequence_materialization._static_sequence_capacity_survey`, `identity_concordance.CorrelationTable._unborn_display_findings` (new) | `claude/project-thread-cz18s3` | in review: born and bounded; the predicate-region placement did not reproduce at `main` |
| A2 | Keep a return surface whole. `emit_outputs` must not fall back to `(output_id,)` when an aggregate layout is absent. The control finish publishes the layout. Accepted-unit accounting must not let one value stand for a tuple/optional. Add the false-acceptance check from doc 34 §1.4 to the bootstrap status tool. | B1, T3 | `fortran_c_shell._class_surface_ssa_program.emit_outputs` (nested, line ~28034); the control-finish member Phi publisher; accepted-unit result accounting | | open: false acceptance |
| A4 | Tire: the result-storage lease mints a record span member's `SSA_VALUE` row as a scalar. Revise it from the record storage contract. `_ssa_structure` reads the registered record storage before calling a shape-empty fact a scalar. | T1 | `structural_concordance._ssa_structure`; the result-storage lease writer in `fortran_c_shell` | | blocked on A0 (re-observe first) |
| A5 | Measure the core-chain cascade. After A0, ask Albert for a full bootstrap. Fill doc 34 §1.2 and re-file units into groups. | B6, B7 | none (measurement) | | blocked on A0 |
| A6 | Variadic returns: the sequence-lane lowering for `return ()`/`(x,)`/`[]`/`[x]` (units 034, 162, 212). | B2 | reducer VARIADIC consumer in `precompile_to_ssa`; `ir_sequence_tables` | `claude/project-thread-kbkzp7` | claimed 2026-10-22; list returns fixed or refused (doc 35), tuple variadic and two-edge list lowering open |
| A7 | `Any`-typed receivers and generic `str()`/`!r` of facts: runtime type dispatch over the fact classes. | B3 | to be scoped | | open: needs a design note first |
| A8 | `'|'.join(tokens)` glyph recipe in f-strings (unit 6). | B5 | f-string glyph recipe builder | | open |

## Lane C: language frontier

Each item turns a LOUD cluster into a lowering, using existing mechanisms,
and adds a corpus program and a fuzz snippet. The order is by how often the
fuzzer and bootstrap meet the cluster (**inferred** from doc 33), with
silent-capable items first.

| rank | item | where the record is missing (doc 33) | claimed functions | branch | status |
|---|---|---|---|---|---|
| C1 | Reductions over a structural aggregate: `any((1 > n, 2 > n))`, `sum(x * n for x in xs)` with `xs` literal | the region takes the aggregate as an unproduced input, and no layer assembles it from its unrolled members (loud in the product at `main`, silent under `emit_ssa_to_c` alone) | region input assembly in `glsl_deployment_strategy`; structural recovery | | open |
| C2 | A display rebound inside a loop: `xs = [i + n, i]` | `descriptor-member-shared`; unaccounted formal (loud in the product at `main`) | descriptor birth in `precompile_to_ssa` | | open |
| C3 | Dict `del d[k]` / `d.pop(k)` | placed by graph-id order into a neighbouring region (a cycle); `pop` has no lowering. It should be one lexical mapping effect. | `_control_block_position` (read only); mapping effect lowering | | open |
| C4 | `d.get(k, 0)` in a loop | control order cycle (c202) | `_sequence_effect_read_edges` | | open |
| C5 | Standalone slice `ys = xs[1:n]` | no lowering; only `r += xs[a:b]` is recognized | slice recognizer; `ir_sequence_tables` | | open |
| C6 | Nested list stores and `xs[i % 4] += i` in a loop | no source cell for the control block; a carried read with no lexical binding row | indexed-store chain in `glsl_deployment_strategy` | | open |
| C7 | Early `return` from a loop over a list parameter | (c216) | loop exit lowering | | open |
| C8 | Membership in a folded literal inside a loop (`i in ts`) | structural recovery builds at entry, where the loop target does not exist; the folded literal has no resident sequence | membership survey | | open |
| C9 | Multi-element replication `[a, b] * k` | no operation writes its rows; fill repeats one value | `_sequence_concat_ops` | | open |
| C10 | Strings as values (`s = "a" if c else "b"`, `len(s)`, `==`, `startswith`) | dispatch metadata; minted as a root input (c114, c217) | to be scoped | | open: large |
| C11 | Variadic returns | shared with A6; take it in Lane A | | | see A6 |
| C12 | A comprehension result mutated afterwards: `ys = [x * n for x in xs]; ys.append(n)` then `len(ys)`, `sum(ys)` | the append node was pruned by loop evaporation; the arena and the reduction kept the comprehension's row count (silent WRONG at `main`) | `loop_composer.evaporate_unrolled_loops` (liveness), `resident_sequence_materialization._static_sequence_capacity_survey`, the comprehension rule in `glsl_deployment_strategy._tensor_descriptor_rule` | `claude/project-thread-cz18s3` (PR #788) | done there (doc 36 §5). Still loud: a set comprehension then `add` (no proven capacity) |

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
| D1 | Private root storage sized `else 1` in the exported C wrapper (`ssa_c_backend.py`, the `element_count` expression near line 5905): make it a named refusal and run the corpus | the private-root-storage block of the C wrapper emitter | `claude/project-thread-h1z2w1` | PR open. **Observed:** the default was reached by 58/159 corpus programs (the lead said none), every one a rank-0, non-dynamic storage requirement whose `element_count` is `None` only because its shape is empty. That is now read as the proof of one cell; anything else is the shortfall `private-root-storage-unproven`. Corpus 159/159 PASS, fuzz seed 11 tier 3 30/30 PASS, gate identical to `main` in all four parts (aa 89, ab 7, ac 26, rec 20). No corpus program reaches the new refusal yet (**unknown** whether one can). |
| D2 | Planner region inputs default to float64. Count the fallbacks across the corpus again (~3,000 measured once), then supply the proofs. This is also the tire's T2 (`bool` actual vs float64 region formal). | `_planned_scalar_dtypes` (read), region-input typing in `glsl_deployment_strategy` (`or "float64"` sites near lines 21277 to 21917) | | open |
| D3 | `except Exception: pass` in decision paths. **Observed** with an instrumented full corpus at `cc6e6f3` (170/170 PASS): 18 swallowed errors in 7 programs. (a) `fortran_c_shell` binding-kind lookup (`materializing_binding_kind`): 5 hits in c62, c271, c272, because a literal binding's source is its value (`[1, 2]`, `'v='`), not a slot. Fixed: only `caller_value`/`caller_alias`/`caller_storage` ask the page, with no try. (b) `_resolve_binding_name` planned-shell walk: 0 hits, made loud. (c) `invalidate_proven_shape` site in `glsl_deployment_strategy`: 13 hits, see D3a. (d) `identity_concordance._attach_failure_claim`: 0 hits; it decorates an error already being raised, so it is diagnostic. Classified benign and left as is. | `fortran_c_shell` binding-kind lookup in the call-argument loop; `_resolve_binding_name` | `claude/project-thread-h1z2w1` | PR open |
| D3a | A shape withdrawal the book refuses, swallowed. In c02, c57, c58 and c62, `invalidate_shape_transformation` posts `shape_transformation_state` for the f-string glyph helpers (`append_decimal_integer_glyphs`, `hyp`). The book refuses it: "REVISE without a changed source; every Derived cell is stamped at or before the row's". Because the refusal is raised before the `proven_shape` invalidation, the stale proven shape stays on the book (**observed**). Whether any reader then uses it is **unknown**; it is **inferred** silent-capable. Next step: trap the post, find why the cause cell is not newer than the row, and supply that record. Then remove the try at the `invalidate_proven_shape` call. | the `invalidate_proven_shape` call site in `glsl_deployment_strategy` (descriptor invalidation cone); `identity_concordance.invalidate_shape_transformation` | | open |
| D4 | `"int"` vs `"int64"` spellings reaching physical checks (rule 7) | to be listed by the inventory | | open |
| D5 | Name, suffix and position matching (`.endswith(`, `[0]` on argument lists) in decision paths | to be listed by the inventory | | open: inventory first |

## Done

- A0 (chained comparison with a builtin conversion middle): branch `claude/project-thread-kbkzp7`, receipts in doc 35.
- A3 (`dict.update()` from a display is a table mutation): branch `claude/project-thread-kbkzp7`, receipts in doc 35.
