# Lanes plan

Ranked work queues for lanes A (bootstrap families), C (language frontier)
and D (defaults and fallbacks). Source: `34_2026_10_22_triage.md`. Rules:
`00_COMPILER_TAKEOVER_PROMPT.md` sections 6 to 10.

Updated 2026-10-23 at `main` `85cf0469`. This revision consolidates the plan edits of #798, #799, #800 and #802 (all but #800 merged), which no longer edit this file. Lanes send plan changes to the planner thread instead of editing it. The daily planner
routine rewrites this file. A worker takes the top unclaimed item of its
lane, writes its branch beside the claim, and does one item per run.

## Ranking this round

Rank is blast radius x depth x whether the group can answer wrongly with a
clean audit. Silent-capable groups come first whatever their size. Merged
since run 1: #786 (A0, A3, list returns, call order, branch-merged lists),
#787 (D1), #788 (S1 = A1, W2, C12, C13, C14), #789 (a set merged from
branch displays), #790 (D3), #791 (indexed stores and concatenations in the
call order, D2-W), #792 (C15, which is also the A2 thread's W4, and the
unrolled form of C2), #794 (D2 step 1). No lane PR is open.

| order | item | lane | silent? | why here |
|---|---|---|---|---|
| 1 | A2 remainder: false acceptance among the 55 `compiled-unverified` units (doc 34 §1.4) | A | yes, **observed** once (unit 027) | one false acceptance there is a silent wrong answer |
| 2 | W3c, the two-unit bootstrap regression from #796 (and W3b) | A | no, loud (W3b silent behind its refusal) | fixed in PR #800 (bootstrap back to 55/370, measured by that thread) |
| 3 | A4 tire `scalar_item_merge` disagreement in `run_superstep` | A | no, loud | the tire's first failure now that R1 is fixed (#801) |
| 4 | D2 next sources (`dict.get`, `StaticReference`, list/set comprehensions; step 2 waits locally) | D | **inferred** yes (a wrong dtype is a TYPE result) | 8,669 fallbacks left in 93 programs; depth is the planner |
| 5 | G1 `formatted glyph component has no resident producer` (12 bootstrap units) | A | no, loud | the largest own-failure family in the bootstrap. D3a (#798) showed nothing went stale in the glyph helpers, so G1 is its own cause (**inferred**) |
| 6 | E4 LLVM refusals, E3 Fortran tuple/list literals | E | no, loud | backend parity |
| 7 | A2b tuple-or-`None` returns | A | no, loud by name | bootstrap units with `fork = None` patterns |
| 8 | C1 remainder (generator over a folded list), then C3-C10 | C | no, loud | |

## Measurement 2026-10-23 (A5)

Run by the planner thread in a cloud container (4 cores, 15 GB). Both runs
took minutes, not hours. Albert said either benchmark may be run at any time
without asking.

**Bootstrap at `d8d41cb6`** (`tools/bootstrap_compiler_whole_program.py
--continue-after-failure`). The provenance core has 370 units (it had 330
on 2026-10-21). 55 are `compiled-unverified`, against 23/330 admitted last
time. 246 are `failed` and 69 are `partial`. The run then stops at
`load_composed_source_units` ("SSA enclosure assembly contains an incomplete
source unit"), so stage 5 and later did not run, and there is no
`bootstrap-state.json` for `tools/compiler_bootstrap_status.py`. All of
this is **observed**. The grouped table is
`evidence/bootstrap_groups_2026_10_23_d8d41cb6.txt`.
- 196 units are refused only because a dependency is incomplete ("source
  enclosure lacks a complete dependency product").
- 69 `partial` units have shortfalls of these kinds: unexplained public
  arguments (50), concordance consumer refusal (48), unresolved boundary
  types (46), structural output shortfalls (39), unresolved required
  source values (30), native result layout mismatch (13). The example is a
  lambda reading `item.mass`, where the attribute is `dispatch_metadata`.
  That is B3/A7.
- Own failures (about 50 units):
  - 12 formatted glyph (G1);
  - 3 undeclared native interface argument roles;
  - 2 or 3 each of union arm frames, provenance findings and multiple
    sequential control owners;
  - about 30 singletons. These include return-slot and carried-arm
    incompleteness, control-order cycles, sequence dtype disagreements
    and a sort scratch layout.

**Tire** (`tools/build_balloon_tire_native.py --managed-dt --backend c`).
The build fails, so validation had no manifest to read. The failure point
moved with each merge (**observed**):
- `af315b9`: the A0 chained comparison.
- `3ffb64d6`: past planning, into `run_superstep`'s SSA lowering (a
  `scalar_item_merge` disagreement, record 78 against a minted record).
- `7b3ec6e1`: past the depth-0 region reduction.
- `6e621cd5` (#791) to `d8d41cb6`: a cycle in the depth-0 region
  reduction (R1).

**Rerun at `4b81b85a` (after #796).** The tire still hits R1. The
bootstrap compiles 53/370 units, down from 55 (**observed**). Unit 35,
`IdentityPage._virtual_fact`, goes from compiled to partial. Its shortfall
is an unexplained public argument, an int64 tag of a pending
Optional[tuple] call result (value 8). Unit 183, `IdentityPage.cell`, then
fails as its dependent. Units 36, 186 and 225 also fail now on the
dependency, which hides their own earlier errors. That is W3c.

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
branch has used, merged or not. On 2026-10-23 two pairs collided (c290-c293 and c330); this table is the settlement.

| range | lane / item | branch | used |
|---|---|---|---|
| c211-c218 | Lane A: A0, A3 and follow-ups | `claude/project-thread-kbkzp7` | c211, c212, c214-c218 |
| c230-c239 | Lane A: A2 optional returns | `claude/project-thread-h4aybe` | c230-c239 (used up) |
| c240-c249 | Lane C: C1 constant-list comprehensions, C15, C2, C16, C17 | `claude/project-thread-2nw77p` | c240-c248 |
| c260-c269 | Lane A: S1 tuple birth, W2 concatenations; Lane C: C12, C13, C14 | `claude/project-thread-cz18s3` | c260-c269 (full) |
| c270-c279 | Lane A: A0 thread follow-ups (calls in source order, branch-merged displays) | `claude/project-thread-kbkzp7` | c270, c271 |
| c280-c289 | Silent lane follow-ups (indexed stores and concatenations in the call order; comprehension births; grown sets) | `claude/project-thread-cz18s3` | c280-c289 (full) |
| c290-c299 | Lane E: list slices, sum of a folded list, slice concatenation widths, loop targets, Fortran list reductions | `claude/project-thread-kbkzp7` | c290-c299 (full) |
| c300-c309 | Lane E: Fortran span cells, callee swaps, dict.get defaults, colliding callee lists, optional returns, list literals | `claude/project-thread-kbkzp7` | c300-c309 (full) |
| c310-c319 | Silent lane: reads before later effects on the same list, returned reads | `claude/project-thread-cz18s3` | c310-c319 (full) |
| c320-c329 | Silent lane: comprehension loops and conditional expressions in the source-call order | `claude/project-thread-cz18s3` | c320, c321 |
| c330-c339 | Silent lane: fixed-capacity full is loud, dict `get` in a loop, callee branch displays | `claude/project-thread-cz18s3` | c330-c332 |
| c340-c349 | free (released by Lane E on 2026-10-23; never used) | | |
| c350-c359 | Lane E: Fortran positional sequence helpers (c350, renumbered from c330, which the silent lane holds); follow-ups (enumerate loops, tuple comprehension targets) from c351 | `claude/project-thread-kbkzp7` | c350 (merged, #799); c351- (local) |
| c360-c369 | Lane A: W3b wrappers forwarding value unions (renumbered from c290-c293, which Lane E holds) | `claude/project-thread-h4aybe` | c360-c363 (PR #800) |
| c370-c379 | next free | | |

## Lane A: bootstrap families (and the tire, where it shares a group)

| rank | item | group (doc 34) | claimed functions | branch | status |
|---|---|---|---|---|---|
| A2 | Keep a return surface whole: the remaining pieces. `emit_outputs` must not fall back to `(output_id,)` when no `Ret` exists; accepted-unit accounting must not let one value stand for a tuple/optional (`project_compilation_product`, `len(paths) == 1 and len(published_ids) == 1`); add the doc 34 §1.4 false-acceptance check to the bootstrap status tool. The optional-return part is done (doc 34 §6, §7). | B1, T3 | `fortran_c_shell._class_surface_ssa_program.emit_outputs`; accepted-unit result accounting | `claude/project-thread-h4aybe` | claimed 2026-10-22; optional-return part, W1, W3 merged (#796). False acceptance has no repro outside a bootstrap run (doc 34 §9): waits on A5 |
| A2b | Lower a returned tuple-or-`None` (`if c: return (a, b)` / `return None` in a bootstrap unit; `fork = (a, b)` / `fork = None; return fork` anywhere). It refuses by name now (`its newest binding ... was not materialized`), was silent before. Observed by the silent-wrong-answers thread: the planner builds the aggregate Phi (presence member 15, member Phis 16, 17), but the root's return reads the handle (10) instead of that layout, and the tuple's region is then pruned as dead. | B1 | the control finish's output resolution in `precompile_to_ssa`; find the pruner with a trap | | open: loud. Run 4 trace in doc 34 §9: the planner's optional merge has no record the control finish can read, and the tuple-tuple name merge is refused too |
| A4 | Tire: after R1 (#801) the managed build stops in `run_superstep`'s SSA lowering: `scalar_item_merge disagreement for (run_superstep@control:0, 80): recorded=ssa_value 78, proposed=a minted value` (**observed** at `3ffb64d6`, and reported again by the silent lane at `6a3ab933`). The doc 32 transport failure (`Metrics.error_present`, span (15,) leased as ()) is behind it and still unobserved at `main`. Repro idea: lift `run_superstep`'s control loop with its merged scalar item and trap the `scalar_item_merge` post. Suspected (**inferred**): a pass mints a fresh value for an item the control merge already recorded, instead of reading record 78. | T1 | to be scoped by the trap; `structural_concordance._ssa_structure` for the later transport row | | next for Lane A after W3c |
| W3c | Bootstrap regression from #796, measured by the planner (55 to 53 of 370 units): unit 35 `IdentityPage._virtual_fact` tests an Optional[tuple] call result with `is None`, and W3's rewrite minted a tag nothing publishes (`unexplained-public-arguments`); unit 183 and its dependants failed behind it. Doc 34 §9. | loud | `precompile_to_ssa._union_none_identity_subject` | `claude/project-thread-h4aybe` | fixed on the branch; bootstrap re-measured in doc 34 §9 |
| G1 | `formatted glyph component has no resident producer: value=N projection='scalar'`: 12 bootstrap units fail on it themselves, for example `id_space.describe`, `CorrelationTable._abi_or_member_key` and `_graph_work_publication_findings`. Repro idea: lift `id_space.describe` and lower it through the public entry. Suspected (**inferred**): the f-string glyph recipe takes a scalar component whose producer was folded or pruned. It may share the `append_decimal_integer_glyphs` path with D3a. | new | f-string glyph recipe builder (shared with A8) | | open |
| A6 | Variadic returns: the sequence-lane lowering for `return ()`/`(x,)`/`[]`/`[x]` (units 034, 162, 212). | B2 | reducer VARIADIC consumer in `precompile_to_ssa`; `ir_sequence_tables` | `claude/project-thread-kbkzp7` | claimed 2026-10-22; list returns fixed or refused (doc 35), tuple variadic and two-edge list lowering open |
| A7 | `Any`-typed receivers and generic `str()`/`!r` of facts: runtime type dispatch over the fact classes. | B3 | to be scoped | | open: needs a design note first |
| A8 | `'|'.join(tokens)` glyph recipe in f-strings (unit 6). | B5 | f-string glyph recipe builder | | open |
| W3 | A caller's `is None` test on a callee returning `x if c else None` (a conditional value union) did not compile. Two lost records: the linker rebound the call's placeholder (the scalar `v`) to the (payload, tag) container, and `v is None` lowered to `Eq(payload, None)`. Doc 34 §8; corpus c237-c239. | silent behind loud | `fortran_c_shell` linker placeholder rebind; `precompile_to_ssa.lower_control_expression` | `claude/project-thread-h4aybe` | done on the branch |
| W3b | A wrapper that forwarded a value-union result (`def wrap(...): return maybe(...)`) was refused, and became a silent `is None` once the refusal was lifted. Three lost records. Doc 34 §9; corpus c360-c363 (renumbered from c290-c293). | loud, silent behind it | `precompile_to_ssa._native_value_union_row_payloads`; `fortran_c_shell._propagate_native_type_tag_results` | `claude/project-thread-h4aybe` | done on the branch |

## Lane C: language frontier

Each item turns a LOUD cluster into a lowering, using existing mechanisms,
and adds a corpus program and a fuzz snippet. The order is by how often the
fuzzer and bootstrap meet the cluster (**inferred** from doc 33), with
silent-capable items first.

| rank | item | where the record is missing (doc 33) | claimed functions | branch | status |
|---|---|---|---|---|---|
| C1 | Reductions over a structural aggregate: `any((1 > n, 2 > n))`, `sum(x * n for x in xs)` with `xs` literal | the region takes the aggregate as an unproduced input, and no layer assembles it from its unrolled members | `resident_sequence_materialization._list_literal_mutations`; `glsl_deployment_strategy._tensor_descriptor_rule` (loop-target branch) | `claude/project-thread-2nw77p` | list/set comprehension part done (e7f4a46d, landed with PR #788; c240-c243). Still loud by name: generator over a folded list (`any(x > n for x in xs)`: the GeneratorExp materializer is not a declared sequence) and tuple displays |
| C3 | Dict `del d[k]` / `d.pop(k)` | placed by graph-id order into a neighbouring region (a cycle); `pop` has no lowering. It should be one lexical mapping effect. | `_control_block_position` (read only); mapping effect lowering | | open |
| C4 | `d.get(k, 0)` in a loop | control order cycle (c202) | `_sequence_effect_read_edges` | | open |
| C5 | Standalone slice `ys = xs[1:n]` | no lowering; only `r += xs[a:b]` is recognized | slice recognizer; `ir_sequence_tables` | | open |
| C6 | Nested list stores and `xs[i % 4] += i` in a loop | no source cell for the control block; a carried read with no lexical binding row | indexed-store chain in `glsl_deployment_strategy` | | open |
| C7 | Early `return` from a loop over a list parameter | (c216) | loop exit lowering | | open |
| C8 | Membership in a folded literal inside a loop (`i in ts`) | structural recovery builds at entry, where the loop target does not exist; the folded literal has no resident sequence | membership survey | | open |
| C9 | Multi-element replication `[a, b] * k` | no operation writes its rows; fill repeats one value | `_sequence_concat_ops` | | open |
| C10 | Strings as values (`s = "a" if c else "b"`, `len(s)`, `==`, `startswith`) | dispatch metadata; minted as a root input (c114, c217) | to be scoped | | open: large |

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
| D2 | Planner region inputs default to float64. **Observed** with an instrumented full corpus at `3ffb64d` (184/184 PASS): 9,547 fallbacks over the 14 `or "float64"` sites in `glsl_deployment_strategy`. `_value_shape_dtype` (the last resort after tensor, specialization, node, descriptor, `_planned_scalar_dtypes` and domain) takes 9,521 of them, in 92 programs. `_tensor_descriptor`'s generic tensor descriptor takes 26, in 18 programs. The other 12 sites take none. The 4 `else "float64"` sites are literal-type mappings and legitimate. On a sample, the unproven nodes were branch-join `Phi`s, then `GeneratorExp`/`comprehension`/`sum` over them, `range(...)` constants, `dict.get`, `StaticReference`s, and arithmetic over any of these. Step 1 (this PR): a `Phi` is typed from its `body`/`orelse` arms, as `IfExp` already was. Result: 8,056 left, in 80 programs (-1,491, and 12 programs now have none). Next steps, one source each: `range` constants (element int64), generator and comprehension elements, `dict.get` (the dict's value join), `StaticReference`. | `_planned_scalar_dtypes` (Phi rule); later sources in the same function | `claude/project-thread-h1z2w1` | step 1 done (PR #794). Step 2 ready locally, PR after #798: `range(...)` constants as int64 rows, a GeneratorExp from its elt's class, `sum(xs, start)` promoted with the start (list and set comprehensions stay with the collection rule). Measured at `2806360` over 205 programs: 9,001 to 8,669 fallbacks (-332), still in 93 programs; corpus 205/205, fuzz seed 15 tier 3 0 WRONG 0 TYPE, gate same as main. Next: `dict.get`, `StaticReference`, list/set comprehensions |
| D4 | `"int"` vs `"int64"` spellings reaching physical checks (rule 7) | to be listed by the inventory | | open |
| D5 | Name, suffix and position matching (`.endswith(`, `[0]` on argument lists) in decision paths | to be listed by the inventory | | open: inventory first |

## Lane E: backend parity

`TYPED_CHECK_BACKENDS=c,llvm,fortran python tools/native_corpus_typed_check.py tests/native_semantics_corpus`
runs every corpus program through LLVM (executed) and Fortran (emitted;
this container has no Fortran compiler) beside C. A `DISAGREE` line or an
LLVM `WRONG`/`RUN-CRASH` is a finding; `FUZZ_BACKENDS=llvm` does the same
for the fuzzer. Corpus block c275-c279. Receipts: doc 37.

| rank | item | claimed functions | branch | status |
|---|---|---|---|---|
| E3 | Fortran: `cannot express literal (tuple/list)` (9 programs) | `ssa_fortran_backend` constant emission | | open |
| E4 | LLVM refusals still standing: `address value ... has no concorded span` (callee reads of a caller list), `aggregate projection ... has no compile-time member` (optional calls), `dict.get` default Phi (c178). Loop targets over comprehensions and `zip` are fixed (E6, c295-c296). Slice concatenation's i32/i64 Phis are fixed (`ir_sequence_tables.lower_sequence_append_slice` is int64) | `ir_sequence_tables` | `claude/project-thread-kbkzp7` | in part |

## Done

- C2 (a list rebound in a retained loop), C16 (layout members kept through loop pruning), C17 (a post-loop subscript when the loop may not run): PR #802, merge `0ef8ce26`, c246-c248.
- Lane E E5-E17 (slices, Fortran reductions, span formals, `dict.get` defaults, LLVM span loads, Fortran optional returns and list literals): PR #799, merge `524c7378`, doc 37.
- D3a (a second withdrawal of a withdrawn shape row is not a failure; nothing went stale): PR #798, merge `85cf0469`.
- R1 (tire region cycle: an order edge against a value edge), reads before later effects, fixed-capacity full is loud, dict `get` in loops: PR #801, merge `fb6b1acb`, doc 36 §10-§14.
- Lane E E1, E2 (LLVM wrong answers, crashes and refusals): PR #795, merge `08ba0171`, doc 37.
- A2 optional return surfaces, W1, W3: PR #796, merge `4b81b85a`, receipts in doc 34 §6 to §8.
- A5 first measurement after the round: see "Measurement 2026-10-23" above.
- A1 / S1 (tuple displays are born and bounded), W2, C12, C13, C14: PR #788, merge `ca9668a4`, receipts in doc 36.
- D1 (private root storage proven or refused): PR #787, merge `cc6e6f36`.
- D3 (no silent except around the binding-kind lookup or the planned-shell walk): PR #790, merge `3ffb64d6`. D3a stays open.
- Indexed stores and list concatenations in the source-call order (fuzz p28_37; D2-W, fuzz p13_5): PR #791, merge `6e621cd5`, doc 36 §8, §9.
- C15 (`sort()` kept beside an unrolled loop; also W4) and C2's unrolled form: PR #792, merge `bbf81da8`.
- A set merged from branch displays (c274): PR #789, merge `4567cb65`.
- D2 step 1 (a branch join's scalar class is the join of its arms): PR #794, merge `c0614a20`.
- A0 (chained comparison with a builtin conversion middle): PR #786, merge `57709b54`, receipts in doc 35.
- A3 (`dict.update()` from a display is a table mutation): PR #786, merge `57709b54`, receipts in doc 35.
