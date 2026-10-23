# Lanes plan

Ranked work queues for lanes A (bootstrap families), C (language frontier)
and D (defaults and fallbacks). Source: `34_2026_10_22_triage.md`. Rules:
`00_COMPILER_TAKEOVER_PROMPT.md` sections 6 to 10.

Updated 2026-10-23 at `main` `4b81b85a` (planner run 2, after the A5 measurement). The daily planner
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
| 1 | A2 remainder: false acceptance. 55 units now report `compiled-unverified`; check each export against its source surface (doc 34 §1.4) | A | yes, **observed** once (unit 027) | the 55 are the only units whose answers would be trusted; one false acceptance there is a silent wrong answer |
| 2 | C2 retained form: a display rebound inside a retained loop reads `len(xs)` as 0 | C | yes, **observed** at `3ffb64d6` | silent with a `descriptor-member-shared` finding; any loop that rebinds a list |
| 3 | D3a stale `proven_shape` after a refused withdrawal | D | **inferred** yes | 4 corpus programs reach it; nobody has shown a reader of the stale shape |
| 4 | R1 tire region-fusion cycle from be0dd1fc (#791), and W3c, the two-unit bootstrap regression from #796 | silent lane, A | no, loud | regressions of the two benchmarks, so they go before new frontier |
| 5 | D2 next sources (`range` constants, generator elements, `dict.get`, `StaticReference`) | D | **inferred** yes (a wrong dtype is a TYPE result) | 8,056 fallbacks left in 80 programs; depth is the planner |
| 6 | A2b tuple-or-`None` returns | A | no: loud by name on the A2 branch | bootstrap units with `fork = None` patterns |
| 7 | G1 `formatted glyph component has no resident producer` (12 bootstrap units fail on it themselves) | A | no, loud | the largest own-failure family in the bootstrap; f-strings are everywhere in the compiler (**inferred** to share A8 and D3a's glyph helpers) |
| 8 | C16 (with C2's constant-member form), C1 remainder, W3b, then the rest of C | C, A | no, loud | C16 is 4/120 at tier 1 seed 11 |

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
branch has used, merged or not.

| range | lane / item | branch | used |
|---|---|---|---|
| c211-c218 | Lane A: A0, A3 and follow-ups | `claude/project-thread-kbkzp7` | c211, c212, c214-c218 |
| c230-c239 | Lane A: A2 optional returns | `claude/project-thread-h4aybe` | c230-c239 (used up) |
| c240-c249 | Lane C: C1 constant-list comprehensions, C15, C2 | `claude/project-thread-2nw77p` | c240-c245 |
| c260-c269 | Lane A: S1 tuple birth, W2 concatenations; Lane C: C12, C13, C14 | `claude/project-thread-cz18s3` | c260-c269 (full) |
| c270-c279 | Lane A: A0 thread follow-ups (calls in source order, branch-merged displays) | `claude/project-thread-kbkzp7` | c270, c271 |
| c280-c289 | Silent lane follow-ups (indexed stores and concatenations in the call order) | `claude/project-thread-cz18s3` | c280-c282 |
| c290-c299 | next free | | |

## Lane A: bootstrap families (and the tire, where it shares a group)

| rank | item | group (doc 34) | claimed functions | branch | status |
|---|---|---|---|---|---|
| A2 | Keep a return surface whole: the false-acceptance remainder. Accepted-unit accounting must not let one value stand for a tuple/optional (`project_compilation_product`, `len(paths) == 1 and len(published_ids) == 1`). Add the doc 34 §1.4 check to the bootstrap status tool, then run it over the 55 `compiled-unverified` units of the 2026-10-23 measurement. The optional surfaces, W1 and W3 landed in #796. | B1, T3 | accepted-unit result accounting in `project_compilation_product`; `tools/compiler_bootstrap_status.py` | | next for Lane A |
| A2b | Lower a returned tuple-or-`None` (`if c: return (a, b)` / `return None` in a bootstrap unit; `fork = (a, b)` / `fork = None; return fork` anywhere). It refuses by name now (`its newest binding ... was not materialized`), was silent before. Observed by the silent-wrong-answers thread: the planner builds the aggregate Phi (presence member 15, member Phis 16, 17), but the root's return reads the handle (10) instead of that layout, and the tuple's region is then pruned as dead. | B1 | the control finish's output resolution in `precompile_to_ssa`; find the pruner with a trap | | open: loud |
| A4 | Tire: the result-storage lease mints a record span member's `SSA_VALUE` row as a scalar. Revise it from the record storage contract. `_ssa_structure` reads the registered record storage before calling a shape-empty fact a scalar. Repro idea: lift `Metrics` with its `error_present` span (15,) into a scratch program that returns the record through one call edge. Suspected pass (**inferred**): the lease writer, which reads the fact's empty shape before the record storage. | T1 | `structural_concordance._ssa_structure`; the result-storage lease writer in `fortran_c_shell` | | open: A0 merged; re-observe at `main` first (needs a tire run, see A5) |
| R1 | Tire: a cycle in the depth-0 region reduction (`reduce_scheduled_shader_regions`, `nx.topological_generations` on the quotient graph). The tire builds past it at `7b3ec6e1` and fails at `6e621cd5`, so be0dd1fc (list concatenations in the source-call order) introduced it. Repro idea: rebuild the tire with a hook on `_sequence_effect_read_edges` that prints the concatenation edges, and find the one that closes the cycle; then lift that function. Suspected (**inferred**): a concatenation ordered after a call whose value it feeds. | T0 | `glsl_deployment_strategy._sequence_effect_read_edges` | `claude/project-thread-cz18s3` (asked) | open: regression |
| W3c | Regression from #796 (W3): a value-union call's `is None` now reads a tag that nothing publishes when the union is Optional[tuple]. Unit 35 `IdentityPage._virtual_fact` shows `unexplained-public-arguments` with `native_type_tag_pending_call_result` for value 8, and unit 183 fails as its dependent. Repro idea: `def f(c): return (1, 2) if c else None` and a caller `x = f(c); return 0 if x is None else x[0]`. Suspected (**inferred**): the W3 tag read needs A2b's tuple-or-None layout, which is not there yet. Supply the tag from that layout, or refuse by name before minting it. | B1 | the W3 tag read in `ssa_optional_values` / `fortran_c_shell` (the code #796 added) | `claude/project-thread-h4aybe` (asked) | open: regression, before A2 |
| G1 | `formatted glyph component has no resident producer: value=N projection='scalar'`: 12 bootstrap units fail on it themselves, for example `id_space.describe`, `CorrelationTable._abi_or_member_key` and `_graph_work_publication_findings`. Repro idea: lift `id_space.describe` and lower it through the public entry. Suspected (**inferred**): the f-string glyph recipe takes a scalar component whose producer was folded or pruned. It may share the `append_decimal_integer_glyphs` path with D3a. | new | f-string glyph recipe builder (shared with A8) | | open |
| A6 | Variadic returns: the sequence-lane lowering for `return ()`/`(x,)`/`[]`/`[x]` (units 034, 162, 212). | B2 | reducer VARIADIC consumer in `precompile_to_ssa`; `ir_sequence_tables` | `claude/project-thread-kbkzp7` | claimed 2026-10-22; list returns fixed or refused (doc 35), tuple variadic and two-edge list lowering open |
| A7 | `Any`-typed receivers and generic `str()`/`!r` of facts: runtime type dispatch over the fact classes. | B3 | to be scoped | | open: needs a design note first |
| A8 | `'|'.join(tokens)` glyph recipe in f-strings (unit 6). | B5 | f-string glyph recipe builder | | open |
| W3 | A caller's `is None` test on a callee returning `x if c else None` (a conditional value union) did not compile. Two lost records: the linker rebound the call's placeholder (the scalar `v`) to the (payload, tag) container, and `v is None` lowered to `Eq(payload, None)`. Doc 34 §8; corpus c237-c239. | silent behind loud | `fortran_c_shell` linker placeholder rebind; `precompile_to_ssa.lower_control_expression` | `claude/project-thread-h4aybe` | done on the branch |
| W3b | A wrapper that forwards a value-union result (`def wrap(...): return maybe(...)`) is refused: `shared heterogeneous sequence payload changes its physical ABI: value=2, payload=(None, ())`. Loud on `main` too. Doc 34 §8. | loud | forwarding of `native_value_union_return_layouts` through an exact wrapper | | open |

## Lane C: language frontier

Each item turns a LOUD cluster into a lowering, using existing mechanisms,
and adds a corpus program and a fuzz snippet. The order is by how often the
fuzzer and bootstrap meet the cluster (**inferred** from doc 33), with
silent-capable items first.

| rank | item | where the record is missing (doc 33) | claimed functions | branch | status |
|---|---|---|---|---|---|
| C1 | Reductions over a structural aggregate: `any((1 > n, 2 > n))`, `sum(x * n for x in xs)` with `xs` literal | the region takes the aggregate as an unproduced input, and no layer assembles it from its unrolled members | `resident_sequence_materialization._list_literal_mutations`; `glsl_deployment_strategy._tensor_descriptor_rule` (loop-target branch) | `claude/project-thread-2nw77p` | list/set comprehension part done (e7f4a46d, landed with PR #788; c240-c243). Still loud by name: generator over a folded list (`any(x > n for x in xs)`: the GeneratorExp materializer is not a declared sequence) and tuple displays |
| C2 | A display rebound inside a loop: `xs = [i + n, i]` | UNROLLED loop (fixed on the branch, c245): the cloned display had no SOURCE_AGGREGATE_LAYOUT row, so its birth refused (`List # lacks one exact element producer`), and a post-loop `xs[0]` folded to a body node that evaporation never remapped. RETAINED loop (still open, **silent**): `xs = [0, 0]; for i in range(4): t += xs[0]; xs = [i + n, i]` then `len(xs)` reads 0 at `main` 3ffb64d6 (audit: `descriptor-member-shared`); the rebound display has no birth into the carried storage. Also loud: a cloned display with a constant member (`ys = [j * 2, j]` in a second loop; `xs = [i + n, 6, i]`, fuzz p11_77): a constant member the cloned layout names is gone by birth time | `loop_composer.evaporate_unrolled_loops` (cloned layouts, last-iteration reads) | `claude/project-thread-2nw77p` | unrolled form done (PR #792, c245); retained form open and next for Lane C. Repro idea: the four-line retained loop above. Suspected (**inferred**): the carried storage's birth, which writes the first display only |
| C3 | Dict `del d[k]` / `d.pop(k)` | placed by graph-id order into a neighbouring region (a cycle); `pop` has no lowering. It should be one lexical mapping effect. | `_control_block_position` (read only); mapping effect lowering | | open |
| C4 | `d.get(k, 0)` in a loop | control order cycle (c202) | `_sequence_effect_read_edges` | | open |
| C5 | Standalone slice `ys = xs[1:n]` | no lowering; only `r += xs[a:b]` is recognized | slice recognizer; `ir_sequence_tables` | | open |
| C6 | Nested list stores and `xs[i % 4] += i` in a loop | no source cell for the control block; a carried read with no lexical binding row | indexed-store chain in `glsl_deployment_strategy` | | open |
| C7 | Early `return` from a loop over a list parameter | (c216) | loop exit lowering | | open |
| C8 | Membership in a folded literal inside a loop (`i in ts`) | structural recovery builds at entry, where the loop target does not exist; the folded literal has no resident sequence | membership survey | | open |
| C9 | Multi-element replication `[a, b] * k` | no operation writes its rows; fill repeats one value | `_sequence_concat_ops` | | open |
| C10 | Strings as values (`s = "a" if c else "b"`, `len(s)`, `==`, `startswith`) | dispatch metadata; minted as a root input (c114, c217) | to be scoped | | open: large |
| C16 | A comprehension over a folded list beside membership in another folded list: `ys = [x - n for x in [4, 4, 8]]` with `n in [5, 0, 2]` | loud at `main` 3ffb64d6: `List # lacks one exact element producer` (from the list birth in `_list_literal_mutations`); fuzzer tier 1 seed 11 p11_0, p11_24, p11_78, p11_104 (and, with the C2 snippet in the pool, p11_0, p11_23, p11_76, p11_102; all loud at `main` too) | to be trapped; *inferred* to share C2's lost constant member | | open: loud, after C2 (same function) |

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
| D2 | Planner region inputs default to float64. **Observed** with an instrumented full corpus at `3ffb64d` (184/184 PASS): 9,547 fallbacks over the 14 `or "float64"` sites in `glsl_deployment_strategy`. `_value_shape_dtype` (the last resort after tensor, specialization, node, descriptor, `_planned_scalar_dtypes` and domain) takes 9,521 of them, in 92 programs. `_tensor_descriptor`'s generic tensor descriptor takes 26, in 18 programs. The other 12 sites take none. The 4 `else "float64"` sites are literal-type mappings and legitimate. On a sample, the unproven nodes were branch-join `Phi`s, then `GeneratorExp`/`comprehension`/`sum` over them, `range(...)` constants, `dict.get`, `StaticReference`s, and arithmetic over any of these. Step 1 (this PR): a `Phi` is typed from its `body`/`orelse` arms, as `IfExp` already was. Result: 8,056 left, in 80 programs (-1,491, and 12 programs now have none). Next steps, one source each: `range` constants (element int64), generator and comprehension elements, `dict.get` (the dict's value join), `StaticReference`. | `_planned_scalar_dtypes` (Phi rule); later sources in the same function | `claude/project-thread-h1z2w1` | step 1 done (PR #794); next sources open, after D3a |
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
| E1 | LLVM wrong answers and crashes (static tuples, root frame cells, owned frame storage) | `ssa_llvm_backend`: aggregate `Const`, `GetElementPtr`, `Load`, `pointer()`, the public wrapper's frame | `claude/project-thread-kbkzp7` | fixed locally, PR after #789 |
| E2 | LLVM refusals: loop literals entering int64/bool Phis, flat reductions | `precompile_to_ssa` sequence query lowerings (`sum`/`any`/`count`/`index`/`remove`); `ssa_llvm_backend` reduction arm | `claude/project-thread-kbkzp7` | fixed locally, PR after #789 |
| E3 | Fortran: `cannot express literal (tuple/list)` (9 programs) | `ssa_fortran_backend` constant emission | | open |
| E4 | LLVM refusals still standing: `address value ... has no concorded span` (callee reads of a caller list), `aggregate projection ... has no compile-time member` (optional calls), integer ops on i64 | to be scoped | | open |

## Done

- A2 optional return surfaces, W1, W3: PR #796, merge `4b81b85a`, receipts in doc 34 §6, §7.
- A5 first measurement after the round: see "Measurement 2026-10-23" above.
- A2, optional return surfaces, W1 (an optional helper's merge has no class) and W3 (a value-union call's `is None`): branch `claude/project-thread-h4aybe`, receipts in doc 34 §6 to §8. Corpus block c230 to c239 belongs to this thread (used up).
- A1 / S1 (tuple displays are born and bounded), W2, C12, C13, C14: PR #788, merge `ca9668a4`, receipts in doc 36.
- D1 (private root storage proven or refused): PR #787, merge `cc6e6f36`.
- D3 (no silent except around the binding-kind lookup or the planned-shell walk): PR #790, merge `3ffb64d6`. D3a stays open.
- Indexed stores and list concatenations in the source-call order (fuzz p28_37; D2-W, fuzz p13_5): PR #791, merge `6e621cd5`, doc 36 §8, §9.
- C15 (`sort()` kept beside an unrolled loop; also W4) and C2's unrolled form: PR #792, merge `bbf81da8`.
- A set merged from branch displays (c274): PR #789, merge `4567cb65`.
- D2 step 1 (a branch join's scalar class is the join of its arms): PR #794, merge `c0614a20`.
- A0 (chained comparison with a builtin conversion middle): PR #786, merge `57709b54`, receipts in doc 35.
- A3 (`dict.update()` from a display is a table mutation): PR #786, merge `57709b54`, receipts in doc 35.
