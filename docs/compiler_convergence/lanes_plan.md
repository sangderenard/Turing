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

## Lane A: bootstrap families (and the tire, where it shares a group)

| rank | item | group (doc 34) | claimed functions | branch | status |
|---|---|---|---|---|---|
| A0 | Lower a chained comparison whose middle operand is a call. The middle is evaluated once, in place, never hoisted ahead of a guarding `and`. Duplicating it is allowed only for a proven-pure builtin conversion of a pure operand. Every other call still refuses by name. Then hand Albert both whole-program commands. | R0 | `ast_control_normalization._normalize_chained_comparisons`, `ast_control_normalization._pure_operand` | | open: blocks 100% of both targets |
| A1 | Give tuple displays a birth. Place the predicate region that produces the members before the effects that consume them, then birth tuples as lists are born. | S1 | `resident_sequence_materialization._list_literal_mutations`; the region-ownership rule that assigns the members' region to the later conditional (find it with a trap first) | | open: silent on `main` with a clean audit |
| A2 | Keep a return surface whole. `emit_outputs` must not fall back to `(output_id,)` when an aggregate layout is absent. The control finish publishes the layout. Accepted-unit accounting must not let one value stand for a tuple/optional. Add the false-acceptance check from doc 34 §1.4 to the bootstrap status tool. | B1, T3 | `fortran_c_shell._class_surface_ssa_program.emit_outputs` (nested, line ~28034); the control-finish member Phi publisher; accepted-unit result accounting | | open: false acceptance |
| A3 | Make `dict.update()` a recorded mutation, so `_mutated_sequence_node_ids` sees it. | B4 | `glsl_deployment_strategy._mutated_sequence_node_ids` (read only); the mutation-record writer for dict methods | | open: inferred silent-capable |
| A4 | Tire: the result-storage lease mints a record span member's `SSA_VALUE` row as a scalar. Revise it from the record storage contract. `_ssa_structure` reads the registered record storage before calling a shape-empty fact a scalar. | T1 | `structural_concordance._ssa_structure`; the result-storage lease writer in `fortran_c_shell` | | blocked on A0 (re-observe first) |
| A5 | Measure the core-chain cascade. After A0, ask Albert for a full bootstrap. Fill doc 34 §1.2 and re-file units into groups. | B6, B7 | none (measurement) | | blocked on A0 |
| A6 | Variadic returns: the sequence-lane lowering for `return ()`/`(x,)`/`[]`/`[x]` (units 034, 162, 212). | B2 | reducer VARIADIC consumer in `precompile_to_ssa`; `ir_sequence_tables` | | open |
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
| D1 | Private root storage sized `else 1` in the exported C wrapper (`ssa_c_backend.py`, the `element_count` expression near line 5905): make it a named refusal and run the corpus | the private-root-storage block of the C wrapper emitter | | open |
| D2 | Planner region inputs default to float64. Count the fallbacks across the corpus again (~3,000 measured once), then supply the proofs. This is also the tire's T2 (`bool` actual vs float64 region formal). | `_planned_scalar_dtypes` (read), region-input typing in `glsl_deployment_strategy` (`or "float64"` sites near lines 21277 to 21917) | | open |
| D3 | `except Exception: pass` in decision paths. The four that sit in compiler decisions come first: `glsl_deployment_strategy.py:11455` (planned-shell walk), `:18206` (`invalidate_proven_shape`), `fortran_c_shell.py:47416` (`materializing_binding_kind`), `identity_concordance.py:9680` (failure-claim record). The telemetry and `old == fact` comparisons are probably benign; classify them, don't change them. | those four `try` blocks | | open |
| D4 | `"int"` vs `"int64"` spellings reaching physical checks (rule 7) | to be listed by the inventory | | open |
| D5 | Name, suffix and position matching (`.endswith(`, `[0]` on argument lists) in decision paths | to be listed by the inventory | | open: inventory first |

## Done

(none yet)
