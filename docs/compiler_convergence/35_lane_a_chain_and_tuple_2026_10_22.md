# 35: Lane A, item A0 (2026-10-22)

Branch `claude/project-thread-kbkzp7`, from `main` `2532577`. Measured in a
cloud container (4 cores, deps from `requirements.txt` plus `ziglang`,
`llvmlite`, `pytest-timeout`).

## A0: a chained comparison whose middle is a builtin conversion

- **Program.** `digits.isdigit() and 0 <= int(digits) <= 9`
  (`identity_concordance.identity_log_lzma_preset`, line 9358; the bootstrap)
  and `eps < float(value) < hi` (`dt_controller.py:663`; the managed tire,
  **inferred** from doc 34). Corpus c211, c212.
- **Records that disagree.** Python evaluates each middle operand once.
  `_pure_operand` classed every `ast.Call` as effectful, so the normalizer
  refused the chain at source normalisation, the first stage of both targets.
- **Rule.** A middle operand may be spelled twice when it is a pure read or
  one of `int`, `float`, `len`, `abs`, `ord` (one positional argument)
  applied to one, and the module never rebinds that name
  (`_rebound_names`: assignment, `def`, `class`, parameter, import,
  `global`, `except ... as`, match captures). Both occurrences stay inside
  the conjunction, in place: a guarding `and` still short-circuits the first,
  and the second runs right after the first returned. Nothing is hoisted.
  Any other call middle (a user function, a method, a shadowed builtin,
  `int(x, 2)`, `int(*v)`) keeps the named refusal. The receipt lists the
  re-evaluated middles (`reevaluated_conversions`).
- **Assumption (inferred).** A second evaluation of these builtins over
  builtin-typed arguments is the first's value with no effect. A user
  `__int__`/`__len__` with an effect would break it; the native lanes do not
  dispatch to one (lanes plan A7).
- **Reach (observed by scan).** Over every `src/**/*.py`, 23 files refused
  before; after, one: `laplace_nd.py:382` (`v.dim()`, a method, refused by
  design). The 22 others held `len(x)`, `abs(x)`, `ord(c)` middles, many of
  them in compiler sources the bootstrap compiles (`fortran_c_shell.py`,
  `precompile_to_ssa.py`, `glsl_deployment_strategy.py`), so a fix for `int`
  and `float` alone would have moved the bootstrap's first refusal by one
  line.
- **Measured at `0d98ef1`.** Unit tests
  (`tests/test_chained_comparison_normalization.py`) 15 passed. Typed corpus
  161/161 PASS, 0 concordance findings. Fuzz `FUZZ_EXT=1` seed 10, 30
  programs (with the two new snippets): 30 PASS. Gate: aa 90, ab 7, ac 26,
  rec 20. The parent `2532577` in the same container: aa 89, ab 7, ac 26,
  rec 20. The one extra aa name,
  `test_sympy_process_graph_is_registered_before_python_dependency_pursuit`,
  fails identically when run alone on the parent's worktree (a cached
  symbolic process graph is returned, `cache_hit=True`), so it is
  order-dependent, not this change (**observed**; its cause is unknown).

## A1: handed over

A1 (tuple display birth) is owned by the "Tuple len silent wrong answer"
thread, whose fix also teaches the audit. Measured here and passed to it:
on `main`, `len(t)` of a tuple display with storage reads 0; a tuple read
only through constant projections has no storage; a display that repeats a
producer (`[m, m, 9]`) or holds another structural tuple's projection
(`[t[1], n]`) is refused on `main` for lists too ("lacks one exact element
producer").

## A3: `dict.update()` from a display

- **Program.** `d = {1: 10, 2: 20}; d.update({1: 5}); return d[1] + n`
  returned 13 for 8 on `main`, with a clean audit (**observed**). Corpus
  c214.
- **Records that disagree.** The emitted C stores both displays and looks
  up `d`, but nothing stores the argument's rows into `d`. `update` on a
  dict was not a mutation record of any kind: `_retain_source_sequence_
  mutation_records` keeps only list/set/bytes receivers, and the dict
  display births (`_dict_literal_mutations`) covered displays only.
- **Rule.** `d.update(<dict display>)` is one `update`/`mapping_items`
  mutation at the call, holding the display's key and value producers in
  order: the same table store `d[k] = v` lowers to
  (`_dict_update_mutations`). The capacity survey adds the rows' literal
  keys to the receiver's universe and no longer unbounds either side for
  that call. An update whose argument is not an unchanged display (it was
  changed after its birth, by `_mutated_sequence_node_ids`; or it is
  another mapping, keywords or `**`) is posted as a `mapping_source`
  update, which the table lowering refuses by name ("mapping update
  requires deterministic key/value pairs ...").
- **Still loud (frontier).** `d.update({1: d[2]})`, an update whose row
  reads its own receiver, gives a control effect-order cycle. A dict
  passed to a callee after an update has no proven capacity (as on
  `main`).
- **Measured at `0718f9a`.** Typed corpus 162/162 PASS, 0 findings. Fuzz
  `FUZZ_EXT=2` seed 12, 40/40 PASS. Gate: aa 89, ab 7, ac 26, rec 20, no
  name outside the parent's failures.

### A3 follow-up: the update orders its receiver's reads

The fuzzer's own update snippet (seed 13, p13_31: `d.update({1: n + 2,
3: 2})` and then reads of `d`) gave a loud control-order cycle. The
update's stores were not an effect in `_sequence_effect_read_edges`, so
`n + 2` was fused with the reads that come after the update. A dict update
is now an in-place effect on its receiver, as an append is. c214 holds it.
Measured at `b4a59c0`: corpus 162/162, fuzz `FUZZ_EXT=2` seed 14 60/60,
gate aa 89 / ab 7 / ac 26 / rec 20.

## A6 (part): lists returned by a callee

While I was reproducing the variadic-return item, three silent families
showed up. Each was **observed** wrong on `main` with a clean audit.

1. **A returned list display.** `def pick(n): return [n]`, then
   `t = pick(n); len(t) * 10 + t[0]` returned 0. The planner expanded the
   returned display to its member path `(0, 0)` (`_authored_output_paths`).
   No caller projection matched it (`t[0]` is `(0,)`), so the call bound no
   result, and the caller read its own arena, which was never born.
   **Rule:** a returned mutable collection is one storage, as an argument
   is (rule 3), so its path is the output itself. Tuples keep their member
   paths. Corpus c215.
2. **Different lists on different return edges.** With `return [n]` on one
   edge and `return []` on the other, the one output named only the last
   site, so every path read the empty list. The variadic sequence return
   is not lowered. `_refuse_divergent_collection_returns` now refuses it by
   name ("returns different collections in return slot k"). Test:
   `tests/test_divergent_collection_return.py`.
3. **The caller mutates a returned list.** `t = pick(n); t.append(9)` lost
   the append, and the caller then read past the length (garbage that
   changes from run to run). A call result carries `result_class_ref`, not
   `aggregate_kind`, and only `aggregate_kind` receivers were retained as
   mutation records. `_receiver_collection_kind` reads either one, for the
   retained records (`_retain_source_sequence_mutation_records`) and for
   the one definition of "changed" (`_method_mutated_receiver_ids`).
   Corpus c216.

Still open in A6: tuple variadic returns (`return (n,)` / `return ()` is
refused by the product, as before), and the lowering of a list returned on
two edges. A dict or set display returned by a callee is loud (a column ABI
conflict, and an undefined CondBr operand). Measured at `12ee174`: corpus
164/164 PASS, 0 findings; fuzz `FUZZ_EXT=2` seed 15, 60/60 PASS (5
programs use the new snippets); gate aa 89 / ab 7 / ac 26 / rec 20, nothing
outside the parent's failures.

## Follow-ups found by the fuzzer and by c986

1. **Capacity of hoisted callee storage** (`4c65f25f`). A callee's own
   list or dict lives in the entry frame (hoisted), but the C wrapper
   published capacity only for the entry's root sequences, so the
   callee's arena had capacity 0. `_linked_frame_sequences` follows the
   call's `callee_input_ids` and respells the callee's descriptors in
   entry ids. Corpus c217.
2. **Rule 7 at the frame ABI** (`265ee20e`). A primitive `int` formal's
   physical type arm used the class name `int` where the receipt says
   `int64`. It now reads the receipt dtype. This commit comes after (1)
   on purpose: alone it turned a loud c986 into a silent capacity-0 read.
3. **Concatenation before its operand's birth** (`a4c29be9`, fuzzer
   p16_29, observed wrong on `main`). `xs = [1, -1] + [n, 1]; d = {1: 6};
   return xs[2] * 10 + d[1]` returned 6. The `+` region completed at its
   own start, which precedes the right operand display, so the copy ran
   before `[n, 1]` was born. **Rule:** a region member that reads a
   display operand completes where that display ends
   (`_member_completion_position`). Corpus c218, fuzz snippet.

Measured at `a4c29be9` on a clean worktree: corpus 166/166 PASS, 0
findings; fuzz `FUZZ_EXT=2` seeds 16 and 17, 60/60 PASS each (seed 16
held the WRONG); gate aa 89 / ab 7 / ac 26 / rec 20, the same failure
names as the previous measurement, which matched the parent.

## Two more silent families (FUZZ_EXT=3 and a scratch probe)

Both were **observed** wrong on `main` (`2532577`) with a clean audit.

1. **Calls keep source order** (`d6639c0c`, `c52f8ef9`; fuzzer p18_11,
   p18_27). `xs0 = [2, n]; grow(xs0, n % 3 + 1); xs1 = [n, 4]; push(xs1,
   3)` then `return len(xs1) * 100 + xs1[-1]` lost the push. Each effect
   was ordered only before its own receiver's reads, but source calls are
   lowered in source order, so the reducer fused `n % 3 + 1` with the
   arithmetic over `xs1` and both calls landed after the reads. **Rule:**
   in `_sequence_effect_read_edges`, the effects and the source calls
   outside loops each precede the next, ordered by where each call ends.
   Native and intrinsic calls (`math.sqrt`, `len`) are operations inside a
   region. My first version also ordered those, which turned `c22_math`
   into a refusal; I caught that in the corpus and fixed it in `c52f8ef9`.
   Corpus c270.
2. **A list bound on two branches** (`753ea047`). `if n > 2: t = [n, 1]
   else: t = [5, 6]` then `len(t)` read 0, and appends after the merge
   were lost. The same happened in elif chains, ternaries and loops. The
   reducer's `aggregate_phi` had member Phis but no birth. **Rule:** the
   merge is its own storage, born where the conditional completes from its
   member Phis (`_list_literal_mutations`, `_authored_node_position`), and
   it is a local list for capacity. A change to the merge changes its arms
   (`_mutated_sequence_node_ids`), so `len(a)` of an aliased arm no longer
   folds. An arm that changes or is shared, or arms of different lengths,
   are refused by name (`tests/test_list_branch_merge.py`). Corpus c271.
   Still loud: such a merge inside a loop in a callee (a pointer Const at
   emission).

Measured at `c52f8ef9` on a clean worktree: corpus 168/168 PASS, 0
findings; fuzz `FUZZ_EXT=3` seeds 18, 19 and 21 and `FUZZ_EXT=2` seed 20,
60/60 PASS each (seed 18 held both WRONGs); gate aa 89 / ab 7 / ac 26 /
rec 20, the same names as the parent. Corpus numbers c270 to c279 belong
to this lane.
