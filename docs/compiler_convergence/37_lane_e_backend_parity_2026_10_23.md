# 37. Lane E: backend parity (2026-10-23)

The typed corpus check now runs each program through LLVM (compiled and
executed) and Fortran (compiled and executed when a Fortran compiler is present; `apt-get install gfortran` in the cloud container, otherwise emitted only)
beside C:

```
TYPED_CHECK_BACKENDS=c,llvm,fortran python tools/native_corpus_typed_check.py tests/native_semantics_corpus
```

Each backend is judged against Python on its own line. `DISAGREE c/llvm`
marks a program whose LLVM values differ from C's on the same input. A
native crash in one program is reported as `RUN-CRASH` for that program,
because each LLVM run happens in a forked child. `FUZZ_BACKENDS=llvm` on
`tools/differential_fuzz.py` adds the same LLVM comparison to the fuzzer.
An LLVM refusal of a program that C ran is reported there as `LLVM-LOUD`.

## Census

| | C | LLVM pass | LLVM wrong | LLVM crash | LLVM refused | Fortran emitted | Fortran refused |
|---|---|---|---|---|---|---|---|
| start (`74d79ec7`, 171 programs) | 171/171 | 132 | 5 | 1 | 33 | 162 | 9 |
| this branch (176 programs) | 176/176 | 157 | 0 | 0 | 19 | 166 | 10 |

The extra Fortran refusal is the new program c275. It is in the same record
as the other nine: `cannot express literal (tuple/list)` (E3, open).

## Items

Each item names the program, the wrong record, and the rule.

1. **Static tuple read as the bits of a double** (c275; it also fixed c156
   and c197). Program: `for x in (1, 5, 9): if x <= n: r += x` returned 0.
   Wrong record: the aggregate `Const` was stored as `double` (the
   `ssa.aggregate` default type), while its element reads were `load i64`.
   Rule: when every use is an element address, the constant is stored in
   its payload's kind (`i1`, `i64` or `double`), and the address stride
   uses the same type. A load whose type differs from its cell's type in
   kind (integer against double) is now a named refusal rather than a
   reinterpretation.
2. **Sort that did not sort** (c276; it also fixed c127, c129 and c134).
   Program: `xs = [5, n, 2, 9]; xs.sort()` gave 5029 in place of 259. Wrong
   record: the LLVM root frame's private cells were bare allocas. The sort's
   scratch length read garbage, its reserve refused, and the status went
   unread. Rule: as in the C wrapper's `memset`, root-owned frame storage
   starts zeroed.
3. **f-string glyph arena overrun** (c277, c62). Program:
   `len(f"value={int(x)}")` in a callee crashed with exit code -11. Wrong
   record: the compiler-frame storage that a function owns was sized from
   its SSA shape (one cell), while its literal glyphs wrote more than one
   cell. Rule: as in C's activation arrays, owned frame storage takes the
   solved storage requirement and starts zeroed.
4. **Loop literals entering int64 and bool Phis** (c278, c279; it also
   fixed c27, c61, c70, c137, c159, c162, c176 and c179). The resident
   `sum`/`any`/`count`/`index`/`remove` loops seeded `int64` index Phis
   with `int` literals, and a `bool` accumulator with an `int` False. LLVM
   refused all of these. Rule: a literal that enters a Phi takes the Phi's
   dtype (`constant_value(..., dtype=)`). The remove shift header now emits
   its Phi before `length - 1`, because LLVM requires Phis to lead a block.
5. **Flat reductions** (c278; it also fixed c150, c152, c153, c154, c165
   and c196). `sum(xs)`, `max(xs)`, `min`, `any` and `all` over a sequence
   had no LLVM emission. Rule, as in the C lane: reduce left to right over
   the static extent or over the sequence's own live length cell. `max` and
   `min` of no rows raise error 74. A NaN row propagates through `max` and
   `min`, and a NaN counts as truthy.

## Still refused in LLVM (19)

These are loud, and are not wrong:

- `address value ... has no concorded span` in a callee that reads a
  caller's list: c188, c189, c190, c192, c193.
- `aggregate projection 3 ... has no compile-time member` on optional
  calls: c31, c32, c33.
- `phi ... i64 disagrees with incoming double`, together with the new
  int-against-double load refusal, in concat and zip programs: c166, c171,
  c172, c173, c178, c210, c218, c273.
- `integer scalar operation has no LLVM emission for i64`: c163, c212.
- A heap load that disagrees with its element layout: c146.

## Gate receipts (at `ddcf059f`, on a clean worktree)

- Corpus: C 176/176 PASS. LLVM: 157 PASS, 0 WRONG, 0 RUN-CRASH, 0 DISAGREE.
- Fuzz: `FUZZ_EXT=3 FUZZ_BACKENDS=llvm` seed 40 ran 40 programs, with 0
  WRONG and 0 TYPE. 30 passed. 9 were LLVM-LOUD, in the refusal families
  listed above plus `Abs` on i64. 1 was LOUD in C: p40_18, "no source
  cell" in a fork copy, the same family as p34_21, which is loud on main.
- Gate: aa 89 / ab 7 / ac 26 / rec 20. These are the same names as the
  parent `74d79ec7`, both run in this container.

## Fortran, executed (2026-10-23, gfortran 14 in this container)

The first run that actually executed Fortran gave 69 PASS, 20 WRONG and
87 refusals out of 176. Every Fortran program before that had only been
emitted. The items below were fixed in the Fortran writer:

6. **Region outputs named by the caller** (c18, c03, c06, c37, c38, c42,
   c121-c123). Program: `while b != 0: a, b = b, a % b` returned `a`
   unchanged. Wrong record: a planned region with no return metadata
   took its output names from its callers' projection loads. The last
   call site won, so the region's body computed `t3` while publishing
   `t955`, and the output was never written. Rule: a function's own single
   `Ret` is its output list. The caller catalogue is used only when the
   function has no single `Ret`.
7. **Floor division** (c04, c15, c24). Fortran's integer `/` truncates
   toward zero. Rule: compute `(a - modulo(a, b)) / b`. This is exact,
   and `modulo` takes the sign of b, as Python's `%` does.
8. **Sequential Phi copies** (c124-c126). Program: `for i in range(10):
   if i * i > n: break` then `return i` returned 10 instead of 9. Wrong
   record: the copies on one edge ran in declaration order, so a later copy
   read a value an earlier copy had already advanced. Rule: the copies on
   an edge are parallel. Each target is written only after every copy that
   reads it, and a copy cycle is refused by name.
9. **Callee sequences not published** (c217). This is the same record as
   C before c217 and LLVM before c271: the runtime bindings and sequence
   tables named only the entry's own sequences. Rule: publish the linked
   descriptors as well. `_linked_frame_sequences` now also respells their
   length and status cells in entry ids.
10. **Static tuples** (c275, c156). Rule, as in LLVM: when every use is an
    element address, the tuple is a rank-1 array of its payload's own
    kind.
11. **Concat arena sized from an operand** (c264, c166, c173, c210, c218,
    c262, c263, c273). This only became visible after #788 typed concat
    rows. Program: `xs = [1, 2] + [n, 8]; xs[2]`. Wrong record: the concat
    column kept the static shape (2,) of its operand. The second
    `append_slice` then reported exhausted into a status nothing read, and
    the read went past the arena. Rule: a sequence column takes the larger
    shape from the storage requirement.

## Census at the PR head `ba464c22` (195 programs)

| | pass | wrong | crash | refused |
|---|---|---|---|---|
| C | 195 | 0 | 0 | 0 |
| LLVM | 169 | 0 | 0 | 26 |
| Fortran | 103 | 1 (c22) | 0 | 89 + 2 harness |

- **c22:** Fortran's `sin` and `cos` use the lane's lookup-table
  approximation (`trig_solver="lut"`), with an error of about 5e-8. The
  check reports this. It does not tolerate it.
- **c188, c197:** the Fortran harness stops with a dynamic extent of an
  internal array that it cannot resolve (RUN-FAIL, loud).
- **Fortran refusals:**
  - 66 EMIT-INCOMPLETE. Most of these are `Alloca`: heap growth has no
    Fortran emission yet.
  - 21 COMPILE-FAIL, mostly rank mismatches in reductions and pops.
  - 2 EMIT-FAIL.

After `ba464c22`, the LLVM `Abs` on i64 (commit `3884c95b`) moved c163
and c212 from refused to PASS, so LLVM is 171 PASS and 24 refused.

## Gate receipts (at `3884c95b`, on a clean worktree)

- Gate: aa 89 / ab 7 / ac 26 / rec 20. The failure names in each part
  match main `4567cb65` exactly, both run in this container.
- Fuzz: `FUZZ_EXT=3 FUZZ_BACKENDS=llvm` seed 42 ran 40 programs. 26
  passed, 10 were LLVM-LOUD (refusal families above), 3 were LOUD in C,
  and 0 were TYPE.
- **1 WRONG, in C, already on main:** p42_33. It is not from this branch.
  It reproduces on main `d8d41cb6`. Reduced:

  ```python
  def swap(xs: list[int]) -> None:
      t = xs[0]
      xs[0] = xs[1]
      xs[1] = t

  def root(n: int) -> int:
      xs0 = [i0 * 4 for i0 in range(n % 5 + 1) if i0 % 2 == 0]
      xs1 = [n, 5]
      swap(xs1)
      c2 = sum(i2 * n for i2 in range(4))
      return (len(xs0) * 100 + sum(xs0)) + (xs1[0] * 10 + xs1[1]) + (c2)
  ```

  C returns the list unswapped (n=0 gives 105, want 150). Without the
  comprehension, or without the `sum`, it passes. It was handed to the
  silent wrong answers lane.

## E4 in part: slice concatenation is int64 (LLVM)

- **Program:** `xs = [1, 2] + [n, 8]; return xs[2]` (c264), and nine
  other concatenations (c166, c173, c210, c218, c262, c263, c273, c281,
  c282).
- **Wrong record:** `lower_sequence_append_slice` minted its bounds,
  lengths and indices as `int`. Scalar settlement widened each `Add` to
  int64 and left the Phis at `int`, so LLVM refused an i32 Phi with i64
  incoming. The helper also loaded the int64 length cells as `int`.
- **Rule:** slice arithmetic has the width of the length cell, int64, like
  `insert_at`. Its status stays `int`. The caller's bounds and its
  absent-bound constants are int64 too.

## E5: list slices (C, silent on main)

These were found while writing a corpus program for the item above. On
main `d8d41cb6`, each of c290-c294 gives a wrong answer in C, with no
refusal.

1. **Folded slice.**
   - Program: `xs = [3, 1, 4, 1, 5]; ys = xs[:3]; return ys[n % 3]`.
   - Wrong record: structural specialization folded `xs[:3]` to the
     Constant `[3, 1, 4]`, with no parents and no leaves. Nothing wrote its
     arena, so the reset left length 0 and every row read gave 0.
     `xs[-4:-1] + [n, 9]` copied zero rows from it.
   - Rule: a folded slice's rows are its literal. It is initialized like a
     compile-time table (`literal_rows=`, under the duplicates policy).
2. **Runtime slice.**
   - Program: `xs = [3, n, 4]; ys = xs[:2]; return ys[1]`.
   - Wrong record: the slice declared its own arena, but no operation
     filled it (0.0 for n).
   - Rule: `dst = src[lower:upper]` is the sequence mutation `slice`.
     It resets `dst`, then appends the clipped rows of `src` with the
     existing `append_slice` helper. Its capacity is the snapshot bound:
     the source's bound plus its own straight-line appends. A stepped slice
     is not admitted, and stays on its old path.
3. **`sum` of a folded list.**
   - Program: `xs = [3, 1, 4, 7]; return sum(xs) + n`.
   - Wrong record: the list's Const result settled to int64 `(4,)`. The
     `sum` operand is another occurrence of the same identity, and it kept
     the provisional `()`, so C reduced one element (7 for 19). The
     concordance audit reported it as a finding.
   - Rule: a published type belongs to the identity. Every occurrence in
     the function that carried the previous type takes the new one
     (`publish_ssa_value_type`).

The corpus adds c290 to c294. The fuzzer's EXTENDED3 pool adds four
slice snippets and a sum of a folded list.

### Receipts for E4 (in part) and E5

- Census before the rebase onto main (195 programs, without c290-c294):
  C 195/195. LLVM 181 PASS, 0 WRONG, 14 refused (it was 171 and 24).
  Fortran unchanged: 103 PASS, 1 WRONG (c22).
- c290-c294: WRONG in C at main `d8d41cb6`, and PASS in C and LLVM here.
- Gate at `f0c046b8` (rebased onto main `08ba0171`): aa 89 / ab 7 /
  ac 26 / rec 20. The failure names in every part are the same as main's,
  with both run in this container.
- Fuzz, `FUZZ_EXT=3 FUZZ_BACKENDS=llvm`, seed 43 (40 programs, with the
  new slice snippets): 30 PASS, 6 LLVM-LOUD, 4 LOUD, 0 WRONG and 0 TYPE.
  All four LOUD programs are also LOUD on main.
- Census at the rebased head `3668d968` (210 programs): C 210/210, LLVM
  196 PASS, 0 WRONG, 0 crashes and 14 refused.

## E6: loop targets over a comprehension or `zip` (C silent on main, LLVM refused)

- **Program:** `for y in [x * n for x in xs]: t += y` (c295, c265) and
  `for p, q in zip(xs, ys): t += p * 3 + q` (c296, c171, c172).
- **Wrong record:** the planner's scalar proof table
  (`glsl_deployment_strategy._planned_scalar_dtypes`) never typed these
  loop targets, so they fell to the float64 domain default.
  - An unrolled comprehension (`sequence_materialization`) holds its
    elements as `arg0`..`argN`, not `elts`, so its element type was never
    joined.
  - A tuple target has no single name.
  - C then added int rows through float64. With n = 2**55 + 1, c295 returns
    216172782113783808 where 216172782113783814 is right. LLVM refused the
    i64/double Phi.
- **Why the corpus did not catch it:** the typed check compared integer
  answers with a relative 1e-12 tolerance, and that tolerance hid the
  rounding. An integer answer is now compared exactly
  (`tools/native_corpus_typed_check.py`).
- **Rule:** an unrolled comprehension's members are its `arg` parents. The
  i-th name of a tuple target over `zip(a, b, ...)` is an element of the
  i-th argument.
- c295 and c296 are WRONG at main `08ba0171` and pass in C and LLVM here.
  c265, c171 and c172 move from LLVM refused to PASS.

### Receipts for E6

- Census at `ca7c606c` (212 programs): C 212/212. LLVM 201 PASS, 0 WRONG,
  11 refused. Fortran (static runtime link) 111 PASS, 1 WRONG (c22).
- Fuzz, `FUZZ_EXT=3 FUZZ_BACKENDS=llvm`, seed 44 (40 programs, with the
  comprehension and `zip` snippets): 33 PASS, 3 LLVM-LOUD, 3 LOUD,
  1 CRASH, 0 WRONG and 0 TYPE. p44_13 (CRASH, an LLVM emission-artifact
  cell) and the LOUD and LLVM-LOUD programs behave the same on main, except
  p44_30: it is a C WRONG on main (305 for 302) and an LLVM refusal here.
- The gate for E6 is the E7 gate below (`d43227df` contains E6).

## E7: Fortran reductions over a resident list (COMPILE-FAIL on main)

- **Program:** `xs = [n, 7, -2, n * 2]; max(xs) * 100 + min(xs)` (c150),
  float `sum`/`max`/`min` (c154), `sum(xs, n)` after an append (c297,
  c298), `any(xs)`/`all(xs)` (c299), and `any([x > n for x in xs])`
  (c240).
- **Wrong record:**
  - The arena operand of a reduction over a resident sequence has a
    shapeless SSA occurrence. The Fortran emitter took the
    scalar-identity path and wrote `t9 = t6`, which gfortran refuses
    (ranks 0 and 1).
  - `any`/`all` with a float64 result passed the mask through `_numeric`,
    so ANY/ALL got a REAL argument, which gfortran refuses.
- **Rule:** `max`/`min`/`sum`/`any`/`all` over a resident sequence of
  scalar rows reduce over its live length, read from the length cell
  (the formal whose `sequence_length_for` names the arena), left to right
  as Python and the C emitter do. `max`/`min` of no rows stop with error
  74 (the C runtime's code) and never read a row. A NaN row propagates.
  `any`/`all` take a mask (a number's truth is `/= 0`), and their LOGICAL
  result is converted where it is assigned.
- Every one of these was loud on main (COMPILE-FAIL or EMIT-INCOMPLETE),
  never wrong.

### Linking the Fortran runtime in a Linux container

The Fortran artifact links libgfortran statically by default. A Linux
libgfortran.a is not position independent, so a library that needs the
runtime (any `error stop`) cannot be linked into a shared object here. That
showed up as COMPILE-FAIL for list programs that are otherwise right (c135,
c137, c138, c176, c266, c267, c268, c278, c279). The typed check now takes
`TYPED_CHECK_FORTRAN_STANDALONE=0` and the fuzzer `FUZZ_FORTRAN_STANDALONE=0`
to link it dynamically. The emitted program is the same, and the default is
unchanged (static, which is what Windows uses).

### Receipts for E7

- Census at `d43227df` (215 programs, `TYPED_CHECK_FORTRAN_STANDALONE=0`):
  C 215/215. LLVM 204 PASS, 0 WRONG, 11 refused. Fortran 126 PASS,
  1 WRONG (c22, the sine table), 4 RUN-FAIL (c142, c177, c188, c197: a
  dynamic extent of an internal array is asked of the caller), 5
  COMPILE-FAIL (c185, c187, c191, c192: a callee's list formal is passed
  as a scalar or assigned rank 0; c64: a dict value passed REAL to
  INTEGER), 3 EMIT-FAIL and 76 EMIT-INCOMPLETE.
- c297-c299 pass in C, LLVM and Fortran here. On main `08ba0171` c297 is
  EMIT-INCOMPLETE, c298 and c299 COMPILE-FAIL in Fortran.
- Fuzz, `FUZZ_EXT=3 FUZZ_BACKENDS=llvm,fortran FUZZ_FORTRAN_STANDALONE=0`,
  seed 45 (40 programs, with the reduction snippets): 15 PASS,
  20 FORTRAN-LOUD (18 `Alloca`, heap growth has no Fortran emission;
  2 `NoneValue`), 2 LLVM-LOUD, 3 LOUD, 0 WRONG and 0 TYPE.
- Gate at `d43227df` (covers E6 and E7), on a clean worktree: aa 89 /
  ab 7 / ac 26 / rec 20. The failure names in every part are the same as
  main `08ba0171`'s, with both run in this container.

## E8: a scalar span formal, and a load folded past a write (Fortran)

- **Program:** `a = outer(xs)` where `outer` returns `xs[0]` after a nested
  callee stores into `xs` (c185, c191, c192, c301), and a callee swap
  `t = xs[0]; xs[0] = xs[1]; xs[1] = t` (c187, c300).
- **Wrong record:**
  - A call result reaches a planned region as a formal with
    `program_abi_storage: span` and `program_abi_rank: 0`. The Fortran
    array-contract analysis (the emitter and `call_array_argument_ids`)
    took every span formal as an array, so the caller declared `t3(extent)`
    beside a callee that writes a scalar `t3`. gfortran refused it (rank
    mismatch).
  - Once that compiled, the swap was silently wrong: the read of `xs[0]`
    was a one-use `Load`, folded into its consumer, a call placed after the
    call that writes `xs[0]`. So `t` read the new row.
- **Rule:** a span formal with an explicit rank of zero is one cell, as the
  LLVM and C emitters already read it (`_declared_span_rank`). A span
  whose rank is unstated keeps the array reading. `call_array_argument_ids`
  takes a `span_formal_is_array` predicate, and only Fortran passes one, so
  C is unchanged. A load is not folded past a store or a call.

## E9: a `dict.get` default keeps its literal's type (C and LLVM silent on main)

- **Program:** `cells.get(5, 36028797018963969)` over int values (c302),
  and `cells.get(2, 9)` (c64).
- **Wrong record:** `fortran_c_shell` minted every lookup default as
  `float(literal)`. The int64 Const therefore carried `36028797018963968.0`,
  and C and LLVM returned 36028797018963990 where 36028797018963991 is
  right. In Fortran the payload was REAL(8) passed to an INTEGER(8)
  formal, which gfortran refused (c64).
- **Rule:** the default keeps the authored literal's own type.
- c302 is WRONG in C and LLVM at main `08ba0171` and passes here. c178's
  LLVM refusal (an i64 Phi with a double incoming) is not this record and
  is still open.

## E10: a callee list whose sequence id collides with the entry's (Fortran silent on main)

- **Program:** `grow(x)` builds `xs = [7, 5]; xs.append(x)` and `root` has
  its own list. Both are sequence 3 (fuzz seed 46 p46_21, corpus c303).
- **Wrong record:** the Fortran runtime bindings skipped a linked callee
  sequence whose id collides with one of the entry's own. Its hoisted
  capacity cell stayed 0, every append reported "full" into a status
  nobody reads, and the callee summed zeros. c303 returns 2 for 39 at
  main.
- **Rule:** every linked sequence's capacity is published. A binding
  carries its own column shapes, so the runtime never looks its descriptor
  up by a sequence id that two functions share. The sequence-table
  metadata is unchanged (it still lists one descriptor per id).
- **Hazard, not fixed here:** a "full" append status is written and never
  read, in every backend. A capacity that is wrong for any other reason is
  silent too.

### Receipts for E8-E10

- Census at `d0801217` (E8, 217 programs, dynamic Fortran runtime): C
  217/217, LLVM 206 PASS / 0 WRONG / 11 refused, Fortran 133 PASS.
- Fuzz seed 46 (`llvm,fortran`) at `d0801217` found p46_21, the E10
  record (Fortran WRONG; FORTRAN-LOUD at main only because another term
  of the same program did not compile there).
- Census at `a005af88` (219 programs, dynamic Fortran runtime): C
  219/219. LLVM 208 PASS, 0 WRONG, 11 refused. Fortran 135 PASS, 1 WRONG
  (c22), 3 RUN-FAIL (c142, c177, c197: a dynamic extent of an internal
  array is asked of the caller), 0 COMPILE-FAIL, 3 EMIT-FAIL and 77
  EMIT-INCOMPLETE (c302 is a new `Alloca` refusal).
- Fuzz, `FUZZ_EXT=3 FUZZ_BACKENDS=llvm,fortran FUZZ_FORTRAN_STANDALONE=0`,
  seed 47 at `a005af88`: 14 PASS, 23 FORTRAN-LOUD, 3 LLVM-LOUD, 0 WRONG
  and 0 TYPE.
- Gate at `a005af88`, on a clean worktree: aa 89 / ab 8 / ac 26 / rec 21.
  aa and ac have the same failure names as main `08ba0171`. ab and rec
  each had one extra failure,
  `test_ir_sequence_tables.py::test_sequence_return_surface_correlates_output_with_storage_descriptor`,
  which pins the exact runtime-binding mapping. A binding now carries
  `column_shapes` (E10), so the expectation adds `"column_shapes": [[]]`.
  With that, the test passes; every other name matches main.

## E11: a dict that accumulates into itself through `get` (C silent on main, LLVM refused)

- **Program:** `d = {}; for i in range(n): d[i % 3] = d.get(i % 3, 0) +
  36028797018963969`, then a membership loop sums `d[k]` (c304, c178).
- **Wrong record:** the planner's scalar proof table types a collection's
  rows as the join of its writers. Here the writer reads `d.get`, and
  `d.get` reads the rows: a cycle that the pessimistic fixpoint never
  closes. So the rows fell to the float64 default, C added 2**55 + 1
  through float64 (c304 returns ...968 for ...969), and LLVM refused the
  i64/double Phi (c178). `get` had no rule at all, and a loop that carries
  a store yielded no element type.
- **Rule:**
  - `d.get(k, default)` is the join of `d`'s rows and the default.
  - A loop whose carried value is a store yields the stored collection's
    rows.
  - For a dict whose rows are unproven and that is read by `get` with a
    typed default, the planner assumes the default's type. It accepts the
    assumption only if, under it, the dict's writers join to exactly that
    type (a consistent typing of the cycle). Otherwise it assumes nothing.
- c304 is WRONG in C at main `08ba0171` and passes in C and LLVM here.
  c178 moves from LLVM refused to PASS.
- **Found beside it, not fixed here (handed to the silent lane):**
  `d[i % 2] = d.get(i % 2, 0) + 5` in a loop, then `return d[0] + len(d)`,
  returns 7 for 12 at n = 3, on main too. The audit reports that the
  post-loop `d[0]` read takes a value defined in the loop body.

### Receipts for E11

- Census at `b40bcf78` (220 programs, dynamic Fortran runtime): C
  220/220. LLVM 210 PASS, 0 WRONG, 10 refused. Fortran 135 PASS, 1 WRONG
  (c22), 3 RUN-FAIL, 0 COMPILE-FAIL.
- Fuzz, `FUZZ_EXT=3 FUZZ_BACKENDS=llvm,fortran FUZZ_FORTRAN_STANDALONE=0`,
  seed 48 (with the accumulating-dict snippet): 19 PASS, 15 FORTRAN-LOUD,
  3 LLVM-LOUD, 3 LOUD, 0 WRONG and 0 TYPE.
- Gate at `b40bcf78` (covers E5-E11), on a clean worktree: aa 89 / ab 7 /
  ac 26 / rec 20. The failure names in every part are the same as main
  `08ba0171`'s.

## E12: a callee's view of a caller list (LLVM refused)

- **Program:** `def size(xs): return len(xs) * 10 + xs[0]` called with a
  list the caller grew (c188, c305). The same record covered c189, c190,
  c192, c193 and c269.
- **Wrong record:** the callee loads its list's length cell, a rank-1 span
  formal, with no element address before the load. Only GEP results had a
  span address in the LLVM emitter, so the load was refused ("address
  value ... has no concorded span").
- **Rule:** a load from a span formal itself (rank 1 or more, not a heap
  pointer cell) reads its first element, the address a zero GEP gives.
- c305 is refused by LLVM at main `08ba0171` and passes here.

### Receipts for E12

- Census at `77784c10` (221 programs, dynamic Fortran runtime): C
  221/221. LLVM 217 PASS, 0 WRONG, 4 refused (c31, c32, c33: an optional
  call's aggregate projection; c146: a heap load layout). Fortran 135 PASS,
  1 WRONG (c22), 3 RUN-FAIL, 0 COMPILE-FAIL.
- Fuzz, `FUZZ_EXT=3 FUZZ_BACKENDS=llvm,fortran FUZZ_FORTRAN_STANDALONE=0`,
  seed 49: 12 PASS, 26 FORTRAN-LOUD, 2 LLVM-LOUD, 0 WRONG and 0 TYPE.

## E13: an optional tuple result's presence projections (LLVM refused)

- **Program:** `fork = fork_of(row) if flag > 0 else None` with `fork_of`
  returning `tuple[int, int] | None`, then `origin, stamp = fork` (c31,
  c32, c33, c306).
- **Wrong record:** `ssa_optional_values` interleaves each payload with
  its presence flag (four outputs for a pair) and rewrote the call's
  `output_ids`. The call's `output_positions` and `output_slots` still
  named the two payload slots, so the LLVM emitter knew aggregate members
  0 and 1 only and refused the presence projections ("aggregate
  projection 3 ... has no compile-time member").
- **Rule:** an interleaved optional result restates its positions and
  slots with its outputs. Both are the callee's result order.
- c306 is refused by LLVM at main `08ba0171` and passes here. c31, c32 and
  c33 move from refused to PASS. C reads the same attributes and is
  unchanged in the census.

### Receipts for E13

- Census at `b27b3c3f` (222 programs, dynamic Fortran runtime): C
  222/222. **LLVM 221 PASS, 0 WRONG, 1 refused** (c146, a heap load whose
  result disagrees with its element layout). Fortran 135 PASS, 1 WRONG
  (c22), 3 RUN-FAIL, 0 COMPILE-FAIL.
- Fuzz, `FUZZ_EXT=3 FUZZ_BACKENDS=llvm,fortran FUZZ_FORTRAN_STANDALONE=0`,
  seed 51 (with the optional-pair snippet): 17 PASS, 22 FORTRAN-LOUD,
  1 LOUD, 0 LLVM-LOUD, 0 WRONG and 0 TYPE.

## E14: a loop target read from an owned heap arena (LLVM refused)

- **Program:** `xs = []; for i in range(n): xs.append(i); if i % 2 == 1:
  xs.pop(0)`, then `for x in xs: total = total * 10 + x` (c146; c307 is
  a companion that also passes at main).
- **Wrong record:** the loop target's Load from the owned heap arena kept
  a provisional declared `int` (i32). Scalar settlement keeps a Load's
  declared width as a memory read (the rule for i32 status cells), though
  the arena's rows are int64. LLVM refused the load ("heap load result
  disagrees with its declared element layout").
- **Rule:** a Load through an element address of a heap arena reads the
  arena's declared element dtype (`heap_element_dtype`). It is not a
  declared-width load.
- c146 is refused by LLVM at main `08ba0171` and passes here. **With E12
  to E14, LLVM refuses no corpus program.**

### Receipts for E14

- Census at `c8452ec5` (223 programs, dynamic Fortran runtime): C
  223/223. **LLVM 223/223 PASS**, 0 WRONG and 0 refused. Fortran 135 PASS,
  1 WRONG (c22), 3 RUN-FAIL, 0 COMPILE-FAIL, 3 EMIT-FAIL and 81
  EMIT-INCOMPLETE.
- Fuzz, `FUZZ_EXT=3 FUZZ_BACKENDS=llvm,fortran FUZZ_FORTRAN_STANDALONE=0`,
  seed 52 (with the pop-front snippet): 19 PASS, 19 FORTRAN-LOUD, 2 LOUD,
  0 LLVM-LOUD, 0 WRONG and 0 TYPE.
- Gate at `c8452ec5` (covers E5-E14), on a clean worktree: aa 89 / ab 7 /
  ac 26 / rec 20. The failure names in every part are the same as main
  `08ba0171`'s.

## E15: a callee that returns an optional value (Fortran refused)

- **Program:** `def maybe_twice(x, y): return x * 2 if x != y else None`,
  then `o = maybe_twice(n, 2); if o is not None: t = o - 2` (c308).
- **Wrong record:** the callee's `None` arm is a `NoneValue` that nothing
  reads once the optional is interleaved, and the callee's single `Ret`
  carries payload and presence while its declared outputs named only the
  payload. Fortran refused the `NoneValue` ("no Fortran intrinsic or
  expression is registered") and would have dropped the presence output.
- **Rule:** an unread `NoneValue` that is no output emits nothing. When a
  function's one `Ret` extends its declared outputs and contains all of
  them, the `Ret` operands are its outputs.
- c308 is refused by Fortran at main and passes here.

## E16: a populated list literal (Fortran refused)

- **Program:** `sum([4, 6, 1]) + n`, `max([2.5, x, -1.0])` (c309, c293).
- **Wrong record:** a `Const` whose payload is a flat list and whose
  declared shape is its own length reached the scalar literal path and
  was refused ("cannot express literal [4, 6, 1]").
- **Rule:** such a `Const` is a rank-1 array constant, written element by
  element in the declared kind. Any other shape still refuses.
- c293 and c309 now pass in Fortran. c271 and c272 move on to the Alloca
  refusal. At main `2806360d`, C is **silently wrong** on c309 (`sum` of
  the literal returns its first element, 604.5 for 1304.5); the folded-list
  rule from earlier in this PR (c293) already fixes that here.

## E17: pop(i) and del xs[i] on a fixed list (Fortran failed to run)

- **Program:** `xs = [5, n, 2]; xs.pop(0)` (c142), `del xs[i]` (c177),
  and c350.
- **Wrong record:** `delete_at`, `insert_at` and `reverse` listed every
  storage formal as an array argument, including the scalar capacity
  cell. Fortran gave the caller's capacity a shape-dynamic extent that
  nothing supplies, and the run stopped with "shape-dynamic Fortran
  extents require explicit positive extent_overrides".
- **Rule:** a positional sequence helper's array arguments are its storage
  formals except the capacity cell (as `allocate` already did), in
  `ir_sequence_tables._storage_array_ids`.
- c142, c177 and c350 pass in Fortran. c197 still fails to run: the
  callee's bool list carries live flags that the caller's list lacks
  (open).

### Receipts for E15-E17

- Census at `86b8314d` (226 programs, dynamic Fortran runtime): C
  226/226. LLVM 226/226 PASS. **Fortran 156 PASS** (135 at `c8452ec5`),
  1 WRONG (c22, the `sin` table), 1 RUN-FAIL (c197), 0 COMPILE-FAIL, 0
  EMIT-FAIL and 68 EMIT-INCOMPLETE (Alloca heap growth).
- Fuzz, `FUZZ_EXT=3 FUZZ_BACKENDS=llvm,fortran FUZZ_FORTRAN_STANDALONE=0`,
  seed 53 (with the optional-scalar, list-literal and pop/del snippets): 25
  PASS, 14 FORTRAN-LOUD, 1 LLVM-LOUD, 0 WRONG and 0 TYPE. The LLVM-LOUD is
  p53_17 (`enumerate` over a list beside a callee loop: "phi physical
  element i64 disagrees with incoming"); it is loud at main `2806360d` too
  (open).
- Gate at `86b8314d` (covers E5-E17), on a clean worktree: aa 89 / ab 7 /
  ac 26 / rec 20. The failure names in every part are the same as main
  `08ba0171`'s (main has since moved to `2806360d`, a docs-only change).

## E18: enumerate loops (C silently wrong; LLVM refused)

- **Program:** `xs = [5, n + 36028797018963969, 6]; for k, x in
  enumerate(xs): t += k * x` (c351). No corpus program used `enumerate`
  before this one.
- **Wrong record:** as with `zip` (E6), the tuple target's names were never
  typed by the planner's scalar proof table, so `x` fell to the float64
  default. At main `2806360d`, C multiplies the large int row through
  float64 and returns 36028797018963984 for 36028797018963983. LLVM refused
  the i64/double Phi (fuzz p53_17).
- **Rule:** over `enumerate(a[, start])` the first name is an int64 count
  and the second is an element of `a`
  (`glsl_deployment_strategy._planned_scalar_dtypes_from`).
- **Second wrong record, same loop:** `enumerate(xs, 3)` bound the count to
  the loop induction, which starts at 0, and dropped `start`. C and LLVM
  both counted from 0 silently. A start has no loop lowering, so
  `loop_composer` now **refuses it by name** ("enumerate with a start
  argument has no loop lowering"). Test:
  `test_loop_composer.py::test_enumerate_start_refuses_rather_than_counting_from_zero`.
  Lowering the offset is open work.
- `[k * x for k, x in enumerate(xs)]` was refused (a resident `sum` over
  float64 rows); E19 fixes it.

### Receipts for E18 (before the rebase onto main `230369c5`)

- Census at `892bdcce` (227 programs): C 227/227, LLVM 227/227, Fortran 157
  PASS, 1 WRONG (c22), 1 RUN-FAIL (c197), 68 EMIT-INCOMPLETE.
- Fuzz seed 54 (`llvm,fortran`, with the enumerate snippet): 24 PASS, 15
  FORTRAN-LOUD, 1 LOUD, 0 LLVM-LOUD, 0 WRONG and 0 TYPE.
- Gate at `892bdcce`, on a clean worktree: aa 89 / ab 7 / ac 26 / rec 20
  (ac and rec each gained the new passing test). Failure names in every
  part are the same as main `08ba0171`'s.

## E19: a tuple comprehension target over enumerate or zip (refused)

- **Program:** `sum([k * x for k, x in enumerate(xs)])` (c352) and
  `sum([p * q for p, q in zip(xs, ys)])` (c353), with a large int row.
- **Wrong record:** the comprehension rule typed only a single-name target,
  so a tuple target's names fell to the float64 default and the resident
  `sum` refused float64 rows ("Python's sum has one result type only for
  integer rows"). Loud at main `2806360d`.
- **Rule:** a tuple comprehension target takes the `for` rule: over `zip`,
  the i-th name is an element of the i-th argument; over `enumerate(a)`,
  an int64 count and an element of `a`.
- c352 and c353 pass in C and LLVM. Fortran refuses both on `Alloca` (the
  comprehension's result grows on the heap).

### Receipts for E19 (before the rebase onto main `230369c5`)

- Census at `3e664c6a` (229 programs): C 229/229, LLVM 229/229, Fortran 157
  PASS, 1 WRONG (c22), 1 RUN-FAIL (c197), 70 EMIT-INCOMPLETE.
- Fuzz seed 55 (`llvm,fortran`): 22 PASS, 15 FORTRAN-LOUD, 2 LOUD, 1 SEGV,
  0 WRONG and 0 TYPE. The SEGV (p55_15) is in C and reproduced at main `2806360d`: `len(xs)` was read before
  callee pushes beside a comprehension loop. The silent lane has since
  fixed it on main (its c332).
- Gate at `3e664c6a`, on a clean worktree: aa 89 / ab 7 / ac 26 / rec 20,
  the same failure names as main `08ba0171` in every part.

### Receipts for E18 and E19 on main `230369c5`

- Census at `09401a20` (258 programs, dynamic Fortran runtime): C 258/258,
  LLVM 258/258, Fortran 173 PASS, 1 WRONG (c22), 1 RUN-FAIL (c197), 83
  EMIT-INCOMPLETE (heap growth).
- Fuzz seed 56 (`FUZZ_EXT=3`, `llvm,fortran`, 30 programs): 13 PASS, 16
  FORTRAN-LOUD, 1 LOUD, 0 LLVM-LOUD, 0 WRONG and 0 TYPE.
- Gate at `09401a20` and at main `230369c5`, each on a clean worktree:
  aa 89 / ab 8 / ac 26 / rec 21 on both. The failure names in every part
  are the same; ac and rec each gain the new passing test.

## E20: enumerate counts from its start

- **Program:** `for k, x in enumerate(xs, 3): t += k * x` (c354), a
  parameter start (c355), `start=` by keyword (c356), and a comprehension
  over `enumerate(xs, 2)` (c357).
- **Wrong record:** the counter target was bound to the loop induction,
  which starts at 0, and `start` was dropped. C and LLVM counted from 0
  silently at main `2806360d`; since #805 it was refused by name.
- **Rule:** `loop_composer` records the start as an induction offset,
  `ControlProgram.induction_offsets` (counter target, start value). It is
  carried through control composition, projection and the hierarchical
  merge, and counted as a control dependency. The counter is the induction
  plus the start (`precompile_to_ssa`, binding `enumerate_start`). A start
  given in any other way still refuses by name. Test:
  `test_loop_composer.py::test_enumerate_start_offsets_the_counter`.

### Receipts for E20

- Census at `eeb508be` on main `4a0f5452` (262 programs): C 262/262, LLVM
  262/262, Fortran 176 PASS, 1 WRONG (c22), 1 RUN-FAIL (c197), 84
  EMIT-INCOMPLETE.
- Fuzz seed 57 (`FUZZ_EXT=3`, `llvm,fortran`, 30 programs, with the
  enumerate-start snippet): 17 PASS, 13 FORTRAN-LOUD, 0 WRONG and 0 TYPE.
- Gate at `eeb508be` and at main `4a0f5452`, each on a clean worktree:
  aa 89 / ab 8 / ac 26 / rec 21 on both, with the same failure names in
  every part.

## E21: a live-flag arena is bounded by its column's rows

- **Program:** `count_true([True, False, True])` where the callee loops
  `for b in bs: if b: c += 1` (c197), `positives([4, -2, 7, -9, 1])` with
  `if x > 0` in its loop (c358), and two flag literals into one callee
  (c359).
- **Wrong record:** the callee's list formal keeps live flags. Storage
  requirements bounded the flags only from the static and comprehension
  capacity bounds, never from the column, which was bounded to the
  literal's rows. So the flag formal, and the caller actual minted for it,
  had no element count. The Fortran entry then asked for
  `extent_dynamic_..._1`, an extent no feed can supply, and each run
  refused (RUN-FAIL on all three programs at main `a67c8a9c`). C and LLVM
  were not affected.
- **Rule:** a live-flag arena holds one flag per row of its sequence, so
  each bound on the first column's rows bounds it
  (`ssa_storage_requirements.module_storage_requirements`). The call
  binding then carries the bound to the caller's actual.

### Receipts for E21

- Census at `9e50732a` on main `a67c8a9c` (273 programs): C 271/273, LLVM
  269/271, Fortran 184 PASS (c197, c358 and c359 included), 1 WRONG (c22,
  by design), 0 RUN-FAIL, 86 EMIT-INCOMPLETE. The C and LLVM misses are
  c334 and c335 (refused at lowering) and c337 and c338 (LLVM refuses a
  phi type). Each does the same at main `a67c8a9c`.
- Fuzz seed 58 (`FUZZ_EXT=3`, `llvm,fortran`, 30 programs, with the
  `positives` snippet in two of them): 15 PASS, 14 FORTRAN-LOUD (Alloca),
  1 LOUD, 0 WRONG and 0 TYPE.
- Gate at `9e50732a` and at main `a67c8a9c`, each on a clean worktree:
  aa 89 / ab 8 / ac 26 / rec 21 on both, with the same failure names in
  every part.

## E22: Fortran emits owned-heap lists

- **Program:** any list grown at run time: `xs = []` then `xs.append(...)`
  in a loop (c400, c402), `[0.5] * (n % 4)` then appends (c401, c201,
  c202), `sorted(xs)` on a grown list (c403, c127), and a list a callee
  returns (c56, c60, c215, c216).
- **Wrong record:** the Fortran writer had no emission for `Alloca
  [allocate]`, so all 86 such programs at main stopped at EMIT-INCOMPLETE.
  C and LLVM were not affected.
- **Rule:** a heap cell is a `type(c_ptr)`. Its elements are read and
  written through a `c_f_pointer` view, refreshed at entry, after an
  allocation, after a store into the cell, and after any call that passes
  the cell. Allocation, `copy_elements` and `release_owner` go to module
  helpers that call libc `malloc`, `memmove` and `free` through bind(C)
  interfaces. An entry releases its heap owner formals at every Ret. A
  caller cell that a callee fills starts at `c_null_ptr`, as C's calloc
  does. The emitter still refuses loudly when a heap cell is indexed with
  no point at which its view can follow the arena.
- **Found on the way:**
  - A callee with two Rets never wrote its status output on the exhausted
    edge, so the caller read a stale status (c202 WRONG, c201 crash). Each
    Ret now writes every output it carries.
  - Element stores were coerced to the cell's dtype (`ptr`) instead of the
    element dtype.
  - Heap capacity cells were given a rank, so the entry asked for a
    dynamic extent (c127). Their ranks are zeroed module-wide.
  - `-9223372036854775808_c_int64_t` is too big a literal for gfortran
    (c57, c62). It is now `(-huge(0_c_int64_t) - 1_c_int64_t)`.
  - The harness bound a heap capacity from a callee binding whose sequence
    id collided with a root list (c193 crash). A ptr column parameter is
    now never bound.
- **Separate finding (C, not fixed here):** `ys = [0.5] * (n % 4)`, two
  appends, then `sum(ys)` fails to compile in C: the reduction reads the
  replication's static extent although two appends write the sequence
  (identity concordance names it). c401 sums with a loop instead.

### Receipts for E22

- Corpus at `ea3e85b5` on main `39fe7283` (303 programs): C 303/303, LLVM
  298/303, Fortran 302 PASS (up from 184 at E21), 1 WRONG (c22, by design),
  0 EMIT-INCOMPLETE. The LLVM misses are c337, c338, c367, c370 and c371
  (LLVM refuses a phi type); c367, c370 and c371 do the same at main
  `39fe7283`.
- c400-c403 are EMIT-INCOMPLETE (Alloca) on Fortran at main `39fe7283`.
- Fuzz seeds 59 to 62 (`FUZZ_EXT=3`, `llvm,fortran`, 30 programs each):
  116 PASS, 3 LOUD, 1 LLVM-LOUD, 0 WRONG and 0 TYPE. The new snippet was
  drawn once (p60_9), and that program is LOUD in C at main `39fe7283`
  too: a list grown in a loop and sorted, followed by `[6] * (n % 4 + 1)`,
  emits a call to `numerical_region_2`, a region C never defines
  (`call to unknown function`). Each half alone passes on all three
  backends. This is a separate C finding.
- Gate at `ea3e85b5` and at main `39fe7283`, each on a clean worktree:
  aa 89 / ab 8 / ac 26 / rec 21 on both, with the same failure names in
  every part.

## E23: dict items tuple targets are a key and a row

- **Program:** `for k, w in d.items(): s += k * 10 + w` over a dict
  holding a large int row (c404) or a large int key (c405), and c337
  (`del d0[1]` beside such a loop).
- **Wrong record:** the planner's scalar rule types the tuple target of
  `zip` and `enumerate` but had no rule for `items`, so both names fell
  to the float64 domain default. C summed int keys and rows through float64
  and returned rounded results (c404 and c405 WRONG at main `da63b7dd`).
  LLVM refused the i64/double phi. Fortran refused int64 actuals for REAL(8)
  formals, which surfaced as c337 COMPILE-FAIL after #827.
- **Rule:** over `d.items()` the first name is a key of `d`: the join of
  its key members and its stores' indices. The second name is a row of `d`
  (`glsl_deployment_strategy._planned_scalar_dtypes_from`, beside the zip
  and enumerate rules).
- **Still open (LLVM, loud):** after a keyed delete, LLVM refuses an i32/i64
  phi (c337, c338, c370, c371, c387, c388, c389, c430-c433). Each does the
  same at main.

### Receipts for E23

- Corpus at `086aac37` on main `da63b7dd` (325 programs): C 325/325, LLVM
  313/325, Fortran 324 PASS, 1 WRONG (c22, by design). The LLVM misses are
  c367 and the delete-family phi refusals above; each does the same at main.
- Fuzz seed 63 (`FUZZ_EXT=3`, `llvm,fortran`, 30 programs): 30 PASS,
  0 WRONG and 0 TYPE. The new snippet was not drawn; c404 is its program.
- Gate at `086aac37` and at main `da63b7dd`, each on a clean worktree: aa 89 /
  ab 7 / ac 26 / rec 20 on both, with the same failure names in every part.

## E24: a table delete counts rows with an int64 counter

- **Program:** `d = {1: n, 2: 3, 5: 7}; del d[2]; len(d)` (c338), and every
  corpus program with a keyed delete (c337, c370, c371, c387-c389,
  c430-c433, c406).
- **Wrong record:** `lower_table_delete` and `lower_table_delete_first`
  scanned rows with an `int` phi. The type linker settles the latch
  increment to the int64 length it counts against, so LLVM refused the
  i32/i64 phi on all twelve programs. C and Fortran were not affected.
- **Rule:** a row counter over an int64 length is int64, as in every other
  scan lowering (`ir_sequence_tables`).
- **Separate finding (LLVM, loud, same at main):** a root whose result folds
  to a constant and never reads `n` (fuzz p64_18:
  `xs0 = [3, n, 0]; return xs0[0] * 7 + xs0[2]`) fails LLVM emission with
  `emission_artifact: no source cell ... value_without_identity_cell`.

### Receipts for E24

- Corpus at `376f1f19` on main `11944336` (328 programs): C 328/328, LLVM
  327/328 (c367, same at main), Fortran 327 PASS, 1 WRONG (c22, by design).
- c406 is LLVM EMIT-INCOMPLETE (the phi) at main `11944336`.
- Fuzz seed 64 (`FUZZ_EXT=3`, `llvm,fortran`, 30 programs, the new snippet
  drawn once in p64_27): 29 PASS, 1 CRASH (p64_18, the separate finding
  above), 0 WRONG and 0 TYPE.
- Gate at `376f1f19` and at main `11944336`, each on a clean worktree: aa 89 /
  ab 7 / ac 26 / rec 20 on both, with the same failure names in every part.
