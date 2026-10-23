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
