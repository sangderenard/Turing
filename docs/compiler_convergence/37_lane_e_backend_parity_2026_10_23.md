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
