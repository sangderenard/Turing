# 37. Lane E: backend parity (2026-10-23)

The typed corpus check now runs each program through LLVM (compiled and
executed) and Fortran (emitted; this container has no Fortran compiler)
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
