# Strings as a managed token library: audit and plan

Started 2026-10-24 13:30 UTC by the planner thread, at Albert's request. Audit only: nothing here was run or tested. Line numbers refer to `main` `ca9b59b5`.

Albert's direction (2026-10-24 13:21): every product uses tokenized strings. Basic tokens compose token sequences, and each sequence is then interned as its own token, in a live, managed system.

## Status of this file

Complete: sections 1 to 6.

## 1. What exists today (audit)

Turing already has three string models. They do not yet form one managed library.

**A. The content token, `string_token(text)`** (`src/compiler/string_table.py`).
- A token is the FNV-1a 64 hash of the UTF-8 bytes, as a signed i64 (`ir_container_ops.fnv1a_64`). `NONE_TOKEN` is the hash of a reserved marker.
- Equal bytes give equal tokens everywhere, with no shared counter. A name hashed at run time from bytes (`wasm_sequence.emit_hash_delimited_prefix`, `ir_byte_idioms`, `STRING_SPLIT_PART_HASH`) meets the compile-time constant's token.
- Every backend emits a constant token as a 64-bit literal: C (`ssa_c_backend.py:868, 3904, 5445`), Fortran (`ssa_fortran_backend.py:859-861, 4290`), LLVM (`3021, 7077`), Wasm (`fused_program_wasm_backend.py:2168, 2774`), JS (`ssa_javascript_backend.py:690`), and the reference evaluator (`ssa_reference_evaluator.py:918`).
- Equality on tokens is an integer compare (`ir_string_interning.tokenize_ssa_string_constants` tags equal/not_equal).
- Dict keys typed `str` use this token (`fortran_c_shell.py:5831`, `"str": "string_token"`). `Literal[...]` str annotations become a `token_vocabulary` on an int64 scalar (`fortran_c_shell.py:5788`, checked in `extraction_contract.py:383-401`).
- A collision is possible in principle. Nothing detects one.

**B. The compile-time table, `StringTable`.**
- A token→text map, persisted at `repo-cache/string-table/strings.json` (ABI tag `turing.string-table.v1`). Every `intern` posts a `STRING_TABLE_ENTRY` concordance fact.
- On load, an entry is kept only if its token re-hashes from its text. That is the only collision guard, and it checks the hash, not uniqueness.
- It lives only in the compiler. `ir_string_ops` says the reverse lookup "goes through the StringTable recorded at compile time, never at run time".
- Constructed ad hoc in three places: `site_bundle.py:3455`, `opportunistic_pipeline.py:474`, and the callers of `intern_string_constants`. No product owns one table.

**C. Resident glyph sequences** (`glyph_sequence_format.py`, `resident_sequence_materialization.py`, `ir_sequence_tables.py`).
- A formatted or built string is a `list[int]` of per-character tokens, each a `string_token` of one character. This is what f-strings and repr lower to today: the "basic tokens" Albert describes.
- `_string_token_glyph_materializations` (`resident_sequence_materialization.py:363`) builds a keyed SSA table `(token, ordinal) -> glyph`. Ordinal -1 holds the glyph count. `lower_glyph_token_copy` (`ir_sequence_tables.py:3539`) expands a known token into its glyphs at run time. Unknown token: status 0. Full destination: status 2.
- `lower_sequence_equal` (`ir_sequence_tables.py:3225`) compares two resident sequences glyph by glyph. Its docstring rules a content hash out of the truth path: "a content hash ... is not evidence of equality".
- So the direction token→sequence exists, but only for tokens known at compile time. The direction sequence→token (interning a sequence built at run time) does not exist.

**D. Byte views** (`ir_string_ops.py`, header only).
- These are i64 fat pointers (ptr high 32 bits, length low 32 bits) with the operations `string_const/length/char/find/slice/concat/split_part`.
- Only the hashing idioms are lowered (Wasm `wasm_sequence.py`, plus the split-hash fold). `STRING_CONCAT` and the rest have no lowering. `intern_string_constants` leaves runtime building "for a real string builder".
- Albert rejected byte strings, so this model should shrink to an ingestion bridge: bytes in, token out.

**E. Edges that refuse today.**
- G1: runtime-built or runtime-selected `str` in 12 bootstrap units.
- A `str` callback argument (doc 41, "Not carried yet": a `str` needs (ptr, len); a dict needs a handle).
- c474x: `len()` of a selected `str`.
- J14 and C10: string frontends held for this decision.
- B5 in doc 34: `'|'.join(tokens)` has no glyph recipe.

## 2. Gaps and findings

1. **There is no runtime interner.** Nothing can turn a sequence built at run time into a token. Every refusal in 1.E has this root cause. The pieces that do exist:
   - the per-character tokens (glyphs);
   - the token→glyph table;
   - collision-free sequence equality;
   - a streaming FNV hash in Wasm.

2. **A collision is silently wrong in two places.**
   - `StringTable.intern` uses `self._entries.setdefault(token, text)`. A second text with the same token keeps the first text's spelling, and the concordance post records the new text against the same row. Nothing refuses.
   - At run time, `==` on two tokens is integer equality. Two different strings that hash equal compare equal.
   - FNV-1a 64 makes this unlikely, but it is not impossible. `lower_sequence_equal` already refuses to treat a hash as evidence. Under Albert's rules this is a silent family: rare, but wrong with a clean audit.
   - A cheap first fix: refuse at compile time when `intern` sees a different text for an existing token.

3. **What "composition" means is not pinned down.**
   - A per-glyph token is `fnv1a_64(char)`. A constant word's token is `fnv1a_64(whole word)`. Nothing defines a sequence token in terms of its members.
   - Suppose a run-time intern hashed the member tokens. Then a runtime-built `"ab"` would not equal the constant `"ab"`, or a dict key `"ab"`, or a name hashed from bytes.
   - The canonical form has to be the flattened glyph run, so `[a, b]`, `[ab]` and the constant `"ab"` are one token.

4. **No product owns a table.** The `StringTable` is built ad hoc in `site_bundle`, `opportunistic_pipeline` and the interning passes, and persisted in a repo cache. A product does not ship its vocabulary. It ships only the `(token, ordinal) -> glyph` table that a given unit asked for.

5. **The backends differ in width.**
   - JS uses BigInt (`ssa_javascript_backend.py:700`). Wasm and LLVM use i64. A Wasm f64 working type carries the token as reinterpreted bits.
   - GLSL and SPIR-V need int64 (`GL_ARB_gpu_shader_int64`) or a `uvec2`. Today nothing on a shader path carries a token.
   - Lane X's open row (a silent Wasm i32 wrap of ints at or above 2**31) applies to any token that passes through an i32 slot.

6. **Byte views are header only.** `string_concat`, `slice`, `find` and the rest have no backend lowering. Since bytes are rejected, these should not be built out. They become ingestion only: bytes in, glyph run or token out.

7. **Lifetime is undefined.** A live table grows. Nothing says who owns it, whether tokens die, or what happens between processes, such as host and kernel, or two products sharing a token over an ABI.

## 3. Proposed design (for Albert to confirm)

**Identity: a hybrid.** It answers the open question with both options.
- *The token stays content-addressed*: FNV-1a 64 over the UTF-8 spelling of the flattened glyph run. This is exactly today's `string_token`, so constants, dict keys, persisted `strings.json` files, extracted names and every backend literal keep working unchanged. Two processes or backends agree with no shared state.
- *The live table is the authority on equality.* Interning a run looks up its hash. A hit whose stored run differs is a collision. In v1 the collision traps loudly. v2 could add a probe-and-salt escape, but a salted token is no longer content-addressed, so that needs Albert's call.
- Under this design, comparing two interned tokens as integers is exact, because the table rejected the collision when the second one was interned.
- Ask Albert: is "trap on a 64-bit collision" acceptable for v1?

**Composition.** A token sequence is a list of tokens, and a member may itself be an interned sequence. Interning flattens to glyphs, hashes as a stream, and stores the run once. Composite members are kept as an optional rope later, to save space. They never decide identity.

**The library, one ABI for every product: `turing-token-library-v1`.**
- A handle `TL*`, the context pointer that callbacks already declare in doc 41, owns an arena with three parts:
  - glyph runs;
  - a token→(offset, length) open-addressing index;
  - a seed section.
- Operations, where everything that returns a string returns a token:
  - `tl_intern(TL*, const int64 *glyphs, int64 n) -> token` (traps on collision);
  - `tl_length(TL*, token) -> int64`;
  - `tl_glyph(TL*, token, i) -> glyph token`;
  - `tl_copy(TL*, token, int64 *dst, cap) -> status`, with the status codes of `lower_glyph_token_copy` (0 unknown, 1 done, 2 full);
  - `tl_concat(TL*, a, b) -> token`;
  - `tl_slice(TL*, t, start, stop) -> token`;
  - `tl_find(TL*, t, needle, from) -> int64`;
  - `tl_from_utf8(TL*, bytes, n) -> token` (ingestion);
  - `tl_to_utf8(TL*, token, dst, cap) -> n` (display, I/O, host).
- Equality is integer `==`. The hash is the glyph-run hash (`tl_intern` returns it), so it is never recomputed.

**The seed.** The compile-time `StringTable` becomes the product's seed section: every constant word and its glyph run, emitted as data. The `(token, ordinal) -> glyph` table that `_string_token_glyph_materializations` builds already is that seed, in SSA form. The plan reuses it; it adds no new machinery.

**Lifetime, v1.** Append only, one table per process, pinned until exit. That needs no refcounts and no GC. Tokens crossing a process or ABI boundary are valid there because they are content-addressed. The receiver's own table may have to intern the spelling, through `tl_from_utf8`, before it can read the glyphs. Epochs or regions are a later item.

**Per backend.**
- C, LLVM and Fortran: one C runtime file, `turing_token_library.c`, linked through the existing shell. Fortran binds it with `bind(C)`.
- Wasm: the same algorithm in-module over the bump heap that `wasm_container` already owns. A JS-host import is an optional fast path, never the meaning.
- JS: a `Map<bigint, Int32Array|BigInt64Array>` plus the same FNV, in the emitted prelude.
- GLSL and SPIR-V: read-only. A kernel may compare tokens and read seed glyphs. It may not intern. A kernel that would intern is refused, and the host interns before dispatch.
- The reference evaluator: a Python implementation of the same table, so the oracle agrees with the products.

## 4. Staged plan (rows for lanes_plan.md once workers resume)

Rows are ordered silent-capable first, as in the lanes plan. The corpus block **c610-c619 is reserved for TOK work**, so the next free block becomes c620.

| id | item | silent? | owner (proposed) | claimed functions | corpus |
|---|---|---|---|---|---|
| TOK0 | `StringTable.intern` refuses a second, different text for an existing token, and `_load` refuses a duplicate | **yes** (fixes a silent collision) | SSA interpreter thread or silent lane | `string_table.StringTable.intern`, `_load` | c610 |
| TOK1 | Specify `turing-token-library-v1` as doc 42: identity rule, flattening, the ABI table above, status codes, lifetime, the collision trap, and the seed format. No code. | no | planner | doc only | none |
| TOK2 | Reference implementation in Python (`src/compiler/token_library.py`), used by `ssa_reference_evaluator` for `string_token` and new `token_*` ops, with the seed loaded from the product's StringTable | no, but it is the oracle | SSA interpreter thread | new module; `ssa_reference_evaluator` string ops | c611 |
| TOK3 | SSA vocabulary: ops `token_intern`, `token_length`, `token_glyph`, `token_concat`, `token_slice`, `token_find`, `token_from_utf8`, `token_to_utf8`, declared in `ir_string_ops` beside `STRING_OPERATORS`. Retire the byte-view ops to ingestion only. | no | SSA interpreter thread | `ir_string_ops` | none |
| TOK4 | C runtime and C, LLVM and Fortran lowering of TOK3 through the shell; the product owns one `TL*`, seeded from its StringTable | **yes**, if interning or a collision check is missed (gate every backend: `TYPED_CHECK_BACKENDS=c,llvm,fortran`) | Lane E | `ssa_c_backend`, `ssa_llvm_backend`, `ssa_fortran_backend` string-token sites; `fortran_c_shell` link | c612-c614 |
| TOK5 | Wasm and JS lowering of TOK3 (BigInt in JS; i64 in Wasm, never an i32 slot) | **yes** (the i32 wrap) | Lane J (JS), Lane X (Wasm) | `ssa_javascript_backend` string ops; `fused_program_wasm_backend` | c615 |
| TOK6 | G1: runtime-built or runtime-selected `str` (f-strings, `''.join`, `+`, a conditional choice) lowers to glyph runs plus `token_intern`, instead of refusing. Covers the 12 bootstrap units and doc 34's B5. | **yes** (equality and keys must meet the constant tokens) | Lane A (bootstrap families) | `glyph_sequence_format`, `resident_sequence_materialization`, the f-string recipe sites | c616-c617 |
| TOK7 | `len()`, indexing and iteration of a selected `str` go through `token_length` and `token_glyph` (c474x) | **yes** (c474x is a silent-lane row) | silent lane | the len/selection sites named in the c474x row | c618 |
| TOK8 | Callback `str` arguments: an i64 token plus the `TL*` the host can read through `tl_to_utf8` (doc 41 "Not carried yet"). Dict arguments stay a separate item. | no, loud today | SSA interpreter thread (it owns #901) | `turing-host-callback-v1` declaration | c619 |
| TOK9 | Shader rule: GLSL and SPIR-V compare tokens and read seed glyphs; an intern in a kernel is refused with a named reason | no | shader thread | `glsl_deployment_strategy` token sites | inside c450-c459 |
| TOK10 | Frontends: J14 (JS strings) and C10 consume TOK3 | no | Lane J, Lane C | per lane | their own blocks |

Order of landing:
1. TOK0 (silent, small).
2. TOK1 (the spec, which Albert reviews).
3. TOK2 and TOK3 together.
4. TOK4.
5. TOK5, TOK6 and TOK7, in parallel once TOK4 is in.
6. TOK8 and TOK9.
7. TOK10.

Each backend item ends with a concordance check that the product's seed section equals the StringTable facts posted at ingestion (`STRING_TABLE_ENTRY`). A product whose seed disagrees is refused.

## 5. Decisions for Albert

1. **Identity.** Keep today's content-addressed token, and make the live table the authority that traps on a collision (my recommendation)? Or a pure live-table ID, which breaks every compiled constant and dict key that relies on the hash?
2. **Collision policy in v1.** A loud trap. The alternative, salting, gives up content addressing for the colliding word.
3. **Lifetime in v1.** Append only for the life of the process. Is that acceptable for the long-running products (Nodus hosts, the compiler service)?
4. **The shader rule.** Read-only on the GPU, with interning on the host?
5. **Byte views.** Retire `string_concat`/`slice`/`split_part` on bytes to ingestion only? Today they are a header with no lowering, so nothing breaks.

## 6. Not audited

These were seen but not read line by line:
- `abstract_ui_vocabulary.py` and `abstract_ui_div_map.py` (UI strings, probably TOK10 consumers);
- `control_source.py`;
- the `topological_reducer` glyph recipe writers;
- `identity_concordance` uses of `STRING_GLYPH_VOCABULARY_MINT`.

Read them before TOK6 is claimed.
