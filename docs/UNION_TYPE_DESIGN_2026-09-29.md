# Union (sum-type) values in the source compiler — design

Date: 2026-09-29. Status: **design for review, nothing implemented.**

Trigger: `examples/python_semantics_gauntlet.py` lowers `for case in CASES:
case()` over a static table of 20 functions whose returns have 20 different
types. The compiler has no value whose type is one of several alternatives.
Three read-only maps of the codebase (type tables; memory and alignment in
turing and nodus; the create→merge→project→ABI path) are condensed here.
Source locations are cited by function name; line numbers drift.

## 0. Direction (user, 2026-09-29)

1. Establish the SSA for unions and put them in a **type table**.
2. Form what the type means for **memory allocation**: unions use aligned
   memory by default at both the small scale (element / SIMD) and the large
   scale (arena / page).
3. nodus's C-based in-memory tensor arena is the reference for the memory
   model.

And the standing rules: the identity concordance is the graph of all
morphing (no private records beside it); valid Python must compile; the
DTYPE manifesto's resolution states are stated, never discovered.

## 1. What exists, in one paragraph each

**Value model.** `SSAValue(id, dtype, shape, device, accounting)`; every
richer fact rides in `accounting` or in a per-function table on `IRModule`:
`tensor_tables` (`SSATensorDescriptor`: dtype, shape, strides, byte_size,
byte_offset, arena_id, alias_of, metadata_state …), `sequence_tables`
(`SSASequenceDescriptor`: per-column arenas, length/capacity/status cells,
`column_shapes`/`column_dtypes`), `record_tables` (book-backed;
`SSARecordFieldDescriptor` with storage kinds SCALAR/SPAN/SEQUENCE/RECORD/
REFERENCE/FUNCTION_TABLE/CLASS_TABLE), `reference_tables`, `call_table`.
All descriptors are frozen dataclasses with `__post_init__` validation and
`to_mapping()`; tables `register()` and raise on conflict. Fortran
republishes the tables into artifact metadata with a schema string.

**dtype authority: none.** Twelve separate dtype→(size | C type | LLVM type |
Fortran kind | numpy) maps (`compiled_program_api._C_TYPES`,
`ssa_c_backend._numpy_dtype/_dtype_for_c_storage/solved_buffer_type`,
`ssa_llvm_backend._value_llvm_type/_LLVM_TYPE_BYTES`,
`ssa_fortran_backend._DTYPE_KIND`, three inline `dtype_bytes` dicts in
`tensor_ssa_lowering`, `ssa_backend._dtype_bytes`, the WASM `_TYPES`, …).
The manifesto says it plainly: "There is no dtype system to extend."
**Alignment appears nowhere**: only byte sizes.

**The existing 2-alternative union: the typed optional.** Contract
`optional: True` → an explicit Boolean presence cell **beside** the typed
payload cell ("the payload is never used to infer absence"). Graph: `is
None` tests become a `<param>.<field>.__present` Input. SSA
(`ssa_optional_values.lower_optional_scalar_returns`): payload keeps its
dtype and gets `accounting["ssa_optional_presence_id"]`; a mixed Phi is
split into a payload Phi and a presence Phi; the absent arm becomes an
inactive `Const 0`; callers receive `dtype="ssa.aggregate", shape=(2,)`
destructured by GEP/Load. Gate: `ssa_self_check.check_optional_merges`
refuses a bare mixed Phi. Backends see an ordinary `bool`. Both cells
always exist; **no storage overlap**.

**Other tag-like things, all static except one.** `token_vocabulary` scalars
(an `int64` holding `index+1` over a finite string set; `isinstance` on them
becomes a runtime `Const`/`Eq` chain) are the only runtime tag. `Phi`
dtype settlement is unanimous-or-unresolved; `result_class_ref`,
`_ASTReferenceAlternatives`, "variant rows" in region planning, and
`ssa.aggregate` heterogeneous returns are all resolved at compile time.

**Memory in turing.** C emission allocates every arena with
`calloc(count, sizeof(*p))` and casts host pointers raw; stack arrays for
literal aggregates. LLVM promises at most `align 8` (`_align`), frames are
`alloca … align 8` under 16 KiB else `malloc`. No `aligned_alloc`, no
`_Alignas`, no vector or struct LLVM types, no arena, no page rounding.
Records are **decomposed into per-field columns**; there is no struct/offset
layout algorithm. `turing_pool` is a worker pool, not memory. GLSL encodes
one std430 rule (doubles at base alignment 8).

**Memory in nodus (the reference).** `in_memory_backend.cpp`: a reserved
virtual range committed incrementally; `kPageAlignment = 4096` for reserve/
commit and base pointer; `kLeaseAlignment = 64` for every lease offset and
size (`align_up`); `kSegmentSize = 1024`; best-fit coalescing free list;
clean/dirty leases zeroed in the background; migration when a range cannot
grow in place. `gp_mem_backend_desc_t.alignment` and `alloc(bytes,
alignment)` exist in the backend vtable (host CPU backend ignores them).
Tensor descriptors carry dtype/shape/strides/`total_bytes` but **no
alignment field** — alignment is an allocator constant. Opaque dtypes
`Bytes/Bytes2/Bytes4/Bytes8/Ptr` exist. The only runtime type tagging is the
tool stack's per-byte `ValueTypeId` mask. turing already mirrors the dtype
codes 0–16 and the `NodusTensorDesc` struct (`nodus_arena.py`,
`nodus_backend.py`); no alignment is declared on either side of that seam.

**Merges that silently pick a winner on type disagreement** (latent
miscompiles the union design replaces): `conditional_result` (first typed
arm), `conditional_carried` (pre-branch dtype), `loop_carried` (initial
dtype), `loop_continue_carried` (incumbent), `loop_result_port` (body
definition), `return_merge` (first typed edge), C Phi edge assignment
`t{res} = value` (implicit conversion), `solved_buffer_type` conflict
fallback. `state_merge` in `lower_state_machine` emits **no Phi at all**.

## 2. Reconciliation with the DTYPE manifesto

A region's dtypes are Resolved, Fanned (finite set; one branch-free body per
member; **select once at region entry** by index into a table of native
functions — a selection, never a conversion), or Dynamic (opaque at compile
time; **must be explicitly declared and loudly reported**). "A value does not
know its type; the code knows. It carries no tag, no header, no descriptor."

Therefore:

- A union is the **representation of a declared Fanned/Dynamic alternative
  set on one value**, not a boxed object. Its tag is an ordinary `int64`
  SSA value in its own cell (the optional's presence bit and the vocabulary
  token are the precedents), never a header inside the payload.
- **Static tag ⇒ no union.** When the tag is a compile-time constant per use
  (the gauntlet: iteration *k* of a loop over a static table takes arm *k*),
  the alternatives collapse to Resolved per slot. The dispatch table's
  result is a static heterogeneous tuple; it needs none of §3–§6. This is
  the manifesto's Fanned form and stays the default.
- **A union-producing merge is loud.** A Phi whose arms do not concord
  produces a union only under an explicit contract declaration
  (`loops`/`values` contract field, default off → the compile refuses with a
  precise report naming the value, the arms and their types). Never inferred
  quietly; the eight silent-winner merges above become refusals or declared
  unions.

## 3. The type table

### 3.1 `SSAUnionDescriptor` (fields drawn from existing descriptors)

| field | from | meaning |
|---|---|---|
| `union_id: int` | `record_id`/`sequence_id` | identity of the union value |
| `identity: str` | `SSARecordDescriptor.identity` | authored spelling (e.g. `int | str`) |
| `tag_value_id: int` | `length_address_id`, presence id | the `int64` tag cell |
| `tag_dtype: str = "int64"` | `token_vocabulary` | tag storage |
| `tag_vocabulary: tuple[str, ...]` | `token_vocabulary` | alternative names, index = tag value |
| `alternatives: tuple[SSAUnionAlternative, ...]` | `SSARecordFieldDescriptor` | per arm: `name, storage (SCALAR/SPAN/SEQUENCE/RECORD/REFERENCE), dtype, shape, value_ids, sequence_id, record_id, byte_offset` |
| `metadata_state: str` | tensor | `"static"` (tag constant) / `"dynamic"` / `"unresolved"` |
| `payload_byte_size: int` | `byte_size` | max alternative inline footprint, rounded to `alignment` |
| `alignment: int` | **new** | payload alignment in bytes (§4) |
| `layout: str` | tensor `layout` | `"overlapping"` (C `union`) or `"side-by-side"` (optional-style cells) — §4.3 |
| `writable: bool` | | |

Frozen dataclass, `__post_init__` (tag dtype must be integral; alternatives
non-empty; names unique; `alignment` a power of two ≥ every alternative's
natural alignment), `to_mapping()`.

### 3.2 Attachment and consumers

- `IRModule.union_tables: dict[str, SSAUnionTable]` beside `record_tables`;
  `SSAUnionTable.register()` raises on conflict like the others.
- Readers that must learn the table: `ssa_c_backend` per-function table
  pull and `solved_buffer_type`/`buffer_type`; `ssa_llvm_backend`
  `_value_llvm_type`/`_declared_span_rank`; `ssa_fortran_backend` snapshot
  and publication (`"union_table_schema": "turing.repository-ssa-union-table.v1"`);
  `ssa_storage_requirements`; `ssa_self_check` (new `check_union_merges`,
  sibling of `check_optional_merges`, wired into
  `_full_native_link_failures`); `extraction_contract` storage kinds (a
  `union` field kind with `alternatives:`); `ssa_optional_values` becomes
  the 2-arm special case (tag 0 = None).

### 3.3 The dtype/alignment authority the table forces into existence

One module (`src/transmogrifier/dtype_layout.py`, name to taste) declaring,
per dtype: byte size, natural alignment, C spelling, LLVM spelling, Fortran
kind, numpy dtype, nodus code. The twelve copies become lookups. This is a
prerequisite, not a side quest: `payload_byte_size` and `alignment` cannot
be computed from twelve disagreeing tables.

## 4. Memory model

### 4.1 Small scale (element / SIMD)

- Every alternative's cells are laid at offsets rounded to the cell's
  natural alignment (8 for `f64`/`i64`, 4 for `i32`, 1 for `i1`), the first
  offset algorithm the compiler has had; it also serves records if they are
  ever laid out rather than decomposed.
- The union payload's alignment is **64 bytes by default**: one cache line,
  every SIMD width in use (SSE 16, AVX 32, AVX-512 64), and exactly nodus's
  `kLeaseAlignment`, so a payload can be handed to a nodus lease without
  re-copying. The work contract can lower it (16 for frame-resident scalars
  only) or raise it; the descriptor records the chosen value.
- Frames: LLVM `alloca … align <alignment>` (today only `align 8` is ever
  emitted); C `_Alignas(alignment)` on the local.

### 4.2 Large scale (arena / page)

- Union storage that outlives a frame lives in arenas allocated **page
  aligned (4096)** and carved into **64-byte leases**, mirroring nodus
  (`kPageAlignment`, `kLeaseAlignment`, `kSegmentSize = 1024`, best-fit
  coalescing free list). turing has no arena today; the first one is either
  (a) a small C allocator emitted beside the program (`_aligned_malloc` /
  `aligned_alloc` for the page, lease carving inline), or (b) delegation to
  nodus's arena through the existing `nodus_arena.py` seam when the program
  is deployed against nodus. Recommendation: (a) for standalone C, with the
  constants shared with (b) so the two agree byte for byte.
- Every private `calloc` arena in the exported wrapper becomes a lease from
  that arena (one allocation, aligned, freed once) — a collateral fix for
  the 393 separate `calloc`s in the Woodshop wrapper.

### 4.3 Layout of the payload: two admissible forms

- **Overlapping (proposed default).** The payload is one region of
  `payload_byte_size` bytes; alternative *k*'s cells alias it at their
  offsets. C: a per-union `typedef union { struct {…} alt0; … } _Alignas(64)
  U_<identity>;` — aligned, strict-aliasing-clean, no reinterpret op needed.
  LLVM: `[payload_byte_size x i8]` with typed GEP/Load at the offsets
  (LLVM has no union type; this is how Clang lowers one).
- **Side-by-side (fallback).** Every alternative's cells exist separately,
  as the optional does today. No aliasing, no new emission concept, memory =
  sum not max. Reuses the optional path unchanged.

The descriptor's `layout` field records which one a union uses; the contract
picks the default.

## 5. Concordance

- New page `union_alternative_state`, row `(scope, value_id)`, fact
  `("alternatives", tag_value_id, (state_0, …, state_n-1))` where each
  `state_k` is the existing 5-tuple `(shape, dtype, rank, metadata_state,
  row_shape)`, so `descriptor_from_shape_transformation_state` and
  `withdraw_superseded_shape_derivations` work per arm unchanged;
  invalidation fact `("invalidated", source_id, reason)` as elsewhere.
- Every injection and projection is an edge: `record_shape_transformation`
  with `stage="union_injection"` (arm *k* value → union value, role
  `alternative:k`) and `stage="union_projection"` (union value → arm *k*
  value). A merge that produces a union records one injection edge per
  incoming control edge, with the tag constant that edge sets.
- No dictionary beside the book. The `SSAUnionTable` is the *physical*
  layout record (like `tensor_tables`); the book is the fact.

## 6. Pipeline points (each extends a named analogue)

| step | analogue | what changes |
|---|---|---|
| create / inject | `Const k` + stores, like optional's `Const 0` inactive payload and `ssa_optional_presence_id` | tag `Const k` into the tag cell; alternative-*k* cells written; accounting `ssa_union_tag_id`, `ssa_union_alternative` |
| merge | `conditional_result`, `return_merge`, `loop_*` Phi sites; `state_merge` (no Phi today) | arms concord → today's Phi; arms disagree → refuse, or under declaration produce a union: tag Phi + per-alternative cell Phis (the optional's split-Phi pattern) |
| project / use | vocabulary `isinstance` → `Const`/`Eq` chain; `StateMachineTick` on a scalar | tag test at the region seam; alternative-*k* loads guarded by it; `isinstance(x, T)` on a union value becomes a tag comparison instead of a static fold |
| gate | `check_optional_merges`, `check_formal_parity` (`program_abi_optional_presence`) | `check_union_merges`; accounting key `program_abi_union_tag` so a tag formal is accounted |
| ABI | `field.__present` descriptor beside payload; sequence `length` + arena | tag slot + payload slot(s) in `buffer_order`; `buffer_dtypes` must admit the tag and the payload byte buffer |
| decode (host) | `StringTable.get`, `NONE_TOKEN` (no consumer today) | a decoder: tag → alternative descriptor → payload view; also gives the rating/gauntlet harnesses their missing token/presence decoding |

## 7. What has no existing mechanism (consolidated)

- A dtype/alignment authority (§3.3).
- Any alignment field, any aligned allocation, any arena, any struct/offset
  layout algorithm, LLVM `align > 8`, vector/struct LLVM types.
- Overlapping storage of different dtypes over one buffer (`alias_of` is
  same-dtype views only); a reinterpret/bitcast op (`Cast` converts).
- A runtime tag with more than two arms; a Phi over disagreeing dtypes that
  is anything but a silent winner; a Phi at `state_merge`.
- Runtime `isinstance` except on vocabulary tokens.
- A host-side decoder for tags, presence bits or string tokens.
- A `tensor_output_descriptors` shape for "one of N descriptors".
- The manifesto's "select once at region entry from a table of native
  functions" has no code.

## 8. Implementation sequence (proposed)

1. `dtype_layout` authority; replace the twelve copies with lookups.
   (Independently valuable; no behaviour change.)
2. `SSAUnionDescriptor`/`SSAUnionTable`/`IRModule.union_tables`; Fortran
   publication schema; `check_union_merges` scaffold (refuses everything —
   i.e. makes today's silent-winner merges loud).
3. Concordance page and the two edge stages.
4. Re-express the optional as the 2-alternative union (behaviour-preserving;
   proves the table end to end through the existing marshalling).
5. Dispatch table for the gauntlet: static tag ⇒ per-slot static result
   (§2); no dynamic union yet. Add a gauntlet case that *requires* a dynamic
   union (`x = f() if runtime_flag else "text"`).
6. Merge lowering under declaration: tag Phi + cell Phis; C `union` +
   `_Alignas`; LLVM byte payload; aligned arena allocator (§4.2, form (a)).
7. Host decoder; wire into the rating and gauntlet harnesses.

## 9. Questions for the user

1. **Layout default:** overlapping C `union` (max, aligned, new emission
   concept) or side-by-side cells (sum, reuses the optional path)? §4.3.
2. **Alignment default:** 64 B (nodus lease, cache line, AVX-512) for every
   union, or 16 B with 64 only for arena-resident unions? §4.1.
3. **Arena:** emit turing's own aligned arena mirroring nodus's constants
   (form a), or delegate to nodus's arena where present (form b), or both
   sharing constants? §4.2.
4. **Declaration:** is the contract switch that admits dynamic unions
   per-program, per-function, or per-value (an annotation such as
   `Union[int, str]` on the binding)? §2.
5. **Tag width:** `int64` (vocabulary precedent, aligns with nodus codes) or
   the narrowest integral type that indexes the alternatives?
