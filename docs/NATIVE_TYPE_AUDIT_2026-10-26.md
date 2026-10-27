# Native compiler type audit — 2026-10-26

Audited integrated Turing main `e31b5522` on Windows. No compiler implementation was changed. This audit covers every non-pointer scalar layout declared by `src/transmogrifier/dtype_layout.py`, its public ABI and allocation views, source array ingestion, sequence mutation, and scalar native emission. C/LLVM were probed dynamically; Fortran declarations were queried; Wasm, JavaScript, SPIR-V, WGSL, KernelIR and GLSL were inspected statically. This is a compiler representation audit, not exhaustive numerical certification of every operator/backend or of AbstractTensor/Precision arithmetic.

## Result

Unsigned support is incomplete across boundaries. Adding uint32 in individual consumers can preserve a spelling while the physical representation remains double. The catalogue declares correct unsigned layouts, but its deliberately preserved legacy lane vocabularies do not expose those layouts consistently. Direct emission can report complete despite a changed representation or invalid LLVM. The public extraction refusal covers explicit unsigned contracts, but internal record arrays can reach these emitters.

A compiled C repro establishes actual value loss:

```text
SSA: uint64 Add(uint64(9007199254740993), uint64(0))
artifact.complete: True
expected: 9007199254740993
actual:   9007199254740992.0
output buffer dtype: float64
```

This repro uses the repository's module-preserving C emitter, compiler and execution API. It is a small emitter-level SSA probe, without an attached source identity book; it does not claim that an explicitly unsigned ProgramABI was accepted. The canonical record-array probes below independently establish the internal source path that admits unsigned rows.

## Measured layout matrix

These are actual lookup results, not claims of end-to-end support. Fortran means the actual declaration query; unsupported declarations raise `FortranEmissionError`.

| Declared dtype | Declared/lookup bytes | LLVM lookup | C scalar | C allocation dtype | Fortran |
| --- | --- | --- | --- | --- | --- |
| bool | 1/1 | i1 | uint8_t | bool | logical(c_bool) |
| int8 | 1/1 | double | double | float64 | refusal |
| uint8 | 1/1 | double | double | bool | refusal |
| int16 | 2/2 | double | double | float64 | refusal |
| uint16 | 2/2 | double | double | float64 | refusal |
| int32 | 4/4 | i32 | int32_t | int32 | integer(c_int32_t) |
| uint32 | **4/8** | double | double | float64 | refusal |
| int64 | 8/8 | i64 | int64_t | int64 | integer(c_int64_t) |
| uint64 | 8/8 | double | double | float64 | refusal |
| float32 | 4/4 | float for spans | double | float32 | real(c_float) |
| float64 | 8/8 | double | double | float64 | real(c_double) |

C/LLVM deliberately widen float32 scalars to double; that is documented existing policy, not counted as an unsigned defect or silently redefined here. It still needs to be distinguished from float32 storage/rounding semantics when validating a program.

## Findings and exact consumers

1. **Silent integer-to-floating representation substitution.** `dtype_layout.py:LLVM_LANE_NAMES` omits int8/int16 and all uint widths. `llvm_type_for_dtype` returns double for them. `ssa_llvm_backend._value_llvm_type` consumes that answer even when `physical_dtype` is present. C's independent `_scalar_c_type` also returns double. Observed: Add for all six types emits floating addition, with no shortfall; all six LLVM modules parse and verify. The declarations and consumers disagree before arithmetic begins.

2. **uint8 allocation loses its numeric class.** `c_lane_numpy_dtype` goes through `c_lane_dtype_for_storage`, whose storage class search identifies uint8_t as bool. Result: uint8 allocation requests produce NumPy bool. Sharing byte width does not establish equivalent value semantics. This was observed in the allocation lookup; this audit did not claim a particular canonical byte program executed that path.

3. **uint32 byte accounting is wrong.** `BYTE_SIZE_NAMES` omits uint32; `dtype_byte_size('uint32')` returns default 8 despite `layout_for('uint32').byte_size == 4`. Consumers include tensor SSA allocation/copy addressing and `SSATensorOperations._dtype_bytes`. This is a measured layout disagreement; a specific corrupting copy or out-of-bounds access has not been demonstrated here.

4. **Complete does not imply valid LLVM.** For each uint8/uint16/uint32/uint64, the ULt, Shr and BitAnd probes return `complete=True` but fail LLVM verification: argument registers are double, while the operation template demands i64. Twelve verified failures. Mod/FloorDiv/Lt instead produce valid floating operations. An int64-to-unsigned Cast produces `sitofp i64 ... to double`, not an unsigned row conversion.

5. **Name mapping alone cannot repair signedness.** `integer_scalar_lines` receives an LLVM width, not signedness. Div, comparisons, Min/Max and Abs select signed instructions, while Shr selects logical shift. Conversion helpers choose signed integer-to-float, signed float-to-integer, and sign extension (apart from bool/explicit unsigned opcodes). Once uint32 maps to i32, its original sign cannot be reconstructed from i32. These are observed code paths and a predicted next failure frontier; they have not been presented as executed unsigned-integer results because current uint values reach double first.

6. **Explicit public ABI protection is incomplete as a global protection.** `extraction_contract.py` rejects unsigned layouts without an exact native lane. All three existing unsigned refusal tests pass. `compiled_program_api`'s C ABI view nevertheless maps uint16/uint32/uint64 to double if directly queried; int8/int16 also default to double. The shell rejects unknown names instead. Thus direct emitters and internally authored record arrays are not covered by a universal refusal. No refusal was relaxed for this audit.

7. **Array ingestion is a partial whitelist.** `_scalar_abi_for_expression` recognizes only array codes B/I/Q, yielding uint8/uint32/uint64. Canonical c632 variants B/I/Q lower with correct descriptor spellings and zero detector findings. The same small source with H/b/h/i/q/f/d refuses at constructor physical identity. uint16 conversion code exists downstream without a corresponding H constructor handoff. This is not evidence that unsupported constructors silently work.

8. **Append and extend have different conversion contracts.** Append now range-checks int64 against the declared unsigned layout before its SSA Cast. Native emission of that Cast still follows the faulty dtype mapping. Extend calls `concord_sequence_row_dtypes` as physical equivalence and requires identical source/destination rows; the existing helper also requires matching heap element layouts. The tiny `array('Q').extend(list[int])` canonical repro fails exactly like whole-core sequence 75 uint64 versus 39 int64. Keep the refusal until a real checked per-element conversion exists; do not relabel either arena.

9. **Aliases are inconsistent across views.** Measured examples: catalogue float means float32, while LLVM lookup float means double and its C allocation means float64; public ABI/shell/Wasm/Fortran float mean float32. Catalogue long means int64; LLVM/C scalar accept it but public C ABI defaults to double and shell rejects it. Catalogue logical means bool; LLVM/C allocation default to double/float64 and byte lookup returns 8. Unknown names also default to double in several views. These choices were intentionally preserved in the catalogue migration; their historical presence is not evidence of cross-boundary equivalence.

10. **Existing tests permit false confidence.** `test_array_u32_receiver_column_keeps_authored_dtype_before_append` verifies the SSA descriptor/Cast and searches the entire LLVM text for `trunc i64`. It neither checks that the uint32 column is i32 nor identifies the relevant conversion nor executes a boundary value. c632 checks source concordance only. Zero identity findings and valid LLVM can coexist with the wrong native representation, as the probes demonstrate. These assertions were not weakened or rewritten during the audit.

## Other emission lanes

| Lane | Observed handling | Evidence limit |
| --- | --- | --- |
| Fortran | Explicit refusal for every unsigned width and int8/int16 | Declaration queries, no native execution |
| Wasm fused program | Working-type table omits unsigned; `_value_type` raises. Separate memory table supports uint8 zero-extending byte load/store | Static inspection; byte memory support is not general unsigned arithmetic |
| JavaScript SSA | uint32 Cast uses `>>> 0`; uint64 Cast defaults to Number. int64/uint64 bitwise ops refuse through `dtype_is_int64` | Static inspection; Number arithmetic beyond 2^53 remains unverified and is not exact integer coverage |
| SPIR-V SSA | Type table lacks unsigned; scalar_type raises | Static inspection |
| WGSL | uint32/u32 are declared; other unsigned widths lack WGSL scalar mapping | Static inspection, no GPU execution or whole-call ABI certification |
| KernelIR | Explicit I32/U32/F32 whitelist | Static inspection, not a general CPU ABI |
| GLSL compute SSA | Component buffers expose f64; integer-index vocabulary does not include uint | Static inspection; typed tensor GLSL backend tests are a separate lane |

The interpreter/reference evaluator was not used as the numerical parity oracle. Precision/ComplexPrecision remain their existing substrates; this audit does not propose changing their arithmetic or using floating tolerances for integer faults.

## Reproduce and validation

```powershell
python -u tools/audit_native_dtype_coverage.py --native-dir build/native_dtype_audit
```

Measured in about five seconds: **32 discrepancies**, exit 1 intentionally. This count includes related observations, not 32 independent root causes: 19 layout/allocation mismatches, 12 invalid complete LLVM modules, and one native value/storage failure. The script prints all evidence inline. Without --native-dir it skips compile/execution and reports 31.

Canonical array probes used c632 with only the array code changed. The separate minimal extend source uses a Ledger containing array('Q'), creates a list, appends one int parameter and extends the field. Both are seconds-scale source lowerings through the sanctioned public entry; no long whole-core run was launched.

Existing focused tests: unsigned refusal plus scalar merge suites: 7 pass, 3 fail on both this branch and untouched main. The failing tests are the synthetic boolean merge chain, integer backedge merge and mixed representation phi cases, all stopping on missing SSA identity. They do not establish native unsigned correctness. Mapping concordance audit remains 26 rows, zero findings, zero unsourced. No full suite/fuzz/Linux/GPU gate or exhaustive mixed-type/overflow/NaN/Precision numerical run was performed.

## Repair order supported by this audit

First make unsupported physical representations refuse at every actual emission entrance, including internal fields, so a complete artifact cannot silently change type. Then establish width and signedness from the existing canonical dtype record through storage, operations, casts, calls and outputs as one contract. Prove each enabled width with exact boundary-value execution before widening its accepted vocabulary. Address checked extend conversion only after native uint rows are truthful. Retain explicit source/target layouts; never pretend list[int] and array(Q) are physically equivalent.

Required exact cases include zero, sign-bit boundaries, every unsigned maximum, 2^53+1, uint64 values above INT64_MAX, widening/narrowing and invalid inputs, mixed signed/unsigned operations, load/store stride, call/return, phi/select, constructor/append/extend, and serialization/output decoding. Overflow behavior must follow each authored operation and existing type contract; this audit does not invent wrapping semantics for Python ints.
