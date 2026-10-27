2026-10-26 LLVM SINGLETON AGGREGATE ELEMENT LAYOUT (MEASURED, Windows):
# LLVM singleton aggregate element layout

The bounded typed float validation source passes `[1.5]` into a loop.
Its static_iterable Const has dtype ssa.aggregate, shape(), payload(1.5,),
and a GEP followed by a float64 Load. LLVM emission refused a double load
from an i64 element. The exact writer was the Const handler: it collapsed
every singleton tuple with empty shape/rank0 to a scalar, including the
aggregate carrier. That skipped aggregate_constant_types publication, so
GEP's structural pointer element fallback selected i64.

The existing singleton scalar-offset rule now additionally requires the
canonical LLVM kind to be a numeric scalar. An ssa.aggregate retains its
element layout at any length. No load conversion, forced float dtype,
integer signedness relaxation, or alternate aggregate representation was
added. The empty aggregate branch is unchanged.

The source float probe now emits with no shortfalls. Direct repository SSA
native tests cover seven singleton/multiple float64, int64 above2**53, and
True/False aggregate cases with repeated reads. Observed:7passed3.13s.
Typed native integer and comparison neighbors observed:22passed6.26s.
`git diff --check` is clean.

The seven source-level counterparts initially report a separate
value-type-abi-disagreement at the loop target: its ordinary type ABI does
not derive directly from the exact source_value_class cell. The integration
lane owns that provenance writer; no source-level clean/native claim is
made by this intermediate receipt. No manager or bootstrap run was launched.
