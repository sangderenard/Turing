# Woodshop outer-program concordance handoff

## Outcome

The Woodshop outer-program probe now passes the complete full-native link
gate. The final linked repository SSA has no unmaterialized extraction
boundaries, unresolved calls, undefined operands, unaccounted formals,
unresolved record rows, optional-merge failures, structural-output failures,
or non-native boundaries.

The probe reaches C emission with 161 functions. Its next failure is separate
from source discovery and call-frame concordance:

```text
TypeError: float() argument must be a string or a real number, not 'slice'
```

`emit_ssa_module_to_c` reaches a surviving Python `slice` constant and sends
it through the scalar numeric-constant path at
`src/compiler/ssa_c_backend.py:3012`. This handoff records that frontier; it
does not attempt to classify or fix it.

## Repairs in this change

### Runtime loop bounds

A source `for` loop now retains its runtime bound whenever its iterable has an
identity, even if target destructuring publishes no target value. Nested
`enumerate` lowering uses the enumerated source iterable as the extent owner.
The compiler still reports `unresolved-loop-bound` when neither a stop nor an
iterable identity/constant exists.

### Optional and reference record representation

Optional record fields retain their typed payload plus presence bit. Declared
reference fields normalize to the repository physical type `opaque_ref` and
do not enter scalar dtype completeness checks. Specialized record instances
retain their established `result_class_ref` identity.

### Record/sequence call-frame identity

Record fields that decompose into keyed or sequence storage link through the
record-field decomposition concordance. A sequence's role-labelled physical
members are authoritative over its flattened inventory. Fixed span fields are
expanded into their declared physical columns, and tuple-backed fields remain
one resident sequence rather than acquiring a parallel opaque object.

### ABI-directed source pursuit

ProgramABI record bindings are published to source pursuit before unresolved
parent expansion. A wrapper such as `world.step(...)` therefore selects the
exact retained record class and walks its same-owner method closure. Closed,
zero-argument lexical factories reached from those methods, including
`empty_channels()`, enter ordinary source pursuit without reopening the whole
retained class catalogue.

### Provisional variant-row settlement

Variant-aware region planning can create a row carrier before a source call's
physical result is linked. Once all non-provisional region-feed occurrences
for that semantic feed agree on one linked span, the compiler publishes the
row-to-resident edge in the existing planning concordance. Dominance-aware
alias settlement then rewrites the views and ordinary paired signature
pruning removes the provisional formal. Multiple physical candidates cause
the pass to abstain.

This is an indexing/view conversion over one physical object. It does not
allocate or publish a second resident.

### Constructor fields and compiler-frame ownership

`WorldContact.penetration_m` exposed a distinction between representation and
ownership. Its value was an indexed member of the local collision-candidate
row and every exact caller supplied compiler-frame storage, but later record
constructor metadata caused the frame reconciler to treat it as an authored
ProgramABI input. Constructor-field metadata may now coexist with proven
compiler-frame ownership when no ProgramABI parameter owns the value and all
incoming calls supply frame storage. Declared parameter fields remain
unchanged.

### Diagnostics

Formal-parity findings now include each orphan formal's accounting metadata.
`TURING_DEBUG_DUMP_FUNCTION=match:path` additionally writes signature ledgers,
value aliases, and matching source-graph nodes for the selected functions.
The full-native failure message also reports unresolved record-sequence rows.

## Verification

The final focused verification completed successfully:

```text
tests/test_ast_parent_ingestion.py                         2 passed
tests/test_loop_composer.py                               2 passed
formal/storage/linking/self-check selection               7 passed
python -m py_compile                                      passed
git diff --check                                          passed
```

The full Woodshop probe completed SSA lowering and cleared the full-native
gate:

```text
[outer] ssa-program ... complete; functions=162 exports=78
OUTER_NATIVE_LOWERED functions=161
```

It then failed in C emission on the `slice` constant described above. No
native library was emitted by this run.

## Reproduction from CMD

This form merges Python stderr in CMD before PowerShell receives the stream,
so it displays and logs the original traceback without PowerShell converting
each stderr line into a `NativeCommandError` record:

```cmd
python -u build\woodshop_outer_native_probe.py 2>&1 | powershell -NoProfile -Command "$input | Tee-Object -FilePath 'build\woodshop_outer_native_probe.log'"
```

## Deferred concordance visualizer

An optional live concordance visualization is feasible but is not part of
this change. The compiler should publish immutable revisioned snapshots or
deltas into a double/triple buffer. A separate Pygame/OpenGL process can own
the window and GPU buffers, consume the newest complete revision, and drop
intermediate revisions when rendering falls behind. Compilation must never
wait for the observer.
