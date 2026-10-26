2026-10-26 IMPORTED ARRAY OWNER BIRTH (MEASURED, Windows):
# Imported constructor array ownership

Manager4 failed after310s at stage9: imported Vocabulary.__init__.domain
required a declared heap owner but its allocation descriptor had none.
Its saved resolved process graph shows SetAttr57 valueedge→Call56, whose
exact lexical callable is array.array. The call retains callee55 and arg0
typecode20. Existing normalization consulted only root static bindings and
external_callee_ref, so it never marked this ingested constructor allocation.

The existing normalization now also consults each function wrapper's exact
static_python_reference binding and compares the actual callable to array.array.
The prior resolved external-reference route remains supported. Both the
resident owner survey and SetAttr dtype handoff count argument edges while
preserving the static callee edge. No field-name/module-name mask, guessed
capacity, dtype override, or second owner model was added.

Bounded actual imported Vocabulary source (`v=Vocabulary();
v.domain.append(1); return len(v.domain)`) lowered in8.07s. Before the fix,
constructor dtype contracts were empty; after the fix, all11 authored arrays
have exact uint8/uint32/uint64 layouts and DYNAMIC owner descriptors.
The bounded source's audit has483rows5functions,0unsourced and4findings only
for unknown keycolumns in the unused _candidates/_prefix fields. Those
unknown layouts remain unknown; this probe is not a whole-core clean claim.

The new imported-module regression requires the exact retained constructor
field owner, appends two values through its imported method, checks count2
on three repeated native activations, and asserts zero concordance findings.
C/LLVM/Fortran all passed. Combined gate:

`python -m pytest -q tests/test_imported_array_owner_birth.py tests/test_owned_array_record_native.py tests/test_constructed_child_pool_native.py --tb=short`

Observed completion:17passed42.01s, one existing cffi deprecation warning.
`git diff --check` is clean. No long manager or whole-core run launched by
this lane for this repair.

Separate integration receipt reported by the root lane: untouched clean
review checkout C:/dev/tg-child-review at5bcc1bba completed11focused suites
(constructed childpool/nativegrowth/computed/optional/recordargs/class
refusals/mappingviews/fixedbounds/returnpruning/boolnumeric/mixedcomparison):
106passed189.21s. That receipt predates this imported allocation birth fix;
manager4's failure remains a separate observed result, not a native success.

Clean integration checkout C:/dev/tg-child-review at1dda48ca: imported array birth, owned array native, constructed child pool, project native cases, and extraction source sheet suites completed31passed129.46s. Manager5 was running separately at this checkpoint; no full-manager native result is claimed.
