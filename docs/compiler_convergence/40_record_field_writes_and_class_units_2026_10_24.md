# 40. Record field writes and whole-class units (2026-10-24)

Thread "Field stores in whole-class compile", branch
`claude/project-thread-umsbgm`. It started from a refusal in a whole-class
DLL compile, `record 'AbstractTensor' field 'track_time' names absent record
1`. It found a silent wrong answer in ordinary programs on the way.

## F1. A field read of a constructed record ignored later writes (SILENT)

**Program** (c396):

```python
s = S(0)          # S is a dataclass with one int field n
if c > 0:
    s.n = 1
return s.n        # c=3: native 0, Python 1
```

At ef49715, every write to a field of a record the function constructs was
lost. c397 (both arms), c398 (constructor from a parameter) and c399 (an
unconditional write, then a read) were all WRONG on main.

**Records that disagree.** The return's `name_binding` row named the
constructor argument (graph value 2). The `reducer_field_state` cursor for
`(s, 'n')` named the merge (`s.n` phi 12). The phi was then removed as an
unused phi (`unused_phi_removals`).

**Writer** (watched with a book-post trap on `name_binding`, row
`result_0`): `topological_reducer.resolve_expression`, `ast.Attribute`
branch. The rule "a field read immediately from an authored record
construction is the exact constructor argument" applied whatever had been
written to the field since construction.

**Fix.** The fold happens only while the read's field-state cursor is empty
or OBSERVED at that same value. When the cursor holds a WRITTEN, MERGED,
ARM_SELECTED or LOOP_EXIT state, the read is the cursor's value, as for a
static receiver's assigned field. The reducer merged no field state at a
loop then, so a field that an enclosing loop body rewrites, or that is read
after such a loop, was refused by name: `local record field written in a
loop has no merged value ... (loop_exit_field_state_unmerged)`. It used to
return the constructor argument. F4 now carries those fields through the
loop; the refusal remains for the cases F4 lists.

**Why the corpus missed it.** The corpus had no class programs. The typed
check, `tests/test_native_semantics_corpus.py` and the fuzzer exec'd each
program with no `__name__`. A class defined there gets `__module__ ==
'builtins'`, and lowering hit a RecursionError walking the builtins module
as static data. All three now exec under the program's own name, and the
pytest file compiles with `dont_inherit=True`, so its `from __future__
import annotations` does not turn the program's annotations into strings.
`FUZZ_EXT=4` gains three record-field snippets.

## F2. Whole-class units through the canonical catalogue

`compile_class_to_dll` / `compile_section_to_dll <Class>` is the deprecated
adapter (`warn_legacy_source_compiler`). It prepares only `__init__`; the
other methods are listed in its class table with `function_name: null`. It
places field stores after their source's producer, ignoring control flow
(`self.n = 0; if c: self.n = 1` stores both at entry), and stores every
field as float64. It was not changed. The canonical product
(`tools.compile_project_catalogue --emit-native`, one unit per method)
lowers fields through the record ABI: typed columns, and stores inside
their arm. With the fixes below, the 10-line classes match Python through
their DLLs (checked by hand with ctypes; catalogue verification reports
"unsupported" for record ABIs):

- `K(scale: int)` with `get` and `mul`: values equal, including 2**60+1.
- `K(track_time: bool, requires_grad: bool)` with a conditional
  `self.n = 1`: all four flag combinations equal, and `n` is left unwritten
  when Python never sets it.

Records fixed:

1. `_authored_dataclass_record_views`: `self.x = p`, where `p` is an
   annotated `__init__` parameter that is never rebound, is a field receipt
   from the annotation. Without it, a getter's unit had no receiver view
   ("missing used parameter self").
2. `partition_authored_source` keeps `__init__` in a method's unit. Dropping
   it gave the unit a receiver view without the constructor's fields, which
   then disagreed with the declared module's view
   (`source_record_view_schema disagreement`, 146 AbstractTensor units).
3. `_inject_field_slot_access`: a slot's offset index is minted int64. It
   was minted with the receiver dtype, so a bool-first receiver indexed with
   `logical + 1` and gfortran refused it.
4. `_field_slot_ops`: a typed scalar parameter carries `result_class_ref`
   (`'int'`) beside `value_kind='scalar'`. It was read as a nested record
   instance ("names absent record 0").

## F3. Ordinary classes constructed in a program

**Program** (c423):

```python
class K:
    def __init__(self, scale: int):
        self.scale = scale
    def mul(self, x: int) -> int:
        return self.scale * x
def root(a: int, x: int) -> int:
    k = K(a)
    return k.mul(x) + k.scale
```

After F2's annotated-parameter receipt, `K` has a projection view, and
`abi_record_for_call` claimed `K(a)` as a record literal. It then refused:
"record constructor requires a complete physical schema, but only an exact
projection view is available". Before F2, the call reached the existing
`__init__` frame link ("materialize class constructors and fields") and
failed there with "record_descriptor ... no incumbent to revise".

Records fixed:

1. `abi_record_for_call`: a projection view (`constructor_complete:
   False`) of a class with an authored `__init__`/`__new__` is not a
   constructor schema. That call is the `__init__` frame link. The refusal
   still stands for a dataclass whose layout is incomplete.
2. The frame link registers the caller's copy of the constructor's record
   DERIVED from the constructor's descriptor cell and the call. A bare
   register posted no cell.
3. Merging constructor call records posts the `call_record` row from the
   call's cells and the row it revises. The bare list write was refused.
4. A bound caller id that becomes a frame slot posts the constructor
   value's type. When `self` is the field slot, the receiver's row was
   minted untyped for the call, and result settlement refused "unresolved
   evidence cannot erase resident type".

c422-c429 pass: two instances, construction in both arms, a caller-side
field write, crossed `__init__` fields, and keyword construction.

**Silent wrong answer found on the way: a computed field.** `self.flag = n >
3` has no field receipt, so the record view has no slot for it. `__init__`
computed it and stored it nowhere, and `pick` read garbage (n=4 gave 3,
where Python gives 8). **Rule.** If a class layout has a field that the
constructor's receiver record lacks, the frame link refuses by name: "class
construction has no physical field for attributes of its layout".

**Silent wrong answer, also on main: a call that stores into a local
record.** `acc = Acc(a); acc.add(b); return acc.total` returned `a`. So did
a free function storing into a dataclass, `add(acc, b)`. The discarded
call was dropped, and the field read took the construction's value. On
main this is the same for dataclasses (checked at dd340a1). **Rule.** The
reducer marks a name passed to a call, as an argument or as a method
receiver, unless every authored definition of that callee provably does
not store into that parameter. A proven read-only parameter is used only as
loaded `p.field` operands, or as `p.method(...)` of a read-only method. A
later field read of a constructed record under that name is refused:
`record_stored_by_call`. A loop body's storing calls mark the name for the
whole body. `return acc.add(b)` still lowers and matches Python, because
the method's frame writes the caller's storage.
`tests/test_class_construction_refusals.py` holds the three refusals.

## F4. A field written in a loop is a loop-carried binding

**Program.** `s = Acc(n, 0); for i in range(m): s.n = s.n + i; return s.n`
(corpus c530). After F1 this was refused as
`loop_exit_field_state_unmerged`.

**Rule.** A loop already has a pre-loop and a post-loop context for names
(the user's point, 2026-10-24): `loop_carried_bindings` maps a name to its
value before the loop and its value at the end of the body, and the loop
header and exit rebind every read through `lexical_read_binding` rows. A
field of a record this function constructed is now carried the same way,
under the binding `s.n`:

- At the top of the loop (before a `while` test), the reducer records each
  field the body writes with its entry value: the field-state cursor's
  value, or the constructor argument when the cursor is empty.
- Each field read in the loop posts a `lexical_read_binding` row for `s.n`,
  so the header Phi is what the read sees on the second trip.
- At the exit, `loop_carried_bindings["s.n"] = (entry value, body-exit
  value)` and the cursor holds a LOOP_EXIT state at the body-exit value.
- An `if` in the body that writes the field in one arm merges from the
  carried value: a key with no pre-branch state (or with a pre-loop write)
  is seeded at the entry value, and each Phi operand equal to the entry
  value gets a read row, so the arm that did not write reads the header Phi.

**Still refused.** A field of an ordinary class (a projection view,
`constructor_complete` false) in a loop: the view's field order is not the
constructor's argument order (`self.b = a`), so the entry value cannot be
folded from the call. A field written in a loop that has no entry value
(the receiver is not a local construction) keeps the old refusal.

Corpus c530-c539: `for`, read-then-write, writes in alternate arms,
`while` test and body, static unroll, nested loops, `break`, two records
swapping fields, a pre-loop write then one arm, and `return` inside the
loop.

## F5. A field merge's arm settled to a constructor argument (SILENT)

**Program.** `r1 = Rec(n, 3); if n > 7: r1.x = n + 6; return r1.x * 10 + n`
(corpus c550, found by fuzz seed 99 as p99_24). For n=9, native returned
165 and Python 159. Wrong on main c6e3f8b1 too.

**Wrong record.** The merge `Phi12(body 8, orelse 11)` is right in the
graph. The else arm is the field read GetAttr 11, which a late record
settlement replaces with the constructor argument `n` (%1). Then
`reconcile_conditional_phi_continuations`, which continues every later use
of a Phi arm (by SSA identity) to the Phi, rewrote the later read of `n`
to Phi12: region 3 was called with feeds (12, 12) instead of (12, 1).
Names avoid this through `preserved_name_binding_arm_positions`, but id 11
has no name binding, so nothing marked the arm.

**Rule.** The conditional Phi records each arm's SSA id as lowered
(`arm_lowered_value_ids`, with its read scope and merge id). An arm whose
id at that position has changed since was settled to a resident value. If
a `name_binding` row in that scope gives the resident a name this merge
does not version (`n`), the continuation pass preserves the arm by
position. If the resident's names include the merged field's receiver (`k`
for `k.scale`: a one-field record's slot is the record itself, and every
alias such as `j = k` binds the same value), the later use is the record
and still continues to the merge (c426, c552). c551 runs field merges in a
loop whose arms read the arguments; c553 is a one-field dataclass.

## F6. Record programs on the Fortran backend

Lane E's backend parity run found the C-passing record programs refused by
Fortran. Both refusals were loud; neither was a wrong answer.

**Program 1.** `s = S(0); if c > 0: s.n = c * 2 else: s.n = 7 - c; return
s.n` (c397; c399 is the same shape). Fortran refused with "record 'S' field
'n' names absent SSA values [2]". The record is a compile-time correlation
only: neither its id nor any field value is an SSA value of `root`, because
the constructor's `0` is overwritten on both arms and never emitted. The C
backend reads record tables only for record parameters, so it never looked.

**Rule.** `emit_module` checks and exports a record descriptor only when the
record, or one of its field values, is resident in the function. A resident
record with an absent field still refuses. So does a record with any field
that names storage (a span, sequence, nested record or reference), since
those name arenas the function must hold
(`test_record_table_rejects_storage_references_absent_from_function`).

**Program 2.** `k = K(n); return k.get() + n` (c423; c422 and c424-c429
too). Fortran refused with "missing Fortran feed 3". The constructor frame
link (`_class_surface_ssa_program`) clones `__init__`'s formals into the
caller as frame slots. The clones kept `program_abi_parameter` and
`program_abi_field`, so the API classified them as public parameters with
no feed. `concord_program_abi_frame_transitions` re-leased any slot still
marked `program_abi_field`.

**Rule.** A cloned formal that is not one of the caller's own parameters is
`linked_call_frame_storage` of the constructor. Both program ABI keys are
dropped from it.

## Measured

F6, at main 830aa1d9: the typed corpus is 380/381 on C, the same as main,
and 373/381 on Fortran, against main's 363. Fortran gains c397, c399,
c422-c427, c429 and c552, and loses nothing. The
one failure on both trees and both backends is c428, which #879 made wrong
and Lane C's #896 fixes. Gate at 8fb85e7b (llvmlite installed): aa and ac
failure names identical to main's. Main's own ab and rec parts were killed
for memory, so those two were read against `tools/gate/baseline_*.txt`: the
only new names were the record-table refusal test (now passing, see F6) and
`test_project_unit_recognizes_exact_lowered_closure_formals`, which fails
the same way on main 8fb85e7b (it loads a `.dll` on Linux). Fuzz seed 101
at `FUZZ_EXT=4`, 60 programs: 55 PASS, 5 LOUD, 0 WRONG.

F4 and F5, rebased onto main ae46579: the typed corpus is 372/373 PASS
and main ae46579 is 358/359. The one failure on both is c428, which #879
(plan callsites placed at their call's end) made silently wrong on main:
the `.get()` call is pulled ahead of the `__init__` of the same record.
It passes at d1950c0c and was handed to that thread. The gate failure
names are identical to ae46579's in all four parts (aa 77, ab 7, ac 25,
rec 20, llvmlite installed). Fuzz at `FUZZ_EXT=4`, 60 programs each: seed
99 was 54 PASS, 5 LOUD, 1 WRONG (p99_24, F5) and is now 55 PASS, 5 LOUD,
0 WRONG; seed 100 is 54 PASS, 6 LOUD, 0 WRONG.

At 48607a8 (the fix rebased onto main caf2e77): the typed corpus is 327/327
PASS, and main caf2e77 is 323/323. The gate failure names are identical to
caf2e77's in all four parts (aa 96, ab 20, ac 56, rec 39 in the cloud
container without llvmlite). The final rebase onto dd340a1 had no conflicts.

Before that, at 2aeab20 on ef49715: the typed corpus was 292/292 PASS
(main ef49715 288/288 with the harness fix). Fuzz seed 97 at
`FUZZ_EXT=4`, 60 programs: 56 PASS, 4 LOUD, 0 WRONG, 0 TYPE. The focused
tests (project product, reducer, precompile, sequence tables, sequence
contracts, loop composer) had the same 26 failures as main, and so did the
gate (aa 96, ab 21, ac 57, rec 40).

AbstractTensor, from a scratch copy with the autograd branch cut from
`__init__` (199 methods, catalogue `--emit-native --jobs 4`, torch installed),
at 8c6ae0d on dd340a1:

- 75 complete, 72 partial, 52 failed.
- Native: 70 shared libraries linked. Catalogue verification reports
  "unsupported" for record ABIs. Five native failures remain: `get_tensor`
  and `to_dtype` have no Fortran expression, one dangling SSA operand at a
  loop exit, and `matmul` given an operand of rank above 2. A run before
  #839 also failed 58 units at the link, because `-static-libgfortran`
  cannot go into a Linux shared object; #839 links it dynamically.
- Partial: mostly `missing_used_parameters` / `unresolved_required_source_values`.
- Failed: API type expectations with no native ABI for unannotated
  parameters (`self`, `idx`), `_pre_autograd`'s closure, and a
  source-value-class disagreement in `compression.coefficient_events`.

## Open frontier

- A field of an ordinary class written in a loop is refused (F4). It needs
  the entry value from the class's `__init__` frame instead of a fold of the
  call's arguments.
- A computed `__init__` field (no field receipt) is refused at
  construction (F3). It needs a receipt from the initializer's type.
- A field read after a call that stores into the record is refused (F3). It
  needs the call's store reloaded into the caller's field state.
- `compile_class_to_dll` remains the deprecated adapter with the field
  lowering above; class DLLs should come from the catalogue.
