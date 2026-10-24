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
static receiver's assigned field. The reducer merges no field state at a
loop (plan 70, section 1.3, held for the user). So a field that an enclosing
loop body rewrites, or that is read after such a loop, is refused by name:
`local record field written in a loop has no merged value ...
(loop_exit_field_state_unmerged)`. It used to return the constructor
argument.

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

## Measured

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

- A field of a local record written in a loop is refused (F1); it needs the
  loop field-state merge that plan 70 holds for the user.
- A computed `__init__` field (no field receipt) is refused at
  construction (F3). It needs a receipt from the initializer's type.
- A field read after a call that stores into the record is refused (F3). It
  needs the call's store reloaded into the caller's field state.
- `compile_class_to_dll` remains the deprecated adapter with the field
  lowering above; class DLLs should come from the catalogue.
