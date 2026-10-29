# Continuation: bootstrap stops (thread "Push the bootstrap past its stops"), 2026-10-29

Parked on Albert's order (2026-10-29 19:01 UTC). This file is the hand-off.
Repro drivers referred to below are in the project folder
`/mnt/project-files/bootstrap-stops/repros/`; their full text is also
quoted here where it matters.

## Standing rules this work runs under

- Fix each bootstrap stop at its writer; one small ready PR from main per
  fix, each with a corpus program. Never bypass a stop. Never run the gate
  or the full corpus for a small step.
- AGENTS.md, repeated by Albert: no alteration in compiling that does not go
  through the concordance; no sort, tolerance, fallback or "repair" where a
  record already has the answer; no design choices on your own authority. If
  a reader needs a record nobody wrote, fix the writer or refuse by name.
- Native only; no live Python host callback unless whitelisted in the
  contract. pickle is replaced by the AbstractTensor codec
  (`src/common/tensors/compression/octet_codec.py`).

## Merged today from this thread

| PR | What it did |
| --- | --- |
| #1172 | Annotated dict filled in a loop gets its own data arena (`precompile_to_ssa._sequence_descriptor` aliasing). c942 |
| #1173 | Tuple key width read from the key, not the value position (`fortran_c_shell._field_slot_ops`). c943 |
| #1180 | Callsite boundary keeps its runtime rank/metadata_state on an untyped formal (`_tensor_descriptor_rule`). Codec stop 3 |
| #1181 | Planner join typing dict-loop targets. SUPERSEDED by #1189 (it violated the concordance rule) |
| #1189 | Dict-loop target classes posted by the class fixed point (`topological_reducer._project_iterated_mapping_key_classes`), from the class rows of the key operands every store / displayed key writes. c944, c945 |

Corpus numbers: the bootstrap block c930-c949 is fully used (c946-c949 were
drafted for a change that was deleted, see below, and are free again in
practice but were never committed). Next free number per memory: c960.

## Open item 1 (the bootstrap's current stop): `Any`-typed tuple rows

Full bootstrap (`PYTHONPATH=<tree> python tools/bootstrap_compiler_concordance.py --output <dir> --jobs 3`)
on main after #1189 stops at:

```
unit 076 | src.compiler.identity_spill._reindex_in_place | ssa-lowering |
ValueError: region value 2 (input 'row' in function '_reindex_in_place') has no
settled dtype ...
```

`_reindex_in_place` runs `for row, column in state["cells"]` over
`IdentityPage.cells: dict[tuple[Any, int], Any]`. `row` is `Any`: it is the
`row` formal of `IdentityPage._stamp` (identity_concordance.py ~6560,
`key = (row, column)`; `self.cells[key] = fact`), which every
`IdentityBook.post(...)` in the compiler fills with tuples of many shapes.

What the rules say: Albert's called-inventory rule (2026-10-28 11:24-11:27)
makes the class of such a formal the union of the classes posted by every
call record reaching it, keeping each exact type. What is missing:

1. No writer collects that inventory into a dict-key position (the `row`
   formal of `_stamp`, then the key column of `IdentityPage.cells`).
2. No native representation of a union of tuple shapes used as a dict key
   (docs/UNION_TYPE_DESIGN_2026-09-29.md covers scalar/span/sequence/record
   arms, not this).

Repro: `repros/reindex.py` (a small copy of `_reindex_in_place` with a
`Page` dataclass). Lower it with `repros/lower.py <driver> root` (it imports
the string-core harness `diffrun` from
`/mnt/project-files/string-core-correctness`). diffrun caches by driver text
and Turing HEAD, so append a random comment to a driver to defeat the cache
after editing the working tree.

This needs Albert's direction before any representation is built; do not
pick a design.

## Open item 2: a whole tuple as a loop variable

Not lowered on main, even when annotated:

```python
def root(n: int) -> int:          # repros/m12t.py (annotated), m12.py (not)
    ps: list[tuple[int, int]] = []
    for i in range(n):
        ps.append((i, n - i))
    total = 0
    for p in ps:
        total += p[0] * p[1]       # region value (Indexed) has no settled dtype
    return total
```

Also `for key in cells` over a tuple-keyed dict followed by `key[1]` or
`cells[key]` (`repros/m4f.py`, `repros/m6u.py`), and `for a, b in ps` over a
list grown by appended pairs (`repros/m14.py`).

A version that rewrote the loop variable into a Tuple node in the graph
worked but was DELETED on Albert's order: it altered the graph outside the
concordance and was a design of my own. The existing record for a whole row
of a fixed-width tuple sequence is the row handle
(`glsl_deployment_strategy._decide_value_shape` / `_fixed_width_row_width`:
a row read `fingerprints[i]` is an `opaque_ref` handle read only by literal
column loads, lowered by `fortran_c_shell._sequence_row_operations`). A
loop target is not covered: `_sequence_row_operations` needs an explicit
row-index node, and a loop target's row index is the loop induction
(bound in `loop_composer` projected_iterable_bindings). Extending that record
to loop targets is the candidate; whether that is acceptable is Albert's
call. `cells[key]` with a whole-row key is not a literal column read and
is refused by name under that record.

## Open item 3: wrong record in the reducer's `.items()` pair projection

`topological_reducer` (the `mapping_layout` block after
`concord_mapping_target_member`, ~17500) projects the `.items()` key/value
pair onto ANY two-name target over a mapping, including
`for row, column in cells` with no `.items()`. For an annotated
`dict[tuple[int, int], float]` that gives `column` the value class. Fix at
the writer: gate the pair projection on the iterator being `.items()`.
`repros/c960_annotated_tuple_keyed_dict_unpacked.py` was drafted to show it,
but on main it stops earlier with an unrelated final-concordance refusal
(`value 4: roles=[(4, 'handle'), (4, 'column0'), (9, 'column0')]` and an
undefined column1 value): an annotated tuple-keyed dict filled in a loop
and iterated with a tuple target. That refusal is its own open stop.

## Open item 4: codec (pickle replacement) compile and benchmark

- Driver: `repros/codec_roundtrip.py` (compress/decompress round trip over
  generated text, five sizes). Last run refused at lowering with
  `AttributeError: 'NoneType' object has no attribute 'Tensor'` (stop 4,
  raise handling inside the codec).
- Raise handling was #1166; Albert reports #1166 is ABANDONED (2026-10-29
  19:01). Raise handling is therefore open with no owner.
- After it compiles: apples-to-apples compiled benchmark against zlib
  (Albert: a 50% ratio is fine if fast, 75% if slow).
- Then Albert's plan (this thread, 2026-10-29): once the codec compiles, use
  the existing compile-told region-marking process so callers use the native
  codec, while compiling the codec itself uses its source.
- Then swap pickle calls in ingested programs for the codec automatically
  (whole-class replacement, like the other swaps).
- `identity_spill`'s whole-page spill still pickles (follow-up).

## Other notes

- #1176 (another thread's record-slot dtype fix) was checked against
  `tests/test_host_callback_fields.py`: 7 passed, 7 errors, identical to main
  (all errors are one `control_block_placement` refusal on `Walker.run|fork`).
- `tests/test_callsite_formal_shape_concordance.py` has 9 failures on main
  that predate this thread's work.
