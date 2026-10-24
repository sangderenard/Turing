# 39. Whole-class trial: the SSA interpreter (2026-10-24)

Albert asked for a whole-class compile of the Python SSA interpreter as a
standing source of frontier errors, the way AbstractTensor's whole-class trial
is one.  The interpreter is `src/compiler/ssa_reference_evaluator.py:
SSAReferenceEvaluator`, the only Python interpreter for repository SSA
(`tools/blas_backend_bakeoff.py` calls it "Repository SSA interpreter").

## How to run it

```
python -u -m tools.trial_whole_class_compile \
    --source src/compiler/ssa_reference_evaluator.py \
    --class SSAReferenceEvaluator --output out/ssa_interpreter_class --catalogue
```

The tool works for any class.  It compiles the class twice and writes
`summary.json`:

1. **Class route** (`compile_class_to_dll`, with the program extraction
   contract and the module's package).  It captures the whole-object SSA
   module and reports each method as `present`, `missing` (a false-acceptance
   candidate) or `unreached` (the route refused before it produced a module).
   Presence is read from each function's `source_function_reference` /
   `source_qualified_name` records, never from a symbol spelling.
2. **Per-method catalogue** (`tools.compile_project_catalogue`), so every
   method reports its own first refusal.

It takes about 30 s for the class route and about 60 s for the catalogue.

## Finding 1 (fixed): the class route compiled `__init__` alone

Observed on `main` at 6058d24, with gfortran installed:
`compile_section_to_dll src/compiler/ssa_reference_evaluator.py
SSAReferenceEvaluator` succeeded and produced `SSAReferenceEvaluator.so`.
Its exports (`nm -D`) were `SSAReferenceEvaluator____init__`, its two planned
regions and its sequence-append helpers.  `run`, `_execute`, `_step`, `_call`,
`_external_call`, `_value`, `_operand` and `_operation_name` were absent, and
nothing refused them.  A two-method class (`Acc.__init__`, `Acc.add`) loses
`add` the same way.  This also applies to every earlier AbstractTensor
class-route run (inferred: same route, same seeds).

Two records disagreed:

- `compile_ast_aot` pursued `entry + dependency_seeds` (`pursuit_roots`) but
  named only the entry as an activation root (`activation_roots=targets`,
  `aot_compile.py`).  The deployment's `activation_root_references` therefore
  held one function, and the planner made shells for `__init__`'s tree only.
  Fix: the seeds are activation roots too.  Only the class route passes
  `dependency_seeds`, so no other caller changes.
- With the seeds activated, the planned shells exist but have no
  `shell_control_program`.  Watched with a `_walk_planned_shells` hook: the
  structural-preparation walk runs while callsite planning is still deferred
  (`activation_root_references == ()`), so it prepares only `__init__`.  The
  seeded shells are planned afterwards and never prepared.
  `_class_surface_ssa_program` then passes over them (`if control is None:
  continue`).  Not fixed here (it is the next item below).  Made loud
  instead: `_emit_class_surface_module` reads the deployment's activation roots
  against each lowered function's `source_function_reference` and refuses by
  name ("class surface planned activation roots it never lowered: 1 'add'").

Pinned by `tests/test_whole_class_compile_trial.py`.

## Triage of the refusals (after the fix)

The class route now stops at the first refusal of family B.  The catalogue
gives each method's own refusal.

| # | Family (record that disagrees) | Units | Silent-capable | Hypothesis (inferred) |
| --- | --- | --- | --- | --- |
| A | Planned activation shells without a control program are passed over by the class surface | class route: every method but `__init__` | **yes** (now loud by the guard; the dropped shell itself is unfixed) | Structural preparation (`glsl_deployment_strategy`, the `_walk_planned_shells` call before "aot: preparing N planned shell(s)") runs before deferred callsite planning adds the seeded roots; nothing prepares the later shells |
| B | `typing.Any` is read as a record identity | class route; `__init__`, `run`, `_operand`, `_step`, `_value` | no | The keyed-value annotation contract (`fortran_c_shell.py`, `value_record = str(value_name)` when the value name is not a scalar) treats every other name as a record, so `Mapping[int, Any]` gives `value_record='Any'` and `module: Any` gives "unknown nested program ABI record 'Any'".  The program ABI has no `Any` record.  What `Any` should lower to (a value union over the values the interpreter stores) is a frontier decision |
| C | `class_field_declaration` for `history`: `annotation=''` recorded, `dict[int, list[Any]]` proposed | `run` (catalogue) | no | `__init__` declares `self.history: dict[int, list[Any]] = {}`; `run` reassigns it without an annotation.  A survey of a projected ClassDef posted the field from the unannotated assignment first (`graph_express2.py`, CLASS_FIELD_DECLARATION post).  An unannotated re-assignment is not a new declaration |
| D | Loop state transition not modelled: receiver class of `self._step(...)` unresolved | `_execute` | no (refused by the loop rule) | In a per-method unit `self` carries no class record, so the method call is an opaque effect.  The class route should supply the record (Albert: methods lower as records); expect this to clear once A is fixed |
| E | Loop iterable `zip(callee.args, instruction.args)` is `closure_aggregate` | `_call` | no | `zip` over two attribute sequences of `Any`-typed objects has no iterable record; depends on B |
| F | `cannot pickle '_struct.Struct'` at compilation-units | `_external_call` | no (a crash, not a refusal) | The unit pulls `shell_external_references` (module-level `struct.Struct` constants) into a graph that the compilation-unit plan pickles.  A crash should become a named refusal or the constant should be a contract boundary |
| G | Nested closure `service` has missing lexical captures (`host`, `identity`, `self`, `ExternalReferenceRequestRecord`) and an unused parameter `operation` | `_external_call.<locals>.service` (partial) | no | Closure captures across a nested def; overlaps the closures-as-values work (#831) |
| H | `_operation_name`: parameter `instruction` unused, one unmaterialized extraction boundary | partial | no | `(instruction.attributes or {}).get(...)` on an `Any` object; depends on B |
| I | `_cast_to`: unresolved program ABI reference `{field: '', parameter: ''}` | partial | no | A reference row with empty identity: a record was written without its names.  Worth a writer trap before anything else in this row |

False-acceptance check: no unit was admitted (no catalogue unit is linkable,
none emitted native).  The only false acceptance is finding 1.

Ranking (silent-capable first, then blast radius times depth): A, B, C, F,
D/E/H (expected to move with A and B), G, I.

## Side finding

A minimal class with a scalar field (`self.n = n`) fails Fortran emission:
`record 'Toy' field 'n' names absent record 0 in 'Toy____init__'`.  The record
table gives the int field `n` a child `record_id` of 0, and no record 0
exists.  Inferred: a falsy-vs-None mix-up in the writer of the field's
`record_id`.  Albert's rule applies (fields lower as records), so this is a
defect, not new work.

## Not measured

The gate was not run in the cloud (it adds about 70 names even on `main`).
Both changes are on the library route only.

## Follow-up: family A fixed (seeded methods get a body)

`compile_ast_aot` called `prepare_graph_precompile` on the entry's shell
alone.  Preparation walks only that shell's callsite tree, so every other
activation root reached the whole-object lowering with no control program
(and since #835, was refused by name).  Each activation root's own shell is
now prepared.  `Acc.add` (a two-method class) compiles and returns `a + b`
natively, the same as Python (`tests/test_whole_class_compile_trial.py`).

The interpreter's class route now reaches `_execute`'s loop and stops at
family D, in a narrower form: `_step` resolves now, and the one opaque effect
left is `self.trace(...)`.  That is a call through an `Optional[Callable]`
field, which has no modelled state transition.  New row: a call through a
callable-valued field.

Corpus 323/323 PASS, fuzz seed 48 30/30 PASS, cloud gate aa 89 / ab 7 /
ac 26 / rec 20 (exactly the baselines).

## Follow-up: family B fixed (`Any` is not a record name)

The keyed-value annotation contract (`_authored_annotation_field_receipt`,
the mapping branch) mapped every non-scalar value name to a record identity.
So `Mapping[int, Any]` declared `value_record='Any'`, a record that no
declaration defines.  `Any` and `object` now take the repository's existing
arbitrary-object boundary (`opaque_ref`), as they already do for parameters,
record fields and collection members.  Pinned by
`tests/test_any_mapping_value_receipt.py`.

Trial after the fix (catalogue): the 'Any' record refusal is gone from
`__init__`, `run`, `_operand`, `_step` and `_value`.

- `__init__`, `_operand` and `_value` are now partial (unresolved program-ABI
  references for the `Any` parameters, and unused parameters).
- `_step` refuses at an f-string: "formatted glyph component has no resident
  producer".
- The class route now stops at "sequence contract concordance disagreement for
  '__init__' value 1": duplicates/1 committed, unique/2 declared.  Inferred:
  `frozenset(int(value) for value in history)` is declared as a unique
  2-column row against its committed duplicates/1 layout.

Not done: a native value check.  A dict display passed to an annotated dict
parameter refuses on main whatever the value type ("no proven capacity").
That is Lane C's frontier, so no corpus program can hold this yet.

Corpus 323/323 PASS, fuzz seed 49 30/30 PASS, cloud gate aa 89 / ab 7 /
ac 26 / rec 20 (exactly the baselines).

## Follow-up: a call through a stored callable field (loud, not lowered)

`self.trace(...)` in `_execute` refuses loudly inside the loop (family D,
"opaque-state-effect").  Out of a loop it did not.  A reduced class:

```python
class Walker:
    def __init__(self, trace: Callable[[int], None]) -> None:
        self.trace = trace
    def poke(self, n: int) -> int:
        self.trace(n)
        return n + 1
```

shipped `Walker.so` with `poke` returning `n + 1` and never calling `trace`.
That was a silent wrong answer: `trace` also spells a tensor operator, so the
unconsumed call was pruned by `_remove_dead_metadata_nodes` as a dead pure
tensor call.  Spelled `self.notify(n)`, the call reached the class surface and
was dropped there without a word.

Fix: `_stored_callable_field_call` reads the method graph's own records
(`method_binding`, `method_owner`, `class_table` fields minus methods, and
the receiver Input bound to the first positional parameter).  Such a call is
an effect, never dead metadata, and the class surface refuses it by
`Owner.method: expression (node N)` until it has a lowered invocation.  Both
spellings are pinned in `tests/test_whole_class_compile_trial.py`.

Not done (Albert's call): the real lowering, a native-to-host callback through
an opaque handle.  The refusal names every site that needs it.

## Follow-up: two per-method units reach their real refusals

- `run` refused at source normalisation: "class_field_declaration
  disagreement for ... 'history'".  A method's unit partitions the class to
  that method alone, so run's unannotated `self.history = {...}` declared
  the field with no annotation, and the unit's declared complete source then
  posted `__init__`'s `dict[int, list[Any]]`.  A view whose declaring
  statements are a strict subset of the other's now reads (or is revised
  to) the fuller declaration.  Two different annotations still disagree
  (`tests/test_partition_field_declaration_view.py`).  `run` now stops at
  "sequence row dtype concordance disagreement ... value 19: ('int64',
  'int64') vs table store ('int64', 'opaque_ref')", the
  `{int(v): [] for v in ...}` dict (inferred).
- `_external_call` crashed at worker hand-off: "cannot pickle
  '_struct.Struct' object".  Resolved-graph checkpoints now restore a Struct
  by its format.  The unit now refuses by name: a loop over a closure
  aggregate, the same as `_call`.

Also found, handed to the silent lane: `len(set(v for v in xs))` and
`len(frozenset(xs))` compile and return 0 (repros c470-c472).

Corpus 325/325 PASS, fuzz seed 54 30/30 PASS, cloud gate aa 89 / ab 7 /
ac 26 / rec 20 (exactly the baselines).

## Follow-up: `run`'s dict value type, then its keyed field

- `values: dict[int, Any] = {}` declared its value column as the int64 handle
  of a record named 'Any' (`_annotation_storage` in the topological
  reducer), so storing `arguments[key]` refused as a row dtype
  disagreement.  A mapping's `Any`/`object` value is now `opaque_ref`.  An
  `Any` key, and `Any` inside a nested aggregate, keep their integer handle
  (five gate tests pin that).
- `run` then crashed in `resolve_keyed_mapping_iterables` with the raw
  invariant "heap sequence length/capacity must be distinct".  The keyed
  field `_root_frame` is published as the caller-supplied exactly-full view
  (length and capacity are one part), and `run` assigns it a dict it then
  grows.  That case now refuses by field.  Real support would give the field
  its own capacity part.

Pinned by `tests/test_any_valued_local_mapping_row.py`.  Corpus 329/329
PASS, fuzz seed 56 30/30 PASS, cloud gate aa 89 / ab 7 / ac 26 / rec 20
(exactly the baselines).

## Where the remaining interpreter blockers converge (inferred, 2026-10-24 09:10)

Every per-method refusal left traces to an `Any`-typed object.
`function: Any`, `instruction: Any` and `module: Any` are really
`transmogrifier.ssa.Function`, `Instr` and the module.

- `_step`: "formatted glyph component has no resident producer: value=333".
  Value 333 is `{function.name}` in the no-semantics error.  `name` is an
  attribute of an opaque `Any`, so native code has nothing to format.
- `_call` and `_external_call`: "iterable-access=closure_aggregate".
  Both are loops over `instruction.args`, an attribute of an opaque `Any`.
- `_execute`: `self.trace(...)` (the stored callable) and `self._step(...)`.
  The second is opaque only in a per-method unit, where the receiver's class
  is unresolved.
- `run`: the keyed field `_root_frame` is given a dict the frame grows.  The
  field view has no capacity part, and real support needs the object to own
  that heap.

So the trial's limit is the interpreter's own `Any` annotations, not one
missing compiler case.  Annotating them with the real dataclasses would let
`op`, `args`, `res` and `name` lower as record fields.  `attributes` and
`metadata` stay `Dict[str, Any]`, so the attribute reads still need opaque
handles or host calls.  Not measured.
