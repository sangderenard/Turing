# 38. Whole-class trial: the SSA interpreter (2026-10-24)

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
