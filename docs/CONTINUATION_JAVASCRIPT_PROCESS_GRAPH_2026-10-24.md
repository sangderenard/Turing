# JavaScript ProcessGraph continuation — 2026-10-24

## Current result

Turing now has one JavaScript-specific semantic entry from source text to an
identity-backed ProcessGraph:

```python
ingest_javascript_source(
    graph, source, owner=filename, entry=function_name,
    assumptions=proved_javascript_facets,
)
```

The route uses the vendored Acorn parser, recursively discovers a selected
function (including a function nested inside another archive function), and
interprets the supported region into the existing ProcessGraph/Handler
vocabulary.  Expressions and whole functions use the same interpreter.

JavaScript source spellings, names, and identity-list entries intentionally
remain strings.  No JavaScript-specific token system was introduced.  The
graph uses the repository's normal ingestion cells, `_set_operands` edge
writer, and downstream canonical lowering; that existing lowering remains the
owner of structural tokenization and `ssa_identity_tokens`.

No generic compiler entrypoint, Handler member, or compiler-core lowering was
changed.

## Files

- `src/compiler/javascript_process_graph.py`
  - source/program/expression entrypoints;
  - one conservative acyclic function interpreter;
  - lexical binding records and identifier occurrences;
  - exact JavaScript Number falsiness (`x === 0 || Number.isNaN(x)`);
  - realm-shadow and call-arity refusal;
  - existing tensor-call attributes for Math identities.
- `src/compiler/javascript_source_tables.py`
  - lexical operator selection;
  - explicit shortfalls where no exact current operation/program exists.
- `src/transmogrifier/graph/javascript_identity_programs.py`
  - Math/Number replacement identities;
  - Array/DataView object boundaries;
  - browser host boundaries.
- `src/transmogrifier/graph/oop_language_translations.py`
  - stable descriptive ESTree graph construction;
  - JavaScript operator strings are node data, not child nodes;
  - no unproved global registration of ESTree functions as runnable compiler
    definitions.
- `tests/test_javascript_process_graph.py`
- `tests/test_oop_language_translations.py`

## Exact supported semantic region

- one selected `FunctionDeclaration`, `FunctionExpression`, or arrow function;
- nested function discovery by the requested authored name;
- identifier parameters;
- initialized `const` declarations;
- numeric literals and Boolean literals;
- proved Number arithmetic and comparisons;
- proved Boolean `&&`, `||`, and `!`;
- strict equality for proved same-type primitive operands;
- conditional expressions with same-kind arms;
- pure declared Math/Number identities at exact arity;
- early-return `if` without `else`, when both paths return values;
- `Ret` publication through the ordinary graph edge path.

The archive fixture
`archives/fit-datastudio-js-2023.zip::eclipse-workspace/DataProcessor/Functions.js`
is exercised directly when present.  Its nested `mapValue` function reaches a
single ProcessGraph region.  The equal-bound case is included so the test
covers the NaN truthiness difference that a bare repository `LNot` would get
wrong.

## Deliberate shortfalls

The route reports, rather than approximates:

- `let`, `var`, assignment, branch mutation, Phi/CFG merges, and loops;
- destructuring, defaults, rest parameters, arrays, objects, and classes;
- JavaScript strings, `null`, and `undefined` as runtime values;
- loose equality;
- remainder, shifts, and bitwise operators pending explicit ECMAScript
  coercion programs;
- `Math.round`, whose ECMAScript tie rule is not the repository round-even
  identity;
- shadowed `Math`/`Number` roots;
- unsupported Math arities (including variadic `Math.min`/`Math.max` cases);
- Array/DataView/browser-host execution.  Those identities are inventoried as
  object/host boundaries but are not falsely emitted as numeric operators.

The result is a ProcessGraph region and lowers through
`process_graph_to_ssa_instrs`; it is not yet a whole repository `IRModule`
function ABI.  The next source-frontend increment should connect parameter
Inputs and returned values to the existing function/ProgramABI route without
adding a JavaScript-only ABI or bypassing the graph interpreter.

## Verification receipt

Focused:

```text
python -m pytest tests/test_javascript_process_graph.py tests/test_oop_language_translations.py -q --tb=short --show-capture=no
53 passed in 3.90s
```

Broader JavaScript/backend/identity batch:

```text
python -m pytest tests/test_javascript_process_graph.py tests/test_oop_language_translations.py tests/test_cpp_shell_desugar.py tests/test_python_identity_programs.py tests/test_backend_identities.py tests/test_ssa_javascript_backend.py -q --tb=short --show-capture=no
91 passed, 1 warning in 7.15s
```

Concordance:

```text
python -u tools/audit_identity_concordance.py view
identity concordance: 496 rows across 24 functions, 0 findings
unsourced: 0 facts, 0 identities
```

`git diff --check` is clean apart from Git's existing LF-to-CRLF notices.
The full suite was not run because `TEST_BASELINE_AND_HAZARDS.md` records that
it does not finish and has no per-test timeout.

## Separate existing Dream frontier

The older `dream_document._javascript_dependency_graph` is a shallow
top-level declaration-reference graph, not this semantic interpreter.  A
broader diagnostic run observed that it still inserts string node IDs into an
integer-only memory graph.  This work did not modify or route around Dream;
that separate JS-specific caller should eventually consume the semantic entry
above or adopt the memory graph's integer identity contract.
