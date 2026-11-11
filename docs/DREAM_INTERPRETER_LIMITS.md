# Dream interpreter: measured limits (2026-11-11)

Scope: `src/compiler/dream_document.py` (`parse_dream_document`,
`compile_sections`, `lower_to_ssa`, `DreamRuntime`, the HTML shell) and the
front ends it dispatches to through `DREAM_LANGUAGE_TRANSLATIONS`. Measured
on branch `claude/repo-world-map` at a worktree of main `a74b3bf5`, Python
3.11.7, pycparser 3.00, sympy 1.13.1, node 22.16. Every verdict below was
observed by running a block through `compile_sections`; nothing is taken
from the handoff prose. Tests pinning these facts:
`tests/test_dream_interpreter_limits.py`.

## What the interpreter is

A `.dream` file is byte-scanned into sentinel blocks before any language
parser runs. Each block names a language; `compile_sections` picks the
route from `DREAM_LANGUAGE_TRANSLATIONS` and builds one graph per block.
`lower_to_ssa` lowers every block to one SSA call and publishes parallel
and shader deployment regions. `DreamRuntime` executes blocks through
caller-supplied handlers. These three stages have different reach, and the
table below keeps them apart.

| stage | reach today |
|---|---|
| framing (`parse_dream_document`) | complete for the v1 sentinel grammar: segment, shader, parallel; encodings; sha256 sentinel; unique ids; parallel membership. Fails closed on unframed bytes. |
| per-block graph (`compile_sections`) | route-dependent, see below |
| deployment IR (`lower_to_ssa`) | complete for what it claims: one `Call` per block, `independent_lanes` region per parallel group, `device_dispatch` region per shader. The call bodies are placeholders (`__dream_<lang>_<stage>__`); no block body is lowered into SSA. |
| execution (`DreamRuntime`) | host-supplied handlers only. Python runs through `exec` when the host opts in; JavaScript is deferred to a browser; shaders go to a host deployer. Nothing is executed by the compiler. |
| HTML shell | emits a launchable page handing display ownership to the document's own controller; carries section graphs and deployment regions as data. |

## Per-route reach

### python → `ast` route (ProcessGraph.build_from_ast + topological reducer)

Admitted, each measured as a resolved function graph: functions, `if`,
`for`/`range`, `while`/`break`, classes with methods, inheritance with
`super()`, generators (`yield`), list comprehensions, lambdas, closures,
decorators, walrus, dict literals, `async def`, `import`, `global`, `match`.

Refused, by the ingestion's own declared gap (`native_exception_control_gap`
in `source_control_ingestion.py`): `raise`, `try`/`except`
(`UnadmittedSourceControlTopology`), and `with`
(`MissingResourceControlTopology`). This is the whole of the Python route's
refusal set as measured. The shipped `examples/torture_five_languages.dream`
Python block fails on its `raise NotImplementedError`; that example has been
red on this point since the gap was declared, and was not remeasured here
against any earlier commit.

Repaired here: a `with` refusal left rows in the one process-global detached
identity book, and every later Python block in the same process then failed
with a `native_exception_control_gap disagreement` it had not caused.
`compile_sections` now opens and closes an identity book per block with the
existing `begin_identity_book`/`end_identity_book`, as the source compiler
does. `raise` and `try` refusals never poisoned later blocks; only `with` did.

### cpp → `cpp-shell` route (desugar → pycparser → ProcessGraph.build_graph)

The shell is the narrow C++-like scope of `CPP_LIKE_SHELL_FOR_C_INTENT.md`:
classes with fields and methods, a constructor, `obj.method()` call sites,
single inheritance. Refused by declaration: templates, operator overloading,
virtual, exceptions, namespaces, multiple inheritance.

Repaired here, in the desugarer:
- `#include` lines are dropped before parsing, the same massaging
  `c_function_token_multigraph` already does. Any other directive
  (`#define`, …) is refused as `CppShellUnsupported`, never silently
  dropped.
- `ClassName obj(args);` is rewritten to `ClassName obj = ClassName__new(args);`.
  Constructors were in scope but their call site reached pycparser raw.
- A receiver whose class is unknown is left as written. An operator
  precedence slip emitted `None__method(&obj)` for it before.

Where the route stops today, for every C block that parses: the reducer's
operand-position gate. Chain, observed:
`dream_document._compile_section` → `graph_express2.build_graph:6319` →
`stage_edge:6297` → `commit_run:6291` → `_commit_edges:6195` →
`topological_reducer._set_operands:3697`, raising
`identity_transition: no source cell for node … the operand rewrite was made
with no operand-position scope`. The gate arrived with `8af44b07`
(2026-10-17, "a post derives from its cells or is refused");
`tests/test_dream_cpp_pipeline.py` was last touched 2026-08-06, is not in
the known-failing list, and fails at this gate on main. The C route needs
its edge commits to carry an operand-position scope; that is the
operand-position-writer migration, not a dream-document change, and it is
not attempted here.

Still outside the shell, measured on the shipped
`examples/mixed_python_cpp_counters.dream`: `std::string`,
`std::unordered_map`, references, `new`, lambdas, `auto`. That example
reaches pycparser now and fails there ("Invalid specifier list").

Line endings: pycparser refuses a carriage return. An autocrlf checkout
turned every `.dream` example into CRLF, changed every block's sha256
`content_hash` (the card `cache_key`), and failed the C++ pipeline test on
`'\r'` before its real failure. `.gitattributes` now declares `*.dream text
eol=lf`, the repository's existing mechanism for exactly this (see the
gate-script receipt above it).

### javascript → `javascript-ast` route (acorn via node → declaration graph)

Reach: module-level declarations (function, class, variable) become nodes;
an edge records that one declaration's body names another. Requires `node`
on PATH; fails closed with the acorn message on a syntax error. Top-level
`await` parses (module source type).

Repaired here: `export function`, `export const`, `export default class`
were invisible (an exported block had no nodes); destructuring declarators
(`const {a, b: c} = …`, `let [d, ...rest] = …`) bound no names. Both are
recognised now.

Limits by design, unchanged: statements are not graphed (a block of only
calls has an empty graph); nested functions are not nodes; an `import`
binding is not a declaration of this block (its name appears nowhere); no
control, data or call graph exists for JavaScript here, only the
declaration dependency graph. `javascript_process_graph.py` is a separate,
expression-level front end not used by this route.

### glsl → `glsl-ssa` route (`lower_glsl_source_to_ssa`)

Reach, measured with zero shortfalls: straight-line scalar arithmetic inside
`void main()` writing declared `out` variables, from `uniform` inputs:
binary/unary operators, ternaries, casts (`int`, `float`, …), the direct
calls in `GLSL_DIRECT_CALLS` (`abs acos … sqrt tan tanh trunc`), and the
decomposed conveniences `mix`, `clamp`, `smoothstep`, `step`; several
outputs.

Shortfalls (the block still produces a graph, the shortfall list says what
was omitted): any control flow (`if`, `for`), user helper functions
(`unsupported GLSL call 'twice'`), vector types and swizzles (`vec3(...)`,
`p.x`), builtins (`gl_FragCoord`, `gl_GlobalInvocationID`), storage buffers
and indexing (`values[i]`), `fract`, `texture*` and derivatives
(`WEBGL_UNLOWERED_CALLS`), and a non-`void main` entry
(`GLSL_ENTRY: no void entry function`). The shipped compute shaders are
therefore mostly shortfall: `register-light-compute` has 17, `glsl-torture`
4. The module header already records the memory-specialisation TODO that
gates storage operations.

### wgsl → `source` route

No parser. The block becomes a `SourceSection` with one `SourceLine` node
per line and `next` edges, `executable: True` because the shader device
runs it, and no shortfall for the absence of a graph. This is the same
fallback an unknown language (`language = java`) receives, except that an
unknown language also gets the "no executable Dream shell artifact"
shortfall.

### sympy → `sympy` route

`sympy.sympify(payload, evaluate=False)` then `ingest_sympy_expression(...,
strict=True)`. Polynomials, trig, `Matrix(...).det()`, `Sum`, `Derivative`,
`Integral` and unknown function heads all ingest. A syntax error raises
`SympifyError` out of `compile_sections`.

## Cross-route facts

- Routes are independent per block. Nothing joins a Python block's graph to
  a C block's graph; `inputs`/`outputs` names on the sentinel become SSA
  call arguments by position, with dtype from the `dtype` decoration, and
  are not checked against the block's source.
- `compile_sections` is all-or-nothing: the first block that raises aborts
  the document. Shortfalls are per block and visible; refusals are
  exceptions.
- The `source` route is the only route that never fails; it is also the one
  that knows nothing.

## Natural-language stand-ins (owner's direction, 2026-11-11, not built)

The owner's intent: a dream block may be a natural-language description
standing in for code. It is irreducible, so it goes to an open work table.
When someone writes the code from the description, the stand-in is
satisfied; the code has an author; the document becomes a tree whose child
of the description is that contribution. This feeds back to the room system:
in both the filesystem map and the code map the stand-in already owns a room,
claimed but unaltered, so ownership belongs to the author from the moment
the description exists.

What exists for it today: the `source` route keeps any undeclared language
as an irreducible line graph with a visible shortfall, so a
`language = prose` block already survives parsing and lands in the section
graphs with its text intact. The world scan (`repo_world_scan.py`) mints a
room per file and the scan's `languages` table says which languages have no
reader. What does not exist: a declared `prose` language in
`DREAM_LANGUAGE_TRANSLATIONS`; a work-table record (identity, description
block, state open/satisfied, author, satisfying block); the satisfaction edge
from description to contribution and the tree it implies; and the room-claim
record that says a room is owned by an author before its content exists.
None of these are started here; the limits above come first, and the
portal/room records in `REPO_WORLD_MAP_DESIGN.md` are where the claim would
be declared.

## Status

Green: `tests/test_dream_interpreter_limits.py` (8), `tests/test_dream_document.py`
(16), `tests/test_cpp_shell_desugar.py` (7). Red on main and here:
`tests/test_dream_cpp_pipeline.py` at the operand-position gate, as
described. Not run: the shell emission against a browser, any shader
deployer, the full suite.
