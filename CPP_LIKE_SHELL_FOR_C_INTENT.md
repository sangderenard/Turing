# A C++-like shell desugaring to real C: statement of intent

**Status: intent, not yet implemented.** Written before writing any code,
per the project's own established practice this round of work (see
`GRAPH_DESCRIPTION_LAYER_SURVEY.md`) of documenting direction before
building, and correcting the document rather than leaving a stale claim in
place once real code contradicts it.

## Why this, and not the alternatives considered first

Three routes for letting C++-flavored source reach `ProcessGraph` were
considered, in order:

1. **Extend `pycparser`'s grammar to accept C++.** Rejected. This is not a
   translation-table problem -- `role_schemas` (`operator_defs.py`) is
   cheap to extend because it operates on already-parsed nodes; getting
   raw C++ text into a tree in the first place is the hard part, because
   C++'s grammar is not context-free the way C's is (resolving `a < b > c`
   as a comparison chain or a template instantiation requires a live
   symbol table during parsing, which `pycparser`'s PLY/LALR grammar has no
   mechanism for), and C is not a strict syntactic subset of C++ either, so
   grafting C++ productions onto the existing C grammar risks quietly
   breaking working C parsing.
2. **Adopt a native parser (`tree-sitter`/`libclang`).** Rejected for now.
   Neither is installed in this environment (`requirements.txt` has
   neither; `import tree_sitter` / `import clang.cindex` both fail).
   More fundamentally: a compiled native library's parsing logic stays
   outside this project's graph system entirely -- exactly the "second
   interpreter" problem the whole compiler effort has been working to
   eliminate this session. Wrapping its *output* into graph nodes (as done
   for JavaScript/acorn, see below) doesn't put the *parsing itself* inside
   something this system can reason about or reduce.
3. **Compile with a real C++ toolchain, lift the resulting binary via the
   existing machine-code-lifting pipeline
   (`machine_code_lifting.py::raise_binary_region_to_ssa`, the same
   infrastructure the reversible-machine-executor work already uses).**
   Not rejected, but shelved as a second, lower-priority track -- it trades
   structure for behavior (by the time g++ has compiled a class hierarchy
   to x86, the classes, names, and template structure are gone; what comes
   back is "what the program does," not "how it was written"), and a real
   SSA→`ProcessGraph` lowering doesn't exist yet either (only the reverse
   direction, and `SSA→FusedProgram` for the numeric half). This is the
   same shape as the "SSA→ProcessGraph decompilation" path already noted
   as deferred in `GRAPH_DESCRIPTION_LAYER_SURVEY.md`; it stays deferred.

## What this route actually is

Not a C++ parser. A **source-to-source desugaring pass**: rewrite a
narrow, deliberately limited "C++-like shell" into real, valid C text,
then hand that unchanged to the existing, working, trusted
`pycparser`-based path (`machine_code_lifting.py::c_function_token_multigraph`
and whatever it grows into). `pycparser` itself is never modified.

This is not a novel idea -- it is how the original C++ compiler worked
(Cfront, "C with Classes": compiled to C source, which a real C compiler
then compiled). It is also already this codebase's own precedent in
miniature: `c_function_token_multigraph`'s docstring already describes
stripping preprocessor lines and MSVC `__declspec` "so the same compilable
Windows fixture can enter the portable parser" -- light text massaging
before parsing, for exactly the reason this document proposes doing more
of, at a larger scale.

## Scope for a first working slice

Desugar:

- `class Foo { fields; methods; };` → `struct Foo { fields; };` plus free
  functions `ReturnType Foo__method(struct Foo* self, params)`, rewriting
  `this->x` and bare in-body field references to `self->x`.
- `obj.method(args)` call sites → `Foo__method(&obj, args)`.
- A constructor → an explicit `Foo__new(args)` function returning an
  initialized `struct Foo`.
- Single inheritance (`class Derived : public Base { ... }`) → struct
  embedding (`struct Derived { struct Base base; ... }`), no virtual
  dispatch, no automatic base-method forwarding beyond what struct
  embedding gives for free in C.

**Explicitly rejected, not silently mishandled**, matching the rule
already established for `ControlProgram` ("Unsupported control is rejected
instead of replaced by the discovery trace"): templates, operator
overloading, multiple inheritance, virtual functions/polymorphism,
exceptions, namespaces beyond the trivial case, and anything else not
listed above. A block using any of these should fail loudly with a clear
"not supported by the C++-like shell" error, never a silent wrong
desugaring.

## Verification plan

Same discipline as the rest of this round of work: no claim of "this
works" without a real, passing test against real source, mirroring how the
JavaScript slice (`oop_language_translations.py`,
`tests/test_oop_language_translations.py`) was verified against real acorn
output, not just "doesn't crash." A first test should desugar a small real
class with one method and one field, confirm the desugared text is valid C
(parses via the existing `pycparser` path with no errors), and confirm the
resulting graph shape is sane (a `Foo__new` function, a `Foo__method`
function taking `self` as its first parameter).

## Open questions this document does not answer

- Where does the desugaring pass live -- a new module alongside
  `machine_code_lifting.py`, or inside `dream_document.py` as a new
  `DreamLanguageTranslation` route (`"cpp-shell"` or similar) feeding into
  the existing `route == "ast"`-style handling? Leaning toward a new,
  separate module (desugaring is not C-token-graph work, it produces text,
  not a graph) that `dream_document.py` calls before delegating to the
  existing C route -- but this hasn't been decided against real code yet.
- Does `node_special_cases.py` need any new entries once real desugared-C
  output starts flowing through it, or does the existing Python/SymPy
  switch already handle whatever `pycparser` node shapes appear? Unknown
  until a real desugared program is actually pushed through.
- How much of the "torture test" C++ block
  (`examples/torture_five_languages.dream`, if extended per the earlier
  discussion) does this scope actually cover? Not yet checked against real
  torture-test content -- the scope above was reasoned from general C++
  feature triage, not from that specific file's actual constructs.
