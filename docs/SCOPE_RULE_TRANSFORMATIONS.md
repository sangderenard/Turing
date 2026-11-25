# Scope rules between front ends and back ends

The compiler translates programs between languages that disagree about
scope:
- **where** a name is visible,
- **how long** a binding lasts,
- **whether** assigning or passing a value copies it or shares it.

Every front end must spell its source's rules in Python, the compiler's
input language, without changing what the program means. Every back end
must spell Python's rules in its target language without changing what
the program means. This document lists each of those transformations that
we know has to be negotiated, says where each one is implemented, and
names the gaps.

Status words used below:
- **Done:** implemented and covered by a test.
- **Gap:** a known wrong or refused case.
- **Open:** not yet examined.

If you add a front end or a back end, go through every row here and state
which rule applies.

## 1. Vocabulary

- **Open scope:** one scope per function. Python works this way. A name
  bound anywhere in a function is the same variable everywhere in that
  function, and it stays bound after the block that bound it ends.
  `for` targets, `if` arms, `with` targets and `try` bodies all bind in
  the function's scope.
- **Block scope:** each `{ }` block, and each `for` initializer, opens its
  own scope. GLSL, C, C++ and WGSL work this way, as do JavaScript's `let`
  and `const`. An inner declaration of an existing name is a different
  variable. When the block ends, its variables cease to exist, and an outer
  variable it shadowed becomes visible again, unchanged.
- **Block to open:** a front end reading a block-scoped language into
  Python. These transformations need **spelling changes**: the same source
  name can become several Python names.
- **Open to block:** a back end emitting a block-scoped target from Python
  or SSA. These transformations need **preallocation**: a variable is
  declared in the scope that encloses every one of its uses.
- **Value semantics:** assigning, passing or returning a value copies it.
  This is GLSL's rule for every type, including vectors, structs and
  arrays.
- **Reference semantics:** assigning or passing names the same object.
  This is Python's rule for objects, records and lists.

## 2. Block to open (front ends into Python)

### 2.1 Durability of an outer definition across an inner block

```glsl
float x = 1.0;
{ float x = 5.0; y += x; }   // the inner x is a different variable
z = x;                       // the outer x: still 1.0
```

In Python, rebinding `x` inside the block overwrites the outer `x`, which
stays overwritten after the block ends.

- **Rule:** a redeclaration in an inner block gets a fresh Python name for
  the rest of that block. The outer name is never written by the inner
  block.
- **Implemented:** `glsl_ast_ingestion.py`, redeclaration handling. It
  produces `x__glsl_scope_N` (the `lexical` stack in `_ASTExpressionParser`,
  plus the redeclaration branch in the declaration reader).
- **Status:** Done for locals.
- **Gap, shadowed global passed to a helper:** a helper that reads or
  writes a GLSL global `g` receives it as a carried parameter. If the
  caller has a local named `g` that shadows the global, the reader hands
  the helper the caller's local instead of the global. Result: `y = 5`
  where GLSL gives `0`. Reported in review comments 623 and 624
  (`glsl_ast_ingestion.py` around line 943, resolved at around line 332).
  The fix is for a carried global to be resolved through the global's own
  binding, never through the caller's lexical stack at the call site. That
  includes the inout write-back.

### 2.2 Per-iteration freshness of a declaration in a loop body

```glsl
for (...) { vec3 rd = vec3(...); ... }   // a new rd each iteration
```

Python reads this as one name, `rd`, rebound on every iteration, and the
previous iteration's object is still bound to it at the top of the next
iteration. That is correct Python, but the compiler must not treat `rd` as
carried around the loop.

- **Rule:** a record bound in a loop's own body is a fresh object on each
  iteration of that loop. A storing call, a field write or a carried-field
  merge from an earlier iteration never reaches it.
- **Implemented:** the reducer posts each name binding's owning loop on the
  book, on page `name_binding_loop_owner`, and `record_built_in_loop` reads
  it (`topological_reducer.py`). It is used by:
  - `loop_stores_into`;
  - `loop_may_change_field`;
  - the loop-exit field rule, `rewritten_by_enclosing_loop`.
- **Tests:** `tests/test_record_field_rebinding.py`, cases
  `built_in_loop_body`, `built_in_loop_body_passed_to_helper` and
  `conditional_write_in_while_body`. A record carried into the loop is
  still refused by name:
  `test_record_carried_into_a_storing_loop_is_refused_by_name`.
- **Gap:** in an unrolled `range` loop, a conditional field write fails.
  The iterations' copies share one field-state cell, which is refused
  with an `ssa_field_version` disagreement. `while` loops are not
  unrolled, so they are not affected.

### 2.3 Value semantics

```glsl
vec3 a = b; a.x = 9.0;      // b is unchanged
vec3 keep(vec3 d) { return d; }
```

In Python, `a = b` makes `a` and `b` name one object, so `a.x = 9.0` also
writes `b.x`. That is a silent wrong value.

- **Rule:** wherever GLSL copies, the Python names a new object.
  - A record (a vector or struct) is copied as its constructor applied to
    its copied members.
  - A fixed array is copied as the list of its copied elements.
  - A scalar is already a value.
  - A resource handle (buffer, image, sampler) is not copied, because GLSL
    does not copy it either.
- **Where GLSL copies:**
  - assignment, and a declaration whose initializer reads existing
    storage;
  - `return` of existing storage;
  - out and inout write-back;
  - an `in` or `inout` parameter that the helper stores into.
- **Implemented:** `_value_copy` and `_copied_read`
  (`glsl_ast_ingestion.py`).
- **Tests:** `tests/test_glsl_value_copies.py`.

### 2.4 Parameters: copy-in, copy-out

GLSL `in` parameters copy in. `out` parameters copy out and are undefined
on entry. `inout` parameters do both.

- **Rule, `in` parameter the helper writes:** the copy is a block-to-open
  spelling change. The helper's copy gets its own name,
  `d__glsl_scope_N`, and the body uses that name. The parameter itself is
  then only read, which is what lets the compiler prove that the caller's
  object is untouched.
- **Rule, `out` and `inout`:** they become results. Each `return` returns
  `(result, outs...)`, and the caller unpacks them in order. An `out`
  parameter starts at its type's zero.
- **Implemented:** `_helper_definition` (`glsl_ast_ingestion.py`).
- **Compiler side:** a parameter that is dead on entry receives no actual.
  The planner and `_is_parameter_entry` admit only a value that the book's
  ingestion record names an Input.
- **Open:** passing one variable as two `inout` arguments. GLSL leaves the
  copy-out order undefined; we must refuse that call or fix an order, and
  neither is decided yet.

### 2.5 Mutable globals

A GLSL global is private to its invocation.

- **Rule:** `main` owns the globals. A helper takes the globals it uses
  after its authored parameters.
  - A written scalar, vector or struct is `inout`: it is returned.
  - An array is passed as its list and written in place. That is the
    caller's storage by design, and it is not copied.
- **Implemented:** `_HelperSignature.globals` and `carried_parameters`
  (`glsl_ast_ingestion.py`).
- **Tests:** `tests/test_glsl_mutable_globals.py`.
- **Gap:** the shadowed-global case in section 2.1.

### 2.6 JavaScript `let` and `const`, against `var`

**Open.** `javascript_source_ingestion.py` has not been examined for block
scope. Every `let` and `const` row in sections 2.1 to 2.3 applies to it:
- A `let` in a `for` header gets a new binding on each iteration.
- `var` is function-scoped and hoisted.
- The temporal dead zone (reading a `let` before its declaration throws) is
  a refusal, not a value.

## 3. Open to block (Python or SSA into block-scoped targets)

### 3.1 A name that outlives its block

```python
for i in range(n):
    last = f(i)
use(last)          # still bound after the loop
use(i)             # the loop target too: its final value
```

A block-scoped target that declares `last` inside the loop's block cannot
read it after the loop.

- **Rule:** preallocate. Declare each variable in the innermost block that
  encloses every one of its definitions and uses (the block that dominates
  them).
- **Rule, Fortran:** declarations belong at the head of the subprogram.
  Everything is preallocated there.

### 3.2 A conditionally defined name

```python
if c:
    v = 1.0
use(v)             # UnboundLocalError in Python when c is false
```

- **Rule:** the back end must not invent a default value. Either the
  compiler proves that every path defines `v`, or the program is refused
  by name. A preallocated slot does not make the read defined: a
  preallocation is storage, not a value.

### 3.3 A loop that runs zero times

A Python `for` over an empty range leaves its target unbound if it was
never bound before the loop, or unchanged if it was.

- **Rule:** the loop result's port carries the initial value. A target
  that was unbound before the loop is the conditional-definition case in
  section 3.2.

### 3.4 Reference semantics into value targets

Python aliasing must be resolved before emitting a language with no
references: GLSL, WGSL, or Fortran value arguments. A record that two
names share is one storage, and every write goes to that storage. The
emitter must never copy a value where the source shares it.

Two compiler gaps belong to this area:
- **Fixed:** returning a record built from a formal's own fields
  (`vec2(v.x, v.y)`) used to leave block geometry with no recorded cause.
  `drop_dead_pure_region_calls` now posts the removed call on
  `pure_call_retirement`.
- **Gap:** a record whose fields are all initialized from one SSA value
  (`vec3(z, z, z)`) reads back wrong after an inout write-back, because
  field storage is identified by its initializing value. The GLSL reader
  never emits this form (it creates one temporary per literal), but valid
  Python can.

### 3.5 Closures and late binding

A Python closure captures the variable, not its value at creation. A
lambda made in a loop sees the variable's final value. Block-scoped
targets with no closures (GLSL, Fortran) must receive the captured
variable as explicit storage. A target with closures (C++, JavaScript)
must capture by reference.

- **Status:** Open as a written rule. The capture machinery exists in the
  book (`name_binding_capture`, `SOURCE_CLOSURE_CAPTURE`), but no
  transformation row for each target has been written yet.

### 3.6 Comprehension scope

A Python 3 comprehension has its own scope. Its target does not leak, and
the outer name is untouched afterwards. A back end that lowers a
comprehension to a loop in the enclosing block must spell its target as
a fresh name.

- **Status:** Open as a written rule.

## 4. How to add a row

- Name the program that showed the rule, with a test that runs it on the
  LLVM lane.
- State the rule in the source language's terms.
- Implement it where the fact is made, as an identity on the book, not as
  a spelling match.
- Add the row here: rule, where it is implemented, its tests, and its
  status.
