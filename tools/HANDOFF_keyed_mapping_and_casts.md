# Handoff: the keyed-mapping seam, from the other side

You are working the keyed-mapping / control-SSA seam from the lowering side
(`tools/repro_keyed_get.py`, `inspect_control_ssa.py`,
`count_control_shortfalls.py`, plus the fusion-ordering fix in
`process_graph_fusion.py`). I arrived at the same seam from the *ingestion*
side while chasing why `train_xor` would not reach a runnable backend. This is
what I found, what I changed, and the two live questions I did not touch
because they are inside your files.

I have not modified `process_graph_fusion.py` or `loop_composer.py` at all.


## 1. One keyed mapping was reaching the IR that never should have

`AbstractTensor.tensor_identity` was doing this:

```python
token = value.__dict__.get("_identity_token")
if token is None:
    token = next(_IDENTITY_COUNTER)
    value.__dict__["_identity_token"] = token
```

`__dict__` is the instance attribute dictionary, so this ingests as a literal
mapping: a `GetAttr('__dict__')` feeding a `get` on the read side and an
`IndexedStore` on the write side. In the `tensor_identity` ProcessGraph that
is nodes 6/7/9 (read chain) and 16/17/18 (write chain).

Symptom, on the tape path only:

```
RuntimeError: observed plain operators are absent from the
ProcessGraph segment plan: (6,)
```

Node 16 survives because it feeds `IndexedStore` and the planner keeps it
(`hierarchy_plan == [17, 18]`). Node 6 terminates in a `phi` that selects
between the looked-up token and a freshly minted one, so nothing numeric
consumes it and the planner correctly excludes it -- but the discovery run
observes it executing, and `capture_fused_programs` rejects the disagreement.

**Fixed at the source, not by widening the check.** `AbstractTensor` overrides
neither `__getattr__` nor `__setattr__`, so the `__dict__` detour bought
nothing; it was purely a second path to attribute access. `_identity_token` is
now a declared class attribute defaulting to `None` (so the read never misses
and never needs `__getattr__`), and the function uses ordinary attribute
access. Verified: tokens stable per tensor, distinct across tensors,
non-tensors still fall back to `id()`.

**Why this matters to you:** this is the same shape your
`3555dfa Walk a keyed mapping as its own key and value vectors` addresses, but
it is the degenerate case -- a mapping that carried no information and should
never have been ingested. Your work makes real mappings lower properly; this
one had no reason to exist. If `repro_keyed_get.py` is enumerating mapping
shapes that must lower, `obj.__dict__[const]` is worth treating as *always*
rewritable to attribute access at ingest rather than as a mapping to support.
The keys are constants in every instance of it I found.

I swept `src/common/tensors` for `.__dict__[`, `.__dict__.get(`,
`.__dict__.setdefault(`, `.__dict__.pop(`. The only other hit is
`aot_compile.py:1256`, which is compiler code manipulating its own objects and
never ingested.


## 2. Fortran SSA shortfalls: 19 -> 5, and the remaining 5 are yours to place

`emit_module` on a lowered `train_xor`-shaped program reported 19 shortfalls.

**14 were transcendentals** -- `exp log sin cos tan asin acos atan sinh cosh
tanh asinh acosh atanh` -- each a `Call` whose callee is the operation's own
name, carrying no `tensor_operation` attribute. Two changes in
`ssa_fortran_backend.py`:

* registered all fourteen in `_UNARY` (every one is a Fortran intrinsic; the
  inverse hyperbolics since Fortran 2008, so nothing here is new capability);
* in `_expression`'s `Call` branch, when the local `llvm.*` dict misses and the
  call is unary, fall back to `_UNARY.get(callee)`. Without that the table
  entries are unreachable, because `instr.op` is `"Call"` and the name lives in
  `attributes['callee']`.

Also removed ~200 columns of stray indentation on the `llvm.sqrt.f64` line.
Pre-existing, unrelated, harmless.

**A concern is recorded in a comment there and I want you to see it directly.**
Fortran intrinsics are ELEMENTAL, so `exp(a)` over an array is one whole-array
operation the compiler may vectorise -- the exact property `_REDUCTION` is
written to exploit ("emitted as whole-array intrinsics rather than explicit
loops so the compiler picks the schedule"). That only holds if the operand is
*still an array* when it reaches the table. If the SSA arriving there has
already been scalarised into a per-element loop, these templates faithfully
emit a scalar call per element and the batch opportunity was lost upstream,
not here. The batch-capable library functions (`unary_double` and friends) are
what would preserve it. **Someone should check whether these callees arrive
with array operands before concluding the emitted Fortran is as fast as it can
be.** That check sits closer to your comprehension-region and fusion work than
to mine.


## 3. The remaining 5 shortfalls, and a defect I did not fix

All five are:

```
op=float  block=entry  callee=cast_double_to_float_values
```

I stopped before registering these, because they are miscategorised and I did
not want to write into a table you may be moving.

`_TENSOR_KERNELS` in `ssa_llvm_backend.py` files
`cast_double_to_float_values` next to `stack_double`, `cat_double`,
`pad_double_nd`. But its LLVM body is:

```llvm
%single = fptrunc double %value to float
%result = fpext   float  %single to double
store double %result
```

Narrow, widen back, store a double. That is **a pure scalar function of one
value** -- no neighbours, no reordering, no shape change. It is not a type
change either: the storage stays `double`, and what is discarded is mantissa
below single precision. It sits among the tensor kernels only because C needed
a buffer loop to express it, and the loop got mistaken for the operation.
`stack`/`cat`/`pad` genuinely belong there; this does not.

Under the AbstractTensor cornerstone -- an operation is an **arity plus a
scalar kernel**, with traversal being each backend's own business
(`_elementwise_unary` -> `_apply_scalar_unary`, `_v1/_v2/_v3_valuewise`
flatten-apply-restore, and `_scalar_kernel`'s "the reference semantics every
backend must agree with") -- this belongs in `ELEMENTWISE_UNARY` with a scalar
reference kernel, exactly as `sigmoid` now is. Fortran then spells it in one
already-elemental expression:

```fortran
real(real(a, c_float), c_double)
```

and the batching concern above does not apply, because a genuinely valuewise
operation has no cross-element structure to lose.

**The defect I want a second opinion on before either of us acts:**
`ssa_llvm_backend.py` maps **both** `"float"` and `"double"` to
`cast_double_to_float_values`:

```python
"float": "cast_double_to_float_values",
"double": "cast_double_to_float_values",
```

Under the scalar-kernel reading these are two different functions --
narrow-to-single versus identity-at-double -- so one of those mappings is
lying, and a program asking for `double` currently gets its mantissa
truncated. I did not change it because I could not tell from the table alone
which behaviour callers depend on, and because five identical narrowing casts
in a single `entry` block smells like a symptom of something upstream rather
than five genuine requests. Your `dump_control_gep.py` /
`probe_comprehension_regions.py` view may show what is emitting them.


## 4. For reference: what is mine and safe to ignore

Trace/observation work, none of it in your files: `trace_manifest.py`,
`influence_observer.py`, a `trace` record kind in `shell_telemetry.py`, a
compile-time-gated trace ring in `profiled_c_shell.py`, `attach_trace` on the
C and GLSL JIT launch sites, and `sigmoid` registered end to end (C op tables,
`ELEMENTWISE_UNARY`, `AbstractTensor.sigmoid`, and `abstract_nn`'s
`_sigmoid_stable` reduced from seven nodes to one).

`sigmoid` is worth one note: it was failing as "no captured basic-operator
lowering" because `abstract_nn.Sigmoid` built it from comparisons/`exp`/
division and then recorded the whole thing as one opaque `sigmoid` op -- an
atom whose contents existed only inside that Python function. Same class of
problem as the `__dict__` chain: a construct that reached the IR in a shape the
IR could not act on. Worth a glance in case other activations do it too.


## 5. Open, in priority order

1. `float` vs `double` both mapping to the narrowing kernel (section 3).
2. Where the five casts in one `entry` block come from -- symptom or request.
3. Whether the transcendental callees arrive with array or scalarised operands
   (section 2) -- decides whether the emitted Fortran batches.
4. `obj.__dict__[const]` as an always-rewrite-at-ingest rule (section 1).
