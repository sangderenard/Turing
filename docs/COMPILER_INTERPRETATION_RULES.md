# How the compiler interprets a program

This is a working reference for the interpretation rules that decide whether a
lowered program means what its source meant. Each section states the rule, what
the compiler used to do, what confused it, and why the current behaviour is the
same intent expressed correctly rather than a new policy.

Every rule here was written while repairing a real defect found by compiling
authored source and comparing the executed result with eager `AbstractTensor`
execution. None of these defects raised. Each produced a complete, compiling
program that read the wrong elements, which is why value comparison, not
inspection, is the gate that matters.

## 1. One storage identity may carry several shaped views

**Rule.** A value id names storage. A shape belongs to an *occurrence* of that
id, not to the id. `b` and `b.reshape((-1, 1, 2))` are the same bytes under two
shapes, and both are true at once.

**What confused it.** Two passes treated the id as owning the shape.
`propagate_repository_ssa_call_metadata` restamped every occurrence with the
call-edge shape when acting authoritatively, and `_intern_unshadowed_formal_uses`
replaced any operand whose id matched a formal with the formal object itself.
Both were right about storage and wrong about shape: the view silently became
its allocation owner's extents.

The damage was invisible. `broadcast_double` received source extents `(8, 2)`
for a `(8, 1, 2)` view being broadcast to `(8, 4, 2)` — not a legal broadcast at
all — and every consumer read misaligned elements.

**How it works now.** A view alias records an `ssa_storage_view` receipt naming
the storage value id, the view shape and the operation. Both passes retain an
occurrence carrying that receipt when its element count equals the storage's,
which is the proof that the two describe the same bytes. Interning still adopts
the formal's dtype, device and accounting, so one storage identity keeps exactly
one physical contract while carrying as many shaped views as the source wrote.

**Invariant.** Physical contract follows the id. Shape follows the occurrence.
A pass that needs one must not overwrite the other.

## 2. An operator that names a catalogued kernel is numerical work

**Rule.** If an operation appears in the tensor catalogue, it lowers to that
kernel over the whole tensor.

**What confused it.** `_SHAPED_SSA_OPERATIONS` had no entry for `Mod` or
`FloorDiv`, so a shaped `%` or `//` fell through to the scalar instruction
emitter. That emitter is correct for scalars: it computed element zero into a
C scalar. The result buffer kept its `calloc` zeros, and nothing reported a
problem, because emitting a scalar for a scalar is not an error — the mistake
was upstream, in deciding the operation was scalar.

**How it works now.** Both spellings map to their catalogue kernels
(`CT_OP_MOD`, `CT_OP_FLOORDIV`). Native output matches eager for all four
operand-sign combinations, so the catalogue's Python floor semantics are
preserved rather than replaced by C truncation.

**Invariant.** The decision "scalar or tensor" is made from the operand
contract, never from the absence of a table entry. A missing spelling must be a
loud shortfall, not a silent demotion to scalar.

## 3. Operation metadata is not a data operand

**Rule.** Some operands describe the operation rather than feed it: an axis, a
slice bound, a dtype spelling. They are metadata and do not make an operation
non-numerical.

**What confused it.** `_is_dispatch_metadata_node` classified a node as
coordinator work if any constant operand was not readable as tensor data. Basic
index literals were already exempt, but the `"int64"` string in
`x.to_dtype("int64")` was not, so the whole cast became coordinator metadata,
left every numerical region, and its result became a region feed nothing
produced. It read as zero.

**How it works now.** A dtype spelling on a cast operation is exempt, exactly as
index literals are.

**Invariant.** Ask what the operand *is to the operation*, not whether it looks
like a number.

## 4. Only a literal integer index projects a call's outputs

**Rule.** `r[1]` selects one of a call's outputs, which is a structural path with
no computation. `m[:, :, 0:2]` is a numerical view that some region must
compute.

**What confused it.** `call_result_projections` walked every `Indexed`
successor of a callsite and removed it from region ownership. A slice therefore
belonged to no region, so nothing computed it, and the caller kept an unnamed
formal for a value that has a perfectly good definition in the source. This was
the `formal_parity` finding on `vehicle_close_contact_graph`,
`vehicle_rig_points_vector` and `vehicle_periodic_terrain_vector`.

**How it works now.** A projection requires exactly one index operand that is a
literal integer. Everything else stays a numerical view its region computes.

**Invariant.** Ownership follows computation. If evaluating it requires
arithmetic, some region owns it.

## 4a. A projection path ends at the first tensor

**Rule.** The walk from a callsite through its projections continues only
while each step is still an intermediate aggregate. `r[1][0]` is a structural
path when `r[1]` is another tuple, and an ordinary element read when `r[1]` is
a tensor.

**What confused it.** The walk descended through every `Indexed` successor
regardless. A tensor output's element read was therefore classified as a
call-boundary projection, belonged to no region, and nothing computed it, so
the caller kept a formal for it. This is the thirteen unnamed formals in
`validator_simulation_advance`, which reads `result[1][0, 0, 6]` from its
tick-vector call.

**How it works now.** A projection is recorded, but the walk only descends
past it when the projected value has no tensor descriptor.

**Invariant.** Selecting an output is structure. Reading inside an output is
computation.

## 5. A coordinator-only operand still has to be produced

**Rule.** A short-circuited `and`/`or` operand may be skipped at runtime, so it
belongs to no numerical region. It is still a value, and whoever consumes it
needs it to exist.

**What confused it.** Structural recovery rebuilds such operands from their
exact graph edges, and handled comparisons, casts, constants and `isfinite`.
It had no reduction case. So in

```python
physical = bool(finite and position.abs().max() < 100.0 and pressure.min() >= 0.0)
```

`pressure.isfinite().all()` failed to recover with reason `operator`, its
enclosing boolean chain failed with it (`operand:9`), and
`validator_simulation_advance` ended up calling its own planned region with a
value nothing defines. That single undefined operand stopped the validator build.

**How it works now.** For a single-operand `all`, `any`, `sum`, `prod`, `mean`,
`min` or `max`, recovery emits exactly the instruction a region body carries for
the same source node, and tensor lowering turns it into the reduction kernel
either way. An explicit axis or keepdim is left unrecovered rather than guessed,
so an unsupported case still announces itself.

**Invariant.** Recovery reproduces the instruction the ordinary path would have
produced. It never invents an operator the ordinary path does not have.

## 6. A capture is the enclosing value, contract included

**Rule.** A nested function reads bindings from its enclosing scope. Those
captures are formals the caller must supply, and their tensor contract is the
contract of the value the caller already holds.

**What confused it.** Two separate gaps.

*Naming.* A capture is not an authored parameter, so it appeared in no
`parameter_names` entry and the signature read as though it had grown values no
caller could name — two such groups on the balloon tire's `_wrench_force` alone.

*Contract.* Callsite specialization gives descriptors to bound arguments only.
A capture is not a callsite argument, so the callee had no contract for it. Its
own result shape then could not be settled, and the whole tensor returned to the
caller as a scalar occurrence, so exactly one element of the result was ever
written.

**How it works now.** Captures are recorded as a `closure_formals` ABI receipt
which `check_formal_parity` accounts, and the callee is offered the enclosing
value's descriptor under the capture's own name. The agreement check that
applies to bound arguments applies to captures too, so a shared catalogue graph
is only specialized when every callsite names the same contract.

The discriminator is the source graph's `closure`/`external` binding kind, never
the name. A name-keyed first attempt claimed
`balloon_tire_reduced_vector_step`'s own `r0`/`z0` tuple-unpack temporaries.
Those are escaped locals, not captures, and masking them would have hidden a
real defect. They still announce themselves.

**Invariant.** A capture is accounted because the caller can name it and knows
its contract. A value that merely lacks a name is never accounted.

## 6a. Still confused: a capture whose producer was folded away

The contract lookup finds the enclosing value through the caller's identity
table. A binding whose producer was folded — `wrench_k = inputs[:, 33].reshape(...)`
— keeps no entry there, and captures have no caller-side binding edge in the
source graph at all: they acquire one during call linking, later. Such a
capture therefore reaches its callee undescribed, and every shape expression
over it (`wrench_k.shape[:2]`) stays unresolved and escapes as an anonymous
formal. This is the six remaining unnamed formals in each of the two
`_wrench_force` specializations. Pinned as
`test_capture_of_a_folded_binding_still_has_a_contract`.

A tempting shortcut is unsound and was tried and reverted: using the capture
node's own value id as the caller's id. Callee-local ids are per-function, so
`plane_point_q` took the shape of whatever value happened to share id 0 in the
caller. Supplying these contracts belongs where captures are actually bound.

## 6b. Still confused: an enclosing parameter read only by a closure

An authored parameter consumed only inside a nested function is dropped from the
enclosing signature and replaced by one anonymous formal per callsite, so no
caller can supply it. This is pinned as
`test_captured_enclosing_parameter_keeps_its_signature_slot` and reproduces
identically at `3af7d206`. The capture rule above covers the contract; the
parameter's own signature slot is a separate, open defect.

## 7. A shape tuple is a structural value, slices included

**Rule.** `x.shape` is a tuple the compiler can read, index, concatenate and
slice, exactly as Python does, because every element is a static extent.

**What confused it.** Indexing folded (`x.shape[0]`) and concatenation folded
(`x.shape + (1,)`), but slicing did not: the structural evaluator had no case
for a `Slice` node, so the index of `x.shape[:2]` never resolved and the
subscript stayed unknown. The reshape consuming it then had no target extents,
kept its source shape, and the next operation saw operands that cannot combine.

The balloon tire writes exactly this spelling:
`wrench_k.reshape(wrench_k.shape[:2] + (1,))`.

**How it works now.** A `Slice` node evaluates to a Python `slice` built from
its resolved bounds, which is the same currency the basic-index reader already
produces. An unresolved bound keeps the whole slice unresolved rather than
guessing `None`.

**Invariant.** Anything the source can do to a tuple of static extents, the
fold can do, or it must say it cannot.

## 8. A recovered definition must dominate every use

**Rule.** Structural recovery inserts real instructions. Like any definition,
they have to be reachable from every use.

**What confused it.** Recovered instructions are inserted before the function's
terminator, then moved in front of their consumer. The move required a single
consuming block: with uses in more than one block the placement could not prove
a common dominator, so it left the definition where it landed. In
`validator_simulation_advance` the recovered boolean is read by a planned
region in `entry` and returned from `if_merge`, so it stayed in `if_merge` and
did not dominate its region call.

**How it works now.** A value used from several blocks is hoisted into the entry
block, which dominates every block by construction, in front of its earliest use
there. Hoisting is allowed only when every external operand of the moved closure
is already available at that point; otherwise the value stays put and the
definition-dominance check reports it rather than the compiler moving something
it cannot justify.

**Invariant.** Recovery may choose where a definition goes, never whether a use
can see it.

## 9. A return may mix tensors and nested aggregates

**Rule.** An authored `return` tuple publishes one member per element, so the
caller's `result[k]` resolves to that member. Members may be tensors, absent,
or nested aggregates.

**What confused it.** Publication required *every* member descriptor to be a
tensor mapping or `None`. The vehicle tick returns fifteen tensors and one
four-member history tuple, and that single nested member failed the test, so
the whole publication was refused. Nothing was published, every sibling
`result[k]` had no member to resolve to, and each became a formal no caller
could fill — the thirteen unnamed formals in `validator_simulation_advance`.

**How it works now.** A nested member is admitted. It is published as a member
with no tensor fact of its own, because an aggregate is not a tensor, and its
structure stays in the ordered output descriptors where the nested projection
path reads it.

**Invariant.** One member the compiler cannot describe as a tensor is a fact
about that member, not grounds to refuse the ones it can.

## 10. Still confused: a nested static loop is preserved, and its list index then has no producer

**Rule as written.** A multi-carried loop is a coordinated recurrence whose
whole `(initial, updated)` vector must advance on one backedge, so it is
preserved rather than unrolled. The protection closes over lexical owners,
because evaporating an owner would erase its retained child control.

**Where that goes wrong.** The protection also covers a loop with a *single*
carried binding purely because it is nested, and the closure then protects its
owner. Two ordinary static loops therefore both become `NATIVE_SOURCE` even
though each one alone is `UNROLL`:

```python
for segment in range(3):                      # alone: UNROLL
    r0, r1 = station_r[segment], station_r[segment + 1]
    for node in nodes:                        # alone: UNROLL
        segment_area = segment_area + chord   # one carried binding
```

Once the outer loop is retained, `station_r[segment]` indexes a Python list of
tensors by a runtime value, which has no lowering. The destructuring
temporaries the compiler itself created for `r0, r1` are then formals with no
producer. In the balloon tire this is exactly `r0`/`r1`/`z0`/`z1` in
`balloon_tire_reduced_vector_step`, four unnamed formals.

Either interpretation would resolve it: unroll the outer loop so the index is
literal, or lower the Python list to a sequence the retained loop can index.
Unrolling innermost-first would make both loops eligible, but the evaporator
clones value producers and cannot yet clone a retained child loop, which is
what the closure protects against. Pinned as
`test_nested_static_loops_unroll_their_list_index`.

`TURING_DEBUG_LOOP_EVAPORATION` prints each candidate's strategy, trip count
and guard components, which is how the demotion was located.

## 11. Publication must honour the declared dtype

**Rule.** The interior may compute in the double working representation. What
crosses the ABI is what the source declared.

**Still confused.** A declared `int64` output is published as raw doubles
whenever a tensor kernel also consumes it. Returned alone, the cast writes
`int64_t` storage and the exported buffer agrees; consumed as well, its storage
settles to `double` for the kernel while the root wrapper still declares
`int64_t`. The call edge has physical input adapters but no output adapter.
Pinned as `test_consumed_integer_output_is_published_in_its_declared_dtype`.

## 12. Still confused: a dispatched job carries no argument ports

**Rule.** A dispatcher operation is a call. Its arguments need ports, and an
argument whose resolved identity is a callable definition needs a port that
names code rather than a value.

**What is confused.** `threading.Thread(target=work, args=(values,))` lowers to
`thread_create` and fails at emission with `dispatcher arguments lack SSA
identities`. Measured at the failure, the check compares

| side | positional | keywords |
|---|---|---|
| resolved from the graph | 0 | `('target',)` |
| authored in the source | 0 | `['target', 'args']` |

So `target` does resolve: it is a `StaticReference` node with
`reference_kind='function_subgraph'` carried as `kw:target`. It is **`args`**
that has no edge at all, and the tuple's contents never enter the graph — the
`values` parameter has no `Input` node, because nothing consumes it.

**Why the order matters.** Giving `target` a function-reference port and
excluding it from the arity check leaves zero resolved keywords against two
authored ones. Relaxing the check further would emit a job submission with no
data ports: a worker that silently receives nothing, which is worse than the
current refusal. The data ports come first; the function port is a real and
separate improvement, because a `StaticReference` value id is not a runtime
value and every higher-order call meets the same wall.

**The shared shape.** This is the third instance in this document of one
defect: *a binding the compiler has already resolved, which the graph
vocabulary has no port to carry, discovered at the far end.* Section 6a is the
same thing for a closure capture, which acquires its caller-side binding only
during call linking. The prescription is the same in both: mint the port where
the binding is minted, while the thing being named is still in hand, rather
than discovering its absence at emission.

Pinned as `tests/test_dispatch_argument_ports.py`.

## Where these rules meet the time field

Three of the sections above are one defect seen three times: a binding the
compiler has already resolved, which the graph vocabulary has no port to
carry, discovered at the far end. Section 6a is a closure capture, which
acquires its caller-side binding only during call linking. Section 12 is a
dispatched job's arguments. The prescription is the same in both: mint the
port where the binding is minted, while the thing being named is still in
hand, rather than discovering its absence at emission.

Section 10 has a deadline attached to it. Retention is decided by carried
state, and `engine_toy/TIME_FIELD_DESIGN.md` makes the time field dynamical:
`log_tau` *and* `dlog_tau_dt`, position and velocity, per zone. A second
carried value is already enough on its own to retain a loop, with no nesting,
and a retained loop cannot index a Python list of tensors by its loop
variable. So that defect moves from the balloon tire into the dt machinery as
soon as the field gains its velocity. Widening the unroll rule would be a fix
that expires; lowering the index in the retained case is the one that
survives.

`engine_toy/NEGATIVE_DRIFT_AND_DIFFUSION_REPORT.md` carries the measurements
behind both statements.

## Diagnostics that find this class of defect

These are env-gated and print to stderr. They exist because each one located a
defect that no exception reported.

| variable | shows |
|---|---|
| `TURING_DEBUG_GRAPH_NODES=fn:id,id` | a node's type, op, tensor descriptor, expression class, attributes and parents |
| `TURING_DEBUG_VIEW_ALIAS` | each view alias decision with its source and view shapes |
| `TURING_DEBUG_STRUCTURAL_RECOVERY` | each recovery attempt and whether it produced a value |
| `TURING_DEBUG_CAPTURE_DESCRIPTOR` | each capture's caller lookup and the descriptor offered |

The reliable method is the same every time: compile the authored source,
execute it, and compare against eager `AbstractTensor` execution on the same
inputs. A structural audit that reports nothing proves nothing on its own.


---

## A bound record resolves and the object parameter is dropped anyway

**This is a defect with a clean reproduction, and the first two
explanations for it were wrong. Both are recorded because they are the
explanations anyone else will reach for.**

### The symptom

Lower a function whose work happens through an object:

```python
def frame(batch, out):
    batch.sync_state_span()
    out[0] = 1.0
    return out
```

with `EngineBatch` declared in `program_abi.records` and bound in
`program_abi.bindings`. It lowers without complaint, emits with **zero
shortfalls**, and produces a function whose body is gone. With two object
parameters and no scalar work, the whole emission is:

```llvm
define void @engine_toy_game__frame(ptr %buffers, ptr %extents) {
entry:
  ret void
}
```

### What it is not

*Not "methods cannot lower".* A free function call lowers correctly: the
same probe with `def bump(x): return x * 2.0 + 1.0` called from `frame`
emits 73 lines of real IR with the call resolved.

*Not "the binding did not match".* It is worth checking, because it is easy
to get wrong -- `records_for_function` matches with `fnmatchcase`, so a
binding written `{"function": "frame", ...}` never matches the qualified
name `engine_toy_game__frame` and silently does nothing. Write `"*frame"`.
But with the glob correct and the binding confirmed resolving:

```
records_for_function('engine_toy_game__frame')
  -> {'batch': 'craft_graph.EngineBatch', 'graph': 'craft_graph.VehicleGraph', ...}
```

the emitted function is still empty.

### What it is

The contract resolves the record. The lowering drops the parameter:

```
parameter_names: {'out': 1}        # `batch` is absent
```

An object-typed parameter does not reach the ABI even when the contract has
said exactly what it is, and everything reached through that parameter goes
with it. The body is then genuinely dead, and an empty body is the correct
lowering of a function with nothing observable in it -- which is why no
shortfall fires. The defect is upstream of the emission, in whatever builds
`parameter_names`.

### Why this is worse than the refusal it replaces

Undeclared, the same program refuses, and refuses well:

```
CompilationSubdivisionRequired: a loop's body regions are scheduled but the
loop itself could not compile, which would otherwise silently run the body
once with no iteration and no effect from its blockers:
  loop_node=252 blockers=('opaque-state-effect',)  batch.step(dt)
```

Declaring the record clears that guard without making the parameter
materialise, so a loud, actionable refusal becomes a silent no-op reported
as success. The message was written to prevent exactly this and now
describes what happens one level up.

### The guard, until it is fixed

**Zero shortfalls on an empty function is not a pass.**

```python
artifact = emit_ssa_function_to_llvm(module, qualified)
assert artifact.shortfalls == ()
assert dict(module.functions[qualified].metadata["parameter_names"])   # NOT empty
assert len(module.functions[qualified].instructions) > 0
```

then execute against eager, which is the only check an empty body cannot
pass.

### The improvement

Either make a bound record materialise its parameter -- the contract
already knows the identity, the fields, the storage and the mutability, so
the information is present and unused -- or emit a shortfall when a
parameter named in `records_for_function` does not appear in
`parameter_names`, so the gap between what was declared and what was
compiled can never be silent.

Reproduction: `engine_toy/time_trials/demo_game.py`, entry `frame`, with
`EngineBatch` and `VehicleGraph` declared and bound `"*frame"`. Lowers in
3.4 s, emits in 0.0 s, zero shortfalls, eight lines of LLVM of which one is
`ret void`.
