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

## 6a. Still confused: an enclosing parameter read only by a closure

An authored parameter consumed only inside a nested function is dropped from the
enclosing signature and replaced by one anonymous formal per callsite, so no
caller can supply it. This is pinned as
`test_captured_enclosing_parameter_keeps_its_signature_slot` and reproduces
identically at `3af7d206`. The capture rule above covers the contract; the
parameter's own signature slot is a separate, open defect.

## 7. Publication must honour the declared dtype

**Rule.** The interior may compute in the double working representation. What
crosses the ABI is what the source declared.

**Still confused.** A declared `int64` output is published as raw doubles
whenever a tensor kernel also consumes it. Returned alone, the cast writes
`int64_t` storage and the exported buffer agrees; consumed as well, its storage
settles to `double` for the kernel while the root wrapper still declares
`int64_t`. The call edge has physical input adapters but no output adapter.
Pinned as `test_consumed_integer_output_is_published_in_its_declared_dtype`.

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
