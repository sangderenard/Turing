# Native exception-control contract

This contract covers authored `raise`, `try`, `except`, `else`, and `finally`
entering `src.compiler.fortran_c_shell.lower_ast_source_to_ssa`. It is a
condition for admitting an authored `Try`, not evidence that the present
compiler implements one. The current `SOURCE_EXCEPTION_ADMISSION` gate refuses
retained authored `Try` before normal SSA emission.

## Source authority

An exceptional transfer starts at the exact retained source operation. Its
identity is the source graph value cell, not its line, spelling, or region
position. The lexical owner of each operation is the exact `Try` and arm in
which it occurs. A `Try` catches only failures from its own body. A failure
from its handler, `else`, or `finally` seeks an outer eligible body. An
`except*` needs exception-group control and remains refused.

The exception class used for dispatch comes from the authored constructor's
resolved static declaration cell, or the native operation's declared failure
class. Matching uses the class subtype relation, in authored handler order.
The first matching arm runs. An unmatched failure leaves the `Try` after its
`finally`; it cannot become a successful result.

The native status vocabulary now distinguishes Python exception statuses from
native width limitations. `NATIVE_STATUS_CLASS` ties each catchable status to
its `STATIC_SYMBOL_DEFINITION`; C and LLVM can report and take a runtime
status as an SSA value. A float-to-int cast reports NaN and infinity before
conversion. SSA has not yet placed the status branch under an authored Try.

## State at a transfer

Each transfer carries its outcome and the exact versions of values that the
destination can read. For a name, the source is its `NAME_BINDING` cell at
the moment of transfer. A failed operation's normal result is never a source
for that edge. If a destination reads a name absent from an incoming edge,
lowering refuses. A join with distinct incoming versions must have a
concordance Phi whose arms cite the individual transfer and binding cells.
The source, canonical, SSA, and physical forms must derive from those same
cells. A raw environment copy is not an authority for any of these forms.

The reducer now begins this chain with `SOURCE_EXCEPTION_SITE_STATE` at each
protected call or authored raise. The fact lists the names demanded by the
active handler and finalizer arms and cites each available `NAME_BINDING`
version at that instant. It also cites each enclosing `Try` source cell, its
active arm, and the exact `ExceptHandler` cell within a handler arm. This is
an ingestion fact only. The site alone does not establish an executable
exceptional successor or native status branch, so it cannot discharge
`SOURCE_EXCEPTION_ADMISSION`.

`SOURCE_EXCEPTION_TRANSFER` now derives the nearest eligible `Try` body and
the inner finalizers crossed by an outward failure from those site scopes.
The transfer cites the site fact and retains its binding-version cells. It
does not yet prove handler class matching or create executable SSA edges.
`SOURCE_EXCEPTION_HANDLER_CLASS` records each authored handler type operand
and its static declaration cell. A type with no unique declaration remains
unresolved and cannot participate in native dispatch.
`SOURCE_EXCEPTION_HANDLER_ENTRY` correlates each handler with the transfers
whose catch owner is its `Try` and the binding cell available for each name
on each candidate edge. These are candidates. Class matching and first-arm
selection must narrow them before an executable handler Phi is admitted.
The reducer now uses the candidate site bindings for handler reads and mints
a Phi when the incoming binding cells differ. `SOURCE_EXCEPTION_HANDLER_PHI`
ties that graph value to the transfer and binding cells. An outward edge
crossing an inner finalizer still refuses when the outer handler reads state:
the post-finalizer binding is not yet represented. The admission gate remains
closed for all of these paths.

For a call with multiple possible native failures, the source operation
has one control site with distinct declared failure outcomes. Its handler
entry includes only outcomes matching that handler. A nested `Try` handles
its own body failures before any outer handler receives an unmatched outcome.
The outer handler's state is the state *after* each inner finalizer that runs
on that path.

## Control outcomes

The pending outcome is one of normal, raise, return, break, or continue.
`else` runs only after a normally completed body. `finally` runs after every
outcome, including an unmatched raise and a handler's outcome. A normally
completed finalizer resumes the pending outcome. A finalizer's abrupt outcome
replaces it. Break and continue retain their exact loop target; return
retains its exact value. A bare `raise` in a handler uses the active caught
exception, not a new class inferred from syntax.

## Admission and verification

Admission requires concordance-backed source transfers, handler and finalizer
joins, and executable SSA successors for every retained outcome. The C and
LLVM backends must preserve the native status and the control successor. A
missing site, class, binding, join arm, loop target, or physical value is a
compile refusal. The existing authored-`Try` gate stays closed until these
conditions hold for the whole retained control subtree.

The current control pipeline constructs loop programs in
`analyze_shader_loop_reductions` and ordinary conditional programs in
`_ordinary_conditional_control_programs`, then overlays those programs before
`precompile_to_ssa` lowers them. The live precompile now constructs a
`NativeTryBlock` for a retained function Try and posts its source control
fact before the admission refusal. Its region markers have exact arm
membership, and handler class values follow ingestion-to-canonical edges
into the forked shell. The control block also cites each exact
`SOURCE_EXCEPTION_CONTINUATION_PHI` fact for its post-Try value joins.
The block is not yet in the executable overlay, and
the SSA lowerer has no branch for it. Folded assignments with no deployment
region, call placement, and pending outcomes still need control ownership;
this Try-shaped block does not discharge admission without SSA failure
successors. The historical constructor only owned selected loop-body Try
syntax and rejected nested control and abrupt finalizer paths.

The bounded native matrix must cover direct raise, caught call failure,
handler order and mismatch, `else`, `finally` on each outcome, nested outward
propagation, handler state written before the failing operation, loop-carried
handler state, and return/break/continue through finalizers in C and LLVM.
Each observed native output and fault must agree with Python's authored
control result. The existing `view` audit currently refuses with `selected
return has no exact source call reachability authority`; that baseline is
separate from exception admission and must not be reported as green.

## Historical prototype boundary

Commit `089eee45` posted exception-site state by copying every integer value
from the reducer's lexical environment. Its handler entry then read those
site states; it could select `body_environment` for a terminal raise. Those
are observed properties of the prototype, not authoritative liveness or
predecessor selection. The prototype did not finish nested outward
propagation and pending return/break/continue. Its native success cases are
useful regression cases, not admission proof.
