# 2026-10-29 Protected linked call reaches the caller's handler

Program: `root` calls a helper that raises ValueError on its third loop
iteration, inside an authored Try whose handler raises TypeError. The
source graph placed the call in the Try's body, but the lexical callsite
scheduler did not traverse NativeTryBlock. It put the marker before the Try.
The loop producer pass did not discover a marker inside a Try and duplicated
it after the Try. An impossible success block after the handler's literal
Raise reached the Try exit in SSA, so frame linking relocated the call to
the loop latch. The native run reported the helper's raw ValueError because
the caller had no protected status edge.

The callsite scheduler now places the marker in the exact Try body from
the source branch compartment. The loop producer pass recognizes markers
in every Try arm. A literal failing validation marks its success block
unreachable, preventing that impossible edge from reaching the Try exit.
The caller's identity-return alias no longer equates the call result with
a changing loop input. NativeTryBlock records protected linked markers;
its control fact and audit verify source call identity and exact marker
placement under the Try's body. SSA takes the native status after the
linked call and dispatches by the concorded exception class.

Native C and LLVM execution of the protected helper call passed: n=0
returned 0, n=1 returned 0, n=2 returned 1, and n=3 raised NativeTypeError
from the caller's handler. The standalone callee Raise and protected
conversion tests also passed. Combined focused native/control gate:
10 passed, 750 deselected. The focused module's CorrelationTable audit
returned zero findings. Syntax compilation and `git diff --check` passed.
The broad `view` audit remains red at its prior selected-return source-call
reachability refusal.

The real octet codec is still blocked at the unaccounted Slice formal 75.
Exceptional finalizers, handler-entry Phi for differing values, nested Try
dispatch, and full native byte conversion remain unfinished.
