2026-10-28 LANE D: A FUNCTION NAMED `unbroadcast` IS NOT THE AUTOGRAD HELPER (MEASURED, cloud):
`_expand_specialized_unbroadcast_identity` replaced any specialized graph
whose function_name was "unbroadcast" (and whose Inputs were spelled `G` and
`target_shape`) with the backward helper's sum/reshape sequence. A program's
own `def unbroadcast(G, target_shape): return G * 2.0` was rewritten: on main
it emits a reshape it never wrote; the same body named `double` refuses
differently. The expansion now requires the authored definition identity
(`authored_function_identity` == ("src.common.tensors.backward_registry",
"unbroadcast")); a probe of a program calling the registry helper shows it
still expands. With the fix the user `unbroadcast` and `double` programs
fail identically (tensor-returning helpers are not yet lowered), so no PASS
corpus program is possible for it yet.
Inventory for the rest of the item: 13 binding_name comparisons in decision
code; 511 startswith/endswith (mostly generated text and paths); about 80
receiver checks spelled "self"/"cls" in topological_reducer, graph_express2
and fortran_c_shell. A class whose receiver formal is not spelled `self`
(`def __init__(me, n)`) refuses on main; all three files together decide it.
