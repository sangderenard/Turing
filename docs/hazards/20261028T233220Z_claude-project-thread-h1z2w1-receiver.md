2026-10-28 A method's receiver was found by its spelling `self` (MEASURED, topological_reducer / graph_express2 / fortran_c_shell / precompile_to_ssa)

Program: `class Bag: def __init__(me, n): me.xs = [n, n + 1, n + 2]` with
`def total(me): return me.xs[0] + me.xs[2]` (corpus c950 uses `me` and
`this`). Wrong record: field declaration, instance-attribute ingestion,
reduced `__init__` list fields, method record views, the receiver's record
ABI and identity history, and control lowering's field-slot accounting each
compared a name against the literal `"self"` (or `{"self", "cls"}`). With
another spelling the receiver had no record view and lowering refused
("region value 3 (Indexed in function 'total') has no settled dtype"); the
same program spelled `self` passed. Rule: Python binds the receiver to the
method's first positional formal whatever its name, and a staticmethod has
none. `method_receiver_formal(definition)` (python_special_cases) reads it
from the AST; `_graph_receiver_formal(graph)` (fortran_c_shell) reads it from
`method_binding` + `positional_parameters`; `lower_control_sections_to_ssa`
takes it as `receiver_parameter` and refuses a receiver value without one.

Still open (same family, not yet changed): about 60 more `"self"` literals in
these files (fortran_c_shell 4977-4992, 7096/7114, 41408, 45019, the 62590-
63890 block; glsl_deployment_strategy and others). The call-binding site in
fortran_c_shell that picks `receiver_spelling` ("self"/"cls") before falling
back to `positional_names[0]` cannot simply take `positional_names[0]`: doing
so broke c950 ("region value 5 (Add in function 'twice') has no settled
dtype"), so that list is not always receiver-first there; it needs its own
look.
