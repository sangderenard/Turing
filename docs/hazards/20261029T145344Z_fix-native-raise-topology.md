# 2026-10-29 Exact lexical states on exceptional source edges

The reducer now captures the runtime lexical bindings at each live Call and
Raise occurrence. Canonical relabel maps both site and value IDs through its
existing mapping and refuses any live site with an unmapped binding. The
exception topology posts those bindings as canonical value cells in each
`SOURCE_EXCEPTION_TRANSFER` fact; its provenance derives from those cells.
The concordance audit checks unique ordered names, same-scope canonical
value refs and the transfer's actual dependency edges.

Measured on the codec source graph: `float(value)` is source call 65 and has
`value=4`, `index=2`, and no `numeric` binding. `int(value)` is source call
80 and has the same `value` and `index`, plus `numeric=65`. These are distinct
exceptional entry states, not a pre-Try snapshot reused for both calls. A
direct structural exception-transfer audit found 0 findings. The codec still
refuses at `try-exception-entry-state-not-composed` / `Raise`; handler Phi and
protected conversion failure edges have not been lowered.

Missing authored Raise and Try graph nodes now refuse. The codec's first
TypeError Raise was removed by a structural `If` specialization; its exact
`SOURCE_CONTROL_SPECIALIZATION` fact selected `orelse` and rejected `body`.
Topology traversal now skips only that proven rejected arm. Its live
exception transfers still post and the structural audit remains at 0.

Native C corpus cases c1166 and c1167 passed together (2 passed, 651
deselected). Syntax compilation passed. The full `view` audit remains red at
the previously observed `selected return has no exact source call
reachability authority` error.
