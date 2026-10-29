2026-10-29 EXACT EXCEPT-ARM MEMBERSHIP (MEASURED, Windows)

Observed repro: a source Try with two handlers, one raising ValueError from
`except TypeError` and the other raising OverflowError from `except ValueError`,
gave both Raise nodes the same `(try_id, "handlers")` branch compartment.
This discarded the authored handler identity before control composition.
`_branch_compartments` now keys each handler arm by its exact graph node ID.
The same graph probe showed the two Raise nodes in distinct compartments:
`(14, "handler:10")` and `(14, "handler:11")`; each ExceptHandler node shared
the compartment of its own Raise. No source position or handler ordinal is
used to select the arm.

This is one producer-side identity correction, not a native try implementation.
The real codec still refuses at its handler Raise because no source Try control
program is composed. Native protected-call failures and finally traversal
remain open. The concordance `view` audit still exits 1 at the pre-existing
`selected return has no exact source call reachability authority` error.
`python -m pytest -q tests/test_loop_composer.py -k
raise_guard_inside_retained_loop` passed 1 (99 deselected). The source graph
probe above ran in the real reducer and deployment membership code; no native
codec artifact was compiled.
