2026-11-13 Retarget conditional initial identities with retained comprehension ports (MEASURED)

Real compile_nodes source: `out_keys = [node.key ...]; if not out_keys: out_keys = list(node_keys)`. After its retained comprehension materializes, the Phi's unchanged arm correctly names LoopResult107, but its cached initial_value_id still named ListComp92. _list_rebound_on_one_branch therefore did not recognize the unchanged initial arm, and incorrectly hit the unequal-display refusal. This was an identity cache bug, not evidence that this source merges incompatible list sizes.

Repair the existing producer-side _retarget_cached_value_ids helper: its scalar replacement now also carries initial_value_id. The graph-wide caller's cache-presence filter includes that key. No relaxed merge predicate, guessed list extent, special case for compile_nodes, or weakened concordance check.

Verification:
- New test_comprehension_initial_identity.py: 3 passed; clean base554ac55e: 2 failed,1 passed. Unit cache identity, complete dynamic-comprehension SSA with zero CorrelationTable findings, and a bounded LLVM differential across five inputs.
- Existing test_rebound_empty_list.py: 4 passed,2 failed on both repaired and baseline versions. The two existing C tests refuse unproven private-root storage capacity; LLVM cases pass.
- Mapping concordance audit unchanged:28 rows,2 functions,0 findings,0 unsourced,latch CLOSED.
- Real class-equipped compile_nodes now passes the list merge and reaches `declared call storage ... callee compile_nodes___import_gs has no recorded dtype`. That source helper imports graph_solver and returns the module. Static module return/caller propagation is the next independent seam, not a numerical dtype to guess.

Limits: the dynamic range variant reaches concordant SSA but LLVM emission still refuses a runtime range and unproven storage capacity. Only the explicitly bounded variant was executed natively. Full GraphSolver LLVM remains unproven. No merge or deployment; shared sources untouched.
