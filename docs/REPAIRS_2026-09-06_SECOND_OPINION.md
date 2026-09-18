# Second-opinion repairs: first verified increment

The user authorized implementation while independently checking the review:
"begin implementing repairs while taking your own time to verify they are the right move".
Existing uncommitted work was preserved. No authored DT source was changed.

## Reachability repair

`src/compiler/ssa_reachability.py` removes CFG edges proved impossible by scalar
literal, Boolean, and equal-constant Phi analysis. It removes unreachable
blocks and filters labelled Phi inputs by actual predecessor edges, including
edges removed between two still-reachable blocks. Dynamic loop Phis remain
dynamic. Mutable ABI fields and multiply defined ids cannot seed constants.

The full-native linker runs this after late source-literal recovery and before
its existing caller/callee signature-pruning transaction. No provenance
exemption was added. Reachable calls and memory operations are retained even
when their results are unused. This is deliberately not an effect-elimination
pass, nor does it fix an incorrectly hoisted call.

An in-memory replay of the saved whole-program SSA removed five private
formals, reducing the count from 23 to 18 with zero undefined operands.

The fresh full driver completed with **19 unaccounted formals**, zero undefined
operands, zero unresolved calls, and zero unmaterialized boundaries. It exited
1 at the strict gate, as required. This fresh input includes the existing
experimental field ledger and is not the same input as the saved 23-finding
artifact; the two counts must not be conflated. The new groups are
run_superstep 6, step 10, and the three singleton helpers. The authoritative
diagnostic files were regenerated; the review's old numerical ids are stale.
Log: `build/reachability_full_formals_20260906.log`.

The five non-native regression cases, seven formal-provenance cases, and the
existing caller/callee pruning regression passed: **13 passed in 4.62s**.
The new compiled return-path test and existing conditional-call-result and
sequence-truth native tests passed: **3 passed in 27.88s**, with each native
execution subprocess bounded at 20 seconds. The outer pytest invocations had
90/120-second limits. This proves the bounded cases, not validator parity.
The guestbook filename validator passed in its default non-mutating mode.
All build/test processes launched by this repair are terminal. Nothing was
committed or pushed.

## Corrections and limits to the review

- The predicted 23-to-16 reduction is not established. In the saved
  `_no_exchange_observed`, the `_scalar` call consuming formal 11 is already
  in `entry`, before the absorbing-false `LAnd`. It remains a reachable call.
  Removing its result's eventual use does not justify deleting its effects.
  The replay also retains `run_superstep` formals 124 and 126. Five step
  formals disappear, including formatted failure-header value 248, beyond
  the review's proposed seven-value set.
- `ssa_c_backend.py` declares **all internal semantic inputs as `void *`**
  and loads scalars through those storage addresses. Therefore scalar SSA
  dtype does not imply a by-value native formal. This rejects the review's
  provisional ABI inference; it does not prove that the advance-result alias
  is wired to the correct storage at every retry.
- Keyed-field assignment must preserve all existing Python aliases. Copying
  an arena and rebinding only one local name is not a generally sufficient
  implementation of reference assignment. This needs an alias-aware storage
  contract and tests before accepting that recommendation.
- Hash-combining formatting operands is not an exact implementation of
  formatted-string equality. Rounding and collisions matter. No such token
  approximation was introduced.

## Remaining work

The formal gate still refuses the full program. Effect ordering, record-field
return propagation, record projection linking, comprehension production,
value-position Boolean semantics, formatted strings, and loop-current state
still require repairs and native/eager proofs. Zero formals would not establish
correctness of dropped writes or execution order. Full validator native build,
frame parity, and performance remain unverified.
