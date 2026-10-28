# Return arms of differing int widths

- Program: Token Orchard's `_fingerprint_span`, reduced to corpus c793. One return is `0, 0` (int64 by class). The other is the combines `high * (1 << 32) + low`, proven `[0, 2**64-1]` (uint64). Python returns an int from either.
- Wrong record: the return-slot merge in `_ControlSSABuilder` only joined int/bool/float classes (`_numeric_join`), so int64 and uint64 arms were refused as `return_arms_dtype_disagree`. That refusal stopped Albert's whole-core audit in all four specializations of `_fingerprint_span`. In the caller, the call's tuple members kept int64 in the planner's region record because the proven-width rule only covered arithmetic operations. C then refused `first % m` with "mixed signed/unsigned operands".
- Rule:
  - A join of scalar int arms of differing widths, at least one wider than int64, takes the narrowest width holding the union of the arms' intervals (`_integer_width_join`).
  - Each arm's interval is its `source_int_range_concordance` row, else a constant's value, else its width's bounds.
  - Each arm converts on its own edge.
  - It is not a promoted int/float join, so `ssa_numeric_joins` does not refuse it.
  - When no width holds the union, the shortfall stays.
  - In the planner's parent-dtype pass, a proven uint64/int128 width overrides an integer dtype for any scalar node.
- Measured:
  - c793 refuses on main. Here it matches Python on C and LLVM for all 5 inputs, and so do c790 and c792.
  - The integer-width test files plus `test_ssa_numeric_joins`: the same 20 failing names on main and here.
  - Corpus `-k "c7 or c6"`: the same 6 failing names on both trees, with one more pass here (c793).
  - No gate, no full corpus, and the whole-core audit was not re-run.
