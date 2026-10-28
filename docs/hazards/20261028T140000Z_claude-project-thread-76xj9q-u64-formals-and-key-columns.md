# uint64 fingerprints through method formals and dict key columns

- Program: Token Orchard's `_intern_validated_span`, reduced to corpus c794. `hi, lo = self._fingerprint_span(...)` (proven `[0, 2**64-1]`) is passed to `self._find_existing_span(domain, atoms, begin, end, hi, lo)` and keys `self._candidates.setdefault((domain, hi, lo, n), []).append(token)`.
- Refusal on main: "24 incompatible physical call inputs; storage types are immutable". The uint64 actuals reached int64 formals of `_find_existing_span` and the int64 key columns of the `_candidates` setdefault/store helpers.
- Wrong records, and the rule for each:
  1. `_solve_int_ranges` counted the bound-method read `self.fingerprint` (GetAttr, carrying the same `method_ref`) as a callsite that binds no argument. That marked every method formal unprovable. Only the call is a callsite.
  2. A builtin `int` scalar ABI arm (`_physical_type_arms`) and a call member's linked contract (`linked_value_abi`) took the `int` receipt's int64. They now take the narrowest width holding the proven interval, citing the range cell.
  3. A specialization minted before the whole-program interval proof has no rows of its own. Readers now fall back to the authored function's rows (`_int_range_scopes` / `_int_range_cell`), since a copy runs a subset of the authored executions. This applies in the planner, the ABI arms, the linked contracts, the SSA return-width join and the mapping keys.
  4. An `int` key column of a mapping field was int64 by class. It now holds the join of every writer's member interval at the narrowest width; an unproven member counts as the int64 range. The evidence views cite the range cells, and the final concordance check re-derives the join from them.
- Measured: c794 refuses on main with the audit's error, and passes here on C and LLVM (4 inputs). c790, c792 and c793 still pass. Focused tests: see the PR.
- Open, not fixed here: reading a key back from a dict-of-lists FIELD (`self.d.setdefault(k, []).append(v)` then `for t in self.d.get(k, ()): return t` in another method) returns nothing natively, with 0 findings. That is a silent wrong result on main: `/mnt/project-files/string-core-verify/dict_of_lists_field_readback_repro.py` gives Python 70, LLVM 0. c794 avoids repeated keys and says so.

## Guarded actuals (corpus c795)

- Program: `intern_preflighted_batch` rejects the caller-supplied pair with `if fingerprint_hi < 0 or fingerprint_hi > U64_MAX ...: raise` and then passes `(fingerprint_hi, fingerprint_lo)` to `_intern_validated_span`, whose `hi, lo` members also receive the computed uint64 fingerprint.
- Refusal on main: the members stayed int64 (the guarded pair was unbounded), so the uint64 fingerprint was an incompatible input. With the members at uint64, the caller's int64 pair was the incompatible input.
- Rules:
  5. `_solve_int_ranges` reads raising guards: past `if A or B or ...: ... raise` with no `else`, each comparison of a value with a constant bounds that value at every later statement of the same body. A formal joins its actuals as read AT each callsite, and a tuple display actual binds the formal's unpacked members position by position (a `None` actual or an omitted `None` default binds none).
  6. The planner's synthesized member formals (`fingerprint[0]`) post the join of the projections they replace.
  7. The caller's value keeps its own interval (it is compared before the guard). The narrowed interval at the call is posted on `guarded_call_actual_range`. The call edge converts int64 into a uint64 formal only where that row (or the value's own row) proves it non-negative, and int64 into int128 always. That conversion loses nothing; any other mismatch stays a named conflict.
- Measured: c795 refuses on main and passes here on C and LLVM (3 inputs).
- Still open: an unproven caller-side product such as `given_hi * 4294967296` is int64 arithmetic, so it wraps when a Python caller passes values past int64. That is the overflow-checked int64 promoting to Precision decided for the next step.

## A tuple no caller passes (corpus c796)

- Program: in the whole-core product, `_intern_validated_span`'s only caller (`intern_batch`) omits `fingerprint` (default `None`). So `hi, lo = fingerprint` never runs, and `hi, lo` are exactly the uint64 pair from `_fingerprint_span`.
- Refusal on main: "sequence row dtype concordance disagreement ... value 78: recorded=('int64', 'int64', 'int64', 'int64', 'int64') ... table store operands says ('int64', 'uint64', 'uint64', 'int64', 'int64')". The `_candidates` key columns settled int64 because the unpacked members were unbounded, which made the `hi, lo` merge unbounded.
- Rules:
  8. When every callsite binds a formal to `None` (or omits it with a `None` default), its unpacked members never hold a value and join as nothing.
  9. A specialized copy's interval is met with its authored function's interval, because the copy runs a subset of the authored executions. A copy cannot see the callsites, but the authored proof can.
- Measured: c796 gives the audit's exact error on main and passes here. The whole-core audit gets past value 78 and stops at the next one: "declared call storage for %t... (callee intern_batch) has no recorded dtype".

## A loop return the late fold disproves (corpus c797)

`validate_batch_envelope`'s `if not isinstance(offset, int) or
isinstance(offset, bool): return 1, 0` sits in its offsets loop.  The loop
is composed during deployment selection; only the hierarchy plan's later
structural fold proves the branch false, removes the `1, 0` literals and
revises the return site to `RETURN_SITE_UNREACHABLE`.  The loop's return
block still named the removed literals, and control lowering minted them as
untyped caller storage ("declared call storage ... has no recorded dtype").
The SSA builder now reads the site's slot rows and emits no edge for a site
the record calls unreachable.

## A name rebound after a carried `while` (corpus c798)

`while ...: checksum += x` then `checksum += 100; return checksum`.  The
loop composer appended the loop's result port to `checksum`'s version
history after the post-loop version, so the returned name resolved to the
loop's exit value and the add was dropped with no error.  The port is now
inserted right after the loop's last version of the name; its
`name_binding` row still takes the next free row.
