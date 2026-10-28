# A descriptor's int64 narrowed a proven uint64 interval

- Program: corpus c790, `first = high * (1 << 32) + low`, with `high` and `low` each `% 2**32`. Also c792 (renumbered from c791, which another lane's program shares).
- Wrong result: after bit operators (`shl`) got elementwise descriptors, C and LLVM returned -1 for n=1 (Python 393808588), with 0 findings. The interval record still proved the combine `[0, 2**64-1]`.
- Wrong record: the parent-dtype pass in `_build_shell_hierarchy_plan` applied the proven uint64/int128 width only inside the branch for a node with no descriptor dtype. `1 << 32` now has an int64 descriptor that only states its class. That dtype reached the product and the sum as `tensor["dtype"]`, so the width rule never ran.
- Rule: a proven interval wider than int64 overrides any integer dtype the parents or the descriptor give (None, int, int64), whether or not a descriptor states one.
- Measured: c790 and c792 FAIL on main and PASS here. c760 passes. 8 integer-width test files: 20 failing names, the same set on main and here. c777 refuses on both trees with the same error ("region value 1 (input 'n') has no settled dtype"), from the float64 lane. No gate, no full corpus.
