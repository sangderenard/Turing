# 35: Lane A, item A0 (2026-10-22)

Branch `claude/project-thread-kbkzp7`, from `main` `2532577`. Measured in a
cloud container (4 cores, deps from `requirements.txt` plus `ziglang`,
`llvmlite`, `pytest-timeout`).

## A0: a chained comparison whose middle is a builtin conversion

- **Program.** `digits.isdigit() and 0 <= int(digits) <= 9`
  (`identity_concordance.identity_log_lzma_preset`, line 9358; the bootstrap)
  and `eps < float(value) < hi` (`dt_controller.py:663`; the managed tire,
  **inferred** from doc 34). Corpus c211, c212.
- **Records that disagree.** Python evaluates each middle operand once.
  `_pure_operand` classed every `ast.Call` as effectful, so the normalizer
  refused the chain at source normalisation, the first stage of both targets.
- **Rule.** A middle operand may be spelled twice when it is a pure read or
  one of `int`, `float`, `len`, `abs`, `ord` (one positional argument)
  applied to one, and the module never rebinds that name
  (`_rebound_names`: assignment, `def`, `class`, parameter, import,
  `global`, `except ... as`, match captures). Both occurrences stay inside
  the conjunction, in place: a guarding `and` still short-circuits the first,
  and the second runs right after the first returned. Nothing is hoisted.
  Any other call middle (a user function, a method, a shadowed builtin,
  `int(x, 2)`, `int(*v)`) keeps the named refusal. The receipt lists the
  re-evaluated middles (`reevaluated_conversions`).
- **Assumption (inferred).** A second evaluation of these builtins over
  builtin-typed arguments is the first's value with no effect. A user
  `__int__`/`__len__` with an effect would break it; the native lanes do not
  dispatch to one (lanes plan A7).
- **Reach (observed by scan).** Over every `src/**/*.py`, 23 files refused
  before; after, one: `laplace_nd.py:382` (`v.dim()`, a method, refused by
  design). The 22 others held `len(x)`, `abs(x)`, `ord(c)` middles, many of
  them in compiler sources the bootstrap compiles (`fortran_c_shell.py`,
  `precompile_to_ssa.py`, `glsl_deployment_strategy.py`), so a fix for `int`
  and `float` alone would have moved the bootstrap's first refusal by one
  line.
- **Measured at `0d98ef1`.** Unit tests
  (`tests/test_chained_comparison_normalization.py`) 15 passed. Typed corpus
  161/161 PASS, 0 concordance findings. Fuzz `FUZZ_EXT=1` seed 10, 30
  programs (with the two new snippets): 30 PASS. Gate: aa 90, ab 7, ac 26,
  rec 20. The parent `2532577` in the same container: aa 89, ab 7, ac 26,
  rec 20. The one extra aa name,
  `test_sympy_process_graph_is_registered_before_python_dependency_pursuit`,
  fails identically when run alone on the parent's worktree (a cached
  symbolic process graph is returned, `cache_hit=True`), so it is
  order-dependent, not this change (**observed**; its cause is unknown).
