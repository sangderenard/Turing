# A2b findings (parked 2026-10-24 13:24Z, nothing in code yet)

- name.py (`fork = None; if c: fork = (a, b); return fork`) and ret.py (`if c: return (a, b)` / `return None`) both refuse in `precompile_to_ssa` finish: "returned name ...: its newest binding (value 8) was not materialized".
- ret.py fails only because `_normalize_top_level_guard_returns` (ast_control_normalization.py) rewrites the root into the name form. Unrewritten forms pass: ifelse.py (if/else returns) and caller_ret.py (pick as callee) compile.
- Working ABI (return arms, OPTIONAL_AGGREGATE): `Ret [p0, q0, p1, q1]`, with an interleaved presence Phi per member (binding optional_return_presence).
- In name.py, the conditional's result_aliases already carry presence (9,10->13) and the members (0,11->14), (1,12->15). External values hold 13, 14 and 15. The merged node 8 (aggregate_phi) has no record that reaches the finish.
- Plan: give the finish a record mapping 8 to its members and presence, and emit the same interleaved ABI as the return arms so callers and the product accept it. Corpus c473-c479.
