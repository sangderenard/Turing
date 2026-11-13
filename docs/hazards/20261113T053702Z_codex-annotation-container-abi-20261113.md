# Authored primitive span entry ownership

Observed: root(a: str,b: str) has physical int64/rank1 span receipts, but its parameter_abi_kind was Unresolved(parameter_abi_undeclared). Source entry publication only admitted tensor annotations. The existing ownership gate correctly refused to mark generated length/capacity members borrowed, and LLVM refused unbounded private capacity.

Change: reuse the existing annotation field receipt for non-optional str/bytes/bytearray source-entry boundaries, retaining the existing selected-entry scope. Nested helpers retain exact call-edge ABI and scalar key-token semantics. Ownership checks and native capacity refusal are unchanged.

Verified: three native length tests pass, with empty inputs, Unicode glyphs/UTF-8 bytes, independent fields, repeated execution, and unchanged borrowed inputs. Existing ABI/annotation suite:22 passed. Immediate HEAD comparison:22 passed and exactly the three new tests failed. Mapping audit:28 rows/2 functions,0 findings,0 unsourced,latch CLOSED.

Limits: this is ownership, not full string-comparison correctness. Plain return a==b still selects elementwise Eq, not full-content sequence equality. Dynamic string fields in Sequence[Record] lack per-field borrowed row descriptors; existing parameter-record table-context machinery is the intended investigation seam. Actual compile_nodes remains blocked; no native GraphSolver product claimed. Prior scalar-token string fixtures were invalid for plain str and were removed to scratch, not committed. No metadata-only fix or guard relaxation was retained.
