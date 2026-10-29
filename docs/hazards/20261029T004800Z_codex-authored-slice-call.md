2026-10-28 Live String Core demo: authored slice method is not a slice value (MEASURED, Windows)

The tracked String Core `demos/live_workload.py` calls Vocabulary.slice. Full lowering stopped at root Indexed283 with no settled dtype. A read-only hook at hierarchy planning observed its parent282 had already become Constant(slice(None,None,None)); the original expression was `vocabulary.slice(token,0,1)`.

A bounded public-source repro in tests/test_authored_slice_method.py uses Box.slice returning (0,token+begin+end). Before this change the runtime case refused twice (C/LLVM) in8.24s. A profile hook on the structural fold's replace writer observed root8, type `slice`, expr ast.Call, exact method_ref0, receiver_class_ref Box. The fold's node-type test treated that Call as an AST slice and published the wrong constant through its existing fold/concordance path. The caller then trusted that constant and its tuple projections had no dtype.

The structural evaluator now excludes ast.Call from its node-type-only slice branch. Its later builtin-slice evaluation also excludes a call with a resolved callee_ref or method_ref, so all-literal authored calls stay calls. No named-class/method exception, dtype guess, SSA patch or new ledger was added. Ordinary AST slices retain their structural evaluation.

After: all6 authored-method runtime/literal and ordinary-slice native C/LLVM tests pass7.56s, each with zero concordance findings and exact outputs for0/7/-3. This checkout also contains the existing materialized-list-return commit1eff49c5 applied to main09e36ef7 as edfeb6c8; its original3 tests pass7.02s. Main has not been edited. Full live-demo native verification remains pending.
