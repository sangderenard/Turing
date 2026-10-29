2026-10-28 LANE D: BROAD EXCEPT HANDLERS IN DECISION PATHS (MEASURED, cloud):
Inventory over src/compiler, the reducer and graph ingestion: 169 handlers
catch Exception (or bare); 58 only pass, continue or return. Fifteen of those
guard a fact read or a proof post; the rest are diagnostics, telemetry,
caches, sympy probing or best-effort static evaluation. Hooked over 76
focused corpus programs, none of the fifteen ever caught. Nine guarded
book reads/posts that cannot legitimately fail (current_identity_book
always returns a book) and turned any failure into "no fact":
fortran_c_shell._loop_rebound_value_ids (-> no rebinds),
formal_shape_ledger._proven_formal_shape / _proven_formal_literal (-> no
proof / conflict), and six in tensor_ssa_lowering (proven_shape_of read and
import, _record_ssa_shape, three record_proven_shape posts). Their handlers
are removed so a failure is loud. The other six stay: int() / item()
conversion attempts in ir_container_ops, an equality probe of arbitrary
facts in identity_concordance, a reference resolution over live objects in
graph_express2, and a diagnostics-only block in call_argument_binding.
76/76 PASS as on main; five focused pytest files give main's failure names.
