import sys, warnings, pathlib, tempfile, traceback, ast, os
warnings.filterwarnings("ignore")
ROOT = os.environ.get("TURING_ROOT", str(__import__("pathlib").Path(__file__).resolve().parents[1]))
sys.path.insert(0, ROOT)
import numpy as np
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_to_c
from src.compiler.identity_concordance import concordance_report
for path in sorted(pathlib.Path(sys.argv[1]).glob("c*.py")):
    src = path.read_text()
    inputs = ast.literal_eval(src.split("# INPUTS", 1)[1].strip())
    ns = {}
    exec(compile(src, str(path), "exec"), ns)
    try:
        module, _, exports = lower_ast_source_to_ssa(src, "root", name="corpus_" + path.stem, extraction_contract=ROOT + "/extraction_contracts/program_extraction.yaml", progress=lambda _: None)
    except Exception as e:
        print(f"{path.stem}: LOWER-FAIL {type(e).__name__}: {str(e)[:300]}"); continue
    rep = " ".join(l.strip() for l in concordance_report(module, limit=3).splitlines()[:1] + [l for l in concordance_report(module, limit=3).splitlines() if l.startswith("     ")][:2])
    try:
        root = module.functions[exports[0]]
        params = dict(root.metadata['parameter_names'])
        from src.compiler.project_compilation_product import _unexplained_root_argument_ids
        if _unexplained_root_argument_ids(root):
            print(f"{path.stem}: PRODUCT-REFUSES unexplained root inputs {_unexplained_root_argument_ids(root)} | {rep}"); continue
        outs = [i.args for b in root.blocks.values() for i in b.instrs if i.op == 'Ret'][0]
        art = emit_ssa_to_c(module, exports[0])
        if not art.complete:
            print(f"{path.stem}: EMIT-INCOMPLETE {art.shortfalls[:2]} | {rep}"); continue
        d = pathlib.Path(tempfile.mkdtemp()); art.compile(d / "native")
    except Exception as e:
        print(f"{path.stem}: EMIT-FAIL {type(e).__name__}: {str(e)[:300]} | {rep}"); continue
    results = []
    for feeds in inputs:
        expect = ns["root"](**feeds)
        try:
            ex = art.prepare_execution({params[k]: np.array([v]) for k, v in feeds.items()}).run()
            got = [ex.buffers[v.id].reshape(-1).tolist() for v in outs]
            kinds = [ex.buffers[v.id].dtype.kind for v in outs]
            flat = [x for g in got for x in g]
            exp = list(expect) if isinstance(expect, tuple) else [expect]
            ok = len(flat) == len(exp) and all(abs(float(a) - float(b)) <= 1e-12 * max(1.0, abs(float(b))) for a, b in zip(flat, exp))
            want_kinds = ["b" if isinstance(e, bool) else "i" if isinstance(e, int) else "f" if isinstance(e, float) else "?" for e in exp]
            type_ok = all(w == "?" or k == w or (w == "i" and k in "iu") or (w == "b" and k in "biu") for k, w in zip(kinds, want_kinds))
            results.append(("ok" if type_ok else f"TYPE {feeds} got kinds {kinds} want {want_kinds}") if ok else f"WRONG {feeds} got {flat} want {exp}")
        except Exception as e:
            results.append(f"RUN-FAIL {type(e).__name__}: {str(e)[:150]}")
    bad = [r for r in results if r != "ok"]
    print(f"{path.stem}: {'PASS' if not bad else 'FAIL'} {bad[:3]} | {rep}")
