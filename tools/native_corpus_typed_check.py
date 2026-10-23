import sys, warnings, pathlib, tempfile, traceback, ast, os
warnings.filterwarnings("ignore")
ROOT = os.environ.get("TURING_ROOT", str(__import__("pathlib").Path(__file__).resolve().parents[1]))
sys.path.insert(0, ROOT)
import numpy as np
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_to_c
from src.compiler.identity_concordance import concordance_report
# Backend parity (lane E): TYPED_CHECK_BACKENDS=c,llvm,fortran runs every
# program through each named backend after C.  Each backend is checked
# against Python on its own line, and a backend whose values differ from
# C's on the same input prints a DISAGREE line: a finding in one of them.
BACKENDS = tuple(
    name.strip() for name in os.environ.get("TYPED_CHECK_BACKENDS", "c").split(",")
    if name.strip()
)
unknown_backends = set(BACKENDS) - {"c", "llvm", "fortran"}
if unknown_backends:
    raise SystemExit(f"unknown TYPED_CHECK_BACKENDS entries: {sorted(unknown_backends)}")


def judge(feeds, expect, flat, kinds):
    exp = list(expect) if isinstance(expect, tuple) else [expect]
    # An integer answer is exact: a float64 round trip of 6 * (2**55 + 1)
    # passed under the relative tolerance (a comprehension row typed float64).
    ok = len(flat) == len(exp) and all(
        (float(a).is_integer() and int(a) == b) if isinstance(b, int) and not isinstance(b, bool)
        else abs(float(a) - float(b)) <= 1e-12 * max(1.0, abs(float(b)))
        for a, b in zip(flat, exp))
    want_kinds = ["b" if isinstance(e, bool) else "i" if isinstance(e, int) else "f" if isinstance(e, float) else "?" for e in exp]
    type_ok = all(w == "?" or k == w or (w == "i" and k in "iu") or (w == "b" and k in "biu") for k, w in zip(kinds, want_kinds))
    return ("ok" if type_ok else f"TYPE {feeds} got kinds {kinds} want {want_kinds}") if ok else f"WRONG {feeds} got {flat} want {exp}"


def run_llvm(module, export, params, outs, inputs, ns):
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )
    try:
        art = emit_ssa_function_to_llvm(module, export)
        if not art.complete:
            return f"EMIT-INCOMPLETE {[s.reason[:150] for s in art.shortfalls[:2]]}", None
        art = compile_artifact(art)
    except Exception as e:
        return f"EMIT-FAIL {type(e).__name__}: {str(e)[:300]}", None
    results, values = [], []
    for feeds in inputs:
        try:
            ex = prepare_artifact_execution(art, {params[k]: np.array([v]) for k, v in feeds.items()}).run()
            arrays = [np.asarray(ex.buffers[v.id]) for v in outs]
            flat = [x for a in arrays for x in a.reshape(-1).tolist()]
            values.append(flat)
            results.append(judge(feeds, ns["root"](**feeds), flat, [a.dtype.kind for a in arrays]))
        except Exception as e:
            values.append(None)
            results.append(f"RUN-FAIL {type(e).__name__}: {str(e)[:150]}")
    bad = [r for r in results if r != "ok"]
    return f"{'PASS' if not bad else 'FAIL'} {bad[:3]}", values


def run_isolated(function, *args):
    """Run ``function`` in a forked child: a native crash in one backend is
    that program's finding, not the end of the corpus run."""
    import multiprocessing
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    child = context.Process(target=lambda: queue.put(function(*args)))
    child.start()
    try:
        result = queue.get(timeout=600)
    except Exception:
        result = None
    child.join(10)
    if result is None:
        if child.is_alive():
            child.kill()
        return f"RUN-CRASH exit code {child.exitcode}", None
    return result


def run_fortran(module, export, params, outs, inputs, ns):
    # Compiled and executed through the packed Fortran core ABI when a
    # compiler is on the machine; without one, emission is all that can be
    # checked, and the line says so rather than passing.
    from src.compiler.ssa_fortran_backend import (
        FortranCoreNative, emit_module, fortran_compiler,
    )
    try:
        fortran_module = emit_module(module, progress=lambda _: None)
    except Exception as e:
        return f"EMIT-FAIL {type(e).__name__}: {str(e)[:300]}", None
    if not getattr(fortran_module, "complete", True):
        return f"EMIT-INCOMPLETE {[s.format()[:150] for s in fortran_module.shortfalls[:2]]}", None
    if fortran_compiler() is None:
        return "EMITTED, NOT RUN (no Fortran compiler on this machine)", None
    try:
        from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
        built = compile_fortran_module_c_shell(
            fortran_module, {}, pathlib.Path(tempfile.mkdtemp()), library=True,
            entrypoint=export, name=export[:48],
        )
        record = next(e for e in fortran_module.api.entry_points if str(e.name) == export)
        native = FortranCoreNative(built.executable_path, record, fortran_module.api)
    except Exception as e:
        return f"COMPILE-FAIL {type(e).__name__}: {str(e)[:300]}", None
    results, values = [], []
    for feeds in inputs:
        try:
            ex = native.prepare_execution({params[k]: np.array([v]) for k, v in feeds.items()}).run()
            arrays = [np.asarray(ex.buffers[v.id]) for v in outs]
            flat = [x for a in arrays for x in a.reshape(-1).tolist()]
            values.append(flat)
            results.append(judge(feeds, ns["root"](**feeds), flat, [a.dtype.kind for a in arrays]))
        except Exception as e:
            values.append(None)
            results.append(f"RUN-FAIL {type(e).__name__}: {str(e)[:150]}")
    bad = [r for r in results if r != "ok"]
    return f"{'PASS' if not bad else 'FAIL'} {bad[:3]}", values


def disagreements(inputs, c_values, other_values):
    return [
        f"{feeds} c={c} other={o}"
        for feeds, c, o in zip(inputs, c_values, other_values)
        if c is not None and o is not None and (
            len(c) != len(o)
            or any(abs(float(a) - float(b)) > 1e-12 * max(1.0, abs(float(a))) for a, b in zip(c, o))
        )
    ]


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
    c_values = []
    for feeds in inputs:
        expect = ns["root"](**feeds)
        try:
            ex = art.prepare_execution({params[k]: np.array([v]) for k, v in feeds.items()}).run()
            got = [ex.buffers[v.id].reshape(-1).tolist() for v in outs]
            kinds = [ex.buffers[v.id].dtype.kind for v in outs]
            flat = [x for g in got for x in g]
            c_values.append(flat)
            results.append(judge(feeds, expect, flat, kinds))
        except Exception as e:
            c_values.append(None)
            results.append(f"RUN-FAIL {type(e).__name__}: {str(e)[:150]}")
    bad = [r for r in results if r != "ok"]
    print(f"{path.stem}: {'PASS' if not bad else 'FAIL'} {bad[:3]} | {rep}")
    if "llvm" in BACKENDS:
        verdict, llvm_values = run_isolated(run_llvm, module, exports[0], params, outs, inputs, ns)
        print(f"{path.stem}: LLVM {verdict}")
        found = disagreements(inputs, c_values, llvm_values or ())
        if found:
            print(f"{path.stem}: DISAGREE c/llvm {found[:2]}")
    if "fortran" in BACKENDS:
        verdict, fortran_values = run_isolated(run_fortran, module, exports[0], params, outs, inputs, ns)
        print(f"{path.stem}: FORTRAN {verdict}")
        found = disagreements(inputs, c_values, fortran_values or ())
        if found:
            print(f"{path.stem}: DISAGREE c/fortran {found[:2]}")
