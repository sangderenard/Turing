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
unknown_backends = set(BACKENDS) - {"c", "llvm", "fortran", "wasm"}
if unknown_backends:
    raise SystemExit(f"unknown TYPED_CHECK_BACKENDS entries: {sorted(unknown_backends)}")
# The Fortran artifact carries its runtime by default: linked in on Windows,
# staged beside the shared object on Linux (whose libgfortran.a is not
# position independent; see fortran_toolchain.stages_shared_runtime).
# TYPED_CHECK_FORTRAN_STANDALONE=0 links the toolchain's libgfortran
# dynamically instead.  The emitted program is the same.
FORTRAN_STANDALONE = os.environ.get("TYPED_CHECK_FORTRAN_STANDALONE", "1") != "0"


def _scalar_list(v):
    return isinstance(v, list) and bool(v) and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)


def _feeds(module, export, params, feeds):
    """The root's input buffers.  A scalar parameter is a one-element array.
    A list parameter is its ABI: the column storage holds the rows, and the
    descriptor's length and capacity cells (caller inputs, not compile-time
    sizes) hold len(list); nothing is fed that the descriptor does not name."""

    out = {}
    table = module.sequence_tables.get(export)
    for name, v in feeds.items():
        if not _scalar_list(v):
            out[params[name]] = np.array([v])
            continue
        descriptor = next(
            d for d in table.sequences.values()
            if int(d.column_value_ids[0]) == int(params[name])
        )
        out[params[name]] = np.array(v)
        out[int(descriptor.length_address_id)] = np.array([len(v)])
        out[int(descriptor.capacity_value_id)] = np.array([len(v)])
    return out


def _leaves(value):
    """A returned tuple's leaves in order: the Ret carries a nested tuple
    ``(x, (x, 1.0))`` as its three leaves (c924)."""

    if isinstance(value, tuple):
        return [leaf for member in value for leaf in _leaves(member)]
    return [value]


def judge(feeds, expect, flat, kinds):
    exp = _leaves(expect)
    # An integer answer is exact: a float64 round trip of 6 * (2**55 + 1)
    # passed under the relative tolerance (a comprehension row typed float64).
    ok = len(flat) == len(exp) and all(
        (a is None) if b is None
        else False if a is None
        else (float(a).is_integer() and int(a) == b) if isinstance(b, int) and not isinstance(b, bool)
        else abs(float(a) - float(b)) <= 1e-12 * max(1.0, abs(float(b)))
        for a, b in zip(flat, exp))
    want_kinds = ["b" if isinstance(e, bool) else "i" if isinstance(e, int) else "f" if isinstance(e, float) else "?" for e in exp]
    type_ok = all(w == "?" or k == w or (w == "i" and k in "iu") or (w == "b" and k in "biu") for k, w in zip(kinds, want_kinds))
    return ("ok" if type_ok else f"TYPE {feeds} got kinds {kinds} want {want_kinds}") if ok else f"WRONG {feeds} got {flat} want {exp}"


_UNION_KINDS = {
    "builtins.int": ("i", int), "builtins.float": ("f", float),
    "builtins.bool": ("b", bool), "builtins.NoneType": ("?", lambda _v: None),
}


def output_values(module, export, outs, buffer_of):
    """The root's outputs as Python sees them: ``(values, kinds)``.

    A value union leaves the program as its frame (an int64 tag and one
    payload cell per arm, ``module.value_union_tables``).  The tag names
    the live arm; only that arm's payload is the value, with that arm's
    class.  Every other output is read as it is.
    """

    table = getattr(module, "value_union_tables", {}).get(export)
    unions = [] if table is None else list(dict(table.unions).values())
    ids = [int(v.id) for v in outs]
    frames = {}
    for union in unions:
        members = {int(union.tag_value_id)} | {
            int(value_id) for arm in union.alternatives for value_id in arm.value_ids
        }
        if members <= set(ids):
            frames[min(ids.index(m) for m in members)] = (union, members)
    covered = {m for _union, members in frames.values() for m in members}
    values, kinds = [], []
    for position, v in enumerate(outs):
        if position in frames:
            union, _members = frames[position]
            tag = int(np.asarray(buffer_of(int(union.tag_value_id))).reshape(-1)[0])
            arm = union.alternatives[tag]
            kind, convert = _UNION_KINDS.get(str(arm.name), ("?", lambda value: value))
            payload = (
                None if not arm.value_ids else
                np.asarray(buffer_of(int(arm.value_ids[0]))).reshape(-1)[0].item()
            )
            values.append(convert(payload) if payload is not None else None)
            kinds.append(kind)
            continue
        if int(v.id) in covered:
            continue
        array = np.asarray(buffer_of(int(v.id)))
        values.extend(array.reshape(-1).tolist())
        kinds.extend([array.dtype.kind] * array.size)
    return values, kinds


def run_llvm(module, export, params, outs, inputs, expects):
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
    for feeds, expect in zip(inputs, expects):
        try:
            ex = prepare_artifact_execution(art, _feeds(module, export, params, feeds)).run()
            flat, kinds = output_values(module, export, outs, lambda i: ex.buffers[i])
            values.append(flat)
            results.append(judge(feeds, expect, flat, kinds))
        except Exception as e:
            values.append(None)
            results.append(f"RUN-FAIL {type(e).__name__}: {str(e)[:150]}")
    bad = [r for r in results if r != "ok"]
    return f"{'PASS' if not bad else 'FAIL'} {bad[:3]}", values


def _isolated_entry(queue, function, args):
    queue.put(function(*args))


def run_isolated(function, *args):
    """Run ``function`` in a forked child: a native crash in one backend is
    that program's finding, not the end of the corpus run."""
    import multiprocessing
    # Windows parity receipt (2026-10-24): the backend corpus stopped before
    # its first Fortran compile because Windows has no ``fork`` context.
    # Spawn a guarded module-level child there; Unix retains fork's lower
    # serialization cost.  Backend inputs contain only authored expectations,
    # never the dynamically exec'd Python function namespace.
    context = multiprocessing.get_context("spawn" if os.name == "nt" else "fork")
    queue = context.Queue()
    child = context.Process(target=_isolated_entry, args=(queue, function, args))
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


def run_fortran(module, export, params, outs, inputs, expects):
    # Compiled and executed through the packed Fortran core ABI when a
    # compiler is on the machine; without one, emission is all that can be
    # checked, and the line says so rather than passing.
    from src.compiler.ssa_fortran_backend import (
        FortranCoreNative, emit_module, fortran_compiler,
    )
    try:
        # The export is the root, as in the C/LLVM lanes (which emit only
        # ``export``) and the compilation product (``extra_roots=
        # export_names``).  Without it every control function became an
        # entry: c829/c855/c856's inlined ``same(a)`` helper was exported
        # on its own and refused for returning its heap list.
        fortran_module = emit_module(
            module, extra_roots=(export,), progress=lambda _: None,
        )
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
            entrypoint=export, name=export[:48], standalone=FORTRAN_STANDALONE,
        )
        record = next(e for e in fortran_module.api.entry_points if str(e.name) == export)
        native = FortranCoreNative(built.executable_path, record, fortran_module.api)
    except Exception as e:
        return f"COMPILE-FAIL {type(e).__name__}: {str(e)[:300]}", None
    results, values = [], []
    for feeds, expect in zip(inputs, expects):
        try:
            ex = native.prepare_execution(_feeds(module, export, params, feeds)).run()
            flat, kinds = output_values(module, export, outs, lambda i: ex.buffers[i])
            values.append(flat)
            results.append(judge(feeds, expect, flat, kinds))
        except Exception as e:
            values.append(None)
            results.append(f"RUN-FAIL {type(e).__name__}: {str(e)[:150]}")
    bad = [r for r in results if r != "ok"]
    return f"{'PASS' if not bad else 'FAIL'} {bad[:3]}", values


def run_wasm(module, export, params, outs, inputs, expects):
    # The independent repository-SSA WebAssembly core (the mobile PWA's
    # execution product), run in node.  Same verdicts as the other lanes.
    from src.compiler.ssa_wasm_backend import (
        emit_ssa_module_to_wasm_core, prepare_wasm_core_execution,
    )
    try:
        art = emit_ssa_module_to_wasm_core(module, export, entry_name=export)
    except Exception as e:
        return f"EMIT-FAIL {type(e).__name__}: {str(e)[:300]}", None
    if not art.complete:
        return f"EMIT-INCOMPLETE {[str(s.reason)[:150] for s in art.shortfalls[:2]]}", None
    # Scalar formals are wasm parameters passed by value; buffers are fed
    # one-element arrays, as in the other lanes.
    by_value = {value_id for kind, value_id in art.parameters if kind != "buffer"}
    results, values = [], []
    for feeds, expect in zip(inputs, expects):
        ex = None
        try:
            fed = _feeds(module, export, params, feeds)
            ex = prepare_wasm_core_execution(art, {
                value_id: (cell[0].item() if value_id in by_value else cell)
                for value_id, cell in fed.items()
            }).run()
            flat, kinds = output_values(module, export, outs, lambda i: ex.buffers[i])
            values.append(flat)
            results.append(judge(feeds, expect, flat, kinds))
        except Exception as e:
            values.append(None)
            results.append(f"RUN-FAIL {type(e).__name__}: {str(e)[:150]}")
        finally:
            if ex is not None:
                ex.close()
    bad = [r for r in results if r != "ok"]
    return f"{'PASS' if not bad else 'FAIL'} {bad[:3]}", values


def disagreements(inputs, c_values, other_values):
    return [
        f"{feeds} c={c} other={o}"
        for feeds, c, o in zip(inputs, c_values, other_values)
        if c is not None and o is not None and (
            len(c) != len(o)
            or any(
                (a is None) != (b is None)
                or type(a) is not type(b)
                or (a is not None and abs(float(a) - float(b)) > 1e-12 * max(1.0, abs(float(a))))
                for a, b in zip(c, o)
            )
        )
    ]


def _unannotated_formals(src):
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "root":
            a = node.args
            return [x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs) if x.annotation is None]
    return []


def main():
    for path in sorted(pathlib.Path(sys.argv[1]).glob("c*.py")):
        src = path.read_text()
        inputs = ast.literal_eval(src.split("# INPUTS", 1)[1].strip())
        ns = {"__name__": "corpus_" + path.stem}
        exec(compile(src, str(path), "exec"), ns)
        expects = [ns["root"](**feeds) for feeds in inputs]
        # The harness IS the caller: for formals the program leaves
        # unannotated, each INPUTS row is a root call record whose argument
        # classes feed the compiler's called inventory.  One exact-typed
        # specialized copy of the entry is lowered per distinct class
        # signature; annotated formals are untouched.
        free = _unannotated_formals(src)
        if not free:
            check_group(path, src, inputs, expects, None)
            continue
        from src.compiler.called_inventory import class_spelling
        groups = {}
        for feeds, expect in zip(inputs, expects):
            sig = tuple((k, class_spelling(feeds[k])) for k in sorted(free))
            groups.setdefault(sig, []).append((feeds, expect))
        calls = [{k: f[k] for k in free} for f in inputs]
        for sig, rows in groups.items():
            label = path.stem + ("[" + ",".join(c for _, c in sig) + "]")
            check_group(path, src, [r[0] for r in rows], [r[1] for r in rows],
                        (calls, dict(sig)), label)


def check_group(path, src, inputs, expects, inventory, label=None):
    if True:
        label = label or path.stem
        # The harness feeds each root parameter as a one-element array.  A
        # container has no such form (its storage, length and capacity are
        # its own ABI), so running it reported a false WRONG for a program
        # nothing had judged.  Say what is true: it was not run.
        containers = sorted({
            k for feeds in inputs for k, v in feeds.items()
            if isinstance(v, (list, tuple, dict, set, frozenset)) and not _scalar_list(v)
        })
        if containers:
            print(f"{label}: CANNOT-FEED container root parameter(s) {containers}; not run (only lists of numbers are fed)")
            return
        try:
            module, _, exports = lower_ast_source_to_ssa(src, "root", name="corpus_" + path.stem + ("" if inventory is None else "__" + "_".join(v.replace("[", "_").replace("]", "") for v in inventory[1].values())), extraction_contract=ROOT + "/extraction_contracts/program_extraction.yaml", progress=lambda _: None, **({} if inventory is None else dict(root_calls=inventory[0], root_signature=inventory[1])))
        except Exception as e:
            print(f"{label}: LOWER-FAIL {type(e).__name__}: {str(e)[:300]}"); return
        rep = " ".join(l.strip() for l in concordance_report(module, limit=3).splitlines()[:1] + [l for l in concordance_report(module, limit=3).splitlines() if l.startswith("     ")][:2])
        try:
            root = module.functions[exports[0]]
            params = dict(root.metadata['parameter_names'])
            from src.compiler.project_compilation_product import _unexplained_root_argument_ids
            if _unexplained_root_argument_ids(root):
                print(f"{label}: PRODUCT-REFUSES unexplained root inputs {_unexplained_root_argument_ids(root)} | {rep}"); return
            outs = [i.args for b in root.blocks.values() for i in b.instrs if i.op == 'Ret'][0]
            art = emit_ssa_to_c(module, exports[0])
            if not art.complete:
                print(f"{label}: EMIT-INCOMPLETE {art.shortfalls[:2]} | {rep}"); return
            d = pathlib.Path(tempfile.mkdtemp()); art.compile(d / "native")
        except Exception as e:
            print(f"{label}: EMIT-FAIL {type(e).__name__}: {str(e)[:300]} | {rep}"); return
        results = []
        c_values = []
        for feeds, expect in zip(inputs, expects):
            try:
                ex = art.prepare_execution(_feeds(module, exports[0], params, feeds)).run()
                flat, kinds = output_values(module, exports[0], outs, lambda i: ex.buffers[i])
                c_values.append(flat)
                results.append(judge(feeds, expect, flat, kinds))
            except Exception as e:
                c_values.append(None)
                results.append(f"RUN-FAIL {type(e).__name__}: {str(e)[:150]}")
        bad = [r for r in results if r != "ok"]
        print(f"{label}: {'PASS' if not bad else 'FAIL'} {bad[:3]} | {rep}")
        if "llvm" in BACKENDS:
            verdict, llvm_values = run_isolated(run_llvm, module, exports[0], params, outs, inputs, expects)
            print(f"{label}: LLVM {verdict}")
            found = disagreements(inputs, c_values, llvm_values or ())
            if found:
                print(f"{label}: DISAGREE c/llvm {found[:2]}")
        if "fortran" in BACKENDS:
            verdict, fortran_values = run_isolated(run_fortran, module, exports[0], params, outs, inputs, expects)
            print(f"{label}: FORTRAN {verdict}")
            found = disagreements(inputs, c_values, fortran_values or ())
            if found:
                print(f"{label}: DISAGREE c/fortran {found[:2]}")
        if "wasm" in BACKENDS:
            verdict, wasm_values = run_isolated(run_wasm, module, exports[0], params, outs, inputs, expects)
            print(f"{label}: WASM {verdict}")
            found = disagreements(inputs, c_values, wasm_values or ())
            if found:
                print(f"{label}: DISAGREE c/wasm {found[:2]}")


if __name__ == "__main__":
    # TYPED_CHECK_NUMERIC_MODE=compute runs the corpus under the compute
    # numeric contract (transcendentals: backend_intrinsic); unset, the
    # contract's default mode (scientific: signal_cores) applies.
    from src.compiler.ssa_call_input_adapters import (
        active_numeric_mode, numeric_mode_scope,
    )
    with numeric_mode_scope(
        os.environ.get("TYPED_CHECK_NUMERIC_MODE") or active_numeric_mode()
    ):
        main()
