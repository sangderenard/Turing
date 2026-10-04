"""Compiled `torch.linalg.solve` against NumPy over a matrix of cases.

usage: probe_solve_matrix.py {c|llvm} [GROUP ...]

GROUPS: s<n>_k<k|v>_<dtype> ..., twice, follow, mixed_refused (default: all
but mixed_refused)

One lowering per lane per group (a group is one function that solves several
systems and returns every result).  Same entry, real program_extraction
contract with a declared ABI, `-O0`, and SENTINEL-poisoned output buffers as
probe_solve_c_lane.py.  Prints one table row per case:
    case | lane | shortfalls | max abs err | MATCH/MISMATCH
A case is a MATCH when its max abs error is within its declared tolerance
(relative to max(1, max|expected|); looser for float32 and for conditioning).
"""

from pathlib import Path
import sys
import time

import numpy as np

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

SENTINEL = -12345.0
rng = np.random.default_rng(7)


def dominant(n, dtype=np.float64):
    return (rng.standard_normal((n, n)) + np.eye(n) * 0.5).astype(dtype)


def rhs(n, k=None, dtype=np.float64):
    shape = (n,) if k is None else (n, k)
    return rng.standard_normal(shape).astype(dtype)


F64, F32 = np.float64, np.float32
ZERO_LEAD = np.asarray([[0.0, 2.0, 1.0], [1.0, 1.0, 1.0], [3.0, 0.0, 2.0]])
TIED = np.asarray([[1.0, 2.0, 3.0], [1.0, 0.0, 1.0], [0.0, 1.0, 4.0]])
TIED4 = np.asarray([[2.0, 1.0, 0.0, 1.0], [-2.0, 3.0, 1.0, 0.0],
                    [2.0, 0.0, 5.0, 1.0], [0.0, 1.0, 1.0, 4.0]])
ILL = np.asarray([[1e-6, 2e-6, 3e-6], [4.0, 1.0, 2.0], [1e6, 3e6, 1e6]])
ILL2 = np.asarray([[1e-8, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1e8]])
HILB5 = 1.0 / (np.arange(5)[:, None] + np.arange(5)[None, :] + 1.0)
NEAR = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0 + 1e-7]])
NEAR2 = np.asarray([[1.0, 1.0], [1.0, 1.0 + 1e-9]])
A_TWO_1 = np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 9.0]])
A_TWO_2 = np.asarray([[5.0, 1.0, 0.0], [2.0, 0.0, 1.0], [1.0, 3.0, 7.0]])
A_FOL = np.asarray([[0.0, 3.0, 1.0], [2.0, 1.0, 0.0], [1.0, 1.0, 4.0]])


def case(name, A, b, dtype=F64, tol=1e-9, post=None):
    """`tol` is the declared acceptable error relative to max(1, max|x|);
    conditioning-limited cases state it from cond(A) * eps."""
    return dict(name=name, A=np.asarray(A, dtype=dtype),
                b=np.asarray(b, dtype=dtype), dtype=dtype, tol=tol, post=post)


def _all_cases():
    cases = [
        *[case(f"size{n}", dominant(n), rhs(n)) for n in (2, 3, 4, 5)],
        case("pivot_zero_lead", ZERO_LEAD, [1.0, 2.0, 3.0]),
        case("pivot_tied3", TIED, [1.0, 2.0, 3.0]),
        case("pivot_tied4", TIED4, [1.0, 2.0, 3.0, 4.0]),
        case("pivot_ill_scaled", ILL, [1.0, 2.0, 3.0], tol=1e-7),
        case("pivot_ill_scaled2", ILL2, [1.0, 2.0, 3.0], tol=1e-7),
        case("near_singular_3x3", NEAR, [1.0, 2.0, 3.0], tol=1e-4),
        case("near_singular_2x2", NEAR2, [1.0, 2.0], tol=1e-4),
        case("hilbert5", HILB5, [1.0, 2.0, 3.0, 4.0, 5.0], tol=1e-6),
        case("multirhs_3x3_k2", ZERO_LEAD, rhs(3, 2)),
        case("multirhs_4x4_k3", TIED4, rhs(4, 3)),
        case("multirhs_2x2_k1", dominant(2), rhs(2, 1)),
        case("f32_size3_pivot", ZERO_LEAD, [1.0, 2.0, 3.0], F32, 1e-4),
        case("f32_size4_tied", TIED4, [1.0, 2.0, 3.0, 4.0], F32, 1e-4),
        case("f32_size5", dominant(5, F32), rhs(5, None, F32), F32, 1e-3),
        case("f32_multirhs", ZERO_LEAD, rhs(3, 2), F32, 1e-4),
    ]
    groups = {}
    # A group holds the cases whose (A shape, b shape, dtype) agree: one
    # lowering of one function solving each of them.  A function that calls
    # solve at two different shapes is refused at lowering (ConcordanceRefusal
    # copy_value_shape REVISE, shape-polymorphic shared callee), which is a
    # lowering refusal and is reported separately.
    for c in cases:
        key = (f"s{c['A'].shape[0]}_k"
               f"{c['b'].shape[1] if c['b'].ndim == 2 else 'v'}_"
               f"{np.dtype(c['dtype']).name}")
        groups.setdefault(key, []).append(c)
    groups["twice"] = [
        case("twice_first", A_TWO_1, [1.0, 2.0, 3.0]),
        case("twice_second", A_TWO_2, [4.0, 5.0, 6.0]),
    ]
    groups["follow"] = [
        case("follow_x2p1", A_FOL, [1.0, 2.0, 3.0], post="x*2+1"),
        case("follow_matmul_A_x", A_FOL, [1.0, 2.0, 3.0], post="matmul"),
        case("follow_sum_x", A_FOL, [1.0, 2.0, 3.0], post="sum"),
    ]
    return groups


GROUPS = _all_cases()
# Opt-in: the mixed-shape lowering the refusal is reproduced with.
GROUPS["mixed_refused"] = [
    case("mixed_size2", dominant(2), rhs(2)),
    case("mixed_size3", dominant(3), rhs(3)),
]

POSTS = {
    "x*2+1": ("{x} * 2 + 1", lambda A, b, x: x * 2 + 1),
    "matmul": ("torch.matmul({A}, {x})", lambda A, b, x: A @ x),
    "sum": ("{x}.sum()", lambda A, b, x: np.asarray(x.sum())),
}


def build_source(cases):
    params, body, rets = [], [], []
    for i, c in enumerate(cases):
        params += [f"m{i}: torch.Tensor", f"r{i}: torch.Tensor"]
        body.append(f"    x{i} = torch.linalg.solve(m{i}, r{i})")
        if c["post"]:
            tmpl = POSTS[c["post"]][0]
            body.append(f"    y{i} = " + tmpl.format(A=f"m{i}", x=f"x{i}"))
            rets.append(f"y{i}")
        else:
            rets.append(f"x{i}")
    return ("import torch\n\ndef solve_group(" + ", ".join(params) + "):\n"
            + "\n".join(body) + "\n    return (" + ", ".join(rets) + ",)\n")


def expected_of(c):
    A64, b64 = c["A"].astype(F64), c["b"].astype(F64)
    x = np.linalg.solve(A64, b64)
    if c["post"]:
        x = POSTS[c["post"]][1](A64, b64, x)
    return x


def run_group(lane, group, cases):
    tag = f"solve_matrix_{lane}_{group}"
    values = []
    for i, c in enumerate(cases):
        dt = np.dtype(c["dtype"]).name
        values.append({"function": "solve_group", "parameter": f"m{i}",
                       "storage": "span", "dtype": dt, "rank": 2,
                       "shape": list(c["A"].shape),
                       "python_type": "AbstractTensor"})
        values.append({"function": "solve_group", "parameter": f"r{i}",
                       "storage": "span", "dtype": dt, "rank": c["b"].ndim,
                       "shape": list(c["b"].shape),
                       "python_type": "AbstractTensor"})
    contract = ExtractionContract(
        root / "extraction_contracts" / "program_extraction.yaml"
    ).with_program_abi({"records": {}, "bindings": [], "values": values})
    source = build_source(cases)
    t0 = time.time()
    try:
        module, outputs, _exports = lower_ast_source_to_ssa(
            source, "solve_group", name=tag, extraction_contract=contract,
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            progress=lambda m: print("PROGRESS", m, flush=True),
        )
    except Exception as error:  # a lowering refusal is reported, not hidden
        print(f"[{group}] LOWER_REFUSED {type(error).__name__}: "
              f"{str(error)[:3000]}", flush=True)
        return [(c["name"], -1, None, "LOWER_REFUSED") for c in cases]
    print(f"[{group}] LOWERED in {time.time() - t0:.1f}s", flush=True)
    qualified = f"{tag}__solve_group"
    function = module.functions[qualified]
    parameters = dict(function.metadata["parameter_names"])
    published = [int(v.id) for v in outputs[qualified]]
    expected = [expected_of(c) for c in cases]
    t0 = time.time()
    if lane == "c":
        from src.compiler.ssa_c_backend import emit_ssa_module_to_c
        artifact = emit_ssa_module_to_c(module, qualified)
        shortfalls = len(artifact.shortfalls)
        print(f"[{group}] C_COMPLETE {artifact.complete} shortfalls {shortfalls}",
              flush=True)
        for s in artifact.shortfalls:
            print("  C_SHORTFALL", s.operation, str(s.reason)[:400], flush=True)
        if not artifact.complete:
            return [(c["name"], shortfalls, None, "INCOMPLETE") for c in cases]
        artifact.compile(root / "build" / tag, optimization="O0")
        prepare = artifact.prepare_execution
    else:
        from src.compiler.ssa_llvm_backend import (
            compile_artifact, emit_ssa_function_to_llvm,
            prepare_artifact_execution,
        )
        artifact = emit_ssa_function_to_llvm(module, qualified, entry_name=tag)
        shortfalls = len(artifact.shortfalls)
        print(f"[{group}] LLVM shortfalls {shortfalls}", flush=True)
        for s in artifact.shortfalls:
            print("  LLVM_SHORTFALL", s, flush=True)
        if shortfalls:
            return [(c["name"], shortfalls, None, "INCOMPLETE") for c in cases]
        native = compile_artifact(artifact, directory=root / "build" / tag,
                                  optimization="O0")
        prepare = lambda feeds: prepare_artifact_execution(native, feeds)
    print(f"[{group}] BUILT in {time.time() - t0:.1f}s", flush=True)

    feeds = {}
    for i, c in enumerate(cases):
        feeds[parameters[f"m{i}"]] = c["A"].copy()
        feeds[parameters[f"r{i}"]] = c["b"].copy()
    if len(published) != len(cases):
        print(f"[{group}] NOTE published={len(published)} cases={len(cases)}",
              flush=True)
    for vid, c, e in zip(published, cases, expected):
        feeds[vid] = np.full(np.shape(e), SENTINEL, dtype=c["dtype"])
    execution = prepare(feeds).run()
    rows = []
    for vid, c, e in zip(published, cases, expected):
        buf = execution.buffers.get(vid)
        if buf is None:
            rows.append((c["name"], shortfalls, None, "MISMATCH(no buffer)"))
            continue
        a = np.asarray(buf)
        if a.size != np.size(e):
            rows.append((c["name"], shortfalls, None,
                         f"MISMATCH(shape {a.shape} vs {np.shape(e)})"))
            continue
        a = a.reshape(np.shape(e))
        info = " untouched" if np.all(a == SENTINEL) else ""
        if a.dtype != np.dtype(c["dtype"]):
            info += f" dtype={a.dtype}"
        print(f"  {c['name']}: produced {a.ravel()[:6]} expected {np.ravel(e)[:6]}",
              flush=True)
        err = float(np.max(np.abs(a.astype(F64) - e)))
        scale = max(1.0, float(np.max(np.abs(e))))
        ok = np.isfinite(err) and err <= c["tol"] * scale
        rows.append((c["name"], shortfalls, err,
                     ("MATCH" if ok else "MISMATCH") + info))
    return rows


lane = sys.argv[1]
wanted = sys.argv[2:] or [g for g in GROUPS if g != 'mixed_refused']
table = []
for group in wanted:
    for row in run_group(lane, group, GROUPS[group]):
        table.append(row)
print()
print(f"{'case':24s} {'lane':5s} {'shortfalls':>10s} {'max_abs_err':>12s}  result")
for name, sf, err, res in table:
    e = "n/a" if err is None else f"{err:.3e}"
    print(f"{name:24s} {lane:5s} {sf:10d} {e:>12s}  {res}")
