"""Compile `torch.linalg.solve` on a 2x2 contract and check it against NumPy.

Same entry and same discipline as `tests/test_native_shaped_view_lowering.py`,
on the LLVM lane instead of the C lane: `lower_ast_source_to_ssa` for the
program, `-O0` for the build, and the executed result compared against the
eager reference.  A shape or identity mistake in this path does not raise --
it produces a complete, compiling program that reads the wrong elements -- so
value comparison is the only honest gate.

Outputs are filled with NaN before the run, as that harness does.  A buffer
the program never writes then reads as NaN instead of as 0.0, which is a
plausible number and has already cost a wrong diagnosis once.
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import (
    concordance_report,
    identity_book,
    render_row,
)
from src.compiler.ssa_llvm_backend import (
    compile_artifact,
    emit_ssa_function_to_llvm,
    prepare_artifact_execution,
)

SOURCE = """
import torch

def solve_two(matrix: torch.Tensor, rhs: torch.Tensor):
    return torch.linalg.solve(matrix, rhs)
"""

MATRIX = np.asarray([[4.0, 1.0], [2.0, 3.0]], dtype=np.float64)
RHS = np.asarray([1.0, 2.0], dtype=np.float64)

root = Path(__file__).resolve().parents[2]
contract = ExtractionContract(
    root / "extraction_contracts" / "program_extraction.yaml"
).with_program_abi({
    "records": {},
    "bindings": [],
    "values": [
        {
            "function": "solve_two", "parameter": "matrix",
            "storage": "span", "dtype": "float64", "rank": 2,
            "shape": list(MATRIX.shape), "python_type": "AbstractTensor",
        },
        {
            "function": "solve_two", "parameter": "rhs",
            "storage": "span", "dtype": "float64", "rank": 1,
            "shape": list(RHS.shape), "python_type": "AbstractTensor",
        },
    ],
})

module, outputs, _exports = lower_ast_source_to_ssa(
    SOURCE,
    "solve_two",
    name="abstract_solve_numeric",
    extraction_contract=contract,
    tensor_ssa_reference=c_backend_repository_ssa_reference(),
    progress=lambda message: print("PROGRESS", message, flush=True),
)

# The concordance is the backbone diagnostic; read it before anything else.
print(concordance_report(module, limit=8), flush=True)
from src.compiler.identity_concordance import (
    CorrelationTable as _CT, identity_book as _ib,
    loop_scope_declarations as _lsd,
)
for _fd in _CT.build(module).findings(module):
    if _fd.kind.startswith("loop-scope") or _fd.kind == "stale-carried-read":
        print("  >>", _fd.kind, _fd.function, _fd.value_id, _fd.detail,
              flush=True)

# What each scheduled callsite resolved its arguments to.  A graph id that
# resolves to itself inside a loop body means the carried machinery never
# mapped it to the current iteration.
_book = _ib(module)
_outer = set()
for _name in module.functions:
    for _decl in _lsd(_book, str(_name)):
        _outer.update(int(r["outer"]) for r in _decl["rebinds"])
_page = _book.page("callsite_argument")
print("CALLSITE ARGUMENTS: outer generations =", sorted(_outer), flush=True)
_seen = 0
for _row in _page.rows():
    _fact = _page.latest(_row)
    if not (isinstance(_fact, tuple) and len(_fact) == 3):
        continue
    _graph, _resolved, _in_loop = _fact
    if not _in_loop or int(_graph) not in _outer:
        continue
    _seen += 1
    print(f"  callsite {_row[1]} arg {_row[2]} in {_row[0]}: graph {_graph}"
          f" -> {_resolved}"
          f" {'SELF (unmapped)' if _graph == _resolved else 'mapped'}",
          flush=True)
print(f"CALLSITE ARGUMENTS: {_seen} in-loop use(s) of an outer generation,"
      f" out of {len(_page.rows())} recorded argument(s)", flush=True)

# Every operand rewrite is receipted in function metadata already.
print("STRUCTURAL REBINDINGS touching a declared loop scope:", flush=True)
_hits = 0
for _fname, _fn in module.functions.items():
    _decls = _lsd(_book, str(_fname))
    if not _decls:
        continue
    _gen = {}
    for _d in _decls:
        for _r in _d["rebinds"]:
            _gen[int(_r["outer"])] = "outer"
            _gen[int(_r["carried"])] = "carried"
            _gen[int(_r["inner"])] = "inner"
    for _c in _fn.metadata.get("structural_identity_rebindings", ()):
        _was, _now = int(_c["value_id"]), int(_c["replacement_value_id"])
        if _was not in _gen and _now not in _gen:
            continue
        _hits += 1
        print(f"  {_fname.split('__')[-1]} {_c['block']}"
              f"#{_c['instruction_index']} operand {_c['operand_index']}:"
              f" {_was} ({_gen.get(_was, '-')}) -> {_now}"
              f" ({_gen.get(_now, '-')})"
              f" [{_c.get('priority')} / {_c.get('tie_policy')}]", flush=True)
print(f"STRUCTURAL REBINDINGS: {_hits} touching a loop scope", flush=True)
book = identity_book(module)
_loop_transitions = book.page("loop_scope_inner_transition")
print("LOOP INNER TRANSITIONS", [
    (row, _loop_transitions.latest(row)) for row in _loop_transitions.rows()
], flush=True)
for _pivot_name, _pivot_fn in module.functions.items():
    if "___pivot_mask__specialized_" not in _pivot_name or "planned_region" in _pivot_name:
        continue
    print("PIVOT FORMALS", _pivot_name, [
        (int(v.id), v.dtype, tuple(v.shape or ()), dict(v.accounting or {}))
        for v in _pivot_fn.args
    ], flush=True)
    print("PIVOT METADATA", {
        key: _pivot_fn.metadata.get(key)
        for key in ("parameter_names", "storage_formals", "closure_formals")
    }, flush=True)
    for _caller_name, _caller_fn in module.functions.items():
        for _block_name, _block in _caller_fn.blocks.items():
            for _instruction in _block.instrs:
                if str(_instruction.attributes.get("callee") or "") != _pivot_name:
                    continue
                print("PIVOT CALL", _caller_name, _block_name, [
                    (int(v.id), v.dtype, dict(v.accounting or {}))
                    for v in _instruction.args
                ], dict(_instruction.attributes), flush=True)
for page_name, page in sorted(book.pages.items()):
    oscillating = page.oscillating_rows()
    if oscillating:
        print(f"OSCILLATING {page_name}: {len(oscillating)} row(s)", flush=True)
        for row, runs in list(oscillating.items())[:5]:
            print("   ", render_row(row),
                  [fact for _start, _end, fact in runs][:6], flush=True)
    if page_name in {"value_shape", "call_edge"}:
        print(f"PAGE {page_name}: {len(page.rows())} row(s)", flush=True)

# What the three shortfalls actually need: does each helper's formal have a
# call edge at all, and did a shape ever land on it?
shapes = book.page("value_shape")
edges = book.page("call_edge")
# Every value the descriptor query answered with a bare scalar while its
# operation is one that produces a tensor.  A wrong shape anywhere in the
# program appears here with the operation that produced it.
descriptors = book.page("proven_shape")
SCALAR_PRODUCERS = {
    "", "const", "constant", "input", "getattr", "dim", "ndim", "ndims",
    "numel", "len", "int", "float", "range", "add", "sub", "mul", "div",
    "truediv", "floordiv", "mod", "lt", "le", "gt", "ge", "eq", "ne",
}
print("SUSPECT SCALARS (tensor-producing operation, no extents):", flush=True)
for row in sorted(descriptors.rows(), key=str):
    if str(row[0]) == "_lu_decompose_inplace" and int(row[1]) in (
        3, 11, 15, 20, 22, 101, 107
    ):
        print("    PROVEN", row, descriptors.latest(row), flush=True)
for name in ("formal_literal", "formal_shape"):
    page = book.page(name)
    print(f"    {name}: {len(page.rows())} row(s)", flush=True)
    for row in sorted(page.rows(), key=str):
        print("      ", row, page.latest(row), flush=True)
stores = {name: book.page(f"shape.{name}") for name in ("node", "linked", "ssa")}
stores["proven"] = descriptors
rows = set()
for page in stores.values():
    rows.update(page.rows())

def _extents(name, row):
    page = stores[name]
    if row not in set(page.rows()):
        return None
    fact = page.latest(row)
    if fact is None:
        return None
    if name == "proven":
        return tuple(fact[1]) if fact[0] == "proven" else None
    return tuple(fact)

print("SHAPE STORES (values whose stores disagree):", flush=True)
disagreeing = 0
for row in sorted(rows, key=str):
    present = {}
    for name in stores:
        value = _extents(name, row)
        if value is not None:
            present[name] = value
    if len({tuple(v) for v in present.values()}) > 1:
        disagreeing += 1
        if disagreeing <= 10:
            print("   ", row, present, flush=True)
print(f"    {disagreeing} disagreeing of {len(rows)} identit(ies)", flush=True)
for name, page in stores.items():
    print(f"    store {name}: {len(page.rows())} row(s)", flush=True)
conflicting = [
    row for row in descriptors.rows()
    if (descriptors.latest(row) or ("",))[0] == "conflicting"
]
print(f"   proven={len(descriptors.rows())} conflicting={len(conflicting)}",
      flush=True)
for row in sorted(conflicting, key=str)[:10]:
    print("    CONFLICT", render_row(row),
          [f for _c, f in descriptors.history(row)][:4], flush=True)

incoming = {}
for row in edges.rows():
    incoming.setdefault((row[0], row[1]), []).append((row[2], row[3]))
for row in sorted(set(shapes.rows()) | set(incoming)):
    history = [fact for _column, fact in shapes.history(row)]
    settled = history[-1] if history else None
    # Say so loudly when a value that callers feed never settled a shape.
    unsettled = settled is None or settled[0] == "polymorphic"
    print("   ", "UNSETTLED" if unsettled else "settled  ",
          render_row(row),
          "from", incoming.get(row, []),
          "|", history, flush=True)

for _n, _f in module.functions.items():
    if not _n.endswith("___back_substitute"):
        continue
    print("FORMALS", [int(a.id) for a in getattr(_f, "args", ())], flush=True)
    print("BLOCKS", list(_f.blocks), flush=True)
    for _want in (20, 24, 12, 57):
        _where = [(b, i, d.op) for b, blk in _f.blocks.items()
                  for i, d in enumerate(blk.instrs)
                  if d.res is not None and int(d.res.id) == _want]
        print("  def", _want, "->", _where or "NOT DEFINED (formal? "
              + str(_want in {int(a.id) for a in getattr(_f, "args", ())}) + ")",
              flush=True)
    _e = _f.blocks["entry"]
    for _i3, _d3 in enumerate(_e.instrs):
        if (_d3.res is not None and int(_d3.res.id) == 63) or 63 in [
            int(getattr(a, "id", -1)) for a in _d3.args
        ]:
            print("SEED entry#%d" % _i3, _d3.op,
                  "res=", getattr(_d3.res, "id", None),
                  "args=", [int(getattr(a, "id", -1)) for a in _d3.args],
                  "attrs=", dict(_d3.attributes), flush=True)
    _uses63 = [(b, i, d.op) for b, blk in _f.blocks.items()
               for i, d in enumerate(blk.instrs)
               if 63 in [int(getattr(a, "id", -1)) for a in d.args]]
    print("USES OF 63:", _uses63, flush=True)
    for _bn in ("loop_body", "loop_header"):
        _blk = _f.blocks.get(_bn)
        if _blk is None:
            continue
        print("== block", _bn, flush=True)
        for _i, _d in enumerate(_blk.instrs):
            print(f"  {_i:3d} {_d.op:14s} res={getattr(_d.res,'id',None)} "
                  f"args={[int(getattr(a,'id',-1)) for a in _d.args]}", flush=True)
            print(f"        ALLATTRS {dict(_d.attributes)}"[:400], flush=True)
    for _cn, _cf in module.functions.items():
        if not _cn.endswith("__planned_region_5"):
            continue
        print("== planned_region_5", _cn,
              "formals=", [int(a.id) for a in getattr(_cf, "args", ())], flush=True)
        for _bn2, _blk2 in _cf.blocks.items():
            print("  --", _bn2, flush=True)
            for _i2, _d2 in enumerate(_blk2.instrs):
                if _d2.op == "Const":
                    continue
                print(f"   {_i2:3d} {_d2.op:14s} res={getattr(_d2.res,'id',None)} "
                      f"args={[int(getattr(a,'id',-1)) for a in _d2.args]} "
                      f"attrs={ {k:v for k,v in _d2.attributes.items() if k in ('callee','index','axis','value','binding')} }",
                      flush=True)
    _hdr = _f.blocks.get("loop_header")
    for _ins in (_hdr.instrs if _hdr else []):
        if str(_ins.op).lower() != "phi":
            continue
        print("PHIATTRS", {k: v for k, v in _ins.attributes.items()}, flush=True)
        print("HDRPHI res=", getattr(_ins.res, "id", None),
              "args=", [int(getattr(a, "id", -1)) for a in _ins.args],
              "binding=", _ins.attributes.get("binding"),
              "incoming=", _ins.attributes.get("incoming_blocks"), flush=True)
        for _a in _ins.args:
            _aid = int(getattr(_a, "id", -1))
            for _b, _blk in _f.blocks.items():
                for _i, _d in enumerate(_blk.instrs):
                    if _d.res is not None and int(_d.res.id) == _aid:
                        print("   def of", _aid, "at", _b, _i, _d.op,
                              "attrs=", {k: v for k, v in _d.attributes.items()
                                          if k in ("callee", "ssa_output_argument", "binding")},
                              flush=True)
    break

for _n, _f in module.functions.items():
    if not _n.endswith("___back_substitute"):
        continue
    print("FORMALS", [int(a.id) for a in getattr(_f, "args", ())], flush=True)
    print("BLOCKS", list(_f.blocks), flush=True)
    for _want in (20, 24, 12, 63):
        _where = [(b, i, d.op) for b, blk in _f.blocks.items()
                  for i, d in enumerate(blk.instrs)
                  if d.res is not None and int(d.res.id) == _want]
        print("  def", _want, "->", _where or "NOT DEFINED", flush=True)
    _uses63 = [(b, i, d.op) for b, blk in _f.blocks.items()
               for i, d in enumerate(blk.instrs)
               if 63 in [int(getattr(a, "id", -1)) for a in d.args]]
    print("USES OF 63:", _uses63, flush=True)
    for _bn in ("loop_body", "loop_header"):
        _blk = _f.blocks.get(_bn)
        if _blk is None:
            continue
        print("== block", _bn, flush=True)
        for _i, _d in enumerate(_blk.instrs):
            print(f"  {_i:3d} {_d.op:14s} res={getattr(_d.res,'id',None)} "
                  f"args={[int(getattr(a,'id',-1)) for a in _d.args]}",
                  flush=True)
            print(f"        ALLATTRS {dict(_d.attributes)}"[:400], flush=True)
    for _cn, _cf in module.functions.items():
        if not _cn.endswith("___back_substitute__planned_region_5"):
            continue
        print("== planned_region_5 formals=",
              [int(a.id) for a in getattr(_cf, "args", ())], flush=True)
        for _bn2, _blk2 in _cf.blocks.items():
            for _i2, _d2 in enumerate(_blk2.instrs):
                if _d2.op == "Const":
                    continue
                print(f"   {_i2:3d} {_d2.op:14s} "
                      f"res={getattr(_d2.res,'id',None)} "
                      f"args={[int(getattr(a,'id',-1)) for a in _d2.args]} "
                      f"attrs={ {k: v for k, v in _d2.attributes.items() if k == 'callee'} }",
                      flush=True)
    for _ins in _f.blocks["loop_header"].instrs:
        if str(_ins.op).lower() != "phi":
            continue
        print("PHIATTRS", dict(_ins.attributes), flush=True)
        print("HDRPHI res=", getattr(_ins.res, "id", None),
              "args=", [int(getattr(a, "id", -1)) for a in _ins.args],
              flush=True)
        for _a in _ins.args:
            _aid = int(getattr(_a, "id", -1))
            for _b, _blk in _f.blocks.items():
                for _i, _d in enumerate(_blk.instrs):
                    if _d.res is not None and int(_d.res.id) == _aid:
                        print("   def of", _aid, "at", _b, _i, _d.op,
                              flush=True)
    break

reconciliation = book.page("loop_result_reconciliation")
outcomes = {}
for row in reconciliation.rows():
    for _column, fact in reconciliation.history(row):
        outcomes[fact[0]] = outcomes.get(fact[0], 0) + 1
print("RECONCILIATION", outcomes, flush=True)
lu_rows = [
    row for row in reconciliation.rows()
    if "lu_decompose" in str(row[0]) and str(row[1]) == "106"
]
print("RECON_106", len(lu_rows), flush=True)
for row in lu_rows[:3]:
    for fact in [f for _c, f in reconciliation.history(row)][:6]:
        print("    106", row[0].split("__specialized_")[0], fact, flush=True)
if not lu_rows:
    any_lu = [row for row in reconciliation.rows() if "lu_decompose" in str(row[0])]
    print("    lu rows on page:", len(any_lu),
          sorted({str(r[1]) for r in any_lu})[:12], flush=True)

qualified = "abstract_solve_numeric__solve_two"
function = module.functions[qualified]
print("ONE HOT CALL EDGES", flush=True)
for _caller_name, _caller_fn in module.functions.items():
    for _block_name, _block in _caller_fn.blocks.items():
        for _instruction_index, _instruction in enumerate(_block.instrs):
            _callee_name = str(_instruction.attributes.get("callee") or "")
            if _instruction.op not in {"Call", "call"} or "one_hot_axis" not in _callee_name:
                continue
            _callee_fn = module.functions.get(_callee_name)
            print("  EDGE", _caller_name, _block_name, _instruction_index,
                  "->", _callee_name, flush=True)
            print("    ACTUAL", [(int(v.id), v.dtype, tuple(v.shape or ()), dict(v.accounting or {}))
                                 for v in _instruction.args], flush=True)
            print("    FORMAL", [(int(v.id), v.dtype, tuple(v.shape or ()), dict(v.accounting or {}))
                                 for v in (_callee_fn.args if _callee_fn else ())], flush=True)
            _previous = _block.instrs[max(0, _instruction_index - 2):_instruction_index]
            print("    PREVIOUS", [(p.op, getattr(p.res, 'id', None), dict(p.attributes))
                                    for p in _previous], flush=True)
_conversion_page = book.page("call_input_conversion")
print("CALL INPUT CONCORDANCE", [
    (row, _conversion_page.latest(row)) for row in _conversion_page.rows()
], flush=True)
artifact = emit_ssa_function_to_llvm(
    module, qualified, entry_name="abstract_solve_numeric_solve_two",
)
print("LLVM_SHORTFALLS", artifact.shortfalls, flush=True)
if artifact.shortfalls:
    raise SystemExit(1)
(root / "build" / "abstract_solve_numeric.ll").write_text(
    artifact.llvm_ir, encoding="utf-8",
)

native = compile_artifact(
    artifact, directory=root / "build" / "abstract_solve_numeric",
    optimization="O0",
)

expected = np.linalg.solve(MATRIX, RHS)
parameters = dict(function.metadata["parameter_names"])
published = [int(value.id) for value in outputs[qualified]]
feeds = {parameters["matrix"]: MATRIX.copy(), parameters["rhs"]: RHS.copy()}
SENTINEL = -12345.0
for value_id in published:
    feeds[value_id] = np.full(np.shape(expected), SENTINEL)

# Poison every buffer, not just the published one.  A buffer the program
# writes comes back as numbers; one it never touches comes back NaN.  The
# first NaN in dependency order is where the computation stops.
_probe = prepare_artifact_execution(native, feeds)
print("ARTIFACT BUFFER ORDER", artifact.buffer_order, flush=True)
print("ARTIFACT BUFFER SHAPES", artifact.buffer_shapes, flush=True)
print("ROOT ARGS", [
    (int(value.id), tuple(value.shape or ()), dict(value.accounting or {}))
    for value in function.args
], flush=True)
for _bid, _buf in _probe.buffers.items():
    if _bid in feeds:
        continue
    # Leave compiler-owned frame storage at its ABI initialization value.
    # Poisoning it here used to conflate an exposed-frame classification bug
    # with a numerical failure in the emitted program.
execution = _probe.run()
_untouched, _wrote_nan, _wrote_real = [], [], []
for _bid, _buf in execution.buffers.items():
    _a = np.asarray(_buf)
    if not _a.size:
        continue
    if np.all(_a == SENTINEL):
        _untouched.append(int(_bid))
    elif np.any(np.isnan(_a)):
        _wrote_nan.append(int(_bid))
    else:
        _wrote_real.append(int(_bid))
print(f"SENTINEL: untouched={len(_untouched)} "
      f"wrote-NaN={len(_wrote_nan)} wrote-real={len(_wrote_real)}", flush=True)
print("SENTINEL wrote-NaN ids:", sorted(_wrote_nan)[:20], flush=True)
print("SENTINEL wrote-real ids:", sorted(_wrote_real)[:20], flush=True)
print("SENTINEL published id:", published, flush=True)
produced = [
    np.asarray(execution.buffers[value_id]).reshape(np.shape(expected))
    for value_id in published
    if value_id in execution.buffers
]
written = {
    key: np.asarray(value).reshape(-1)[:4]
    for key, value in execution.buffers.items()
    if np.asarray(value).size and np.any(np.asarray(value) != 0)
    and not np.all(np.isnan(np.asarray(value)))
}
print("WROTE", len(written), "of", len(execution.buffers),
      "buffers; sample:", {k: list(v) for k, v in list(written.items())[:6]},
      flush=True)
print("EXPECTED", expected, flush=True)
print("PRODUCED", produced, flush=True)
_solve_name = next(
    name for name in module.functions
    if "__solve__specialized_" in name and "planned_region" not in name
)
_solve_fn = module.functions[_solve_name]
_watch_ids = (65, 67, 69, 58, 59, 60, 62)
_watched_artifact = emit_ssa_function_to_llvm(
    module, _solve_name,
    entry_name="abstract_solve_numeric_debug_solve",
    watch=_watch_ids,
)
print("WATCH SHORTFALLS", _watched_artifact.watch_shortfalls, flush=True)
print("WATCH ABI", _watched_artifact.buffer_order, flush=True)
if not _watched_artifact.shortfalls:
    _watched_native = compile_artifact(
        _watched_artifact,
        directory=root / "build" / "abstract_solve_numeric_debug_solve",
        optimization="O0",
    )
    _solve_parameters = dict(_solve_fn.metadata.get("parameter_names", ()))
    _solve_feeds = {
        _solve_parameters.get("matrix", 0): MATRIX.copy(),
        _solve_parameters.get("rhs", 1): RHS.copy(),
    }
    _watched_execution = prepare_artifact_execution(
        _watched_native, _solve_feeds,
    ).run()
    print("WATCH VALUES", {
        value_id: np.asarray(_watched_execution.buffers[value_id]).tolist()
        for value_id in _watched_artifact.buffer_order
        if value_id in set(_watched_artifact.watched)
        or value_id in {60, 62}
    }, flush=True)
_lu_name = next(
    name for name in module.functions
    if "___lu_decompose_inplace__specialized_" in name
    and "planned_region" not in name
)
_lu_fn = module.functions[_lu_name]
_lu_watch_ids = (
    3, 38, 11, 22, 40, 47, 59, 63, 75, 76, 80, 82, 83,
    91, 94, 95, 100, 103, 107, 109, 111, 113, 114, 115, 116,
    2305843010213694219, 2305843010213694220,
    2305843010213694221, 2305843010213694222,
    2305843010213695021, 2305843010213695022,
    2305843010213695023,
)
_lu_artifact = emit_ssa_function_to_llvm(
    module, _lu_name,
    entry_name="abstract_solve_numeric_debug_lu",
    watch=_lu_watch_ids,
)
print("LU WATCH SHORTFALLS", _lu_artifact.watch_shortfalls, flush=True)
if not _lu_artifact.shortfalls:
    _lu_native = compile_artifact(
        _lu_artifact,
        directory=root / "build" / "abstract_solve_numeric_debug_lu",
        optimization="O0",
    )
    _lu_execution = prepare_artifact_execution(
        _lu_native, {int(_lu_fn.args[0].id): MATRIX.copy()},
    ).run()
    print("LU WATCH VALUES", {
        value_id: np.asarray(_lu_execution.buffers[value_id]).tolist()
        for value_id in _lu_artifact.buffer_order
        if value_id in set(_lu_artifact.watched)
    }, flush=True)
if not any(np.allclose(value, expected) for value in produced):
    unwritten = [value for value in produced if np.all(np.isnan(value))]
    raise AssertionError(
        "compiled solve did not publish the NumPy solution"
        + (" (output never written)" if unwritten else "")
    )
print("NUMERIC_MATCH", flush=True)
