import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import identity_book, render_row
import src.compiler.identity_concordance as _identity_concordance
import src.compiler.glsl_deployment_strategy as _deployment

_formal_queries = []
_formal_events = []
_original_proven_formal_shape = _deployment._proven_formal_shape
_original_publish_formal_shape = _deployment._publish_formal_shape

def _traced_proven_formal_shape(graph, name):
    answer = _original_proven_formal_shape(graph, name)
    function_name = str(graph.G.graph.get("function_name"))
    if "first_occurrence" in function_name or "masked_pivot_rows" in function_name:
        _formal_queries.append((function_name, name, answer))
    if "first_occurrence" in function_name and name == "mask":
        _formal_events.append(("query", function_name, name, answer))
    return answer

_deployment._proven_formal_shape = _traced_proven_formal_shape

def _traced_publish_formal_shape(function, parameter, descriptor, caller):
    if "first_occurrence" in str(function) and parameter == "mask":
        _formal_events.append(("publish", function, parameter, dict(descriptor), caller))
    return _original_publish_formal_shape(function, parameter, descriptor, caller)

_deployment._publish_formal_shape = _traced_publish_formal_shape

_pivot_graph_snapshot = []
_original_propagate = _deployment._propagate_callsite_tensor_specializations

def _traced_propagate(graph):
    answer = _original_propagate(graph)
    table = getattr(graph, "function_table", None)
    if table is not None:
        for entry in table:
            candidate = getattr(entry, "graph", None)
            if candidate is None or candidate.G.graph.get("function_name") != "_pivot_mask":
                continue
            for node_id, data in candidate.G.nodes(data=True):
                ref = (data.get("attributes") or {}).get("callee_ref")
                if ref is None:
                    continue
                try:
                    callee = table.entry(int(ref)).graph
                except Exception:
                    continue
                if callee is None or callee.G.graph.get("function_name") != "_first_occurrence":
                    continue
                parents = []
                for parent, role in data.get("parents") or ():
                    parent_data = dict(candidate.G.nodes[int(parent)])
                    grandparents = [
                        (int(gp), str(grole), dict(candidate.G.nodes[int(gp)]))
                        for gp, grole in parent_data.get("parents") or ()
                    ]
                    parents.append((int(parent), str(role), parent_data, grandparents))
                _pivot_graph_snapshot.append((int(node_id), dict(data), parents))
    return answer

_deployment._propagate_callsite_tensor_specializations = _traced_propagate

_original_record_proven_shape = _identity_concordance.record_proven_shape

def _traced_record_proven_shape(function, value_id, extents, dtype, level=0):
    if "masked_pivot_rows" in str(function) and int(value_id) == 2:
        import traceback
        print("RECORD MASKED %2", function, tuple(extents or ()), dtype, level)
        print("".join(traceback.format_stack(limit=8)))
    return _original_record_proven_shape(
        function, value_id, extents, dtype, level,
    )

_identity_concordance.record_proven_shape = _traced_record_proven_shape

SOURCE = """
import torch

def solve_two(matrix: torch.Tensor, rhs: torch.Tensor):
    return torch.linalg.solve(matrix, rhs)
"""
MATRIX = np.asarray([[4.0, 1.0], [2.0, 3.0]], dtype=np.float64)
RHS = np.asarray([1.0, 2.0], dtype=np.float64)
root = Path.cwd()
contract = ExtractionContract(root / "extraction_contracts" / "program_extraction.yaml").with_program_abi({
    "records": {}, "bindings": [],
    "values": [
        {"function": "solve_two", "parameter": "matrix", "storage": "span", "dtype": "float64", "rank": 2, "shape": list(MATRIX.shape), "python_type": "AbstractTensor"},
        {"function": "solve_two", "parameter": "rhs", "storage": "span", "dtype": "float64", "rank": 1, "shape": list(RHS.shape), "python_type": "AbstractTensor"},
    ],
})

def _settlement_trace(frame, event, arg):
    if event != "line":
        return _settlement_trace
    if frame.f_lineno not in {14459, 14463, 14501, 14503, 14555, 14582}:
        return _settlement_trace
    values = frame.f_locals
    caller_graph = values.get("caller_graph")
    callee_graph = values.get("callee_graph")
    if caller_graph is None or callee_graph is None:
        return _settlement_trace
    caller_name = str(caller_graph.graph.get("function_name"))
    callee_name = str(callee_graph.graph.get("function_name"))
    caller_id = values.get("caller_id")
    callee_id = values.get("callee_id")
    watched = (
        ("first_occurrence" in caller_name and "pivot_mask" in callee_name)
        or ("pivot_mask" in caller_name and "lu_decompose" in callee_name)
        or ("lu_decompose" in caller_name and "masked_pivot_rows" in callee_name)
    )
    if watched:
        print("SETTLE", frame.f_lineno, caller_name, caller_id, "->", callee_name, callee_id,
              "source=", values.get("source"), "descriptor=", values.get("descriptor"),
              "existing=", values.get("existing"), flush=True)
        if "first_occurrence" in caller_name and int(caller_id) == 10:
            node = dict(caller_graph.nodes[int(caller_id)])
            print("FIRST_OCCURRENCE OUTPUT NODE", node, flush=True)
            for parent, role in node.get("parents") or ():
                print("  PARENT", parent, role,
                      dict(caller_graph.nodes[int(parent)]), flush=True)
            raise SystemExit("captured first-occurrence descriptor frontier")
    return _settlement_trace

def _global_trace(frame, event, arg):
    if (
        event == "call"
        and frame.f_code.co_filename.endswith("fortran_c_shell.py")
        and frame.f_code.co_name == "_class_surface_ssa_program"
    ):
        return _settlement_trace
    return None

# Enable only while capturing one fixed-point frontier.
# sys.settrace(_global_trace)
module, outputs, _exports = lower_ast_source_to_ssa(
    SOURCE, "solve_two", name="abstract_solve_numeric",
    extraction_contract=contract,
    tensor_ssa_reference=c_backend_repository_ssa_reference(),
    progress=lambda m: None,
)
sys.settrace(None)

owner = next(
    function for name, function in module.functions.items()
    if "___masked_pivot_rows__specialized_" in name
    and "__planned_region" not in name
)
fn = next(
    function for name, function in module.functions.items()
    if name.startswith(owner.name + "__planned_region_0")
)
print("REGION formal 2 object id:", id(fn.args[0]), "shape:", fn.args[0].shape)
print("OWNER formal 2 object id:", id(owner.args[2]), "shape:", owner.args[2].shape)
print("SAME OBJECT?", fn.args[0] is owner.args[2])
print("FORMALS:", [(int(a.id), a.dtype, a.shape) for a in fn.args])
for bname, blk in fn.blocks.items():
    print("== block", bname)
    for idx, instr in enumerate(blk.instrs):
        print(f"  {idx:3d} {instr.op:14s} res={getattr(instr.res,'id',None)} args={[int(getattr(a,'id',-1)) for a in instr.args]} attrs={dict(instr.attributes)}"[:300])

print("== OWNER CALLS INTO _row ==")
for bname, blk in owner.blocks.items():
    for idx, instr in enumerate(blk.instrs):
        callee = str(instr.attributes.get("callee") or "")
        if "___row__specialized" not in callee:
            continue
        print(bname, idx, callee)
        print("  args", [(int(a.id), a.dtype, tuple(a.shape or ()), dict(a.accounting or {})) for a in instr.args])
        print("  attrs", dict(instr.attributes))
        target = module.functions[callee]
        print("  formals", [(int(a.id), a.dtype, tuple(a.shape or ()), dict(a.accounting or {})) for a in target.args])

print("== CALLERS INTO _masked_pivot_rows ==")
for caller_name, caller in module.functions.items():
    for bname, blk in caller.blocks.items():
        for idx, instr in enumerate(blk.instrs):
            callee = str(instr.attributes.get("callee") or "")
            if callee != owner.name:
                continue
            print(caller_name, bname, idx)
            print("  actual-count/formal-count", len(instr.args), len(owner.args))
            print("  args", [(int(a.id), a.dtype, tuple(a.shape or ()), dict(a.accounting or {})) for a in instr.args])
            print("  formals", [(int(a.id), a.dtype, tuple(a.shape or ()), dict(a.accounting or {})) for a in owner.args])
            print("  attrs", dict(instr.attributes))

print("== PIVOT RETURN CHAIN ==")
for name, candidate in module.functions.items():
    if not any(token in name for token in ("___pivot_mask__specialized", "___first_occurrence__specialized")):
        continue
    if "planned_region" in name:
        continue
    print(name, "formals", [(int(a.id), tuple(a.shape or ())) for a in candidate.args])
    for bname, blk in candidate.blocks.items():
        for idx, instr in enumerate(blk.instrs):
            if instr.op in {"Call", "call", "Ret", "ret"}:
                print(" ", bname, idx, instr.op,
                      "res", None if instr.res is None else (int(instr.res.id), tuple(instr.res.shape or ())),
                      "args", [(int(a.id), tuple(a.shape or ())) for a in instr.args],
                      "attrs", dict(instr.attributes))

book = identity_book(module)
for page_name in ("call_edge", "value_shape", "proven_shape", "shape.node", "shape.linked", "shape.ssa"):
    page = book.page(page_name)
    print("== PAGE", page_name, "==")
    for row in sorted(page.rows(), key=str):
        if any(token in str(row[0]) for token in ("masked_pivot_rows", "pivot_mask", "lu_decompose", "___row__specialized")):
            if str(row[0]).endswith("lu_decompose_inplace") and 40 not in [int(x) for x in row[1:] if isinstance(x, int)]:
                continue
            print(" ", render_row(row), "latest=", page.latest(row), "history=", page.history(row))

specialization_page = book.page("callsite_return_specialization")
formal_page = book.page("formal_shape")
proven_page = book.page("proven_shape")
polymorphism_page = book.page("linked_value_abi_polymorphism")
value_shape_page = book.page("value_shape")
row_functions = [
    (name, [(int(arg.id), tuple(arg.shape or ())) for arg in function.args])
    for name, function in module.functions.items()
    if "___row__specialized_" in name and "__planned_region" not in name
]
row_regions = [
    (name, [(int(arg.id), tuple(arg.shape or ())) for arg in function.args],
     [(instruction.op,
       [(int(arg.id), tuple(arg.shape or ())) for arg in instruction.args],
       dict(instruction.attributes))
      for block in function.blocks.values() for instruction in block.instrs])
    for name, function in module.functions.items()
    if "___row__specialized_" in name and "__planned_region" in name
]
row_calls = []
for caller_name, caller_function in module.functions.items():
    for block_name, block in caller_function.blocks.items():
        for instruction in block.instrs:
            callee = str(instruction.attributes.get("callee") or "")
            if "___row__specialized_" in callee and "__planned_region" not in callee:
                row_calls.append((caller_name, block_name, callee,
                                  [(int(arg.id), tuple(arg.shape or ())) for arg in instruction.args]))
cached_first_occurrence = []
for cached_shell in _deployment._CALLSITE_SHELL_TYPE_CACHE.values():
    cached_graph = cached_shell.process_graph
    if "first_occurrence" not in str(cached_graph.G.graph.get("function_name")):
        continue
    cached_identities = cached_graph.G.graph.get("identity_table") or {}
    cached_outputs = tuple(cached_graph.G.graph.get("function_outputs") or ())
    cached_first_occurrence.append((
        cached_graph.G.graph.get("function_name"),
        cached_graph.G.graph.get("planner_tensor_descriptors"),
        cached_outputs,
        {name: tuple(cached_identities.get(str(name), ())) for name in cached_outputs},
        {int(node_id): _deployment._structured_output_descriptor(cached_graph, int(node_id))
         for name in cached_outputs for node_id in cached_identities.get(str(name), ())
         if int(node_id) in cached_graph.G},
        {int(node_id): dict(cached_graph.G.nodes[int(node_id)])
         for node_id in (1, 9, 10) if int(node_id) in cached_graph.G},
        {int(node_id): _deployment._tensor_descriptor(cached_graph, int(node_id))
         for node_id in (1, 9, 10) if int(node_id) in cached_graph.G},
    ))
Path("build/_shape_pages.txt").write_text("\n".join([
    f"REGION FORMALS {[(int(a.id), tuple(a.shape or ())) for a in fn.args]!r}",
    f"OWNER FORMALS {[(int(a.id), tuple(a.shape or ())) for a in owner.args]!r}",
    f"FORMAL QUERIES {_formal_queries!r}",
    f"FORMAL EVENTS {_formal_events!r}",
    f"PIVOT GRAPH {_pivot_graph_snapshot!r}",
    f"CACHED FIRST OCCURRENCE {cached_first_occurrence!r}",
    f"ROW FUNCTIONS {row_functions!r}",
    f"ROW REGIONS {row_regions!r}",
    f"ROW CALLS {row_calls!r}",
    f"POLYMORPHISM {[(row, polymorphism_page.history(row)) for row in polymorphism_page.rows()]!r}",
    f"ROW VALUE SHAPES {[(row, value_shape_page.history(row)) for row in value_shape_page.rows() if '_row' in str(row)]!r}",
    "CALLSITE RETURN SPECIALIZATION",
    *(f"{render_row(row)} latest={specialization_page.latest(row)!r} history={specialization_page.history(row)!r}"
      for row in sorted(specialization_page.rows(), key=str)
      if any(token in str(row) for token in ("first_occurrence", "pivot_mask", "lu_decompose", "masked_pivot_rows", "_row"))),
    "FORMAL SHAPE",
    *(f"{render_row(row)} latest={formal_page.latest(row)!r} history={formal_page.history(row)!r}"
      for row in sorted(formal_page.rows(), key=str)
      if any(token in str(row) for token in ("first_occurrence", "pivot_mask", "lu_decompose", "masked_pivot_rows"))),
    "PROVEN SHAPE",
    *(f"{render_row(row)} latest={proven_page.latest(row)!r} history={proven_page.history(row)!r}"
      for row in sorted(proven_page.rows(), key=str)
      if any(token in str(row) for token in ("first_occurrence", "pivot_mask", "lu_decompose", "masked_pivot_rows", "_row"))),
]), encoding="utf-8")
