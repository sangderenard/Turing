"""Lower solve to SSA and stop, for diagnosis in seconds instead of minutes.

The compile log puts ``ssa-program complete`` at about six seconds; the rest
of the four minutes is LLVM and clang.  Everything the concordance knows is
settled by then, so a question about identity or shape does not need the
native build to answer it.

Reports the shape stores that disagree and, for each, the instruction that
defines the value -- a value the graph calls ``(2,)`` and the descriptor
proves ``(1,)`` is a buffer with room for one element holding two, which
reads uninitialized memory and publishes NaN without any shortfall.
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
    authored_function_name, identity_book, shape_store_report,
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
    "records": {}, "bindings": [],
    "values": [
        {"function": "solve_two", "parameter": "matrix", "storage": "span",
         "dtype": "float64", "rank": 2, "shape": list(MATRIX.shape),
         "python_type": "AbstractTensor"},
        {"function": "solve_two", "parameter": "rhs", "storage": "span",
         "dtype": "float64", "rank": 1, "shape": list(RHS.shape),
         "python_type": "AbstractTensor"},
    ],
})

module, outputs, _exports = lower_ast_source_to_ssa(
    SOURCE, "solve_two", name="ssa_only",
    extraction_contract=contract,
    tensor_ssa_reference=c_backend_repository_ssa_reference(),
    progress=lambda _message: None,
)

book = identity_book(module)
print(shape_store_report(book), flush=True)

# Where does each disagreeing value come from?
node_page = book.page("shape.node")
proven_page = book.page("proven_shape")
disagreeing = []
for row in proven_page.rows():
    proven = proven_page.latest(row)
    if not (isinstance(proven, tuple) and proven and proven[0] == "proven"):
        continue
    node = node_page.latest(row)
    if node is None or tuple(node) == tuple(proven[1]):
        continue
    disagreeing.append((row, tuple(node), tuple(proven[1])))

# How did the losing answer win?  The page keeps every level it was told.
print("PROVEN HISTORY for the disagreeing rows:", flush=True)
_rows = {r for r, _n, _pv in disagreeing}
for row in proven_page.rows():
    if row not in _rows:
        continue
    print(f"  {row}", flush=True)
    for column, fact in sorted(proven_page.history(row)):
        print(f"    level {column}: {fact}", flush=True)
    for _c, _f in sorted(book.page("descriptor_op").history(row)):
        print(f"    descriptor answer {_c}: {_f}", flush=True)
    print(f"    binary sides: {book.page(chr(34)+chr(34)) if False else book.page('binary_sides').latest(row)}", flush=True)
    print('    binary parents:', book.page('binary_parents').latest(row), flush=True)
    node_hist = node_page.history(row)
    for column, fact in sorted(node_hist):
        print(f"    node level {column}: {fact}", flush=True)

# Everything the concordance knows about each disagreeing identity, from
# every page at once -- which is the question, not which page to ask.
print("EVERY PAGE for the disagreeing rows:", flush=True)
for _r in sorted(_rows, key=str):
    print(f"  {_r}", flush=True)
    for _pn, _pf in sorted(book.latest_by_page(_r).items()):
        print(f"    {_pn}: {_pf}"[:180], flush=True)
    _clash = book.disagreements(_r)
    if _clash:
        print(f"    DISAGREEMENTS: {_clash}"[:300], flush=True)

print(f"\nDISAGREEING: {len(disagreeing)}", flush=True)
for row, node, proven in disagreeing:
    authored, value_id = row
    print(f"\n  {authored} value {value_id}: node={node} proven={proven}",
          flush=True)
    for name, function in module.functions.items():
        if authored_function_name(name) != authored:
            continue
        for block_name, block in function.blocks.items():
            for index, instruction in enumerate(block.instrs):
                if instruction.res is None:
                    continue
                if int(instruction.res.id) != int(value_id):
                    continue
                print(f"    def {name.split('__')[-1]} {block_name}#{index} "
                      f"{instruction.op} "
                      f"args={[int(getattr(a, 'id', -1)) for a in instruction.args]}",
                      flush=True)
                print(f"      attrs={dict(instruction.attributes)}"[:500],
                      flush=True)
                print(f"      res.shape={instruction.res.shape} "
                      f"dtype={instruction.res.dtype}", flush=True)
        # Any instruction that CONSUMES it, and what shape that use expects.
        for block_name, block in function.blocks.items():
            for index, instruction in enumerate(block.instrs):
                positions = [
                    position
                    for position, argument in enumerate(instruction.args)
                    if int(getattr(argument, "id", -1)) == int(value_id)
                ]
                if not positions:
                    continue
                feeds = instruction.attributes.get("feed_shapes")
                print(f"    use {name.split('__')[-1]} {block_name}#{index} "
                      f"{instruction.op} at {positions}"
                      + (f" feed_shapes={feeds}" if feeds else ""), flush=True)

print("UPSTREAM operands that answered ():", flush=True)
for _v in (80, 47, 82, 75, 76):
    _ur = ("_lu_decompose_inplace", _v)
    print(f"  value {_v}", flush=True)
    for _pn, _pf in sorted(book.latest_by_page(_ur).items()):
        print(f"    {_pn}: {_pf}"[:200], flush=True)
    for _name, _fn in module.functions.items():
        if authored_function_name(_name) != "_lu_decompose_inplace":
            continue
        for _bn, _blk in _fn.blocks.items():
            for _i, _ins in enumerate(_blk.instrs):
                if _ins.res is not None and int(_ins.res.id) == _v:
                    print(
                        f"    def {_bn}#{_i} {_ins.op}"
                        f" args={[int(getattr(a, 'id', -1)) for a in _ins.args]}"
                        f" callee={_ins.attributes.get('callee')}"
                        f" res.shape={_ins.res.shape}", flush=True)

# What does the entry actually emit?  Emission is fast; clang is not.
from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm

_qualified = "ssa_only__solve_two"
_artifact = emit_ssa_function_to_llvm(
    module, _qualified, entry_name="ssa_only_solve_two",
)
print("ENTRY shortfalls:", _artifact.shortfalls, flush=True)
_ir = _artifact.llvm_ir
_start = _ir.index("define") if "define" in _ir else 0
for _line in _ir.splitlines():
    if _line.startswith("define") and "solve_two" in _line:
        _at = _ir.index(_line)
        _body = _ir[_at:_ir.index("\n}", _at)]
        print("=== ENTRY BODY ===", flush=True)
        print(_body[:3000], flush=True)
        break

# Does anything write a CALLER-PASSED pointer, anywhere in the module?
import re as _re
_stores_to_args = 0
_stores_total = 0
_memcpy_to_args = 0
_memcpy_total = 0
for _line in _ir.splitlines():
    _s = _line.strip()
    if _s.startswith("store "):
        _stores_total += 1
        if _re.search(r"ptr %arg\.\d+\s*(,|$)", _s):
            _stores_to_args += 1
    if "llvm.memcpy" in _s:
        _memcpy_total += 1
        if _re.search(r"\(ptr %arg\.\d+", _s):
            _memcpy_to_args += 1
print(f"IR: stores={_stores_total} to-arg={_stores_to_args} "
      f"memcpy={_memcpy_total} to-arg={_memcpy_to_args}", flush=True)
_defs = [l.split("@")[1].split("(")[0]
         for l in _ir.splitlines() if l.startswith("define")]
print("IR: functions defined =", len(_defs), flush=True)

_out_defs = _re.findall(r"%out\.\d+", _ir)
print("IR: %out. occurrences =", len(_out_defs),
      "distinct =", len(set(_out_defs)), flush=True)
_writes = [l.strip() for l in _ir.splitlines()
           if ("%out." in l and ("store " in l or "memcpy" in l))]
print("IR: writes naming %out. =", len(_writes), flush=True)
for _w in _writes[:8]:
    print("   ", _w[:150], flush=True)
_entry_params = _re.search(r"define internal void @__ssa_ssa_only__solve_two\(([^)]*)\)", _ir)
print("IR: entry param count =",
      0 if not _entry_params else len(_entry_params.group(1).split(",")),
      flush=True)
