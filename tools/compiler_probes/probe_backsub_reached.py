"""Ask the running program which blocks it reaches.

Every link of the publication chain reads correctly in the emitted module and
the output buffer is still untouched, so the question is no longer what the
module says but what it does.  `turing_validation_error` keeps the FIRST
non-zero code it is given and the compiler links its runtime whenever the IR
mentions it, so one probe call per build answers "was this block entered?"
without a text sink or a debugger.
"""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
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
    SOURCE, "solve_two", name="reached",
    extraction_contract=contract,
    tensor_ssa_reference=c_backend_repository_ssa_reference(),
    progress=lambda _message: None,
)
qualified = "reached__solve_two"
function = module.functions[qualified]
artifact = emit_ssa_function_to_llvm(
    module, qualified, entry_name="reached_solve_two",
)
assert not artifact.shortfalls, artifact.shortfalls

DECLARE = "declare void @turing_validation_error(i32)"
STORE_SENTINEL = "  store double 7.0, ptr %out.0, align 8" + chr(10)


def function_span(text: str, symbol: str) -> tuple[int, int]:
    start = text.index(f"define internal void @{symbol}(")
    return start, text.index("\n}", start)


def probe_after(text: str, symbol: str, label: str, code: int) -> str:
    """Insert one probe call as the first instruction of a block."""

    start, end = function_span(text, symbol)
    body = text[start:end]
    marker = f"\n{label}:\n"
    at = body.index(marker) + len(marker)
    # Phi nodes must stay grouped at the top of their block.
    while body[at:].lstrip().startswith("%") and " = phi " in body[
        at:body.index("\n", at) + 1
    ]:
        at = body.index("\n", at) + 1
    body = body[:at] + f"  call void @turing_validation_error(i32 {code})\n" + body[at:]
    text = text[:start] + body + text[end:]
    if DECLARE not in text:
        # `source_filename` has to stay at the head of the module, so the
        # declaration goes at the end.
        text = text.rstrip() + "\n" + DECLARE + "\n"
    return text


expected = np.linalg.solve(MATRIX, RHS)
parameters = dict(function.metadata["parameter_names"])
published = [int(value.id) for value in outputs[qualified]]

def sentinel_before_memcpy(text: str, symbol: str) -> str:
    """Write a sentinel into %out.0 just before the publishing memcpy.

    If the result comes back as the sentinel the copy never ran; if it comes
    back NaN the copy ran and its SOURCE was NaN.
    """

    start, end = function_span(text, symbol)
    body = text[start:end]
    at = body.index("  call void @llvm.memcpy.p0.p0.i64(ptr %out.0")
    body = body[:at] + STORE_SENTINEL + body[at:]
    return text[:start] + body + text[end:]


SITES = (
    ("solve entered", "__ssa_reached__solve__specialized_", "entry", 43),
    ("back_substitute entered", "__ssa_reached___back_substitute", "entry", 41),
    ("back_substitute loop_exit", "__ssa_reached___back_substitute", "loop_exit", 42),
    ("back_substitute loop_body", "__ssa_reached___back_substitute", "loop_body", 44),
)

symbols = [
    line.split("@")[1].split("(")[0]
    for line in artifact.llvm_ir.splitlines()
    if line.startswith("define internal void @")
]

for label, prefix, block, code in SITES:
    symbol = next(
        (s for s in symbols if s.startswith(prefix) and "planned_region" not in s),
        None,
    )
    if symbol is None:
        print(f"{label:28s} NO SUCH FUNCTION ({prefix})", flush=True)
        continue
    try:
        probed = probe_after(artifact.llvm_ir, symbol, block, code)
    except ValueError:
        print(f"{label:28s} NO SUCH BLOCK ({block})", flush=True)
        continue
    native = compile_artifact(
        replace(artifact, llvm_ir=probed),
        directory=root / "build" / f"reached_{code}",
        optimization="O0",
    )
    feeds = {parameters["matrix"]: MATRIX.copy(), parameters["rhs"]: RHS.copy()}
    for value_id in published:
        feeds[value_id] = np.full(np.shape(expected), np.nan)
    execution = prepare_artifact_execution(native, feeds)
    try:
        execution.run()
        observed = 0
    except RuntimeError as error:
        observed = int(str(error).rsplit(" ", 1)[-1])
    result = np.asarray(execution.buffers[published[0]]).reshape(-1)
    import ctypes as _ct
    _order = [int(v) for v in native.buffer_order]
    _slot = _order.index(published[0])
    _typed = _ct.cast(execution.pointers, _ct.POINTER(_ct.c_void_p))
    _entry = int(_typed[_slot] or 0)
    _array = int(execution.buffers[published[0]].ctypes.data)
    print(f"    slot={_slot} table={hex(_entry)} array={hex(_array)} "
          f"same={_entry == _array}", flush=True)
    print(
        f"{label:28s} code={observed} reached={observed == code} "
        f"out={result}",
        flush=True,
    )


# The sentinel run: did the copy happen at all?
_symbol = next(
    s for s in symbols
    if s.startswith("__ssa_reached___back_substitute")
    and "planned_region" not in s
)
_native = compile_artifact(
    replace(artifact, llvm_ir=sentinel_before_memcpy(artifact.llvm_ir, _symbol)),
    directory=root / "build" / "reached_sentinel",
    optimization="O0",
)
_feeds = {parameters["matrix"]: MATRIX.copy(), parameters["rhs"]: RHS.copy()}
for _value_id in published:
    _feeds[_value_id] = np.full(np.shape(expected), np.nan)
_execution = prepare_artifact_execution(_native, _feeds).run()
_out = np.asarray(_execution.buffers[published[0]]).reshape(-1)
print("sentinel run out =", _out,
      "-> copy ran" if not np.all(_out == 7.0) else "-> copy did NOT run",
      flush=True)


# Which pointer carries the NaN: the clone result, or the latch value?
def sentinel_into(text, symbol, after_substring, pointer_name, value):
    start, end = function_span(text, symbol)
    body = text[start:end]
    at = body.index(after_substring)
    at = body.index(chr(10), at) + 1
    store = "  store double " + value + ", ptr " + pointer_name + ", align 8" + chr(10)
    body = body[:at] + store + body[at:]
    return text[:start] + body + text[end:]


for _label, _after, _ptr in (
    ("clone result %value.12", "planned_region_0(ptr %arg.1, ptr %value.12)", "%value.12"),
    ("y (arg.1) at entry", "entry:", "%arg.1"),
):
    try:
        _ir = sentinel_into(artifact.llvm_ir, _symbol, _after, _ptr, "7.0")
    except ValueError:
        print(_label, "NOT FOUND", flush=True)
        continue
    _n = compile_artifact(
        replace(artifact, llvm_ir=_ir),
        directory=root / "build" / ("sent_" + _ptr.strip("%").replace(".", "_")),
        optimization="O0",
    )
    _f = {parameters["matrix"]: MATRIX.copy(), parameters["rhs"]: RHS.copy()}
    for _v in published:
        _f[_v] = np.full(np.shape(expected), np.nan)
    _e = prepare_artifact_execution(_n, _f).run()
    print("sentinel into", _label, "-> out =",
          np.asarray(_e.buffers[published[0]]).reshape(-1), flush=True)
