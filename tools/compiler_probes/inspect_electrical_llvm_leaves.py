from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)


root = Path(__file__).resolve().parents[3]
turing = root / "turing"
source_path = root / "spectral-analyzer" / "electrical_dt_engine.py"
sys.path.insert(0, str(source_path.parent))
contract = ExtractionContract(
    turing / "extraction_contracts" / "program_extraction.yaml"
).with_sources([
    ("electrical_dt_engine", source_path),
    ("electrical_tensor_network", root / "spectral-analyzer" / "electrical_tensor_network.py"),
    ("engine_toy.dc_power", turing / "engine_toy" / "dc_power.py"),
])

module, _outputs, _exports = lower_ast_source_to_ssa(
    source_path.read_text(encoding="utf-8"),
    "ComplexElectricalEngine.step",
    name="complex_electrical_engine",
    extraction_contract=contract,
    runtime_closure_only=True,
    tensor_ssa_reference=c_backend_repository_ssa_reference(),
    progress=lambda message: print(f"[lower] {message}", flush=True),
)

leaf_operations = {
    "max", "real", "sum", "Load", "Const", "GetElementPtr",
    "repeat", "solve", "getattr", "full_like",
}
for function_name, function in module.functions.items():
    if function_name != "complex_electrical_engine___kcl_residual__planned_region_4":
        continue
    table = module.tensor_tables.get(function_name)
    for block_name, block in function.blocks.items():
        for index, instruction in enumerate(block.instrs):
            operation = str(
                instruction.attributes.get("tensor_operation")
                or instruction.attributes.get("tensor")
                or instruction.op
            )
            if operation not in leaf_operations and instruction.op not in {"Indexed", "indexed"}:
                continue
            values = (*instruction.args,) + (
                (() if instruction.res is None else (instruction.res,))
            )
            print("LEAF", repr({
                "function": function_name,
                "block": block_name,
                "index": index,
                "op": instruction.op,
                "operation": operation,
                "args": tuple((
                    int(value.id), str(value.dtype), tuple(value.shape or ()),
                    dict(value.accounting or {}),
                ) for value in instruction.args),
                "result": None if instruction.res is None else (
                    int(instruction.res.id), str(instruction.res.dtype),
                    tuple(instruction.res.shape or ()),
                    dict(instruction.res.accounting or {}),
                ),
                "attributes": dict(instruction.attributes),
                "tensor_descriptors": tuple(
                    None if table is None or table.by_id(int(value.id)) is None
                    else table.by_id(int(value.id))
                    for value in values
                ),
            }), flush=True)

print("LATE_SHORTFALLS", repr(
    module.metadata.get("late_tensor_lowering_shortfalls", ())
), flush=True)
