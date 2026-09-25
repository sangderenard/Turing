from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.ssa_llvm_backend import (
    compile_artifact,
    emit_ssa_function_to_llvm,
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

module, _outputs, exports = lower_ast_source_to_ssa(
    source_path.read_text(encoding="utf-8"),
    "ComplexElectricalEngine.step",
    name="complex_electrical_engine",
    extraction_contract=contract,
    runtime_closure_only=True,
    tensor_ssa_reference=c_backend_repository_ssa_reference(),
    progress=lambda message: print(f"[lower] {message}", flush=True),
)
print(f"exports={exports!r}", flush=True)
book = module.metadata["identity_book"]
print("identity_pages=" + repr({
    name: len(page.rows()) for name, page in sorted(book.pages.items())
}), flush=True)
for page_name, page in sorted(book.pages.items()):
    selected = []
    for row in page.rows():
        if (
            (isinstance(row, tuple) and any(
                "kcl_residual" in str(item) for item in row
            ) and any(item == 75 for item in row))
            or row == 75
        ):
            selected.append((row, page.history(row)))
    if selected:
        print(f"identity_75={page_name}:{selected!r}", flush=True)
for function_name, function in module.functions.items():
    for block_name, block in function.blocks.items():
        for instruction in block.instrs:
            if instruction.op not in {"Call", "call"}:
                continue
            callee = instruction.attributes.get("callee")
            if callee is not None and str(callee) in module.functions:
                continue
            print("unlinked_call=" + repr((
                function_name, block_name, callee,
                [(value.id, value.dtype, value.shape)
                 for value in instruction.args],
                instruction.attributes,
            )), flush=True)
entry = next(
    symbol for symbol in exports
    if symbol.endswith("__step") and "planned_region" not in symbol
)
emitted = emit_ssa_function_to_llvm(
    module, entry, entry_name="complex_electrical_engine_step"
)
print(f"llvm_shortfalls={emitted.shortfalls!r}", flush=True)

for function_name, function in module.functions.items():
    if any(fragment in function_name for fragment in (
        "src__common__tensors__linalg__solve",
        "forward_substitute",
        "back_substitute",
    )) and "planned_region" not in function_name:
        print("TENSOR_FUNCTION=" + repr((
            function_name,
            [(value.id, value.dtype, value.shape, value.accounting)
             for value in function.args[:4]],
        )), flush=True)
