from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.compiler.extraction_contract import ExtractionContract
from src.compiler import fortran_c_shell as shell
from src.compiler.identity_concordance import current_identity_book

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

original = shell._publish_concordant_function_aliases

def traced(function, bindings):
    try:
        return original(function, bindings)
    except ValueError:
        name = str(function.name)
        page = current_identity_book().page("planning_value_concordance")
        metadata = dict(function.metadata or {})
        print("CONCORDANCE_FAILURE", name, flush=True)
        print("INCOMING", dict(bindings), flush=True)
        print("VALUE_ALIASES", metadata.get("value_aliases"), flush=True)
        print("OUTPUT_ALIASES", metadata.get("output_identity_aliases"), flush=True)
        print("PAGE_BINDINGS", page.alias_bindings(name), flush=True)
        values = {}
        for argument in function.args:
            values[int(argument.id)] = ("formal", argument.dtype, argument.shape, argument.accounting)
        for block_name, block in function.blocks.items():
            for index, instruction in enumerate(block.instrs):
                if instruction.res is not None:
                    values[int(instruction.res.id)] = (
                        f"{block_name}#{index}:{instruction.op}",
                        instruction.res.dtype,
                        instruction.res.shape,
                        instruction.res.accounting,
                    )
        for value_id in (2, 50, 60):
            print(
                "VALUE", value_id,
                "history", page.history((name, value_id)),
                "ownership", values.get(value_id),
                flush=True,
            )
        raise

shell._publish_concordant_function_aliases = traced
try:
    shell.lower_ast_source_to_ssa(
        source_path.read_text(encoding="utf-8"),
        "ComplexElectricalEngine.step",
        name="complex_electrical_engine",
        extraction_contract=contract,
        runtime_closure_only=True,
    )
finally:
    shell._publish_concordant_function_aliases = original
