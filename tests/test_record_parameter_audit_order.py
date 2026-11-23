"""A record parameter of a helper survives the final concordance.

The final audit (``_source_parameter_memory_contract_findings``) reads the
function table's graphs: the parameter's Input and its installed contract.
The native-only frontend detach used to run first and release those graphs,
so every program passing a dataclass to a helper was refused with
"parameter 'd' lost selected contract ... formal binding".  The detach now
runs once the audit has accepted the module.
"""

from pathlib import Path
import textwrap

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

ROOT = Path(__file__).resolve().parents[1]


def test_dataclass_helper_parameter_compiles_and_frontend_detaches_after():
    source = textwrap.dedent('''
        from dataclasses import dataclass
        @dataclass
        class V:
            x: float
            y: float
        def weigh(d: V, s: float) -> float:
            return (d.x + 2.0 * d.y) * s
        def main(a: float, b: float) -> float:
            return weigh(V(a, b), 2.0)
    ''')
    contract = ExtractionContract(
        ROOT / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({"records": {}, "bindings": [], "values": [
        {"function": "main", "parameter": name, "storage": "scalar",
         "dtype": "float64", "rank": 0, "shape": [],
         "python_type": "builtins.float"}
        for name in ("a", "b")
    ]}).with_execution_file(
        ROOT / "extraction_contracts/vehicle_full_native_execution.yaml"
    )
    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "main", extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        runtime_closure_only=True, progress=lambda _: None,
        progress_bars=False,
    )
    assert exports == ("main__main",)
    # Accepted, then released by the established finalization transaction.
    assert module.metadata["frontend_representations_detached"] is True
    assert all(entry.graph is None for entry in module.function_table)
    assert "frontend_detach_pending" not in module.metadata
