from pathlib import Path

import pytest

from src.compiler import structural_concordance
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import ConcordanceRefusal
from src.transmogrifier.ssa import SSAValue


CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml"


def test_shared_source_entry_refuses_actual_final_discordance(monkeypatch):
    publish = structural_concordance.publish_module_structures

    def corrupt_final_return(module):
        publish(module)
        for function in module.functions.values():
            for block in function.blocks.values():
                for instruction in block.instrs:
                    if instruction.op == "Ret" and instruction.args:
                        instruction.args[0] = SSAValue(987654321, "int64")
                        return
        raise AssertionError("probe must corrupt an actual return")

    monkeypatch.setattr(structural_concordance, "publish_module_structures", corrupt_final_return)
    with pytest.raises(ConcordanceRefusal, match="final concordance refused") as raised:
        lower_ast_source_to_ssa(
            "def root():\n    return 7\n", "root", name="concordance_gate",
            extraction_contract=CONTRACT, progress=lambda _: None,
        )
    assert "987654321" in str(raised.value)


def test_shared_source_entry_accepts_concordant_program():
    module, _outputs, exports = lower_ast_source_to_ssa(
        "def root():\n    return 7\n", "root", name="concordance_gate_valid",
        extraction_contract=CONTRACT, progress=lambda _: None,
    )
    assert exports and module.functions
