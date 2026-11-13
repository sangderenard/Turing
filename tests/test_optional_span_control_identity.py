"""Control branches consume the declared optional span tag, not its payload."""
from pathlib import Path
import pytest
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable

@pytest.mark.parametrize("operator", ["is", "is not"])
def test_optional_list_control_uses_declared_tag(operator):
    source = ("from typing import Optional\n"
              "def root(items: Optional[list[str]]) -> int:\n"
              f"    if items {operator} None:\n"
              "        return 7\n"
              "    return 3\n")
    contract = ExtractionContract(Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml")
    module, _, exports = lower_ast_source_to_ssa(source, "root", extraction_contract=contract,
                                                progress=lambda _: None, progress_bars=False)
    assert CorrelationTable.build(module).findings(module) == []
    function = module.functions[exports[0]]
    tags = [value for value in function.args if value.accounting.get("native_type_tag_parameter") == "items"]
    assert len(tags) == 1
    tag = tags[0]
    assert "native_type_tag_pending_for_value_id" not in tag.accounting
    tests = [i for b in function.blocks.values() for i in b.instrs
             if i.attributes.get("source_native_type_test")]
    assert tests and all(i.attributes["source_native_type_test"]["tag_value_id"] == tag.id for i in tests)
    assert any(i.op == "Eq" and tag.id in [a.id for a in i.args]
               for b in function.blocks.values() for i in b.instrs)
