"""Direct record sequences share the contract's recursive leaf ABI."""
from pathlib import Path
import numpy as np
import pytest
from src.compiler.extraction_contract import (ExtractionContract, ExtractionContractError,
    ProgramABIContract)
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable

SOURCE = """from dataclasses import dataclass
@dataclass
class Inner:
    x: float
    y: int
@dataclass
class Node:
    key: int
    spare: Inner
    another: int
def root(nodes: list[Node]):
    total = 0.0
    for node in nodes:
        total += node.spare.x + node.spare.y * 2 + node.another * 3
    return total
"""

@pytest.mark.parametrize("optional", [False, True])
def test_recursive_row_layout_and_native_leaf_projection(tmp_path, optional):
    from src.compiler.ssa_llvm_backend import (
        emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution)
    source = SOURCE.replace("spare: Inner", "spare: Inner | None") if optional else SOURCE
    contract = ExtractionContract(Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml")
    module, _, exports = lower_ast_source_to_ssa(source, "root", extraction_contract=contract,
                                                progress=lambda _: None, progress_bars=False)
    assert CorrelationTable.build(module).findings(module) == []
    fn = module.functions[exports[0]]
    sid = dict(fn.metadata["parameter_names"])["nodes"]
    desc = module.sequence_tables[exports[0]].sequences[sid]
    expected = ("int64", *(("bool",) if optional else ()), "float64", "int64", "int64")
    assert desc.column_dtypes == expected
    emitted = emit_ssa_function_to_llvm(module, exports[0])
    assert emitted.shortfalls == ()
    artifact = compile_artifact(emitted, directory=tmp_path / "llvm")
    for n in (0, 1, 4):
        arrays = [np.arange(n, dtype=np.int64)]
        if optional:
            arrays.append(np.ones(n, dtype=np.bool_))
        arrays.extend([np.arange(n, dtype=np.float64) + .25,
                       np.arange(n, dtype=np.int64) + 2, np.arange(n, dtype=np.int64) + 5])
        feeds = dict(zip(desc.column_value_ids, arrays))
        feeds[desc.length_address_id] = np.array([n], dtype=np.int64)
        feeds[desc.capacity_value_id] = np.array([n], dtype=np.int64)
        feeds[desc.status_address_id] = np.array([0], dtype=np.int32)
        result = prepare_artifact_execution(artifact, feeds).run()
        output = next(i.args[0].id for b in fn.blocks.values() for i in b.instrs
                      if i.op == "Ret" and not i.attributes.get("fault_exit"))
        assert float(result.buffers[output].reshape(-1)[0]) == sum((i + .25) + (i + 2) * 2 + (i + 5) * 3 for i in range(n))


def test_shared_row_normalizer_preserves_optional_order_and_detects_cycle():
    from src.compiler.extraction_contract import record_row_columns
    records = {
        "Inner": {"fields": {"x": {"storage": "scalar", "dtype": "float64"}}},
        "Outer": {"fields": {"inner": {"storage": "record", "record": "Inner", "optional": True},
                              "end": {"storage": "scalar", "dtype": "int64"}}},
        "Owner": {"fields": {"rows": {"storage": "table", "row_record": "Outer"}}},
    }
    columns = record_row_columns(records, "Outer")
    assert tuple(c["name"] for c in columns) == ("inner.__present", "inner.x", "end")
    contract = ProgramABIContract.from_mapping({"records": records, "bindings": [], "values": []})
    assert contract.records["Owner"].fields["rows"].table_columns == columns
    records["Inner"]["fields"]["cycle"] = {"storage": "record", "record": "Outer"}
    with pytest.raises(ExtractionContractError, match="row_record cycle: Outer -> Inner -> Outer"):
        record_row_columns(records, "Outer")
