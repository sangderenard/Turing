"""Nullable collections retain their tag and existing borrowed row ABI."""
from pathlib import Path
import numpy as np
import pytest
from src.compiler.authored_parameter_abi import _authored_sequence_annotation_contract
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable

@pytest.mark.parametrize("annotation", ["Optional[list[int]]", "Union[None, list[int]]", "list[int] | None", "'Optional[list[int]]'", "Union[list[int], None]", "typing.Union[None, list[int]]", "'typing.Optional[list[int]]'", "Optional[Union[None, list[int]]]"])
@pytest.mark.parametrize("mutate", [False, True])
def test_nullable_list_iteration_keeps_tag_and_borrowed_payload(tmp_path, annotation, mutate):
    from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
    source = ("import typing\nfrom typing import Optional, Union\n"
              f"def root(items: {annotation}) -> int:\n"
              "    if items is None:\n        return -7\n"
              + ("    if len(items) > 0:\n        items[0] += 1\n" if mutate else "")
              + "    total = 0\n    for value in items:\n        total += value\n    return total\n")
    contract = ExtractionContract(Path(__file__).resolve().parents[1]/"extraction_contracts/program_extraction.yaml")
    module, _, exports = lower_ast_source_to_ssa(source,"root",extraction_contract=contract,progress=lambda _:None,progress_bars=False)
    assert CorrelationTable.build(module).findings(module) == []
    fn=module.functions[exports[0]]
    identity=dict(fn.metadata["parameter_names"])["items"]
    seq=module.sequence_tables[exports[0]].sequences[identity]
    tag=next(a for a in fn.args if a.accounting.get("native_type_tag_parameter")=="items")
    tags=dict(tag.accounting["native_type_tags"])
    assert set(tags)=={"builtins.NoneType","builtins.list"}
    emitted=emit_ssa_function_to_llvm(module,exports[0]);assert emitted.shortfalls==()
    artifact=compile_artifact(emitted,directory=tmp_path/"llvm")
    output=next(i.args[0].id for b in fn.blocks.values() for i in b.instrs if i.op=="Ret" and not i.attributes.get("fault_exit"))
    for absent,data in [(True,[91]),(False,[]),(False,[4,-2,7])]:
        for _ in range(2):
            array=np.array(data,dtype=np.int64)
            feeds={seq.column_value_ids[0]:array,seq.length_address_id:np.array([len(data)],dtype=np.int64),
                   seq.capacity_value_id:np.array([len(data)],dtype=np.int64),seq.status_address_id:np.array([0],dtype=np.int64),
                   tag.id:np.array([tags["builtins.NoneType" if absent else "builtins.list"]],dtype=np.int64)}
            expected_data=list(data)
            if mutate and not absent and expected_data:expected_data[0]+=1
            result=prepare_artifact_execution(artifact,feeds).run()
            assert int(result.buffers[output].reshape(-1)[0])==(-7 if absent else sum(expected_data))
            np.testing.assert_array_equal(array,np.array(expected_data,dtype=np.int64))

@pytest.mark.parametrize("annotation", ["Union[list[int], list[float], None]", "list[int] | set[int] | None", "Optional[Union[list[int], int]]"])
def test_nullable_contract_does_not_collapse_distinct_material_arms(annotation):
    assert _authored_sequence_annotation_contract(annotation) is None


def test_union_physical_receipt_preserves_distinct_material_arms():
    import ast
    from src.compiler.fortran_c_shell import (
        _authored_annotation_field_receipt, _authored_annotation_python_types,
    )
    annotation = ast.parse("Union[list[int], list[float], None]", mode="eval").body
    assert _authored_annotation_field_receipt(annotation) is None
    # Runtime classes may agree; their physical element layouts do not.
    assert _authored_annotation_python_types(annotation) == (
        "builtins.NoneType", "builtins.list",
    )
