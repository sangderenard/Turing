"""Container element dependencies are not runtime union arms."""
import ast
from pathlib import Path
import pytest
from src.compiler.fortran_c_shell import (
    lower_ast_source_to_ssa,
)
from src.compiler.extraction_contract import ExtractionContract

@pytest.mark.parametrize("annotation, expected", [
    ("list[Node]", {"list"}),
    ("Optional[list[Node]]", {"list"}),
    ("Node | None", {"Node"}),
    ("Union[Node, Other]", {"Node", "Other"}),
    ("dict[Node, Other]", {"dict"}),
    ('"Node | None"', {"Node"}),
])
def test_runtime_names_exclude_element_dependencies(annotation, expected):
    from src.compiler.fortran_c_shell import _authored_annotation_runtime_class_names
    assert _authored_annotation_runtime_class_names(ast.parse(annotation, mode="eval").body) == expected


def test_nested_record_parameter_is_not_union_with_its_element():
    source = """from dataclasses import dataclass
@dataclass
class Inner:
    x: float
    y: int
@dataclass
class Node:
    key: str
    spare: Inner
    another: int
def keys(nodes: list[Node]):
    keys = {node.key for node in nodes}
    return len(keys)
"""
    contract = ExtractionContract(Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml")
    module, _, _ = lower_ast_source_to_ssa(
        source, "keys", extraction_contract=contract,
        progress=lambda _: None, progress_bars=False)
    function = module.functions["keys__keys"]
    parameter_id = dict(function.metadata["parameter_names"])["nodes"]
    fact = function.metadata["native_value_type_abi"].get(parameter_id)
    assert fact is None or fact.alternatives == ("builtins.list",)
    assert not any(v.accounting.get("native_type_tag_parameter") == "nodes" for v in function.args)
