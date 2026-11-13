"""Imported typing aliases retain the same sequence-element class receipts."""
from pathlib import Path

import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable


@pytest.mark.parametrize("declaration, annotation", [
    ("", "list[Node]"),
    ("from typing import List", "List[Node]"),
])
def test_sequence_annotation_alias_preserves_record_fields(declaration, annotation):
    source = f'''from dataclasses import dataclass
{declaration}
@dataclass
class Node:
    key: str

def keys(nodes: {annotation}):
    return {{node.key: node.key for node in nodes if node.key}}
'''
    module, _, _ = lower_ast_source_to_ssa(
        source, "keys",
        extraction_contract=ExtractionContract(
            Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml"
        ),
        progress=lambda _message: None,
    )
    assert CorrelationTable.build(module).findings(module) == []
