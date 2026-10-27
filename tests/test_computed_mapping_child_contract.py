"""Computed child payloads follow their exact source values across methods."""
from pathlib import Path

import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import concordance_report

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("rebind,dtype", [(False, "int64"), (True, "float64")])
def test_computed_payload_and_named_key_use_their_source_contracts(rebind, dtype):
    source = (ROOT / "tests/record_view_corpus/c640_computed_mapping_child.py").read_text()
    if rebind:
        source = source.replace("        self.domain.append(domain)",
                                "        token = 1.5\n        self.domain.append(domain)")
    graphs = []
    module, _, _ = lower_ast_source_to_ssa(
        source, "root", name="computed_mapping_child",
        extraction_contract=ROOT / "extraction_contracts/program_extraction.yaml",
        resolved_process_graph_sink=graphs.append,
        progress=lambda _: None, progress_bars=False,
    )
    evidence = module.metadata["mapping_child_contract_evidence"]
    assert len(evidence) == 2
    assert {item["key_width"] for item in evidence} == {2, 4}
    assert {item["dtype"] for item in evidence} == {dtype}
    assert all(item["class_cell"].row[1] == evidence[0]["class_cell"].row[1]
               for item in evidence)
    for entry in graphs[0].function_table:
        if entry.name not in {"__init__", "put", "read"}:
            continue
        contracts = entry.graph.G.graph["class_field_mapping_contracts"]
        assert contracts["_candidates"]["mapping_key_column_count"] == 4
        assert contracts["_prefix"]["mapping_key_column_count"] == 2
        assert contracts["_candidates"]["mapping_child_column_dtypes"] == (dtype,)
        assert contracts["_prefix"]["mapping_child_column_dtypes"] == (dtype,)
    pools = [descriptor.child_table_pool
             for table in module.sequence_tables.values()
             for descriptor in table.sequences.values()
             if descriptor.child_table_pool is not None]
    assert pools
    assert all(pool.column_dtypes == (dtype,) for pool in pools)
    report = concordance_report(module)
    assert "0 finding(s)" in report, report
    # A copied contract is a view of the referenced row, not an independent
    # claim: poison each derived field and require the audit to notice.
    evidence[0]["dtype"] = "uint64"
    assert "[mapping-child-contract-disagreement]" in concordance_report(module)
    evidence[0]["dtype"] = dtype
    evidence[0]["key_width"] += 1
    assert "[mapping-child-contract-disagreement]" in concordance_report(module)
