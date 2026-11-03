"""A linked scalar result does not demand a source sequence alias chase."""

import json
from pathlib import Path

import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.project_compilation_product import (
    _load_pickle_artifact,
    compile_partitioned_source,
    compile_resolved_process_graph_unit,
)


@pytest.mark.parametrize(
    "callee_body, root_expression, has_sequence",
    (
        ("return x + 1", "scalar(x)", False),
        ("row = [x, x + 1]\n    return row", "scalar(x)[0]", True),
    ),
    ids=("scalar-no-sequence", "linked-sequence-result"),
)
def test_linked_result_from_cut_callee_keeps_sequence_identity(
    tmp_path, callee_body, root_expression, has_sequence,
):
    source = (
        "def scalar(x):\n    " + callee_body + "\n\n"
        f"def root(x):\n    return {root_expression}\n"
    )
    contract = ExtractionContract(
        "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            {"function": name, "parameter": "x", "storage": "scalar",
             "dtype": "int64", "rank": 0, "shape": [],
             "python_type": "builtins.int"}
            for name in ("scalar", "root")
        ],
        "records": {}, "bindings": [],
    })
    planned = compile_partitioned_source(
        source, tmp_path / "split", extraction_contract=contract,
        ssa_composition=True, plan_only=True, max_unit_source_nodes=512,
    )
    unit_plan = planned["process_graph_unit_plan"]
    units = json.loads(Path(unit_plan).read_text(encoding="utf-8"))["units"]
    root_index = next(
        index for index, unit in enumerate(units)
        if unit["qualified_names"] == ["root"]
    )
    callee_index = next(
        index for index, unit in enumerate(units)
        if unit["qualified_names"] == ["scalar"]
    )
    assert callee_index in units[root_index]["dependency_units"]

    leaf_root = tmp_path / "leaf"
    leaf = compile_resolved_process_graph_unit(
        planned["resolved_process_graph"], unit_plan,
        callee_index, leaf_root, progress=lambda _message: None,
    )
    assert leaf["repository_ssa_complete"] is True
    leaf_module, _leaf_outputs, leaf_exports = _load_pickle_artifact(
        leaf_root / "repository-ssa.pkl"
    )
    assert bool(leaf_module.sequence_tables[leaf_exports[0]].sequences) is has_sequence
    caller = compile_resolved_process_graph_unit(
        planned["resolved_process_graph"], unit_plan,
        root_index, tmp_path / "caller",
        linked_units={callee_index: leaf_root},
        progress=lambda _message: None,
    )
    assert caller["repository_ssa_complete"] is True

    root_cut = _load_pickle_artifact(
        tmp_path / "split/inputs" / f"unit_{root_index:03d}.pkl"
    )
    callee_entry = root_cut.function_table.entry("scalar")
    assert callee_entry.graph is None
