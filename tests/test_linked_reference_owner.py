"""Reference witnesses of accepted regions stay in their owning book."""

import hashlib
import json
from pathlib import Path

import pytest

from src.compiler.concordance_declarations import CROSS_FUNCTION_REFERENCES
from src.compiler.identity_concordance import (
    ConcordanceRefusal, begin_identity_book, end_identity_book,
)
from src.compiler.tensor_ssa_lowering import propagate_repository_ssa_call_metadata
from src.compiler.ssa_call_input_adapters import adapt_physical_call_inputs
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.project_compilation_product import (
    _load_pickle_artifact, compile_partitioned_source,
    compile_resolved_process_graph_unit,
)
from src.transmogrifier.ssa import BasicBlock, Function, Instr, IRModule, SSAValue
from tests.compiler_provenance_inputs import publish_authored_ssa


@pytest.mark.parametrize("valid_digest", (True, False))
def test_accepted_region_reference_witnesses_are_imported_from_the_owner(valid_digest):
    value = SSAValue(5, "int64")
    functions = {
        name: Function(name, [value], {
            "entry": BasicBlock("entry", [Instr("Ret", [value], None)]),
        }, metadata={
            "tensor_shape_concordance_scope": "accepted@control:0",
            "source_region_integral": {"owner": "accepted"},
        })
        for name in ("region_a", "region_b")
    }
    owner, token = begin_identity_book()
    try:
        publish_authored_ssa(functions)
    finally:
        end_identity_book(token)
    digest = hashlib.sha256(b"accepted-reference-fixture").hexdigest()
    for function in functions.values():
        function.metadata["accepted_ssa_component"] = {
            "schema": "turing.accepted-ssa-component.v1",
            "identity_book": owner,
            "artifact_sha256": digest if valid_digest else None,
        }

    destination, token = begin_identity_book()
    try:
        module = IRModule(functions, metadata={"identity_book": destination})
        if not valid_digest:
            with pytest.raises(ConcordanceRefusal, match="verified artifact digest"):
                propagate_repository_ssa_call_metadata(module)
            return
        assert not propagate_repository_ssa_call_metadata(module)
        row = ("accepted", 5)
        assert destination.page(CROSS_FUNCTION_REFERENCES).latest(row) == (
            "region_a", "region_b",
        )
        imported = destination.page("concordance_imported_cell")
        assert imported.rows()
        for row in imported.rows():
            assert row[0] == digest
            receipt = imported.latest(row)
            assert receipt.owner is owner
            owner._source_stamp(receipt.source)
    finally:
        end_identity_book(token)


def test_accepted_region_formal_is_immutable_under_caller_enrichment():
    accepted_formal = SSAValue(5, "int64", (2,))
    accepted = Function("accepted__planned_region_4", [accepted_formal], {
        "entry": BasicBlock("entry", [Instr("Ret", [accepted_formal], None)]),
    }, metadata={"source_region_integral": {"owner": "accepted"}})
    owner, token = begin_identity_book()
    try:
        publish_authored_ssa({accepted.name: accepted})
    finally:
        end_identity_book(token)
    accepted.metadata["accepted_ssa_component"] = {
        "schema": "turing.accepted-ssa-component.v1",
        "identity_book": owner,
        "artifact_sha256": hashlib.sha256(b"accepted-immutable").hexdigest(),
    }
    actual = SSAValue(20, "int64", (3,))
    result = SSAValue(21)
    caller = Function("caller", [actual], {
        "entry": BasicBlock("entry", [
            Instr("Call", [actual], result, attributes={"callee": accepted.name}),
            Instr("Ret", [result], None),
        ]),
    })
    ambient, token = begin_identity_book()
    try:
        publish_authored_ssa({caller.name: caller})
        module = IRModule({caller.name: caller, accepted.name: accepted},
                          metadata={"identity_book": ambient})
        propagate_repository_ssa_call_metadata(module)
        assert accepted_formal.shape == (2,)
        assert accepted_formal.dtype == "int64"
        assert all(
            row[0] != accepted.name
            for row in ambient.page("tensor_shape_enrichment").rows()
        )
        assert owner.page("ssa_value").latest(
            ("accepted__planned_region_4", 5)
        ).shape == (2,)
    finally:
        end_identity_book(token)


def test_caller_enrichment_imports_accepted_return_value_proof():
    returned = SSAValue(5, "int64", (3,), accounting={
        "physical_dtype": "int64", "program_abi_storage": "span",
        "program_abi_rank": 1,
    })
    accepted = Function("accepted_leaf", [], {
        "entry": BasicBlock("entry", [
            Instr("Const", [], returned, attributes={"values": (1, 2, 3)}),
            Instr("Ret", [returned], None),
        ]),
    })
    owner, token = begin_identity_book()
    try:
        publish_authored_ssa({accepted.name: accepted})
    finally:
        end_identity_book(token)
    digest = hashlib.sha256(b"accepted-return").hexdigest()
    accepted.metadata["accepted_ssa_component"] = {
        "schema": "turing.accepted-ssa-component.v1",
        "identity_book": owner, "artifact_sha256": digest,
    }
    result = SSAValue(9)
    caller = Function("caller", [], {
        "entry": BasicBlock("entry", [
            Instr("Call", [], result, attributes={"callee": accepted.name}),
            Instr("Ret", [result], None),
        ]),
    })
    destination, token = begin_identity_book()
    try:
        publish_authored_ssa({caller.name: caller})
    finally:
        end_identity_book(token)
    module = IRModule({caller.name: caller, accepted.name: accepted},
                      metadata={"identity_book": destination})
    ambient, token = begin_identity_book()
    try:
        assert propagate_repository_ssa_call_metadata(module)
        from src.compiler.identity_concordance import current_identity_book
        assert current_identity_book() is ambient
        assert result.dtype == "int64"
        assert result.shape == (3,)
        assert returned.shape == (3,)
        assert owner.page("ssa_value").latest((accepted.name, 5)).shape == (3,)
        assert any(row[0] == digest for row in destination.page(
            "concordance_imported_cell"
        ).rows())
        assert not any(row[0] == digest for row in ambient.page(
            "concordance_imported_cell"
        ).rows())
    finally:
        end_identity_book(token)


def test_physical_call_formal_reads_accepted_owner_and_imports_proof():
    formal = SSAValue(5, "int64", (), accounting={
        "physical_dtype": "int64", "program_abi_storage": "scalar",
        "program_abi_rank": 0, "program_abi_parameter": "x",
    })
    accepted = Function("accepted_leaf", [formal], {
        "entry": BasicBlock("entry", [Instr("Ret", [formal], None)]),
    })
    owner, token = begin_identity_book()
    try:
        publish_authored_ssa({accepted.name: accepted})
    finally:
        end_identity_book(token)
    digest = hashlib.sha256(b"accepted-physical-formal").hexdigest()
    accepted.metadata["accepted_ssa_component"] = {
        "schema": "turing.accepted-ssa-component.v1",
        "identity_book": owner, "artifact_sha256": digest,
    }
    actual = SSAValue(10, "int64", (), accounting={
        "physical_dtype": "int64", "program_abi_storage": "scalar",
        "program_abi_rank": 0, "program_abi_parameter": "x",
    })
    result = SSAValue(11)
    caller = Function("caller", [actual], {
        "entry": BasicBlock("entry", [
            Instr("Call", [actual], result, attributes={"callee": accepted.name}),
            Instr("Ret", [result], None),
        ]),
    })
    destination, token = begin_identity_book()
    try:
        publish_authored_ssa({caller.name: caller})
        assert adapt_physical_call_inputs({caller.name: caller,
                                           accepted.name: accepted}) == 0
        assert any(row[0] == digest for row in destination.page(
            "concordance_imported_cell"
        ).rows())
        assert formal.dtype == "int64"
        assert owner.page("ssa_value").latest((accepted.name, 5)).physical_dtype == "int64"
    finally:
        end_identity_book(token)


def test_fresh_linked_source_retains_accepted_callee_receipt(tmp_path):
    source = (
        "def leaf(x: int) -> int:\n    return x + 1\n\n"
        "def root(x: int) -> int:\n    return leaf(x)\n"
    )
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            {"function": name, "parameter": "x", "storage": "scalar",
             "dtype": "int64", "rank": 0, "shape": [],
             "python_type": "builtins.int"}
            for name in ("leaf", "root")
        ],
        "records": {}, "bindings": [],
    })
    planned = compile_partitioned_source(
        source, tmp_path / "plan", extraction_contract=contract,
        ssa_composition=True, plan_only=True, max_unit_source_nodes=512,
    )
    unit_plan = planned["process_graph_unit_plan"]
    units = json.loads(Path(unit_plan).read_text(encoding="utf-8"))["units"]
    leaf_index = next(index for index, unit in enumerate(units)
                      if unit["qualified_names"] == ["leaf"])
    root_index = next(index for index, unit in enumerate(units)
                      if unit["qualified_names"] == ["root"])
    assert leaf_index in units[root_index]["dependency_units"]
    leaf_root = tmp_path / "leaf"
    leaf = compile_resolved_process_graph_unit(
        planned["resolved_process_graph"], unit_plan, leaf_index,
        leaf_root, progress=lambda _message: None,
    )
    assert leaf["repository_ssa_complete"] is True
    leaf_module, _outputs, leaf_exports = _load_pickle_artifact(
        leaf_root / "repository-ssa.pkl"
    )
    accepted_name = leaf_exports[0]
    accepted_values = tuple(
        (value.id, value.dtype, tuple(value.shape))
        for value in leaf_module.functions[accepted_name].args
    )
    caller = compile_resolved_process_graph_unit(
        planned["resolved_process_graph"], unit_plan, root_index,
        tmp_path / "caller", linked_units={leaf_index: leaf_root},
        progress=lambda _message: None,
    )
    assert caller["repository_ssa_complete"] is True
    composed, _outputs, _exports = _load_pickle_artifact(
        tmp_path / "caller/repository-ssa.pkl"
    )
    accepted = composed.functions[accepted_name]
    assert accepted.metadata.get("accepted_ssa_component")
    assert tuple((value.id, value.dtype, tuple(value.shape))
                 for value in accepted.args) == accepted_values
