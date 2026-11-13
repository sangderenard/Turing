"""Native string comparisons use complete glyph spans and exact borrowed storage."""
from pathlib import Path
import numpy as np
import pytest
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.ssa_storage_requirements import is_compiler_owned_storage
from src.compiler.string_table import string_token

@pytest.mark.parametrize("operator", ["==", "!="])
def test_authored_string_equality_native(tmp_path, operator):
    from src.compiler.ssa_llvm_backend import (
        emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution)
    source = f"def root(a: str, b: str):\n    return a {operator} b\n"
    contract = ExtractionContract(Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml")
    module, _, exports = lower_ast_source_to_ssa(
        source, "root", extraction_contract=contract,
        progress=lambda _: None, progress_bars=False)
    assert CorrelationTable.build(module).findings(module) == []
    fn = module.functions[exports[0]]
    names = dict(fn.metadata["parameter_names"])
    args = {value.id: value for value in fn.args}
    sequences = module.sequence_tables[exports[0]].sequences
    for name in ("a", "b"):
        desc = sequences[names[name]]
        for member in (*desc.column_value_ids, desc.length_address_id, desc.capacity_value_id):
            assert args[member].accounting["program_abi_parameter"] == name
            assert not is_compiler_owned_storage(args[member])
    emitted = emit_ssa_function_to_llvm(module, exports[0])
    assert emitted.shortfalls == ()
    artifact = compile_artifact(emitted, directory=tmp_path / "llvm")
    output = next(i.args[0].id for block in fn.blocks.values() for i in block.instrs
                  if i.op == "Ret" and not i.attributes.get("fault_exit"))
    cases = [("", ""), ("", "a"), ("a", ""), ("master", "mast"), ("master", "master"), ("ab", "ac"), ("\u03bb", "\u03bb"), ("\u03bb", "\u03b2"), ("", "")]
    for left, right in cases:
        feeds = {}
        for name, word in (("a", left), ("b", right)):
            desc = sequences[names[name]]
            encoded = [string_token(char) for char in word]
            feeds[desc.column_value_ids[0]] = np.array(encoded, dtype=np.int64)
            feeds[desc.length_address_id] = np.array([len(word)], dtype=np.int64)
            feeds[desc.capacity_value_id] = np.array([len(word)], dtype=np.int64)
        before = {key: value.copy() for key, value in feeds.items()}
        result = prepare_artifact_execution(artifact, feeds).run()
        assert bool(result.buffers[output].reshape(-1)[0]) == ((left == right) if operator == "==" else (left != right))
        for key in feeds:
            np.testing.assert_array_equal(feeds[key], before[key])


@pytest.fixture(scope="module")
def equality_module():
    contract = ExtractionContract(Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml")
    module, _, exports = lower_ast_source_to_ssa(
        "def root(a: str, b: str):\n    return a == b\n", "root",
        extraction_contract=contract, progress=lambda _: None, progress_bars=False)
    return module, exports[0]


@pytest.mark.parametrize("damage", ["absent", "wrong_actual", "wrong_scope", "divergent_layout"])
def test_borrowed_helper_requires_exact_occurrence(equality_module, monkeypatch, damage):
    from dataclasses import replace
    from src.compiler.concordance_declarations import FORMAL_ACTUAL_OCCURRENCE
    from src.compiler.ssa_record_return_state import function_scope_of
    import src.compiler.source_evidence_resolver as resolver
    module, root_name = equality_module
    book = module.metadata["identity_book"]
    root = module.functions[root_name]
    helper = next(fn for fn in module.functions.values()
                  if fn.metadata.get("ssa_sequence_operation") == "equal")
    scope = function_scope_of(helper)
    value_id = module.sequence_tables[root_name].sequences[dict(root.metadata["parameter_names"])["a"]].length_address_id
    original = resolver.require_physical_layout
    valid = original(scope, value_id, book=book)
    assert isinstance(valid, resolver.PhysicalLayoutEvidence)
    assert any(cell.page == FORMAL_ACTUAL_OCCURRENCE for cell in valid.proof_cells)
    page = book.page(FORMAL_ACTUAL_OCCURRENCE)
    if damage == "absent":
        rows = page.scope_rows
        monkeypatch.setattr(page, "scope_rows", lambda key: () if key == scope else rows(key))
    elif damage == "wrong_actual":
        latest = page.latest
        monkeypatch.setattr(page, "latest", lambda row: -100 if row[0] == scope else latest(row))
    elif damage == "wrong_scope":
        rows = page.scope_rows
        monkeypatch.setattr(page, "scope_rows", lambda key: tuple((*row[:-1], "unrelated") for row in rows(key)) if key == scope else rows(key))
    else:
        root_scope = function_scope_of(root)
        def changed(key, identity, **kwargs):
            result = original(key, identity, **kwargs)
            if key == root_scope and isinstance(result, resolver.PhysicalLayoutEvidence):
                return replace(result, physical_dtype="float64")
            return result
        monkeypatch.setattr(resolver, "require_physical_layout", changed)
    rejected = original(scope, value_id, book=book)
    assert isinstance(rejected, resolver.UnresolvedSourceDemand)
    assert rejected.missing == "missing_unique_span_abi"


def test_borrowed_span_cycles_are_refused(equality_module):
    from src.compiler.source_evidence_resolver import require_physical_layout, UnresolvedSourceDemand
    from src.compiler.ssa_record_return_state import function_scope_of
    module, root_name = equality_module
    root = module.functions[root_name]
    scope, value_id = function_scope_of(root), root.args[0].id
    result = require_physical_layout(scope, value_id, book=module.metadata["identity_book"],
                                     _seen=frozenset({(scope, value_id)}))
    assert isinstance(result, UnresolvedSourceDemand)
    assert result.missing == "cyclic_formal_layout"


@pytest.mark.parametrize("kind", ["token", "tensor"])
def test_scalar_tokens_and_numeric_arrays_keep_numerical_equality(tmp_path, kind):
    from src.compiler.concordance_declarations import EXECUTABLE_NODE, MetadataRule
    from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
    tensor = kind == "tensor"
    abi = {"storage": "span" if tensor else "scalar", "dtype": "int64",
           "rank": 1 if tensor else 0,
           "python_type": "src.common.tensors.abstraction.AbstractTensor" if tensor else "builtins.str"}
    if tensor:
        abi["shape"] = [3]
    contract = ExtractionContract(Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml").with_program_abi({
        "values": [{"function": "root", "parameter": name, **abi} for name in ("a", "b")],
        "records": {}, "bindings": [],
    })
    module, _, exports = lower_ast_source_to_ssa(
        "def root(a, b):\n    return a == b\n", "root",
        extraction_contract=contract, progress=lambda _: None, progress_bars=False)
    page = module.metadata["identity_book"].page(EXECUTABLE_NODE)
    assert all(page.latest(row).rule.value != "resident_string_comparison" for row in page.rows())
    fn = module.functions[exports[0]]
    emitted = emit_ssa_function_to_llvm(module, exports[0])
    assert emitted.shortfalls == ()
    artifact = compile_artifact(emitted, directory=tmp_path / "llvm")
    names = dict(fn.metadata["parameter_names"])
    left = np.array([1, 2, 3] if tensor else [string_token("one")], dtype=np.int64)
    right = np.array([1, 9, 3] if tensor else [string_token("two")], dtype=np.int64)
    result = prepare_artifact_execution(artifact, {names["a"]: left, names["b"]: right}).run()
    output = next(i.args[0].id for block in fn.blocks.values() for i in block.instrs
                  if i.op == "Ret" and not i.attributes.get("fault_exit"))
    np.testing.assert_array_equal(result.buffers[output].reshape(-1), left == right)
