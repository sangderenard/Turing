"""Authored primitive source-entry spans remain caller-owned native storage."""
from pathlib import Path
import numpy as np
import pytest
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.ssa_storage_requirements import is_compiler_owned_storage
from src.compiler.string_table import string_token

@pytest.mark.parametrize("annotation", ["str", "bytes", "bytearray"])
def test_primitive_span_entry_owns_native_lengths_and_capacity(tmp_path, annotation):
    from src.compiler.ssa_llvm_backend import (
        emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution)
    source = f"def root(a: {annotation}, b: {annotation}):\n    return len(a) + len(b)\n"
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
    cases = [("", ""), ("", "a"), ("master", "mast"), ("é🦉", "é🐈"), ("", "")]
    for left, right in cases:
        if annotation != "str":
            left, right = left.encode("utf-8"), right.encode("utf-8")
        feeds = {}
        for name, word in (("a", left), ("b", right)):
            desc = sequences[names[name]]
            encoded = [string_token(char) for char in word] if annotation == "str" else list(word)
            feeds[desc.column_value_ids[0]] = np.array(encoded, dtype=np.int64)
            feeds[desc.length_address_id] = np.array([len(word)], dtype=np.int64)
            feeds[desc.capacity_value_id] = np.array([len(word)], dtype=np.int64)
        before = {key: value.copy() for key, value in feeds.items()}
        result = prepare_artifact_execution(artifact, feeds).run()
        assert int(result.buffers[output].reshape(-1)[0]) == len(left) + len(right)
        for key in feeds:
            np.testing.assert_array_equal(feeds[key], before[key])
