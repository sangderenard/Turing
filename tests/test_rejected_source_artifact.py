"""The normal product retains refused work without admitting it as native SSA."""
import hashlib
import pickle
from pathlib import Path

import pytest

from src.compiler import project_compilation_product as product
from src.compiler import structural_concordance
from src.compiler.identity_concordance import ConcordanceRefusal, CorrelationTable
from src.transmogrifier.ssa import SSAValue


CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml"


@pytest.fixture
def refused_source(monkeypatch, tmp_path):
    original = structural_concordance.publish_module_structures

    def poison(module):
        original(module)
        for function in module.functions.values():
            for block in function.blocks.values():
                for instruction in block.instrs:
                    if instruction.op == "Ret" and instruction.args:
                        instruction.args[0] = SSAValue(987654321, "int64")
                        return
        raise AssertionError("expected a real return to poison")

    monkeypatch.setattr(structural_concordance, "publish_module_structures", poison)
    source = tmp_path / "probe.py"
    source.write_text("def root():\n    return 7\n", encoding="utf-8")
    return source


def test_rejected_product_keeps_exact_book_and_immutable_artifacts(refused_source, tmp_path):
    root = tmp_path / "product"
    artifacts = []
    for _ in range(2):
        with pytest.raises(ConcordanceRefusal, match="987654321") as raised:
            product.compile_project_call(refused_source, "root", root,
                                         extraction_contract=CONTRACT, progress_bars=False)
        receipt = raised.value.to_failure_mapping()
        artifact = root / receipt["rejected_repository_ssa"]
        payload = artifact.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == receipt["rejected_repository_ssa_sha256"]
        assert receipt["rejected_repository_ssa_linkable"] is False
        module, _outputs, exports = pickle.loads(payload)
        assert module.call_table.book is module.metadata["identity_book"]
        assert any(f.value_id == 987654321 for f in CorrelationTable.build(module).findings(module))
        with pytest.raises(ConcordanceRefusal):
            product._require_native_root_semantics(module, exports[0])
        artifacts.append((artifact, payload))
    assert artifacts[0][0] != artifacts[1][0]
    assert artifacts[0][0].read_bytes() == artifacts[0][1]
    assert not (root / "repository-ssa.pkl").exists()
    assert not (root / "unit.json").exists()


def test_capture_failure_preserves_original_refusal(refused_source, tmp_path, monkeypatch):
    def fail_capture(*_args, **_kwargs):
        raise OSError("capture unavailable")

    monkeypatch.setattr(product, "_dump_pickle_artifact", fail_capture)
    with pytest.raises(ConcordanceRefusal, match="987654321") as raised:
        product.compile_project_call(refused_source, "root", tmp_path / "product",
                                     extraction_contract=CONTRACT, progress_bars=False)
    assert raised.value.to_failure_mapping() == {
        "rejected_repository_ssa_error": "capture unavailable",
    }
