"""Resident glyph keys must stop before scalar keyed helper emission."""
from pathlib import Path
import pytest
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.precompile_to_ssa import SSALoweringIncompleteError
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.precompile_to_ssa import _ControlSSABuilder

PREFIX = """from dataclasses import dataclass
@dataclass
class Node:
    key: str
"""

def lower(body):
    return lower_ast_source_to_ssa(
        PREFIX + "def root(nodes: list[Node]) -> int:\n" + body, "root",
        extraction_contract=ExtractionContract(Path(__file__).parents[1] /
            "extraction_contracts/program_extraction.yaml"),
        progress=lambda _: None, progress_bars=False)

@pytest.mark.parametrize("body", [
    "    keys = {node.key for node in nodes}\n    return len(keys)\n",
    "    keys = {node.key: 7 for node in nodes}\n    return len(keys)\n",
    "    keys = {node.key for node in nodes}\n    out = list(keys)\n    return len(out)\n",
    "    keys = set()\n    for node in nodes:\n        keys.add(node.key)\n    return len(keys)\n",
    "    keys = {'a': 7}\n    total = 0\n    for node in nodes:\n        if node.key in keys:\n            total += 1\n    return total\n",
    "    keys = {'a': 7}\n    total = 0\n    for node in nodes:\n        total += keys[node.key]\n    return total\n",
    "    keys = set()\n    for node in nodes:\n        k = node.key\n        keys.add(k)\n    return len(keys)\n",
    "    keys = {'a': 7}\n    total = 0\n    for node in nodes:\n        k = node.key\n        total += keys[k]\n    return total\n",
])
def test_resident_glyph_key_refuses_before_native(body):
    with pytest.raises(SSALoweringIncompleteError) as caught:
        lower(body)
    assert any(f.name == "resident-glyph-key" for _, f in caught.value.failures)


@pytest.mark.parametrize("body", [
    "    keys = {'a': 7, 'b': 9}\n    return keys['a']\n",
    "    keys = {j for j in range(4)}\n    return len(keys)\n",
])
def test_scalar_keys_pass_the_guard_before_existing_later_audits(monkeypatch, body):
    # Stop at this exact successful dispatch. Some current whole-source keyed
    # fixtures subsequently fail the independent formal-occurrence audit.
    class Observed(Exception):
        pass
    original = _ControlSSABuilder._refuse_resident_glyph_keys
    def inspect(builder, ids):
        original(builder, ids)
        raise Observed()
    monkeypatch.setattr(_ControlSSABuilder, '_refuse_resident_glyph_keys', inspect)
    with pytest.raises(Observed):
        lower(body)


def test_owned_glyph_key_use_is_rejected_by_existing_read_phase():
    with pytest.raises(SSALoweringIncompleteError) as caught:
        lower("    out: list[str] = []\n    for node in nodes:\n        out.append(node.key)\n    keys = set()\n    for word in out:\n        keys.add(word)\n    return len(keys)\n")
    assert any(f.name == "string-row-read-phase" for _, f in caught.value.failures)


def test_annotated_runtime_string_key_is_not_a_vocabulary_token():
    source = "def root(key: str) -> int:\n    keys = {'a': 7}\n    return keys[key]\n"
    with pytest.raises(SSALoweringIncompleteError) as caught:
        lower_ast_source_to_ssa(source, 'root', extraction_contract=ExtractionContract(
            Path(__file__).parents[1] / 'extraction_contracts/program_extraction.yaml'),
            progress=lambda _: None, progress_bars=False)
    assert any(f.name == 'resident-glyph-key' for _, f in caught.value.failures)
