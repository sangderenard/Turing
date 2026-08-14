import json
import math
from pathlib import Path

import pytest

from src.compiler.extraction_contract import (
    ExtractionAction,
    ExtractionContract,
    ExtractionContractError,
)
from src.common.tensors.abstract_nn import Adam
from src.transmogrifier.graph.graph_express2 import ProcessGraph


CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts"
    / "program_extraction.yaml"
)


def _contract_inner():
    return 1


def _contract_outer():
    return _contract_inner()


def test_default_contract_draws_python_native_and_decompile_lines():
    contract = ExtractionContract(CONTRACT)

    assert contract.decide(Adam).action is ExtractionAction.INGEST_PYTHON
    assert contract.decide(range).action is ExtractionAction.INTRINSIC
    assert contract.decide(print).action is ExtractionAction.PYTHON_HOST_CALL
    assert contract.decide(math.sin).action is ExtractionAction.USE_NATIVE
    assert not contract.decide(math.sin).ingest_parent
    assert all(
        receipt["action"] != ExtractionAction.DECOMPILE_MACHINE.value
        for receipt in contract.receipts()
    )


def test_contract_must_cover_every_class_and_explicitly_opt_into_decompile(tmp_path):
    incomplete = tmp_path / "incomplete.yaml"
    incomplete.write_text(
        "version: 1\nmode: exhaustive\ndefaults:\n  builtin: intrinsic\n",
        encoding="utf-8",
    )
    with pytest.raises(ExtractionContractError, match="cover exactly"):
        ExtractionContract(incomplete)

    raw = {
        "version": 1,
        "mode": "exhaustive",
        "defaults": {
            name: (
                {"action": "decompile_machine", "parameters": {
                    "max_functions": 1,
                    "max_total_bytes": 1,
                    "max_dependency_depth": 1,
                }}
                if name == "unknown"
                else {"action": "reject", "parameters": {"reason": "test"}}
            )
            for name in (
                "authored_python", "repository_python", "third_party_python",
                "stdlib_python", "builtin", "native_extension",
                "dynamic_library", "unknown",
            )
        },
    }
    unsafe = tmp_path / "unsafe.yaml"
    import yaml
    unsafe.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ExtractionContractError, match="explicit_opt_in"):
        ExtractionContract(unsafe)


def test_parent_expansion_records_intrinsic_without_host_decompilation():
    contract = ExtractionContract(CONTRACT)
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(
        "def kernel():\n    return float(1)\n",
        resolve_unresolved_parents=True,
        pursuit_roots=("kernel",),
        parent_include=contract,
    )

    receipts = graph.G.graph["extraction_contract_receipts"]
    float_receipt = next(item for item in receipts if item["identity"] == "builtins.float")
    assert float_receipt["action"] == "intrinsic"
    assert float_receipt["rule_id"] == "control-and-scalar-builtins"
    assert graph.G.graph["extraction_contract_fingerprint"] == contract.fingerprint
    assert json.loads(contract.receipt_json())[0]["identity"] == "builtins.float"


def test_python_dependency_depth_is_a_hard_contract_ceiling():
    contract = ExtractionContract(CONTRACT)
    contract.limits["python_source"]["max_dependency_depth"] = 1
    graph = ProcessGraph(materialize_memory=False)

    with pytest.raises(RuntimeError, match="max_dependency_depth.*2/1"):
        graph.build_from_ast(
            "def kernel():\n    return outer()\n",
            resolve_unresolved_parents=True,
            pursuit_roots=("kernel",),
            parent_bindings={"outer": _contract_outer},
            parent_include=contract,
        )
