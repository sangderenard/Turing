import ast
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
from src.compiler.evolution_metagraph import record_evolution
from src.transmogrifier.graph.graph_express2 import ProcessGraph
from src.transmogrifier.graph.python_special_cases import (
    interpret_python_special_case,
)


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


def _call_nodes(graph):
    return [
        (node_id, data)
        for node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Call)
    ]


def test_python_adapter_materializes_intrinsic_receipt_on_canonical_node():
    contract = ExtractionContract(CONTRACT)
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(
        "def kernel():\n    return float(1)\n",
        resolve_unresolved_parents=True,
        pursuit_roots=("kernel",),
        parent_include=contract,
    )

    _, call = _call_nodes(graph)[0]
    assert call["type"] == "float"
    assert call["attributes"]["extraction_action"] == "intrinsic"
    assert call["attributes"]["extraction_identity"] == "builtins.float"
    assert call["extraction_contract"]["rule_id"] == "control-and-scalar-builtins"


@pytest.mark.parametrize(
    ("source", "action", "identity"),
    [
        (
            "import math\ndef kernel(x):\n    return math.sin(x)\n",
            "use_native",
            "math.sin",
        ),
        (
            "import inspect\ndef kernel(x):\n    return inspect.signature(x)\n",
            "reject",
            "inspect.signature",
        ),
    ],
)
def test_python_boundary_keeps_call_arguments_but_never_ingests_callee(
    source, action, identity,
):
    contract = ExtractionContract(CONTRACT)
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(
        source,
        resolve_unresolved_parents=True,
        pursuit_roots=("kernel",),
        parent_include=contract,
    )

    call_id, call = _call_nodes(graph)[0]
    assert call["type"] == "Call"
    assert call["attributes"]["extraction_action"] == action
    assert call["attributes"]["extraction_identity"] == identity
    assert graph.G.in_degree(call_id) >= 2  # authored callee spelling + argument
    receipt = graph.G.graph["extraction_boundary_calls"][0][
        "extraction_contract"
    ]
    assert receipt["action"] == action
    assert receipt["identity"] == identity
    assert "resolved_ast_parent" not in call["attributes"]
    if action == "reject":
        assert graph.G.graph["rejected_extraction_calls"]


def test_first_live_call_event_is_already_contract_classified():
    contract = ExtractionContract(CONTRACT)
    with record_evolution() as metagraph:
        graph = ProcessGraph(materialize_memory=False)
        graph.build_from_ast(
            "import math\ndef kernel(x):\n    return math.sin(x)\n",
            resolve_unresolved_parents=True,
            pursuit_roots=("kernel",),
            parent_include=contract,
        )

    process_graph = next(
        item for item in metagraph.snapshot().graphs
        if item.stage == "process-graph"
    )
    call_id, _ = _call_nodes(graph)[0]
    events = [
        event for event in metagraph.snapshot().events
        if event.graph == process_graph
        and event.component is not None
        and event.component.local_id == str(call_id)
    ]
    assert events[0].kind == "component-spawn"
    assert events[0].detail["attributes"]["extraction_action"] == "use_native"
    assert events[0].detail["attributes"]["extraction_identity"] == "math.sin"


def test_admitted_python_call_keeps_definition_link_and_receipt():
    contract = ExtractionContract(CONTRACT)
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(
        "def kernel():\n    return inner()\n",
        resolve_unresolved_parents=True,
        pursuit_roots=("kernel",),
        parent_bindings={"inner": _contract_inner},
        parent_include=contract,
    )

    _, call = _call_nodes(graph)[0]
    assert call["type"] == "Call"
    assert call["attributes"]["extraction_action"] == "ingest_python"
    assert call["attributes"]["resolved_ast_parent"] in graph.G


def test_print_host_boundary_uses_existing_stream_operator():
    contract = ExtractionContract(CONTRACT)
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(
        "def kernel():\n    print('ready')\n",
        resolve_unresolved_parents=True,
        pursuit_roots=("kernel",),
        parent_include=contract,
    )

    _, call = _call_nodes(graph)[0]
    assert call["type"] == "stream_publish"
    assert call["attributes"]["extraction_action"] == "python_host_call"
    assert call["attributes"]["extraction_identity"] == "builtins.print"


@pytest.mark.parametrize("action", ["ingest_python", "decompile_machine"])
def test_source_admitting_actions_remain_call_shaped(action):
    call = ast.parse("target(value)").body[0].value
    call._extraction_contract = {
        "identity": "example.target",
        "action": action,
        "rule_id": "test",
        "classification": "repository_python",
        "parameters": {},
    }

    special = interpret_python_special_case(call)

    assert special.type == "Call"
    assert not special.terminal
    assert special.attributes["extraction_action"] == action


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
