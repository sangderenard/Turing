import ast
import contextlib
import io
import json
import math
from pathlib import Path

import pytest
import yaml

from src.compiler.extraction_contract import (
    ExtractionAction,
    ExtractionContract,
    ExtractionContractError,
)
from src.common.tensors.abstract_nn import Adam
from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.autograd import autograd
from src.common.tensors.topological_reducer import (
    reduce_abstract_tensor_topology,
)
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

    assert contract.execution.host_runtime == "python"
    assert contract.execution.dependency_search == "reachable"
    assert contract.execution.native_lowering == "opportunistic"
    assert contract.execution.dispatch_unit == "isolated_numeric_subgraph"
    assert contract.execution.unlowered_behavior == "execute_in_python"
    assert contract.execution.require_full_native is False
    assert contract.execution.backward_source == "process_graph"
    assert contract.execution.numeric_semantics == "abstract_tensor"
    assert contract.execution.scalar_promotion == "all_numeric"
    state_abi = contract.program_abi.records_for_function(
        "symbolic_fluid_advance"
    )["state"]
    assert state_abi.fields["height"].storage == "span"
    assert state_abi.fields["height"].rank == 2
    assert state_abi.fields["dx"].dtype == "float64"
    assert state_abi.fields["dx"].mutable is False
    dt_abi = contract.program_abi.values_for_function(
        "step_with_dt_control_used"
    )["dt"]
    assert dt_abi.field.storage == "scalar"
    assert dt_abi.field.dtype == "float64"
    assert dt_abi.python_type == "builtins.float"

    assert contract.decide(Adam).action is ExtractionAction.INGEST_PYTHON
    assert contract.decide(range).action is ExtractionAction.INTRINSIC
    assert contract.decide(print).action is ExtractionAction.PYTHON_HOST_CALL
    assert contract.decide(math.sin).action is ExtractionAction.USE_NATIVE
    assert not contract.decide(math.sin).ingest_parent
    assert all(
        receipt["action"] != ExtractionAction.DECOMPILE_MACHINE.value
        for receipt in contract.receipts()
    )


@pytest.mark.parametrize("target", [open, io.open, Path.open, Path.read_bytes])
def test_python_filesystem_uses_the_existing_shell_file_broker(target):
    decision = ExtractionContract(CONTRACT).decide(target)

    assert decision.action is ExtractionAction.PYTHON_HOST_CALL
    assert decision.rule_id == "python-filesystem-is-a-shell-boundary"
    assert decision.parameters["execution"] == "shell_io.file_broker"
    assert decision.parameters["shell_capability"] == "files"
    assert decision.parameters["shell_abi"] == "turing-shell-io-abi.files"


def test_process_graph_retains_filesystem_call_as_a_shell_boundary():
    contract = ExtractionContract(CONTRACT)
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(
        """
from pathlib import Path
def kernel(name):
    path = Path(name)
    return path.read_bytes()
""",
        resolve_unresolved_parents=True,
        pursuit_roots=("kernel",),
        parent_include=contract,
    )

    boundaries = graph.G.graph["extraction_boundary_calls"]
    file_boundary = next(
        item for item in boundaries
        if (item.get("extraction_contract") or {}).get("identity")
        == "pathlib.Path.read_bytes"
    )
    receipt = file_boundary["extraction_contract"]
    assert receipt["action"] == "python_host_call"
    assert receipt["parameters"]["execution"] == "shell_io.file_broker"


def test_program_abi_correlates_unannotated_method_receiver_by_class_identity():
    contract = ExtractionContract(CONTRACT)

    records = contract.program_abi.records_for_function(
        "pi_update",
        method_owner="STController",
        parameters=("self", "dt_prev", "dt_pen"),
    )

    assert records["self"].identity.endswith(".STController")


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
    unsafe.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ExtractionContractError, match="explicit_opt_in"):
        ExtractionContract(unsafe)


def test_program_abi_rejects_untyped_physical_spans(tmp_path):
    raw = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    raw["program_abi"]["records"]["Broken"] = {
        "fields": {"values": {"storage": "span", "rank": 1}}
    }
    raw["program_abi"]["bindings"].append({
        "function": "*", "parameter": "broken", "record": "Broken",
    })
    path = tmp_path / "broken-program-abi.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ExtractionContractError, match="dtype is required"):
        ExtractionContract(path)


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
    assert graph.G.graph["execution_contract"] == contract.execution.receipt()
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
    assert call["attributes"]["backend_intrinsic_candidate"] == {
        "semantic_identity": "builtins.float",
        "lowering_namespace": "python_language",
        "ingested_fallback": False,
    }
    assert call["extraction_contract"]["rule_id"] == "control-and-scalar-builtins"


def test_intrinsic_candidate_can_retain_an_ingested_semantic_fallback(tmp_path):
    raw = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    raw["rules"].insert(0, {
        "id": "test-fallback-intrinsic",
        "match": {"identity": "*._contract_inner"},
        "action": "intrinsic",
        "parameters": {
            "lowering_namespace": "test-native",
            "ingest_fallback_source": True,
        },
    })
    path = tmp_path / "fallback-intrinsic.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    decision = ExtractionContract(path).decide(_contract_inner)

    assert decision.action is ExtractionAction.INTRINSIC
    assert decision.ingest_parent
    assert decision.parameters["lowering_namespace"] == "test-native"


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


def test_native_boundary_receipt_survives_function_graph_reduction():
    contract = ExtractionContract(CONTRACT)
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(
        "import pickle\ndef kernel(stream):\n    return pickle.load(stream)\n",
        resolve_unresolved_parents=True,
        pursuit_roots=("kernel",),
        parent_include=contract,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        reduce_abstract_tensor_topology(graph)

    function_graph = graph.function_table.entry("kernel").graph
    call = next(
        data for _node_id, data in function_graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Call)
        and ast.unparse(data["expr_obj"]) == "pickle.load(stream)"
    )
    attributes = call["attributes"]
    assert attributes["extraction_action"] == "use_native"
    assert attributes["extraction_identity"] == "_pickle.load"
    assert attributes["extraction_contract"]["parameters"] == {
        "loader": "existing_module",
        "symbol_resolution": "in_place",
        "callbacks": "reject",
        "execution": "shell_io.external_references",
        "shell_capability": "host_references",
        "shell_abi": "turing-shell-io-abi.external_references",
        "external_domain": "host_system",
    }


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

    call_id, call = _call_nodes(graph)[0]
    assert call["type"] == "stream_publish"
    assert call["attributes"]["extraction_action"] == "python_host_call"
    assert call["attributes"]["extraction_identity"] == "builtins.print"
    assert graph.G.in_degree(call_id) == 1


def test_nested_identity_replacements_retain_authored_dataflow():
    contract = ExtractionContract(CONTRACT)
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(
        "def kernel(loss):\n    print(float(loss))\n",
        resolve_unresolved_parents=True,
        pursuit_roots=("kernel",),
        parent_include=contract,
    )

    calls = {
        data["attributes"]["extraction_identity"]: (node_id, data)
        for node_id, data in _call_nodes(graph)
    }
    publish_id, publish = calls["builtins.print"]
    cast_id, cast = calls["builtins.float"]

    assert publish["type"] == "stream_publish"
    assert cast["type"] == "float"
    assert graph.G.has_edge(cast_id, publish_id)
    assert graph.G.in_degree(publish_id) == 1
    assert graph.G.in_degree(cast_id) == 1
    assert publish["attributes"]["python_replacement_kind"] == "operator"
    assert cast["attributes"]["python_replacement_kind"] == "operator"


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


def test_autograd_instance_occurrences_pursue_the_real_tape_source():
    contract = ExtractionContract(CONTRACT)
    assert contract.decide(autograd).receipt()["occurrence"] == "instance"
    assert contract.decide(autograd.tape).parameters["occurrence_pursuit"] == (
        "referenced_attributes"
    )

    graph = ProcessGraph(materialize_memory=False)
    graph.python_bindings = {"loss": AbstractTensor}
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            ast.parse(
                "class Unrelated:\n"
                "    def backward(self):\n"
                "        return None\n\n"
                "def kernel():\n"
                "    loss.backward()\n"
            ),
            resolve_unresolved_parents=True,
            pursuit_roots=("kernel",),
            parent_include=contract,
        )

    identities = {
        tuple(identity)
        for _node_id, data in graph.G.nodes(data=True)
        if (identity := getattr(
            data.get("expr_obj"),
            "_python_source_identity",
            None,
        ))
    }
    assert ("src.common.tensors.autograd", "backward") in identities
    assert (
        "src.common.tensors.autograd",
        "GradTape.mark_loss",
    ) in identities
    assert (
        "src.common.tensors.autograd",
        "GradTape.parameter_tensors",
    ) in identities
    assert ("src.common.tensors.autograd", "Autograd.grad") in identities

    occurrences = {
        receipt["identity"]
        for _node_id, data in graph.G.nodes(data=True)
        for receipt in (data.get("attributes") or {}).get(
            "extraction_occurrences",
            (),
        )
    }
    assert "src.common.tensors.autograd.Autograd" in occurrences
    assert "src.common.tensors.autograd.GradTape" in occurrences
