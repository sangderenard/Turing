import ast
import contextlib
import io
import json
import math
import re
import time
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


def test_program_abi_table_row_retains_dynamic_span_rank():
    contract = ExtractionContract(CONTRACT)
    steps = contract.program_abi.records["CompilerFusedProgram"].fields["steps"]
    assert steps.table_columns == (
        {"name": "input_ids", "dtype": "int64", "rank": 1},
        {"name": "result_id", "dtype": "int64", "rank": 0},
    )
    round_trip = contract.with_program_abi(contract.program_abi.receipt())
    assert (
        round_trip.program_abi.records["CompilerFusedProgram"]
        .fields["steps"].table_columns
        == steps.table_columns
    )


def test_program_abi_table_row_keeps_nested_table_as_child_handle():
    contract = ExtractionContract(CONTRACT).with_program_abi({
        "records": {
            "Instruction": {
                "identity": "src.transmogrifier.ssa.Instr",
                "fields": {
                    "op": {"storage": "scalar", "dtype": "int64"},
                    "args": {
                        "storage": "table", "rank": 1,
                        "row_record": "Value",
                    },
                },
            },
            "Value": {
                "identity": "src.transmogrifier.ssa.SSAValue",
                "fields": {
                    "dtype": {"storage": "scalar", "dtype": "int64"},
                },
            },
            "Block": {
                "identity": "src.transmogrifier.ssa.BasicBlock",
                "fields": {
                    "instrs": {
                        "storage": "table", "rank": 1,
                        "row_record": "Instruction",
                    },
                },
            },
        },
        "bindings": [],
        "values": [],
    })

    instrs = contract.program_abi.records["Block"].fields["instrs"]
    assert instrs.table_columns == (
        {"name": "op", "dtype": "int64", "rank": 0},
        {"name": "args", "dtype": "int64", "rank": 0},
    )
    assert instrs.row_record == "Instruction"
    args = contract.program_abi.records["Instruction"].fields["args"]
    assert args.row_record == "Value"
    assert args.table_columns == (
        {"name": "dtype", "dtype": "int64", "rank": 0},
    )

    round_trip = contract.with_program_abi(contract.program_abi.receipt())
    assert (
        round_trip.program_abi.records["Block"].fields["instrs"].table_columns
        == instrs.table_columns
    )


def test_source_availability_is_probed_once_per_exact_definition(monkeypatch):
    import inspect
    contract = ExtractionContract(CONTRACT)
    original = inspect.getsource
    probes = []

    def source(value):
        probes.append(value)
        return original(value)

    monkeypatch.setattr(inspect, "getsource", source)
    for _ in range(20):
        assert contract.subject(_contract_inner).source_available
    assert probes == [_contract_inner]
    # Two definitions can carry the same displayed name without being the
    # same source object. Their availability must still be probed separately.
    alternate = type(_contract_inner)(_contract_inner.__code__, globals(),
                                      _contract_inner.__name__)
    assert contract.subject(alternate).source_available
    assert len(probes) == 2
    contract.with_program_abi(contract.program_abi).subject(_contract_inner)
    assert len(probes) == 3
    for _ in range(300):
        definition = type(_contract_inner)(_contract_inner.__code__, globals(),
                                           _contract_inner.__name__)
        assert contract.subject(definition).source_available
    assert len(contract._source_availability) == 256
    before = len(probes)
    contract.subject(_contract_inner)
    assert len(probes) == before + 1


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
    report = contract.program_abi.records["Metrics"].fields[
        "unresolved_report"
    ]
    assert report.storage == "table"
    assert report.rank == 1
    assert report.mutable is True
    assert report.table_columns == ({"name": "token", "dtype": "int64", "rank": 0},)
    targets = contract.program_abi.records["Targets"]
    assert targets.fields["energy_exchange_fraction"].optional is True
    assert targets.fields["shadow_growth_max"].optional is True
    assert targets.fields["energy_exchange_fraction"].dtype == "float64"
    controller = contract.program_abi.records["STController"]
    assert controller.fields["dt_min"].optional is True
    assert controller.fields["dt_max"].optional is True
    assert controller.fields["Kp"].optional is False
    assert controller.receipt()["fields"]["dt_min"]["optional"] is True

    assert contract.decide(Adam).action is ExtractionAction.INGEST_PYTHON
    assert contract.decide(range).action is ExtractionAction.INTRINSIC
    assert contract.decide(print).action is ExtractionAction.PYTHON_HOST_CALL
    assert contract.decide(math.sin).action is ExtractionAction.USE_NATIVE
    assert not contract.decide(math.sin).ingest_parent
    assert all(
        receipt["action"] != ExtractionAction.DECOMPILE_MACHINE.value
        for receipt in contract.receipts()
    )


def test_program_abi_input_accepts_closed_type_expectation_union():
    contract = ExtractionContract(CONTRACT).with_program_abi({
        "records": {},
        "bindings": [],
        "values": [{
            "function": "root",
            "parameter": "value",
            "storage": "reference",
            "type_expectations": ["builtins.dict", "builtins.list"],
        }],
    })

    binding = contract.program_abi.values_for_function("root")["value"]
    assert binding.admitted_types == ("builtins.dict", "builtins.list")
    assert binding.receipt()["type_expectations"] == [
        "builtins.dict", "builtins.list",
    ]


def test_reference_field_normalizes_its_repository_physical_dtype():
    contract = ExtractionContract(CONTRACT).with_program_abi({
        "records": {
            "State": {
                "identity": "tests.State",
                "fields": {
                    "payload": {"storage": "reference", "optional": True},
                },
            },
        },
        "bindings": [],
        "values": [],
    })

    field = contract.program_abi.records["State"].fields["payload"]
    assert field.dtype == "opaque_ref"
    assert field.receipt()["dtype"] == "opaque_ref"


@pytest.mark.parametrize("native_abi,expected", [
    (None, ExtractionAction.REJECT),
    ("cpython-c-api", ExtractionAction.REJECT),
    ("c", ExtractionAction.USE_NATIVE),
])
def test_full_native_extension_requires_declared_non_python_abi(monkeypatch, native_abi, expected):
    from src.compiler.extraction_contract import ExtractionSubject

    contract = ExtractionContract(CONTRACT).with_execution_file(
        CONTRACT.with_name("vehicle_full_native_execution.yaml"))
    subject = ExtractionSubject("unregistered_extension", "draw", "builtin",
                                "call", "", "native_extension", False)
    monkeypatch.setattr(contract, "subject", lambda value: subject)
    parameters = {"loader": "existing_module", "callbacks": "reject"}
    if native_abi is not None:
        parameters["native_abi"] = native_abi
    contract.defaults["native_extension"] = (ExtractionAction.USE_NATIVE, parameters)
    decision = contract.decide(object())
    assert decision.action is expected
    if expected is ExtractionAction.REJECT:
        assert decision.parameters["reason"] == "native_extension_requires_non_python_abi"
        assert decision.rule_id == "execution:native_extension_abi_required"


def test_full_native_link_gate_rejects_python_and_unresolved_boundaries():
    from types import SimpleNamespace

    from src.compiler.fortran_c_shell import (
        _full_native_link_failures,
        _undefined_repository_ssa_operands,
    )
    from src.transmogrifier.ssa import Instr, SSAValue

    supplied = SSAValue(1, dtype="float64")
    missing = SSAValue(2, dtype="float64")
    result = SSAValue(3, dtype="float64")
    module = SimpleNamespace(functions={
        "root": SimpleNamespace(
            args=[supplied],
            metadata={},
            blocks={"entry": SimpleNamespace(instrs=[
                Instr("Add", [supplied, missing], result),
            ])},
        ),
    })
    undefined = _undefined_repository_ssa_operands(module)
    assert len(undefined) == 1
    assert {key: undefined[0][key] for key in (
        "function", "block", "operation", "value_id",
    )} == {
        "function": "root",
        "block": "entry",
        "operation": "Add",
        "value_id": 2,
    }

    failures = _full_native_link_failures(
        ({
            "extraction_contract": {
                "identity": "json.loads",
                "action": "use_native",
                "parameters": {
                    "native_abi": "cpython-c-api",
                    "callbacks": "reject",
                    "shell_profiles": ["python", "cpython-c"],
                },
            },
        },),
        ({"identity": "pathlib.Path.read_bytes"},),
        ({"caller": "main", "callsite_id": 7, "callee": "helper"},),
        undefined,
    )

    assert len(failures["unmaterialized_boundaries"]) == 1
    assert len(failures["unresolved_call_records"]) == 1
    assert failures["undefined_operands"] == undefined
    assert failures["non_native_boundaries"] == ({
        "identity": "json.loads",
        "action": "use_native",
        "reasons": ("cpython-c-api", "python-shell-profile"),
    },)


@pytest.mark.parametrize("target", [open, io.open, Path.open, Path.read_bytes])
def test_python_filesystem_uses_the_existing_shell_file_broker(target):
    decision = ExtractionContract(CONTRACT).decide(target)

    assert decision.action is ExtractionAction.PYTHON_HOST_CALL
    assert decision.rule_id == "python-filesystem-is-a-shell-boundary"
    assert decision.parameters["execution"] == "shell_io.file_broker"
    assert decision.parameters["shell_capability"] == "files"
    assert decision.parameters["shell_abi"] == "turing-shell-io-abi.files"


@pytest.mark.parametrize(
    ("target", "rule_id"),
    [
        (time.perf_counter, "native-performance-counter-scalar-result"),
        (time.time, "native-wall-clock-scalar-result"),
    ],
)
def test_clock_readings_return_an_inline_float64_scalar(target, rule_id):
    decision = ExtractionContract(CONTRACT).decide(target)

    assert decision.action is ExtractionAction.USE_NATIVE
    assert decision.rule_id == rule_id
    assert decision.parameters["result_dtype"] == "float64"
    assert decision.parameters["runtime_owner"] == "shell"


def test_frozen_stdlib_definition_is_stdlib_not_repository_source():
    # os.path.exists is genericpath.exists, whose origin is the pseudo-file
    # "<frozen genericpath>". Resolved as a path it fell inside the
    # repository root (ticket b043a7c9, case 6b7d429f).
    import genericpath
    from importlib.machinery import FrozenImporter

    spec = genericpath.__spec__
    if spec.origin != "frozen" or spec.loader is not FrozenImporter:
        pytest.skip("this interpreter does not freeze genericpath")
    decision = ExtractionContract(CONTRACT).decide(genericpath.exists)

    assert decision.subject.classification == "stdlib_python"
    assert decision.rule_id == "default:stdlib_python"
    assert decision.action is ExtractionAction.USE_NATIVE


def test_wall_clock_return_is_supplied_by_the_native_callsite_as_float64():
    # Ticket b043a7c9 / case 6b7d429f: the contract's result_dtype must reach
    # the lowered program, not stop at ExtractionContract.decide().
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    with contextlib.redirect_stdout(io.StringIO()):
        module, outputs, exports = lower_ast_source_to_ssa(
            "import time\ndef entry():\n    return time.time()\n",
            "entry", name="wall_clock",
            extraction_contract=ExtractionContract(CONTRACT),
            progress=lambda _message: None, progress_bars=False,
        )

    assert exports == ("wall_clock__entry",)
    returned, = outputs["wall_clock__entry"]
    assert returned.dtype == "float64"
    operations = tuple(
        instruction
        for block in module.functions["wall_clock__entry"].blocks.values()
        for instruction in block.instrs
    )
    native, = (
        instruction for instruction in operations
        if (instruction.attributes or {}).get("external_reference")
    )
    assert native.attributes["external_identity"] == "time.time"
    assert native.attributes["runtime_owner"] == "shell"
    assert native.res.id == returned.id
    assert native.res.dtype == "float64"
    ret, = (instruction for instruction in operations if instruction.op == "Ret")
    assert [value.id for value in ret.args] == [returned.id]


@pytest.mark.parametrize(
    "source,expected_identity,expected_count",
    (
        ("import time\ndef entry():\n    t = time.time()\n    return t + 1.0\n",
         "time.time", 1),
        ("import time\ndef entry():\n"
         "    t0 = time.perf_counter()\n    return time.perf_counter() - t0\n",
         "time.perf_counter", 2),
    ),
    ids=("wall_clock_plus_constant", "performance_counter_difference"),
)
def test_clock_reading_used_in_arithmetic_is_a_float64_host_scalar(
    source, expected_identity, expected_count,
):
    # Ticket b043a7c9 / case 6b7d429f: a use_native float64 result consumed
    # by arithmetic was refused by _refuse_unsettled_host_scalars as
    # missing_source_value_class.  The contract's result_dtype now publishes
    # the call's class, so the arithmetic lowers to a float64 return.  Each
    # clock read stays its own native call: two perf_counter reads must not
    # collapse into one (review 753568f5, comment 508).  The whole sequence
    # of external calls is pinned by identity, so a missing clock read, an
    # extra unrelated external call or a wrong identity all fail (review
    # f7b4f6ca, comments 520-523).
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    with contextlib.redirect_stdout(io.StringIO()):
        module, outputs, exports = lower_ast_source_to_ssa(
            source, "entry", name="clock_arithmetic",
            extraction_contract=ExtractionContract(CONTRACT),
            progress=lambda _message: None, progress_bars=False,
        )

    assert exports == ("clock_arithmetic__entry",)
    returned, = outputs["clock_arithmetic__entry"]
    assert returned.dtype == "float64"
    natives = tuple(
        instruction
        for block in module.functions["clock_arithmetic__entry"].blocks.values()
        for instruction in block.instrs
        if (instruction.attributes or {}).get("external_reference")
    )
    assert tuple(
        native.attributes["external_identity"] for native in natives
    ) == (expected_identity,) * expected_count
    assert all(native.res.dtype == "float64" for native in natives)


def test_returned_file_broker_result_is_refused_with_a_named_shortfall():
    # Ticket b043a7c9 / case 6b7d429f, plan C: the shell runs the file-broker
    # plan after the entry, so a returned f.write() count has no producer in
    # the entry.  The refusal must name that cause, not the hidden symptom
    # "Ret returns none" from final concordance.
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.precompile_to_ssa import SSALoweringIncompleteError

    source = (
        "def entry(p):\n"
        "    with open(p, 'wb') as f:\n"
        "        n = f.write(b'ab')\n"
        "    return n\n"
    )
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(
        SSALoweringIncompleteError
    ) as refused:
        lower_ast_source_to_ssa(
            source, "entry", name="file_result",
            extraction_contract=ExtractionContract(CONTRACT),
            progress=lambda _message: None, progress_bars=False,
        )

    message = str(refused.value)
    assert "shell-file-result" in message
    assert "file-broker operation 'write'" in message
    # The callsite is part of the diagnostic contract (review 87abc6dc,
    # comments 544 and 546): dropping it from the message must fail here.
    assert re.search(r"file-broker operation 'write' at callsite \d+", message)
    assert "returns none" not in message


def test_broker_result_consumed_by_arithmetic_is_not_a_shell_file_result():
    # Control for the guard above (review 87abc6dc, comments 544 and 546):
    # the shell-file-result rule is keyed to source outputs that are the
    # broker call's own value.  A broker result consumed by arithmetic is not
    # such an output and must not be classified as one.  It is refused
    # instead as consumed-file-broker-result, naming the operation, callsite
    # and consumer (ticket b043a7c9, case 6b7d429f).
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    source = (
        "def entry(p):\n"
        "    with open(p, 'wb') as f:\n"
        "        n: int = f.write(b'ab')\n"
        "    return n + 1\n"
    )
    with pytest.raises(Exception) as refused:
        with contextlib.redirect_stdout(io.StringIO()):
            lower_ast_source_to_ssa(
                source, "entry", name="file_result_arithmetic",
                extraction_contract=ExtractionContract(CONTRACT),
                progress=lambda _message: None, progress_bars=False,
            )
    message = str(refused.value)
    assert "shell-file-result" not in message
    # Job e74ff1de: final concordance refused this program on "value 8:
    # dtype=int64 accounting_keys=[...]" without naming the broker or the
    # consumer.  The named shortfall now comes first.
    assert "consumed-file-broker-result" in message
    assert re.search(
        r"file-broker operation 'write' at callsite \d+ "
        r"is consumed by Add node \d+ as lhs",
        message,
    )
    assert "final concordance" not in message


def test_effect_only_file_broker_write_is_not_a_consumed_result():
    # Control for the consumed-file-broker-result rule: a write whose value
    # nothing reads, and the write reading open's handle, stay inside the
    # broker plan and must not be classified as consumed.  Job ecd255a6
    # showed this fixture lowers at c4e06679, so the control requires
    # successful lowering (review d2344047, comments 563 and 565): any
    # refusal, whatever it names, fails here.
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    source = (
        "def entry(p):\n"
        "    with open(p, 'wb') as f:\n"
        "        f.write(b'ab')\n"
        "    return 1\n"
    )
    with contextlib.redirect_stdout(io.StringIO()):
        lowered = lower_ast_source_to_ssa(
            source, "entry", name="file_write_effect_only",
            extraction_contract=ExtractionContract(CONTRACT),
            progress=lambda _message: None, progress_bars=False,
        )
    assert lowered is not None


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
        "native_abi": "cpython-c-api",
        "runtime_owner": "shell",
        "shell_profiles": ["python", "cpython-c"],
        "result_dtype": "opaque_ref",
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


def test_with_sources_admits_one_named_file_and_nothing_beside_it(
    tmp_path, monkeypatch,
):
    """A source file under no declared root is ``unknown`` and rejected.

    The sheet's roots are relative to the sheet, so a program that sits
    BESIDE the repository (engine_toy beside turing) has every class it
    imports from its own directory rejected as ``provenance_not_declared``.
    ``with_sources`` admits exactly the files it names, by module name AND
    path: a sibling in the same directory stays unknown, the same file
    under another module name stays unknown, and the declaration survives
    the other builders in either order.
    """
    import importlib

    for name in ("beside_module", "sibling_module"):
        (tmp_path / f"{name}.py").write_text(
            "class Beside:\n    def step(self, dt):\n        return dt\n",
            encoding="utf-8",
        )
    monkeypatch.syspath_prepend(str(tmp_path))
    beside = importlib.import_module("beside_module")
    sibling = importlib.import_module("sibling_module")

    base = ExtractionContract(CONTRACT)
    undeclared = base.decide(beside.Beside.step)
    assert undeclared.subject.classification == "unknown"
    assert undeclared.action is ExtractionAction.REJECT
    assert undeclared.parameters["reason"] == "provenance_not_declared"

    declared = base.with_sources([
        ("beside_module", tmp_path / "beside_module.py"),
    ])
    admitted = declared.decide(beside.Beside.step)
    assert admitted.subject.classification == "authored_python"
    assert admitted.action is ExtractionAction.INGEST_PYTHON
    assert admitted.ingest_parent
    assert declared.decide(sibling.Beside.step).action is ExtractionAction.REJECT
    assert declared.sources_receipt() == [
        {"name": "beside_module", "path": str((tmp_path / "beside_module.py").resolve())},
    ]
    assert declared.fingerprint != base.fingerprint

    # the same file under a different module name is not the declaration
    misnamed = base.with_sources([("other_name", tmp_path / "beside_module.py")])
    assert misnamed.decide(beside.Beside.step).action is ExtractionAction.REJECT

    # builders compose in either order without losing the declaration
    chained = declared.with_program_abi(declared.program_abi.receipt())
    assert chained.decide(beside.Beside.step).ingest_parent
    reversed_order = base.with_program_abi(
        base.program_abi.receipt()
    ).with_sources([("beside_module", tmp_path / "beside_module.py")])
    assert reversed_order.decide(beside.Beside.step).ingest_parent
    assert chained.execution.host_runtime == base.execution.host_runtime

    with pytest.raises(ExtractionContractError):
        base.with_sources([("beside_module", tmp_path / "beside_module.txt")])
    with pytest.raises(ExtractionContractError):
        base.with_sources([
            ("beside_module", tmp_path / "beside_module.py"),
            ("renamed", tmp_path / "beside_module.py"),
        ])
