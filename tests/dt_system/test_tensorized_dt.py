"""The shared dt ABI carries values and presence, with a pinned column order."""

from pathlib import Path

import numpy as np
import pytest

from src.common.dt_system.dt_controller import Targets, _propose_dt_pen
from src.common.dt_system.dt_scaler import Metrics, coerce_metrics
from src.common.dt_system.error_channels import (
    DT_CHANNEL_NAMES, channel_fields, declare_channel,
)
from src.common.tensors import AbstractTensor


@pytest.mark.fast
def test_canonical_metrics_coercion_preserves_tensorized_record():
    metric = AbstractTensor.tensor(0.51)
    metrics = Metrics(metric, metric, metric, metric)

    assert coerce_metrics(metrics) is metrics
    assert metrics.max_vel is metric


@pytest.mark.fast
def test_dt_columns_do_not_depend_on_process_declaration_order(monkeypatch):
    from src.common.dt_system import error_channels as ec
    monkeypatch.setattr(ec, "_CHANNEL_NAMES", [])
    monkeypatch.setattr(ec, "_CHANNEL_BY_NAME", {})
    declare_channel("unrelated_plugin_column")
    declare_channel("power_w")
    data = channel_fields({"energy_j": 3.0, "power_w": 0.0})
    assert data["error_channels"].tolist()[:3] == [3.0, 0.0, 0.0]
    assert data["error_present"].tolist()[:3] == [1.0, 1.0, 0.0]
    assert len(data["error_channels"].tolist()) == len(DT_CHANNEL_NAMES)
    with pytest.raises(ValueError, match="undeclared"):
        channel_fields({"unrelated_plugin_column": 1.0})


@pytest.mark.fast
def test_unpublished_nan_and_zero_limit_do_not_become_measurements():
    metrics = Metrics(1.0, 0.0, 0.0, 0.0)
    targets = Targets(1.0, 1.0, 1.0, **channel_fields(
        {"height_positivity": 0.0}, limits=True))
    metrics.error_channels[5] = float("nan")
    assert _propose_dt_pen(metrics, targets, 1.0, None) == 1.0
    metrics.error_channels[5] = 2e-30
    metrics.error_present[5] = 1.0
    assert _propose_dt_pen(metrics, targets, 1.0, None) == 0.5


def test_real_energy_helper_lowers_without_keyed_arenas(tmp_path):
    import pickle
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )

    source = (
        "from src.common.dt_system.dt_controller import _energy_time_limit\n"
        "def root(metrics, targets):\n"
        "    limit, present = _energy_time_limit(metrics, targets)\n"
        "    return limit if present else -1.0\n"
    )
    reference = c_backend_repository_ssa_reference()
    reference_before = pickle.dumps(reference.module)
    module, _, _ = lower_ast_source_to_ssa(
        source, "root", name="tensorized_energy",
        extraction_contract=ExtractionContract(Path("extraction_contracts/program_extraction.yaml")),
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=reference,
    )
    assert pickle.dumps(reference.module) == reference_before
    for function in module.functions.values():
        for block in function.blocks.values():
            assert not any(i.attributes.get("ssa_sequence_operation") in
                           ("table_load", "table_store", "contains") for i in block.instrs)
    from src.compiler.identity_concordance import concordance_report
    print(concordance_report(module))
    artifact = emit_ssa_module_to_c(module, "tensorized_energy__root")
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path, optimization="O0")
    root = module.functions["tensorized_energy__root"]
    from src.compiler.vehicle_python_compilation import _managed_native_feeds_by_id
    from types import SimpleNamespace
    interface = SimpleNamespace(module=module, root_name=root.name)
    metrics = Metrics(0.0, 0.0, 0.0, 0.0, **channel_fields({"energy_j": 12.0, "power_w": 3.0}))
    targets = Targets(1.0, 1.0, 1.0, energy_exchange_fraction=0.25)
    output_id = int(root.metadata["named_outputs"][0][1])
    for present, energy, power, fraction, expected in (
        (1.0, 12.0, 3.0, 0.25, 1.0),
        (0.0, 12.0, 3.0, 0.25, -1.0),
        (1.0, 12.0, 0.0, 0.25, -1.0),
        (1.0, float("nan"), 3.0, 0.25, -1.0),
        (1.0, 12.0, 3.0, None, -1.0),
    ):
        metrics.error_present[1] = present
        metrics.error_channels[0] = energy
        metrics.error_channels[1] = power
        targets.energy_exchange_fraction = fraction
        feeds = _managed_native_feeds_by_id(interface, {"metrics": metrics, "targets": targets})
        execution = artifact.prepare_execution(feeds)
        execution.run()
        actual = np.asarray(execution.buffers[output_id]).reshape(-1)[0]
        assert actual == expected


def test_native_publication_uses_declared_participant_channel_extent(tmp_path):
    import sys
    from types import SimpleNamespace
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
    from llvm_dt_system import dt_system_contract
    from src.common.dt_system.participants import StepSpans
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c
    from src.compiler.vehicle_python_compilation import _managed_native_feeds_by_id
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    source = (
        "from src.common.dt_system.dt_scaler import Metrics, coerce_metrics\n"
        "def root(state):\n"
        "    metrics = Metrics(0.0, 0.0, 0.0, 0.0, "
        + ", ".join(f"{field}=state.{field}" for field in (
            "pub_tau", "pub_tau_present", "pub_contract", "pub_dt_limit", "pub_dt_limit_present",
            "pub_values", "pub_present", "pub_limits", "pub_limits_present"))
        + ")\n"
        "    metrics = coerce_metrics(metrics)\n"
        "    metrics.pub_values[29] = metrics.pub_values[29] + 2.0\n"
        "    metrics.pub_present[29] = 1.0\n"
        "    return metrics.pub_values[29]\n"
    )
    module, _, _ = lower_ast_source_to_ssa(
        source, "root", name="publication_extent",
        extraction_contract=dt_system_contract("root", (), 1, 2),
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
    )
    root = module.functions["publication_extent__root"]
    from src.compiler.identity_concordance import concordance_report
    print(concordance_report(module))
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path, optimization="O0")
    spans = StepSpans(*[AbstractTensor.zeros((2,)) for _ in range(5)],
                      *[AbstractTensor.zeros((30,)) for _ in range(4)],
                      channel_names=DT_CHANNEL_NAMES)
    spans.pub_values[29] = 7.0
    feeds = _managed_native_feeds_by_id(
        SimpleNamespace(module=module, root_name=root.name),
        {"state": spans},
    )
    execution = artifact.prepare_execution(feeds)
    output_id = int(root.metadata["named_outputs"][0][1])
    for expected in (9.0, 11.0):
        execution.run()
        assert np.asarray(execution.buffers[output_id]).reshape(-1)[0] == expected
    for argument in root.args:
        if (argument.accounting or {}).get("program_abi_field") == "pub_values":
            values = np.asarray(execution.buffers[int(argument.id)])
            assert values.size == 30
            assert np.count_nonzero(values) == 1


def test_coerced_metrics_uses_declared_publication_extent():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
    from llvm_dt_system import dt_system_contract
    from src.compiler.fortran_c_shell import (
        _undefined_repository_ssa_operands,
        lower_ast_source_to_ssa,
    )
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )

    source = (
        "from src.common.dt_system.dt_scaler import coerce_metrics\n"
        "def root(metrics):\n"
        "    metrics = coerce_metrics(metrics)\n"
        "    has_participants = int(metrics.pub_tau.shape[0]) > 0\n"
        "    values = (metrics.error_channels if not has_participants "
        "else metrics.pub_values)\n"
        "    return values[0]\n"
    )
    module, _, _ = lower_ast_source_to_ssa(
        source, "root", name="coerced_publication_extent",
        extraction_contract=dt_system_contract("root", (), 1, 2),
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
    )
    root = module.functions["coerced_publication_extent__root"]
    artifact = emit_ssa_module_to_c(module, root.name)

    assert not _undefined_repository_ssa_operands(module)
    assert artifact.complete, artifact.shortfalls
    assert not any(
        int(argument.id) in {value_id for _name, value_id in root.metadata.get(
            "value_names", ()
        ) if _name == "has_participants"}
        for argument in root.args
        if not (argument.accounting or {})
    )


@pytest.mark.parametrize("participants", [0, 2])
def test_real_proposal_consumes_channel_spans_natively(tmp_path, participants):
    from types import SimpleNamespace
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c
    from src.compiler.vehicle_python_compilation import _managed_native_feeds_by_id
    from src.compiler.identity_concordance import concordance_report
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    source = (
        "from src.common.dt_system.dt_controller import _propose_dt_pen\n"
        "def root(metrics, targets):\n"
        "    return _propose_dt_pen(metrics, targets, 1.0, None)\n"
    )
    contract = ExtractionContract("extraction_contracts/program_extraction.yaml")
    if participants:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
        from llvm_dt_system import dt_system_contract
        contract = dt_system_contract("root", (), 1, participants)
    module, _, _ = lower_ast_source_to_ssa(
        source, "root", name="tensorized_proposal",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
    )
    print(concordance_report(module))
    root = module.functions["tensorized_proposal__root"]
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path, optimization="O0")
    metrics = Metrics(2.0, 0.0, 0.0, 0.0)
    targets = Targets(1.0, 1.0, 1.0, **channel_fields({"height_positivity": 2.0}, limits=True))
    if participants:
        for field in ("pub_tau", "pub_tau_present", "pub_contract", "pub_dt_limit", "pub_dt_limit_present"):
            setattr(metrics, field, AbstractTensor.zeros((participants,)))
        for field in ("pub_values", "pub_present", "pub_limits", "pub_limits_present"):
            setattr(metrics, field, AbstractTensor.zeros((participants * len(DT_CHANNEL_NAMES),)))
        metrics.pub_limits[-1] = 2.0
        metrics.pub_limits_present[-1] = 1.0
    output_id = int(root.metadata["named_outputs"][0][1])
    for present, measure, expected in ((0.0, 4.0, 0.5), (1.0, 4.0, 0.25), (1.0, 0.0, 0.5)):
        if participants:
            metrics.pub_values[-1] = measure
            metrics.pub_present[-1] = present
        else:
            metrics.error_channels[5] = measure
            metrics.error_present[5] = present
        feeds = _managed_native_feeds_by_id(
            SimpleNamespace(module=module, root_name=root.name),
            {"metrics": metrics, "targets": targets},
        )
        execution = artifact.prepare_execution(feeds)
        execution.run()
        assert np.asarray(execution.buffers[output_id]).reshape(-1)[0] == expected
