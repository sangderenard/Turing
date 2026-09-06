from __future__ import annotations

import pytest

from src.compiler.graph_appendage_replacement import (
    AppendageDutyTelemetry,
    GraphAppendageReplacementContract,
    validate_parameter_artifact,
)


def test_appendage_contract_is_content_addressed_and_keeps_parameters_external():
    contract = GraphAppendageReplacementContract(
        identity="probe", exact_entrypoint="probe_exact",
        input_names=("x",), state_names=("q",),
        output_names=("force", "temperature"),
        learned_output_names=("force",), novelty_channels=("contact",),
    )
    manifest = contract.as_manifest()
    assert manifest["invariants"]["replacement_boundary_is_graph_cut_not_host_wrapper"]
    assert manifest["parameter_transport"] == "json-dtype-shape-content-hash"
    assert len(contract.digest) == 64


def test_appendage_parameter_json_requires_dtype_shape_and_content_hash():
    digest = validate_parameter_artifact({"arrays": {"weight": {
        "dtype": "float32", "shape": [3, 2], "sha256": "0" * 64,
    }}}, required_arrays=("weight",))
    assert len(digest) == 64
    with pytest.raises(ValueError):
        validate_parameter_artifact({"arrays": {}}, required_arrays=("weight",))


def test_duty_telemetry_exposes_sampled_attachment_activity_without_physics_state():
    telemetry = AppendageDutyTelemetry(True, True, 0.1, 0.25)
    assert telemetry.sampled_attachment_pulse(12) > 0.0
    assert telemetry.learned_active


def test_stateful_appendage_cannot_enter_live_duty_with_force_only_candidate():
    with pytest.raises(ValueError, match="complete next-state"):
        GraphAppendageReplacementContract(
            identity="stateful", exact_entrypoint="exact",
            input_names=("input",), state_names=("position", "velocity"),
            output_names=("force",), learned_output_names=("force",),
            novelty_channels=("energy",), permits_live_duty=True,
        )
