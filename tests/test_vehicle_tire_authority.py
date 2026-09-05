from __future__ import annotations

import json

import pytest

from src.compiler.vehicle_tire_authority import (
    build_tire_authority_definition,
    write_native_tire_authority,
)
from src.compiler.vehicle_tire_local_acceleration_network import TireLocalNetworkSpec


def test_authority_resolves_one_topology_state_and_operator_abi():
    authority = build_tire_authority_definition()
    manifest = authority.manifest
    topology = manifest["topology"]
    assert manifest["identity"] == "compiled-balloon-skin-v1"
    assert topology["vertex_count"] == 128
    assert topology["face_count"] == 256
    assert len(manifest["state"]["per_wheel"]) == 6 * topology["vertex_count"]
    assert manifest["network"]["spec"]["batch_size"] == 4
    assert manifest["network"]["output_order"] == [
        "local_acceleration_x_m_s2", "local_acceleration_y_m_s2",
        "local_acceleration_z_m_s2",
    ]
    assert "terrain-geometry" in manifest["network"]["excluded_inputs"]
    assert manifest["physics"]["hub_wrench"].startswith("emergent")
    assert manifest["physics"]["rest_torus_runtime_authority"] is False
    assert len(authority.digest) == 64


def test_authority_rejects_a_network_field_that_is_not_the_skin():
    with pytest.raises(ValueError, match="exactly match"):
        build_tire_authority_definition(network_spec=TireLocalNetworkSpec(
            circumferential_segments=8,
            section_segments=8,
            batch_size=4,
        ))


def test_native_authority_writer_uses_resolved_paths_and_records_every_kernel(tmp_path):
    written = write_native_tire_authority(tmp_path / "relative-safe")
    assert written.library_path.is_file()
    manifest = json.loads(written.manifest_path.read_text(encoding="utf-8"))
    assert manifest["authority_digest"] == written.definition.digest
    assert set(manifest["native"]["abi"]) == set(manifest["kernels"].values())
    assert all(path.is_file() for path in written.source_paths)
