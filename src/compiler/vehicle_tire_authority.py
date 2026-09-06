"""One authoritative tyre definition shared by rig, trainer, and deployment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .abstract_ui_vehicles import WHEEL_NAMES, load_default_car_configuration
from .vehicle_balloon_tire import balloon_tire_graph_abi
from .vehicle_tire_local_acceleration_network import (
    TIRE_LOCAL_ACCELERATION_NAMES,
    TireLocalNetworkSpec,
    tire_local_feature_names,
    tire_local_feature_scales,
)
from .vehicle_tire_force_workshare import TireForceWorkShareConfig
from .graph_appendage_replacement import GraphAppendageReplacementContract
from .vehicle_balloon_tire_program import balloon_tire_python_program
from .vehicle_python_compilation import emit_balloon_tire_python_c


@dataclass(frozen=True, slots=True)
class TireAuthorityDefinition:
    manifest: Mapping[str, Any]
    digest: str


@dataclass(frozen=True, slots=True)
class WrittenNativeTireAuthority:
    destination: Path
    library_path: Path
    manifest_path: Path
    source_paths: tuple[Path, ...]
    definition: TireAuthorityDefinition


def open_native_tire_authority(
    destination: str | Path,
    *,
    definition: TireAuthorityDefinition | None = None,
) -> WrittenNativeTireAuthority:
    """Open an existing bundle only when its content identity still matches."""

    authority = definition or build_tire_authority_definition()
    output = Path(destination).resolve()
    manifest_path = output / "vehicle_tire_authority.json"
    library = output / "vehicle_tire_authority.dll"
    if not manifest_path.is_file() or not library.is_file():
        raise FileNotFoundError("native tire authority bundle is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("authority_digest") != authority.digest:
        raise ValueError("native tire authority bundle is stale for this definition")
    source_paths = tuple(
        output / f"{name}.c" for name in manifest["native"]["abi"]
    ) + (output / f"{manifest['native']['appendage']['step']}.c",)
    if not all(path.is_file() for path in source_paths):
        raise FileNotFoundError("native tire authority source inventory is incomplete")
    return WrittenNativeTireAuthority(
        destination=output,
        library_path=library,
        manifest_path=manifest_path,
        source_paths=source_paths,
        definition=authority,
    )


def build_tire_authority_definition(
    *,
    vehicle_config=None,
    network_spec: TireLocalNetworkSpec | None = None,
    workshare_config: TireForceWorkShareConfig = TireForceWorkShareConfig(),
) -> TireAuthorityDefinition:
    """Resolve every tyre consumer onto one content-addressed ABI."""

    config = vehicle_config or load_default_car_configuration()
    source = config.source if hasattr(config, "source") else vehicle_config
    graph_abi = balloon_tire_graph_abi(source)
    topology = graph_abi["topology"]
    network = network_spec or TireLocalNetworkSpec(
        circumferential_segments=topology.circumferential_segments,
        section_segments=topology.section_segments,
        batch_size=len(WHEEL_NAMES),
    )
    if (
        network.circumferential_segments != topology.circumferential_segments
        or network.section_segments != topology.section_segments
    ):
        raise ValueError("network UV field must exactly match the authoritative skin topology")
    if network.batch_size != len(WHEEL_NAMES):
        raise ValueError("live vehicle tyre operator batch must contain all four wheel corners")

    replacement = GraphAppendageReplacementContract(
        identity=str(source["tire_skin"]["model"]),
        exact_entrypoint="balloon_tire_appendage_step",
        input_names=tuple(tire_local_feature_names(network)),
        state_names=tuple(graph_abi["state"]),
        output_names=tuple(TIRE_LOCAL_ACCELERATION_NAMES),
        learned_output_names=tuple(TIRE_LOCAL_ACCELERATION_NAMES),
        novelty_channels=("plastic_activity", "contact_novelty", "thermodynamic_novelty"),
        learned_state_names=tuple(graph_abi["state"]),
        permits_live_duty=True,
    )

    manifest = {
        "schema": "turing.vehicle-tire-authority.v1",
        "identity": str(source["tire_skin"]["model"]),
        "vehicle_config_digest": getattr(config, "digest", None),
        "wheel_order": list(WHEEL_NAMES),
        "coordinate_system": "hub-local-x-forward-y-up-z-right",
        "topology": {
            "circumferential_segments": topology.circumferential_segments,
            "section_segments": topology.section_segments,
            "vertex_count": len(topology.rest_positions),
            "face_count": len(topology.faces),
            "edge_count": len(topology.edges),
            "bead_rings": [list(ring) for ring in topology.bead_rings],
            "reference_volume_m3": topology.reference_volume_m3,
            "rest_positions": [list(row) for row in topology.rest_positions],
            "faces": [list(row) for row in topology.faces],
            "face_rest_data": [list(row) for row in topology.face_rest_data],
        },
        "state": {
            "per_wheel": list(graph_abi["state"]),
            "scalar_count_per_wheel": len(graph_abi["state"]),
            "integration_order": "position-then-velocity-by-vertex-index-and-xyz",
            "derived_thermodynamic_outputs": [
                "gas_pressure_pa", "volume_ratio", "gas_temperature_k",
            ],
            "thermodynamic_closure": "polytropic-from-closed-skin-volume",
        },
        "runtime_parameters": dict(graph_abi["parameters"]),
        "physics": {
            "material": "StVK-membrane-plus-Kelvin-strain-rate-loss",
            "gas": "closed-volume-polytropic",
            "reference_shape": "inflated-construction-state",
            "construction_prestress": (
                "conservative-reference-face-load-cancels-reference-gas-pressure"
            ),
            "reference_quiescence": "zero-face-force-at-reference-pressure-and-state",
            "bead": "two-ring-equal-opposite-rim-wrench-backward-euler-junction",
            "membrane_time_discretization": "explicit-force-with-passivity-limited-Kelvin-impulse",
            "contact": "deformed-skin-vertex-triangle-CCD-active-set-impulse",
            "rest_torus_runtime_authority": False,
            "teacher_output": "per-vertex-hub-local-acceleration",
            "hub_wrench": "emergent-equal-opposite-bead-reaction-after-integration",
        },
        "network": {
            "spec": asdict(network),
            "feature_order": list(tire_local_feature_names(network)),
            "feature_scales": tire_local_feature_scales(network).tolist(),
            "output_order": list(TIRE_LOCAL_ACCELERATION_NAMES),
            "output_shape": list(network.output_shape),
            "output_scale": network.output_scale.reshape(-1).tolist(),
            "excluded_inputs": ["terrain-geometry", "road-history", "chassis-state", "engine-state"],
            "surface_free_mode": "zero-external-boundary-force-field",
            "deployment_device": "gpu-only",
            "cpu_native_role": "parity-oracle-and-fallback-not-performance-candidate",
            "candidates": [
                "single-augmented-full-state-linear-gemm",
                "two-layer-periodic-spatial-convolution",
            ],
            "selection": "held-out-reference-loss-subject-to-passivity-and-novelty-gates",
        },
        "workshare": {
            "alpha_meaning": "scientific-reference-local-acceleration-fraction-and-duty-budget",
            "initial_alpha": 1.0,
            "config": asdict(workshare_config),
            "plastic_contact_and_thermodynamic_novelty_override": "immediate-exact",
        },
        "appendage_replacement": {
            **replacement.as_manifest(),
            "contract_digest": replacement.digest,
            "duty_visualization": "sampled-hub-bead-attachment-pulse-after-local-field-integration",
        },
        "multiplayer": {
            "server_authority": "compiled-balloon-teacher-not-learned-operator",
            "client_prediction": "gpu-learned-local-acceleration-with-authoritative-skin-state-correction",
            "reconciliation_boundary": "ordered-hub-local-skin-position-and-velocity",
            "checkpoint_state": "ordered-skin-position-and-velocity-plus-wheel-hub-state",
            "fixed_step_required": True,
            "configuration_identity": "authority-digest-plus-network-parameter-digest",
            "network_parameter_transport": "json-with-explicit-dtype-shape-and-content-hash",
            "bitwise_cross-platform_claim": False,
            "current_determinism": "fixed-order-fp-contract-off-tolerance-checked",
            "bitwise_upgrade": "reproducible-transcendental-core-or-quantized-authority",
        },
        "kernels": {
            "membrane_face": "balloon_tire_membrane_face",
            "gas": "balloon_tire_gas",
            "bead": "balloon_tire_bead_implicit_step",
            "contact_geometry": "balloon_tire_contact_geometry",
            "cylinder_contact_geometry": "balloon_tire_cylinder_contact_geometry",
            "contact_impulse": "balloon_tire_contact_impulse",
            "workshare": "tire_force_reference_workshare",
        },
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return TireAuthorityDefinition(manifest=manifest, digest=digest)


def write_native_tire_authority(
    destination: str | Path,
    *,
    definition: TireAuthorityDefinition | None = None,
) -> WrittenNativeTireAuthority:
    """Emit and compile the vectorized Python/AbstractTensor tyre authority."""

    authority = definition or build_tire_authority_definition()
    # Native compiler invocation changes cwd so every path in the command must
    # remain unambiguous.  Resolve once here; a relative build destination must
    # not accidentally become ``destination/destination``.
    output = Path(destination).resolve()
    output.mkdir(parents=True, exist_ok=True)
    program = balloon_tire_python_program()
    python_path = output / "balloon_tire.abstract_tensor.py"
    python_path.write_text(program.source, encoding="utf-8")
    artifact = emit_balloon_tire_python_c()
    c_path = output / f"{artifact.name}.c"
    c_path.write_text(artifact.source, encoding="utf-8")
    artifact.compile(output)
    library = artifact.library_path
    if library is None or not library.is_file():
        raise RuntimeError("compiler-owned tyre authority library was not produced")
    manifest = {
        **authority.manifest,
        "authority_digest": authority.digest,
        "native": {
            "library": library.name,
            "source_language": "python",
            "pipeline": ["python-ast", "ProcessGraph", "repository-SSA", "C"],
            "vehicle_specific_c_input": False,
            "appendage": {
                "step": artifact.name,
                "inputs": list(program.input_names),
                "outputs": list(program.output_names),
                "state_scalar_count": program.state_scalar_count,
                "vertex_count": program.vertex_count,
                "face_count": program.face_count,
                "python_source_sha256": hashlib.sha256(
                    program.source.encode("utf-8")
                ).hexdigest(),
                "source_sha256": hashlib.sha256(
                    artifact.source.encode("utf-8")
                ).hexdigest(),
            },
        },
    }
    manifest_path = output / "vehicle_tire_authority.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return WrittenNativeTireAuthority(
        destination=output,
        library_path=library,
        manifest_path=manifest_path,
        source_paths=(python_path, c_path),
        definition=authority,
    )
