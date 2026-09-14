"""Compile an AbstractTensor perforated network to LLVM with a named ABI.

The reverse artifact is acquired through the ProcessGraph backward generator;
it never traces a tape backward.  The SSA-authored forward is first isolated
as a semantic ProcessGraph, then fused with its generated VJP for lowering to
the repository SSA and LLVM backends.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from ..common.tensors.abstract_nn.perforated import PerforatedLinear
from ..common.tensors.accelerator_backends.ssa_backend import (
    SSATensorOperations,
    SSATensorProgram,
)
from .llvm_training_runtime import (
    NativeGraphForward,
    NativeGraphReverse,
    compile_native_graph_forward,
    compile_native_graph_reverse,
)
from .ssa_llvm_backend import LLVMFunctionArtifact


@dataclass(frozen=True)
class TensorPort:
    name: str
    value_id: int
    shape: tuple[int, ...]
    role: str
    dtype: str = "float64"
    layout: str = "row-major-contiguous"

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value_id": self.value_id,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "layout": self.layout,
            "role": self.role,
        }


@dataclass(frozen=True)
class PerforatedLLVMContract:
    inputs: tuple[TensorPort, ...]
    prediction: TensorPort
    prediction_adjoint: TensorPort
    gradients: tuple[TensorPort, ...]
    dendrites_per_neuron: int

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "turing.perforated-network-llvm",
            "version": 2,
            "execution": {
                "forward_source": "isolated-abstract-tensor-ssa",
                "backward_source": "process-graph-generator",
                "backward_packaging": "combined-forward-vjp",
                "tape_autograd": False,
            },
            "network": {
                "kind": "perforated-linear",
                "dendrites_per_neuron": self.dendrites_per_neuron,
                "equation": (
                    "authority*(x@base_weight+base_bias+(dendrite_mask*"
                    "dendrite_gain*tanh(x@dendrite_weight+dendrite_bias))@"
                    "dendrite_route)+(1-authority)*simulator_delta"
                ),
                "runtime_topology_gating": True,
                "simulator_fallback": True,
            },
            "inputs": [port.manifest() for port in self.inputs],
            "prediction": self.prediction.manifest(),
            "prediction_adjoint": self.prediction_adjoint.manifest(),
            "gradients": [port.manifest() for port in self.gradients],
        }


@dataclass(frozen=True)
class CompiledPerforatedNetwork:
    contract: PerforatedLLVMContract
    forward: NativeGraphForward
    forward_vjp: NativeGraphReverse
    manifest_path: Path


def _compiler_fingerprint() -> str:
    """Invalidate native artifacts when their tensor/compiler sources change."""
    src_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    roots = (src_root / "compiler", src_root / "common" / "tensors")
    for source_path in sorted(
            path for root in roots for path in root.rglob("*.py")):
        stat = source_path.stat()
        digest.update(
            f"{source_path.relative_to(src_root)}:{stat.st_size}:"
            f"{stat.st_mtime_ns}".encode("utf-8"))
    return digest.hexdigest()


def _artifact_manifest(native) -> dict[str, Any]:
    artifact = native.artifact
    return {
        "entry": artifact.name,
        "library": str(Path(artifact.library_path).resolve()),
        "buffer_order": list(artifact.buffer_order),
        "buffer_shapes": [list(shape) for shape in artifact.buffer_shapes],
        "buffer_dtypes": list(artifact.buffer_dtypes),
        "extent_order": [list(item) for item in artifact.extent_order],
    }


def _load_cached_network(directory: Path, *, batch: int, in_dim: int,
                         out_dim: int, branches: int
                         ) -> CompiledPerforatedNetwork | None:
    manifest_path = directory / "contract.json"
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (payload.get("schema") != "turing.perforated-network-llvm"
                or payload.get("version") != 2
                or payload.get("build_fingerprint") != _compiler_fingerprint()
                or payload.get("network", {}).get("dendrites_per_neuron") != branches):
            return None
        inputs = tuple(TensorPort(
            port["name"], int(port["value_id"]), tuple(port["shape"]),
            port["role"], port.get("dtype", "float64"),
            port.get("layout", "row-major-contiguous"))
            for port in payload["inputs"])
        ports = {port.name: port for port in inputs}
        if (ports["x"].shape != (batch, in_dim)
                or ports["base_weight"].shape != (in_dim, out_dim)):
            return None
        prediction = TensorPort(**{
            **payload["prediction"], "shape": tuple(payload["prediction"]["shape"])
        })
        prediction_adjoint = TensorPort(**{
            **payload["prediction_adjoint"],
            "shape": tuple(payload["prediction_adjoint"]["shape"])
        })
        gradients = tuple(TensorPort(**{**port, "shape": tuple(port["shape"])})
                          for port in payload["gradients"])

        def load_artifact(record):
            library = Path(record["library"])
            if not library.is_file() or "buffer_shapes" not in record:
                raise FileNotFoundError(library)
            return LLVMFunctionArtifact(
                name=record["entry"], llvm_ir="",
                buffer_order=tuple(map(int, record["buffer_order"])),
                buffer_shapes=tuple(tuple(shape)
                                    for shape in record["buffer_shapes"]),
                extent_order=tuple(tuple(item)
                                   for item in record.get("extent_order", ())),
                shortfalls=(),
                buffer_dtypes=tuple(record["buffer_dtypes"]),
                library_path=library.resolve())

        forward_artifact = load_artifact(payload["artifacts"]["forward"])
        reverse_artifact = load_artifact(payload["artifacts"]["forward_vjp"])
        input_ids = {port.name: port.value_id for port in inputs}
        gradient_ids = {
            ports[gradient.name.removeprefix("grad_")].value_id: gradient.value_id
            for gradient in gradients
        }
        forward = NativeGraphForward(
            payload["artifacts"]["forward"]["entry"], "cached", "cached",
            input_ids, (prediction.value_id,), forward_artifact)
        reverse = NativeGraphReverse(
            payload["artifacts"]["forward_vjp"]["entry"], input_ids,
            tuple(gradient.value_id for gradient in gradients), gradient_ids,
            {prediction.value_id: prediction_adjoint.value_id},
            reverse_artifact, 0)
        contract = PerforatedLLVMContract(
            inputs, prediction, prediction_adjoint, gradients, branches)
        return CompiledPerforatedNetwork(
            contract, forward, reverse, manifest_path.resolve())
    except (KeyError, TypeError, ValueError, OSError):
        return None


def _build_graph(batch: int, in_dim: int, out_dim: int, branches: int):
    program = SSATensorProgram("perforated_linear")
    width = out_dim * branches
    shapes = {
        "x": (batch, in_dim),
        "base_weight": (in_dim, out_dim),
        "base_bias": (1, out_dim),
        "dendrite_weight": (in_dim, width),
        "dendrite_bias": (1, width),
        "dendrite_gain": (1, width),
        "dendrite_route": (width, out_dim),
        "dendrite_mask": (1, width),
        "network_authority": (batch, out_dim),
        "simulator_delta": (batch, out_dim),
    }
    bindings = {
        name: SSATensorOperations.input(program, shape)
        for name, shape in shapes.items()
    }
    layer = PerforatedLinear(
        in_dim,
        out_dim,
        like=bindings["x"],
        dendrites_per_neuron=branches,
        active=True,
    )
    layer.base.W = bindings["base_weight"]
    layer.base.b = bindings["base_bias"]
    layer.dendrite_weight = bindings["dendrite_weight"]
    layer.dendrite_bias = bindings["dendrite_bias"]
    layer.dendrite_gain = bindings["dendrite_gain"]
    layer.dendrite_route = bindings["dendrite_route"]
    network_prediction = layer.forward(
        bindings["x"], dendrite_mask=bindings["dendrite_mask"])
    authority = bindings["network_authority"]
    prediction = (network_prediction * authority
                  + bindings["simulator_delta"] * (authority * -1.0 + 1.0))
    return prediction, bindings, shapes


def compile_perforated_network(
    directory: str | Path,
    *,
    batch: int,
    in_dim: int,
    out_dim: int,
    dendrites_per_neuron: int = 2,
    name: str = "perforated_linear",
    use_cache: bool = True,
) -> CompiledPerforatedNetwork:
    """Compile standalone forward and combined ProcessGraph forward/VJP DLLs."""
    if min(batch, in_dim, out_dim, dendrites_per_neuron) < 1:
        raise ValueError("all perforated-network dimensions must be positive")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if use_cache:
        cached = _load_cached_network(
            directory, batch=batch, in_dim=in_dim, out_dim=out_dim,
            branches=dendrites_per_neuron)
        if cached is not None:
            return cached

    prediction, bindings, shapes = _build_graph(
        batch, in_dim, out_dim, dendrites_per_neuron
    )
    forward = compile_native_graph_forward(
        prediction,
        bindings=bindings,
        source="PerforatedLinear.forward(AbstractTensor)",
        name=f"{name}__forward",
        directory=directory / "forward",
    )

    # Build again because forward compilation appends a Ret to its SSA module.
    prediction, bindings, shapes = _build_graph(
        batch, in_dim, out_dim, dendrites_per_neuron
    )
    ids = {name: int(value.data.value.id) for name, value in bindings.items()}
    prediction_id = int(prediction.data.value.id)
    parameter_names = (
        "base_weight", "base_bias", "dendrite_weight", "dendrite_bias",
        "dendrite_gain",
    )
    differentiated_ids = tuple(ids[port_name] for port_name in parameter_names)
    forward_vjp = compile_native_graph_reverse(
        prediction,
        bindings=bindings,
        wrt_value_ids=differentiated_ids,
        name=f"{name}__forward_vjp",
        directory=directory / "forward_vjp",
    )

    inputs = tuple(
        TensorPort(
            port_name,
            ids[port_name],
            tuple(shapes[port_name]),
            (
                "data" if port_name == "x"
                else "fixed-structure" if port_name == "dendrite_route"
                else "topology-mask" if port_name == "dendrite_mask"
                else "authority-mask" if port_name == "network_authority"
                else "simulator-fallback" if port_name == "simulator_delta"
                else "parameter"
            ),
        )
        for port_name in bindings
    )
    prediction_port = TensorPort(
        "prediction", prediction_id, (batch, out_dim), "output"
    )
    seed_port = TensorPort(
        "prediction_adjoint",
        int(forward_vjp.seed_value_ids[prediction_id]),
        (batch, out_dim),
        "reverse-seed",
    )
    gradients = tuple(
        TensorPort(
            f"grad_{port.name}",
            int(forward_vjp.gradient_value_ids[port.value_id]),
            port.shape,
            "gradient",
        )
        for port in inputs if port.role == "parameter"
    )
    contract = PerforatedLLVMContract(
        inputs,
        prediction_port,
        seed_port,
        gradients,
        dendrites_per_neuron,
    )
    payload = contract.manifest()
    payload["build_fingerprint"] = _compiler_fingerprint()
    payload["artifacts"] = {
        "forward": _artifact_manifest(forward),
        "forward_vjp": _artifact_manifest(forward_vjp),
    }
    manifest_path = directory / "contract.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return CompiledPerforatedNetwork(
        contract, forward, forward_vjp, manifest_path
    )


__all__ = [
    "CompiledPerforatedNetwork",
    "PerforatedLLVMContract",
    "TensorPort",
    "compile_perforated_network",
]
