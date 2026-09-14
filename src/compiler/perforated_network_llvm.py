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


@dataclass(frozen=True)
class CompiledPerforatedAdamChunk:
    """One LLVM entry containing loss, generated VJP, and an Adam cycle."""

    artifact: LLVMFunctionArtifact
    input_value_ids: dict[str, int]
    parameter_value_ids: dict[str, int]
    gradient_value_ids: dict[str, int]
    output_value_ids: dict[str, int]
    optimizer_state_value_ids: dict[str, Any]
    cycle_length: int
    manifest_path: Path
    cache_hit: bool = False


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


def _build_training_graph(batch: int, in_dim: int, out_dim: int, branches: int):
    prediction, bindings, shapes = _build_graph(
        batch, in_dim, out_dim, branches)
    program = prediction.data.program
    shapes.update({
        "target": (batch, out_dim),
        "sample_weight": (batch, 1),
        # A full-shaped scale avoids scalar-shape ambiguity at linked helper
        # boundaries while retaining exact valid-row normalization.
        "loss_scale": (batch, out_dim),
    })
    for name in ("target", "sample_weight", "loss_scale"):
        bindings[name] = SSATensorOperations.input(program, shapes[name])
    error = prediction - bindings["target"]
    loss = (
        error * error * bindings["sample_weight"] * bindings["loss_scale"]
    ).sum()
    return loss, prediction, bindings, shapes


def _load_cached_adam_chunk(
    directory: Path,
    *,
    batch: int,
    in_dim: int,
    out_dim: int,
    cycle_length: int,
    branches: int,
    gradient_accumulation_steps: int,
    max_global_gradient_norm: float | None,
) -> CompiledPerforatedAdamChunk | None:
    manifest_path = directory / "contract.json"
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        shape = payload["shape"]
        if (
            payload.get("schema") != "turing.perforated-adam-chunk-llvm"
            or payload.get("version") != 2
            or payload.get("build_fingerprint") != _compiler_fingerprint()
            or shape != {
                "batch": batch,
                "cycle_length": cycle_length,
                "dendrites_per_neuron": branches,
                "input": in_dim,
                "output": out_dim,
            }
            or payload.get("optimizer_options") != {
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "max_global_gradient_norm": max_global_gradient_norm,
            }
        ):
            return None
        record = payload["artifact"]
        library = Path(record["library"])
        if not library.is_file():
            return None
        state = dict(payload["optimizer_state"])
        state["first_moment"] = {
            int(key): int(value) for key, value in state["first_moment"].items()
        }
        state["second_moment"] = {
            int(key): int(value) for key, value in state["second_moment"].items()
        }
        state["gradient_accumulator"] = {
            int(key): int(value)
            for key, value in state["gradient_accumulator"].items()
        }
        state["cycled_value_ids"] = tuple(map(int, state["cycled_value_ids"]))
        for key in (
            "steps", "learning_rate", "beta1", "beta2", "epsilon",
            "beta1_power", "beta2_power", "iteration", "cycle_length",
            "gradient_norm", "clipped_gradient_norm",
        ):
            state[key] = int(state[key])
        state["gradient_accumulation_steps"] = int(
            state["gradient_accumulation_steps"])
        artifact = LLVMFunctionArtifact(
            name=str(record["entry"]),
            llvm_ir="",
            buffer_order=tuple(map(int, record["buffer_order"])),
            buffer_shapes=tuple(tuple(shape) for shape in record["buffer_shapes"]),
            extent_order=tuple(tuple(item) for item in record.get("extent_order", ())),
            shortfalls=(),
            buffer_dtypes=tuple(map(str, record["buffer_dtypes"])),
            library_path=library.resolve(),
            training_steps_value_id=int(state["steps"]),
            learning_rate_value_id=int(state["learning_rate"]),
            optimizer_state_value_ids=state,
        )
        return CompiledPerforatedAdamChunk(
            artifact=artifact,
            input_value_ids={
                str(key): int(value) for key, value in payload["inputs"].items()
            },
            parameter_value_ids={
                str(key): int(value) for key, value in payload["parameters"].items()
            },
            gradient_value_ids={
                str(key): int(value) for key, value in payload["gradients"].items()
            },
            output_value_ids={
                str(key): int(value) for key, value in payload["outputs"].items()
            },
            optimizer_state_value_ids=state,
            cycle_length=cycle_length,
            manifest_path=manifest_path.resolve(),
            cache_hit=True,
        )
    except (KeyError, TypeError, ValueError, OSError):
        return None


def compile_perforated_adam_chunk(
    directory: str | Path,
    *,
    batch: int,
    in_dim: int,
    out_dim: int,
    cycle_length: int = 4,
    dendrites_per_neuron: int = 2,
    gradient_accumulation_steps: int = 1,
    max_global_gradient_norm: float | None = None,
    name: str = "perforated_adam_chunk",
    use_cache: bool = True,
) -> CompiledPerforatedAdamChunk:
    """Compile changing minibatches through one native loss/VJP/Adam cycle."""
    if min(batch, in_dim, out_dim, cycle_length, dendrites_per_neuron,
           gradient_accumulation_steps) < 1:
        raise ValueError("all perforated Adam chunk dimensions must be positive")
    from .process_graph_autograd import (
        lower_training_motion_to_repository_ssa,
        obtain_graph_reverse,
    )
    from .ssa_llvm_backend import (
        compile_artifact,
        emit_ssa_function_to_llvm,
        with_native_adam_loop,
    )

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if use_cache:
        cached = _load_cached_adam_chunk(
            directory,
            batch=batch,
            in_dim=in_dim,
            out_dim=out_dim,
            cycle_length=cycle_length,
            branches=dendrites_per_neuron,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_global_gradient_norm=max_global_gradient_norm,
        )
        if cached is not None:
            return cached
    loss, prediction, bindings, _shapes = _build_training_graph(
        batch, in_dim, out_dim, dendrites_per_neuron)
    ids = {
        binding_name: int(value.data.value.id)
        for binding_name, value in bindings.items()
    }
    parameter_names = (
        "base_weight", "base_bias", "dendrite_weight",
        "dendrite_bias", "dendrite_gain",
    )
    parameter_ids = tuple(ids[item] for item in parameter_names)
    product = obtain_graph_reverse(
        loss,
        bindings=bindings,
        wrt=parameter_ids,
        packaging="combined",
        unit_output_seed=True,
    )
    if product.motion is None:
        raise RuntimeError("perforated Adam chunk produced no training motion")
    lowering = lower_training_motion_to_repository_ssa(
        product.motion,
        function_name=f"{name}__motion",
        observed_outputs={"prediction": int(prediction.data.value.id)},
    )
    if lowering.shortfalls:
        raise RuntimeError(
            f"{name} repository-SSA shortfalls: {lowering.shortfalls!r}")
    emitted = emit_ssa_function_to_llvm(
        lowering.module,
        lowering.function_name,
        entry_name=lowering.function_name,
    )
    if emitted.shortfalls:
        raise RuntimeError(f"{name} LLVM shortfalls: {emitted.shortfalls!r}")
    gradient_ids = {
        parameter_name: int(product.motion.gradient_value_ids[ids[parameter_name]])
        for parameter_name in parameter_names
    }
    outputs = dict(lowering.outputs)
    wrapped = with_native_adam_loop(
        emitted,
        parameter_gradient_pairs=tuple(
            (ids[parameter_name], gradient_ids[parameter_name])
            for parameter_name in parameter_names
        ),
        cycled_value_ids=tuple(
            ids[item] for item in (
                "x", "target", "sample_weight", "loss_scale",
                "dendrite_mask",
            )
        ) + (int(outputs["loss_0"]),),
        cycle_length=cycle_length,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_global_gradient_norm=max_global_gradient_norm,
        entry_name=name,
    )
    artifact = compile_artifact(wrapped, directory=directory / "native")
    optimizer_state = dict(artifact.optimizer_state_value_ids or {})
    manifest = {
        "schema": "turing.perforated-adam-chunk-llvm",
        "version": 2,
        "build_fingerprint": _compiler_fingerprint(),
        "execution": {
            "motion": "combined-forward-loss-process-graph-vjp",
            "optimizer": "adam",
            "cycle_owner": "native-llvm-entry",
            "tape_autograd": False,
            "changing_minibatches": True,
        },
        "shape": {
            "cycle_length": cycle_length,
            "batch": batch,
            "input": in_dim,
            "output": out_dim,
            "dendrites_per_neuron": dendrites_per_neuron,
        },
        "optimizer_options": {
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_global_gradient_norm": max_global_gradient_norm,
        },
        "inputs": ids,
        "parameters": {name: ids[name] for name in parameter_names},
        "gradients": gradient_ids,
        "outputs": outputs,
        "optimizer_state": optimizer_state,
        "artifact": {
            "entry": artifact.name,
            "library": str(Path(artifact.library_path).resolve()),
            "buffer_order": list(artifact.buffer_order),
            "buffer_shapes": [list(shape) for shape in artifact.buffer_shapes],
            "buffer_dtypes": list(artifact.buffer_dtypes),
            "extent_order": [list(item) for item in artifact.extent_order],
        },
    }
    manifest_path = directory / "contract.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return CompiledPerforatedAdamChunk(
        artifact=artifact,
        input_value_ids=ids,
        parameter_value_ids={name: ids[name] for name in parameter_names},
        gradient_value_ids=gradient_ids,
        output_value_ids=outputs,
        optimizer_state_value_ids=optimizer_state,
        cycle_length=cycle_length,
        manifest_path=manifest_path.resolve(),
        cache_hit=False,
    )


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
    "CompiledPerforatedAdamChunk",
    "CompiledPerforatedNetwork",
    "PerforatedLLVMContract",
    "TensorPort",
    "compile_perforated_adam_chunk",
    "compile_perforated_network",
]
