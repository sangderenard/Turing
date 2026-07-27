"""Torch native compilation target for ProcessGraph fusion regions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..common.tensors.fused_ir import (
    ELEMENTWISE_BINARY,
    ELEMENTWISE_UNARY,
    FusedProgram,
    canonical_elementwise_op,
    ordered_feed_ids,
)
from ..common.tensors.torch_backend import PyTorchTensorOperations
from .process_graph_fusion import (
    BackendFusionProfile,
    DispatchRegion,
    ProcessGraphDispatchPlan,
    dispatch_region_to_fused_program,
    plan_process_graph_dispatches,
)


TORCH_FUSIBLE_OPS = frozenset(ELEMENTWISE_UNARY | ELEMENTWISE_BINARY)


def _raw_tensor(value):
    return value.data if isinstance(value, PyTorchTensorOperations) else value


def _region_callable(program: FusedProgram):
    """Build a traceable callable from the existing Torch primitive dispatch."""

    feed_ids = ordered_feed_ids(program)
    output_items = tuple(program.outputs.items())
    operations = PyTorchTensorOperations()

    def run(*feed_values):
        store = dict(zip(feed_ids, feed_values))
        for step in program.steps:
            op, prefix_reverse = canonical_elementwise_op(step.op_name)
            attrs = dict(step.attrs)
            reverse = prefix_reverse ^ bool(attrs.pop("reverse", False))
            has_scalar = "right_scalar" in attrs
            scalar = attrs.pop("right_scalar", None)
            if attrs:
                raise ValueError(
                    f"Torch region step {step.step_id} has unsupported attrs: "
                    f"{sorted(attrs)}"
                )
            args = [store[value_id] for value_id in step.input_ids]
            if op in ELEMENTWISE_UNARY:
                if len(args) != 1 or has_scalar:
                    raise ValueError(f"unary operation {op} has invalid operands")
                result = operations._apply_operator__(op, args[0], None)
            elif len(args) == 2 and not has_scalar:
                left, right = args
                if reverse:
                    left, right = right, left
                result = operations._apply_operator__(op, left, right)
            elif len(args) == 1 and has_scalar:
                left, right = args[0], scalar
                if reverse:
                    left, right = right, left
                result = operations._apply_operator__(op, left, right)
            else:
                raise ValueError(f"binary operation {op} has invalid operands")
            store[step.result_id] = result
        return tuple(store[value_id] for _, value_id in output_items)

    return run


@dataclass
class CompiledTorchRegion:
    """One ProcessGraph region captured by ``torch.compile``."""

    region: DispatchRegion
    program: FusedProgram
    compiler_backend: str
    fullgraph: bool
    dynamic: bool
    _compiled: Any

    @property
    def feed_ids(self) -> tuple[int, ...]:
        return ordered_feed_ids(self.program)

    def __call__(
        self,
        feeds: Mapping[int, Any] | Sequence[Any],
    ) -> dict[str, Any]:
        if isinstance(feeds, Mapping):
            missing = set(self.feed_ids) - set(feeds)
            if missing:
                raise ValueError(f"missing Torch region feeds: {sorted(missing)}")
            values = [_raw_tensor(feeds[value_id]) for value_id in self.feed_ids]
        else:
            values = [_raw_tensor(value) for value in feeds]
        if len(values) != len(self.feed_ids):
            raise ValueError(
                f"expected {len(self.feed_ids)} Torch feeds, got {len(values)}"
            )
        results = self._compiled(*values)
        if not isinstance(results, tuple):
            results = (results,)
        return dict(zip(self.program.outputs, results))


@dataclass(frozen=True)
class TorchProcessGraphCompilation:
    """The shared dispatch plan and Torch-native compiled region callables."""

    plan: ProcessGraphDispatchPlan
    regions: tuple[CompiledTorchRegion, ...]


def compile_torch_region(
    graph,
    region: DispatchRegion,
    *,
    compiler_backend: str = "inductor",
    fullgraph: bool = True,
    dynamic: bool = False,
) -> CompiledTorchRegion:
    """Compile one selected ProcessGraph region with Torch's compiler."""

    import torch

    program = dispatch_region_to_fused_program(graph, region)
    eager = _region_callable(program)
    compiled = torch.compile(
        eager,
        backend=compiler_backend,
        fullgraph=fullgraph,
        dynamic=dynamic,
    )
    return CompiledTorchRegion(
        region,
        program,
        compiler_backend,
        fullgraph,
        dynamic,
        compiled,
    )


def compile_process_graph_torch(
    graph,
    *,
    compiler_backend: str = "inductor",
    fullgraph: bool = True,
    dynamic: bool = False,
    max_steps: int = 4096,
) -> TorchProcessGraphCompilation:
    """Plan a ProcessGraph for Torch and compile every selected region.

    Compilation remains lazy in the standard ``torch.compile`` manner: the
    first region invocation specializes against its actual shapes/dtypes and
    device. CUDA feeds therefore select TorchInductor's CUDA path naturally.
    """

    profile = BackendFusionProfile(
        "torch",
        TORCH_FUSIBLE_OPS,
        max_bindings=1 << 20,
        max_steps=max_steps,
    )
    plan = plan_process_graph_dispatches(graph, profile)
    regions = tuple(
        compile_torch_region(
            graph,
            region,
            compiler_backend=compiler_backend,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        for region in plan.regions
    )
    return TorchProcessGraphCompilation(plan, regions)


__all__ = [
    "CompiledTorchRegion",
    "TORCH_FUSIBLE_OPS",
    "TorchProcessGraphCompilation",
    "compile_process_graph_torch",
    "compile_torch_region",
]
