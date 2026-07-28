"""Ephemeral Python execution and forward capture for ``ProcessGraph``.

``ProcessGraph`` remains the semantic graph.
``EphemeralProcessGraphCallable`` is only an in-process Python projection of
that graph through the public ``AbstractTensor`` operation table.  Calling
:meth:`capture_forward` records the operations taken by one invocation and
lowers that executed path to the existing backend-neutral ``FusedProgram`` IR.

The callable and its capture are deliberately transient: neither replaces the
source graph, and a capture does not claim to retain dormant control-flow
branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from ..common.tensors.autograd import autograd
from ..transmogrifier.graph.graph_deep_compiler import GraphDeepCompiler
from ..transmogrifier.operator_defs import (
    abstract_tensor_funcs,
    abstract_tensor_sigs,
)

if TYPE_CHECKING:
    from ..common.tensors.accelerator_backends.c_primitive_program import (
        CapturedFusedProgram,
    )


@dataclass
class EphemeralProcessGraphCallable:
    """A transient Python callable projected from one ``ProcessGraph``."""

    graph: Any

    def __post_init__(self) -> None:
        # GraphDeepCompiler installs its Store adapter into the supplied table,
        # so give each callable its own shallow copy of the canonical mapping.
        compiler = GraphDeepCompiler(
            self.graph,
            dict(abstract_tensor_funcs),
            abstract_tensor_sigs,
        )
        self._callable = compiler.build_function()
        self.generated_source = compiler._code

    def __call__(self, **inputs: Any) -> tuple[Any, ...]:
        """Execute the graph through ordinary AbstractTensor operations."""

        return self._callable(**inputs)

    def capture_forward(
        self,
        *,
        dynamic_scalar_ids: tuple[int, ...] = (),
        backward_overrides: Mapping[str, Any] | None = None,
        **inputs: Any,
    ) -> "CapturedFusedProgram":
        """Capture the path taken by one call as an ephemeral fused program."""

        from ..common.tensors.accelerator_backends.c_primitive_program import (
            compile_elementwise_tape,
        )

        with autograd.forward_capture(
            backward_overrides=dict(backward_overrides or {}),
        ) as tape:
            outputs = self(**inputs)

        if len(outputs) == 1:
            captured_outputs: Any = outputs[0]
        else:
            captured_outputs = {
                f"result_{index}": value
                for index, value in enumerate(outputs)
            }
        return compile_elementwise_tape(
            tape,
            captured_outputs,
            dynamic_scalar_ids=dynamic_scalar_ids,
        )


def make_process_graph_callable(graph: Any) -> EphemeralProcessGraphCallable:
    """Return the ephemeral AbstractTensor callable for ``graph``."""

    return EphemeralProcessGraphCallable(graph)


__all__ = ["EphemeralProcessGraphCallable", "make_process_graph_callable"]
