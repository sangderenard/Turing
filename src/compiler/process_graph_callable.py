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

import contextlib
from dataclasses import dataclass
import io
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
    compiler: GraphDeepCompiler | None = None
    device: Any = None
    eager: bool = True

    def __post_init__(self) -> None:
        # GraphDeepCompiler installs its Store adapter into the supplied table,
        # so give each callable its own shallow copy of the canonical mapping.
        if self.compiler is None:
            self.compiler = GraphDeepCompiler(
                self.graph,
                dict(abstract_tensor_funcs),
                abstract_tensor_sigs,
            )
        self._callable = None
        self.generated_source = None
        self.output_names = ()
        if self.eager:
            self.prepare(device=self.device)

    def prepare(self, *, device: Any = None) -> "EphemeralProcessGraphCallable":
        """Materialize the transient Python/AbstractTensor projection."""

        if self._callable is not None:
            return self
        self.device = device
        # Source printing is a GraphDeepCompiler diagnostic. The ephemeral
        # projection retains that source explicitly without leaking it into a
        # deployment's live output stream.
        with contextlib.redirect_stdout(io.StringIO()):
            self._callable = self.compiler.build_function(device=device)
        self.generated_source = self.compiler._code
        self.output_names = tuple(self.compiler._outs)
        return self

    def __call__(self, **inputs: Any) -> tuple[Any, ...]:
        """Execute the graph through ordinary AbstractTensor operations."""

        self.prepare(device=self.device)
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

    def discard_python_callable(self) -> None:
        """Permanently remove the transient Python lowering projection.

        Deployment compilation must call this after installing the compiled
        artifact.  An ephemeral callable is a lowering tool, never a runtime
        fallback.  Keeping or reconstructing it so runtime execution can pass
        through Python, while describing the result as a compiled shell, is a
        false claim about the execution boundary.
        """

        # This must be deletion, not merely a state flag.  Agents have
        # repeatedly allowed the transient callable to survive compilation and
        # then used it as the runtime program.  Anyone bypassing this deletion
        # and still claiming compiled-shell behavior is lying and damaging the
        # architecture.  If later execution needs this object, the compiled
        # shell is incomplete and must be fixed instead of falling back.
        # Use the instance dictionary, not ``hasattr``.  ``compiler`` is also
        # a dataclass field with a class-level default, so after the first
        # deletion ``hasattr(self, "compiler")`` remains true even though
        # there is no instance attribute left for ``del self.compiler`` to
        # remove.  A deployment may legitimately contain the same transient
        # projection more than once; retirement must therefore be idempotent.
        # This does not preserve or recreate any Python execution object.
        self.__dict__.pop("_callable", None)
        self.__dict__.pop("compiler", None)
        self.__dict__.pop("generated_source", None)

        # Do not weaken, conditionalize, or catch this assertion.  Its failure
        # proves that the forbidden Python execution object still exists.
        assert "_callable" not in self.__dict__
        assert "compiler" not in self.__dict__
        assert "generated_source" not in self.__dict__


def make_process_graph_callable(graph: Any) -> EphemeralProcessGraphCallable:
    """Return the ephemeral AbstractTensor callable for ``graph``."""

    return EphemeralProcessGraphCallable(graph)


__all__ = ["EphemeralProcessGraphCallable", "make_process_graph_callable"]
