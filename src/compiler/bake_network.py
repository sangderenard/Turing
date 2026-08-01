"""Freeze a trained AbstractNN and emit it as compilable Python source.

A trained network is a graph plus a pile of numbers. Once the numbers stop
changing, the graph stops needing an autograd engine, a parameter store, or
a framework -- a ``Linear`` layer whose weights are frozen is no longer a
matmul against a variable, it is ``sum(w_i * x_i) + b`` against literals.

So the way to get a trained model into a compiled backend is not to teach
every backend about layers. It is to write the frozen model out as ordinary
Python and hand that to the compiler that already exists: source -> AST ->
ProcessGraph -> AOT -> whichever backend. The model arrives as a program
like any other, and everything downstream -- the SSA lowering, Fortran,
SPIR-V, GLSL, WebAssembly -- serves it without knowing a network was
involved.

The activation is the one thing that has to be spelled out, because a
backend may not have it natively. ``tanh`` reaches WebAssembly through a
baked, error-bounded table (see ``fused_program_wasm_backend.tanh_table``);
here it is simply written as ``.tanh()`` and the backend decides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class FrozenLayer:
    """One affine layer plus the activation that follows it."""

    weight: np.ndarray  # (inputs, outputs)
    bias: np.ndarray  # (outputs,)
    activation: str = "identity"


def _as_array(value: Any) -> np.ndarray:
    data = getattr(value, "data", value)
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    return np.asarray(data, dtype=np.float64)


def freeze_sequential(model: Any) -> tuple[FrozenLayer, ...]:
    """Read a trained ``Sequential`` into plain arrays.

    Reads the layers rather than the flat parameter list, so a weight is
    never paired with the wrong bias by position -- the failure that would
    produce a network which still runs and is simply wrong.
    """

    activations = list(getattr(model, "activations", []) or [])
    frozen: list[FrozenLayer] = []
    for index, layer in enumerate(getattr(model, "layers", []) or []):
        weight = _as_array(getattr(layer, "W", None))
        bias = getattr(layer, "b", None)
        bias_array = (
            np.zeros(weight.shape[-1]) if bias is None else _as_array(bias)
        ).reshape(-1)
        name = "identity"
        if index < len(activations):
            name = type(activations[index]).__name__.lower()
        frozen.append(FrozenLayer(weight=weight, bias=bias_array, activation=name))
    if not frozen:
        raise ValueError("this model exposes no layers to freeze")
    return tuple(frozen)


def _literal(value: float) -> str:
    # repr keeps full double precision; a rounded weight is a different
    # network than the one that was trained.
    return repr(float(value))


def emit_network_source(
    layers: Sequence[FrozenLayer],
    *,
    feature_mean: Sequence[float] | None = None,
    feature_scale: Sequence[float] | None = None,
    function_name: str = "score",
    inner_name: str = "network",
) -> str:
    """Write the frozen network as a compilable Python function.

    Inputs arrive as one argument per feature, because the compiler's feeds
    are named scalars-per-element and a matrix argument would have to be
    indexed -- which is a shape operation, not an elementwise one.

    Normalization is folded in rather than left to the caller: it was part of
    what was trained, and a caller who forgot it would get confident nonsense.
    """

    inputs = int(layers[0].weight.shape[0])
    names = [f"x{index}" for index in range(inputs)]
    lines: list[str] = [f"def {inner_name}({', '.join(names)}):"]

    mean = list(feature_mean) if feature_mean is not None else [0.0] * inputs
    scale = list(feature_scale) if feature_scale is not None else [1.0] * inputs
    current: list[str] = []
    for index, name in enumerate(names):
        term = f"n{index}"
        lines.append(
            f"    {term} = ({name} - {_literal(mean[index])}) "
            f"* {_literal(1.0 / (scale[index] or 1.0))}"
        )
        current.append(term)

    for depth, layer in enumerate(layers):
        weight = np.asarray(layer.weight, dtype=np.float64)
        bias = np.asarray(layer.bias, dtype=np.float64).reshape(-1)
        produced: list[str] = []
        for unit in range(weight.shape[1]):
            # The bias leads, so the first term is a real tensor expression
            # rather than a bare Python float: every feed here is a tensor,
            # and starting from a scalar would make the sum a scalar.
            terms = " + ".join(
                f"{current[source]} * {_literal(weight[source, unit])}"
                for source in range(weight.shape[0])
            )
            name = f"h{depth}_{unit}"
            lines.append(f"    {name} = {terms} + {_literal(bias[unit])}")
            if layer.activation == "tanh":
                lines.append(f"    {name} = {name}.tanh()")
            elif layer.activation not in ("identity", "", None):
                raise ValueError(
                    f"no source form for activation {layer.activation!r}; "
                    "add one rather than silently dropping it"
                )
            produced.append(name)
        current = produced

    if len(current) != 1:
        raise ValueError(
            f"the network ends with {len(current)} outputs; this emitter "
            "writes a single score"
        )
    lines.append(f"    return {current[0]}")
    lines.append("")
    lines.append("")
    lines.append(f"def {function_name}({', '.join(names)}):")
    # The AOT front end wants the entry point to call another function
    # rather than compute inline (see aot_compile's docstring).
    lines.append(
        f"    return {inner_name}({', '.join(f'{n}={n}' for n in names)})"
    )
    lines.append("")
    return "\n".join(lines)


__all__ = ["FrozenLayer", "emit_network_source", "freeze_sequential"]
