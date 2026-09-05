"""The eager run executes each sympy law's AbstractTensor stage.

One program, three stages: authored Python -> AbstractTensor Python ->
native.  The bindings the authored graph calls by name ARE the middle stage,
the program the native product is lowered from.  The sympy reference used
here to prove that stage is the parity harness's lambdify authority; it is a
proof instrument, not an execution path.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.common.tensors import AbstractTensor
from src.compiler.vehicle_python_compilation import (
    symbolic_abstract_tensor_source,
    symbolic_law_compilations,
    vehicle_python_runtime_bindings,
)


def _vectorized_reference(compilation):
    import functools

    import sympy

    metadata = compilation.function.metadata
    by_name = {str(equation.lhs): equation.rhs for equation in compilation.equations}
    # A single explicit namespace (no "numpy" printer): sympy's NumPy printer
    # spells Max/Min as amax/amin over a stacked list, which is not
    # elementwise; these are, and they evaluate the whole column at once.
    modules = [{
        "Max": lambda *xs: functools.reduce(np.maximum, xs),
        "Min": lambda *xs: functools.reduce(np.minimum, xs),
        "Abs": np.abs, "sqrt": np.sqrt, "tanh": np.tanh, "exp": np.exp,
        "sin": np.sin, "cos": np.cos, "log": np.log, "pi": np.pi, "E": np.e,
    }]
    function = sympy.lambdify(
        tuple(sympy.Symbol(name) for name in metadata["argument_names"]),
        tuple(by_name[name] for name in metadata["output_names"]),
        modules=modules, cse=True,
    )

    def evaluate(*columns):
        outputs = function(*columns)
        return [np.broadcast_to(np.asarray(value, dtype=np.float64), columns[0].shape)
                for value in outputs]

    return evaluate


def _law(name):
    if name == "abstract_ui_wheel_contact":
        from src.compiler.abstract_ui_vehicles import compile_wheel_contact_ssa
        return compile_wheel_contact_ssa()
    return symbolic_law_compilations(True)[name]


@pytest.mark.parametrize("law", ("vehicle_member_material_step", "abstract_ui_vehicle_step", "abstract_ui_wheel_contact"))
def test_tensor_stage_matches_the_sympy_reference_on_batch_columns(law):
    compilation = _law(law)
    names = tuple(compilation.function.metadata["argument_names"])
    outputs = tuple(compilation.function.metadata["output_names"])
    rng = np.random.default_rng(7)
    batch = rng.uniform(0.5, 2.0, size=(3, len(names)))
    columns = {name: AbstractTensor.tensor(batch[:, i].copy()) for i, name in enumerate(names)}

    from src.compiler.vehicle_python_compilation import _abstract_tensor_stage_callable
    stage = (vehicle_python_runtime_bindings()[law] if law != "abstract_ui_wheel_contact"
             else _abstract_tensor_stage_callable(compilation, law))
    got = stage(*(columns[name] for name in names))
    # The sympy reference evaluated on the whole batch column at once: a
    # vectorized lambdify over numpy arrays, no lane loop anywhere.
    reference = _vectorized_reference(compilation)
    want = np.stack(reference(*(batch[:, i] for i in range(len(names)))), axis=0)

    assert len(got) == len(outputs)
    for index, value in enumerate(got):
        value = np.broadcast_to(np.asarray(getattr(value, "data", value), dtype=np.float64), (3,))
        assert np.allclose(value, want[index], rtol=1.0e-10, atol=1.0e-12), (law, outputs[index], value, want[index])


def test_tensor_stage_source_is_batch_column_python():
    compilation = symbolic_law_compilations(False)["vehicle_member_material_step"]
    source = symbolic_abstract_tensor_source(compilation, "vehicle_member_material_step")
    assert source.startswith("def vehicle_member_material_step(")
    # The compiler's own vocabulary: Max/Min as the tensor methods, the
    # square root as the SSA's Pow with a constant exponent (``t ** t_half``),
    # and no scalar-Python spellings anywhere.
    assert ".maximum(" in source and " ** " in source
    assert "lambdify" not in source and "math." not in source and "abs(" not in source


def test_bindings_are_the_tensor_stage_and_nothing_else():
    bindings = vehicle_python_runtime_bindings()
    assert bindings["vehicle_member_material_step"].__name__ == "vehicle_member_material_step"
    assert bindings["abstract_ui_vehicle_step"].__name__ == "abstract_ui_vehicle_step"
