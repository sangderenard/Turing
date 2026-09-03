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
from tools.frame_parity import python_backend


@pytest.mark.parametrize("law", ("vehicle_member_material_step", "abstract_ui_vehicle_step"))
def test_tensor_stage_matches_the_sympy_reference_on_batch_columns(law):
    compilation = symbolic_law_compilations(True)[law]
    names = tuple(compilation.function.metadata["argument_names"])
    outputs = tuple(compilation.function.metadata["output_names"])
    rng = np.random.default_rng(7)
    batch = rng.uniform(0.5, 2.0, size=(3, len(names)))
    columns = {name: AbstractTensor.tensor(batch[:, i].copy()) for i, name in enumerate(names)}

    got = vehicle_python_runtime_bindings()[law](*(columns[name] for name in names))
    reference = python_backend(compilation)
    want = np.stack([reference(batch[lane]) for lane in range(batch.shape[0])], axis=1)

    assert len(got) == len(outputs)
    for index, value in enumerate(got):
        value = np.broadcast_to(np.asarray(getattr(value, "data", value), dtype=np.float64), (3,))
        assert np.allclose(value, want[index], rtol=1.0e-10, atol=1.0e-12), (law, outputs[index], value, want[index])


def test_tensor_stage_source_is_batch_column_python():
    compilation = symbolic_law_compilations(False)["vehicle_member_material_step"]
    source = symbolic_abstract_tensor_source(compilation, "vehicle_member_material_step")
    assert source.startswith("def vehicle_member_material_step(")
    assert ".maximum(" in source and ".sqrt()" in source
    assert "lambdify" not in source and "math." not in source


def test_bindings_are_the_tensor_stage_and_nothing_else():
    bindings = vehicle_python_runtime_bindings()
    assert bindings["vehicle_member_material_step"].__name__ == "vehicle_member_material_step"
    assert bindings["abstract_ui_vehicle_step"].__name__ == "abstract_ui_vehicle_step"
