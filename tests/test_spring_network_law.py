"""The batched cloth/rope law agrees at SymPy, AbstractTensor and LLVM stages."""

import numpy as np

from src.common.tensors import AbstractTensor
from src.compiler.spring_network_law import compile_spring_network_step
from src.compiler.vehicle_python_compilation import symbolic_abstract_tensor_source


def columns():
    compilation = compile_spring_network_step()
    result = {name: np.zeros(24, dtype=np.float64)
              for name in compilation.function.metadata["argument_names"]}
    result["inverse_mass"][:] = 1
    result["stiffness"][:] = 10
    result["dt"][:] = .1
    result["occupied"][:3] = 1
    result["position"][0] = 1
    result["edge_0"][0] = result["edge_1"][0] = 1
    result["pinned"][1] = 1
    result["anchor"][1] = 3
    result["velocity"][1] = 5
    result["position"][3] = 7
    result["velocity"][3] = 2
    return compilation, result


def test_batched_spring_step_abstract_tensor():
    compilation, values = columns()
    source = symbolic_abstract_tensor_source(compilation, "spring_step")
    namespace = {"AbstractTensor": AbstractTensor}
    exec(source, namespace)
    outputs = namespace["spring_step"](*(AbstractTensor.tensor(values[name])
                                          for name in compilation.function.metadata["argument_names"]))
    position, velocity = (np.asarray(value.data) for value in outputs)
    assert np.isclose(velocity[0], -2)
    assert np.isclose(position[0], .8)
    assert position[1] == 3 and velocity[1] == 0
    assert position[3] == 7 and velocity[3] == 2


def test_batched_spring_step_native_llvm():
    from src.compiler.native_law_kernels import law_kernel

    compilation, values = columns()
    native = law_kernel(compilation, "spring_network_step", 24, "llvm")
    output = native(values)
    assert np.isclose(output["velocity_next"][0], -2)
    assert np.isclose(output["position_next"][0], .8)
    assert output["position_next"][1] == 3 and output["velocity_next"][1] == 0
    assert output["position_next"][3] == 7 and output["velocity_next"][3] == 2
