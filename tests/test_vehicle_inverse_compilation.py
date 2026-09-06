import sympy
import inspect

from src.common.tensors.abstraction import AbstractTensor
from src.compiler.symbolic_equation_compiler import compile_sympy_equations
from src.compiler.vehicle_inverse_compilation import (
    VehicleInverseSpecification,
    VehicleObjectiveMetric,
    prepare_vehicle_inverse_adam_python,
    vehicle_rig_outfit_contract,
)


def test_static_inverse_adam_program_preserves_runtime_parameter_abi_and_two_limbs():
    stiffness, displacement, predicted = sympy.symbols(
        "stiffness displacement predicted")
    compilation = compile_sympy_equations(
        (sympy.Eq(predicted, stiffness * displacement),), name="tiny_vehicle")
    specification = VehicleInverseSpecification(
        parameters=("stiffness",),
        metrics=(VehicleObjectiveMetric("predicted", 0.0, 1.0),),
        adam_learning_rate=.01, adam_beta1=.9, adam_beta2=.999, adam_epsilon=1e-8,
    )
    program = prepare_vehicle_inverse_adam_python(compilation, specification)
    assert program.optimized_parameter_names == ("stiffness",)
    assert "objective_target__predicted" in program.objective_parameter_names
    assert "adam_beta1" in program.optimizer_parameter_names
    assert "Precision.of(stiffness, 2)" in program.source
    assert "beta1=adam_beta1" in program.source
    assert "GradTape" not in program.source
    assert "capture(" not in program.source
    assert program.manifest["json_values_constant_folded"] is False
    assert program.manifest["wasm"] == {
        "scope": "canonical-forward-only", "inverse": False, "adam": False}

    namespace = {}
    exec(program.source, namespace)
    entry = namespace[program.two_limb_entrypoint]
    values = {
        "displacement": 3.0, "stiffness": 2.0,
        "objective_target__predicted": 5.0,
        "objective_weight__predicted": 1.0,
        "adam_m__stiffness": 0.0, "adam_v__stiffness": 0.0,
        "adam_step_index": 0.0, "adam_learning_rate": .01,
        "adam_beta1": .9, "adam_beta2": .999, "adam_epsilon": 1e-8,
    }
    result = entry(**{
        argument: AbstractTensor.get_tensor(values[argument])
        for argument in inspect.signature(entry).parameters
    })
    assert len(result) == 5
    assert all(value.tolist() == value.tolist() for value in result)
    assert float(result[1].tolist()) < 2.0


def test_rig_outfit_contract_requires_each_installed_configuration_to_resolve():
    contract = vehicle_rig_outfit_contract()
    assert contract["starts_with_equipment"] is False
    assert {"body", "engine", "clutch"}.issubset(contract["installable_part_classes"])
    assert contract["stages"][0]["identity"] == "bare-structural-mockup"
    assert contract["stages"][-1]["identity"] == "deployment"
    assert "every outfitted configuration" in contract["rule"]
