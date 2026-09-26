import ctypes
import subprocess
import sys

import pytest

from src.compiler.vehicle_mechanical_material import compile_vehicle_member_material_c


def _run(function, names, output_names, **updates):
    values = {
        "dt": .001, "axial_strain": 0.0, "bending_strain": 0.0,
        "shear_strain": 0.0, "axial_strain_rate": 0.0,
        "bending_strain_rate": 0.0, "shear_strain_rate": 0.0,
        "plastic_axial_previous": 0.0, "plastic_bending_previous": 0.0,
        "plastic_shear_previous": 0.0,
        "accumulated_plastic_strain_previous": 0.0,
        "dissipated_energy_previous": 0.0, "failed_previous": 0.0,
        "youngs_modulus_pa": 200e9, "shear_modulus_pa": 77e9,
        "initial_yield_stress_pa": 350e6, "ultimate_stress_pa": 520e6,
        "hardening_modulus_pa": 1.4e9, "fracture_plastic_strain": .12,
        "hardening_fragility": 3.0, "material_volume_m3": 8e-4,
        "axial_viscosity_pa_s": 8e5, "bending_viscosity_pa_s": 8e5,
        "shear_viscosity_pa_s": 5e5,
    }
    values.update(updates)
    inputs = (ctypes.c_double * len(names))(*(values[name] for name in names))
    outputs = (ctypes.c_double * len(output_names))()
    function(inputs, outputs)
    return dict(zip(output_names, outputs))


def test_compiled_member_material_is_elastic_then_plastic_and_irreversibly_fractures(tmp_path):
    artifact = compile_vehicle_member_material_c()
    source = tmp_path / "member.c"
    library = tmp_path / "member.dll"
    source.write_text(artifact.source, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "ziglang", "cc", "-shared", "-O2",
         str(source), "-o", str(library)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    function = ctypes.CDLL(str(library)).vehicle_member_material_step
    function.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]

    elastic = _run(function, artifact.input_names, artifact.output_names,
                   axial_strain=8e-4)
    assert elastic["plastic_axial_next"] == pytest.approx(0.0)
    assert elastic["failed_next"] == 0.0
    assert elastic["elastic_energy_j"] > 0.0

    yielded = _run(function, artifact.input_names, artifact.output_names,
                   axial_strain=.002)
    assert yielded["plastic_axial_next"] > 0.0
    assert yielded["work_hardening_next"] > 0.0
    assert yielded["plastic_work_increment_j"] > 0.0
    assert yielded["failed_next"] == 0.0

    fractured = _run(function, artifact.input_names, artifact.output_names,
                     axial_strain=.2)
    assert fractured["failed_next"] == 1.0
    assert fractured["axial_stress_pa"] == pytest.approx(0.0)
    assert fractured["elastic_energy_j"] == pytest.approx(0.0)
    irreversible = _run(function, artifact.input_names, artifact.output_names,
                        axial_strain=0.0, failed_previous=fractured["failed_next"])
    assert irreversible["failed_next"] == 1.0


def test_work_hardening_raises_yield_and_reduces_remaining_ductility(tmp_path):
    artifact = compile_vehicle_member_material_c()
    source = tmp_path / "member.c"
    library = tmp_path / "member.dll"
    source.write_text(artifact.source, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "ziglang", "cc", "-shared", "-O2",
         str(source), "-o", str(library)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    function = ctypes.CDLL(str(library)).vehicle_member_material_step
    function.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    fresh = _run(function, artifact.input_names, artifact.output_names)
    worked = _run(function, artifact.input_names, artifact.output_names,
                  accumulated_plastic_strain_previous=.04)
    assert worked["current_yield_stress_pa"] > fresh["current_yield_stress_pa"]
    assert worked["remaining_ductility_next"] < fresh["remaining_ductility_next"]
