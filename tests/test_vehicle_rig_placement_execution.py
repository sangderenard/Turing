"""Exercise the real authored support installation without preparing tires."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.compiler.abstract_ui_archetypes import LivingDocument, LivingNode
from src.compiler.abstract_ui_validator_rig import place_validator_support
from src.compiler.mechanical_ports import bind_placed_rig_point


def test_dually_support_stage_places_and_binds_every_point_in_every_lane():
    path = Path(__file__).resolve().parents[1] / "tools/run_vehicle_native_assembly.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    entry = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == "_run_dually_python_profile")
    stage = next(n for n in entry.body if isinstance(n, ast.FunctionDef)
                 and n.name == "apply_stage")
    install = next(n for n in stage.body if isinstance(n, ast.If)
                   and "grasp_configured" in ast.unparse(n.test))
    material_class = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                          and n.name == "_PythonVehicleMaterial")
    method = next(n for n in material_class.body if isinstance(n, ast.FunctionDef)
                  and n.name == "install_rig_binding")
    setup = [n for n in entry.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id in
                     {"rig_world_identity", "rig_body_identity", "rig_document"}
                     for t in n.targets)]
    namespace = dict(
        LivingDocument=LivingDocument, LivingNode=LivingNode,
        place_validator_support=place_validator_support,
        bind_placed_rig_point=bind_placed_rig_point,
        profile=SimpleNamespace(identity="rig", model={"identity": "loaded-vehicle"},
                                stages=tuple(range(8)),
                                structural_support_positions=tuple((i, 1, 2) for i in range(19))),
        stage=4, grasp_configured=False,
    )
    exec(compile(ast.Module(body=[method, *setup], type_ignores=[]), str(path), "exec"), namespace)
    data = np.zeros((2, 19, 21))
    material = SimpleNamespace(feeds={"rig_points": SimpleNamespace(data=data)},
                               rig_component_identities=("",) * 19)
    material.install_rig_binding = lambda slot, binding: namespace["install_rig_binding"](material, slot, binding)
    namespace["material"] = material
    code = compile(ast.Module(body=[install], type_ignores=[]), str(path), "exec")
    exec(code, namespace)
    document = namespace["rig_document"]
    assert document.revision == 57
    assert len([n for n in document.nodes if n.kind == "attachment-point"]) == 38
    assert namespace["rig_body_identity"] == "loaded-vehicle"
    assert material.rig_component_identities == tuple(f"rig/world/supports/{i}" for i in range(19))
    np.testing.assert_array_equal(data[0], data[1])
    np.testing.assert_array_equal(data[0, :, 2], np.arange(19))
    np.testing.assert_array_equal(data[0, :, 2:5], data[0, :, 5:8])
    np.testing.assert_array_equal(data[0, :, :2], np.ones((19, 2)))
    np.testing.assert_array_equal(data[0, 18, 14:], [80000, 120000, 80000, 800, 1200, 800, 60000])
    # Re-entering the stage must not install duplicate objects or reset commands.
    data[0, 0, 11] = 17
    exec(code, namespace)
    assert namespace["rig_document"] is document and data[0, 0, 11] == 17
