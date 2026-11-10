"""Seconds-scale adapter checks against authored machine input identities."""
import src.compiler.abstract_ui_vehicles as v


def test_gpu_adapter_accepts_actual_unsprung_machine_inputs():
    vehicle_equations, _ = v._symbolic_vehicle_equations_authored()
    contact_equations, _ = v._symbolic_wheel_contact_equations_authored()
    vehicle_inputs = tuple(sorted({str(s) for equation in vehicle_equations for s in equation.rhs.free_symbols}))
    contact_inputs = tuple(sorted({str(s) for equation in contact_equations for s in equation.rhs.free_symbols}))
    for wheel in v.WHEEL_NAMES:
        assert f"target_compression_{wheel}" not in vehicle_inputs
        assert f"wheel_support_{wheel}" not in vehicle_inputs
        assert f"contact_normal_force_{wheel}" in vehicle_inputs
        assert f"compression_velocity_{wheel}" in vehicle_inputs
    outputs = tuple(str(equation.lhs) for equation in vehicle_equations)
    adapters = v._vehicle_gpu_graph_adapters(contact_inputs, v.CONTACT_PATCH_OUTPUTS, vehicle_inputs, outputs)
    assembly = adapters['assembly']['source']
    commit = adapters['commit']['source']
    for wheel in v.WHEEL_NAMES:
        assert f"vehicle_feed[{vehicle_inputs.index('contact_normal_force_' + wheel)}u]" in assembly
        assert f"vehicle_feed[{vehicle_inputs.index('compression_velocity_' + wheel)}u]" in commit
    assert 'chassis_force -= normal_force_' in assembly
    assert 'chassis_torque -= cross(' in assembly


def test_vehicle_slot_leveling_metadata_reads_configuration_without_unbound_local():
    import ast
    import inspect
    source = ast.parse(inspect.getsource(v.vehicle_slot_model))
    config = v.load_default_car_configuration()
    checked = []
    for node in ast.walk(source):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if not isinstance(key, ast.Constant) or key.value not in {
                'piston_area_m2', 'manifold_pressure_pa', 'maximum_flow_m3_s',
                'efficiency', 'stroke_m', 'maximum_rate_m_s',
            }:
                continue
            expression = ast.unparse(value)
            if 'leveling_' not in expression:
                continue
            # Execute the exact authored metadata expression with the function's
            # real configuration. No full vehicle/compiler build is needed.
            result = eval(compile(ast.Expression(value), '<leveling-metadata>', 'eval'),
                          {'float': float}, {'config': config})
            assert result > 0
            checked.append(key.value)
    assert len(checked) == 6
