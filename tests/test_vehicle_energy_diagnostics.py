import sympy
from src.compiler.vehicle_energy_diagnostics import vehicle_energy_expressions, ENERGY_OUTPUTS, ENERGY_DIAGNOSTICS_CONTRACT


def test_compartments_use_actual_state_and_owner_inertias():
    from collections import defaultdict
    s=defaultdict(lambda: sympy.Integer(0))
    s.update(inverse_mass=sympy.Rational(1,1000), velocity_x=3, velocity_y=4,
        inverse_inertia_roll=sympy.Rational(1,100),inverse_inertia_pitch=sympy.Rational(1,200),
        inverse_inertia_yaw=sympy.Rational(1,300), roll_velocity=2, gravity=-10, position_y=2,
        engine_angular_speed=10, wheel_inertia=4, wheel_omega_front_left=3,
        differential_brake_rotor_inertia=2, external_differential_inertia_front=1,
        differential_wrench_shaft_omega_front=4)
    result=vehicle_energy_expressions(s, sympy.Integer(6))
    assert tuple(result)==ENERGY_OUTPUTS
    assert result['energy_chassis_translation_j']==12500
    assert result['energy_chassis_rotation_j']==200
    assert result['energy_chassis_gravity_j']==20000
    assert result['energy_engine_rotation_j']==300
    assert result['energy_wheel_rotation_front_left_j']==18
    assert result['energy_differential_rotation_front_j']==24
    assert ENERGY_DIAGNOSTICS_CONTRACT['complete_budget'] is False


def test_diagnostic_equations_lower_through_existing_compiler(tmp_path):
    from src.compiler.symbolic_equation_compiler import compile_symbolic_program, SymbolicPublication
    def authored_energy_slice():
        from src.compiler.abstract_ui_vehicles import _symbols
        s = _symbols()
        # Same callable used by the vehicle transition; no independent formula.
        values = vehicle_energy_expressions(s, s['engine_rotating_inertia'])
        return tuple(sympy.Eq(sympy.Symbol(name, real=True),value,evaluate=False) for name,value in values.items()),s
    compiled=compile_symbolic_program(authored_energy_slice,name='vehicle_energy_diagnostic_slice',
        publications=tuple(SymbolicPublication(name,f'world.vehicle.{name}') for name in ENERGY_OUTPUTS))
    from src.compiler.identity_concordance import concordance_report
    assert '0 finding(s)' in concordance_report(compiled.module)
    from src.compiler.ssa_wasm_backend import emit_ssa_function_to_wasm
    from src.compiler.ssa_webgpu_backend import emit_module
    wasm=emit_ssa_function_to_wasm(compiled.module,compiled.function.name)
    assert wasm.complete,wasm.shortfalls
    assert set(wasm.output_names)==set(ENERGY_OUTPUTS)
    compiled=compile_symbolic_program(authored_energy_slice,name='vehicle_energy_diagnostic_slice_gpu',dtype='float32',
        publications=tuple(SymbolicPublication(name,f'world.vehicle.{name}') for name in ENERGY_OUTPUTS),key_sources=(vehicle_energy_expressions,))
    returned=next(i.args for b in compiled.function.blocks.values() for i in b.instrs if i.op.lower() in {'ret','return'})
    gpu=emit_module(compiled.module,name=compiled.function.name,outputs={compiled.function.name:returned},count=1,packed_outputs=True)
    assert gpu.complete,gpu.shortfalls
    import json,subprocess
    _,binary=wasm.write(tmp_path)
    values={name:1.0 for name in wasm.input_names}
    values.update(velocity_x=3,velocity_y=4,velocity_z=0,inverse_mass=.001)
    payload={'inputs':list(zip(wasm.input_offsets,[values[n] for n in wasm.input_names])),
        'outputs':list(zip(wasm.output_names,wasm.output_offsets)),'entry':'vehicle_energy_diagnostic_slice'}
    script="""const fs=require('fs');(async()=>{const {instance}=await WebAssembly.instantiate(fs.readFileSync(process.argv[1]));
const p=JSON.parse(process.argv[2]),v=new DataView(instance.exports.memory.buffer);
for(const [offset,value] of p.inputs)v.setFloat64(offset,value,true);instance.exports[p.entry](0);
console.log(JSON.stringify(Object.fromEntries(p.outputs.map(([name,offset])=>[name,v.getFloat64(offset,true)]))));})();"""
    execution=subprocess.run(['node','-e',script,str(binary),json.dumps(payload)],capture_output=True,text=True,check=True)
    actual=json.loads(execution.stdout)
    assert actual['energy_chassis_translation_j']==12500
    assert all(isinstance(value,(float,int)) for value in actual.values())




def test_actual_vehicle_output_contract_appends_energy_publications():
    import src.compiler.abstract_ui_vehicles as v
    equations, _ = v._symbolic_vehicle_equations_authored()
    names=tuple(str(equation.lhs) for equation in equations)
    assert set(names)==set(v.VEHICLE_STATE_OUTPUTS)
    assert names[-len(ENERGY_OUTPUTS):]==ENERGY_OUTPUTS
    assert v.VEHICLE_STATE_OUTPUTS[-len(ENERGY_OUTPUTS):]==ENERGY_OUTPUTS
