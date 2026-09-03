"""Fast compiled chassis/suspension dyno without constructing the living-map page."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.compiler.abstract_ui_vehicles import (
    WHEEL_NAMES,
    _vehicle_mechanical_graph,
    compile_symbolic_vehicle_suspension_rig_wasm,
    compile_wheel_contact_wasm,
    fit_vehicle_chassis_to_power_unit,
    load_default_car_configuration,
    vehicle_configuration_from_mapping,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--engine-start-seconds", type=float, default=10.0)
    parser.add_argument("--engine-mass-kg", type=float, default=220.0)
    parser.add_argument("--engine-length-m", type=float, default=1.05)
    parser.add_argument("--engine-width-m", type=float, default=.62)
    parser.add_argument("--engine-height-m", type=float, default=.76)
    parser.add_argument("--pan-length-m", type=float, default=.72)
    parser.add_argument("--pan-width-m", type=float, default=.48)
    parser.add_argument("--pan-depth-m", type=float, default=.18)
    parser.add_argument("--linear-torque-nm", type=float, default=285.0)
    parser.add_argument("--torque-slope-per-redline", type=float, default=-.22)
    parser.add_argument("--engine-inertia-kg-m2", type=float, default=.34)
    parser.add_argument("--idle-rpm", type=float, default=650.0)
    parser.add_argument("--redline-rpm", type=float, default=4500.0)
    parser.add_argument("--tire-pressure-pa", type=float, default=None)
    parser.add_argument("--spring-stiffness-n-m", type=float, default=None)
    parser.add_argument("--compression-damping-n-s-m", type=float, default=None)
    parser.add_argument("--rebound-damping-n-s-m", type=float, default=None)
    parser.add_argument("--bump-stop-stiffness-n-m", type=float, default=None)
    parser.add_argument("--lvl-m", type=float, default=0.0)
    for corner in WHEEL_NAMES:
        parser.add_argument(f"--lvl-{corner.replace('_', '-')}-m", type=float, default=0.0)
    parser.add_argument("--rig-boundary", choices=("hub-supported", "hub-downforce"),
                        default="hub-supported")
    parser.add_argument("--hub-downforce-n", type=float, default=0.0)
    parser.add_argument("--roller-radius-m", type=float, default=.16)
    parser.add_argument("--roller-separation-m", type=float, default=.30)
    parser.add_argument("--roller-offset-x-m", type=float, default=0.0)
    parser.add_argument("--roller-offset-y-m", type=float, default=0.0)
    parser.add_argument("--roller-offset-z-m", type=float, default=0.0)
    parser.add_argument("--roller-pcm-rate-hz", type=float, default=48_000.0)
    parser.add_argument("--roller-pcm-wave", choices=("sine", "sweep", "impulse", "silence"),
                        default="silence")
    parser.add_argument("--roller-signal-frequency-hz", type=float, default=18.0)
    parser.add_argument("--roller-signal-x-m", type=float, default=0.0)
    parser.add_argument("--roller-signal-y-m", type=float, default=0.0)
    parser.add_argument("--roller-signal-z-m", type=float, default=0.0)
    parser.add_argument("--exciter", action="append", default=[], metavar="NODE:MODE:X,Y,Z[:HZ[:PHASE]]",
                        help="attach a PCM graph exciter; MODE is force, velocity, or position")
    parser.add_argument("--exciter-position-stiffness-n-m", type=float, default=80_000.0)
    parser.add_argument("--exciter-position-damping-n-s-m", type=float, default=4_500.0)
    parser.add_argument("--exciter-velocity-gain-n-s-m", type=float, default=3_200.0)
    parser.add_argument("--fixture-roll-deg", type=float, default=0.0)
    parser.add_argument("--fixture-pitch-deg", type=float, default=0.0)
    parser.add_argument("--fixture-yaw-deg", type=float, default=0.0)
    parser.add_argument("--gravity-frame", choices=("world", "fixture"), default="world")
    parser.add_argument("--html-output", default="docs/generated/vehicle_chassis_dyno.html")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _exciters(args: argparse.Namespace, config) -> list[dict[str, Any]]:
    graph = _vehicle_mechanical_graph(config)
    positions = {node["identity"]: node["reference_position"] for node in graph["nodes"]}
    result = []
    for specification in args.exciter:
        fields = specification.split(":")
        if len(fields) < 3 or fields[1] not in {"force", "velocity", "position"}:
            raise ValueError(f"invalid graph exciter {specification!r}")
        node, mode = fields[:2]
        if node not in positions:
            raise ValueError(f"graph exciter node {node!r} does not exist")
        vector = [float(value) for value in fields[2].split(",")]
        if len(vector) != 3:
            raise ValueError("graph exciter XYZ vector must have three components")
        result.append({"node": node, "mode": mode, "amplitude": vector,
                       "frequency_hz": float(fields[3]) if len(fields) > 3 else args.roller_signal_frequency_hz,
                       "phase_rad": float(fields[4]) if len(fields) > 4 else 0.0,
                       "reference_position": positions[node]})
    return result


def _fit_configuration(args: argparse.Namespace):
    config, shared_fit = fit_vehicle_chassis_to_power_unit(
        load_default_car_configuration(),
        engine_envelope_m=(args.engine_length_m, args.engine_height_m, args.engine_width_m),
        oil_pan_envelope_m=(args.pan_length_m, args.pan_depth_m, args.pan_width_m),
        engine_mass_kg=args.engine_mass_kg,
    )
    source = copy.deepcopy(config.source)
    chassis, wheels = source["chassis"], source["wheels"]
    bay_length, bay_width = shared_fit["minimum_expanded_bay_m"]
    added_frame_mass = shared_fit["added_frame_mass_kg"]
    suspension = source["suspension"]
    tires = source["tires"]
    if args.tire_pressure_pa is not None:
        tires["pressure_pa"] = args.tire_pressure_pa
    if args.spring_stiffness_n_m is not None:
        suspension["stiffness"] = args.spring_stiffness_n_m
    if args.compression_damping_n_s_m is not None:
        suspension["pneumatic_compression_damping"] = args.compression_damping_n_s_m
    if args.rebound_damping_n_s_m is not None:
        suspension["pneumatic_rebound_damping"] = args.rebound_damping_n_s_m
    if args.bump_stop_stiffness_n_m is not None:
        suspension["bump_stop_stiffness_n_per_m"] = args.bump_stop_stiffness_n_m
    config = vehicle_configuration_from_mapping(source)
    mass = config.mass_properties()
    derived_front = float(mass["derived_axle_fractions"]["front"])
    corner_fractions = {
        "front_left": derived_front / 2, "front_right": derived_front / 2,
        "rear_left": (1 - derived_front) / 2, "rear_right": (1 - derived_front) / 2,
    }
    lvl = {corner: args.lvl_m + float(getattr(args, f"lvl_{corner}_m")) for corner in WHEEL_NAMES}
    fit = {
        "engine_envelope_m": [args.engine_length_m, args.engine_height_m, args.engine_width_m],
        "oil_pan_envelope_m": [args.pan_length_m, args.pan_depth_m, args.pan_width_m],
        "minimum_expanded_bay_m": [bay_length, bay_width],
        "chassis_half_length_m": chassis["half_length"],
        "chassis_half_width_m": chassis["half_width"],
        "wheelbase_m": 2 * float(wheels["wheelbase_half_length"]),
        "track_m": 2 * float(wheels["track_half_width"]),
        "axle_group_offset_x_m": float(wheels["axle_group_offset_x_m"]),
        "wheel_placement_changed_by_pan_fit": False,
        "wheel_placement": shared_fit["wheel_placement"],
        "added_frame_mass_kg": added_frame_mass,
        "total_mass_kg": source["mass"],
        "sprung_mass_kg": config.sprung_mass(),
        "unsprung_mass_per_corner_kg": config.unsprung_mass_per_corner(),
        "derived_corner_mass_fractions": corner_fractions,
        "center_of_mass_local_m": mass["center_of_mass"],
        "lvl_offsets_m": lvl,
    }
    return config, fit, corner_fractions, lvl


def _artifact(artifact) -> dict[str, Any]:
    return {
        "name": artifact.name,
        "bytes": base64.b64encode(artifact.binary).decode("ascii"),
        "inputs": artifact.input_names,
        "outputs": artifact.output_names,
        "inputOffsets": artifact.input_offsets,
        "outputOffsets": artifact.output_offsets,
        "shortfalls": [item.reason for item in artifact.shortfalls],
    }


def _compiled_artifact_specs() -> tuple[dict[str, Any], dict[str, Any], float, bool]:
    dependency_paths = (
        REPOSITORY_ROOT / "src/compiler/abstract_ui_vehicles.py",
        REPOSITORY_ROOT / "src/compiler/symbolic_equation_compiler.py",
        REPOSITORY_ROOT / "src/compiler/ssa_wasm_backend.py",
        REPOSITORY_ROOT / "src/compiler/wasm_binary.py",
    )
    digest = hashlib.sha256(b"vehicle-chassis-dyno-compiled-kernels-v1\0" + b"".join(
        path.read_bytes() for path in dependency_paths)).hexdigest()
    cache = REPOSITORY_ROOT / ".cache" / "vehicle_chassis_dyno" / digest / "artifacts.json"
    if cache.exists():
        value = json.loads(cache.read_text(encoding="utf-8"))
        return value["contact"], value["rig"], 0.0, True
    started = time.perf_counter()
    contact = compile_wheel_contact_wasm()
    rig = compile_symbolic_vehicle_suspension_rig_wasm()
    compile_seconds = time.perf_counter() - started
    if contact.shortfalls or rig.shortfalls:
        raise RuntimeError("compiler shortfalls: " + "; ".join(
            [item.reason for item in (*contact.shortfalls, *rig.shortfalls)]))
    value = {"contact": _artifact(contact), "rig": _artifact(rig)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(value), encoding="utf-8")
    return value["contact"], value["rig"], compile_seconds, False


NODE_HARNESS = r"""
const payload=JSON.parse(require("fs").readFileSync(0,"utf8"));
async function runner(spec){
  const bytes=Buffer.from(spec.bytes,"base64"),made=await WebAssembly.instantiate(bytes,{}),view=new DataView(made.instance.exports.memory.buffer);
  return {spec,view,fn:made.instance.exports[spec.name]};
}
function run(r,values){
  r.spec.inputs.forEach((name,index)=>r.view.setFloat64(Number(r.spec.inputOffsets[index]),Number(values[name]??0),true));
  r.fn(0);const out={};r.spec.outputs.forEach((name,index)=>out[name]=r.view.getFloat64(Number(r.spec.outputOffsets[index]),true));return out;
}
(async()=>{
  const contact=await runner(payload.contact),rig=await runner(payload.rig),p=payload.parameters,cfg=p.config,s=cfg.suspension,t=cfg.tires,w=cfg.wheels,ch=cfg.chassis;
  const names=["front_left","front_right","rear_left","rear_right"],audioSignalActive=p.roller.wave!=="silence"||p.exciters.length>0,
    simulationRate=audioSignalActive?p.roller.pcmRate:120*3,dt=1/simulationRate,gravity=Math.abs(cfg.world.gravity),Ms=p.fit.sprung_mass_kg,mu=p.fit.unsprung_mass_per_corner_kg,
    fixtureAngles=p.fixture.rotation_deg.map(value=>value*Math.PI/180),sr=Math.sin(fixtureAngles[0]),cr=Math.cos(fixtureAngles[0]),sp=Math.sin(fixtureAngles[1]),cp=Math.cos(fixtureAngles[1]),
    localGravity=p.fixture.gravity_frame==="fixture"?[0,-gravity,0]:[-gravity*sp,-gravity*cp*cr,gravity*cp*sr],supportGravity=Math.max(gravity*.02,-localGravity[1]);
  const state={q:{},qd:{},N:{},Fs:{},x:0,y:0,z:0,vx:0,vy:0,vz:0,roll:0,rollRate:0,pitch:0,pitchRate:0,yaw:0,yawRate:0,rpm:0};
  for(const name of names){state.q[name]=Math.min(s.travel,Ms*supportGravity*p.cornerFractions[name]/s.stiffness);state.qd[name]=0;}
  function contactInputs(name,penetration,radialVelocity,normal=[0,1,0]){return {...p.contactDefaults,dt,support:penetration>0?1:0,
    previous_compression:state.q[name],compression_velocity:state.qd[name],geometric_compression:state.q[name],
    tire_radial_compression:Math.max(0,penetration),tire_radial_velocity:radialVelocity,
    normal_x:normal[0],normal_y:normal[1],normal_z:normal[2],
    corner_weight:(Ms*p.cornerFractions[name]+mu)*supportGravity};}
  function forceFor(name,penetration){return run(contact,contactInputs(name,penetration,0)).chassis_force_y;}
  const desired={};for(const name of names){const target=(Ms*p.cornerFractions[name]+mu)*supportGravity/2;let lo=0,hi=t.toroid_section_radius_m*1.65;
    for(let i=0;i<50;i++){const mid=(lo+hi)/2;if(forceFor(name,mid)<target)lo=mid;else hi=mid;}desired[name]=(lo+hi)/2;}
  const front=ch.clearance+s.rest_length-state.q.front_left+p.lvl.front_left,
    rear=ch.clearance+s.rest_length-state.q.rear_left+p.lvl.rear_left;
  state.y=(front+rear)/2;state.pitch=(front-rear)/(2*w.wheelbase_half_length);
  state.initialNodePositions=p.exciters.map(exciter=>{const local=exciter.reference_position.map(Number);
    return [state.x+local[0]-state.pitch*local[1],state.y+local[1]+state.pitch*local[0],state.z+local[2]];});
  const rollerBaseY={};names.forEach((name,index)=>{const frontSign=index<2?1:-1,sideSign=index%2===0?-1:1,
    hubY=state.y-ch.clearance-s.rest_length+state.q[name]+t.radius+p.lvl[name]+state.pitch*(Number(w.axle_group_offset_x_m||0)+frontSign*w.wheelbase_half_length)-state.roll*sideSign*w.track_half_width,
    distance=t.radius+p.roller.radius-desired[name],halfGap=p.roller.separation/2;
    rollerBaseY[name]=hubY-Math.sqrt(Math.max(1e-8,distance*distance-halfGap*halfGap));});
  let pcmFrame=0;
  function pcmWindow(time){const count=Math.max(1,Math.round(p.roller.pcmRate*dt)),sum=[0,0,0],first=[0,0,0],last=[0,0,0],exciterSums=p.exciters.map(()=>0);
    for(let sample=0;sample<count;sample++){const sampleTime=(pcmFrame+sample)/p.roller.pcmRate,phase=2*Math.PI*p.roller.frequency*sampleTime,
      carrier=p.roller.wave==="sine"?Math.sin(phase):p.roller.wave==="sweep"?Math.sin(phase*(1+sampleTime/Math.max(.001,p.seconds))):
        p.roller.wave==="impulse"&&sampleTime<.002?Math.sin(Math.PI*sampleTime/.002):0,
      value=p.roller.amplitude.map(component=>component*carrier);if(sample===0)value.forEach((v,i)=>first[i]=v);
      value.forEach((v,i)=>{sum[i]+=v;last[i]=v;});p.exciters.forEach((exciter,index)=>{
        exciterSums[index]+=Math.sin(2*Math.PI*exciter.frequency_hz*sampleTime+exciter.phase_rad);});}
    pcmFrame+=count;return {position:sum.map(v=>v/count),velocity:last.map((v,i)=>(v-first[i])/dt),
      exciterCarrier:exciterSums.map(v=>v/count)};}
  let initialEnergy=null,minEnergy=Infinity,maxEnergyCreation=0,maxPenetration=0,maxAbsQd=0,dropoutTicks=0,longestDropout=0,nonfinite=false,oscillations=0,lastVy=0,epsilonAtStart=null;
  const totalSteps=Math.round(p.seconds/dt),startStep=Math.round(p.engineStart/dt),angularDamping=4.2;
  for(let step=0;step<totalSteps;step++){
    const signal=pcmWindow(step*dt),contactNormal={},exciterForce=[0,0,0],exciterTorque=[0,0,0];
    p.exciters.forEach((exciter,index)=>{const local=exciter.reference_position.map(Number),carrier=signal.exciterCarrier[index],
      arm=[local[0]+state.yaw*local[2]-state.pitch*local[1],local[1]+state.pitch*local[0]-state.roll*local[2],
        local[2]+state.roll*local[1]-state.yaw*local[0]],position=[state.x+arm[0],state.y+arm[1],state.z+arm[2]],
      velocity=[state.vx+state.yawRate*arm[2]-state.pitchRate*arm[1],state.vy+state.pitchRate*arm[0]-state.rollRate*arm[2],
        state.vz+state.rollRate*arm[1]-state.yawRate*arm[0]],target=exciter.amplitude.map(value=>value*carrier);let force=[0,0,0];
      if(exciter.mode==="force")force=target;else if(exciter.mode==="velocity")force=target.map((value,axis)=>(value-velocity[axis])*p.exciterGains.velocity);
      else force=target.map((value,axis)=>p.exciterGains.positionStiffness*(state.initialNodePositions[index][axis]+value-position[axis])-
        p.exciterGains.positionDamping*velocity[axis]);
      force.forEach((value,axis)=>exciterForce[axis]+=value);const torque=[arm[1]*force[2]-arm[2]*force[1],
        arm[2]*force[0]-arm[0]*force[2],arm[0]*force[1]-arm[1]*force[0]];torque.forEach((value,axis)=>exciterTorque[axis]+=value);});
    names.forEach((name,index)=>{const frontSign=index<2?1:-1,sideSign=index%2===0?-1:1,
      hubY=state.y-ch.clearance-s.rest_length+state.q[name]+t.radius+p.lvl[name]+state.pitch*(Number(w.axle_group_offset_x_m||0)+frontSign*w.wheelbase_half_length)-state.roll*sideSign*w.track_half_width,
      wheelVy=state.vy+state.qd[name]+state.pitchRate*(Number(w.axle_group_offset_x_m||0)+frontSign*w.wheelbase_half_length)-state.rollRate*sideSign*w.track_half_width,
      hubX=Number(w.axle_group_offset_x_m||0)+frontSign*w.wheelbase_half_length,hubZ=sideSign*w.track_half_width;
      let verticalForce=0;for(const pairSign of [-1,1]){const center=[hubX+pairSign*p.roller.separation/2+p.roller.offset[0]+signal.position[0],
        rollerBaseY[name]+p.roller.offset[1]+signal.position[1],hubZ+p.roller.offset[2]+signal.position[2]],
        delta=[hubX-center[0],hubY-center[1],hubZ-center[2]],distance=Math.max(1e-8,Math.hypot(...delta)),normal=delta.map(v=>v/distance),
        penetration=Math.max(0,t.radius+p.roller.radius-distance),hubVelocity=[0,wheelVy,0],relativeVelocity=hubVelocity.map((v,i)=>v-signal.velocity[i]),
        radialVelocity=relativeVelocity.reduce((sum,v,i)=>sum+v*normal[i],0),out=run(contact,contactInputs(name,penetration,radialVelocity,normal));
        verticalForce+=Math.max(0,Number(out.chassis_force_y));maxPenetration=Math.max(maxPenetration,penetration);}
      contactNormal[name]=verticalForce;state.N[name]=verticalForce;
    });
    const input={...p.rigDefaults,gravity:localGravity[1],dt,velocity_y:state.vy,roll_velocity:state.rollRate,pitch_velocity:state.pitchRate,
      total_force_y:exciterForce[1]+(p.rigBoundary==="hub-downforce"?-p.hubDownforceN:0),contact_wrench_force_y:0};
    names.forEach(name=>{input[`compression_${name}`]=state.q[name];input[`compression_velocity_${name}`]=state.qd[name];
      input[`contact_normal_force_${name}`]=contactNormal[name];});
    const out=run(rig,input);state.vy=out.velocity_y_next;state.y+=state.vy*dt;
    state.vx+=(localGravity[0]+exciterForce[0]/Ms)*dt;state.x+=state.vx*dt;state.vz+=(localGravity[2]+exciterForce[2]/Ms)*dt;state.z+=state.vz*dt;
    names.forEach(name=>{state.q[name]=out[`compression_${name}_next`];state.qd[name]=out[`compression_velocity_${name}_next`];
      state.Fs[name]=out[`spring_force_${name}`];maxAbsQd=Math.max(maxAbsQd,Math.abs(state.qd[name]));});
    let engineReaction=0;if(step>=startStep){if(step===startStep)epsilonAtStart=Math.hypot(state.vy,...Object.values(state.qd));
      const normalized=Math.min(1,state.rpm/p.redlineRpm),torque=Math.max(0,p.linearTorqueNm*(1+p.torqueSlope*normalized)),load=.72*torque;
      state.rpm=Math.min(p.redlineRpm,Math.max(p.idleRpm,state.rpm+(torque-load)/p.engineInertia*dt*60/(2*Math.PI)));engineReaction=-(torque-load);}
    const com=p.fit.center_of_mass_local_m,gravityTorque=[com[1]*Ms*localGravity[2]-com[2]*Ms*localGravity[1],
      com[2]*Ms*localGravity[0]-com[0]*Ms*localGravity[2],com[0]*Ms*localGravity[1]-com[1]*Ms*localGravity[0]],
      rollTorque=(state.Fs.front_left+state.Fs.rear_left-state.Fs.front_right-state.Fs.rear_right)*w.track_half_width+engineReaction+exciterTorque[0]+gravityTorque[0],
      pitchTorque=(state.Fs.front_left+state.Fs.front_right-state.Fs.rear_left-state.Fs.rear_right)*w.wheelbase_half_length+
        (state.Fs.front_left+state.Fs.front_right+state.Fs.rear_left+state.Fs.rear_right)*Number(w.axle_group_offset_x_m||0)+exciterTorque[2]+gravityTorque[2];
    state.rollRate=(state.rollRate+rollTorque*p.inverseInertiaRoll*dt)/(1+angularDamping*dt);state.roll+=state.rollRate*dt;
    state.pitchRate=(state.pitchRate+pitchTorque*p.inverseInertiaPitch*dt)/(1+angularDamping*dt);state.pitch+=state.pitchRate*dt;
    state.yawRate=(state.yawRate+(exciterTorque[1]+gravityTorque[1])*p.inverseInertiaYaw*dt)/(1+angularDamping*dt);state.yaw+=state.yawRate*dt;
    const contacts=names.filter(name=>state.N[name]>20).length;dropoutTicks=contacts<4?dropoutTicks+1:0;longestDropout=Math.max(longestDropout,dropoutTicks);
    const energy=.5*Ms*state.vy**2+names.reduce((sum,name)=>sum+.5*mu*(state.vy+state.qd[name])**2+.5*s.stiffness*state.q[name]**2,0);
    if(step<startStep){if(initialEnergy===null)initialEnergy=energy;minEnergy=Math.min(minEnergy,energy);maxEnergyCreation=Math.max(maxEnergyCreation,energy-minEnergy);
      if(state.vy*lastVy<0&&(Math.abs(state.vy)>.002||Math.abs(lastVy)>.002))oscillations++;}lastVy=state.vy;
    if(!Number.isFinite(state.y)||!Number.isFinite(state.vy)||!Object.values(state.q).every(Number.isFinite))nonfinite=true;
  }
  const failures=[];if(nonfinite)failures.push("non-finite-state");if(longestDropout*dt>.25)failures.push("contact-dropout");
  if(maxPenetration>t.toroid_section_radius_m*1.60)failures.push("excessive-tire-crossing");if(maxEnergyCreation>Math.max(250,(initialEnergy||0)*.03))failures.push("passive-energy-creation");
  console.log(JSON.stringify({pass:failures.length===0,failures,compiler:{contactShortfalls:payload.contact.shortfalls,rigShortfalls:payload.rig.shortfalls},
    simulatedSeconds:p.seconds,substeps:totalSteps,dynamicSubframeRateHz:simulationRate,pcmFrames:pcmFrame,engineStartedAtSeconds:p.engineStart,engineStartEpsilon:epsilonAtStart,
    maximumTirePenetrationM:maxPenetration,maximumCompressionSpeedMps:maxAbsQd,longestContactDropoutSeconds:longestDropout*dt,
    passiveEnergyCreationJ:maxEnergyCreation,passiveOscillations:oscillations,final:{...state},fit:p.fit},null,2));
})().catch(error=>{console.error(error.stack||error);process.exit(1)});
"""


def _write_orthographic_report(path: Path, result: dict[str, Any]) -> None:
    template = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Compiled vehicle chassis dyno</title><style>body{margin:0;background:#09110f;color:#d8eee7;font:14px ui-monospace,monospace}main{max-width:1200px;margin:auto;padding:20px}.views{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}canvas,pre{background:#101b18;border:1px solid #31534a;border-radius:8px}canvas{width:100%;height:280px}.gimbal{grid-column:1/-1;height:420px;cursor:grab}.gimbal:active{cursor:grabbing}pre{padding:14px;overflow:auto}.pass{color:#85e0b8}.fail{color:#ff9188}@media(max-width:800px){.views{grid-template-columns:1fr}.gimbal{grid-column:auto}}</style></head>
<body><main><h1>Compiled chassis dyno · <span id="status"></span></h1><p>Drag to move the independent inspection camera around the operating fixture. Physical fixture rotation and its gravity frame are separate tested parameters. The orthographic views remain measurement references.</p><div class="views"><canvas class="gimbal" id="gimbal" width="1120" height="420"></canvas><canvas id="front" width="360" height="280"></canvas><canvas id="side" width="360" height="280"></canvas><canvas id="top" width="360" height="280"></canvas></div><pre id="report"></pre></main><script>
const r=__DATA__,status=document.querySelector('#status');status.textContent=r.pass?'PASS':'FAIL';status.className=r.pass?'pass':'fail';document.querySelector('#report').textContent=JSON.stringify(r,null,2);
function view(id,kind){const c=document.querySelector('#'+id),x=c.getContext('2d'),fit=r.fit,rig=r.rig,scale=kind==='side'?105:kind==='top'?100:125,ox=c.width/2,oy=c.height*.52;x.clearRect(0,0,c.width,c.height);x.strokeStyle='#34554d';for(let i=0;i<c.width;i+=24){x.beginPath();x.moveTo(i,0);x.lineTo(i,c.height);x.stroke()}for(let i=0;i<c.height;i+=24){x.beginPath();x.moveTo(0,i);x.lineTo(c.width,i);x.stroke()}x.lineWidth=3;x.strokeStyle='#8fd8bd';const halfL=fit.chassis_half_length_m*scale,halfW=fit.chassis_half_width_m*scale;if(kind==='side')x.strokeRect(ox-halfL,oy-45,2*halfL,55);else if(kind==='front')x.strokeRect(ox-halfW,oy-45,2*halfW,55);else x.strokeRect(ox-halfL,oy-halfW,2*halfL,2*halfW);const wheelR=.57*scale,rollerR=rig.radius_m*scale,gap=rig.separation_m*scale;
const hubs=kind==='side'?[[ox-fit.wheelbase_m/2*scale,oy+25],[ox+fit.wheelbase_m/2*scale,oy+25]]:kind==='front'?[[ox-fit.track_m/2*scale,oy+25],[ox+fit.track_m/2*scale,oy+25]]:[[ox-fit.wheelbase_m/2*scale,oy-fit.track_m/2*scale],[ox-fit.wheelbase_m/2*scale,oy+fit.track_m/2*scale],[ox+fit.wheelbase_m/2*scale,oy-fit.track_m/2*scale],[ox+fit.wheelbase_m/2*scale,oy+fit.track_m/2*scale]];for(const h of hubs){x.strokeStyle='#dceae4';x.beginPath();x.arc(h[0],h[1],kind==='top'?16:wheelR,0,Math.PI*2);x.stroke();x.fillStyle='#78aeca';for(const s of [-1,1]){x.beginPath();x.arc(h[0]+(kind==='front'?0:s*gap/2),h[1]+(kind==='top'?24:wheelR+rollerR-6),rollerR,0,Math.PI*2);x.fill()}}x.fillStyle='#dceae4';x.fillText(kind.toUpperCase(),12,20)}view('front','front');view('side','side');view('top','top');
const gc=document.querySelector('#gimbal'),gx=gc.getContext('2d');let gy=-.65,gp=.38,zoom=145,drag=false,last=[0,0];gc.onpointerdown=e=>{drag=true;last=[e.clientX,e.clientY];gc.setPointerCapture(e.pointerId)};gc.onpointerup=()=>drag=false;gc.onpointermove=e=>{if(!drag)return;gy+=(e.clientX-last[0])*.008;gp=Math.max(-1.45,Math.min(1.45,gp+(e.clientY-last[1])*.008));last=[e.clientX,e.clientY]};gc.onwheel=e=>{zoom=Math.max(70,Math.min(260,zoom-e.deltaY*.12));e.preventDefault()};
function gimbal(time){gx.clearRect(0,0,gc.width,gc.height);const cy=Math.cos(gy),sy=Math.sin(gy),cp=Math.cos(gp),sp=Math.sin(gp),pulse=Math.sin(time*.001*2*Math.PI*r.rig.pcm.frequency_hz),amp=r.rig.pcm.amplitude_xyz_m.map(v=>v*pulse),P=v=>{const x=v[0]*cy-v[2]*sy,z=v[0]*sy+v[2]*cy,y=v[1]*cp-z*sp,d=z*sp+v[1]*sp;return[gc.width/2+x*zoom,gc.height*.48-y*zoom,1+d*.08]},line=(a,b,color='#8fd8bd')=>{const A=P(a),B=P(b);gx.strokeStyle=color;gx.lineWidth=2;gx.beginPath();gx.moveTo(A[0],A[1]);gx.lineTo(B[0],B[1]);gx.stroke()},hl=r.fit.chassis_half_length_m,hw=r.fit.chassis_half_width_m,box=[[-hl,.05,-hw],[-hl,.05,hw],[hl,.05,hw],[hl,.05,-hw],[-hl,.42,-hw],[-hl,.42,hw],[hl,.42,hw],[hl,.42,-hw]],edges=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];edges.forEach(e=>line(box[e[0]],box[e[1]]));const hubs=[];for(const x of [-r.fit.wheelbase_m/2,r.fit.wheelbase_m/2])for(const z of [-r.fit.track_m/2,r.fit.track_m/2])hubs.push([x,0,z]);hubs.forEach(h=>{const H=P(h);gx.fillStyle='#dceae4';gx.beginPath();gx.arc(H[0],H[1],12,0,Math.PI*2);gx.strokeStyle='#dceae4';gx.stroke();for(const side of [-1,1]){const roller=[h[0]+side*r.rig.separation_m/2+r.rig.offset_xyz_m[0]+amp[0],-.42+r.rig.offset_xyz_m[1]+amp[1],h[2]+r.rig.offset_xyz_m[2]+amp[2]],R=P(roller);gx.fillStyle='#70a9c6';gx.beginPath();gx.arc(R[0],R[1],Math.max(5,r.rig.radius_m*zoom*.45),0,Math.PI*2);gx.fill();line(h,roller,'#567a70')}});for(const ex of r.rig.graph_exciters){const E=P(ex.reference_position);gx.fillStyle=ex.mode==='force'?'#ff9f72':ex.mode==='velocity'?'#d39cff':'#ffe279';gx.beginPath();gx.arc(E[0],E[1],7,0,Math.PI*2);gx.fill()}gx.fillStyle='#d8eee7';gx.fillText('independent camera rotator · drag to orbit · wheel to zoom',14,22);gx.fillText('fixture '+r.rig.fixture_rotator.rotation_deg.join(', ')+'° · gravity '+r.rig.fixture_rotator.gravity_frame,14,42);requestAnimationFrame(gimbal)}requestAnimationFrame(gimbal);
</script></body></html>'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template.replace("__DATA__", json.dumps(result)), encoding="utf-8")


def main() -> int:
    args = _arguments()
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("Node.js is required to execute the compiled Wasm dyno")
    config, fit, fractions, lvl = _fit_configuration(args)
    exciters = _exciters(args, config)
    contact, rig, compile_seconds, cache_hit = _compiled_artifact_specs()
    defaults = config.parameter_defaults()
    suspension, tires, wheels, chassis = (config.source["suspension"], config.source["tires"],
                                           config.source["wheels"], config.source["chassis"])
    contact_defaults = {
        "normal_y": 1, "forward_x": 1, "right_z": 1, "suspension_alignment": 1,
        "wheelbase_half_length": wheels["wheelbase_half_length"],
        "axle_group_offset_x_m": wheels["axle_group_offset_x_m"],
        "track_half_width": wheels["track_half_width"], "corner_front_sign": 1,
        "corner_side_sign": 1, "suspension_rest_length": suspension["rest_length"],
        "chassis_clearance": chassis["clearance"], "suspension_travel": suspension["travel"],
        "spring_stiffness": suspension["stiffness"],
        "bump_stop_stiffness": suspension["bump_stop_stiffness_n_per_m"],
        "bump_stop_progressive_stiffness": suspension["bump_stop_progressive_stiffness_n_per_m2"],
        "bump_stop_damping": suspension["bump_stop_damping_n_s_per_m"],
        "linkage_motion_ratio": 1, "pneumatic_compression_damping": suspension["pneumatic_compression_damping"],
        "pneumatic_rebound_damping": suspension["pneumatic_rebound_damping"],
        "pneumatic_efficiency": suspension["pneumatic_efficiency"],
        "maximum_compression_speed": suspension["maximum_compression_speed"],
        "active_damping_minimum_scale": suspension["active_damping_minimum_scale"],
        "active_damping_maximum_scale": suspension["active_damping_maximum_scale"],
        "active_damping_body_velocity_gain_s_per_m": suspension["active_damping_body_velocity_gain_s_per_m"],
        "active_damping_rebound_release_gain_s_per_m": suspension["active_damping_rebound_release_gain_s_per_m"],
        "tire_pressure": tires["pressure_pa"], "minimum_contact_area": tires["minimum_contact_area"],
        "maximum_contact_area": tires["maximum_contact_area"], "mu_static": tires["static_friction"],
        "mu_kinetic": tires["kinetic_friction"], "load_sensitivity": tires["load_sensitivity"],
        "slip_transition_speed": tires["slip_transition_speed"],
        "tire_major_radius": tires["radius"] - tires["toroid_section_radius_m"],
        "tire_section_radius": tires["toroid_section_radius_m"],
        "tire_effective_tread_width": tires["width"] * tires["effective_tread_width_fraction"],
        "tire_reference_volume": 2 * math.pi ** 2 * (tires["radius"] - tires["toroid_section_radius_m"])
        * tires["toroid_section_radius_m"] ** 2,
        "tire_gas_polytropic_exponent": tires["gas_polytropic_exponent"],
        "radial_carcass_loss": tires["radial_carcass_loss_n_s_per_m"],
        "tire_radial_effective_mass": config.unsprung_mass_per_corner()
        * tires["radial_contact_effective_mass_fraction_of_unsprung"],
        "sidewall_shear_stiffness_longitudinal": tires["sidewall_shear_stiffness_longitudinal_n_per_m"],
        "sidewall_shear_stiffness_lateral": tires["sidewall_shear_stiffness_lateral_n_per_m"],
        "sidewall_shear_damping": tires["sidewall_shear_damping_n_s_per_m"],
    }
    payload = {
        "contact": contact, "rig": rig,
        "parameters": {
            "seconds": args.seconds, "engineStart": args.engine_start_seconds,
            "linearTorqueNm": args.linear_torque_nm, "torqueSlope": args.torque_slope_per_redline,
            "engineInertia": args.engine_inertia_kg_m2, "idleRpm": args.idle_rpm,
            "redlineRpm": args.redline_rpm, "config": config.source, "fit": fit,
            "cornerFractions": fractions, "lvl": lvl, "contactDefaults": contact_defaults,
            "rigDefaults": defaults,
            "rigBoundary": args.rig_boundary, "hubDownforceN": args.hub_downforce_n,
            "roller": {"radius": args.roller_radius_m, "separation": args.roller_separation_m,
                       "offset": [args.roller_offset_x_m, args.roller_offset_y_m,
                                  args.roller_offset_z_m],
                       "pcmRate": args.roller_pcm_rate_hz, "wave": args.roller_pcm_wave,
                       "frequency": args.roller_signal_frequency_hz,
                       "amplitude": [args.roller_signal_x_m, args.roller_signal_y_m,
                                     args.roller_signal_z_m]},
            "exciters": exciters,
            "exciterGains": {"positionStiffness": args.exciter_position_stiffness_n_m,
                             "positionDamping": args.exciter_position_damping_n_s_m,
                             "velocity": args.exciter_velocity_gain_n_s_m},
            "fixture": {"rotation_deg": [args.fixture_roll_deg, args.fixture_pitch_deg,
                                           args.fixture_yaw_deg],
                        "gravity_frame": args.gravity_frame},
            "inverseInertiaRoll": defaults["inverse_inertia_roll"],
            "inverseInertiaPitch": defaults["inverse_inertia_pitch"],
            "inverseInertiaYaw": defaults["inverse_inertia_yaw"],
        },
    }
    completed = subprocess.run([node, "-e", NODE_HARNESS], input=json.dumps(payload),
                               capture_output=True, text=True, check=False)
    if completed.returncode:
        print(completed.stderr, file=sys.stderr)
        return completed.returncode
    result = json.loads(completed.stdout)
    result["compiler"]["compileSeconds"] = compile_seconds
    result["compiler"]["cacheHit"] = cache_hit
    result["rig"] = {
        "boundary": args.rig_boundary, "hub_downforce_n": args.hub_downforce_n,
        "radius_m": args.roller_radius_m, "separation_m": args.roller_separation_m,
        "offset_xyz_m": [args.roller_offset_x_m, args.roller_offset_y_m, args.roller_offset_z_m],
        "pcm": {"sample_rate_hz": args.roller_pcm_rate_hz, "wave": args.roller_pcm_wave,
                "frequency_hz": args.roller_signal_frequency_hz,
                "amplitude_xyz_m": [args.roller_signal_x_m, args.roller_signal_y_m,
                                    args.roller_signal_z_m]},
        "graph_exciters": exciters,
        "fixture_rotator": {"rotation_deg": [args.fixture_roll_deg, args.fixture_pitch_deg,
                                               args.fixture_yaw_deg],
                            "gravity_frame": args.gravity_frame},
        "camera_rotator": "independent-interactive-orbit",
    }
    _write_orthographic_report(REPOSITORY_ROOT / args.html_output, result)
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
