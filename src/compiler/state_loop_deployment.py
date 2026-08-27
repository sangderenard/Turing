"""Backend-neutral state-loop discovery and JavaScript worker deployment."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable, Iterable, TypeVar


STATE_LOOP_DEPLOYMENT_VERSION = "state-loop-deployment-v0"
_F = TypeVar("_F", bound=Callable[..., object])


def state_loop(*, domain: str, clock: str = "event",
               frequency_hz: float | None = None,
               isolation: str = "auto", identity: str | None = None):
    """Attach portable deployment intent while leaving Python behavior intact."""
    if clock not in {"event", "fixed-step", "animation-frame"}:
        raise ValueError(f"unsupported state-loop clock: {clock}")
    if isolation not in {"auto", "main", "worker"}:
        raise ValueError(f"unsupported state-loop isolation: {isolation}")
    if clock == "fixed-step" and (frequency_hz is None or frequency_hz <= 0):
        raise ValueError("fixed-step state loops require a positive frequency_hz")
    metadata = {"domain": domain, "clock": clock, "frequency_hz": frequency_hz,
                "isolation": isolation, "identity": identity}

    def decorate(function: _F) -> _F:
        setattr(function, "__abstract_ui_state_loop__", metadata)
        return function

    return decorate


@dataclass(frozen=True)
class StateLoop:
    identity: str
    domain: str
    clock: str
    writes: tuple[str, ...]
    reads: tuple[str, ...] = ()
    frequency_hz: float | None = None
    effects: tuple[str, ...] = ()
    isolation: str = "auto"

    def to_data(self) -> dict[str, object]:
        return {"identity": self.identity, "domain": self.domain,
                "clock": self.clock, "frequency_hz": self.frequency_hz,
                "reads": list(self.reads), "writes": list(self.writes),
                "effects": list(self.effects), "isolation": self.isolation}


def _literal_keywords(decorator: ast.Call) -> dict[str, object]:
    values: dict[str, object] = {}
    for keyword in decorator.keywords:
        if keyword.arg:
            try:
                values[keyword.arg] = ast.literal_eval(keyword.value)
            except (ValueError, TypeError):
                pass
    return values


def identify_state_loops(source: str) -> tuple[StateLoop, ...]:
    """Recognize explicit loop annotations without importing user code.

    A backend may infer candidates, but isolation is deliberately attached to
    the source-level ``@state_loop`` contract so timing is never guessed from
    an arbitrary ``while`` loop.
    """
    tree = ast.parse(source)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decorator = next((item for item in node.decorator_list
                          if isinstance(item, ast.Call)
                          and isinstance(item.func, ast.Name)
                          and item.func.id == "state_loop"), None)
        if decorator is None:
            continue
        options = _literal_keywords(decorator)
        reads, writes, effects = set(), set(), set()
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name) \
                    and child.value.id in {"self", "state"}:
                (writes if isinstance(child.ctx, ast.Store) else reads).add(child.attr)
            if isinstance(child, ast.Name) and child.id == "document":
                effects.add("dom")
            if isinstance(child, ast.Name) and child.id in {"gl", "webgl"}:
                effects.add("webgl")
        domain = str(options.get("domain", node.name))
        found.append(StateLoop(
            str(options.get("identity", f"{domain}/{node.name}")), domain,
            str(options.get("clock", "event")), tuple(sorted(writes)),
            tuple(sorted(reads - writes)),
            float(options["frequency_hz"]) if "frequency_hz" in options else None,
            tuple(sorted(effects)), str(options.get("isolation", "auto")),
        ))
    return tuple(found)


def plan_state_loops(loops: Iterable[StateLoop]) -> dict[str, object]:
    values = tuple(loops)
    writers: dict[str, str] = {}
    placements = []
    for loop in values:
        for field in loop.writes:
            if field in writers:
                raise ValueError(f"state {field!r} has two writers: {writers[field]} and {loop.identity}")
            writers[field] = loop.identity
        presentation = bool({"dom", "webgl", "canvas"} & set(loop.effects))
        if presentation and loop.isolation == "worker":
            raise ValueError(
                f"state loop {loop.identity!r} requests a worker but has main-thread presentation effects"
            )
        host = "main" if presentation or loop.clock == "animation-frame" else (
            f"worker:{loop.identity}" if loop.isolation != "main" else "main")
        placements.append({"loop": loop.to_data(), "execution_host": host,
                           "reason": "presentation-affinity" if host == "main" else "independent-state-owner"})
    hosts = {item["loop"]["identity"]: item["execution_host"] for item in placements}
    channels = []
    for producer in values:
        for consumer in values:
            shared = sorted(set(producer.writes) & set(consumer.reads))
            if producer is consumer or not shared or hosts[producer.identity] == hosts[consumer.identity]:
                continue
            channels.append({"producer": producer.identity, "consumer": consumer.identity,
                             "fields": shared, "policy": "latest-complete-snapshot",
                             "capacity": 1, "stale": "overwrite"})
    return {"schema": STATE_LOOP_DEPLOYMENT_VERSION,
            "placements": placements, "channels": channels}


def emit_javascript_physics_worker(frequency_hz: float = 120.0) -> str:
    """Emit a long-lived state owner around the compiler's scalar physics ABI."""
    interval = 1000.0 / frequency_hz
    return f'''"use strict";
const SNAPSHOT_STRIDE=71, SNAPSHOT_POOL_SIZE=3,FIXED_DT={1.0 / frequency_hz!r};
let instance=null, abi=null, vehicleInstance=null,vehicleDynoInstance=null,vehicleAbi=null,contactInstance=null,contactAbi=null,
  vehicleGpu=null, timer=null, sequence=0, snapshotInFlight=false,tickInFlight=false;
let idleTicks=0, coastTicks=0, engineStage="full-dynamics",lastVehicleGpuError=null;
let snapshotCapacity=0, snapshotBuffers=[];
let parameters={{}}, colliders=[], bodies=new Map(), previous=performance.now(),
  worldBottom={{top_y:0,thickness:8,minimum_y:-8}};
function contact(p,r,excluded){{
  let best=null;
  for(const c of colliders){{
    if(c.objectIdentity===excluded||c.surface) continue;
    const mn=c.minimum,mx=c.maximum;
    if(p[1]+r<=mn[1]||p[1]-r>=mx[1]||p[0]<mn[0]-r||p[0]>mx[0]+r||p[2]<mn[2]-r||p[2]>mx[2]+r) continue;
    const cx=Math.max(mn[0],Math.min(mx[0],p[0])), cz=Math.max(mn[2],Math.min(mx[2],p[2]));
    if((p[0]-cx)**2+(p[2]-cz)**2>r*r) continue;
    for(const f of [{{d:p[0]+r-mn[0],n:[1,0],q:mn[0]}},{{d:mx[0]+r-p[0],n:[-1,0],q:-mx[0]}},{{d:p[2]+r-mn[2],n:[0,1],q:mn[2]}},{{d:mx[2]+r-p[2],n:[0,-1],q:-mx[2]}}])
      if(f.d>=0&&(!best||f.d<best.d)) best={{...f,identity:c.identity,runtimePartId:c.runtimePartId||0}};
  }} return best;
}}
function topSupport(previous,next,r,excluded){{
  if(next[1]>=previous[1])return null;let best=null;
  for(const c of colliders){{
    if(c.objectIdentity===excluded||c.role==="projectile-body"||c.surface)continue;
    const mn=c.minimum,mx=c.maximum,top=mx[1];
    if(next[0]<mn[0]-r||next[0]>mx[0]+r||next[2]<mn[2]-r||next[2]>mx[2]+r)continue;
    if(previous[1]-r>=top-.012&&next[1]-r<=top+.006&&(!best||top>best.top))
      best={{top,identity:c.identity,runtimePartId:c.runtimePartId||0}};
  }}return best;
}}
function sampledTerrainSupport(previous,next,r,excluded){{
  let best=null;
  for(const collider of colliders){{const surface=collider.surface,domain=surface?.domain;
    if(collider.objectIdentity===excluded||!surface||!domain||
      next[0]<domain.minimum_x||next[0]>domain.maximum_x||next[2]<domain.minimum_z||next[2]>domain.maximum_z)continue;
    const before=sampleSurface(surface,previous[0],previous[2]),after=sampleSurface(surface,next[0],next[2]),
      previousBottom=previous[1]-r,nextBottom=next[1]-r,
      wasSupported=Math.abs(previousBottom-before.height)<=.045,
      crossed=previousBottom>=before.height-.012&&nextBottom<=after.height+.012,
      followsSurface=wasSupported&&nextBottom<=after.height+.09&&nextBottom>=after.height-.18;
    if(!crossed&&!followsSurface)continue;
    const raw=[-after.gradient[0],1,-after.gradient[1]],length=Math.max(1e-9,Math.hypot(...raw)),
      candidate={{height:after.height,normal:raw.map(value=>value/length),identity:collider.identity,
        runtimePartId:collider.runtimePartId||0}};
    if(!best||candidate.height>best.height)best=candidate;
  }}return best;
}}
function sampleSurface(s,x,z){{
  if(s.kind!=="sampled-height-field")return {{height:s.origin[1]+s.gradient[0]*(x-s.origin[0])+s.gradient[1]*(z-s.origin[2]),gradient:[...s.gradient]}};
  const [columns,rows]=s.resolution,[cellX,cellZ]=s.cell_size,
    u=Math.max(0,Math.min(columns-1,(x-s.origin[0])/cellX)),v=Math.max(0,Math.min(rows-1,(z-s.origin[2])/cellZ)),
    column=Math.min(columns-2,Math.floor(u)),row=Math.min(rows-2,Math.floor(v)),tx=u-column,tz=v-row,
    at=(ix,iz)=>Number(s.heights[iz*columns+ix]),h00=at(column,row),h10=at(column+1,row),
    h01=at(column,row+1),h11=at(column+1,row+1);
  if(tx>=tz)return {{height:h00+(h10-h00)*tx+(h11-h10)*tz,
    gradient:[(h10-h00)/cellX,(h11-h10)/cellZ]}};
  return {{height:h00+(h11-h01)*tx+(h01-h00)*tz,
    gradient:[(h11-h01)/cellX,(h01-h00)/cellZ]}};
}}
function contactSurfaceAt(x,z,bodyY,reach,excluded=null){{
  let best=null;
  const consider=(height,normal,identity,runtimePartId)=>{{
    if(height<bodyY-reach||height>bodyY+reach)return;
    const candidate={{height,normal,identity,runtimePartId:runtimePartId||0}};
    if(!best||height>best.height||height===best.height&&candidate.runtimePartId<best.runtimePartId)best=candidate;
  }};
  const terrainReplacesFloor=colliders.some(c=>{{const s=c.surface,d=s?.domain;return s?.kind==="sampled-height-field"&&
    x>=d.minimum_x&&x<=d.maximum_x&&z>=d.minimum_z&&z<=d.maximum_z;}});
  if(!terrainReplacesFloor)consider(0,[0,1,0],"world-floor",0);
  for(const c of colliders){{if(c.objectIdentity===excluded)continue;
    const s=c.surface;
    if(s){{const d=s.domain;if(x<d.minimum_x||x>d.maximum_x||z<d.minimum_z||z>d.maximum_z)continue;
      const sampled=sampleSurface(s,x,z),h=sampled.height,n=[-sampled.gradient[0],1,-sampled.gradient[1]],
        l=Math.max(1e-9,Math.hypot(...n));
      consider(h,n.map(v=>v/l),c.identity,c.runtimePartId);continue;}}
    if(c.role==="projectile-body"||!c.minimum||!c.maximum)continue;
    if(x<c.minimum[0]||x>c.maximum[0]||z<c.minimum[2]||z>c.maximum[2])continue;
    consider(c.maximum[1],[0,1,0],c.identity,c.runtimePartId);
  }}return best;
}}
async function checkedShaderModule(device,code,label){{
  const module=device.createShaderModule({{code,label}}),info=await module.getCompilationInfo(),
    errors=info.messages.filter(message=>message.type==="error");
  if(errors.length)throw new Error(`${{label}} WGSL invalid · `+errors.map(message=>
    `line ${{message.lineNum}}:${{message.linePos}} ${{message.message}}`).join(" · "));
  return module;
}}
async function initializeVehicleGpu(program){{
  if(!program)throw new Error("resident vehicle WebGPU program is missing");
  if(!globalThis.navigator?.gpu)throw new Error("resident vehicle graph requires worker WebGPU");
  const adapter=await navigator.gpu.requestAdapter();
  if(!adapter)throw new Error("resident vehicle graph could not acquire a WebGPU adapter");
  const device=await adapter.requestDevice(),contact=program.tensor_contact_precompile,
    reduction=program.wrench_reduction,integration=program.vehicle_integration,
    geometry=program.terrain_contact_geometry,adapters=program.graph_adapters;
  if(contact?.packed_outputs&&contact.kernel?.source&&reduction?.kernel?.source){{
    try{{const wheelCount=contact.kernel.invocations,
      contactModule=await checkedShaderModule(device,contact.kernel.source,"compiled-contact"),
      contactPipeline=await device.createComputePipelineAsync({{layout:"auto",compute:{{module:contactModule,entryPoint:"main"}}}}),
      reductionModule=await checkedShaderModule(device,reduction.kernel.source,"backend-gemm-reduction"),
      reductionPipeline=await device.createComputePipelineAsync({{layout:"auto",compute:{{module:reductionModule,entryPoint:"main"}}}}),
      geometryModule=await checkedShaderModule(device,geometry.kernel.source,"terrain-wall-radial-gather"),
      geometryPipeline=await device.createComputePipelineAsync({{layout:"auto",compute:{{module:geometryModule,
        entryPoint:geometry.kernel.entrypoint}}}}),
      assemblyModule=await checkedShaderModule(device,adapters.assembly.source,"vehicle-graph-assembly"),
      assemblyPipeline=await device.createComputePipelineAsync({{layout:"auto",compute:{{module:assemblyModule,
        entryPoint:adapters.assembly.entrypoint}}}}),
      integrationModule=await checkedShaderModule(device,integration.kernel.source,"compiled-vehicle-transition"),
      integrationPipeline=await device.createComputePipelineAsync({{layout:"auto",compute:{{module:integrationModule,
        entryPoint:integration.kernel.entrypoint}}}}),
      commitModule=await checkedShaderModule(device,adapters.commit.source,"vehicle-state-commit"),
      commitPipeline=await device.createComputePipelineAsync({{layout:"auto",compute:{{module:commitModule,
        entryPoint:adapters.commit.entrypoint}}}}),
      feed=device.createBuffer({{size:contact.inputs.length*wheelCount*4,
        usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}}),
      packed=device.createBuffer({{size:contact.outputs.length*wheelCount*4,
        usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}}),
      resultRead=device.createBuffer({{size:(contact.outputs.length*wheelCount+6)*4,
        usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}}),
      unitColumn=device.createBuffer({{size:wheelCount*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}}),
      reduced=device.createBuffer({{size:6*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}}),
      scalars=device.createBuffer({{size:8,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}}),
      vehicleFeed=device.createBuffer({{size:integration.inputs.length*4,
        usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC}}),
      vehicleOutputs=device.createBuffer({{size:integration.outputs.length*4,
        usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}}),
      controls=device.createBuffer({{size:adapters.control_abi.length*4,
        usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}}),
      terrainHeights=device.createBuffer({{size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}}),
      wallColliders=device.createBuffer({{size:24,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}}),
      terrainParameters=device.createBuffer({{size:geometry.terrain_parameter_abi.length*4,
        usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}}),
      contactBindGroup=device.createBindGroup({{layout:contactPipeline.getBindGroupLayout(0),entries:[
        {{binding:0,resource:{{buffer:feed}}}},{{binding:1,resource:{{buffer:packed}}}}]}}),
      reductionBindGroup=device.createBindGroup({{layout:reductionPipeline.getBindGroupLayout(0),entries:[
        {{binding:0,resource:{{buffer:packed}}}},{{binding:1,resource:{{buffer:unitColumn}}}},
        {{binding:2,resource:{{buffer:reduced}}}},{{binding:3,resource:{{buffer:scalars}}}}]}}),
      geometryBindGroup=device.createBindGroup({{layout:geometryPipeline.getBindGroupLayout(0),entries:[
        {{binding:0,resource:{{buffer:terrainHeights}}}},{{binding:1,resource:{{buffer:terrainParameters}}}},
        {{binding:2,resource:{{buffer:vehicleFeed}}}},{{binding:3,resource:{{buffer:feed}}}},
        {{binding:4,resource:{{buffer:controls}}}},{{binding:5,resource:{{buffer:wallColliders}}}}]}}),
      assemblyBindGroup=device.createBindGroup({{layout:assemblyPipeline.getBindGroupLayout(0),entries:[
        {{binding:0,resource:{{buffer:packed}}}},{{binding:1,resource:{{buffer:reduced}}}},
        {{binding:2,resource:{{buffer:feed}}}},{{binding:3,resource:{{buffer:vehicleFeed}}}},
        {{binding:4,resource:{{buffer:controls}}}}]}}),
      integrationBindGroup=device.createBindGroup({{layout:integrationPipeline.getBindGroupLayout(0),entries:[
        {{binding:0,resource:{{buffer:vehicleFeed}}}},{{binding:1,resource:{{buffer:vehicleOutputs}}}}]}}),
      commitBindGroup=device.createBindGroup({{layout:commitPipeline.getBindGroupLayout(0),entries:[
        {{binding:0,resource:{{buffer:vehicleOutputs}}}},{{binding:1,resource:{{buffer:vehicleFeed}}}}]}}),
      snapshotValueCount=integration.outputs.length+contact.outputs.length*wheelCount+6+contact.inputs.length*wheelCount,
      snapshotReads=Array.from({{length:3}},()=>({{busy:false,buffer:device.createBuffer({{size:snapshotValueCount*4,
        usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}})}}));
      device.queue.writeBuffer(unitColumn,0,new Float32Array(wheelCount).fill(1));
      device.queue.writeBuffer(scalars,0,new Float32Array([
        Number(reduction.kernel.scalars?.alpha??1),Number(reduction.kernel.scalars?.beta??0)]));
      return {{mode:"resident-vehicle-graph",device,wheelCount,feed,packed,resultRead,reduced,
        contactPipeline,reductionPipeline,geometryPipeline,assemblyPipeline,integrationPipeline,commitPipeline,
        contactBindGroup,reductionBindGroup,geometryBindGroup,assemblyBindGroup,integrationBindGroup,commitBindGroup,
        vehicleFeed,vehicleOutputs,controls,terrainHeights,terrainParameters,wallColliders,snapshotReads,snapshotValueCount,
        residentGraph:true,terrainReady:false,snapshotCounter:0,program,contact,reduction,integration,geometry,adapters}};
    }}catch(error){{throw new Error(`resident vehicle graph pipeline creation failed: ${{String(error?.message||error)}}`);}}
  }}
  throw new Error("resident vehicle graph artifact is incomplete");
}}
''' + r'''
function configureResidentVehicleTerrain(gpu,field,allColliders=[]){
  if(!gpu?.residentGraph||!field?.surface||field.surface.kind!=="sampled-height-field")return false;
  const surface=field.surface,heights=new Float32Array(surface.heights.map(Number)),domain=surface.domain,
    walls=allColliders.filter(collider=>!collider.surface&&collider.role!=="projectile-body"&&
      Array.isArray(collider.minimum)&&Array.isArray(collider.maximum)),
    wallValues=new Float32Array(Math.max(6,walls.length*6));
  walls.forEach((wall,index)=>wallValues.set([...wall.minimum,...wall.maximum].map(Number),index*6));
  const
    parameters=new Float32Array([surface.origin[0],surface.origin[1],surface.origin[2],surface.cell_size[0],
      surface.cell_size[1],surface.resolution[0],surface.resolution[1],domain.minimum_x,domain.maximum_x,
      domain.minimum_z,domain.maximum_z,walls.length]),oldTerrain=gpu.terrainHeights,oldWalls=gpu.wallColliders,
    terrainHeights=gpu.device.createBuffer({size:Math.max(4,heights.byteLength),
      usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),
    wallColliders=gpu.device.createBuffer({size:wallValues.byteLength,
      usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  gpu.device.queue.writeBuffer(terrainHeights,0,heights);gpu.device.queue.writeBuffer(wallColliders,0,wallValues);
  gpu.device.queue.writeBuffer(gpu.terrainParameters,0,parameters);
  gpu.terrainHeights=terrainHeights;gpu.wallColliders=wallColliders;gpu.geometryBindGroup=gpu.device.createBindGroup({
    layout:gpu.geometryPipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:terrainHeights}},{binding:1,resource:{buffer:gpu.terrainParameters}},
      {binding:2,resource:{buffer:gpu.vehicleFeed}},{binding:3,resource:{buffer:gpu.feed}},
      {binding:4,resource:{buffer:gpu.controls}},{binding:5,resource:{buffer:wallColliders}}]});gpu.terrainReady=true;
  void gpu.device.queue.onSubmittedWorkDone().then(()=>{oldTerrain.destroy();oldWalls.destroy();});return true;
}
function residentVehicleControlValues(body){
  const t=body.config.transmission,state=body.transmission||{},range=state.lowRange?Number(t.ultra_low_range_ratio):1,
    ratio=Number(state.engagedRatio||t.forward_ratios[Math.max(0,Number(state.gear||t.starting_gear)-1)]);
  if(Math.abs(Number(body.controls?.throttle||0))>.02)body.driveDirection=Math.sign(Number(body.controls.throttle));
  if(!Number.isFinite(body.driveDirection)||body.driveDirection===0)body.driveDirection=1;
  return new Float32Array([Number(body.controls?.throttle||0),Number(body.controls?.steering||0),
    Number(body.controls?.brake||0),ratio/range,Number(t.reverse_ratio),range,state.diffLock?1:0,body.driveDirection]);
}
function initializeResidentVehicleState(gpu,body){
  if(!gpu?.residentGraph)return false;const inputs=gpu.integration.inputs,values=new Float32Array(inputs.length),
    names=["front_left","front_right","rear_left","rear_right"],state={...(body.defaults||{}),
      position_x:body.position[0],position_y:body.position[1],position_z:body.position[2],
      velocity_x:body.velocity[0],velocity_y:body.velocity[1],velocity_z:body.velocity[2],
      roll:body.roll||0,pitch:body.pitch||0,yaw:body.yaw||0,roll_velocity:body.rollVelocity||0,
      pitch_velocity:body.pitchVelocity||0,yaw_velocity:body.yawVelocity||0,dt:FIXED_DT};
  names.forEach(name=>{state[`wheel_omega_${name}`]=body.wheelOmegas?.[name]||0;
    state[`compression_${name}`]=body.compressions?.[name]||0;
    state[`previous_slip_longitudinal_${name}`]=body.previousSlips?.[name]||0;});
  inputs.forEach((name,index)=>values[index]=Number(state[name]||0));gpu.device.queue.writeBuffer(gpu.vehicleFeed,0,values);
  gpu.device.queue.writeBuffer(gpu.controls,0,residentVehicleControlValues(body));body.gpuResidentInitialized=true;
  body.gpuStateDirty=false;body.gpuControlsDirty=false;return true;
}
function applyResidentVehicleSnapshot(gpu,body,values){
  const outputs=gpu.integration.outputs,output=Object.fromEntries(outputs.map((name,index)=>[name,values[index]])),
    names=["front_left","front_right","rear_left","rear_right"],packedStart=outputs.length,
    packedCount=gpu.contact.outputs.length*gpu.wheelCount,reducedStart=packedStart+packedCount,
    feedStart=reducedStart+6,contactFeed=values.subarray(feedStart),ci=Object.fromEntries(
      gpu.contact.inputs.map((name,index)=>[name,index])),co=Object.fromEntries(gpu.contact.outputs.map((name,index)=>[name,index]));
  body.position=[output.position_x_next,output.position_y_next,output.position_z_next];
  body.velocity=[output.velocity_x_next,output.velocity_y_next,output.velocity_z_next];
  body.roll=output.roll_next;body.pitch=output.pitch_next;body.yaw=output.yaw_next;
  body.rollVelocity=output.roll_velocity_next;body.pitchVelocity=output.pitch_velocity_next;body.yawVelocity=output.yaw_velocity_next;
  names.forEach(name=>{body.wheelOmegas[name]=output[`wheel_omega_${name}_next`];
    body.compressions[name]=output[`compression_${name}_next`];body.previousSlips[name]=output[`slip_longitudinal_${name}_next`];});
  body.springForces=names.map(name=>output[`spring_force_${name}`]);
  body.tractionScales=names.map(name=>output[`traction_scale_${name}`]);
  body.brakeScales=names.map(name=>output[`brake_scale_${name}`]);
  body.damperScales=names.map(name=>output[`damper_scale_${name}`]);
  body.contactAreas=names.map((_,lane)=>values[packedStart+co.contact_area*4+lane]);
  body.frictionUtilizations=names.map((_,lane)=>{const force=["x","y","z"].map(axis=>
      values[packedStart+co[`chassis_force_${axis}`]*4+lane]),normal=["x","y","z"].map(axis=>
      contactFeed[ci[`normal_${axis}`]*4+lane]),load=Math.max(0,force.reduce((sum,value,index)=>sum+value*normal[index],0)),
      tangent=Math.sqrt(Math.max(0,force.reduce((sum,value)=>sum+value*value,0)-load*load)),
      mu=contactFeed[ci.mu_static*4+lane];return tangent/Math.max(1e-5,mu*load);});
  body.contactModes=names.map((_,lane)=>{const area=body.contactAreas[lane],u=body.frictionUtilizations[lane];
    return area<=0?0:u<.78?1:u<=1.08?2:3;});
  body.powertrain={engineTorque:output.engine_torque,clutchTorque:output.clutch_torque,
    transmissionOutputTorque:output.transmission_output_torque,drivelineTorque:output.driveline_torque,
    frontDifferentialTorque:output.front_differential_torque,rearDifferentialTorque:output.rear_differential_torque,
    engineAccelerationTorque:output.engine_acceleration_torque,engineAngularAcceleration:output.engine_angular_acceleration,
    reactionTorque:[output.powertrain_reaction_torque_x,output.powertrain_reaction_torque_y,output.powertrain_reaction_torque_z],
    mountTorque:[output.engine_mount_torque_x,output.engine_mount_torque_y,output.engine_mount_torque_z],
    engineAngularSpeed:output.engine_angular_speed_next,engineRPM:output.engine_rpm};
}
function residentVehicleStep(body,dt){
  const gpu=vehicleGpu;if(!gpu?.residentGraph||!gpu.terrainReady)throw new Error("resident vehicle GPU graph is not initialized");
  if(!body.gpuResidentInitialized||body.gpuStateDirty)initializeResidentVehicleState(gpu,body);
  if(body.gpuControlsDirty){gpu.device.queue.writeBuffer(gpu.controls,0,residentVehicleControlValues(body));body.gpuControlsDirty=false;}
  if(gpu.dispatchActive)throw new Error("vehicle graph scheduler attempted a concurrent GPU dispatch");gpu.dispatchActive=true;
  try{const encoder=gpu.device.createCommandEncoder(),run=(pipeline,bindGroup,dispatch)=>{const pass=encoder.beginComputePass();
      pass.setPipeline(pipeline);pass.setBindGroup(0,bindGroup);pass.dispatchWorkgroups(...dispatch);pass.end();};
    run(gpu.geometryPipeline,gpu.geometryBindGroup,gpu.geometry.kernel.dispatch);
    run(gpu.contactPipeline,gpu.contactBindGroup,gpu.contact.kernel.dispatch);
    run(gpu.reductionPipeline,gpu.reductionBindGroup,gpu.reduction.kernel.dispatch);
    run(gpu.assemblyPipeline,gpu.assemblyBindGroup,gpu.adapters.assembly.dispatch);
    run(gpu.integrationPipeline,gpu.integrationBindGroup,gpu.integration.kernel.dispatch);
    run(gpu.commitPipeline,gpu.commitBindGroup,gpu.adapters.commit.dispatch);
    let snapshot=null;if((gpu.snapshotCounter++&3)===0)snapshot=gpu.snapshotReads.find(item=>!item.busy)||null;
    if(snapshot){snapshot.busy=true;const outputBytes=gpu.integration.outputs.length*4,
        packedBytes=gpu.contact.outputs.length*gpu.wheelCount*4,reducedBytes=24;
      encoder.copyBufferToBuffer(gpu.vehicleOutputs,0,snapshot.buffer,0,outputBytes);
      encoder.copyBufferToBuffer(gpu.packed,0,snapshot.buffer,outputBytes,packedBytes);
      encoder.copyBufferToBuffer(gpu.reduced,0,snapshot.buffer,outputBytes+packedBytes,reducedBytes);
      encoder.copyBufferToBuffer(gpu.feed,0,snapshot.buffer,outputBytes+packedBytes+reducedBytes,
        gpu.contact.inputs.length*gpu.wheelCount*4);}
    gpu.device.queue.submit([encoder.finish()]);if(snapshot)void snapshot.buffer.mapAsync(GPUMapMode.READ).then(()=>{
      const values=new Float32Array(snapshot.buffer.getMappedRange().slice(0));snapshot.buffer.unmap();snapshot.busy=false;
      applyResidentVehicleSnapshot(gpu,body,values);
    },error=>{snapshot.busy=false;postMessage({type:"vehicle-gpu-error",error:String(error?.message||error)});});
  }finally{gpu.dispatchActive=false;}
}
''' + f'''
function rotateBodyVector(body,v){{
  const cr=Math.cos(body.roll||0),sr=Math.sin(body.roll||0),cp=Math.cos(body.pitch||0),sp=Math.sin(body.pitch||0),
    cy=Math.cos(body.yaw||0),sy=Math.sin(body.yaw||0),
    r=[v[0],v[1]*cr-v[2]*sr,v[1]*sr+v[2]*cr],p=[r[0]*cp-r[1]*sp,r[0]*sp+r[1]*cp,r[2]];
  return [p[0]*cy-p[2]*sy,p[1],p[0]*sy+p[2]*cy];
}}
function c2Unit(value){{const t=Math.max(0,Math.min(1,Number(value)||0));return t*t*t*(10+t*(-15+6*t));}}
function c2Positive(value,width){{return Number(value)*c2Unit(Number(value)/Math.max(1e-9,Number(width)));}}
function c2Clamp(value,lower,upper,width){{const raised=c2Positive(Number(value)-Number(lower),width);
  return Number(upper)-c2Positive(Number(upper)-Number(lower)-raised,width);}}
function secondOrderChannel(owner,valueKey,velocityKey,target,frequency,damping,dt){{
  let value=Number(owner[valueKey]??0),velocity=Number(owner[velocityKey]??0),omega=2*Math.PI*Number(frequency),
    acceleration=omega*omega*(Number(target)-value)-2*Number(damping)*omega*velocity;
  velocity+=acceleration*dt;value+=velocity*dt;owner[valueKey]=value;owner[velocityKey]=velocity;return value;
}}
function solveMechanicalGraph(body){{
  const graph=body.config.mechanical_graph;if(!graph)return null;
  const positions=new Map(graph.nodes.map(node=>[node.identity,node.reference_position.map(Number)])),
    fixed=new Set(graph.nodes.filter(node=>node.fixed_to==="chassis").map(node=>node.identity)),
    ch=body.config.chassis,w=body.config.wheels,t=body.config.tires,s=body.config.suspension,
    names=["front_left","front_right","rear_left","rear_right"];
  names.forEach(name=>{{const prefix=`suspension.${{name}}`,hub=positions.get(`${{prefix}}.hub`),
      target=-ch.clearance-s.rest_length+(body.compressions[name]||0)+t.radius,delta=target-hub[1];
    graph.nodes.filter(node=>node.generalized_coordinate===`compression_${{name}}`).forEach(node=>
      positions.get(node.identity)[1]+=delta);}});
  const constraints=graph.edges.filter(edge=>(edge.constraint==="rigid-distance"||edge.constraint==="rigid-offset")&&
    !(fixed.has(edge.a)&&fixed.has(edge.b)));
  for(let iteration=0;iteration<8;iteration+=1){{constraints.forEach(edge=>{{const a=positions.get(edge.a),b=positions.get(edge.b);
      if(!a||!b)return;const delta=b.map((value,index)=>value-a[index]),length=Math.max(1e-8,Math.hypot(...delta)),
        error=(length-Number(edge.rest_length))/length,aFixed=fixed.has(edge.a),bFixed=fixed.has(edge.b),
        aScale=aFixed?0:bFixed?1:.5,bScale=bFixed?0:aFixed?1:.5;
      for(let axis=0;axis<3;axis+=1){{a[axis]+=delta[axis]*error*aScale;b[axis]-=delta[axis]*error*bScale;}}}});
    names.forEach(name=>{{const hub=positions.get(`suspension.${{name}}.hub`),
        target=-ch.clearance-s.rest_length+(body.compressions[name]||0)+t.radius;hub[1]+=(target-hub[1])*.38;}});}}
  names.forEach(name=>{{const hub=positions.get(`suspension.${{name}}.hub`),patch=positions.get(`suspension.${{name}}.contact_patch`);
    patch[0]=hub[0];patch[1]=hub[1]-t.radius;patch[2]=hub[2];}});return positions;
}}
function radialTireContact(body,worldHub,forwardAxis,rightAxis,steer,radius,width,travel,reach){{
  const cs=Math.cos(steer),ss=Math.sin(steer),rollingAxis=[
      forwardAxis[0]*cs+rightAxis[0]*ss,forwardAxis[1]*cs+rightAxis[1]*ss,forwardAxis[2]*cs+rightAxis[2]*ss],
    axle=[rightAxis[0]*cs-forwardAxis[0]*ss,rightAxis[1]*cs-forwardAxis[1]*ss,
      rightAxis[2]*cs-forwardAxis[2]*ss],down=rotateBodyVector(body,[0,-1,0]),
    radialAngles=[-.95,-.48,0,.48,.95],lateralFractions=[-.38,0,.38];let best=null;
  for(const angle of radialAngles){{const ca=Math.cos(angle),sa=Math.sin(angle),radial=[
      down[0]*ca+rollingAxis[0]*sa,down[1]*ca+rollingAxis[1]*sa,down[2]*ca+rollingAxis[2]*sa];
    for(const lateralFraction of lateralFractions){{const probe=[
        worldHub[0]+radial[0]*radius+axle[0]*width*lateralFraction,
        worldHub[1]+radial[1]*radius+axle[1]*width*lateralFraction,
        worldHub[2]+radial[2]*radius+axle[2]*width*lateralFraction],
      surface=contactSurfaceAt(probe[0],probe[2],body.position[1],reach,body.identity);if(!surface)continue;
      const point=[probe[0],surface.height,probe[2]],hubToSurface=worldHub.map((value,index)=>value-point[index]),
        normalDistance=hubToSurface.reduce((sum,value,index)=>sum+value*surface.normal[index],0),
        radialDistance=Math.max(1e-8,Math.hypot(...hubToSurface)),normalAlignment=normalDistance/radialDistance;
      if(normalDistance>radius+travel+.025||normalAlignment<.12)continue;
      const penetration=radius-normalDistance,score=penetration+normalAlignment*.003;
      if(!best||score>best.score)best={{...surface,point,axle,radial,angle,lateralFraction,
        normalDistance,normalAlignment,penetration,score}};
    }}
  }}return best;
}}
function wheelContactRecords(body,dt){{
  const c=body.config,w=c.wheels,ch=c.chassis,s=c.suspension,t=c.tires,
    names=["front_left","front_right","rear_left","rear_right"],
    graphPositions=solveMechanicalGraph(body),offsets=[[w.wheelbase_half_length,-w.track_half_width],
      [w.wheelbase_half_length,w.track_half_width],[-w.wheelbase_half_length,-w.track_half_width],
      [-w.wheelbase_half_length,w.track_half_width]],
    forwardAxis=rotateBodyVector(body,[1,0,0]),rightAxis=rotateBodyVector(body,[0,0,1]),
    angular=[forwardAxis[0]*(body.rollVelocity||0)+rightAxis[0]*(body.pitchVelocity||0),
      body.yawVelocity||0,forwardAxis[2]*(body.rollVelocity||0)+rightAxis[2]*(body.pitchVelocity||0)],
    steeringAngle=-(body.appliedSteering||0)*c.controls.maximum_steering_angle_degrees*Math.PI/180,
    centerOfMass=c.mechanical_graph?.load_audit?.center_of_mass||[0,0,0];
  return names.map((name,i)=>{{const o=offsets[i],localHub=graphPositions?.get(`suspension.${{name}}.hub`)||
      [o[0],-ch.clearance-s.rest_length+t.radius,o[1]],
    hubOffset=rotateBodyVector(body,localHub),worldHub=body.position.map((value,index)=>value+hubOffset[index]),
    coiloverA=graphPositions?.get(`suspension.${{name}}.coilover_chassis`),
    coiloverB=graphPositions?.get(`suspension.${{name}}.lower_ball_joint`),
    coiloverDelta=coiloverA&&coiloverB?coiloverB.map((value,index)=>value-coiloverA[index]):[0,-1,0],
    linkageMotionRatio=Math.max(.25,Math.min(1.25,Math.abs(coiloverDelta[1])/Math.max(1e-8,Math.hypot(...coiloverDelta)))),
    wx=worldHub[0],wz=worldHub[2],reach=ch.clearance+s.rest_length+s.travel+t.radius+.08,
    steer=i<2?steeringAngle:0,candidate=radialTireContact(body,worldHub,forwardAxis,rightAxis,steer,
      Number(t.radius),Number(t.width),Number(s.travel),reach),n=candidate?.normal||[0,1,0],
    suspensionDown=rotateBodyVector(body,[0,-1,0]),alignment=-(suspensionDown[0]*n[0]+suspensionDown[1]*n[1]+
      suspensionDown[2]*n[2]),surfacePoint=candidate?.point||[wx,0,wz],originToSurface=surfacePoint.map((value,index)=>
      value-body.position[index]),distanceAlongSuspension=originToSurface.reduce((sum,value,index)=>
      sum+value*suspensionDown[index],0),geometricCompression=Math.max(0,Math.min(s.travel,
      ch.clearance+s.rest_length-distanceAlongSuspension)),supportWeight=candidate&&distanceAlongSuspension>0
        ?c2Unit(geometricCompression/.025)*c2Unit((alignment-.18)/.34):0,
      supported=supportWeight>1e-6,support=supported?candidate:null,
    radialOut=support?surfacePoint.map((value,index)=>(value-worldHub[index])/Math.max(1e-8,
      Math.hypot(...surfacePoint.map((component,axis)=>component-worldHub[axis])))):suspensionDown,
    axle=candidate?.axle||rightAxis,rollingRaw=[axle[1]*radialOut[2]-axle[2]*radialOut[1],
      axle[2]*radialOut[0]-axle[0]*radialOut[2],axle[0]*radialOut[1]-axle[1]*radialOut[0]],
    headingNormal=rollingRaw[0]*n[0]+rollingRaw[1]*n[1]+rollingRaw[2]*n[2],
    tangentRaw=rollingRaw.map((value,axis)=>value-headingNormal*n[axis]),
    tangentLength=Math.max(1e-8,Math.hypot(...tangentRaw)),forward=tangentRaw.map(value=>value/tangentLength),
    right=[forward[1]*n[2]-forward[2]*n[1],forward[2]*n[0]-forward[0]*n[2],
      forward[0]*n[1]-forward[1]*n[0]],
    worldCenterOfMass=rotateBodyVector(body,centerOfMass),attachment=surfacePoint.map((value,index)=>
      value-body.position[index]-worldCenterOfMass[index]),
    pointVelocity=[body.velocity[0]+angular[1]*attachment[2]-angular[2]*attachment[1],
      body.velocity[1]+angular[2]*attachment[0]-angular[0]*attachment[2],
      body.velocity[2]+angular[0]*attachment[1]-angular[1]*attachment[0]],
    forwardSpeed=pointVelocity.reduce((sum,value,axis)=>sum+value*forward[axis],0),
    lateralSpeed=pointVelocity.reduce((sum,value,axis)=>sum+value*right[axis],0);
    return {{dt,support:supportWeight,hub_height:worldHub[1],hub_velocity_y:pointVelocity[1],
      chassis_velocity_y:body.velocity[1],roll_velocity:body.rollVelocity||0,pitch_velocity:body.pitchVelocity||0,
      wheelbase_half_length:w.wheelbase_half_length,track_half_width:w.track_half_width,
      corner_front_sign:i<2?1:-1,corner_side_sign:i%2===0?-1:1,
      geometric_compression:geometricCompression,suspension_alignment:alignment,
      previous_compression:body.compressions[name]||0,surface_height:support?.height||0,
      contact_x:surfacePoint[0],contact_z:surfacePoint[2],runtime_part_id:support?.runtimePartId||0,
      radial_contact_angle:support?.angle||0,radial_lateral_fraction:support?.lateralFraction||0,
      normal_x:n[0],normal_y:n[1],normal_z:n[2],forward_x:forward[0],forward_y:forward[1],forward_z:forward[2],
      right_x:right[0],right_y:right[1],right_z:right[2],slip_longitudinal:forwardSpeed-(body.wheelOmegas?.[name]||0)*t.radius,
      slip_lateral:lateralSpeed,attachment_x:attachment[0],attachment_y:attachment[1],attachment_z:attachment[2],
      corner_weight:c.mass*Math.abs(c.world.gravity)*c.mass_distribution[name],suspension_rest_length:s.rest_length,
      chassis_clearance:ch.clearance,suspension_travel:s.travel,spring_stiffness:s.stiffness,
      linkage_motion_ratio:linkageMotionRatio,
      pneumatic_compression_damping:s.pneumatic_compression_damping,
      pneumatic_rebound_damping:s.pneumatic_rebound_damping,pneumatic_efficiency:s.pneumatic_efficiency,
      active_damping_minimum_scale:s.active_damping_minimum_scale,
      active_damping_maximum_scale:s.active_damping_maximum_scale,
      active_damping_body_velocity_gain_s_per_m:s.active_damping_body_velocity_gain_s_per_m,
      active_damping_rebound_release_gain_s_per_m:s.active_damping_rebound_release_gain_s_per_m,
      maximum_compression_speed:s.maximum_compression_speed,
      tire_pressure:t.pressure_pa,minimum_contact_area:t.minimum_contact_area,
      maximum_contact_area:t.maximum_contact_area,mu_static:t.static_friction,mu_kinetic:t.kinetic_friction,
      load_sensitivity:t.load_sensitivity,longitudinal_stiffness:t.longitudinal_stiffness,
      lateral_stiffness:t.lateral_stiffness,slip_transition_speed:t.slip_transition_speed}};}});
}}
function resolveVehicleSolidContact(body,dt){{
  const layer=body.config.solid_contact,hit=contact(body.position,body.radius,body.identity);if(!layer||!hit)return;
  const normal=[-hit.n[0],0,-hit.n[1]],penetration=Math.max(0,hit.d),
    correction=Math.min(layer.maximum_correction_speed*dt,penetration*layer.penetration_bias);
  body.position[0]+=normal[0]*correction;body.position[2]+=normal[2]*correction;
  const vn=body.velocity[0]*normal[0]+body.velocity[2]*normal[2];if(vn>=0)return;
  const tx=body.velocity[0]-vn*normal[0],tz=body.velocity[2]-vn*normal[2],speed=Math.hypot(tx,tz),
    normalDelta=-(1+layer.restitution)*vn,staticCapacity=layer.static_friction*normalDelta;
  let tangentScale=1;
  if(speed<=staticCapacity)tangentScale=0;
  else if(speed>1e-8)tangentScale=Math.max(0,1-layer.kinetic_friction*normalDelta/speed);
  body.velocity[0]=tx*tangentScale-vn*layer.restitution*normal[0];
  body.velocity[2]=tz*tangentScale-vn*layer.restitution*normal[2];
  body.solidContactMode=tangentScale===0?1:2;body.contactRuntimePartId=hit.runtimePartId||body.contactRuntimePartId;
}}
function vehicleCageContactWrench(body,dt){{
  const c=body.config,layer=c.solid_contact,graph=c.mechanical_graph,
    nodes=graph?.nodes?.filter(node=>node.kind==="roll-cage-node")||[],
    centerOfMass=graph?.load_audit?.center_of_mass||[0,0,0],force=[0,0,0],torque=[0,0,0];
  const nodePositions=new Map((graph?.nodes||[]).map(node=>[node.identity,node.reference_position.map(Number)])),
    barMidpoints=(graph?.edges||[]).filter(edge=>edge.identity.startsWith("cage.")).map(edge=>{{
      const a=nodePositions.get(edge.a),b=nodePositions.get(edge.b);return a&&b?a.map((value,index)=>(value+b[index])*.5):null;}}).filter(Boolean),
    points=[...nodes.map(node=>node.reference_position.map(Number)),...barMidpoints];
  let count=0,contactId=0;if(!points.length)return{{force,torque,count,contactId}};
  const forward=rotateBodyVector(body,[1,0,0]),right=rotateBodyVector(body,[0,0,1]),
    angular=[forward[0]*(body.rollVelocity||0)+right[0]*(body.pitchVelocity||0),
      body.yawVelocity||0,forward[2]*(body.rollVelocity||0)+right[2]*(body.pitchVelocity||0)],
    radius=Number(layer.cage_contact_radius),share=Math.max(1,points.reduce((sum,local)=>{{
      const offset=rotateBodyVector(body,local),point=body.position.map(
        (value,index)=>value+offset[index]),surface=contactSurfaceAt(point[0],point[2],body.position[1],2.5,body.identity);
      return sum+(surface&&surface.height+radius-point[1]>0?1:0);}},0));
  for(const local of points){{const offset=rotateBodyVector(body,local),
      attachment=rotateBodyVector(body,local.map((value,index)=>value-Number(centerOfMass[index]||0))),
      point=body.position.map((value,index)=>value+offset[index]),surface=contactSurfaceAt(
        point[0],point[2],body.position[1],2.5,body.identity);if(!surface)continue;
    const penetration=surface.height+radius-point[1];if(penetration<=0)continue;
    const n=surface.normal,pointVelocity=[
      body.velocity[0]+angular[1]*attachment[2]-angular[2]*attachment[1],
      body.velocity[1]+angular[2]*attachment[0]-angular[0]*attachment[2],
      body.velocity[2]+angular[0]*attachment[1]-angular[1]*attachment[0]],
      normalSpeed=pointVelocity.reduce((sum,value,index)=>sum+value*n[index],0),
      normalForce=c2Clamp(Number(layer.cage_contact_stiffness)*penetration-
        Number(layer.cage_contact_damping)*normalSpeed,0,Number(layer.cage_contact_maximum_force),80),
      tangent=pointVelocity.map((value,index)=>value-normalSpeed*n[index]),speed=Math.hypot(...tangent),
      stopRequest=Number(c.mass)*speed/(Math.max(dt,1e-6)*share),
      staticLimit=Number(layer.cage_static_friction)*normalForce,
      kineticLimit=Number(layer.cage_kinetic_friction)*normalForce,
      kineticBlend=c2Unit((stopRequest-staticLimit)/Math.max(1,staticLimit*.18)),
      frictionMagnitude=(1-kineticBlend)*stopRequest+kineticBlend*kineticLimit,
      smoothSpeed=Math.sqrt(speed*speed+1e-8),
      f=n.map((value,index)=>value*normalForce-tangent[index]/smoothSpeed*frictionMagnitude);
    for(let axis=0;axis<3;axis+=1)force[axis]+=f[axis];
    torque[0]+=attachment[1]*f[2]-attachment[2]*f[1];
    torque[1]+=attachment[2]*f[0]-attachment[0]*f[2];
    torque[2]+=attachment[0]*f[1]-attachment[1]*f[0];
    count+=1;contactId=surface.runtimePartId||contactId;
  }}return{{force,torque,count,contactId}};
}}
function resolveVehicleCagePenetration(body,dt){{
  const c=body.config,layer=c.solid_contact,graph=c.mechanical_graph,nodes=graph?.nodes?.filter(
    node=>node.kind==="roll-cage-node")||[],nodePositions=new Map((graph?.nodes||[]).map(node=>[
      node.identity,node.reference_position.map(Number)])),barMidpoints=(graph?.edges||[]).filter(edge=>
      edge.identity.startsWith("cage.")).map(edge=>{{const a=nodePositions.get(edge.a),b=nodePositions.get(edge.b);
        return a&&b?a.map((value,index)=>(value+b[index])*.5):null;}}).filter(Boolean),
    points=[...nodes.map(node=>node.reference_position.map(Number)),...barMidpoints],
    radius=Number(layer.cage_contact_radius);let resolved=0;
  for(let iteration=0;iteration<3;iteration+=1){{let deepest=null;
    for(const local of points){{const offset=rotateBodyVector(body,local),
        point=body.position.map((value,index)=>value+offset[index]),surface=contactSurfaceAt(
          point[0],point[2],body.position[1],2.5,body.identity),penetration=surface?surface.height+radius-point[1]:0;
      if(penetration>0&&(!deepest||penetration>deepest.penetration))deepest={{penetration,normal:surface.normal,
        runtimePartId:surface.runtimePartId||0}};}}
    if(!deepest)break;const correction=Math.min(deepest.penetration,Number(layer.maximum_correction_speed)*dt);
    for(let axis=0;axis<3;axis+=1)body.position[axis]+=deepest.normal[axis]*correction;
    const inward=body.velocity.reduce((sum,value,index)=>sum+value*deepest.normal[index],0);
    if(inward<0)for(let axis=0;axis<3;axis+=1)body.velocity[axis]-=deepest.normal[axis]*inward*(1+Number(layer.restitution));
    body.contactRuntimePartId=deepest.runtimePartId||body.contactRuntimePartId;resolved+=1;
  }}body.cageContactCount=resolved;return resolved;
}}
function resolveVehicleSuspensionTravelStop(body){{
  const c=body.config,w=c.wheels,ch=c.chassis,s=c.suspension,
    offsets=[[w.wheelbase_half_length,-w.track_half_width],[w.wheelbase_half_length,w.track_half_width],
      [-w.wheelbase_half_length,-w.track_half_width],[-w.wheelbase_half_length,w.track_half_width]],
    reach=ch.clearance+s.rest_length+s.travel+.08;let minimumBodyY=-Infinity,contactId=0;
  for(const offset of offsets){{const attachment=rotateBodyVector(body,[offset[0],-ch.clearance,offset[1]]),
    support=contactSurfaceAt(body.position[0]+attachment[0],body.position[2]+attachment[2],
      body.position[1],reach,body.identity),down=rotateBodyVector(body,[0,-1,0]),alignment=support?-(
      down[0]*support.normal[0]+down[1]*support.normal[1]+down[2]*support.normal[2]):0;
    if(!support||alignment<=.18)continue;
    minimumBodyY=Math.max(minimumBodyY,support.height+s.rest_length-s.travel-attachment[1]);
    contactId=support.runtimePartId||contactId;}}
  if(!Number.isFinite(minimumBodyY)||body.position[1]>=minimumBodyY)return false;
  const penetration=minimumBodyY-body.position[1],engagement=c2Unit(penetration/.025),
    correction=c2Clamp(penetration,0,Number(s.maximum_compression_speed)*dt,.004);
  body.suspensionStopPenetration=penetration;body.position[1]+=correction*engagement;
  if(body.velocity[1]<0)body.velocity[1]+=c2Positive(-body.velocity[1],.18)*engagement;
  body.contactRuntimePartId=contactId||body.contactRuntimePartId;return true;
}}
function resolveWorldBottom(body,previousPosition){{
  const previous=Array.isArray(previousPosition)?previousPosition:[body.position[0],Number(previousPosition),body.position[2]],
    supportOffset=body.kind==="vehicle"?Math.max(.02,Number(body.config?.chassis?.clearance||0)):
      Math.max(.001,Number(body.radius||0)),sampled=colliders.find(c=>{{const s=c.surface,d=s?.domain;
        return s?.kind==="sampled-height-field"&&body.position[0]>=d.minimum_x&&body.position[0]<=d.maximum_x&&
          body.position[2]>=d.minimum_z&&body.position[2]<=d.maximum_z;}});
  if(sampled){{const previousSample=sampleSurface(sampled.surface,previous[0],previous[2]),
      nextSample=sampleSurface(sampled.surface,body.position[0],body.position[2]),
      previousBase=previous[1]-supportOffset,nextBase=body.position[1]-supportOffset,
      crossed=previousBase>=previousSample.height-.012&&nextBase<=nextSample.height+.006;
    if(crossed&&body.velocity[1]<=0){{body.position[1]=nextSample.height+supportOffset;
      body.velocity[1]=0;body.contactRuntimePartId=sampled.runtimePartId||body.contactRuntimePartId;return true;}}}}
  const top=Number(worldBottom?.top_y??0),thickness=Math.max(.25,Number(worldBottom?.thickness??8)),
    surfaceHeight=sampled?
      sampleSurface(sampled.surface,body.position[0],body.position[2]).height:null,
    guardDepth=Math.max(.25,Number(worldBottom?.sampled_surface_guard_depth??.75)),
    rejectionTop=surfaceHeight===null?top+supportOffset:surfaceHeight-guardDepth,
    recoveryHeight=surfaceHeight===null?top+supportOffset:surfaceHeight+supportOffset,
    bottom=rejectionTop-thickness,boundary=rejectionTop,nextY=body.position[1],
    swept=previous[1]>=boundary&&nextY<bottom,contained=nextY<boundary&&nextY>=bottom;
  if(!swept&&!contained)return false;
  body.position[1]=recoveryHeight;if(body.velocity[1]<0)body.velocity[1]=0;
  body.bottomRejected=true;body.contactRuntimePartId=0;return true;
}}
function updateVehicleTransmission(body,dt){{
  const c=body.config,t=c.transmission,p=c.powertrain,d=c.drivetrain,w=c.wheels,tire=c.tires,
    ratios=t.forward_ratios.map(Number),maximum=ratios.length,throttle=Number(body.appliedThrottle||0),
    drivenNames=["front_left","front_right","rear_left","rear_right"],
    wheelSpeed=drivenNames.reduce((sum,name)=>sum+Math.abs(body.wheelOmegas?.[name]||0),0)/drivenNames.length,
    roadSpeed=Math.hypot(body.velocity[0],body.velocity[2]),state=body.transmission||{{
      mode:t.mode_default,gear:Number(t.starting_gear),shiftAge:Number(t.minimum_shift_interval_s),reason:"initial-second",
      engagedRatio:ratios[Number(t.starting_gear)-1],ratioVelocity:0,downshiftDemand:0,downshiftDemandVelocity:0,
      lowRange:false,diffLock:false}};
  if(!Number.isFinite(state.shiftAge))state.shiftAge=Number(t.minimum_shift_interval_s);
  if(!Number.isFinite(state.engagedRatio))state.engagedRatio=ratios[Number(t.starting_gear)-1];
  if(!Number.isFinite(state.ratioVelocity))state.ratioVelocity=0;
  if(!Number.isFinite(state.downshiftDemand))state.downshiftDemand=0;
  if(!Number.isFinite(state.downshiftDemandVelocity))state.downshiftDemandVelocity=0;
  state.shiftAge+=dt;state.gear=Math.max(1,Math.min(maximum,Math.round(state.gear||t.starting_gear)));
  const rangeMultiplier=state.lowRange?Number(t.ultra_low_range_ratio):1,
    indicated=p.brake_mean_effective_pressure_pa*(p.displacement_liters/1000)/(4*Math.PI)*p.combustion_efficiency,
    demand=(body.longitudinalForces||[0,0,0,0]).reduce((sum,value)=>sum+Math.abs(value),0)*tire.radius+
      4*d.rolling_resistance_torque_nm,
    available=gear=>Math.abs(throttle)*indicated*ratios[gear-1]*rangeMultiplier*p.final_drive_ratio*
      p.clutch_efficiency*p.driveline_efficiency*d.transfer_case_efficiency,
    reserve=gear=>available(gear)/Math.max(1,demand),canShift=state.shiftAge>=t.minimum_shift_interval_s,
    gear=state.gear,speedThreshold=Number(t.downshift_wheel_speed_rad_s[gear-1]||0),
    speedNeed=c2Unit((speedThreshold-wheelSpeed)/Math.max(1,speedThreshold*.35)),
    reserveNeed=c2Unit((Number(t.downshift_torque_reserve)-reserve(gear))/.32),
    launchNeed=c2Unit((Number(t.crawler_entry_speed_m_s)-roadSpeed)/Math.max(.2,Number(t.crawler_entry_speed_m_s))) *
      c2Unit((Math.abs(throttle)-.22)/.42),
    downshiftTarget=gear>1?1-(1-speedNeed)*(1-reserveNeed)*(1-launchNeed):0;
  secondOrderChannel(state,"downshiftDemand","downshiftDemandVelocity",downshiftTarget,
    t.downshift_demand_frequency_hz,t.downshift_demand_damping_ratio,dt);
  if(state.mode==="automatic"&&throttle>0&&canShift){{const gear=state.gear;
    if(gear===1){{if(wheelSpeed>t.upshift_wheel_speed_rad_s[0]&&reserve(2)>=t.upshift_torque_reserve){{
      state.gear=2;state.shiftAge=0;state.reason="crawler-clear";}}}}
    else if(gear<maximum&&wheelSpeed>t.upshift_wheel_speed_rad_s[gear-1]&&
      reserve(gear+1)>=t.upshift_torque_reserve){{state.gear+=1;state.shiftAge=0;state.reason="next-ratio-has-reserve";}}
    else if(gear>1&&state.downshiftDemand>=Number(t.downshift_commit_level)){{
      state.gear-=1;state.shiftAge=0;state.reason=gear===2?"crawler-demand-integrated":"downshift-demand-integrated";
      state.downshiftDemand=0;state.downshiftDemandVelocity=0;}}
  }}
  const rangeRatio=rangeMultiplier,targetRatio=ratios[state.gear-1]*rangeRatio;
  state.engagedRatio=secondOrderChannel(state,"engagedRatio","ratioVelocity",
    targetRatio,t.ratio_response_frequency_hz,t.ratio_response_damping_ratio,dt);
  state.displayGear=throttle<0?-1:state.gear;state.torqueReserve=reserve(state.gear);
  body.transmission=state;return{{forwardRatio:state.engagedRatio/rangeRatio,
    reverseRatio:Number(t.reverse_ratio),transferCaseRatio:rangeRatio}};
}}
function applyPendingVehicleCommands(body){{
  if(body.pendingControls){{body.controls=body.pendingControls;body.pendingControls=null;body.gpuControlsDirty=true;}}
  if(body.pendingRecovery){{const lift=Number(body.pendingRecovery.lift||.5);body.pendingRecovery=null;
    body.position[1]+=Math.max(.1,lift);body.roll=0;body.pitch=0;body.rollVelocity=0;body.pitchVelocity=0;
    body.yawVelocity=0;body.velocity[1]=Math.max(0,body.velocity[1]);body.gpuStateDirty=true;}}
  if(body.pendingRespawn){{const command=body.pendingRespawn;body.pendingRespawn=null;
    body.position=[...command.position];body.velocity=[0,0,0];body.roll=Number(command.roll||0);
    body.pitch=Number(command.pitch||0);body.yaw=Number(command.yaw||0);
    body.rollVelocity=0;body.pitchVelocity=0;body.yawVelocity=0;
    body.wheelOmegas={{front_left:0,front_right:0,rear_left:0,rear_right:0}};
    body.previousSlips={{front_left:0,front_right:0,rear_left:0,rear_right:0}};body.gpuStateDirty=true;}}
  if(body.pendingTransmission){{const command=body.pendingTransmission;body.pendingTransmission=null;
    const t=body.config.transmission,state=body.transmission||{{mode:t.mode_default,gear:t.starting_gear,
      shiftAge:0,reason:"control"}};
    if(command.mode==="automatic"){{state.mode="automatic";state.reason="driver-auto";}}
    if(typeof command.lowRange==="boolean"){{state.lowRange=command.lowRange;state.ratioVelocity=0;
      state.reason=command.lowRange?"driver-ultra-low":"driver-high-range";}}
    if(typeof command.diffLock==="boolean"){{state.diffLock=command.diffLock;
      state.reason=command.diffLock?"driver-diff-lock":"driver-diff-open";}}
    if(Number.isFinite(command.gearDelta)){{state.mode="manual";state.gear=Math.max(1,
      Math.min(t.forward_ratios.length,Math.round(state.gear+Number(command.gearDelta))));
      state.shiftAge=0;state.reason="driver-manual";}}body.transmission=state;body.gpuControlsDirty=true;}}
}}
function stepWorld(body,dt){{
  const previousPosition=[...body.position];
  const previousVelocity=[...body.velocity];
  const c=contact(body.position,body.radius,body.identity), x={{position_x:body.position[0],position_y:body.position[1],position_z:body.position[2],velocity_x:body.velocity[0],velocity_y:body.velocity[1],velocity_z:body.velocity[2],dt,obstacle_active:c?1:0,obstacle_normal_x:c?.n[0]||0,obstacle_normal_z:c?.n[1]||0,obstacle_plane:c?.q||0,radius:body.radius,...body.overrides}};
  const memory=new Float64Array(instance.exports.memory.buffer);
  abi.input_names.forEach((name,i)=>memory[abi.input_offsets[i]/8]=x[name]??parameters[name]??0);
  instance.exports[abi.entrypoint](0); const out={{}};
  abi.output_names.forEach((name,i)=>out[name]=memory[abi.output_offsets[i]/8]);
  const nextPosition=[out.position_x_next,out.position_y_next,out.position_z_next];
  const support=topSupport(previousPosition,nextPosition,body.radius,body.identity),
    terrainSupport=sampledTerrainSupport(previousPosition,nextPosition,body.radius,body.identity);
  body.position[0]=nextPosition[0];body.position[1]=terrainSupport?terrainSupport.height+body.radius:
    support?support.top+body.radius:nextPosition[1];body.position[2]=nextPosition[2];
  const force=body.force||[0,0,0],moment=body.moment||[0,0,0],inverseMass=Number(body.inverseMass??parameters.inverse_mass??0),
    inverseInertia=body.inverseInertia||[inverseMass,inverseMass,inverseMass],angular=body.angularVelocity||[0,0,0];
  body.velocity[0]=out.velocity_x_next+dt*force[0]*inverseMass;
  body.velocity[1]=out.velocity_y_next+dt*force[1]*inverseMass;
  body.velocity[2]=out.velocity_z_next+dt*force[2]*inverseMass;
  if(terrainSupport){{const inward=body.velocity.reduce((sum,value,index)=>sum+value*terrainSupport.normal[index],0);
    if(inward<0)for(let axis=0;axis<3;axis+=1)body.velocity[axis]-=terrainSupport.normal[axis]*inward;}}
  for(let axis=0;axis<3;axis+=1)angular[axis]+=dt*moment[axis]*Number(inverseInertia[axis]||0);
  body.angularVelocity=angular;body.roll=(body.roll||0)+dt*angular[0];
  body.pitch=(body.pitch||0)+dt*angular[1];body.yaw=(body.yaw||0)+dt*angular[2];
  body.rollVelocity=angular[0];body.pitchVelocity=angular[1];body.yawVelocity=angular[2];
  resolveWorldBottom(body,previousPosition);
  body.accelerationMagnitude=Math.hypot(body.velocity[0]-previousVelocity[0],
    body.velocity[1]-previousVelocity[1],body.velocity[2]-previousVelocity[2])/Math.max(dt,1e-6);
  body.contactRuntimePartId=terrainSupport?.runtimePartId||support?.runtimePartId||c?.runtimePartId||0;
}}
function coastStep(body,dt){{
  const next=body.position.map((value,axis)=>value+body.velocity[axis]*dt),r=body.radius;
  const value=name=>body.overrides?.[name]??parameters[name];
  if(next[0]<value("minimum_x")+r||next[0]>value("maximum_x")-r||
     next[1]<value("minimum_y")+r||next[1]>value("maximum_y")-r||
     next[2]<value("minimum_z")+r||next[2]>value("maximum_z")-r||
     contact(next,r,body.identity))return false;
  body.position=next;body.contactRuntimePartId=0;return true;
}}
function publishSnapshot(now){{
  if(snapshotInFlight||!snapshotBuffers.length)return;
  const buffer=snapshotBuffers.pop(), values=new Float64Array(buffer); values.fill(0);
  for(const body of bodies.values()){{const o=body.slot*SNAPSHOT_STRIDE;if(o<0||o+SNAPSHOT_STRIDE>values.length)continue;
    values[o]=1;values[o+1]=body.generation;values[o+2]=body.position[0];values[o+3]=body.position[1];values[o+4]=body.position[2];
    values[o+5]=body.velocity[0];values[o+6]=body.velocity[1];values[o+7]=body.velocity[2];
    values[o+8]=body.roll||0;values[o+9]=body.pitch||0;values[o+10]=body.yaw||0;
    values[o+11]=body.rollVelocity||0;values[o+12]=body.pitchVelocity||0;values[o+13]=body.yawVelocity||0;
    values[o+14]=body.contactRuntimePartId||0;
    const springs=body.springForces||[0,0,0,0],areas=body.contactAreas||[0,0,0,0],
      utilization=body.frictionUtilizations||[0,0,0,0],modes=body.contactModes||[0,0,0,0],
      compression=Object.values(body.compressions||{{}});
    for(let i=0;i<4;i++){{values[o+15+i]=springs[i]||0;values[o+19+i]=areas[i]||0;
      values[o+23+i]=utilization[i]||0;values[o+27+i]=modes[i]||0;
      values[o+31+i]=compression[i]||0;values[o+35+i]=body.wheelOmegas?.[Object.keys(body.compressions||{{}})[i]]||0;}}
    for(let i=0;i<4;i++){{values[o+39+i]=body.tractionScales?.[i]??1;values[o+43+i]=body.brakeScales?.[i]??1;}}
    const powertrain=body.powertrain||{{}},reaction=powertrain.reactionTorque||[0,0,0],mount=powertrain.mountTorque||[0,0,0];
    values[o+47]=powertrain.engineTorque||0;values[o+48]=powertrain.clutchTorque||0;
    values[o+49]=powertrain.transmissionOutputTorque||0;values[o+50]=powertrain.drivelineTorque||0;
    values[o+51]=powertrain.frontDifferentialTorque||0;values[o+52]=powertrain.rearDifferentialTorque||0;
    values[o+53]=powertrain.engineAccelerationTorque||0;values[o+54]=powertrain.engineAngularAcceleration||0;
    for(let i=0;i<3;i++){{values[o+55+i]=reaction[i]||0;values[o+58+i]=mount[i]||0;}}
    values[o+61]=body.transmission?.gear||0;values[o+62]=body.transmission?.displayGear||0;
    values[o+63]=body.transmission?.mode==="automatic"?1:0;values[o+64]=body.controlGeneration||0;
    for(let i=0;i<4;i++)values[o+65+i]=body.damperScales?.[i]??1;
    values[o+69]=body.powertrain?.engineAngularSpeed||0;values[o+70]=body.powertrain?.engineRPM||0;
  }}
  snapshotInFlight=true;postMessage({{type:"snapshot-buffer",sequence:++sequence,time:now,buffer}},[buffer]);
}}
function setEngineStage(stage,reason){{
  if(timer){{clearInterval(timer);timer=null;}}
  engineStage=stage;idleTicks=0;coastTicks=0;previous=performance.now();
  if(instance&&stage!=="asleep")timer=setInterval(tick,
    stage==="kinematic-coast"?1000/30:{interval!r});
  postMessage({{type:"engine-state",stage,sleeping:stage==="asleep",reason,members:bodies.size}});
}}
function armEngine(reason){{
  idleTicks=0;coastTicks=0;
  if(!instance)return;
  if(engineStage!=="full-dynamics"||!timer)setEngineStage("full-dynamics",reason);
}}
function tick(){{
  if(!instance||tickInFlight)return;tickInFlight=true;try{{const now=performance.now(),
    dt=engineStage==="kinematic-coast"?1/30:FIXED_DT;previous=now;
  if(engineStage==="kinematic-coast"){{
    let clear=true,moving=false;
    for(const body of bodies.values()){{
      moving=moving||Math.hypot(body.velocity[0],body.velocity[1],body.velocity[2])>.012;
      if(!coastStep(body,dt))clear=false;
    }}
    if(!clear){{setEngineStage("full-dynamics","coast-contact-or-boundary");return;}}
    publishSnapshot(now);idleTicks=moving?0:idleTicks+1;
    if(idleTicks>=45)setEngineStage("asleep","kinematic-coast-became-still");
    return;
  }}
  let integrationOnly=true,moving=false,constantVelocity=true;
  for(const body of bodies.values()){{
    if(body.kind==="vehicle"){{
      if(!vehicleGpu?.residentGraph){{body.accelerationMagnitude=0;continue;}}
      if(!vehicleGpu.terrainReady){{body.accelerationMagnitude=0;continue;}}
      applyPendingVehicleCommands(body);try{{residentVehicleStep(body,dt);if(lastVehicleGpuError)
        postMessage({{type:"vehicle-gpu-recovered"}});lastVehicleGpuError=null;}}
      catch(error){{const message=String(error?.message||error);if(message!==lastVehicleGpuError){{
        lastVehicleGpuError=message;postMessage({{type:"vehicle-gpu-error",error:message}});}}
        body.accelerationMagnitude=0;continue;}}
    }}else{{
      stepWorld(body,dt);
    }}
    const speed=Math.hypot(body.velocity[0],body.velocity[1],body.velocity[2]);
    if(speed>.012){{integrationOnly=false;moving=true;}}
    if((body.accelerationMagnitude||0)>.001||body.kind==="vehicle")constantVelocity=false;
  }}
  publishSnapshot(now);
  idleTicks=integrationOnly?idleTicks+1:0;
  coastTicks=moving&&constantVelocity?coastTicks+1:0;
  if(idleTicks>=180)setEngineStage("asleep","integration-only-quiescence");
  else if(coastTicks>=12)setEngineStage("kinematic-coast","constant-velocity-no-acceleration");
  }}finally{{tickInFlight=false;}}
}}
onmessage=async event=>{{const m=event.data||{{}};
  if(m.type==="init"){{const made=await WebAssembly.instantiate(m.wasm,{{}});instance=made.instance;abi={{...m.abi,entrypoint:m.entrypoint}};parameters=m.parameters||{{}};
    worldBottom={{...worldBottom,...(m.worldBottom||{{}})}};
    snapshotCapacity=m.snapshotCapacity||128;snapshotBuffers=Array.from({{length:SNAPSHOT_POOL_SIZE}},()=>new ArrayBuffer(snapshotCapacity*SNAPSHOT_STRIDE*8));
    previous=performance.now();postMessage({{type:"ready",snapshotCapacity,stride:SNAPSHOT_STRIDE,poolSize:SNAPSHOT_POOL_SIZE,
      vehicleCompute:"resident-webgpu-pending"}});armEngine("initialized");
    // Shader compilation is an initialization dependency of the vehicle graph,
    // not of the general world loop.  Let balls and the player advance while
    // the purpose-built graph is validated, then atomically publish it.
    initializeVehicleGpu(m.vehicleWebgpu).then(gpu=>{{vehicleGpu=gpu;
      const field=colliders.find(collider=>collider.surface?.kind==="sampled-height-field");
      if(field)configureResidentVehicleTerrain(vehicleGpu,field,colliders);
      for(const body of bodies.values())if(body.kind==="vehicle")initializeResidentVehicleState(vehicleGpu,body);
      postMessage({{type:"vehicle-gpu-recovered"}});armEngine("vehicle-graph-ready");
    }}).catch(error=>{{vehicleGpu=null;postMessage({{type:"vehicle-gpu-error",error:String(error?.message||error)}});}});}}
  else if(m.type==="upsert"){{const body={{force:[0,0,0],moment:[0,0,0],angularVelocity:[0,0,0],
    ...m.body,position:[...m.body.position],velocity:[...m.body.velocity],force:[...(m.body.force||[0,0,0])],
    moment:[...(m.body.moment||[0,0,0])],angularVelocity:[...(m.body.angularVelocity||[0,0,0])],
    wheelOmegas:{{front_left:0,front_right:0,rear_left:0,rear_right:0,...m.body.wheelOmegas}},
    previousSlips:{{front_left:0,front_right:0,rear_left:0,rear_right:0,...m.body.previousSlips}}}};
    bodies.set(m.body.identity,body);if(body.kind==="vehicle"&&vehicleGpu?.residentGraph)initializeResidentVehicleState(vehicleGpu,body);
    armEngine("body-upsert");}}
  else if(m.type==="remove") bodies.delete(m.identity);
  else if(m.type==="control"){{const b=bodies.get(m.identity);if(b&&m.position){{b.position[0]=m.position[0];b.position[2]=m.position[1];b.controlGeneration=m.generation||0;armEngine("control");}}}}
  else if(m.type==="vehicle-control"){{const b=bodies.get(m.identity);if(b){{b.pendingControls={{
    throttle:m.throttle||0,steering:m.steering||0,brake:m.brake||0}};armEngine("vehicle-control");}}}}
  else if(m.type==="vehicle-recover"){{const b=bodies.get(m.identity);if(b){{b.pendingRecovery={{lift:m.lift}};
    armEngine("vehicle-recover");}}}}
  else if(m.type==="vehicle-respawn"){{const b=bodies.get(m.identity);if(b&&Array.isArray(m.position)){{
    b.pendingRespawn={{position:[...m.position],roll:m.roll,pitch:m.pitch,yaw:m.yaw}};armEngine("vehicle-respawn");}}}}
  else if(m.type==="vehicle-dyno"){{const b=bodies.get(m.identity),p=b?.powertrain||{{}},springs=b?.springForces||[];
    postMessage({{type:"vehicle-dyno-result",result:{{requestId:m.requestId||0,status:"telemetry",
      compute:vehicleGpu?.residentGraph?"resident-webgpu-graph":"resident-webgpu-fault",pass:Boolean(vehicleGpu?.residentGraph),
      forceY:springs,forceX:[0,0,0,0],wheelTorque:[p.frontDifferentialTorque||0,p.frontDifferentialTorque||0,
        p.rearDifferentialTorque||0,p.rearDifferentialTorque||0],high:{{drivelineTorque:p.drivelineTorque||0}},
      ultraLow:{{drivelineTorque:p.drivelineTorque||0}},reason:"passive resident-graph snapshot"
    }}}});}}
  else if(m.type==="support"){{const b=bodies.get(m.identity);if(b&&Number.isFinite(m.y)){{b.position[1]=m.y;b.velocity[1]=0;}}}}
  else if(m.type==="impulse"){{const b=bodies.get(m.identity);if(b&&m.velocity){{b.velocity[0]=m.velocity[0];b.velocity[1]=m.velocity[1];b.velocity[2]=m.velocity[2];armEngine("impulse");}}}}
  else if(m.type==="wrench"){{const b=bodies.get(m.identity);if(b){{b.force=[...(m.force||[0,0,0])];
    b.moment=[...(m.moment||[0,0,0])];armEngine("wrench-change");}}}}
  else if(m.type==="vehicle-transmission"){{const b=bodies.get(m.identity);if(b?.kind==="vehicle"){{
    b.pendingTransmission={{mode:m.mode,gearDelta:m.gearDelta,lowRange:m.lowRange,diffLock:m.diffLock}};
    armEngine("transmission-control");}}}}
  else if(m.type==="colliders"){{colliders=m.colliders||[];if(vehicleGpu?.residentGraph){{const field=colliders.find(
      collider=>collider.surface?.kind==="sampled-height-field");if(field)configureResidentVehicleTerrain(vehicleGpu,field,colliders);}}
    armEngine("collider-field-change");}}
  else if(m.type==="parameters"){{Object.assign(parameters,m.parameters||{{}});armEngine("physics-field-change");}}
  else if(m.type==="recycle"){{snapshotBuffers.push(m.buffer);snapshotInFlight=false;}}
  else if(m.type==="stop"){{if(timer)clearInterval(timer);close();}}
}};'''


def living_map_loop_deployment(root: str, physics_program: str) -> dict[str, object]:
    physics = StateLoop(f"{root}/loops/world-physics", physics_program, "fixed-step",
                        ("world.body-pose",), ("control.intent",), 120.0, (), "worker")
    graphics = StateLoop(f"{root}/loops/graphics", root, "animation-frame", (),
                         ("world.body-pose",), effects=("dom", "webgl", "canvas"), isolation="main")
    actions = StateLoop(f"{root}/loops/actions", root, "event", ("control.intent",), effects=("dom",), isolation="main")
    plan = plan_state_loops((physics, graphics, actions))
    for channel in plan["channels"]:
        if "world.body-pose" in channel["fields"]:
            channel.update({
                "transport": "transferable-array-buffer-pool",
                "pool_size": 3,
                "record_capacity": 128,
                "record_layout": [
                    "active", "generation", "position.x", "position.y",
                    "position.z", "velocity.x", "velocity.y", "velocity.z",
                    "vehicle.roll", "vehicle.pitch", "vehicle.yaw",
                    "vehicle.roll-velocity", "vehicle.pitch-velocity", "vehicle.yaw-velocity",
                    "contact.runtime-part-id", "vehicle.spring-force[4]",
                    "vehicle.contact-area[4]", "vehicle.friction-utilization[4]",
                    "vehicle.contact-mode[4]", "vehicle.compression[4]",
                    "vehicle.wheel-omega[4]", "vehicle.traction-scale[4]",
                    "vehicle.brake-scale[4]", "vehicle.engine-torque", "vehicle.clutch-torque",
                    "vehicle.transmission-output-torque", "vehicle.driveline-torque",
                    "vehicle.front-differential-torque", "vehicle.rear-differential-torque",
                    "vehicle.engine-acceleration-torque", "vehicle.engine-angular-acceleration",
                    "vehicle.powertrain-reaction-torque[3]", "vehicle.engine-mount-torque[3]",
                    "vehicle.transmission-gear", "vehicle.transmission-display-gear",
                    "vehicle.transmission-automatic", "control.generation",
                ],
                "allocation": "preallocated-before-first-tick",
                "synchronization": "ownership-transfer-no-locks",
            })
    plan["workers"] = [{"identity": f"{physics.identity}/javascript-worker",
                        "loop": physics.identity, "source_language": "state-loop-ir",
                        "target": "javascript-worker", "source": emit_javascript_physics_worker()}]
    plan["scheduler"] = {"ownership": "single-writer-per-state-field",
                         "presentation": "request-animation-frame",
                         "simulation": "single-fixed-step-worker-with-joined-stage-barriers",
                         "backpressure": "recycled-latest-snapshot",
                         "snapshot_memory": "preallocated-transferable-triple-buffer",
                         "membership": {
                             "policy": "dynamic-awake-set",
                             "drop": "host-observed-supported-low-speed-body",
                             "restore": ["collision-touch", "physics-field-change"],
                             "identity": "retained-outside-solver",
                         },
                         "engine_gears": {
                             "full_dynamics": "force-integration-at-120hz",
                             "kinematic_coast": {
                                 "condition": "moving-with-constant-velocity",
                                 "acceleration_epsilon": 0.001,
                                 "confirmation_ticks": 12,
                                 "action": "skip-force-wasm-and-advance-at-30hz",
                                 "guards": ["bounds", "static-collision"],
                             },
                             "asleep": {
                                 "condition": "all-members-quiescent",
                                 "delay_ticks": 180,
                                 "velocity_epsilon": 0.012,
                                 "action": "disarm-fixed-step-timer",
                             },
                              "wake_events": [
                                  "body-upsert", "control", "impulse", "wrench-change",
                                 "collider-field-change", "physics-field-change",
                             ],
                             "telemetry": "full-dynamics-kinematic-coast-or-asleep",
                          },
                          "synchronization": "lock-free-exclusive-buffer-ownership"}
    plan["wrench_abi"] = {
        "schema": "abstract-ui-physics-wrench-v0",
        "applies_to": "every-physics-body",
        "force": ["force_x", "force_y", "force_z"],
        "moment": ["moment_x", "moment_y", "moment_z"],
        "state": ["roll", "pitch", "yaw", "roll_velocity", "pitch_velocity", "yaw_velocity"],
        "message": "wrench",
        "compatibility": "zero force and zero moment preserve existing translational behavior",
    }
    return plan


__all__ = ["STATE_LOOP_DEPLOYMENT_VERSION", "StateLoop", "state_loop", "identify_state_loops", "plan_state_loops",
           "emit_javascript_physics_worker", "living_map_loop_deployment"]
