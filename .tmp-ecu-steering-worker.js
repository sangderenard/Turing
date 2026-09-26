"use strict";
const SNAPSHOT_STRIDE=162, SNAPSHOT_POOL_SIZE=3,FIXED_DT=0.008333333333333333;
let instance=null, abi=null, vehicleInstance=null,vehicleDynoInstance=null,vehicleAbi=null,contactInstance=null,contactAbi=null,
  vehicleGpu=null, timer=null, sequence=0, snapshotInFlight=false,tickInFlight=false;
let idleTicks=0, coastTicks=0, engineStage="full-dynamics",lastVehicleGpuError=null;
let snapshotCapacity=0, snapshotBuffers=[];
let parameters={}, colliders=[], bodies=new Map(), previous=performance.now(),
  worldBottom={top_y:0,thickness:8,minimum_y:-8};
function contact(p,r,excluded){
  let best=null;
  for(const c of colliders){
    if(c.objectIdentity===excluded||c.surface) continue;
    const mn=c.minimum,mx=c.maximum;
    if(p[1]+r<=mn[1]||p[1]-r>=mx[1]||p[0]<mn[0]-r||p[0]>mx[0]+r||p[2]<mn[2]-r||p[2]>mx[2]+r) continue;
    const cx=Math.max(mn[0],Math.min(mx[0],p[0])), cz=Math.max(mn[2],Math.min(mx[2],p[2]));
    if((p[0]-cx)**2+(p[2]-cz)**2>r*r) continue;
    for(const f of [{d:p[0]+r-mn[0],n:[1,0],q:mn[0]},{d:mx[0]+r-p[0],n:[-1,0],q:-mx[0]},{d:p[2]+r-mn[2],n:[0,1],q:mn[2]},{d:mx[2]+r-p[2],n:[0,-1],q:-mx[2]}])
      if(f.d>=0&&(!best||f.d<best.d)) best={...f,identity:c.identity,runtimePartId:c.runtimePartId||0};
  } return best;
}
function topSupport(previous,next,r,excluded){
  if(next[1]>=previous[1])return null;let best=null;
  for(const c of colliders){
    if(c.objectIdentity===excluded||c.role==="projectile-body"||c.surface)continue;
    const mn=c.minimum,mx=c.maximum,top=mx[1];
    if(next[0]<mn[0]-r||next[0]>mx[0]+r||next[2]<mn[2]-r||next[2]>mx[2]+r)continue;
    if(previous[1]-r>=top-.012&&next[1]-r<=top+.006&&(!best||top>best.top))
      best={top,identity:c.identity,runtimePartId:c.runtimePartId||0};
  }return best;
}
function sampledTerrainSupport(previous,next,r,excluded){
  let best=null;
  for(const collider of colliders){const surface=collider.surface,domain=surface?.domain;
    if(collider.objectIdentity===excluded||!surface||!domain||
      next[0]<domain.minimum_x||next[0]>domain.maximum_x||next[2]<domain.minimum_z||next[2]>domain.maximum_z)continue;
    const before=sampleSurface(surface,previous[0],previous[2]),after=sampleSurface(surface,next[0],next[2]),
      previousBottom=previous[1]-r,nextBottom=next[1]-r,
      wasSupported=Math.abs(previousBottom-before.height)<=.045,
      crossed=previousBottom>=before.height-.012&&nextBottom<=after.height+.012,
      followsSurface=wasSupported&&nextBottom<=after.height+.09&&nextBottom>=after.height-.18;
    if(!crossed&&!followsSurface)continue;
    const raw=[-after.gradient[0],1,-after.gradient[1]],length=Math.max(1e-9,Math.hypot(...raw)),
      candidate={height:after.height,normal:raw.map(value=>value/length),identity:collider.identity,
        runtimePartId:collider.runtimePartId||0};
    if(!best||candidate.height>best.height)best=candidate;
  }return best;
}
function sampleSurface(s,x,z){
  if(s.kind!=="sampled-height-field")return {height:s.origin[1]+s.gradient[0]*(x-s.origin[0])+s.gradient[1]*(z-s.origin[2]),gradient:[...s.gradient]};
  const [columns,rows]=s.resolution,[cellX,cellZ]=s.cell_size,
    u=Math.max(0,Math.min(columns-1,(x-s.origin[0])/cellX)),v=Math.max(0,Math.min(rows-1,(z-s.origin[2])/cellZ)),
    column=Math.min(columns-2,Math.floor(u)),row=Math.min(rows-2,Math.floor(v)),tx=u-column,tz=v-row,
    at=(ix,iz)=>Number(s.heights[iz*columns+ix]),h00=at(column,row),h10=at(column+1,row),
    h01=at(column,row+1),h11=at(column+1,row+1);
  if(tx>=tz)return {height:h00+(h10-h00)*tx+(h11-h10)*tz,
    gradient:[(h10-h00)/cellX,(h11-h10)/cellZ]};
  return {height:h00+(h11-h01)*tx+(h01-h00)*tz,
    gradient:[(h11-h01)/cellX,(h01-h00)/cellZ]};
}
function contactSurfaceAt(x,z,bodyY,reach,excluded=null,includeSolidTops=true){
  let best=null;
  const consider=(height,normal,identity,runtimePartId,material={})=>{
    if(height<bodyY-reach||height>bodyY+reach)return;
    const candidate={height,normal,identity,runtimePartId:runtimePartId||0,material};
    if(!best||height>best.height||height===best.height&&candidate.runtimePartId<best.runtimePartId)best=candidate;
  };
  const terrainReplacesFloor=colliders.some(c=>{const s=c.surface,d=s?.domain;return s?.kind==="sampled-height-field"&&
    x>=d.minimum_x&&x<=d.maximum_x&&z>=d.minimum_z&&z<=d.maximum_z;});
  if(!terrainReplacesFloor)consider(0,[0,1,0],"world-floor",0,
    {identity:"compacted-world-floor",maximum_sink_depth_m:.018,equilibrium_sink_depth_m:.003,restitution:.10});
  for(const c of colliders){if(c.objectIdentity===excluded)continue;
    const s=c.surface;
    if(s){const d=s.domain;if(x<d.minimum_x||x>d.maximum_x||z<d.minimum_z||z>d.maximum_z)continue;
      const sampled=sampleSurface(s,x,z),h=sampled.height,n=[-sampled.gradient[0],1,-sampled.gradient[1]],
        l=Math.max(1e-9,Math.hypot(...n));
      consider(h,n.map(v=>v/l),c.identity,c.runtimePartId,s.contact_material||c.contact_material||{});continue;}
    if(!includeSolidTops||c.role==="projectile-body"||!c.minimum||!c.maximum)continue;
    if(x<c.minimum[0]||x>c.maximum[0]||z<c.minimum[2]||z>c.maximum[2])continue;
    consider(c.maximum[1],[0,1,0],c.identity,c.runtimePartId,c.contact_material||{});
  }return best;
}
async function checkedShaderModule(device,code,label){
  const module=device.createShaderModule({code,label}),info=await module.getCompilationInfo(),
    errors=info.messages.filter(message=>message.type==="error");
  if(errors.length)throw new Error(`${label} WGSL invalid · `+errors.map(message=>
    `line ${message.lineNum}:${message.linePos} ${message.message}`).join(" · "));
  return module;
}
async function initializeVehicleGpu(program){
  if(!program)throw new Error("resident vehicle WebGPU program is missing");
  if(!globalThis.navigator?.gpu)throw new Error("resident vehicle graph requires worker WebGPU");
  const adapter=await navigator.gpu.requestAdapter();
  if(!adapter)throw new Error("resident vehicle graph could not acquire a WebGPU adapter");
  const device=await adapter.requestDevice(),contact=program.tensor_contact_precompile,
    reduction=program.wrench_reduction,integration=program.vehicle_integration,
    geometry=program.terrain_contact_geometry,adapters=program.graph_adapters;
  if(contact?.packed_outputs&&contact.kernel?.source&&reduction?.kernel?.source){
    try{const wheelCount=contact.kernel.invocations,
      contactModule=await checkedShaderModule(device,contact.kernel.source,"compiled-contact"),
      contactPipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module:contactModule,entryPoint:"main"}}),
      reductionModule=await checkedShaderModule(device,reduction.kernel.source,"backend-gemm-reduction"),
      reductionPipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module:reductionModule,entryPoint:"main"}}),
      geometryModule=await checkedShaderModule(device,geometry.kernel.source,"terrain-wall-radial-gather"),
      geometryPipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module:geometryModule,
        entryPoint:geometry.kernel.entrypoint}}),
      assemblyModule=await checkedShaderModule(device,adapters.assembly.source,"vehicle-graph-assembly"),
      assemblyPipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module:assemblyModule,
        entryPoint:adapters.assembly.entrypoint}}),
      integrationStages=integration.stages?.length?integration.stages:[{identity:"monolithic",outputs:integration.outputs,
        output_offset_floats:0,kernel:integration.kernel}],
      integrationModules=await Promise.all(integrationStages.map(stage=>checkedShaderModule(
        device,stage.kernel.source,`compiled-vehicle-${stage.identity}`))),
      integrationPipelines=await Promise.all(integrationModules.map((module,index)=>device.createComputePipelineAsync({
        layout:"auto",compute:{module,entryPoint:integrationStages[index].kernel.entrypoint}}))),
      commitModule=await checkedShaderModule(device,adapters.commit.source,"vehicle-state-commit"),
      commitPipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module:commitModule,
        entryPoint:adapters.commit.entrypoint}}),
      feed=device.createBuffer({size:contact.inputs.length*wheelCount*4,
        usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),
      packed=device.createBuffer({size:contact.outputs.length*wheelCount*4,
        usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),
      resultRead=device.createBuffer({size:(contact.outputs.length*wheelCount+6)*4,
        usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),
      unitColumn=device.createBuffer({size:wheelCount*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),
      reduced=device.createBuffer({size:6*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),
      scalars=device.createBuffer({size:8,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),
      vehicleFeed=device.createBuffer({size:integration.inputs.length*4,
        usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC}),
      vehicleOutputs=device.createBuffer({size:Number(integration.output_buffer_floats||integration.outputs.length)*4,
        usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),
      controls=device.createBuffer({size:adapters.control_abi.length*4,
        usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),
      radialProbes=device.createBuffer({size:60*4,
        usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),
      terrainHeights=device.createBuffer({size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),
      wallColliders=device.createBuffer({size:24,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),
      terrainParameters=device.createBuffer({size:8,
        usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),
      contactBindGroup=device.createBindGroup({layout:contactPipeline.getBindGroupLayout(0),entries:[
        {binding:0,resource:{buffer:feed}},{binding:1,resource:{buffer:packed}}]}),
      reductionBindGroup=device.createBindGroup({layout:reductionPipeline.getBindGroupLayout(0),entries:[
        {binding:0,resource:{buffer:packed}},{binding:1,resource:{buffer:unitColumn}},
        {binding:2,resource:{buffer:reduced}},{binding:3,resource:{buffer:scalars}}]}),
      geometryBindGroup=device.createBindGroup({layout:geometryPipeline.getBindGroupLayout(0),entries:[
        {binding:0,resource:{buffer:terrainHeights}},{binding:1,resource:{buffer:terrainParameters}},
        {binding:2,resource:{buffer:vehicleFeed}},{binding:3,resource:{buffer:feed}},
        {binding:4,resource:{buffer:controls}},{binding:5,resource:{buffer:wallColliders}},
        {binding:6,resource:{buffer:radialProbes}}]}),
      assemblyBindGroup=device.createBindGroup({layout:assemblyPipeline.getBindGroupLayout(0),entries:[
        {binding:0,resource:{buffer:packed}},{binding:1,resource:{buffer:reduced}},
        {binding:2,resource:{buffer:feed}},{binding:3,resource:{buffer:vehicleFeed}},
        {binding:4,resource:{buffer:controls}}]}),
      integrationBindGroups=integrationStages.map((stage,index)=>device.createBindGroup({
        layout:integrationPipelines[index].getBindGroupLayout(0),entries:[
          {binding:0,resource:{buffer:vehicleFeed}},{binding:1,resource:{buffer:vehicleOutputs,
            offset:Number(stage.output_offset_floats||0)*4,size:stage.outputs.length*4}}]})),
      commitBindGroup=device.createBindGroup({layout:commitPipeline.getBindGroupLayout(0),entries:[
        {binding:0,resource:{buffer:vehicleOutputs}},{binding:1,resource:{buffer:vehicleFeed}}]}),
      snapshotValueCount=Number(integration.output_buffer_floats||integration.outputs.length)+
        contact.outputs.length*wheelCount+6+contact.inputs.length*wheelCount+60,
      snapshotReads=Array.from({length:3},()=>({busy:false,buffer:device.createBuffer({size:snapshotValueCount*4,
        usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ})}));
      device.queue.writeBuffer(unitColumn,0,new Float32Array(wheelCount).fill(1));
      device.queue.writeBuffer(scalars,0,new Float32Array([
        Number(reduction.kernel.scalars?.alpha??1),Number(reduction.kernel.scalars?.beta??0)]));
      return {mode:"resident-vehicle-graph",device,wheelCount,shellLaneStart:Number(geometry.shell_lane_start||wheelCount),
        shellLaneCount:Number(geometry.shell_lane_count||0),feed,packed,resultRead,reduced,
        contactPipeline,reductionPipeline,geometryPipeline,assemblyPipeline,integrationPipelines,commitPipeline,
        contactBindGroup,reductionBindGroup,geometryBindGroup,assemblyBindGroup,integrationBindGroups,commitBindGroup,
        integrationStages,
        vehicleFeed,vehicleOutputs,controls,radialProbes,terrainHeights,terrainParameters,wallColliders,snapshotReads,snapshotValueCount,
        residentGraph:true,terrainReady:false,snapshotCounter:0,dispatchSerial:0,
        program,contact,reduction,integration,geometry,adapters};
    }catch(error){throw new Error(`resident vehicle graph pipeline creation failed: ${String(error?.message||error)}`);}
  }
  throw new Error("resident vehicle graph artifact is incomplete");
}

function configureResidentVehicleTerrain(gpu,allColliders=[]){
  if(!gpu?.residentGraph)return false;
  const fields=allColliders.filter(collider=>collider.surface?.kind==="sampled-height-field"),
    packedHeights=[],descriptors=[];
  fields.forEach(field=>{const surface=field.surface,domain=surface.domain,offset=packedHeights.length;
    packedHeights.push(...surface.heights.map(Number));descriptors.push(
      Number(surface.origin[0]),Number(surface.origin[1]),Number(surface.origin[2]),
      Number(surface.cell_size[0]),Number(surface.cell_size[1]),Number(surface.resolution[0]),
      Number(surface.resolution[1]),Number(domain.minimum_x),Number(domain.maximum_x),
      Number(domain.minimum_z),Number(domain.maximum_z),offset);});
  const heights=new Float32Array(Math.max(1,packedHeights.length));heights.set(packedHeights),
    walls=allColliders.filter(collider=>!collider.surface&&collider.role!=="projectile-body"&&
      Array.isArray(collider.minimum)&&Array.isArray(collider.maximum)),
    wallValues=new Float32Array(Math.max(6,walls.length*6));
  walls.forEach((wall,index)=>wallValues.set([...wall.minimum,...wall.maximum].map(Number),index*6));
  const
    parameters=new Float32Array([fields.length,walls.length,...descriptors]),oldTerrain=gpu.terrainHeights,
    oldWalls=gpu.wallColliders,oldParameters=gpu.terrainParameters,
    terrainHeights=gpu.device.createBuffer({size:Math.max(4,heights.byteLength),
      usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),
    wallColliders=gpu.device.createBuffer({size:wallValues.byteLength,
      usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),
    terrainParameters=gpu.device.createBuffer({size:Math.max(8,parameters.byteLength),
      usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  gpu.device.queue.writeBuffer(terrainHeights,0,heights);gpu.device.queue.writeBuffer(wallColliders,0,wallValues);
  gpu.device.queue.writeBuffer(terrainParameters,0,parameters);
  gpu.terrainHeights=terrainHeights;gpu.wallColliders=wallColliders;gpu.terrainParameters=terrainParameters;
  gpu.geometryBindGroup=gpu.device.createBindGroup({
    layout:gpu.geometryPipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:terrainHeights}},{binding:1,resource:{buffer:terrainParameters}},
      {binding:2,resource:{buffer:gpu.vehicleFeed}},{binding:3,resource:{buffer:gpu.feed}},
      {binding:4,resource:{buffer:gpu.controls}},{binding:5,resource:{buffer:wallColliders}},
      {binding:6,resource:{buffer:gpu.radialProbes}}]});gpu.terrainReady=true;
  void gpu.device.queue.onSubmittedWorkDone().then(()=>{oldTerrain.destroy();oldWalls.destroy();oldParameters.destroy();});return true;
}
async function ensureParametricVehiclePipelines(gpu){
  if(gpu.parametricIntegrationPipelines)return gpu.parametricIntegrationPipelines;
  if(gpu.parametricPipelinePromise)return gpu.parametricPipelinePromise;
  gpu.parametricPipelinePromise=(async()=>{const modules=await Promise.all(gpu.integrationStages.map(stage=>
      checkedShaderModule(gpu.device,stage.parametric_kernel.source,`parametric-vehicle-${stage.identity}`))),
    pipelines=await Promise.all(modules.map((module,index)=>gpu.device.createComputePipelineAsync({
      layout:"auto",compute:{module,entryPoint:gpu.integrationStages[index].parametric_kernel.entrypoint}}))),
    bindGroups=gpu.integrationStages.map((stage,index)=>gpu.device.createBindGroup({
      layout:pipelines[index].getBindGroupLayout(0),entries:[
        {binding:0,resource:{buffer:gpu.vehicleFeed}},{binding:1,resource:{buffer:gpu.vehicleOutputs,
          offset:Number(stage.output_offset_floats||0)*4,size:stage.outputs.length*4}}]}));
    gpu.parametricIntegrationPipelines=pipelines;gpu.parametricIntegrationBindGroups=bindGroups;return pipelines;
  })();return gpu.parametricPipelinePromise;
}
function requestParametricVehiclePipelines(body,reason){
  const gpu=vehicleGpu;if(!gpu?.residentGraph)return false;body.gpuKernelVariant="parametric-pending";
  void ensureParametricVehiclePipelines(gpu).then(()=>{if(vehicleGpu!==gpu||bodies.get(body.identity)!==body)return;
    body.gpuKernelVariant="parametric";body.gpuStateDirty=true;
    postMessage({type:"vehicle-gpu-recovered",reason:`parametric layer passes ready · ${reason}`});armEngine("parametric-layers-ready");
  }).catch(error=>{if(vehicleGpu===gpu)vehicleGpu=null;postMessage({type:"vehicle-wasm-fallback",
    reason:`parametric layer compile failed · ${String(error?.message||error)}`});});return true;
}
function residentVehicleControlValues(body){
  const t=body.config.transmission,state=body.transmission||{},range=state.lowRange?Number(t.ultra_low_range_ratio):1,
    ratio=Number(state.engagedRatio||t.forward_ratios[Math.max(0,Number(state.gear||t.starting_gear)-1)]),
    throttle=Number(body.effectiveThrottle??body.controls?.throttle??0),locks=body.brakeLocks||{},
    defaultFront=Number(body.config.drivetrain.front_drive_fraction||.5),
    coupling=mode=>mode==="locked"?1:mode==="limited-slip"?.32:0,
    frontCoupling=state.frontDiffMode?coupling(state.frontDiffMode):(state.frontDiffLock?1:0),
    rearCoupling=state.rearDiffMode?coupling(state.rearDiffMode):(state.rearDiffLock?1:0),
    centerCoupling=state.centerDiffMode?coupling(state.centerDiffMode):(state.centerDiffLock?1:0),
    frontShare=centerCoupling>=.999 ? .5 : Math.max(.05,Math.min(.95,Number(state.frontDriveShare??defaultFront)));
  if(!Number.isFinite(body.driveDirection)||body.driveDirection===0)body.driveDirection=1;
  return new Float32Array([throttle,Number(body.appliedSteering??body.controls?.steering??0),
    Number(body.effectiveBrake??body.controls?.brake??0),ratio/range,Number(t.reverse_ratio),range,frontCoupling,
    body.driveDirection,Number(locks.front_left||0),Number(locks.front_right||0),
    Number(locks.rear_left||0),Number(locks.rear_right||0),frontShare,
    rearCoupling,centerCoupling>=.999?1:0,
    state.tractionControlEnabled===false?0:1,state.absEnabled===false?0:1,
    Number(body.frontKnuckleSteerAngle||0),Number(body.rearKnuckleSteerAngle||0),
    Math.max(0,Math.min(1,Number(state.tractionControlAuthority??1))),
    Math.max(0,Math.min(1,Number(state.absAuthority??1))),centerCoupling,
    state.frontDifferentialBrake?1:0,Math.max(state.rearDifferentialBrake?1:0,Number(
      body.rearDifferentialBrakeCommand||0)),
    body.bodyShell==="bare-frame"||(body.config.mechanical_graph?.edges||[]).filter(edge=>edge.identity.startsWith("body_shell.mount."))
      .every(edge=>ensureVehicleDamage(body).members[edge.identity]?.failed)?0:1,
    Number(body.wheelSteerAngles?.front_left??body.frontKnuckleSteerAngle??0),
    Number(body.wheelSteerAngles?.front_right??body.frontKnuckleSteerAngle??0),
    Number(body.wheelSteerAngles?.rear_left??body.rearKnuckleSteerAngle??0),
    Number(body.wheelSteerAngles?.rear_right??body.rearKnuckleSteerAngle??0),
    Number(ensureVehicleEnergy(body).tirePressurePa||body.config.tires?.pressure_pa||155000)]);
}
function initializeResidentVehicleState(gpu,body){
  if(!gpu?.residentGraph)return false;const inputs=gpu.integration.inputs,values=new Float32Array(inputs.length),
    names=["front_left","front_right","rear_left","rear_right"],state={...(body.defaults||{}),...(body.wasmState||{}),
      position_x:body.position[0],position_y:body.position[1],position_z:body.position[2],
      velocity_x:body.velocity[0],velocity_y:body.velocity[1],velocity_z:body.velocity[2],
      roll:body.roll||0,pitch:body.pitch||0,yaw:body.yaw||0,roll_velocity:body.rollVelocity||0,
      pitch_velocity:body.pitchVelocity||0,yaw_velocity:body.yawVelocity||0,dt:FIXED_DT};
  names.forEach(name=>{state[`wheel_omega_${name}`]=body.wheelOmegas?.[name]||0;
    state[`compression_${name}`]=body.compressions?.[name]||0;
    state[`previous_slip_longitudinal_${name}`]=body.previousSlips?.[name]||0;});
  inputs.forEach((name,index)=>values[index]=Number(state[name]||0));gpu.device.queue.writeBuffer(gpu.vehicleFeed,0,values);
  gpu.device.queue.writeBuffer(gpu.controls,0,residentVehicleControlValues(body));body.gpuResidentInitialized=true;
  body.gpuEpoch=Number(body.gpuEpoch||0);body.gpuAppliedSnapshotSerial=Number(body.gpuAppliedSnapshotSerial||0);
  body.gpuStateDirty=false;body.gpuControlsDirty=false;return true;
}
function applyResidentVehicleSnapshot(gpu,body,values){
  const outputs=gpu.integration.outputs,slots=gpu.integration.output_slots||Object.fromEntries(outputs.map((name,index)=>[name,index])),
    output=Object.fromEntries(outputs.map(name=>[name,values[slots[name]]])),
    names=["front_left","front_right","rear_left","rear_right"],
    packedStart=Number(gpu.integration.output_buffer_floats||outputs.length),
    packedCount=gpu.contact.outputs.length*gpu.wheelCount,reducedStart=packedStart+packedCount,
    feedStart=reducedStart+6,probeStart=feedStart+gpu.contact.inputs.length*gpu.wheelCount,
    contactFeed=values.subarray(feedStart,probeStart),ci=Object.fromEntries(
      gpu.contact.inputs.map((name,index)=>[name,index])),co=Object.fromEntries(gpu.contact.outputs.map((name,index)=>[name,index]));
  if(!Object.values(output).every(Number.isFinite)){
    resetVehicleDrivetrainState(body,"resident GPU produced a non-finite output");
    body.gpuEpoch=Number(body.gpuEpoch||0)+1;body.gpuStateDirty=true;
    throw new Error("resident vehicle snapshot contained a non-finite output");
  }
  body.bottomRejected=false;body.supportLatchRejected=false;
  const previousPosition=[...body.position],previousVelocity=[...body.velocity];
  body.position=[output.position_x_next,output.position_y_next,output.position_z_next];
  body.velocity=[output.velocity_x_next,output.velocity_y_next,output.velocity_z_next];
  body.accelerationMagnitude=Math.hypot(body.velocity[0]-previousVelocity[0],body.velocity[1]-previousVelocity[1],
    body.velocity[2]-previousVelocity[2])/(FIXED_DT*4);
  body.roll=output.roll_next;body.pitch=output.pitch_next;body.yaw=output.yaw_next;
  body.rollVelocity=output.roll_velocity_next;body.pitchVelocity=output.pitch_velocity_next;body.yawVelocity=output.yaw_velocity_next;
  names.forEach(name=>{body.wheelOmegas[name]=output[`wheel_omega_${name}_next`];
    body.compressions[name]=output[`compression_${name}_next`];body.previousSlips[name]=output[`slip_longitudinal_${name}_next`];});
  body.springForces=names.map(name=>output[`spring_force_${name}`]);
  body.tractionScales=names.map(name=>output[`traction_scale_${name}`]);
  body.brakeScales=names.map(name=>output[`brake_scale_${name}`]);
  body.damperScales=names.map(name=>output[`damper_scale_${name}`]);
  body.contactAreas=names.map((_,lane)=>values[packedStart+co.contact_area*gpu.wheelCount+lane]);
  body.frictionUtilizations=names.map((_,lane)=>{const force=["x","y","z"].map(axis=>
      values[packedStart+co[`chassis_force_${axis}`]*gpu.wheelCount+lane]),normal=["x","y","z"].map(axis=>
      contactFeed[ci[`normal_${axis}`]*gpu.wheelCount+lane]),load=Math.max(0,force.reduce((sum,value,index)=>sum+value*normal[index],0)),
      tangent=Math.sqrt(Math.max(0,force.reduce((sum,value)=>sum+value*value,0)-load*load)),
      mu=contactFeed[ci.mu_static*gpu.wheelCount+lane];return tangent/Math.max(1e-5,mu*load);});
  body.contactModes=names.map((_,lane)=>{const area=body.contactAreas[lane],u=body.frictionUtilizations[lane];
    return area<=0?0:u<.78?1:u<=1.08?2:3;});
  body.bodyShellContactLoadN=Array.from({length:gpu.shellLaneCount},(_,index)=>gpu.shellLaneStart+index)
    .reduce((sum,lane)=>sum+Math.hypot(...["x","y","z"].map(axis=>values[
      packedStart+co[`chassis_force_${axis}`]*gpu.wheelCount+lane])),0);
  body.radialProbePenetrations=names.map((_,wheel)=>Array.from(values.slice(
    probeStart+wheel*15,probeStart+(wheel+1)*15)));
  body.radialProbeActiveCounts=body.radialProbePenetrations.map(samples=>samples.filter(value=>value>0).length);
  body.powertrain={engineTorque:output.engine_torque,clutchTorque:output.clutch_torque,
    transmissionOutputTorque:output.transmission_output_torque,drivelineTorque:output.driveline_torque,
    frontDifferentialTorque:output.front_differential_torque,rearDifferentialTorque:output.rear_differential_torque,
    engineAccelerationTorque:output.engine_acceleration_torque,engineAngularAcceleration:output.engine_angular_acceleration,
    reactionTorque:[output.powertrain_reaction_torque_x,output.powertrain_reaction_torque_y,output.powertrain_reaction_torque_z],
    mountTorque:[output.engine_mount_torque_x,output.engine_mount_torque_y,output.engine_mount_torque_z],
    engineAngularSpeed:output.engine_angular_speed_next,engineRPM:output.engine_rpm,
    tractionControlDissipationTorque:output.traction_control_dissipation_torque,
    serviceBrakeReactionTorque:output.service_brake_reaction_torque,
    rollingResistanceReactionTorque:output.rolling_resistance_reaction_torque,
    tireContactReactionTorque:output.tire_contact_reaction_torque,
    drivetrainChassisReactionTorque:output.drivetrain_chassis_reaction_torque};
  body.wasmState={engine_angular_speed:output.engine_angular_speed_next};
  names.forEach(name=>{body.wasmState[`slip_sensor_velocity_${name}`]=output[`slip_sensor_velocity_${name}_next`];
    body.wasmState[`friction_utilization_${name}`]=output[`friction_utilization_${name}_next`];
    body.wasmState[`friction_utilization_sensor_velocity_${name}`]=
      output[`friction_utilization_sensor_velocity_${name}_next`];});
  if(resolveWorldBottom(body,previousPosition)){body.gpuEpoch=Number(body.gpuEpoch||0)+1;body.gpuStateDirty=true;
    if(body.bottomRejected)resetVehicleDrivetrainState(body,"resident GPU crossed the world-bottom guard");}
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
    const parametric=body.gpuKernelVariant==="parametric",integrationPipelines=parametric?
      gpu.parametricIntegrationPipelines:gpu.integrationPipelines,integrationBindGroups=parametric?
      gpu.parametricIntegrationBindGroups:gpu.integrationBindGroups;
    gpu.integrationStages.forEach((stage,index)=>run(integrationPipelines[index],integrationBindGroups[index],
      stage.kernel.dispatch));
    run(gpu.commitPipeline,gpu.commitBindGroup,gpu.adapters.commit.dispatch);
    let snapshot=null;if((gpu.snapshotCounter++&3)===0)snapshot=gpu.snapshotReads.find(item=>!item.busy)||null;
    if(snapshot){snapshot.busy=true;snapshot.epoch=Number(body.gpuEpoch||0);snapshot.serial=++gpu.dispatchSerial;
      const outputBytes=Number(gpu.integration.output_buffer_floats||gpu.integration.outputs.length)*4,
        packedBytes=gpu.contact.outputs.length*gpu.wheelCount*4,reducedBytes=24;
      encoder.copyBufferToBuffer(gpu.vehicleOutputs,0,snapshot.buffer,0,outputBytes);
      encoder.copyBufferToBuffer(gpu.packed,0,snapshot.buffer,outputBytes,packedBytes);
      encoder.copyBufferToBuffer(gpu.reduced,0,snapshot.buffer,outputBytes+packedBytes,reducedBytes);
      encoder.copyBufferToBuffer(gpu.feed,0,snapshot.buffer,outputBytes+packedBytes+reducedBytes,
        gpu.contact.inputs.length*gpu.wheelCount*4);
      encoder.copyBufferToBuffer(gpu.radialProbes,0,snapshot.buffer,
        outputBytes+packedBytes+reducedBytes+gpu.contact.inputs.length*gpu.wheelCount*4,60*4);}
    gpu.device.queue.submit([encoder.finish()]);if(snapshot)void snapshot.buffer.mapAsync(GPUMapMode.READ).then(()=>{
      const values=new Float32Array(snapshot.buffer.getMappedRange().slice(0));snapshot.buffer.unmap();snapshot.busy=false;
      if(bodies.get(body.identity)!==body||Number(body.gpuEpoch||0)!==snapshot.epoch||
          snapshot.serial<=Number(body.gpuAppliedSnapshotSerial||0))return;
      body.gpuAppliedSnapshotSerial=snapshot.serial;try{applyResidentVehicleSnapshot(gpu,body,values);}
      catch(error){const reason=String(error?.message||error);vehicleGpu=null;
        postMessage({type:vehicleInstance&&contactInstance?"vehicle-wasm-fallback":"vehicle-gpu-error",
          reason,error:reason});armEngine("gpu-snapshot-fallback");}
    },error=>{snapshot.busy=false;postMessage({type:"vehicle-gpu-error",error:String(error?.message||error)});});
  }finally{gpu.dispatchActive=false;}
}

function rotateBodyVector(body,v){
  const cr=Math.cos(body.roll||0),sr=Math.sin(body.roll||0),cp=Math.cos(body.pitch||0),sp=Math.sin(body.pitch||0),
    cy=Math.cos(body.yaw||0),sy=Math.sin(body.yaw||0),
    r=[v[0],v[1]*cr-v[2]*sr,v[1]*sr+v[2]*cr],p=[r[0]*cp-r[1]*sp,r[0]*sp+r[1]*cp,r[2]];
  return [p[0]*cy-p[2]*sy,p[1],p[0]*sy+p[2]*cy];
}
function c2Unit(value){const t=Math.max(0,Math.min(1,Number(value)||0));return t*t*t*(10+t*(-15+6*t));}
function c2Positive(value,width){return Number(value)*c2Unit(Number(value)/Math.max(1e-9,Number(width)));}
function c2Clamp(value,lower,upper,width){const raised=c2Positive(Number(value)-Number(lower),width);
  return Number(upper)-c2Positive(Number(upper)-Number(lower)-raised,width);}
function secondOrderChannel(owner,valueKey,velocityKey,target,frequency,damping,dt){
  let value=Number(owner[valueKey]??0),velocity=Number(owner[velocityKey]??0),omega=2*Math.PI*Number(frequency),
    acceleration=omega*omega*(Number(target)-value)-2*Number(damping)*omega*velocity;
  velocity+=acceleration*dt;value+=velocity*dt;owner[valueKey]=value;owner[velocityKey]=velocity;return value;
}
function resetVehicleDrivetrainState(body,reason){
  const names=["front_left","front_right","rear_left","rear_right"],
    idle=Math.max(0,Number(body.defaults?.engine_idle_angular_speed||0));
  body.wasmState={engine_angular_speed:idle};
  body.wheelOmegas=Object.fromEntries(names.map(name=>[name,0]));
  body.wheelSteerAngles=Object.fromEntries(names.map(name=>[name,0]));
  body.previousSlips=Object.fromEntries(names.map(name=>[name,0]));
  body.appliedThrottle=0;body.throttleVelocity=0;
  body.powertrain={engineTorque:0,clutchTorque:0,transmissionOutputTorque:0,drivelineTorque:0,
    frontDifferentialTorque:0,rearDifferentialTorque:0,engineAccelerationTorque:0,
    engineAngularAcceleration:0,reactionTorque:[0,0,0],mountTorque:[0,0,0],
    engineAngularSpeed:idle,engineRPM:idle*60/(2*Math.PI),tractionControlDissipationTorque:0,
    serviceBrakeReactionTorque:0,rollingResistanceReactionTorque:0,tireContactReactionTorque:0,
    drivetrainChassisReactionTorque:0};
  body.gpuEpoch=Number(body.gpuEpoch||0)+1;body.gpuStateDirty=true;body.gpuControlsDirty=true;
  body.lastDrivetrainRecoveryReason=String(reason||"vehicle-state-recovery");
}
function ensureVehicleDamage(body){
  if(body.damage)return body.damage;const names=["front_left","front_right","rear_left","rear_right"];
  body.damage={mode:"elastic",springPlasticSet:Object.fromEntries(names.map(name=>[name,0])),
    springHealth:Object.fromEntries(names.map(name=>[name,1])),
    halfshaftHealth:Object.fromEntries(names.map(name=>[name,1])),members:{},revision:0};
  for(const edge of body.config.mechanical_graph?.edges||[])if(edge.damage)body.damage.members[edge.identity]={
    elasticStrain:0,plasticStrain:0,restLength:Number(edge.damage.natural_rest_length??edge.rest_length),failed:false};
  return body.damage;
}
function ensureVehicleEnergy(body){
  if(body.energy)return body.energy;const fuel=body.config.fuel_system||{},electrical=body.config.electrical||{},seed=body.energySeed||{},
    shell=body.config.body_shell||{},initialFuel=Math.max(0,Number(fuel.initial_fuel_mass_kg||0)),initialMass=Number(body.config.mass||1),
    configuredShellAssemblyMassKg=Number(shell.shell_mass_kg||0)+Number(shell.mount_mass_kg||0);
  body.energy={fuelMassKg:Number.isFinite(seed.fuelMassKg)?Math.max(0,Number(seed.fuelMassKg)):initialFuel,
    fuelCapacityKg:Number(fuel.capacity_kg||initialFuel),
    dryMassKg:Math.max(1,initialMass-initialFuel),batteryChargeWh:Number.isFinite(seed.batteryChargeWh)?Number(seed.batteryChargeWh):
      Number(electrical.battery_capacity_wh||1)*Number(electrical.initial_state_of_charge||1),
    batteryCapacityWh:Number(electrical.battery_capacity_wh||1),
    ignitionOn:seed.ignitionOn!==false,starterEngaged:false,headlightsOn:Boolean(seed.headlightsOn),hornOn:false,alternatorPowerW:0,
    electricalLoadW:0,fuelFlowKgS:0,lastPublishedAt:0,lastMassKg:initialMass,configuredShellAssemblyMassKg,
    activeShellAssemblyMassKg:body.bodyShell==="bare-frame"?0:configuredShellAssemblyMassKg,
    fuelIdentity:seed.fuelIdentity||body.fuelProfile||"pump-gasoline-93",
    requestedIgnitionProfileIdentity:seed.requestedIgnitionProfileIdentity||seed.ignitionProfileIdentity||
      body.ignitionProfile||"gasoline-distributor",
    ignitionProfileIdentity:seed.ignitionProfileIdentity||body.ignitionProfile||"gasoline-distributor",
    ignitionTimingOffsetCycles:14/720,combustionSharpness:1,timingErrorDegrees:0,combustionStress:0,
    computerOnline:true,ecuOnline:true,lightingCircuitOnline:true,tailLightsOn:false,brakeLightsOn:false,
    tirePressurePa:Number(seed.tirePressurePa||body.config.tires?.pressure_pa||155000),
    tirePressureTargetPa:Number(seed.tirePressureTargetPa||seed.tirePressurePa||body.config.tires?.pressure_pa||155000),
    referenceTirePressurePa:Number(seed.referenceTirePressurePa||body.config.tires?.pressure_pa||155000),
    pneumaticCompressorOn:false,pneumaticCompressorPowerW:0,hydraulicPumpOn:false,hydraulicPumpPowerW:0,
    steeringServoOnline:true,steeringServoPowerW:0,
    combustionDamage:Number(seed.combustionDamage||0)};return body.energy;
}
function ensureVehicleDriverAssistance(body){
  if(body.driverAssistance)return body.driverAssistance;const redline=Number(body.config.powertrain?.redline_rpm||6500);
  body.driverAssistance={drivingMode:body.drivingMode||"road",governorRpm:redline,cruiseEnabled:false,
    cruiseTargetSpeedMps:0,cruiseIntegral:0,cruiseThrottle:0,cruiseBrake:0,tiltEnabled:true,
    tiltAuthority:0,tiltRisk:0,tiltGovernorRpm:redline,rearDifferentialBrakeCommand:0,
    previousForwardSpeedMps:0};return body.driverAssistance;
}
function prepareVehicleEnergy(body,dt){
  const state=ensureVehicleEnergy(body),fuel=body.config.fuel_system||{},electrical=body.config.electrical||{},
    fuelProfile=(body.fuelProfiles||[]).find(item=>item.identity===state.fuelIdentity)||{
      energy_density_j_per_kg:fuel.fuel_energy_density_j_per_kg||44e6,torque_scale:1,preferred_advance_degrees:14,
      combustion_sharpness:1},requestedIgnitionProfile=(body.ignitionProfiles||[]).find(item=>
      item.identity===(state.requestedIgnitionProfileIdentity||state.ignitionProfileIdentity))||{
        identity:"gasoline-distributor",advance_degrees:14,dispatch:"ecu-electronic"},
    powertrain=body.powertrain||{},omega=Math.max(0,Number(powertrain.engineAngularSpeed||
      body.wasmState?.engine_angular_speed||0)),rpm=omega*60/(2*Math.PI),
    mechanicalPower=Math.max(0,Number(powertrain.engineTorque||0)*omega),efficiency=Math.max(.05,
      Number(body.defaults?.combustion_efficiency||body.config.powertrain?.combustion_efficiency||.3)),
    energyDensity=Math.max(1,Number(fuelProfile.energy_density_j_per_kg||fuel.fuel_energy_density_j_per_kg||44e6));
  state.computerOnline=state.batteryChargeWh>.05;state.ecuOnline=state.computerOnline&&state.ignitionOn;
  state.lightingCircuitOnline=state.computerOnline;
  const electronicIgnition=requestedIgnitionProfile.dispatch==="ecu-electronic",
    ignitionDispatched=!electronicIgnition||state.ecuOnline,rpmBlend=Math.max(0,Math.min(1,(rpm-700)/3200)),
    engineLoad=Math.max(0,Math.min(1,Math.abs(Number(body.effectiveThrottle||0)))),
    dispatchedAdvance=Number(requestedIgnitionProfile.advance_degrees||0)+rpmBlend*Number(
      requestedIgnitionProfile.rpm_advance_degrees||0)-engineLoad*Number(requestedIgnitionProfile.load_retard_degrees||0),
    preferredAdvance=Number(fuelProfile.preferred_advance_degrees||0)+rpmBlend*Number(
      fuelProfile.preferred_rpm_advance_degrees||0);
  state.ignitionProfileIdentity=requestedIgnitionProfile.identity;
  state.timingErrorDegrees=dispatchedAdvance-preferredAdvance;
  const ignitionTorqueScale=(ignitionDispatched?1:0)*Math.max(.25,1-Math.abs(state.timingErrorDegrees)/38);
  state.ignitionTimingOffsetCycles=dispatchedAdvance/720;
  state.dispatchedAdvanceDegrees=dispatchedAdvance;
  state.combustionSharpness=Number(fuelProfile.combustion_sharpness||1);
  state.combustionStress=Math.abs(state.timingErrorDegrees)/18*Math.abs(Number(body.effectiveThrottle||0))+
    Math.max(0,state.timingErrorDegrees)/24;
  state.combustionDamage=Math.min(1,state.combustionDamage+Math.max(0,state.combustionStress-.35)*dt*.0025);
  state.fuelFlowKgS=state.ignitionOn&&state.fuelMassKg>0?mechanicalPower/(energyDensity*efficiency):0;
  state.fuelMassKg=Math.max(0,state.fuelMassKg-state.fuelFlowKgS*dt);
  const starterAllowed=state.starterEngaged&&state.ignitionOn&&state.batteryChargeWh>1,
    starterLoad=starterAllowed?Number(electrical.starter_power_w||0):0;
  state.tailLightsOn=state.lightingCircuitOnline&&state.headlightsOn;
  state.brakeLightsOn=state.lightingCircuitOnline&&Number(body.effectiveBrake??body.controls?.brake??0)>.025;
  const minimumPressure=Number(electrical.minimum_tire_pressure_pa||45000),maximumPressure=Number(
      electrical.maximum_tire_pressure_pa||260000),pressureRate=Number(electrical.pneumatic_pressure_rate_pa_s||9000);
  state.tirePressureTargetPa=Math.max(minimumPressure,Math.min(maximumPressure,Number(state.tirePressureTargetPa||155000)));
  const names=["front_left","front_right","rear_left","rear_right"],priorCompressions=state.lastShockCompressions||{},
    shockAirDemand=names.reduce((sum,name)=>sum+Math.abs(Number(body.compressions?.[name]||0)-Number(
      priorCompressions[name]||body.compressions?.[name]||0))/Math.max(1e-6,dt),0);
  state.lastShockCompressions=Object.fromEntries(names.map(name=>[name,Number(body.compressions?.[name]||0)]));
  const tireInflationDemand=state.tirePressurePa<state.tirePressureTargetPa-250;
  state.pneumaticCompressorOn=state.computerOnline&&(tireInflationDemand||shockAirDemand>.06);
  const pressureDelta=state.tirePressureTargetPa-state.tirePressurePa,pressureStep=(pressureDelta>=0?
      (state.pneumaticCompressorOn?pressureRate:0):pressureRate*.45)*dt;
  state.tirePressurePa+=Math.sign(pressureDelta)*Math.min(Math.abs(pressureDelta),pressureStep);
  state.pneumaticCompressorPowerW=state.pneumaticCompressorOn?Number(electrical.pneumatic_compressor_power_w||0)*
    (tireInflationDemand?1:Math.min(.55,.12+shockAirDemand*.35)):0;
  const leveling=body.chassisLeveling||body.config.chassis_leveling||{},levelingVelocities=Object.entries(
      body.levelingState||{}).filter(([name])=>name.endsWith("Velocity")).map(([,value])=>Math.abs(Number(value||0))),
    hydraulicDemand=Math.min(1,(levelingVelocities.length?Math.max(...levelingVelocities):0)/Math.max(.001,
      Number(leveling.pose_lerp_rate_m_s||leveling.maximum_actuator_rate_m_s||.055)));
  state.hydraulicPumpOn=Boolean(leveling.enabled)&&hydraulicDemand>.01&&state.computerOnline;
  state.hydraulicPumpPowerW=state.hydraulicPumpOn?Number(electrical.hydraulic_pump_power_w||0)*hydraulicDemand:0;
  state.electricalLoadW=(state.ignitionOn?Number(electrical.base_load_w||0):0)+
    (state.computerOnline?Number(electrical.ecu_load_w||0):0)+
    (state.lightingCircuitOnline&&state.headlightsOn?Number(electrical.headlight_load_w||0):0)+
    (state.tailLightsOn?Number(electrical.tail_light_load_w||0):0)+
    (state.brakeLightsOn?Number(electrical.brake_light_load_w||0):0)+
    (state.lightingCircuitOnline&&state.hornOn?Number(electrical.horn_load_w||0):0)+starterLoad;
  state.electricalLoadW+=state.pneumaticCompressorPowerW+state.hydraulicPumpPowerW+
    Number(state.steeringServoPowerW||0);
  const alternatorSpeed=Math.max(0,Math.min(1,(rpm-350)/1800)),alternatorAvailable=state.ignitionOn?
      Number(electrical.alternator_max_power_w||0)*alternatorSpeed:0,
    regulatorChargeDemand=Math.max(0,Math.min(Number(electrical.alternator_max_power_w||0)*.55,
      (state.batteryCapacityWh-state.batteryChargeWh)*120));
  state.alternatorPowerW=Math.min(alternatorAvailable,state.electricalLoadW+regulatorChargeDemand);
  const accessoryLoadTorque=omega>8?state.alternatorPowerW/(omega*.72):0;
  state.batteryChargeWh=Math.max(0,Math.min(state.batteryCapacityWh,state.batteryChargeWh+
    (state.alternatorPowerW-state.electricalLoadW)*dt/3600));
  // Combustion cannot bootstrap a stationary crank.  The starter or a wheel-driven
  // clutch must first carry the engine through cranking speed; an authored running
  // vehicle is seeded at idle when it is initially inserted below.
  const engineEnabled=state.ignitionOn&&ignitionDispatched&&state.fuelMassKg>0&&(omega>12||starterAllowed)?1:0,
    fuelTorqueScale=Number(fuelProfile.torque_scale||1);
  if(Math.abs(Number(body.defaults?.tire_pressure||0)-state.tirePressurePa)>5){body.defaults={...(body.defaults||{}),
      tire_pressure:state.tirePressurePa};body.config.tires.pressure_pa=state.tirePressurePa;body.gpuStateDirty=true;}
  if(Number(body.defaults?.engine_enabled)!==engineEnabled||Number(body.defaults?.fuel_torque_scale)!==fuelTorqueScale||
      Number(body.defaults?.ignition_torque_scale)!==ignitionTorqueScale||Math.abs(Number(
        body.defaults?.accessory_load_torque||0)-accessoryLoadTorque)>.02){body.defaults={...(body.defaults||{}),engine_enabled:engineEnabled,
      fuel_torque_scale:fuelTorqueScale,ignition_torque_scale:ignitionTorqueScale,accessory_load_torque:accessoryLoadTorque};
    body.gpuStateDirty=true;}
  if(starterAllowed&&omega<Number(electrical.starter_cranking_speed_rad_s||28)){const inertia=Math.max(.01,
      Number(body.defaults?.engine_rotating_inertia||.2)),started=omega+Number(electrical.starter_torque_nm||0)*dt/inertia;
    body.wasmState={...(body.wasmState||{}),engine_angular_speed:started};body.gpuStateDirty=true;}
  const liveMass=state.dryMassKg+state.fuelMassKg-state.configuredShellAssemblyMassKg+state.activeShellAssemblyMassKg;
  if(Math.abs(liveMass-state.lastMassKg)>=.05){const scale=liveMass/Math.max(1,state.lastMassKg);
    body.config.mass=liveMass;body.defaults={...(body.defaults||{}),inverse_mass:1/liveMass,
      inverse_inertia_roll:Number(body.defaults?.inverse_inertia_roll||0)/scale,
      inverse_inertia_pitch:Number(body.defaults?.inverse_inertia_pitch||0)/scale,
      inverse_inertia_yaw:Number(body.defaults?.inverse_inertia_yaw||0)/scale};
    state.lastMassKg=liveMass;body.gpuStateDirty=true;}
  const now=performance.now();if(now-state.lastPublishedAt>250){state.lastPublishedAt=now;postMessage({
    type:"vehicle-energy",identity:body.identity,energy:{...state,stateOfCharge:state.batteryChargeWh/
      Math.max(1e-9,state.batteryCapacityWh),totalMassKg:liveMass},
    driverAssistance:{...ensureVehicleDriverAssistance(body)}});}
}
function applyVehicleChassisProfile(body,profile){
  const graph=body.config.mechanical_graph,reference=body.chassisReferenceDefaults||={
      mass:Number(body.config.mass),inverse_mass:Number(body.defaults?.inverse_mass||0),
      inverse_inertia_roll:Number(body.defaults?.inverse_inertia_roll||0),
      inverse_inertia_pitch:Number(body.defaults?.inverse_inertia_pitch||0),
      inverse_inertia_yaw:Number(body.defaults?.inverse_inertia_yaw||0)},
    newMass=Math.max(1,Number(profile.vehicle_mass_kg||reference.mass)),massScale=newMass/Math.max(1,reference.mass);
  for(const edge of graph?.edges||[]){if(!edge.chassis_profile_member)continue;
    edge.radius=Number(profile.outer_diameter_m)/2;
    if(edge.damage)Object.assign(edge.damage,{material:profile.material,section_area_m2:profile.section_area_m2,
      youngs_modulus_pa:profile.youngs_modulus_pa,yield_strength_pa:profile.yield_strength_pa,
      shear_strength_pa:profile.shear_strength_pa,axial_yield_force_n:profile.axial_yield_force_n,
      shear_force_limit_n:profile.shear_force_limit_n,axial_stiffness_n_per_m:Number(profile.youngs_modulus_pa)*
        Number(profile.section_area_m2)/Math.max(1e-9,Number(edge.rest_length))});}
  body.config.mass=newMass;body.defaults={...(body.defaults||{}),inverse_mass:1/newMass,
    inverse_inertia_roll:reference.inverse_inertia_roll/massScale,
    inverse_inertia_pitch:reference.inverse_inertia_pitch/massScale,
      inverse_inertia_yaw:reference.inverse_inertia_yaw/massScale};
  const energy=ensureVehicleEnergy(body),initialFuel=Number(body.config.fuel_system?.initial_fuel_mass_kg||0);
  energy.dryMassKg=Math.max(1,newMass-initialFuel);energy.lastMassKg=energy.dryMassKg+energy.fuelMassKg-
    energy.configuredShellAssemblyMassKg+energy.activeShellAssemblyMassKg;
  body.chassisProfile=profile.identity;requestParametricVehiclePipelines(body,`chassis · ${profile.identity}`);
  body.gpuEpoch=Number(body.gpuEpoch||0)+1;body.gpuStateDirty=true;
}
function enterParametricDamageMode(body,reason){
  const damage=ensureVehicleDamage(body);if(damage.mode==="parametric-damage")return;
  damage.mode="parametric-damage";damage.reason=reason;damage.revision+=1;
  if(vehicleGpu?.residentGraph)requestParametricVehiclePipelines(body,`damage · ${reason}`);
}
function vehicleDriveFractions(body){
  const state=body.transmission||{},damage=ensureVehicleDamage(body),health=damage.halfshaftHealth,
    axleHealth={front:Number(health.front_left>0||health.front_right>0),rear:Number(health.rear_left>0||health.rear_right>0)},
    authoredFront=Math.max(.05,Math.min(.95,Number(state.frontDriveShare??body.config.drivetrain.front_drive_fraction??.5)));
  let frontShare=state.centerDiffLock ? .5 : authoredFront;
  if(state.centerDiffLock&&axleHealth.front!==axleHealth.rear)frontShare=axleHealth.front?1:0;
  const axle=(name,share,locked)=>{const left=health[`${name}_left`],right=health[`${name}_right`];
    if(locked){const total=left+right;return total>0?[share*left/total,share*right/total]:[0,0];}
    if(left<=0||right<=0)return [0,0];return [share*.5,share*.5];},front=axle("front",frontShare,state.frontDiffLock),
    rear=axle("rear",1-frontShare,state.rearDiffLock);
  return {front_left:front[0],front_right:front[1],rear_left:rear[0],rear_right:rear[1]};
}
function updateVehicleDamage(body,dt){
  const damage=ensureVehicleDamage(body),c=body.config,s=c.suspension,names=["front_left","front_right","rear_left","rear_right"],
    travel=Number(s.travel),bumpStart=travel*.82,bumpSpan=Math.max(.01,travel-bumpStart),
    bumpStiffness=Number(s.stiffness)*6.5,inverseMass=Number(body.defaults?.inverse_mass||0),
    inverseInertia=["roll","pitch","yaw"].map(axis=>Number(body.defaults?.[`inverse_inertia_${axis}`]||0)),
    offsets=[[c.wheels.wheelbase_half_length,-c.wheels.track_half_width],[c.wheels.wheelbase_half_length,c.wheels.track_half_width],
      [-c.wheels.wheelbase_half_length,-c.wheels.track_half_width],[-c.wheels.wheelbase_half_length,c.wheels.track_half_width]];
  let changed=false,totalBump=0;
  names.forEach((name,index)=>{const compression=Number(body.compressions?.[name]||0),over=Math.max(0,compression-bumpStart),
      bumpForce=bumpStiffness*over*over/bumpSpan,load=Math.abs(Number(body.springForces?.[index]||0))+bumpForce,
      yieldLoad=Number(s.stiffness)*bumpStart*1.28,fractureLoad=Number(s.stiffness)*travel*4.2;
    if(bumpForce>1){enterParametricDamageMode(body,`${name} bump stop`);totalBump+=bumpForce;
      const up=rotateBodyVector(body,[0,1,0]),force=up.map(value=>value*bumpForce),arm=rotateBodyVector(body,[offsets[index][0],0,offsets[index][1]]),
        torque=[arm[1]*force[2]-arm[2]*force[1],arm[2]*force[0]-arm[0]*force[2],arm[0]*force[1]-arm[1]*force[0]];
      for(let axis=0;axis<3;axis++)body.velocity[axis]+=dt*force[axis]*inverseMass;
      body.rollVelocity=(body.rollVelocity||0)+dt*torque[0]*inverseInertia[0];
      body.pitchVelocity=(body.pitchVelocity||0)+dt*torque[2]*inverseInertia[1];}
    if(load>yieldLoad){enterParametricDamageMode(body,`${name} spring yielded`);const before=damage.springPlasticSet[name];
      damage.springPlasticSet[name]=Math.min(travel*.28,before+dt*.004*Math.min(8,load/yieldLoad-1));
      changed=changed||damage.springPlasticSet[name]!==before;}
    if(load>fractureLoad&&damage.springHealth[name]>0){damage.springHealth[name]=0;changed=true;
      enterParametricDamageMode(body,`${name} coilover fractured`);}
  });
  const torqueByAxle={front:Math.abs(Number(body.powertrain?.frontDifferentialTorque||0)),
    rear:Math.abs(Number(body.powertrain?.rearDifferentialTorque||0))};
  for(const name of names){const edge=(c.mechanical_graph?.edges||[]).find(item=>item.identity===`drivetrain.${name}_halfshaft`),
      limit=Number(edge?.torsional_fracture_torque_nm||6500),torque=torqueByAxle[name.startsWith("front")?"front":"rear"]*.5;
    if(torque>limit&&damage.halfshaftHealth[name]>0){damage.halfshaftHealth[name]=0;changed=true;
      enterParametricDamageMode(body,`${name} halfshaft fractured`);}}
  const impactLoad=Number(c.mass||0)*Math.max(0,Number(body.accelerationMagnitude||0)-18),edges=c.mechanical_graph?.edges||[],
    intactShellMounts=Math.max(1,edges.filter(edge=>edge.identity.startsWith("body_shell.mount.")&&
      !damage.members[edge.identity]?.failed).length),shellMountLoad=Number(body.bodyShellContactLoadN||0)/intactShellMounts;
  for(const edge of edges){const state=damage.members[edge.identity];if(!state||state.failed)continue;
    const corner=names.find(name=>edge.identity.includes(name)),cornerIndex=corner?names.indexOf(corner):-1,
      shellMount=edge.identity.startsWith("body_shell.mount."),engineMount=edge.identity.startsWith("mount.engine."),
      pathLoad=shellMount?shellMountLoad:(cornerIndex>=0?Math.abs(Number(body.springForces?.[cornerIndex]||0)):
        impactLoad/12+(engineMount?Number(body.energy?.combustionStress||0)*5200:0)),
      axial=Number(shellMount?edge.mount_yield_force_n:edge.damage.axial_yield_force_n||Infinity),
      shear=Number(shellMount?edge.mount_fracture_force_n:edge.damage.shear_force_limit_n||Infinity),
      axialStiffness=Number(edge.damage.axial_stiffness_n_per_m||Infinity),natural=Number(edge.damage.natural_rest_length),
      utilization=Math.max(pathLoad/axial,(pathLoad*.38)/shear);
    state.elasticStrain=Math.min(Number(edge.damage.plastic_strain_limit||.0025),
      pathLoad/Math.max(1e-9,axialStiffness*natural));
    if(utilization>1){const before=state.plasticStrain;state.plasticStrain=Math.min(Number(edge.damage.fracture_strain),
      before+dt*.0015*Math.min(12,utilization-1));changed=changed||state.plasticStrain!==before;}
    state.restLength=natural*(1+state.elasticStrain+state.plasticStrain);
    if((shellMount&&pathLoad>=shear)||state.plasticStrain>=Number(edge.damage.fracture_strain)){state.failed=true;changed=true;
      enterParametricDamageMode(body,`${edge.identity} sheared`);}
  }
  if(changed){damage.revision+=1;body.gpuEpoch=Number(body.gpuEpoch||0)+1;body.gpuStateDirty=true;
    postMessage({type:"vehicle-damage",identity:body.identity,damage});}
  body.bumpStopForce=totalBump;
}
function solveMechanicalGraph(body){
  const graph=body.config.mechanical_graph;if(!graph)return null;
  const positions=new Map(graph.nodes.map(node=>[node.identity,node.reference_position.map(Number)])),
    fixed=new Set(graph.nodes.filter(node=>node.fixed_to==="chassis").map(node=>node.identity)),
    ch=body.config.chassis,w=body.config.wheels,t=body.config.tires,s=body.config.suspension,
    names=["front_left","front_right","rear_left","rear_right"],damage=ensureVehicleDamage(body);
  names.forEach(name=>{const prefix=`suspension.${name}`,hub=positions.get(`${prefix}.hub`),
      actuator=Number(body.levelingOffsets?.[name]||0),
      target=-ch.clearance-(s.rest_length+actuator-damage.springPlasticSet[name])+(body.compressions[name]||0)+t.radius,delta=target-hub[1];
    graph.nodes.filter(node=>node.generalized_coordinate===`compression_${name}`).forEach(node=>
      positions.get(node.identity)[1]+=delta);});
  const constraints=graph.edges.filter(edge=>(edge.constraint==="rigid-distance"||edge.constraint==="rigid-offset")&&
    !damage.members[edge.identity]?.failed&&
    !(fixed.has(edge.a)&&fixed.has(edge.b)));
  for(let iteration=0;iteration<8;iteration+=1){constraints.forEach(edge=>{const a=positions.get(edge.a),b=positions.get(edge.b);
      if(!a||!b)return;const delta=b.map((value,index)=>value-a[index]),length=Math.max(1e-8,Math.hypot(...delta)),
        error=(length-Number(damage.members[edge.identity]?.restLength??edge.rest_length))/length,aFixed=fixed.has(edge.a),bFixed=fixed.has(edge.b),
        aScale=aFixed?0:bFixed?1:.5,bScale=bFixed?0:aFixed?1:.5;
      for(let axis=0;axis<3;axis+=1){a[axis]+=delta[axis]*error*aScale;b[axis]-=delta[axis]*error*bScale;}});
    names.forEach(name=>{const hub=positions.get(`suspension.${name}.hub`),
        actuator=Number(body.levelingOffsets?.[name]||0),target=-ch.clearance-(s.rest_length+actuator-
          damage.springPlasticSet[name])+(body.compressions[name]||0)+t.radius;hub[1]+=(target-hub[1])*.38;});}
  names.forEach(name=>{const hub=positions.get(`suspension.${name}.hub`),patch=positions.get(`suspension.${name}.contact_patch`);
    patch[0]=hub[0];patch[1]=hub[1]-t.radius;patch[2]=hub[2];});return positions;
}
function terrainSegmentCrossing(start,finish,body,reach){
  const at=fraction=>{const p=start.map((value,index)=>value+(finish[index]-value)*fraction),
      surface=contactSurfaceAt(p[0],p[2],p[1],reach,body.identity);
    return surface?{fraction,p,surface,clearance:p[1]-surface.height}:null;};
  let previous=at(0);
  for(let subdivision=1;subdivision<=8;subdivision++){const candidate=at(subdivision/8);
    if(previous&&candidate&&previous.clearance>=0&&candidate.clearance<=0){let lower=previous.fraction,
        upper=candidate.fraction,hit=candidate;
      for(let iteration=0;iteration<8;iteration++){const middle=at((lower+upper)/2);
        if(middle&&middle.clearance<=0){upper=middle.fraction;hit=middle;}else lower=(lower+upper)/2;}
      return {...hit.surface,fraction:upper,point:[hit.p[0],hit.surface.height,hit.p[2]]};}
    previous=candidate;
  }return null;
}
function radialTireContact(body,worldHub,forwardAxis,rightAxis,steer,radius,width,travel,reach,angular,dt){
  const cs=Math.cos(steer),ss=Math.sin(steer),rollingAxis=[
      forwardAxis[0]*cs+rightAxis[0]*ss,forwardAxis[1]*cs+rightAxis[1]*ss,forwardAxis[2]*cs+rightAxis[2]*ss],
    axle=[rightAxis[0]*cs-forwardAxis[0]*ss,rightAxis[1]*cs-forwardAxis[1]*ss,
      rightAxis[2]*cs-forwardAxis[2]*ss],down=rotateBodyVector(body,[0,-1,0]),
    radialAngles=[-.95,-.48,0,.48,.95],lateralFractions=[-.38,0,.38],sampleCount=radialAngles.length*lateralFractions.length,
    probes=Array(sampleCount).fill(0),pointSum=[0,0,0],normalSum=[0,0,0],radialSum=[0,0,0],positionSum=[0,0,0];
  let probeIndex=0,weightSum=0,penetrationSum=0,lateralSum=0,angleSinSum=0,angleCosSum=0,dominant=null;
  for(const angle of radialAngles){const ca=Math.cos(angle),sa=Math.sin(angle),radial=[
      down[0]*ca+rollingAxis[0]*sa,down[1]*ca+rollingAxis[1]*sa,down[2]*ca+rollingAxis[2]*sa];
    for(const lateralFraction of lateralFractions){const index=probeIndex++,probe=[
        worldHub[0]+radial[0]*radius+axle[0]*width*lateralFraction,
        worldHub[1]+radial[1]*radius+axle[1]*width*lateralFraction,
        worldHub[2]+radial[2]*radius+axle[2]*width*lateralFraction],
      attachment=probe.map((value,index)=>value-body.position[index]),probeVelocity=[
        body.velocity[0]+angular[1]*attachment[2]-angular[2]*attachment[1],
        body.velocity[1]+angular[2]*attachment[0]-angular[0]*attachment[2],
        body.velocity[2]+angular[0]*attachment[1]-angular[1]*attachment[0]],
      nextProbe=probe.map((value,index)=>value+probeVelocity[index]*dt),
      currentSurface=contactSurfaceAt(probe[0],probe[2],probe[1],reach,body.identity),
      spatialHit=terrainSegmentCrossing(worldHub,probe,body,reach),
      temporalHit=spatialHit?null:terrainSegmentCrossing(probe,nextProbe,body,reach),
      hit=spatialHit||temporalHit,surface=hit||currentSurface;if(!surface)continue;
      const fraction=temporalHit?temporalHit.fraction:0,hubAttachment=worldHub.map((value,index)=>value-body.position[index]),
        hubVelocity=[body.velocity[0]+angular[1]*hubAttachment[2]-angular[2]*hubAttachment[1],
          body.velocity[1]+angular[2]*hubAttachment[0]-angular[0]*hubAttachment[2],
          body.velocity[2]+angular[0]*hubAttachment[1]-angular[1]*hubAttachment[0]],
        evaluationHub=worldHub.map((value,index)=>value+hubVelocity[index]*dt*fraction),
        evaluationPosition=body.position.map((value,index)=>value+body.velocity[index]*dt*fraction),
        point=hit?hit.point:[probe[0],surface.height,probe[2]],hubToSurface=evaluationHub.map((value,index)=>value-point[index]),
        normalDistance=hubToSurface.reduce((sum,value,index)=>sum+value*surface.normal[index],0),
        radialDistance=Math.max(1e-8,Math.hypot(...hubToSurface)),normalAlignment=normalDistance/radialDistance;
      if(normalDistance>radius+travel+.025||normalAlignment<.12)continue;
      const endSurface=temporalHit?contactSurfaceAt(nextProbe[0],nextProbe[2],nextProbe[1],
          reach+Math.hypot(...probeVelocity)*dt,body.identity):null,
        endPoint=endSurface?[nextProbe[0],endSurface.height,nextProbe[2]]:nextProbe,
        sweptCompression=endSurface?Math.max(0,endPoint.map((value,index)=>value-nextProbe[index]).reduce(
          (sum,value,index)=>sum+value*endSurface.normal[index],0)):0,
        penetration=Math.max(0,radius-normalDistance,sweptCompression);probes[index]=penetration;if(penetration<=0&&!hit)continue;
      const weight=Math.max(penetration,hit?1e-5:0);weightSum+=weight;penetrationSum+=penetration;
      for(let axis=0;axis<3;axis++){pointSum[axis]+=point[axis]*weight;normalSum[axis]+=surface.normal[axis]*weight;
        radialSum[axis]+=radial[axis]*weight;positionSum[axis]+=evaluationPosition[axis]*weight;}
      lateralSum+=lateralFraction*weight;angleSinSum+=Math.sin(angle)*weight;angleCosSum+=Math.cos(angle)*weight;
      if(!dominant||weight>dominant.weight)dominant={...surface,weight,crossingFraction:fraction};
    }
  }if(weightSum<=1e-9)return {integral:null,probes,activeProbeCount:0};
  const unit=value=>{const length=Math.max(1e-9,Math.hypot(...value));return value.map(component=>component/length);},
    point=pointSum.map(value=>value/weightSum),normal=unit(normalSum),radial=unit(radialSum),
    integralCompression=penetrationSum/sampleCount,normalDistance=radius-integralCompression,
    angle=Math.atan2(angleSinSum,angleCosSum),lateralFraction=lateralSum/weightSum;
  return {integral:{...dominant,point,normal,axle,radial,angle,lateralFraction,normalDistance,
    normalAlignment:Math.max(0,unit(worldHub.map((value,index)=>value-point[index])).reduce(
      (sum,value,index)=>sum+value*normal[index],0)),penetration:integralCompression,score:integralCompression,
      evaluationPosition:positionSum.map(value=>value/weightSum)},
    probes,activeProbeCount:probes.filter(value=>value>0).length};
}
function wheelContactRecords(body,dt){
  const c=body.config,w=c.wheels,ch=c.chassis,s=c.suspension,t=c.tires,
    names=["front_left","front_right","rear_left","rear_right"],
    graphPositions=solveMechanicalGraph(body),offsets=[[w.wheelbase_half_length,-w.track_half_width],
      [w.wheelbase_half_length,w.track_half_width],[-w.wheelbase_half_length,-w.track_half_width],
      [-w.wheelbase_half_length,w.track_half_width]],
    forwardAxis=rotateBodyVector(body,[1,0,0]),rightAxis=rotateBodyVector(body,[0,0,1]),
    angular=[forwardAxis[0]*(body.rollVelocity||0)+rightAxis[0]*(body.pitchVelocity||0),
      body.yawVelocity||0,forwardAxis[2]*(body.rollVelocity||0)+rightAxis[2]*(body.pitchVelocity||0)],
    wheelSteerAngles=body.wheelSteerAngles||{},
    centerOfMass=c.mechanical_graph?.load_audit?.center_of_mass||[0,0,0],damage=ensureVehicleDamage(body);
  return names.map((name,i)=>{const o=offsets[i],localHub=graphPositions?.get(`suspension.${name}.hub`)||
      [o[0],-ch.clearance-s.rest_length+t.radius,o[1]],
    actuator=Number(body.levelingOffsets?.[name]||0),hubOffset=rotateBodyVector(body,localHub),worldHub=body.position.map((value,index)=>value+hubOffset[index]),
    coiloverA=graphPositions?.get(`suspension.${name}.coilover_chassis`),
    coiloverB=graphPositions?.get(`suspension.${name}.lower_ball_joint`),
    coiloverDelta=coiloverA&&coiloverB?coiloverB.map((value,index)=>value-coiloverA[index]):[0,-1,0],
    linkageMotionRatio=Math.max(.25,Math.min(1.25,Math.abs(coiloverDelta[1])/Math.max(1e-8,Math.hypot(...coiloverDelta)))),
    wx=worldHub[0],wz=worldHub[2],reach=ch.clearance+s.rest_length+Math.abs(actuator)+s.travel+t.radius+.08,
    steer=Number(wheelSteerAngles[name]??(i<2?body.frontKnuckleSteerAngle:body.rearKnuckleSteerAngle)??0),
    radialContact=radialTireContact(body,worldHub,forwardAxis,rightAxis,steer,
      Number(t.radius),Number(t.width),Number(s.travel),reach,angular,dt),candidate=radialContact.integral,n=candidate?.normal||[0,1,0],
    suspensionDown=rotateBodyVector(body,[0,-1,0]),alignment=-(suspensionDown[0]*n[0]+suspensionDown[1]*n[1]+
      suspensionDown[2]*n[2]),surfacePoint=candidate?.point||[wx,0,wz],contactOrigin=candidate?.evaluationPosition||body.position,
    originToSurface=surfacePoint.map((value,index)=>value-contactOrigin[index]),distanceAlongSuspension=originToSurface.reduce((sum,value,index)=>
      sum+value*suspensionDown[index],0),geometricCompression=Math.max(0,Math.min(s.travel,
      ch.clearance+(s.rest_length+Number(body.levelingOffsets?.[name]||0)-damage.springPlasticSet[name])-
        distanceAlongSuspension)),supportWeight=candidate&&distanceAlongSuspension>0
        ?Math.max(c2Unit(geometricCompression/.025)*c2Unit((alignment-.18)/.34),
          c2Unit(Number(candidate.penetration||0)/.012)):0,
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
      value-contactOrigin[index]-worldCenterOfMass[index]),
    pointVelocity=[body.velocity[0]+angular[1]*attachment[2]-angular[2]*attachment[1],
      body.velocity[1]+angular[2]*attachment[0]-angular[0]*attachment[2],
      body.velocity[2]+angular[0]*attachment[1]-angular[1]*attachment[0]],
    forwardSpeed=pointVelocity.reduce((sum,value,axis)=>sum+value*forward[axis],0),
    lateralSpeed=pointVelocity.reduce((sum,value,axis)=>sum+value*right[axis],0);
    return {dt,support:supportWeight,hub_height:worldHub[1],hub_velocity_y:pointVelocity[1],
      chassis_velocity_y:body.velocity[1],roll_velocity:body.rollVelocity||0,pitch_velocity:body.pitchVelocity||0,
      wheelbase_half_length:w.wheelbase_half_length,track_half_width:w.track_half_width,
      corner_front_sign:i<2?1:-1,corner_side_sign:i%2===0?-1:1,
      geometric_compression:geometricCompression,suspension_alignment:alignment,
      previous_compression:body.compressions[name]||0,surface_height:support?.height||0,
      contact_x:surfacePoint[0],contact_z:surfacePoint[2],runtime_part_id:support?.runtimePartId||0,
      radial_contact_angle:support?.angle||0,radial_lateral_fraction:support?.lateralFraction||0,
      radial_probe_active_count:radialContact.activeProbeCount,radial_probe_penetrations:radialContact.probes,
      normal_x:n[0],normal_y:n[1],normal_z:n[2],forward_x:forward[0],forward_y:forward[1],forward_z:forward[2],
      right_x:right[0],right_y:right[1],right_z:right[2],slip_longitudinal:forwardSpeed-(body.wheelOmegas?.[name]||0)*t.radius,
      slip_lateral:lateralSpeed,
      tire_radial_compression:support?Math.max(0,Number(support.penetration||0)):0,
      tire_radial_velocity:support?pointVelocity.reduce((sum,value,axis)=>sum+value*n[axis],0):0,
      attachment_x:attachment[0],attachment_y:attachment[1],attachment_z:attachment[2],
      corner_weight:c.mass*Math.abs(c.world.gravity)*c.mass_distribution[name],suspension_rest_length:s.rest_length,
      chassis_clearance:ch.clearance,suspension_travel:s.travel,
      spring_stiffness:s.stiffness*damage.springHealth[name],
      linkage_motion_ratio:linkageMotionRatio,
      pneumatic_compression_damping:s.pneumatic_compression_damping,
      pneumatic_rebound_damping:s.pneumatic_rebound_damping,pneumatic_efficiency:s.pneumatic_efficiency,
      active_damping_minimum_scale:s.active_damping_minimum_scale,
      active_damping_maximum_scale:s.active_damping_maximum_scale,
      active_damping_body_velocity_gain_s_per_m:s.active_damping_body_velocity_gain_s_per_m,
      active_damping_rebound_release_gain_s_per_m:s.active_damping_rebound_release_gain_s_per_m,
      maximum_compression_speed:s.maximum_compression_speed,
      tire_pressure:t.pressure_pa,minimum_contact_area:t.minimum_contact_area,
      tire_radial_stiffness:t.radial_stiffness_n_per_m*Math.sqrt(Math.max(.2,Math.min(1.8,
        Number(t.pressure_pa)/Math.max(1,Number(ensureVehicleEnergy(body).referenceTirePressurePa||155000))))),
      tire_radial_damping:t.radial_damping_n_s_per_m,
      maximum_contact_area:t.maximum_contact_area,mu_static:t.static_friction,mu_kinetic:t.kinetic_friction,
      load_sensitivity:t.load_sensitivity,longitudinal_stiffness:t.longitudinal_stiffness,
      lateral_stiffness:t.lateral_stiffness,slip_transition_speed:t.slip_transition_speed};});
}
function runScalarWasm(wasm,abi,inputs){
  if(!wasm||!abi)throw new Error("compiled vehicle Wasm fallback is not initialized");
  const view=new DataView(wasm.exports.memory.buffer);
  abi.input_names.forEach((name,index)=>view.setFloat64(Number(abi.input_offsets[index]),
    Number(inputs[name]??0),true));
  wasm.exports[abi.entrypoint](0);const output={};
  abi.output_names.forEach((name,index)=>output[name]=view.getFloat64(Number(abi.output_offsets[index]),true));
  return output;
}
function applyVehicleWasmOutput(body,output,contacts,contactOutputs,previousPosition,dt){
  const names=["front_left","front_right","rear_left","rear_right"];
  body.position=[output.position_x_next,output.position_y_next,output.position_z_next];
  body.velocity=[output.velocity_x_next,output.velocity_y_next,output.velocity_z_next];
  body.roll=output.roll_next;body.pitch=output.pitch_next;body.yaw=output.yaw_next;
  body.rollVelocity=output.roll_velocity_next;body.pitchVelocity=output.pitch_velocity_next;
  body.yawVelocity=output.yaw_velocity_next;
  names.forEach((name,index)=>{body.wheelOmegas[name]=output[`wheel_omega_${name}_next`];
    body.compressions[name]=output[`compression_${name}_next`];
    body.previousSlips[name]=output[`slip_longitudinal_${name}_next`];});
  body.springForces=names.map(name=>output[`spring_force_${name}`]);
  body.tractionScales=names.map(name=>output[`traction_scale_${name}`]);
  body.brakeScales=names.map(name=>output[`brake_scale_${name}`]);
  body.damperScales=names.map(name=>output[`damper_scale_${name}`]);
  body.contactAreas=contactOutputs.map(value=>Number(value.contact_area||0));
  body.frictionUtilizations=contactOutputs.map((value,index)=>{const record=contacts[index],force=[
      value.chassis_force_x,value.chassis_force_y,value.chassis_force_z],normal=[record.normal_x,record.normal_y,record.normal_z],
      load=Math.max(0,force.reduce((sum,component,axis)=>sum+component*normal[axis],0)),
      tangent=Math.sqrt(Math.max(0,force.reduce((sum,component)=>sum+component*component,0)-load*load));
    return tangent/Math.max(1e-5,Number(record.mu_static)*load);});
  body.contactModes=body.contactAreas.map((area,index)=>area<=0?0:
    body.frictionUtilizations[index]<.78?1:body.frictionUtilizations[index]<=1.08?2:3);
  body.powertrain={engineTorque:output.engine_torque,clutchTorque:output.clutch_torque,
    transmissionOutputTorque:output.transmission_output_torque,drivelineTorque:output.driveline_torque,
    frontDifferentialTorque:output.front_differential_torque,rearDifferentialTorque:output.rear_differential_torque,
    engineAccelerationTorque:output.engine_acceleration_torque,engineAngularAcceleration:output.engine_angular_acceleration,
    reactionTorque:[output.powertrain_reaction_torque_x,output.powertrain_reaction_torque_y,
      output.powertrain_reaction_torque_z],mountTorque:[output.engine_mount_torque_x,
      output.engine_mount_torque_y,output.engine_mount_torque_z],
    engineAngularSpeed:output.engine_angular_speed_next,engineRPM:output.engine_rpm,
    tractionControlDissipationTorque:output.traction_control_dissipation_torque,
    serviceBrakeReactionTorque:output.service_brake_reaction_torque,
    rollingResistanceReactionTorque:output.rolling_resistance_reaction_torque,
    tireContactReactionTorque:output.tire_contact_reaction_torque,
    drivetrainChassisReactionTorque:output.drivetrain_chassis_reaction_torque};
  body.wasmState={...body.wasmState,engine_angular_speed:output.engine_angular_speed_next};
  names.forEach(name=>{body.wasmState[`slip_sensor_velocity_${name}`]=output[`slip_sensor_velocity_${name}_next`];
    body.wasmState[`friction_utilization_${name}`]=output[`friction_utilization_${name}_next`];
    body.wasmState[`friction_utilization_sensor_velocity_${name}`]=
      output[`friction_utilization_sensor_velocity_${name}_next`];});
  // No post-step positional rejection: wheel, cage and body-shell wrenches own
  // contact. This keeps the fallback from adding energy after integration.
}
function residentVehicleWasmStep(body,dt){
  if(!vehicleInstance||!contactInstance)throw new Error("compiled vehicle Wasm fallback is unavailable");
  const previousPosition=[...body.position];
  body.bottomRejected=false;body.supportLatchRejected=false;
  if(resolveWorldBottom(body,previousPosition)&&body.bottomRejected)
    resetVehicleDrivetrainState(body,"vehicle entered the world-bottom guard before integration");
  // The pedal/cable actuator already owns any optional feathering and elastic
  // response. A second generic throttle slew here made the Wasm fallback lag
  // behind the resident GPU path and double-filtered the physical control.
  body.appliedThrottle=Number(body.effectiveThrottle??body.controls?.throttle??0);
  body.throttleVelocity=0;
  const controls=body.config.controls,frequency=Number(controls.input_response_frequency_hz),
    damping=Number(controls.input_response_damping_ratio);
  // prepareVehicleControls owns the ECU speed-sensitive steering command for
  // both resident GPU and Wasm paths. Do not filter the same column twice.
  body.appliedSteering=Number(body.appliedSteering??body.controls?.steering??0);
  body.appliedBrake=secondOrderChannel(body,"appliedBrake","brakeVelocity",
    Number(body.effectiveBrake??body.controls?.brake??0),frequency,damping,dt);
  if(!Number.isFinite(body.driveDirection)||body.driveDirection===0)body.driveDirection=1;
  const contacts=wheelContactRecords(body,dt),contactOutputs=contacts.map(record=>
      runScalarWasm(contactInstance,contactAbi,{...(body.defaults||{}),...record})),
    names=["front_left","front_right","rear_left","rear_right"],wheelForce=[0,0,0],wheelTorque=[0,0,0];
  contactOutputs.forEach(output=>{for(let axis=0;axis<3;axis++){wheelForce[axis]+=Number(output[`chassis_force_${"xyz"[axis]}`]||0);
      wheelTorque[axis]+=Number(output[`chassis_torque_${"xyz"[axis]}`]||0);}});
  const cage=vehicleCageContactWrench(body,dt);for(let axis=0;axis<3;axis++){wheelForce[axis]+=cage.force[axis];
    wheelTorque[axis]+=cage.torque[axis];}body.contactRuntimePartId=cage.contactId||
      contacts.find(record=>record.runtime_part_id)?.runtime_part_id||0;
  body.longitudinalForces=contactOutputs.map((output,index)=>["x","y","z"].reduce((sum,axis)=>
    sum+Number(output[`chassis_force_${axis}`]||0)*Number(contacts[index][`forward_${axis}`]||0),0));
  const transmission=updateVehicleTransmission(body,dt),driveFractions=vehicleDriveFractions(body),
    input={...(body.defaults||{}),...(body.wasmState||{}),dt,
    position_x:body.position[0],position_y:body.position[1],position_z:body.position[2],
    velocity_x:body.velocity[0],velocity_y:body.velocity[1],velocity_z:body.velocity[2],
    roll:body.roll||0,pitch:body.pitch||0,yaw:body.yaw||0,roll_velocity:body.rollVelocity||0,
    pitch_velocity:body.pitchVelocity||0,yaw_velocity:body.yawVelocity||0,
    yaw_cos:Math.cos(body.yaw||0),yaw_sin:Math.sin(body.yaw||0),throttle:body.appliedThrottle,
    steering:body.appliedSteering,brake:body.appliedBrake,drive_direction:body.driveDirection,
    forward_gear_ratio:transmission.forwardRatio,reverse_gear_ratio:transmission.reverseRatio,
    transfer_case_ratio:transmission.transferCaseRatio,
    traction_control_enabled:body.transmission?.tractionControlEnabled===false?0:1,
    abs_enabled:body.transmission?.absEnabled===false?0:1,
    traction_control_authority:Math.max(0,Math.min(1,Number(body.transmission?.tractionControlAuthority??1))),
    abs_authority:Math.max(0,Math.min(1,Number(body.transmission?.absAuthority??1))),
    differential_lock:(body.transmission?.frontDiffMode==="locked"||body.transmission?.rearDiffMode==="locked")?1:0,
    front_differential_lock:body.transmission?.frontDiffMode==="locked"?1:
      body.transmission?.frontDiffMode==="limited-slip"?.32:body.transmission?.frontDiffLock?1:0,
    rear_differential_lock:body.transmission?.rearDiffMode==="locked"?1:
      body.transmission?.rearDiffMode==="limited-slip"?.32:body.transmission?.rearDiffLock?1:0,
    center_differential_lock:body.transmission?.centerDiffMode==="locked"?1:
      body.transmission?.centerDiffMode==="limited-slip"?.32:body.transmission?.centerDiffLock?1:0,
    front_differential_brake:body.transmission?.frontDifferentialBrake?1:0,
    rear_differential_brake:Math.max(body.transmission?.rearDifferentialBrake?1:0,Number(
      body.rearDifferentialBrakeCommand||0)),
    drive_fraction_front_left:driveFractions.front_left,drive_fraction_front_right:driveFractions.front_right,
    drive_fraction_rear_left:driveFractions.rear_left,drive_fraction_rear_right:driveFractions.rear_right,
    contact_wrench_force_x:wheelForce[0],contact_wrench_force_y:wheelForce[1],contact_wrench_force_z:wheelForce[2],
    contact_wrench_torque_x:wheelTorque[0],contact_wrench_torque_y:wheelTorque[1],contact_wrench_torque_z:wheelTorque[2]};
  names.forEach((name,index)=>{const record=contacts[index],output=contactOutputs[index],force=[
      output.chassis_force_x,output.chassis_force_y,output.chassis_force_z],normal=[record.normal_x,record.normal_y,record.normal_z],
      load=Math.max(0,force.reduce((sum,value,axis)=>sum+value*normal[axis],0)),
      tangent=Math.sqrt(Math.max(0,force.reduce((sum,value)=>sum+value*value,0)-load*load));
    Object.assign(input,{[`compression_${name}`]:body.compressions[name]||0,
      [`wheel_height_${name}`]:record.surface_height,[`wheel_support_${name}`]:record.support,
      [`target_compression_${name}`]:record.geometric_compression,[`wheel_omega_${name}`]:body.wheelOmegas[name]||0,
      [`longitudinal_force_${name}`]:body.longitudinalForces[index],[`slip_longitudinal_${name}`]:record.slip_longitudinal,
      [`previous_slip_longitudinal_${name}`]:body.previousSlips[name]||0,
      [`measured_friction_utilization_${name}`]:tangent/Math.max(1e-5,Number(record.mu_static)*load),
      [`linkage_motion_ratio_${name}`]:record.linkage_motion_ratio,
      [`brake_lock_${name}`]:body.brakeLocks?.[name]?1:0});});
  const output=runScalarWasm(vehicleInstance,vehicleAbi,input);
  if(!Object.values(output).every(Number.isFinite)){resetVehicleDrivetrainState(body,
      "resident Wasm produced a non-finite output");body.accelerationMagnitude=0;return;}
  body.radialProbePenetrations=contacts.map(record=>record.radial_probe_penetrations);
  body.radialProbeActiveCounts=contacts.map(record=>record.radial_probe_active_count);
  applyVehicleWasmOutput(body,output,contacts,contactOutputs,previousPosition,dt);
  body.accelerationMagnitude=Math.hypot(body.velocity[0]-input.velocity_x,body.velocity[1]-input.velocity_y,
    body.velocity[2]-input.velocity_z)/Math.max(dt,1e-6);
}
function resolveVehicleSolidContact(body,dt){
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
}
function vehicleExteriorContactSamples(body){
  const graph=body.config.mechanical_graph,damage=ensureVehicleDamage(body),nodePositions=new Map(
      (graph?.nodes||[]).map(node=>[node.identity,node.reference_position.map(Number)])),
    cageNodes=(graph?.nodes||[]).filter(node=>node.kind==="roll-cage-node").map(node=>({local:node.reference_position.map(Number),shell:false})),
    cageBars=(graph?.edges||[]).filter(edge=>edge.identity.startsWith("cage.")&&!damage.members[edge.identity]?.failed)
      .map(edge=>{const a=nodePositions.get(edge.a),b=nodePositions.get(edge.b);return a&&b?{local:a.map((value,index)=>(value+b[index])*.5),shell:false}:null;}).filter(Boolean),
    samples=[...cageNodes,...cageBars];
  if(body.bodyShell!=="bare-frame"){const chassis=body.config.chassis,halfLength=Number(chassis.half_length),
      halfWidth=Number(chassis.half_width),mountEdges=(graph?.edges||[]).filter(edge=>edge.identity.startsWith("body_shell.mount.")&&
        !damage.members[edge.identity]?.failed);
    if(mountEdges.length){for(const x of [-.72,0,.72])for(const z of [-1.02,1.02])samples.push({
      local:[halfLength*x,x===0?.55:.31,halfWidth*z],shell:true});
      for(const x of [-.55,.38])samples.push({local:[halfLength*x,x<0?.70:.39,0],shell:true});}
  }return samples;
}
function vehicleCageContactWrench(body,dt){
  const c=body.config,layer=c.solid_contact,graph=c.mechanical_graph,shellLayer=c.body_shell||{},
    centerOfMass=graph?.load_audit?.center_of_mass||[0,0,0],force=[0,0,0],torque=[0,0,0],samples=vehicleExteriorContactSamples(body);
  let count=0,contactId=0,shellLoad=0;if(!samples.length)return{force,torque,count,contactId};
  const forward=rotateBodyVector(body,[1,0,0]),right=rotateBodyVector(body,[0,0,1]),
    angular=[forward[0]*(body.rollVelocity||0)+right[0]*(body.pitchVelocity||0),
      body.yawVelocity||0,forward[2]*(body.rollVelocity||0)+right[2]*(body.pitchVelocity||0)],
    share=Math.max(1,samples.reduce((sum,sample)=>{const local=sample.local,
      radius=Number(sample.shell?shellLayer.contact_radius_m:layer.cage_contact_radius),
      offset=rotateBodyVector(body,local),point=body.position.map(
        (value,index)=>value+offset[index]),surface=contactSurfaceAt(point[0],point[2],body.position[1],2.5,body.identity);
      return sum+(surface&&surface.height+radius-point[1]>0?1:0);},0));
  for(const sample of samples){const local=sample.local,radius=Number(sample.shell?shellLayer.contact_radius_m:layer.cage_contact_radius),
      stiffness=Number(sample.shell?shellLayer.contact_stiffness_n_per_m:layer.cage_contact_stiffness),
      damping=Number(sample.shell?shellLayer.contact_damping_n_s_per_m:layer.cage_contact_damping),
      maximumForce=Number(sample.shell?shellLayer.contact_maximum_force_n:layer.cage_contact_maximum_force),offset=rotateBodyVector(body,local),
      attachment=rotateBodyVector(body,local.map((value,index)=>value-Number(centerOfMass[index]||0))),
      point=body.position.map((value,index)=>value+offset[index]),surface=contactSurfaceAt(
        point[0],point[2],body.position[1],2.5,body.identity);if(!surface)continue;
    const penetration=surface.height+radius-point[1];if(penetration<=0)continue;
    const n=surface.normal,pointVelocity=[
      body.velocity[0]+angular[1]*attachment[2]-angular[2]*attachment[1],
      body.velocity[1]+angular[2]*attachment[0]-angular[0]*attachment[2],
      body.velocity[2]+angular[0]*attachment[1]-angular[1]*attachment[0]],
      normalSpeed=pointVelocity.reduce((sum,value,index)=>sum+value*n[index],0),
      normalForce=c2Clamp(stiffness*penetration-damping*normalSpeed,0,maximumForce,80),
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
    count+=1;if(sample.shell)shellLoad+=normalForce;contactId=surface.runtimePartId||contactId;
  }body.bodyShellContactLoadN=shellLoad;return{force,torque,count,contactId};
}
function resolveVehicleCagePenetration(body,dt){
  const c=body.config,layer=c.solid_contact,shellLayer=c.body_shell||{},samples=vehicleExteriorContactSamples(body);let resolved=0;
  for(let iteration=0;iteration<3;iteration+=1){let deepest=null;
    for(const sample of samples){const local=sample.local,radius=Number(sample.shell?shellLayer.contact_radius_m:layer.cage_contact_radius),
        offset=rotateBodyVector(body,local),
        point=body.position.map((value,index)=>value+offset[index]),surface=contactSurfaceAt(
          point[0],point[2],body.position[1],2.5,body.identity),penetration=surface?surface.height+radius-point[1]:0;
      if(penetration>0&&(!deepest||penetration>deepest.penetration))deepest={penetration,normal:surface.normal,
        runtimePartId:surface.runtimePartId||0};}
    if(!deepest)break;const correction=Math.min(deepest.penetration,Number(layer.maximum_correction_speed)*dt);
    for(let axis=0;axis<3;axis+=1)body.position[axis]+=deepest.normal[axis]*correction;
    const inward=body.velocity.reduce((sum,value,index)=>sum+value*deepest.normal[index],0);
    if(inward<0)for(let axis=0;axis<3;axis+=1)body.velocity[axis]-=deepest.normal[axis]*inward*(1+Number(layer.restitution));
    body.contactRuntimePartId=deepest.runtimePartId||body.contactRuntimePartId;resolved+=1;
  }body.cageContactCount=resolved;return resolved;
}
function resolveVehicleSuspensionTravelStop(body){
  const c=body.config,w=c.wheels,ch=c.chassis,s=c.suspension,
    names=["front_left","front_right","rear_left","rear_right"],
    offsets=[[w.wheelbase_half_length,-w.track_half_width],[w.wheelbase_half_length,w.track_half_width],
      [-w.wheelbase_half_length,-w.track_half_width],[-w.wheelbase_half_length,w.track_half_width]],
    reach=ch.clearance+s.rest_length+s.travel+Number(c.chassis_leveling?.maximum_corner_offset_m||0)+.08;let minimumBodyY=-Infinity,contactId=0;
  for(let index=0;index<offsets.length;index++){const offset=offsets[index],actuator=Number(body.levelingOffsets?.[names[index]]||0),
    attachment=rotateBodyVector(body,[offset[0],-ch.clearance,offset[1]]),
    support=contactSurfaceAt(body.position[0]+attachment[0],body.position[2]+attachment[2],
      body.position[1],reach,body.identity),down=rotateBodyVector(body,[0,-1,0]),alignment=support?-(
      down[0]*support.normal[0]+down[1]*support.normal[1]+down[2]*support.normal[2]):0;
    if(!support||alignment<=.18)continue;
    minimumBodyY=Math.max(minimumBodyY,support.height+s.rest_length+actuator-s.travel-attachment[1]);
    contactId=support.runtimePartId||contactId;}
  if(!Number.isFinite(minimumBodyY)||body.position[1]>=minimumBodyY)return false;
  const penetration=minimumBodyY-body.position[1],engagement=c2Unit(penetration/.025),
    correction=c2Clamp(penetration,0,Number(s.maximum_compression_speed)*dt,.004);
  body.suspensionStopPenetration=penetration;body.position[1]+=correction*engagement;
  if(body.velocity[1]<0)body.velocity[1]+=c2Positive(-body.velocity[1],.18)*engagement;
  body.contactRuntimePartId=contactId||body.contactRuntimePartId;return true;
}
function resolveWorldBottom(body,previousPosition){
  // Vehicles have no terrain underside or positional recovery boundary. Their
  // tyres and exterior samples must resolve the actual swept top-skin crossing;
  // a failed solve remains observable and recoverable by the explicit respawn.
  if(body.kind==="vehicle"){body.supportSurfaceLatch=null;body.bottomRejected=false;return false;}
  const previous=Array.isArray(previousPosition)?previousPosition:[body.position[0],Number(previousPosition),body.position[2]],
    supportOffset=body.kind==="vehicle"?Math.max(.02,Number(body.config?.chassis?.clearance||0)):
      Math.max(.001,Number(body.radius||0)),support=contactSurfaceAt(body.position[0],body.position[2],
        body.position[1],Number.POSITIVE_INFINITY,body.identity,false),previousSupport=contactSurfaceAt(previous[0],previous[2],
        previous[1],Number.POSITIVE_INFINITY,body.identity,false);
  // Every declared support is a half-space constraint backed by a persistent
  // local-domain latch, never a zero-thickness visual sheet. Once a body enters
  // the inequality, losing a radial probe cannot release it; moving out of that
  // surface's authored x/z mask (or onto another higher support) is the release.
  if(!support)body.supportSurfaceLatch=null;
  if(support){const material=support.material||{},previousBase=previous[1]-supportOffset,
      nextBase=body.position[1]-supportOffset,crossed=previousSupport?.identity===support.identity&&
        previousBase>=previousSupport.height-.012&&nextBase<=support.height+.006,
      entered=nextBase<support.height||crossed,
      maximumSink=Math.max(0,Number(material.maximum_sink_depth_m??.045)),
      equilibriumSink=Math.max(0,Math.min(maximumSink,Number(material.equilibrium_sink_depth_m??maximumSink*.35))),
      minimumBase=support.height-maximumSink;
    if(entered||body.supportSurfaceLatch?.identity===support.identity){body.supportSurfaceLatch={identity:support.identity,
        runtimePartId:support.runtimePartId||0,material:material.identity||"declared-support"};
      if(nextBase<minimumBase){const targetBase=support.height-equilibriumSink,normal=support.normal,
          inward=body.velocity.reduce((sum,value,index)=>sum+value*normal[index],0),
          restitution=Math.max(0,Math.min(1,Number(material.restitution??.08)));
        body.position[1]=targetBase+supportOffset;
        if(inward<0)for(let axis=0;axis<3;axis++)body.velocity[axis]-=normal[axis]*inward*(1+restitution);
        body.contactRuntimePartId=support.runtimePartId||body.contactRuntimePartId;body.supportLatchRejected=true;return true;}
    }}
  const top=Number(worldBottom?.top_y??0),thickness=Math.max(.25,Number(worldBottom?.thickness??8)),
    surfaceHeight=support?.height??null,
    guardDepth=Math.max(.25,Number(worldBottom?.sampled_surface_guard_depth??.75)),
    rejectionTop=surfaceHeight===null?top+supportOffset:surfaceHeight-guardDepth,
    recoveryHeight=surfaceHeight===null?top+supportOffset:surfaceHeight+supportOffset,
    bottom=rejectionTop-thickness,boundary=rejectionTop,nextY=body.position[1],
    swept=previous[1]>=boundary&&nextY<bottom,contained=nextY<boundary&&nextY>=bottom;
  if(!swept&&!contained)return false;
  body.position[1]=recoveryHeight;if(body.velocity[1]<0)body.velocity[1]=0;
  body.bottomRejected=true;body.contactRuntimePartId=0;return true;
}
function updateVehicleTransmission(body,dt){
  const c=body.config,t=c.transmission,p=c.powertrain,d=c.drivetrain,w=c.wheels,tire=c.tires,
    ratios=t.forward_ratios.map(Number),maximum=ratios.length,throttle=Number(body.appliedThrottle||0),
    drivenNames=["front_left","front_right","rear_left","rear_right"],
    wheelSpeed=drivenNames.reduce((sum,name)=>sum+Math.abs(body.wheelOmegas?.[name]||0),0)/drivenNames.length,
    roadSpeed=Math.hypot(body.velocity[0],body.velocity[2]),state=body.transmission||{
      mode:t.mode_default,gear:Number(t.starting_gear),shiftAge:Number(t.minimum_shift_interval_s),reason:"initial-second",
      engagedRatio:ratios[Number(t.starting_gear)-1],ratioVelocity:0,downshiftDemand:0,downshiftDemandVelocity:0,
      lowRange:false,frontDiffLock:false,rearDiffLock:false,centerDiffLock:false,
      frontDiffMode:"open",rearDiffMode:"open",centerDiffMode:"open",
      frontDifferentialBrake:false,rearDifferentialBrake:false,
      tractionControlEnabled:true,absEnabled:true,tiltEnabled:true,tractionControlAuthority:1,absAuthority:1,
      frontDriveShare:Number(d.front_drive_fraction||.5)};
  if(typeof state.tractionControlEnabled!=="boolean")state.tractionControlEnabled=true;
  if(typeof state.absEnabled!=="boolean")state.absEnabled=true;
  if(typeof state.tiltEnabled!=="boolean")state.tiltEnabled=true;
  if(!Number.isFinite(state.tractionControlAuthority))state.tractionControlAuthority=1;
  if(!Number.isFinite(state.absAuthority))state.absAuthority=1;
  if(!["open","limited-slip","locked"].includes(state.frontDiffMode))state.frontDiffMode=state.frontDiffLock?"locked":"open";
  if(!["open","limited-slip","locked"].includes(state.rearDiffMode))state.rearDiffMode=state.rearDiffLock?"locked":"open";
  if(!["open","limited-slip","locked"].includes(state.centerDiffMode))state.centerDiffMode=state.centerDiffLock?"locked":"open";
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
  if(state.mode==="automatic"&&throttle>0&&canShift){const gear=state.gear;
    if(gear===1){if(wheelSpeed>t.upshift_wheel_speed_rad_s[0]&&reserve(2)>=t.upshift_torque_reserve){
      state.gear=2;state.shiftAge=0;state.reason="crawler-clear";}}
    else if(gear<maximum&&wheelSpeed>t.upshift_wheel_speed_rad_s[gear-1]&&
      reserve(gear+1)>=t.upshift_torque_reserve){state.gear+=1;state.shiftAge=0;state.reason="next-ratio-has-reserve";}
    else if(gear>1&&state.downshiftDemand>=Number(t.downshift_commit_level)){
      state.gear-=1;state.shiftAge=0;state.reason=gear===2?"crawler-demand-integrated":"downshift-demand-integrated";
      state.downshiftDemand=0;state.downshiftDemandVelocity=0;}
  }
  const rangeRatio=rangeMultiplier,targetRatio=ratios[state.gear-1]*rangeRatio;
  state.engagedRatio=secondOrderChannel(state,"engagedRatio","ratioVelocity",
    targetRatio,t.ratio_response_frequency_hz,t.ratio_response_damping_ratio,dt);
  state.displayGear=Number(body.driveDirection||1)<0?-1:state.gear;state.torqueReserve=reserve(state.gear);
  body.transmission=state;return{forwardRatio:state.engagedRatio/rangeRatio,
    reverseRatio:Number(t.reverse_ratio),transferCaseRatio:rangeRatio};
}
function applyPendingVehicleCommands(body){
  if(body.pendingControls){body.controls=body.pendingControls;body.pendingControls=null;body.gpuControlsDirty=true;}
  if(body.pendingPowerUnit){const preset=body.pendingPowerUnit;body.pendingPowerUnit=null;
    body.defaults={...(body.defaults||{}),...(preset.parameters||{})};
    body.config.powertrain={...body.config.powertrain,...(preset.configuration||{})};
    body.powerUnitPreset=preset.identity;body.wasmState={...(body.wasmState||{}),
      engine_angular_speed:Number(preset.parameters?.engine_idle_angular_speed||0)};
    requestParametricVehiclePipelines(body,`power unit · ${preset.identity}`);
    body.gpuEpoch=Number(body.gpuEpoch||0)+1;body.gpuStateDirty=true;}
  if(body.pendingChassisProfile){const profile=body.pendingChassisProfile;body.pendingChassisProfile=null;
    applyVehicleChassisProfile(body,profile);}
  if(body.pendingChassisLeveling){body.chassisLeveling={...(body.config.chassis_leveling||{}),
      ...body.pendingChassisLeveling};body.pendingChassisLeveling=null;}
  if(body.pendingSteeringSystem){body.steeringSystem={...(body.config.steering_control||{}),
      ...body.pendingSteeringSystem};body.pendingSteeringSystem=null;body.gpuControlsDirty=true;}
  if(body.pendingVehicleParameters){const parameters=body.pendingVehicleParameters;body.pendingVehicleParameters=null;
    Object.assign(body.defaults||={},parameters);const suspension=body.config.suspension;
    const mapping={suspension_rest_length:"rest_length",suspension_travel:"travel",spring_stiffness:"stiffness",
      pneumatic_compression_damping:"pneumatic_compression_damping",pneumatic_rebound_damping:"pneumatic_rebound_damping",
      pneumatic_efficiency:"pneumatic_efficiency",active_damping_minimum_scale:"active_damping_minimum_scale",
      active_damping_maximum_scale:"active_damping_maximum_scale",
      active_damping_body_velocity_gain_s_per_m:"active_damping_body_velocity_gain_s_per_m",
      active_damping_rebound_release_gain_s_per_m:"active_damping_rebound_release_gain_s_per_m"};
    for(const [parameter,key] of Object.entries(mapping))if(Number.isFinite(parameters[parameter]))suspension[key]=Number(parameters[parameter]);
    requestParametricVehiclePipelines(body,"shock parameter control");body.gpuEpoch=Number(body.gpuEpoch||0)+1;
    body.gpuStateDirty=true;}
  if(body.pendingRecovery){const lift=Number(body.pendingRecovery.lift||.5);body.pendingRecovery=null;
    body.position[1]+=Math.max(.1,lift);body.roll=0;body.pitch=0;body.rollVelocity=0;body.pitchVelocity=0;
    body.yawVelocity=0;body.velocity[1]=Math.max(0,body.velocity[1]);body.gpuEpoch=Number(body.gpuEpoch||0)+1;
    body.gpuStateDirty=true;}
  if(body.pendingRespawn){const command=body.pendingRespawn;body.pendingRespawn=null;
    body.position=[...command.position];body.velocity=[0,0,0];body.roll=Number(command.roll||0);
    body.pitch=Number(command.pitch||0);body.yaw=Number(command.yaw||0);
    body.rollVelocity=0;body.pitchVelocity=0;body.yawVelocity=0;
    body.wheelOmegas={front_left:0,front_right:0,rear_left:0,rear_right:0};
    body.previousSlips={front_left:0,front_right:0,rear_left:0,rear_right:0};
    body.wasmState={};body.effectiveThrottle=0;body.featheredThrottle=0;body.featheredThrottleVelocity=0;
    body.levelingOffsets={front_left:0,front_right:0,rear_left:0,rear_right:0};body.levelingState={};
    body.supportSurfaceLatch=null;body.supportLatchRejected=false;
    body.frontKnuckleSteerAngle=0;body.rearKnuckleSteerAngle=0;
    body.frontKnuckleSteerVelocity=0;body.rearKnuckleSteerVelocity=0;
    body.controls={throttle:0,steering:0,brake:0};
    body.brakeLocks={front_left:false,front_right:false,rear_left:false,rear_right:false};
    body.gpuEpoch=Number(body.gpuEpoch||0)+1;body.gpuStateDirty=true;body.gpuControlsDirty=true;}
  if(body.pendingTransmission){const command=body.pendingTransmission;body.pendingTransmission=null;
    if(command.gearset&&Array.isArray(command.gearset.forward_ratios))
      body.config.transmission={...body.config.transmission,...command.gearset};
    const t=body.config.transmission,state=body.transmission||{mode:t.mode_default,gear:t.starting_gear,
      shiftAge:0,reason:"control"};
    if(command.gearset){state.gear=Math.max(1,Math.min(t.forward_ratios.length,Number(t.starting_gear||1)));
      state.engagedRatio=Number(t.forward_ratios[state.gear-1]);state.ratioVelocity=0;
      state.transmissionPreset=command.transmissionPreset||"custom";state.reason="driver-gearset";}
    if(command.mode==="automatic"){state.mode="automatic";state.reason="driver-auto";}
    if(typeof command.lowRange==="boolean"){state.lowRange=command.lowRange;state.ratioVelocity=0;
      state.reason=command.lowRange?"driver-ultra-low":"driver-high-range";}
    if(typeof command.frontDiffLock==="boolean"){state.frontDiffLock=command.frontDiffLock;
      state.frontDiffMode=command.frontDiffLock?"locked":"open";
      state.reason=command.frontDiffLock?"driver-front-diff-lock":"driver-front-diff-open";}
    if(typeof command.rearDiffLock==="boolean"){state.rearDiffLock=command.rearDiffLock;
      state.rearDiffMode=command.rearDiffLock?"locked":"open";
      state.reason=command.rearDiffLock?"driver-rear-diff-lock":"driver-rear-diff-open";}
    if(typeof command.centerDiffLock==="boolean"){state.centerDiffLock=command.centerDiffLock;
      state.centerDiffMode=command.centerDiffLock?"locked":"open";
      state.reason=command.centerDiffLock?"driver-center-lock":"driver-center-open";}
    for(const axle of ["front","rear","center"]){const key=`${axle}DiffMode`,mode=command[key];
      if(["open","limited-slip","locked"].includes(mode)){state[key]=mode;state[`${axle}DiffLock`]=mode==="locked";
        state.reason=`driver-${axle}-diff-${mode}`;}}
    if(Number.isFinite(command.frontDriveShare)){state.frontDriveShare=Math.max(.05,Math.min(.95,
      Number(command.frontDriveShare)));state.reason="driver-torque-proportion";}
    if(typeof command.smoothLaunch==="boolean"){state.smoothLaunch=command.smoothLaunch;
      state.reason=command.smoothLaunch?"driver-smooth-launch":"driver-direct-launch";}
    if(typeof command.frontDifferentialBrake==="boolean"){state.frontDifferentialBrake=command.frontDifferentialBrake;
      state.reason=command.frontDifferentialBrake?"driver-front-differential-brake-on":"driver-front-differential-brake-off";}
    if(typeof command.rearDifferentialBrake==="boolean"){state.rearDifferentialBrake=command.rearDifferentialBrake;
      state.reason=command.rearDifferentialBrake?"driver-rear-differential-brake-on":"driver-rear-differential-brake-off";}
    if(typeof command.tractionControlEnabled==="boolean"){state.tractionControlEnabled=command.tractionControlEnabled;
      state.reason=command.tractionControlEnabled?"driver-traction-control-on":"driver-traction-control-off";}
    if(typeof command.absEnabled==="boolean"){state.absEnabled=command.absEnabled;
      state.reason=command.absEnabled?"driver-abs-on":"driver-abs-off";}
    if(typeof command.tiltEnabled==="boolean"){state.tiltEnabled=command.tiltEnabled;
      state.reason=command.tiltEnabled?"driver-tilt-on":"driver-tilt-off";}
    if(Number.isFinite(command.tractionControlAuthority)){state.tractionControlAuthority=Math.max(0,
      Math.min(1,Number(command.tractionControlAuthority)));state.reason="driver-traction-authority";}
    if(Number.isFinite(command.absAuthority)){state.absAuthority=Math.max(0,Math.min(1,
      Number(command.absAuthority)));state.reason="driver-abs-authority";}
    if(command.brakeLock&&typeof command.brakeLock.name==="string"){
      body.brakeLocks={...(body.brakeLocks||{}),[command.brakeLock.name]:Boolean(command.brakeLock.locked)};
      state.reason=`driver-${command.brakeLock.name}-brake-${command.brakeLock.locked?"locked":"released"}`;}
    if(command.releaseAllBrakes){body.brakeLocks={front_left:false,front_right:false,rear_left:false,rear_right:false};
      state.frontDifferentialBrake=false;state.rearDifferentialBrake=false;
      state.reason="driver-release-all-brake-holds";}
    if(Number.isFinite(command.gearDelta)){state.mode="manual";state.gear=Math.max(1,
      Math.min(t.forward_ratios.length,Math.round(state.gear+Number(command.gearDelta))));
      state.shiftAge=0;state.reason="driver-manual";}body.transmission=state;body.gpuControlsDirty=true;}
}
function prepareVehicleControls(body,dt){
  const energy=ensureVehicleEnergy(body),driver=ensureVehicleDriverAssistance(body),rawThrottle=Number(body.controls?.throttle||0),
    rawBrake=Math.max(0,Number(body.controls?.brake||0)),mode=(body.drivingModes||[]).find(item=>
      item.identity===driver.drivingMode)||{throttle_exponent:1,throttle_rate_scale:1},
    curvedThrottle=Math.sign(rawThrottle)*Math.pow(Math.min(1,Math.abs(rawThrottle)),Number(mode.throttle_exponent||1)),
    roadSpeed=Math.hypot(Number(body.velocity?.[0]||0),Number(body.velocity?.[2]||0)),
    forward=rotateBodyVector(body,[1,0,0]),forwardSpeed=Number(body.velocity?.[0]||0)*forward[0]+
      Number(body.velocity?.[1]||0)*forward[1]+Number(body.velocity?.[2]||0)*forward[2];
  const steeringPolicy=body.steeringSystem||body.config.steering_control||{},rawSteering=Math.max(-1,Math.min(1,
      Number(body.controls?.steering||0))),steeringDamage=ensureVehicleDamage(body),
    ecuFeedFailed=Boolean(steeringDamage.members?.["electrical.wire.ecu_feed"]?.failed),
    servoFeedFailed=Boolean(steeringDamage.members?.["electrical.wire.steering_servo_feed"]?.failed),
    servoCouplingFailed=Boolean(steeringDamage.members?.["steering.assist.motor_to_column"]?.failed),
    servoPowered=energy.ignitionOn&&energy.batteryChargeWh>.05&&!servoFeedFailed&&!servoCouplingFailed,
    ecuAvailable=energy.ecuOnline&&!ecuFeedFailed,ecuSteeringActive=ecuAvailable&&servoPowered&&
      steeringPolicy.velocity_rate_control_enabled!==false,
    parkingRate=Math.max(.1,Number(steeringPolicy.parking_steering_rate_per_s||3.2)),
    highwayRate=Math.max(.1,Math.min(parkingRate,Number(steeringPolicy.highway_steering_rate_per_s||.85))),
    referenceSpeed=Math.max(1,Number(steeringPolicy.steering_rate_reference_speed_m_s||22)),
    rateCurve=Math.max(.2,Number(steeringPolicy.steering_rate_curve_exponent||1.35)),
    speedBlend=Math.pow(c2Unit(roadSpeed/referenceSpeed),rateCurve),
    ecuRate=parkingRate+(highwayRate-parkingRate)*speedBlend,priorSteering=Number(body.appliedSteering??0),
    steeringError=rawSteering-priorSteering,frontNormalLoad=Math.max(0,Number(body.springForces?.[0]||0)+Number(
      body.springForces?.[1]||0)),steeringRatio=Math.max(1,Number(steeringPolicy.steering_ratio||16)),
    scrubResistance=frontNormalLoad*Number(steeringPolicy.manual_static_friction_estimate||.72)*Number(
      steeringPolicy.tire_scrub_radius_m||.032)/(steeringRatio*(1+roadSpeed/1.8)),
    casterResistance=roadSpeed*roadSpeed*Number(steeringPolicy.caster_resistance_nm_per_m_s_squared||.012),
    resistingTorque=Math.max(0,scrubResistance+casterResistance),maximumHumanTorque=Math.max(1,Number(
      steeringPolicy.maximum_human_steering_wheel_torque_nm||38)),humanTorque=Math.min(maximumHumanTorque,
      Math.abs(steeringError)*Number(steeringPolicy.human_torque_per_normalized_error_nm||52)+
      Math.abs(rawSteering)*resistingTorque),
    viscous=Math.max(1,Number(steeringPolicy.manual_steering_viscous_nm_s||12)),
    manualRate=Math.min(Number(steeringPolicy.manual_maximum_rate_per_s||1.45),Math.max(0,humanTorque-resistingTorque)/viscous),
    assistedRate=Math.min(Number(steeringPolicy.assist_without_ecu_maximum_rate_per_s||2.25),Math.max(0,
      humanTorque*Number(steeringPolicy.assist_torque_multiplier||3.4)-resistingTorque)/viscous),
    allowedRate=ecuSteeringActive?ecuRate:servoPowered?assistedRate:manualRate,
    steeringDelta=Math.max(-allowedRate*dt,Math.min(allowedRate*dt,rawSteering-priorSteering));
  body.appliedSteering=Math.max(-1,Math.min(1,priorSteering+steeringDelta));
  body.ecuSteering={active:ecuSteeringActive,ecuAvailable,servoPowered,mode:ecuSteeringActive?
    "ecu-speed-map":servoPowered?"local-torque-assist":"manual-human-force",roadSpeedMps:roadSpeed,
    ratePerS:allowedRate,humanTorqueNm:humanTorque,resistingTorqueNm:resistingTorque,
    driverRequest:rawSteering,command:body.appliedSteering};
  energy.steeringServoOnline=servoPowered;energy.steeringServoPowerW=servoPowered?
    Number(body.config.electrical?.steering_servo_idle_power_w||18)+Number(
      body.config.electrical?.steering_servo_peak_power_w||680)*Math.min(1,Math.abs(steeringDelta)/Math.max(1e-6,dt)/parkingRate):0;
  if(driver.cruiseEnabled&&(rawBrake>.02||rawThrottle<-.02)){driver.cruiseEnabled=false;driver.cruiseIntegral=0;
    driver.cruiseThrottle=0;driver.cruiseBrake=0;}
  if(driver.cruiseEnabled){const error=driver.cruiseTargetSpeedMps-roadSpeed;
    driver.cruiseIntegral=Math.max(-8,Math.min(8,driver.cruiseIntegral+error*dt));
    driver.cruiseThrottle=Math.max(0,Math.min(1,.055+error*.16+driver.cruiseIntegral*.045));
    driver.cruiseBrake=Math.max(0,Math.min(1,-error*.11-.04));
  }else{driver.cruiseThrottle=0;driver.cruiseBrake=0;}
  const requestedDirection=Math.abs(curvedThrottle)>.02?Math.sign(curvedThrottle):0,
    reversingAgainstMotion=requestedDirection!==0&&Math.abs(forwardSpeed)>.35&&Math.sign(forwardSpeed)!==requestedDirection;
  if(reversingAgainstMotion)body.pendingDriveDirection=requestedDirection;
  else if(requestedDirection!==0&&Math.abs(forwardSpeed)<=.35){body.driveDirection=requestedDirection;
    body.pendingDriveDirection=0;}
  const demandedThrottle=driver.cruiseEnabled?Math.max(curvedThrottle,driver.cruiseThrottle):curvedThrottle,
    target=energy.ignitionOn&&energy.fuelMassKg>0&&!reversingAgainstMotion?demandedThrottle:0,
    directionChangeBrake=reversingAgainstMotion?Math.min(1,Math.abs(curvedThrottle)):0,
    smooth=Boolean(body.transmission?.smoothLaunch),
    previous=Number(body.effectiveThrottle??target);
  if(reversingAgainstMotion){body.effectiveThrottle=0;body.featheredThrottle=0;body.featheredThrottleVelocity=0;}
  else if(smooth)body.effectiveThrottle=secondOrderChannel(body,"featheredThrottle","featheredThrottleVelocity",
    target,1.35,1.15,dt);
  else body.effectiveThrottle=secondOrderChannel(body,"featheredThrottle","featheredThrottleVelocity",target,
    Number(body.config.controls?.input_response_frequency_hz||4.5)*Number(mode.throttle_rate_scale||1),
    Number(body.config.controls?.input_response_damping_ratio||.9),dt);
  body.effectiveBrake=Math.max(rawBrake,driver.cruiseBrake,directionChangeBrake);
  body.directionChange={requestedDirection,pendingDirection:Number(body.pendingDriveDirection||0),
    braking:reversingAgainstMotion,forwardSpeedMps:forwardSpeed};
  const acceleration=(forwardSpeed-Number(driver.previousForwardSpeedMps||0))/Math.max(1e-6,dt),
    graph=body.config.mechanical_graph||{},center=graph.load_audit?.center_of_mass||[0,.35,0],
    rearPatches=(graph.nodes||[]).filter(node=>node.identity==="suspension.rear_left.contact_patch"||
      node.identity==="suspension.rear_right.contact_patch").map(node=>node.reference_position),
    rearContactX=rearPatches.length?rearPatches.reduce((sum,point)=>sum+Number(point[0]),0)/rearPatches.length:
      -Number(body.config.wheels?.wheelbase_half_length||1),rearContactY=rearPatches.length?
      rearPatches.reduce((sum,point)=>sum+Number(point[1]),0)/rearPatches.length:-Number(body.config.chassis?.clearance||.3),
    rearLever=Math.max(.05,Number(center[0]||0)-rearContactX),cgHeight=Math.max(.08,Number(center[1]||.35)-rearContactY),
    tippingAngle=Math.atan2(rearLever,cgHeight),predictedArc=Number(body.pitch||0)+Number(body.pitchVelocity||0)*.18+
      Math.atan2(Math.max(0,acceleration),Math.abs(Number(body.config.world?.gravity||-9.81))),
    risk=(predictedArc-(tippingAngle-.22))/.22,targetTilt=body.transmission?.tiltEnabled===false?0:c2Unit(risk);
  driver.previousForwardSpeedMps=forwardSpeed;driver.tiltEnabled=body.transmission?.tiltEnabled!==false;
  driver.tiltRisk=risk;driver.tiltAuthority=secondOrderChannel(driver,
    "tiltFilteredAuthority","tiltAuthorityVelocity",targetTilt,7.5,1.05,dt);
  const userGovernor=Math.max(500,Number(driver.governorRpm||6500));driver.tiltGovernorRpm=Math.max(
    Number(body.config.powertrain?.idle_rpm||850)+180,userGovernor*(1-.72*driver.tiltAuthority));
  const previousTiltBrake=Number(body.rearDifferentialBrakeCommand||0);
  driver.rearDifferentialBrakeCommand=Math.max(0,Math.min(1,driver.tiltAuthority));
  body.rearDifferentialBrakeCommand=Math.max(body.transmission?.rearDifferentialBrake?1:0,
    driver.rearDifferentialBrakeCommand);
  if(Math.abs(body.rearDifferentialBrakeCommand-previousTiltBrake)>1e-5)body.gpuControlsDirty=true;
  const governorOmega=driver.tiltGovernorRpm*2*Math.PI/60;
  if(Math.abs(Number(body.defaults?.governor_angular_speed||0)-governorOmega)>1e-4){body.defaults={...(body.defaults||{}),
    governor_angular_speed:governorOmega};body.gpuStateDirty=true;}
  if(Math.abs(body.effectiveThrottle-previous)>1e-7)body.gpuControlsDirty=true;
  updateVehicleSteeringWrench(body,dt);
  updateVehicleChassisLeveling(body,dt);
}
function updateVehicleSteeringWrench(body,dt){
  const policy=body.steeringSystem||body.config.steering_control||{},share=Math.max(0,Math.min(1,
      Number(policy.front_share??.5))),frontEnabled=policy.front_axle_enabled!==false,rearEnabled=policy.rear_axle_enabled!==false,
    damage=ensureVehicleDamage(body),failed=identity=>Boolean(damage.members?.[identity]?.failed),
    frontShaftConnected=frontEnabled&&!failed("steering.proportioner.front"),
    rearShaftConnected=rearEnabled&&!failed("steering.proportioner.rear"),
    frontGain=frontShaftConnected?Math.min(1,share*2):0,rearGain=rearShaftConnected?Math.min(1,(1-share)*2):0,
    steeringCommand=Number(body.appliedSteering??body.controls?.steering??0),
    humanInputTorque=Math.sign(steeringCommand)*Number(body.ecuSteering?.humanTorqueNm||0),
    servoAssistTorque=body.ecuSteering?.servoPowered?humanInputTorque*Math.max(0,Number(
      policy.assist_torque_multiplier||3.4)-1):0,inputTorque=humanInputTorque+servoAssistTorque,
    maximum=Number(body.config.controls.maximum_steering_angle_degrees)*Math.PI/180,phase=Number(policy.rear_phase??-1),
    frequency=Math.max(.1,Number(policy.knuckle_response_frequency_hz||5.5)),damping=Math.max(.1,
      Number(policy.knuckle_damping_ratio||.92)),freeFrequency=Math.max(.05,Number(policy.free_knuckle_caster_frequency_hz||.75)),
    pinionRadius=Math.max(.001,Number(policy.pinion_radius_m||.018)),columnTarget=-steeringCommand*1.75,
    columnAngle=secondOrderChannel(body,"steeringColumnAngle","steeringColumnVelocity",columnTarget,frequency,damping,dt),
    graph=body.config.mechanical_graph,nodes=new Map((graph?.nodes||[]).map(node=>[node.identity,node.reference_position]));
  const solveCorner=(corner,rackTravel,connected)=>{const rack=nodes.get(`suspension.${corner}.steering_rack`),
      arm=nodes.get(`suspension.${corner}.steering_arm`),knuckle=nodes.get(`suspension.${corner}.knuckle`),
      prior=Number(body.wheelSteerAngles?.[corner]||0);if(!connected||!rack||!arm||!knuckle)
        return secondOrderChannel(body,`${corner}CasterAngle`,`${corner}CasterVelocity`,0,freeFrequency,.38,dt);
    const dx=Number(arm[0])-Number(knuckle[0]),dz=Number(arm[2])-Number(knuckle[2]),
      rx=Number(rack[0]),rz=Number(rack[2])+rackTravel,
      neutralLength2=(Number(arm[0])-Number(rack[0]))**2+(Number(arm[2])-Number(rack[2]))**2;
    let angle=Math.max(-maximum,Math.min(maximum,prior));for(let iteration=0;iteration<10;iteration+=1){
      const evaluate=value=>{const c=Math.cos(value),s=Math.sin(value),x=Number(knuckle[0])+dx*c-dz*s,
        z=Number(knuckle[2])+dx*s+dz*c;return (x-rx)**2+(z-rz)**2-neutralLength2;},
        error=evaluate(angle),epsilon=1e-4,gradient=(evaluate(angle+epsilon)-evaluate(angle-epsilon))/(2*epsilon);
      if(Math.abs(gradient)<1e-8)break;angle=Math.max(-maximum,Math.min(maximum,angle-error/gradient));}
    return secondOrderChannel(body,`${corner}KnuckleAngle`,`${corner}KnuckleVelocity`,angle,frequency,damping,dt);};
  const frontTravel=columnAngle*pinionRadius*frontGain,rearTravel=columnAngle*pinionRadius*rearGain*phase,
    angles=body.wheelSteerAngles||={};
  for(const corner of ["front_left","front_right"])angles[corner]=solveCorner(corner,frontTravel,
    frontShaftConnected&&!failed(`suspension.${corner}.tie_rod`));
  for(const corner of ["rear_left","rear_right"])angles[corner]=solveCorner(corner,rearTravel,
    rearShaftConnected&&!failed(`suspension.${corner}.tie_rod`));
  const beforeFront=Number(body.frontKnuckleSteerAngle||0),beforeRear=Number(body.rearKnuckleSteerAngle||0);
  body.frontKnuckleSteerAngle=(angles.front_left+angles.front_right)/2;
  body.rearKnuckleSteerAngle=(angles.rear_left+angles.rear_right)/2;
  body.steeringWrench={inputTorque,humanInputTorque,servoAssistTorque,columnAngle,
    frontRackTravel:frontTravel,rearRackTravel:rearTravel,
    ecuVelocityRateControl:{...(body.ecuSteering||{})},
    frontRackForce:inputTorque*frontGain/pinionRadius,rearRackForce:inputTorque*rearGain*phase/pinionRadius,
    frontConnected:frontShaftConnected,rearConnected:rearShaftConnected,wheelAngles:{...angles}};
  if(Math.abs(body.frontKnuckleSteerAngle-beforeFront)>1e-7||Math.abs(body.rearKnuckleSteerAngle-beforeRear)>1e-7)
    body.gpuControlsDirty=true;
}
function updateVehicleChassisLeveling(body,dt){
  const policy=body.chassisLeveling||body.config.chassis_leveling||{},names=["front_left","front_right","rear_left","rear_right"],
    offsets=body.levelingOffsets||={front_left:0,front_right:0,rear_left:0,rear_right:0},
    state=body.levelingState||={},w=body.config.wheels,maxOffset=Math.max(0,Number(policy.maximum_corner_offset_m||0)),
    maxRate=Math.max(0,Number(policy.pose_lerp_rate_m_s||policy.maximum_actuator_rate_m_s||0)),ride=policy.enabled?
      Number(policy.target_ride_height_offset_m||0):0,rollError=policy.enabled?(Number(body.roll||0)-Number(policy.target_roll_rad||0)):0,
    pitchError=policy.enabled?(Number(body.pitch||0)-Number(policy.target_pitch_rad||0)):0,
    omega=2*Math.PI*Math.max(.05,Number(policy.response_frequency_hz||.55)),damping=Math.max(.2,Number(policy.damping_ratio||1.05));
  names.forEach(name=>{const front=name.startsWith("front")?1:-1,right=name.endsWith("right")?1:-1,
      x=front*Number(w.wheelbase_half_length),z=right*Number(w.track_half_width),derivedTarget=ride+rollError*z-pitchError*x,
      requestedTarget=policy.mode==="manual-wheel"?Number(policy.manual_corner_targets_m?.[name]||0):derivedTarget,
      target=Math.max(-maxOffset,Math.min(maxOffset,policy.enabled?requestedTarget:0)),
      valueKey=`${name}Value`,velocityKey=`${name}Velocity`,
      prior=Number(offsets[name]||0);
    let value=Number(state[valueKey]??prior),velocity=Number(state[velocityKey]||0);
    const acceleration=omega*omega*(target-value)-2*damping*omega*velocity;
    velocity+=acceleration*dt;value+=velocity*dt;const delta=Math.max(-maxRate*dt,Math.min(maxRate*dt,value-prior));
    offsets[name]=Math.max(-maxOffset,Math.min(maxOffset,prior+delta));state[valueKey]=offsets[name];
    state[velocityKey]=delta===value-prior?velocity:delta/Math.max(1e-9,dt);});
  body.levelingOffsets=offsets;body.levelingState=state;
}
function stepWorld(body,dt){
  const previousPosition=[...body.position];
  const previousVelocity=[...body.velocity];
  const c=contact(body.position,body.radius,body.identity), x={position_x:body.position[0],position_y:body.position[1],position_z:body.position[2],velocity_x:body.velocity[0],velocity_y:body.velocity[1],velocity_z:body.velocity[2],dt,obstacle_active:c?1:0,obstacle_normal_x:c?.n[0]||0,obstacle_normal_z:c?.n[1]||0,obstacle_plane:c?.q||0,radius:body.radius,...body.overrides};
  const memory=new Float64Array(instance.exports.memory.buffer);
  abi.input_names.forEach((name,i)=>memory[abi.input_offsets[i]/8]=x[name]??parameters[name]??0);
  instance.exports[abi.entrypoint](0); const out={};
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
  if(terrainSupport){const inward=body.velocity.reduce((sum,value,index)=>sum+value*terrainSupport.normal[index],0);
    if(inward<0)for(let axis=0;axis<3;axis+=1)body.velocity[axis]-=terrainSupport.normal[axis]*inward;}
  for(let axis=0;axis<3;axis+=1)angular[axis]+=dt*moment[axis]*Number(inverseInertia[axis]||0);
  body.angularVelocity=angular;body.roll=(body.roll||0)+dt*angular[0];
  body.pitch=(body.pitch||0)+dt*angular[1];body.yaw=(body.yaw||0)+dt*angular[2];
  body.rollVelocity=angular[0];body.pitchVelocity=angular[1];body.yawVelocity=angular[2];
  resolveWorldBottom(body,previousPosition);
  body.accelerationMagnitude=Math.hypot(body.velocity[0]-previousVelocity[0],
    body.velocity[1]-previousVelocity[1],body.velocity[2]-previousVelocity[2])/Math.max(dt,1e-6);
  body.contactRuntimePartId=terrainSupport?.runtimePartId||support?.runtimePartId||c?.runtimePartId||0;
}
function coastStep(body,dt){
  const next=body.position.map((value,axis)=>value+body.velocity[axis]*dt),r=body.radius;
  const value=name=>body.overrides?.[name]??parameters[name];
  if(next[0]<value("minimum_x")+r||next[0]>value("maximum_x")-r||
     next[1]<value("minimum_y")+r||next[1]>value("maximum_y")-r||
     next[2]<value("minimum_z")+r||next[2]>value("maximum_z")-r||
     contact(next,r,body.identity))return false;
  body.position=next;body.contactRuntimePartId=0;return true;
}
function publishSnapshot(now){
  if(snapshotInFlight||!snapshotBuffers.length)return;
  const buffer=snapshotBuffers.pop(), values=new Float64Array(buffer); values.fill(0);
  for(const body of bodies.values()){const o=body.slot*SNAPSHOT_STRIDE;if(o<0||o+SNAPSHOT_STRIDE>values.length)continue;
    values[o]=1;values[o+1]=body.generation;values[o+2]=body.position[0];values[o+3]=body.position[1];values[o+4]=body.position[2];
    values[o+5]=body.velocity[0];values[o+6]=body.velocity[1];values[o+7]=body.velocity[2];
    values[o+8]=body.roll||0;values[o+9]=body.pitch||0;values[o+10]=body.yaw||0;
    values[o+11]=body.rollVelocity||0;values[o+12]=body.pitchVelocity||0;values[o+13]=body.yawVelocity||0;
    values[o+14]=body.contactRuntimePartId||0;
    const springs=body.springForces||[0,0,0,0],areas=body.contactAreas||[0,0,0,0],
      utilization=body.frictionUtilizations||[0,0,0,0],modes=body.contactModes||[0,0,0,0],
      compression=Object.values(body.compressions||{});
    for(let i=0;i<4;i++){values[o+15+i]=springs[i]||0;values[o+19+i]=areas[i]||0;
      values[o+23+i]=utilization[i]||0;values[o+27+i]=modes[i]||0;
      values[o+31+i]=compression[i]||0;values[o+35+i]=body.wheelOmegas?.[Object.keys(body.compressions||{})[i]]||0;}
    for(let i=0;i<4;i++){values[o+39+i]=body.tractionScales?.[i]??1;values[o+43+i]=body.brakeScales?.[i]??1;}
    const powertrain=body.powertrain||{},reaction=powertrain.reactionTorque||[0,0,0],mount=powertrain.mountTorque||[0,0,0];
    values[o+47]=powertrain.engineTorque||0;values[o+48]=powertrain.clutchTorque||0;
    values[o+49]=powertrain.transmissionOutputTorque||0;values[o+50]=powertrain.drivelineTorque||0;
    values[o+51]=powertrain.frontDifferentialTorque||0;values[o+52]=powertrain.rearDifferentialTorque||0;
    values[o+53]=powertrain.engineAccelerationTorque||0;values[o+54]=powertrain.engineAngularAcceleration||0;
    for(let i=0;i<3;i++){values[o+55+i]=reaction[i]||0;values[o+58+i]=mount[i]||0;}
    values[o+61]=body.transmission?.gear||0;values[o+62]=body.transmission?.displayGear||0;
    values[o+63]=body.transmission?.mode==="automatic"?1:0;values[o+64]=body.controlGeneration||0;
    for(let i=0;i<4;i++)values[o+65+i]=body.damperScales?.[i]??1;
    values[o+69]=body.powertrain?.engineAngularSpeed||0;values[o+70]=body.powertrain?.engineRPM||0;
    const probes=body.radialProbePenetrations||[];
    for(let wheel=0;wheel<4;wheel++)for(let probe=0;probe<15;probe++)
      values[o+71+wheel*15+probe]=Number(probes[wheel]?.[probe]||0);
    for(let wheel=0;wheel<4;wheel++)values[o+131+wheel]=Number(body.radialProbeActiveCounts?.[wheel]||0);
    values[o+135]=body.transmission?.lowRange?1:0;values[o+136]=body.transmission?.frontDiffLock?1:0;
    values[o+137]=body.transmission?.rearDiffLock?1:0;values[o+138]=body.transmission?.centerDiffLock?1:0;
    values[o+139]=Number(body.transmission?.frontDriveShare??body.config?.drivetrain?.front_drive_fraction??.5);
    values[o+140]=body.transmission?.tractionControlEnabled===false?0:1;
    values[o+141]=body.transmission?.absEnabled===false?0:1;values[o+142]=body.transmission?.smoothLaunch?1:0;
    values[o+143]=powertrain.tractionControlDissipationTorque||0;
    values[o+144]=powertrain.serviceBrakeReactionTorque||0;
    values[o+145]=powertrain.rollingResistanceReactionTorque||0;
    values[o+146]=powertrain.tireContactReactionTorque||0;
    values[o+147]=powertrain.drivetrainChassisReactionTorque||0;
    values[o+148]=Number(body.transmission?.tractionControlAuthority??1);
    values[o+149]=Number(body.transmission?.absAuthority??1);
    const coupling=mode=>mode==="locked"?1:mode==="limited-slip"?.32:0;
    values[o+150]=coupling(body.transmission?.frontDiffMode||(body.transmission?.frontDiffLock?"locked":"open"));
    values[o+151]=coupling(body.transmission?.rearDiffMode||(body.transmission?.rearDiffLock?"locked":"open"));
    values[o+152]=coupling(body.transmission?.centerDiffMode||(body.transmission?.centerDiffLock?"locked":"open"));
    values[o+153]=Number(body.frontKnuckleSteerAngle||0);values[o+154]=Number(body.rearKnuckleSteerAngle||0);
    const wheelSteer=body.wheelSteerAngles||{};values[o+155]=Number(wheelSteer.front_left||0);
    values[o+156]=Number(wheelSteer.front_right||0);values[o+157]=Number(wheelSteer.rear_left||0);
    values[o+158]=Number(wheelSteer.rear_right||0);
    values[o+159]=Number(body.steeringWrench?.columnAngle||0);
    values[o+160]=Number(body.steeringWrench?.frontRackTravel||0);
    values[o+161]=Number(body.steeringWrench?.rearRackTravel||0);
  }
  snapshotInFlight=true;postMessage({type:"snapshot-buffer",sequence:++sequence,time:now,buffer},[buffer]);
}
function setEngineStage(stage,reason){
  if(timer){clearInterval(timer);timer=null;}
  engineStage=stage;idleTicks=0;coastTicks=0;previous=performance.now();
  if(instance&&stage!=="asleep")timer=setInterval(tick,
    stage==="kinematic-coast"?1000/30:8.333333333333334);
  postMessage({type:"engine-state",stage,sleeping:stage==="asleep",reason,members:bodies.size});
}
function armEngine(reason){
  idleTicks=0;coastTicks=0;
  if(!instance)return;
  if(engineStage!=="full-dynamics"||!timer)setEngineStage("full-dynamics",reason);
}
function tick(){
  if(!instance||tickInFlight)return;tickInFlight=true;try{const now=performance.now(),
    dt=engineStage==="kinematic-coast"?1/30:FIXED_DT;previous=now;
  if(engineStage==="kinematic-coast"){
    let clear=true,moving=false;
    for(const body of bodies.values()){
      moving=moving||Math.hypot(body.velocity[0],body.velocity[1],body.velocity[2])>.012;
      if(!coastStep(body,dt))clear=false;
    }
    if(!clear){setEngineStage("full-dynamics","coast-contact-or-boundary");return;}
    publishSnapshot(now);idleTicks=moving?0:idleTicks+1;
    if(idleTicks>=45)setEngineStage("asleep","kinematic-coast-became-still");
    return;
  }
  let integrationOnly=true,moving=false,constantVelocity=true;
  for(const body of bodies.values()){
    if(body.kind==="vehicle"){
      applyPendingVehicleCommands(body);prepareVehicleEnergy(body,dt);prepareVehicleControls(body,dt);try{
        if(vehicleGpu?.residentGraph&&vehicleGpu.terrainReady&&body.gpuKernelVariant!=="parametric-pending"){residentVehicleStep(body,dt);
          if(lastVehicleGpuError)postMessage({type:"vehicle-gpu-recovered"});lastVehicleGpuError=null;}
        else if(vehicleInstance&&contactInstance)residentVehicleWasmStep(body,dt);
        else{body.accelerationMagnitude=0;continue;}
      }catch(error){const message=String(error?.message||error);if(vehicleInstance&&contactInstance){
          try{residentVehicleWasmStep(body,dt);vehicleGpu=null;
            postMessage({type:"vehicle-wasm-fallback",reason:message});lastVehicleGpuError=message;}
          catch(fallbackError){const fallbackMessage=String(fallbackError?.message||fallbackError);
            if(fallbackMessage!==lastVehicleGpuError)postMessage({type:"vehicle-gpu-error",error:fallbackMessage});
            lastVehicleGpuError=fallbackMessage;body.accelerationMagnitude=0;continue;}
        }else{if(message!==lastVehicleGpuError)postMessage({type:"vehicle-gpu-error",error:message});
          lastVehicleGpuError=message;body.accelerationMagnitude=0;continue;}}
      updateVehicleDamage(body,dt);
    }else{
      stepWorld(body,dt);
    }
    const speed=Math.hypot(body.velocity[0],body.velocity[1],body.velocity[2]);
    if(speed>.012){integrationOnly=false;moving=true;}
    if((body.accelerationMagnitude||0)>.001||body.kind==="vehicle")constantVelocity=false;
  }
  publishSnapshot(now);
  idleTicks=integrationOnly?idleTicks+1:0;
  coastTicks=moving&&constantVelocity?coastTicks+1:0;
  if(idleTicks>=180)setEngineStage("asleep","integration-only-quiescence");
  else if(coastTicks>=12)setEngineStage("kinematic-coast","constant-velocity-no-acceleration");
  }finally{tickInFlight=false;}
}
onmessage=async event=>{const m=event.data||{};
  if(m.type==="init"){const made=await WebAssembly.instantiate(m.wasm,{});instance=made.instance;abi={...m.abi,entrypoint:m.entrypoint};parameters=m.parameters||{};
    if(m.vehicleWasm&&m.contactWasm){const fallback=await Promise.all([
      WebAssembly.instantiate(m.vehicleWasm.wasm,{}),WebAssembly.instantiate(m.contactWasm.wasm,{})]);
      vehicleInstance=fallback[0].instance;contactInstance=fallback[1].instance;
      vehicleAbi={...m.vehicleWasm.abi,entrypoint:m.vehicleWasm.entrypoint};
      contactAbi={...m.contactWasm.abi,entrypoint:m.contactWasm.entrypoint};}
    worldBottom={...worldBottom,...(m.worldBottom||{})};
    snapshotCapacity=m.snapshotCapacity||128;snapshotBuffers=Array.from({length:SNAPSHOT_POOL_SIZE},()=>new ArrayBuffer(snapshotCapacity*SNAPSHOT_STRIDE*8));
    previous=performance.now();postMessage({type:"ready",snapshotCapacity,stride:SNAPSHOT_STRIDE,poolSize:SNAPSHOT_POOL_SIZE,
      vehicleCompute:vehicleInstance&&contactInstance?"resident-wasm-fallback":"resident-webgpu-pending"});armEngine("initialized");
    // Shader compilation is an initialization dependency of the vehicle graph,
    // not of the general world loop.  Let balls and the player advance while
    // the purpose-built graph is validated, then atomically publish it.
    initializeVehicleGpu(m.vehicleWebgpu).then(gpu=>{vehicleGpu=gpu;
      configureResidentVehicleTerrain(vehicleGpu,colliders);
      for(const body of bodies.values())if(body.kind==="vehicle")initializeResidentVehicleState(vehicleGpu,body);
      postMessage({type:"vehicle-gpu-recovered"});armEngine("vehicle-graph-ready");
    }).catch(error=>{vehicleGpu=null;if(vehicleInstance&&contactInstance)
      postMessage({type:"vehicle-wasm-fallback",reason:String(error?.message||error)});
      else postMessage({type:"vehicle-gpu-error",error:String(error?.message||error)});});}
  else if(m.type==="upsert"){const body={force:[0,0,0],moment:[0,0,0],angularVelocity:[0,0,0],
    ...m.body,position:[...m.body.position],velocity:[...m.body.velocity],force:[...(m.body.force||[0,0,0])],
    moment:[...(m.body.moment||[0,0,0])],angularVelocity:[...(m.body.angularVelocity||[0,0,0])],
    wheelOmegas:{front_left:0,front_right:0,rear_left:0,rear_right:0,...m.body.wheelOmegas},
    wheelSteerAngles:{front_left:0,front_right:0,rear_left:0,rear_right:0,...m.body.wheelSteerAngles},
    previousSlips:{front_left:0,front_right:0,rear_left:0,rear_right:0,...m.body.previousSlips}};
    bodies.set(m.body.identity,body);if(body.kind==="vehicle"){ensureVehicleDamage(body);ensureVehicleEnergy(body);
      ensureVehicleDriverAssistance(body);if(!Number.isFinite(body.wasmState?.engine_angular_speed))
        body.wasmState={...(body.wasmState||{}),engine_angular_speed:Number(body.defaults?.engine_idle_angular_speed||0)};
      if(vehicleGpu?.residentGraph)initializeResidentVehicleState(vehicleGpu,body);}
    armEngine("body-upsert");}
  else if(m.type==="remove") bodies.delete(m.identity);
  else if(m.type==="control"){const b=bodies.get(m.identity);if(b&&m.position){b.position[0]=m.position[0];b.position[2]=m.position[1];b.controlGeneration=m.generation||0;armEngine("control");}}
  else if(m.type==="vehicle-control"){const b=bodies.get(m.identity);if(b){b.pendingControls={
    throttle:m.throttle||0,steering:m.steering||0,brake:m.brake||0};armEngine("vehicle-control");}}
  else if(m.type==="vehicle-auxiliary"){const b=bodies.get(m.identity);if(b?.kind==="vehicle"){const state=ensureVehicleEnergy(b);
    if(typeof m.ignitionOn==="boolean")state.ignitionOn=m.ignitionOn;
    if(typeof m.starterEngaged==="boolean")state.starterEngaged=m.starterEngaged;
    if(typeof m.headlightsOn==="boolean")state.headlightsOn=m.headlightsOn;
    if(typeof m.hornOn==="boolean")state.hornOn=m.hornOn;armEngine("vehicle-auxiliary");}}
  else if(m.type==="vehicle-body-shell"){const b=bodies.get(m.identity);if(b?.kind==="vehicle"){
    b.bodyShell=String(m.shellIdentity||"bare-frame");const energy=ensureVehicleEnergy(b);
    energy.activeShellAssemblyMassKg=b.bodyShell==="bare-frame"?0:energy.configuredShellAssemblyMassKg;
    armEngine("vehicle-body-shell");}}
  else if(m.type==="vehicle-fuel-ignition"){const b=bodies.get(m.identity);if(b?.kind==="vehicle"){const energy=ensureVehicleEnergy(b);
    if(typeof m.fuelIdentity==="string")energy.fuelIdentity=m.fuelIdentity;
    if(typeof m.ignitionProfileIdentity==="string")energy.requestedIgnitionProfileIdentity=m.ignitionProfileIdentity;
    b.gpuStateDirty=true;armEngine("vehicle-fuel-ignition");}}
  else if(m.type==="vehicle-driver-assistance"){const b=bodies.get(m.identity);if(b?.kind==="vehicle"){const state=
      ensureVehicleDriverAssistance(b);
    if(typeof m.drivingMode==="string"&&(b.drivingModes||[]).some(item=>item.identity===m.drivingMode))
      state.drivingMode=m.drivingMode;
    if(Number.isFinite(m.governorRpm))state.governorRpm=Math.max(500,Math.min(Number(
      b.config.powertrain?.redline_rpm||6500),Number(m.governorRpm)));
    if(typeof m.cruiseEnabled==="boolean"){state.cruiseEnabled=m.cruiseEnabled;
      state.cruiseIntegral=0;if(m.cruiseEnabled)state.cruiseTargetSpeedMps=Number.isFinite(m.cruiseTargetSpeedMps)?
        Math.max(0,Number(m.cruiseTargetSpeedMps)):Math.hypot(Number(b.velocity?.[0]||0),Number(b.velocity?.[2]||0));}
    if(Number.isFinite(m.cruiseTargetSpeedMps))state.cruiseTargetSpeedMps=Math.max(0,Number(m.cruiseTargetSpeedMps));
    armEngine("vehicle-driver-assistance");}}
  else if(m.type==="vehicle-pneumatics"){const b=bodies.get(m.identity);if(b?.kind==="vehicle"){const state=ensureVehicleEnergy(b),
      electrical=b.config.electrical||{},minimum=Number(electrical.minimum_tire_pressure_pa||45000),maximum=Number(
        electrical.maximum_tire_pressure_pa||260000);
    if(Number.isFinite(m.tirePressureTargetPa))state.tirePressureTargetPa=Math.max(minimum,Math.min(maximum,
      Number(m.tirePressureTargetPa)));armEngine("vehicle-pneumatics");}}
  else if(m.type==="vehicle-disable-gpu"){const reason=String(m.reason||"host-requested GPU failover");vehicleGpu=null;
    postMessage({type:vehicleInstance&&contactInstance?"vehicle-wasm-fallback":"vehicle-gpu-error",reason,error:reason});
    armEngine("vehicle-disable-gpu");}
  else if(m.type==="vehicle-power-unit"){const b=bodies.get(m.identity);if(b?.kind==="vehicle"){
    b.pendingPowerUnit={...(m.preset||{})};armEngine("vehicle-power-unit");}}
  else if(m.type==="vehicle-chassis-profile"){const b=bodies.get(m.identity);if(b?.kind==="vehicle"){
    b.pendingChassisProfile={...(m.profile||{})};armEngine("vehicle-chassis-profile");}}
  else if(m.type==="vehicle-chassis-leveling"){const b=bodies.get(m.identity);if(b?.kind==="vehicle"){
    b.pendingChassisLeveling={...(m.leveling||{})};armEngine("vehicle-chassis-leveling");}}
  else if(m.type==="vehicle-steering-system"){const b=bodies.get(m.identity);if(b?.kind==="vehicle"){
    b.pendingSteeringSystem={...(m.steering||{})};armEngine("vehicle-steering-system");}}
  else if(m.type==="vehicle-parameters"){const b=bodies.get(m.identity);if(b?.kind==="vehicle"){
    b.pendingVehicleParameters={...(b.pendingVehicleParameters||{}),...(m.parameters||{})};
    armEngine("vehicle-parameters");}}
  else if(m.type==="vehicle-recover"){const b=bodies.get(m.identity);if(b){b.pendingRecovery={lift:m.lift};
    armEngine("vehicle-recover");}}
  else if(m.type==="vehicle-respawn"){const b=bodies.get(m.identity);if(b&&Array.isArray(m.position)){
    b.pendingRespawn={position:[...m.position],roll:m.roll,pitch:m.pitch,yaw:m.yaw};armEngine("vehicle-respawn");}}
  else if(m.type==="vehicle-dyno"){const b=bodies.get(m.identity),p=b?.powertrain||{},springs=b?.springForces||[];
    postMessage({type:"vehicle-dyno-result",result:{requestId:m.requestId||0,status:"telemetry",
      compute:vehicleGpu?.residentGraph?"resident-webgpu-graph":vehicleInstance&&contactInstance?
        "resident-wasm-fallback":"resident-webgpu-fault",
      pass:Boolean(vehicleGpu?.residentGraph||vehicleInstance&&contactInstance),
      forceY:springs,forceX:[0,0,0,0],wheelTorque:[p.frontDifferentialTorque||0,p.frontDifferentialTorque||0,
        p.rearDifferentialTorque||0,p.rearDifferentialTorque||0],high:{drivelineTorque:p.drivelineTorque||0},
      ultraLow:{drivelineTorque:p.drivelineTorque||0},reason:"passive resident-graph snapshot"
    }});}
  else if(m.type==="support"){const b=bodies.get(m.identity);if(b&&Number.isFinite(m.y)){b.position[1]=m.y;b.velocity[1]=0;}}
  else if(m.type==="player-jump"){const b=bodies.get(m.identity);if(b&&Number.isFinite(m.deltaVelocity)){
    b.velocity[1]+=Number(m.deltaVelocity);b.contactRuntimePartId=0;armEngine("player-jump-force");}}
  else if(m.type==="impulse"){const b=bodies.get(m.identity);if(b&&m.velocity){b.velocity[0]=m.velocity[0];b.velocity[1]=m.velocity[1];b.velocity[2]=m.velocity[2];armEngine("impulse");}}
  else if(m.type==="wrench"){const b=bodies.get(m.identity);if(b){b.force=[...(m.force||[0,0,0])];
    b.moment=[...(m.moment||[0,0,0])];armEngine("wrench-change");}}
  else if(m.type==="vehicle-transmission"){const b=bodies.get(m.identity);if(b?.kind==="vehicle"){
    b.pendingTransmission={mode:m.mode,gearDelta:m.gearDelta,lowRange:m.lowRange,
      frontDiffLock:m.frontDiffLock,rearDiffLock:m.rearDiffLock,centerDiffLock:m.centerDiffLock,
      frontDiffMode:m.frontDiffMode,rearDiffMode:m.rearDiffMode,centerDiffMode:m.centerDiffMode,
      frontDriveShare:m.frontDriveShare,smoothLaunch:m.smoothLaunch,brakeLock:m.brakeLock,
      frontDifferentialBrake:m.frontDifferentialBrake,rearDifferentialBrake:m.rearDifferentialBrake,
      tractionControlEnabled:m.tractionControlEnabled,absEnabled:m.absEnabled,tiltEnabled:m.tiltEnabled,
      tractionControlAuthority:m.tractionControlAuthority,absAuthority:m.absAuthority,
      gearset:m.gearset,transmissionPreset:m.transmissionPreset,
      releaseAllBrakes:m.releaseAllBrakes};
    armEngine("transmission-control");}}
  else if(m.type==="colliders"){colliders=m.colliders||[];if(vehicleGpu?.residentGraph)
      configureResidentVehicleTerrain(vehicleGpu,colliders);
    armEngine("collider-field-change");}
  else if(m.type==="parameters"){Object.assign(parameters,m.parameters||{});armEngine("physics-field-change");}
  else if(m.type==="recycle"){snapshotBuffers.push(m.buffer);snapshotInFlight=false;}
  else if(m.type==="stop"){if(timer)clearInterval(timer);close();}
};