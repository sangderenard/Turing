// Additive shell adapter: existing frame, registered geometry, declared controls.
const arenaLineRuntime={driver:null,loading:null,sequence:0,panel:null};
function arenaLineSampleSupport(identity,x,z){
  // The original contact sampler defines this exact floor identity at y=0;
  // sampled terrain replaces it in its domain, even when the terrain is low.
  if(identity==='world-floor'){
    const replaced=shaderViewer.colliders.some(c=>{const d=c.surface?.domain;
      return c.surface?.kind==='sampled-height-field'&&d&&x>=d.minimum_x&&x<=d.maximum_x&&z>=d.minimum_z&&z<=d.maximum_z;});
    if(replaced)throw new Error('Named world-floor is replaced by sampled terrain here');
    return 0;
  }
  const collider=shaderViewer.colliders.find(c=>c.identity===identity);
  if(!collider)throw new Error(`Support is absent: ${identity}`);
  const authored=shaderViewer.geometry.find(box=>box.identity===collider.objectIdentity);
  if(authored&&Math.abs(boxYawDegrees(authored))>1e-8)throw new Error('Rotated box support requires an exact oriented surface');
  const domain=collider.surface?.domain;
  const minX=domain?.minimum_x??collider.minimum[0],maxX=domain?.maximum_x??collider.maximum[0];
  const minZ=domain?.minimum_z??collider.minimum[2],maxZ=domain?.maximum_z??collider.maximum[2];
  if(x<minX||x>maxX||z<minZ||z>maxZ)throw new Error('Path leaves its named support');
  // Box tops and declared height fields are distinct named supports. No
  // height ranking or camera-ground heuristic is used to select a surface.
  return collider.surface?sampleDeclaredSurface(collider.surface,x,z).height:collider.maximum[1];
}
async function arenaLineInitialize(){
  if(arenaLineRuntime.driver)return arenaLineRuntime.driver;
  if(!arenaLineRuntime.loading)arenaLineRuntime.loading=(async()=>{
    const [{ArenaLineDriver},{TuringGraphProgram}]=await Promise.all([
      import('./arena-line-driver/arena_line_driver.js'),import('./arena-line-driver/turing-wasm-shell.js')]);
    const program=await TuringGraphProgram.load('./arena-line-driver/pilot.wasm','./arena-line-driver/pilot.abi.json');
    const host={sampleSupport:arenaLineSampleSupport,
      ready:()=>Boolean(stateLoopRuntime.ready&&!document.hidden),
      vehicle:()=>vehicleRuntime.active&&vehicleRuntime.state?{id:vehicleRuntime.active.identity,state:vehicleRuntime.state}:null,
      basis:state=>[[1,0,0],[0,1,0],[0,0,1]].map(axis=>rotateVehiclePresentationVector(axis,state))};
    arenaLineRuntime.driver=new ArenaLineDriver(program,host);return arenaLineRuntime.driver;
  })().catch(error=>{arenaLineRuntime.loading=null;throw error;});
  return arenaLineRuntime.loading;
}
function arenaLineGuardFocus(){
  const driver=arenaLineRuntime.driver;
  if(driver?.active&&(!viewportControls.highlighted||controlFocus.mode!=="game"||document.hidden)){
    driver.stop('Paused after leaving game controls');
    if(vehicleRuntime.active&&stateLoopRuntime.ready)stateLoopRuntime.worker.postMessage({type:'vehicle-control',identity:vehicleRuntime.active.identity,throttle:0,steering:0,brake:1});
  }
}
function arenaLineControls(manual){
  const driver=arenaLineRuntime.driver;if(!driver)return manual;
  const controls=driver.controls(manual);
  if(arenaLineRuntime.panel){
    arenaLineRuntime.panel.status.textContent=arenaLineRuntime.panel.error||driver.status;
    const state=vehicleRuntime.state,observation=driver.observation;
    if(state){const forward=rotateVehiclePresentationVector([1,0,0],state),speed=state.velocity.reduce((n,v,i)=>n+v*forward[i],0);
      arenaLineRuntime.panel.telemetry.textContent=`yaw ${(state.yaw*180/Math.PI).toFixed(1)} deg | local forward ${speed.toFixed(3)} m/s\n`+
        `command steer ${Number(controls.steering||0).toFixed(3)} throttle ${Number(controls.throttle||0).toFixed(3)} brake ${Number(controls.brake||0).toFixed(3)}\n`+
        (observation?`line progress ${driver.distance.toFixed(2)} m | near forward ${observation.localPoints[1][0].toFixed(2)} m | near lateral ${observation.localPoints[1][2].toFixed(2)} m`:'line observation not sampled');}
  }
  return controls;
}
function arenaLinePaintVertices(){
  const vertices=[];
  for(const path of arenaLineRuntime.driver?.paint()||[]){
    const rgb=[1,3,5].map(i=>parseInt(path.color.slice(i,i+2),16)/255);
    for(let i=1;i<path.points.length;i++){
      const a=path.points[i-1].position,b=path.points[i].position,dx=b[0]-a[0],dz=b[2]-a[2],len=Math.hypot(dx,dz);
      if(len<1e-8)continue;const px=-dz/len*.045,pz=dx/len*.045;
      const p=[a[0]+px,a[1]+.018,a[2]+pz],q=[a[0]-px,a[1]+.018,a[2]-pz];
      const r=[b[0]-px,b[1]+.018,b[2]-pz],s=[b[0]+px,b[1]+.018,b[2]+pz];
      for(const point of [p,q,r,p,r,s])vertices.push(...point,0,1,0,...rgb);
    }
  }return vertices;
}
function arenaLineInstallPanel(){
  const panel=document.createElement('details');panel.id='arena-line-driver-panel';
  panel.style.cssText='position:fixed;left:12px;bottom:100px;z-index:111;max-width:min(330px,calc(100vw - 24px));max-height:40vh;overflow:auto;padding:8px;background:#13202eee;color:white;font:12px sans-serif;border:1px solid #5c7c99;border-radius:7px';
  const title=document.createElement('summary');title.textContent='Arena line driver · 5-neuron starter';panel.append(title);
  const status=document.createElement('p');status.textContent='Experimental starter weights. Upload replaces paths and weights; Start drives through real controls.';
  const telemetry=document.createElement('pre');telemetry.style.cssText='font:10px monospace;white-space:pre-wrap';telemetry.textContent='Vehicle/line telemetry appears after mounting';
  const input=document.createElement('input');input.type='file';input.accept='.json,application/json';input.setAttribute('aria-label','Upload arena paths and network weights');
  const select=document.createElement('select');select.setAttribute('aria-label','Circuit');
  const demo=document.createElement('button');demo.textContent='Load test circuits';
  const start=document.createElement('button');start.textContent='Start';start.disabled=true;
  const stop=document.createElement('button');stop.textContent='Stop';
  const errorStatus=error=>{arenaLineRuntime.panel.error=error.message;status.textContent=error.message;};
  const load=async(text,sequence)=>{const driver=await arenaLineInitialize();if(sequence!==arenaLineRuntime.sequence)return;
    const paths=driver.load(text);select.replaceChildren();for(const path of paths){const option=document.createElement('option');option.value=path.id;option.textContent=path.id;select.append(option);}
    start.disabled=false;arenaLineRuntime.panel.error='';status.textContent=driver.status;rebuildPortableSceneMesh();};
  input.addEventListener('change',async()=>{const file=input.files?.[0];if(!file)return;const sequence=++arenaLineRuntime.sequence;
    try{if(file.size>262144)throw new Error('Upload exceeds 256 KiB');await load(await file.text(),sequence);}
    catch(error){errorStatus(error);}finally{input.value='';}});
  demo.addEventListener('click',async()=>{const sequence=++arenaLineRuntime.sequence;try{const r=await fetch('./arena-line-driver/starter-circuits.json');if(!r.ok)throw new Error('Test circuits unavailable');await load(await r.text(),sequence);}catch(error){errorStatus(error);}});
  start.addEventListener('click',()=>{try{arenaLineRuntime.driver.start(select.value);requestViewportControls();arenaLineRuntime.panel.error='';status.textContent=arenaLineRuntime.driver.status;}catch(error){errorStatus(error);}});
  stop.addEventListener('click',()=>{++arenaLineRuntime.sequence;arenaLineRuntime.driver?.stop('Stopped by user');arenaLineRuntime.panel.error='';status.textContent='Stopped by user';
    if(vehicleRuntime.active)updateActiveVehicle(0,0,0,1);});
  panel.append(status,telemetry,input,document.createElement('br'),demo,select,start,stop);document.body.append(panel);arenaLineRuntime.panel={panel,status,telemetry,error:''};
  document.addEventListener('visibilitychange',()=>{if(document.hidden){arenaLineRuntime.driver?.stop('Paused while hidden');if(vehicleRuntime.active&&stateLoopRuntime.ready)stateLoopRuntime.worker.postMessage({type:'vehicle-control',identity:vehicleRuntime.active.identity,throttle:0,steering:0,brake:1});}});
}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',arenaLineInstallPanel,{once:true});else arenaLineInstallPanel();

