// Arena path data adapter. Inference belongs to the existing TuringGraphProgram;
// this module never evaluates uploaded code and never writes vehicle poses.
export const SCHEMA = 'turing.arena-line-driver.v1';
export const SHAPES = Object.freeze({GW1:[15,5],Gb1:[1,5],GM1:[15,5],GW2:[5,3],Gb2:[1,3],GM2:[5,3]});
const assert = (ok, message) => { if (!ok) throw new Error(message); };
const finite = x => typeof x === 'number' && Number.isFinite(x);
const keys = (v, names, at) => {
  assert(v && typeof v === 'object' && !Array.isArray(v), `${at}: expected object`);
  assert(Object.keys(v).every(k=>names.includes(k)), `${at}: unknown field`);
};
const vector = (v,n,at,limit=100000) => {
  assert(Array.isArray(v)&&v.length===n&&v.every(x=>finite(x)&&Math.abs(x)<=limit), `${at}: expected ${n} bounded numbers`);
  return [...v];
};
export function parseArenaProgram(text) {
  assert(typeof text==='string' && new TextEncoder().encode(text).length<=262144,'Upload exceeds 256 KiB');
  const p=JSON.parse(text);
  keys(p,['schema','network','circuits'],'program');
  assert(p.schema===SCHEMA,'Unsupported arena program version');
  keys(p.network,['architecture','weights'],'network');
  assert(p.network.architecture==='graph-15-5-3-tanh-gates-v1','Unsupported network architecture');
  keys(p.network.weights,Object.keys(SHAPES),'weights');
  const weights={};
  for(const [name,shape] of Object.entries(SHAPES)){
    const row=p.network.weights[name];keys(row,['shape','data'],name);
    assert(JSON.stringify(row.shape)===JSON.stringify(shape),`${name}: wrong shape`);
    const data=vector(row.data,shape.reduce((a,b)=>a*b,1),name,100);
    if(name.startsWith('GM'))assert(data.every(x=>x===0||x===1),`${name}: masks must be 0 or 1`);
    weights[name]={shape:[...shape],data};
  }
  assert(Array.isArray(p.circuits)&&p.circuits.length>0&&p.circuits.length<=8,'Expected 1..8 circuits');
  const seen=new Set();let count=0;
  const circuits=p.circuits.map((c,i)=>{
    keys(c,['id','color','segments'],`circuit ${i}`);
    assert(typeof c.id==='string'&&/^[a-zA-Z0-9_-]{1,48}$/.test(c.id)&&!seen.has(c.id),'Invalid or duplicate circuit id');seen.add(c.id);
    assert(typeof c.color==='string'&&/^#[0-9a-fA-F]{6}$/.test(c.color),'Expected #RRGGBB line color');
    assert(Array.isArray(c.segments)&&c.segments.length>0&&c.segments.length<=256,'Expected 1..256 segments');
    const segments=c.segments.map((s,j)=>{
      keys(s,['support','from','to'],`segment ${j}`);
      assert(typeof s.support==='string'&&s.support.length>0&&s.support.length<=256,'Missing support identity');
      const from=vector(s.from,2,'from'),to=vector(s.to,2,'to');
      assert(Math.hypot(to[0]-from[0],to[1]-from[1])>.001,'Zero length segment');
      count++;assert(count<=512,'Too many segments');
      if(j){const last=c.segments[j-1].to;assert(Math.hypot(last[0]-from[0],last[1]-from[1])<1e-6,'Circuit segments must connect in order');}
      return {support:s.support,from,to};
    });return {id:c.id,color:c.color,segments};
  });
  return {schema:SCHEMA,network:{architecture:p.network.architecture,weights},circuits};
}

// The caller resolves an exact registered support identity and uses the game's
// sampleDeclaredSurface. Missing/quarantined supports fail, never fall back to
// highest terrain. Each sample is projected AFTER x/z interpolation.
export function projectCircuit(circuit, sampleSupport, spacing=.25) {
  const points=[];let length=0;
  for(const segment of circuit.segments){
    const distance=Math.hypot(segment.to[0]-segment.from[0],segment.to[1]-segment.from[1]);
    const steps=Math.ceil(distance/spacing);
    assert(points.length+steps+1<=16384,'Circuit sampling budget exceeded');
    for(let i=0;i<=steps;i++){
      const t=i/steps,x=segment.from[0]+t*(segment.to[0]-segment.from[0]),z=segment.from[1]+t*(segment.to[1]-segment.from[1]);
      const y=sampleSupport(segment.support,x,z);
      assert(finite(y),'Support missing or point outside its domain');
      const position=[x,y,z],previous=points.at(-1);
      if(i===0&&previous){assert(Math.abs(y-previous.position[1])<.05,'Disconnected support layers');continue;}
      if(previous)length+=Math.hypot(...position.map((v,a)=>v-previous.position[a]));
      points.push({position,distance:length,support:segment.support});
    }
  }
  return {id:circuit.id,color:circuit.color,points,length};
}
const dot=(a,b)=>a.reduce((sum,v,i)=>sum+v*b[i],0);
const interpolate=(a,b,t)=>a.map((v,i)=>v+(b[i]-v)*t);
export function pointAt(path,distance){
  const d=Math.max(0,Math.min(path.length,distance));
  let i=0;while(i<path.points.length-2&&path.points[i+1].distance<d)i++;
  const a=path.points[i],b=path.points[i+1],t=(d-a.distance)/(b.distance-a.distance);
  return interpolate(a.position,b.position,t);
}
export function observeLine(path,state,basis,previousDistance=0){
  assert(state.position?.every(finite)&&state.velocity?.every(finite)&&finite(state.yawVelocity),'Nonfinite vehicle state');
  // Restrict to the current connected section so crossing/stacked roads cannot
  // switch the driver to a different later part of the circuit.
  let best=null;
  for(let i=0;i<path.points.length-1;i++){
    const a=path.points[i],b=path.points[i+1];
    if(b.distance<previousDistance-1||a.distance>previousDistance+12)continue;
    const ab=b.position.map((v,j)=>v-a.position[j]),ap=state.position.map((v,j)=>v-a.position[j]);
    const t=Math.max(0,Math.min(1,dot(ap,ab)/dot(ab,ab))),point=interpolate(a.position,b.position,t);
    const error=Math.hypot(...point.map((v,j)=>v-state.position[j]));
    if(!best||error<best.error)best={error,distance:a.distance+t*(b.distance-a.distance)};
  }
  assert(best&&best.error<=8,'Vehicle is too far from the current line section');
  const distance=Math.max(previousDistance,best.distance),localPoints=[0,1,3,6].map(offset=>{
    const delta=pointAt(path,distance+offset).map((v,i)=>v-state.position[i]);
    return basis.map(axis=>dot(delta,axis));
  });
  const features=[...localPoints.flat().map(v=>v/10),dot(state.velocity,basis[0])/10,dot(state.velocity,basis[2])/10,state.yawVelocity];
  assert(features.every(finite),'Invalid line observation');
  return {features,distance,remaining:path.length-distance,localPoints,error:best.error};
}

export class ArenaLineDriver {
  constructor(program,host){this.program=program;this.host=host;this.active=false;this.loaded=null;this.status='No circuit loaded';this.distance=0;this.holdBrake=false;}
  load(text){
    const parsed=parseArenaProgram(text);
    const paths=parsed.circuits.map(c=>projectCircuit(c,(...args)=>this.host.sampleSupport(...args)));
    assert(paths.reduce((n,p)=>n+p.points.length,0)<=16384,'Total circuit sampling budget exceeded');
    // Validate all data and supports before mutating the running program.
    for(const [name,weight] of Object.entries(parsed.network.weights))this.program.setInput(name,weight.data);
    this.stop('Loaded; choose a circuit and Start');this.loaded={parsed,paths};return paths;
  }
  start(id){
    assert(this.loaded,'Load a network and circuits first');
    const path=this.loaded.paths.find(p=>p.id===id);assert(path,'Unknown circuit');
    const vehicle=this.host.vehicle();assert(vehicle&&this.host.ready(),'Mount a vehicle with ready physics first');
    // Refresh projection at Start to reject edited or removed support surfaces.
    this.path=projectCircuit(this.loaded.parsed.circuits.find(c=>c.id===id),(...args)=>this.host.sampleSupport(...args));
    observeLine(this.path,vehicle.state,this.host.basis(vehicle.state),0);
    this.vehicleId=vehicle.id;this.distance=0;this.active=true;this.holdBrake=false;this.status=`Following ${id}`;
  }
  stop(reason='Stopped'){this.holdBrake=this.active||this.holdBrake;this.active=false;this.status=reason;}
  controls(manual){
    if(['throttle','steering','brake'].some(k=>Math.abs(Number(manual[k]||0))>.05)){if(this.active)this.stop('Manual control');this.holdBrake=false;return manual;}
    if(!this.active)return this.holdBrake?{throttle:0,steering:0,brake:1}:manual;
    try{
      const vehicle=this.host.vehicle();assert(vehicle?.id===this.vehicleId&&this.host.ready(),'Vehicle or physics unavailable');
      const o=observeLine(this.path,vehicle.state,this.host.basis(vehicle.state),this.distance);this.distance=o.distance;this.observation=o;
      if(o.remaining<.35){this.stop('Circuit complete');return {throttle:0,steering:0,brake:1};}
      this.program.setInput('x',o.features);this.program.run();
      const output=Array.from(this.program.output('action'));
      assert(output.length===3&&output.every(finite),'Network returned invalid controls');
      // This is actuator bounding, not a replacement steering policy.
      return {steering:Math.max(-1,Math.min(1,output[0])),throttle:Math.max(0,Math.min(.5,output[1])),brake:Math.max(0,Math.min(1,output[2]))};
    }catch(error){this.stop(`Stopped: ${error.message}`);return {throttle:0,steering:0,brake:1};}
  }
  paint(){return this.loaded?.paths||[];}
}
