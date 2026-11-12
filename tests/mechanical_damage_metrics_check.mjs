import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const source=fs.readFileSync(new URL('../src/compiler/abstract_ui_div_map.py',import.meta.url),'utf8');
const start=source.indexOf('function vehicleGraphProblem('),end=source.indexOf('function drawVehicleShaderHud(',start);
const context=vm.createContext({});vm.runInContext(source.slice(start,end),context);
const run=context.vehicleDamageGraphMetrics;
const graph={nodes:['a','b','c','isolated'].map(identity=>({identity,reference_position:[0,0,0]})),edges:[
  {identity:'ab',a:'a',b:'b',damage:{fracture_strain:.1}},
  {identity:'bc',a:'b',b:'c',damage:{fracture_strain:.1}},
  {identity:'missing',a:'a',b:'absent'}]};
const damage={members:{ab:{plasticStrain:.025,failed:false},bc:{plasticStrain:.1,failed:true}},
  junctions:{ab:{powerW:12,energyJ:34},bc:{powerW:0,energyJ:4}}};
const before=JSON.stringify({graph,damage});
let a=run(graph,damage,'graph');assert.equal(a.componentCount,3);assert.equal(a.isolated,2);assert.equal(a.dangling,1);
assert.equal(a.components.get('a'),a.components.get('b'));assert.notEqual(a.components.get('b'),a.components.get('c'));
assert.equal(a.records[1].color,'#ff3f62');assert.equal(a.records[2].color,'#ff00ff');
a=run(graph,damage,'energy');assert.equal(a.maximum,34);assert.equal(a.records[0].value,34);assert.equal(a.records[2].value,null);
a=run(graph,damage,'power');assert.equal(a.maximum,12);assert.equal(a.records[0].value,12);
a=run(graph,damage,'strain');assert.equal(a.records[0].value,.025);
a=run(graph,damage,'health');assert.equal(a.records[0].value,.25);assert.equal(a.records[2].value,null);
assert.equal(run(graph,null,'energy').missing,3,'Missing telemetry is unavailable, never fabricated zero');
assert.equal(JSON.stringify({graph,damage}),before,'Presentation must not mutate graph or dynamics');
const model=JSON.parse(fs.readFileSync(new URL('../build/dot-living-map-runtime-20261110/matched/model.json',import.meta.url),'utf8'));
function find(v){if(!v||typeof v!=='object')return null;if(v.mechanical_graph)return v.mechanical_graph;for(const item of Object.values(v)){const g=find(item);if(g)return g;}return null;}
const real=find(model),analysis=run(real,null,'graph');assert.equal(real.nodes.length,241);assert.equal(real.edges.length,339);
assert.equal(analysis.dangling,0);assert.equal(analysis.isolated,10);assert.equal(analysis.components.size,241);
const worker=fs.readFileSync(new URL('../src/compiler/state_loop_deployment.py',import.meta.url),'utf8');
assert.match(worker,/damageMetrics:body\.damage\|\|null/);assert.match(source,/if\(message\.damageMetrics\)vehicleRuntime\.damage=message\.damageMetrics/);
assert.match(source,/action:"damage-metric"/);assert.match(source,/nodes\)\{if\(!valid\(p\)\)continue/);
console.log('PASS graph metrics: real241nodes/339edges/10isolated, broken connectivity, dangling IDs, unknown telemetry, exact existing W/J, read-only state and HUD dispatch');

// Exercise the actual original HUD draw and pointer dispatcher. This is a
// command-recording unit test, not a claim of browser/GPU validation.
const drawStart=source.indexOf('function drawVehicleShaderHud('),drawEnd=source.indexOf('function drawVehicleCanvasHud(',drawStart);
const colorStart=source.indexOf('function colorVector(value)'),colorEnd=source.indexOf('\nfunction ',colorStart+1);
vm.runInContext(source.slice(colorStart,colorEnd),context);
const commands=[];
const gl=new Proxy({TRIANGLES:4},{get:(target,key)=>key in target?target[key]:(...args)=>{if(key.startsWith('uniform'))assert(args.slice(1).every(v=>Array.isArray(v)||Number.isFinite(v)));commands.push([key,...args]);}});
const canvas={width:420,height:460,getBoundingClientRect:()=>({left:0,top:0,width:420,height:460})};
context.window={devicePixelRatio:1};context.shaderViewer={shaderOnly:true,canvas,vehicleHudProgram:{},vehicleHudLocations:{},vehicleHudVao:{}};
context.vehicleRuntime={active:model.vehicle_slot.vehicles[0],damage:null,damageMetric:'graph',mechanicalNodePositions:new Map(),compressions:[],damperScales:[],tractionScales:[],brakeScales:[],transmission:{},shaderHudHitRegions:[]};
vm.runInContext(source.slice(drawStart,drawEnd),context);
for(const metric of ['health','strain','power','energy','graph','laplacian','compiled']){
  context.vehicleRuntime.damageMetric=metric;context.drawVehicleShaderHud(gl,420,460);
  const buttons=context.vehicleRuntime.shaderHudHitRegions.filter(h=>h.action==='damage-metric');assert.equal(buttons.length,7);
  for(const hit of buttons){assert.equal(context.handleVehicleShaderHudPointer({button:0,clientX:hit.x+1,clientY:hit.y+1}),true);assert.equal(context.vehicleRuntime.damageMetric,hit.metric);}
}
assert(commands.filter(c=>c[0]==='drawArrays').length>real.nodes.length*5);
console.log('PASS actual HUD draw/dispatch command test for all7metric buttons; browser GPU verification remains separate');

const bar={nodes:[{identity:'a',reference_position:[0,0,0]},{identity:'b',reference_position:[2,0,0]}],
  edges:[{identity:'bar',a:'a',b:'b',damage:{axial_stiffness_n_per_m:100}}]};
const strain={members:{bar:{plasticStrain:0}}}, points=new Map([['a',[0,0,0]],['b',[2.1,0,0]]]);
a=run(bar,strain,'laplacian',points);
assert(Math.abs(a.total-.5)<1e-12);assert(Math.abs(a.laplacianAction.get('b')[0]-10)<1e-12);
assert(Math.abs([...a.nodeEnergy.values()].reduce((x,y)=>x+y,0)-a.total)<1e-12);
assert.equal(run(bar,strain,'laplacian',new Map([['a',[1,3,5]],['b',[3,3,5]]])).total,0);
assert.equal(run(bar,strain,'laplacian').missing,1);
assert.equal(run(bar,{members:{bar:{plasticStrain:.01}}},'laplacian',points).missing,1);
assert.equal(run(bar,{members:{bar:{failed:true,plasticStrain:0}}},'laplacian',points).missing,1);
assert.equal(context.vehicleGraphProblem(bar.edges[0],{junctions:{bar:{powerW:NaN}}},null,null),'invalid');
assert.equal(context.vehicleGraphProblem(bar.edges[0],null,null,null),null);
assert.equal(context.vehicleGraphProblem(bar.edges[0],{members:{bar:{failed:true}}},null,null),'broken');
assert.equal(context.vehicleGraphProblem(bar.edges[0],null,[Infinity,0,0],[0,0,0]),'invalid');
assert.equal(run(bar,{members:{bar:{plasticStrain:0}},junctions:{bar:{powerW:NaN}}},'graph',points).records[0].color,'#ff00ff');
assert.match(source,/if\(box.diagnostic_problem\)/);
assert.match(source,/updateVehiclePresentation\(vehicle,state,dt,0\);updateVehicleBodyPresentation\(vehicle,state\)/);
console.log('PASS weighted axial Laplacian quadratic, translation nullspace, action, energy allocation, exclusions, invalid/broken/missing diagnostic separation');

context.vertices=[];context.semanticPartSpans=[];context.spans=[];
context.colors={line:'#ffffff'};context.performance={now:()=>0};context.turingWorld={partRuntimeId:()=>1};
context.normalized3=v=>{const n=Math.hypot(...v)||1;return v.map(x=>x/n);};
const linkStart=source.indexOf('  function vehicleLink('),linkEnd=source.indexOf('  function vehiclePowertrainPart(',linkStart);
vm.runInContext(source.slice(linkStart,linkEnd),context);
const link={identity:'test',kind:'vehicle-mechanical-edge',appearance:{face_color:'#123456'},mechanical_edge:{},
  link_state:{localA:[0,0,0],localB:[1,0,0],radius:.01}};
context.vehicleLink(link,0);const baselineVertices=context.vertices.length;
context.vertices=[];link.diagnostic_problem='invalid';context.vehicleLink(link,0);
assert(context.vertices.length>baselineVertices);assert(context.vertices.every(Number.isFinite));
assert.equal(link.appearance.face_color,'#123456');
assert(context.vertices.some((v,i)=>i%9===6&&v===1&&context.vertices[i+1]===0&&context.vertices[i+2]===1));
console.log('PASS actual shader link mesh: finite additive diagnostic collars preserve underlying material');

const captureStart=worker.indexOf('function captureVehicleCompiledEnergy('),captureEnd=worker.indexOf('function applyResidentVehicleSnapshot(',captureStart);
vm.runInContext(worker.slice(captureStart,captureEnd),context);
const energyBody={config:{energy_diagnostics:{outputs:['known','missing','invalid'],complete_budget:false}}};
context.captureVehicleCompiledEnergy(energyBody,{known:12,invalid:Infinity});
assert.equal(energyBody.compiledEnergy.values.known,12);assert.equal(energyBody.compiledEnergy.values.missing,null);
assert.deepEqual([...energyBody.compiledEnergy.invalid],['invalid']);assert.deepEqual([...energyBody.compiledEnergy.missing],['missing']);
const compiled={schema:'browser-vehicle-compiled-energy-compartments-v1',node_outputs:{a:['spin']},values:{spin:5}};
a=run(bar,strain,'compiled',points,compiled);assert.equal(a.compiledNodes.get('a'),5);assert.equal(a.nodeColor('b'),'#6d7773');
console.log('PASS compiled energy transport preserves missing/nonfinite values and exact graph-node identity mapping');
