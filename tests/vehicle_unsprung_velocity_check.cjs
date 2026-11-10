const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert/strict');
const root=path.resolve(__dirname,'..'),source=fs.readFileSync(path.join(root,'src/compiler/state_loop_deployment.py'),'utf8').replaceAll('\r\n','\n');
function extract(name){const a=source.indexOf('function '+name+'('),b=source.indexOf('\nfunction ',a+1);assert(a>=0&&b>a);return source.slice(a,b).replaceAll('{{','{').replaceAll('}}','}');}
const original={},candidate={colliders:[]};vm.createContext(original);vm.createContext(candidate);
const sharedGeometry=require('child_process').execFileSync('python',['-c','from src.compiler.closed_solid_geometry import emit_closed_solid_javascript; print(emit_closed_solid_javascript())'],{cwd:root,encoding:'utf8'});vm.runInContext(sharedGeometry,candidate);vm.runInContext(sharedGeometry,original);
const dependencies=['ensureVehicleDamage','rotateBodyVector','c2Unit'].map(extract).join('\n');
vm.runInContext(dependencies+'\n'+extract('solveMechanicalGraph'),original);
vm.runInContext(dependencies+'\n'+['solveMechanicalGraph','analyticTorusTireContact','wheelContactRecords'].map(extract).join('\n'),candidate);
// Self-contained linkage fixture: no ignored build artifacts or force kernels.
const names=['front_left','front_right','rear_left','rear_right'],nodes=[],edges=[];
for(const [i,name] of names.entries()){
 const x=i<2?.62:-.62,z=i%2===0?-.78:.78,prefix=`suspension.${name}`;
 nodes.push({identity:prefix+'.mount',reference_position:[x+.1,.1,z+.1],fixed_to:'chassis'},
  {identity:prefix+'.hub',reference_position:[x,-.15,z],generalized_coordinate:`compression_${name}`},
  {identity:prefix+'.contact_patch',reference_position:[x,-.45,z]});
 edges.push({identity:prefix+'.arm',a:prefix+'.mount',b:prefix+'.hub',constraint:'rigid-distance',rest_length:.29,damage:{natural_rest_length:.29}});
}
edges.push({identity:'cross-link',a:'suspension.front_left.hub',b:'suspension.front_right.hub',constraint:'steering-link',rest_length:1.56});
const base={identity:'velocity-boundary-fixture',config:{
 chassis:{clearance:.2},suspension:{rest_length:.25,travel:.3},
 wheels:{wheelbase_half_length:.62,track_half_width:.78},tires:{radius:.3,toroid_section_radius_m:.08,width:.2},
 mass:100,world:{gravity:-9.81},mass_distribution:Object.fromEntries(names.map(n=>[n,.25])),drivetrain:{wheel_mass_kg:5,tire_mass_kg:3},
 mechanical_graph:{nodes,edges}},position:[0,.20,0],velocity:[0,0,0],roll:.12,pitch:-.08,yaw:.3,rollVelocity:0,pitchVelocity:0,yawVelocity:0,
 compressions:Object.fromEntries(names.map((n,i)=>[n,.03+i*.012])),compressionVelocities:Object.fromEntries(names.map((n,i)=>[n,[.5,-.23,.17,-.31][i]])),wheelAlignment:{corners:Object.fromEntries(names.map((n,i)=>[n,{camber_deg:i-2,caster_deg:5+i}]))}};
const cases=[['healthy',b=>{}],['damage-and-link-modifiers',b=>{
 original.ensureVehicleDamage(b);b.damage.springPlasticSet.front_left=.012;b.levelingOffsets={rear_right:.013};
 b.linkLengthModifiers={};let k=0;for(const e of b.config.mechanical_graph.edges){if(['rigid-distance','rigid-offset','steering-link','replaceable-sacrificial-knuckle-bushing'].includes(e.constraint)){
 b.linkLengthModifiers[e.identity]=((k++%3)-1)*.007;if(b.damage.members[e.identity])b.damage.members[e.identity].restLength*=1.01;}}
 const e=b.config.mechanical_graph.edges.find(e=>b.damage.members[e.identity]);if(e)b.damage.members[e.identity].failed=true;
}],['zero-rates',b=>{for(const n of names)b.compressionVelocities[n]=0;}]];
let allError=0,nodeChecks=0,contactChecks=0,sweepChecks=0;
for(const [label,modify] of cases){const body=structuredClone(base);modify(body);const before=structuredClone(body),zero=structuredClone(body);for(const n of names)zero.compressionVelocities[n]=0;const old=original.solveMechanicalGraph(zero),positions=candidate.solveMechanicalGraph(body);
 for(const [id,pos] of old)assert.deepEqual(Array.from(positions.get(id)),Array.from(pos),'position changed: '+id);
 const h=1e-6;function shifted(sign){const b=structuredClone(before);for(const n of names)b.compressions[n]+=sign*h*b.compressionVelocities[n];return original.solveMechanicalGraph(b);}
 const lo=shifted(-1),hi=shifted(1);let maxError=0;
 for(const [id,pos] of positions)for(let a=0;a<3;a++){const fd=(hi.get(id)[a]-lo.get(id)[a])/(2*h),got=body.mechanicalGraphVelocities.get(id)[a],error=Math.abs(fd-got);maxError=Math.max(maxError,error);assert(error<2e-7,`${label} ${id}[${a}]: ${fd} vs ${got}`);nodeChecks++;}
 allError=Math.max(allError,maxError);
 // Explicit static planar terrain test seam. No tire force kernel is replaced or run.
 const normal=[.02,1,-.01],norm=Math.hypot(...normal);normal.forEach((x,i)=>normal[i]=x/norm);let sweeps=[];
 candidate.contactSurfaceAt=(x,z)=>({normal,height:-(normal[0]*x+normal[2]*z)/normal[1],point:[0,0,0]});
 candidate.terrainSegmentCrossing=(start,end)=>{sweeps.push({start:Array.from(start),end:Array.from(end)});return null;};candidate.solidSegmentCrossing=()=>null;
 const records=candidate.wheelContactRecords(body,1/120);assert.equal(records.length,4);
 records.forEach((r,i)=>{const name=names[i],local=body.mechanicalGraphVelocities.get(`suspension.${name}.hub`),world=candidate.rotateBodyVector(body,local),expected=world.reduce((s,x,a)=>s+x*normal[a],0);
 assert(r.support>0,`${label} ${name}: test must have contact`);assert(Math.abs(r.tire_radial_velocity-expected)<1e-12);contactChecks++;
 for(let a=0;a<3;a++){assert(Math.abs((sweeps[i].end[a]-sweeps[i].start[a])*120-world[a])<1e-12);sweepChecks++;}
 });
 console.log(JSON.stringify({case:label,positionParity:'exact',nodes:positions.size,maximumDirectionalDerivativeError:maxError,contacts:records.length,sweepVelocity:'matches transported hub velocity',fixedCamberCaster:true,rolledPitchedYawed:true}));
}
console.log(JSON.stringify({result:'PASS',nodeComponentChecks:nodeChecks,contactChecks,sweepComponentChecks:sweepChecks,maximumDerivativeError:allError,scope:'production-source geometry only; no compiler, force kernel, or vehicle stability test'}));
