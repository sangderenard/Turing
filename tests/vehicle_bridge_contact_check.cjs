// Current worker geometry only. Closed-solid cases mirror the existing WGSL
// capsule/box contract; this does not execute tire force or suspension kernels.
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert/strict');
const root=path.resolve(__dirname,'..'),source=fs.readFileSync(path.join(root,'src/compiler/state_loop_deployment.py'),'utf8').replaceAll('\r\n','\n');
function extract(name){const a=source.indexOf('function '+name+'('),b=a+1+source.slice(a+1).search(/\n(?:async )?function /);assert(a>=0&&b>a);return source.slice(a,b).replaceAll('{{','{').replaceAll('}}','}');}
const ctx={colliders:[]};vm.createContext(ctx);
vm.runInContext(require('child_process').execFileSync('python',['-c','from src.compiler.closed_solid_geometry import emit_closed_solid_javascript; print(emit_closed_solid_javascript())'],{cwd:root,encoding:'utf8'}),ctx);
vm.runInContext(['rotateBodyVector','c2Unit','ensureVehicleDamage','solveMechanicalGraph','sampleSurface','contactSurfaceAt','terrainSegmentCrossing','solidSegmentCrossing','analyticTorusTireContact','wheelContactRecords'].map(extract).join('\n'),ctx);
const names=['front_left','front_right','rear_left','rear_right'];
const body={identity:'car',position:[0,0,0],velocity:[0,0,0],roll:0,pitch:0,yaw:0,compressions:Object.fromEntries(names.map(n=>[n,0])),
 config:{chassis:{clearance:.1},suspension:{rest_length:.2,travel:1},wheels:{wheelbase_half_length:0,track_half_width:0},
 tires:{radius:.3,toroid_section_radius_m:.08,width:.2},mass:100,world:{gravity:-9.81},mass_distribution:Object.fromEntries(names.map(n=>[n,.25])),drivetrain:{wheel_mass_kg:5,tire_mass_kg:3}}};
const plate={identity:'plate',objectIdentity:'bridge',minimum:[-2,3.42,-2],maximum:[2,4,2],runtimePartId:17};
function check(label,hub,expected){body.position=[...hub];const r=ctx.analyticTorusTireContact(body,hub,ctx.rotateBodyVector(body,[1,0,0]),ctx.rotateBodyVector(body,[0,0,1]),0,0,0,.3,.08,.2,1,1.6,[0,0,0],1/120,[0,0,0]);
 const records=ctx.wheelContactRecords(body,1/120);
 if(!expected){assert.equal(r.integral,null,label);for(const input of records)assert.equal(input.support,0,label);}
 else{assert.equal(r.integral.identity,expected.identity||'plate',label);assert(Math.abs(r.integral.penetration-expected.depth)<1e-10,label);
  for(let a=0;a<3;a++)assert(Math.abs(r.integral.normal[a]-expected.normal[a])<1e-10,label);
  // These are the actual contact-law inputs, after the support gate, not only
  // the intermediate analytic geometry candidate.
  for(const input of records){assert(input.support>0,label+' support gate');assert(Math.abs(input.tire_radial_compression-expected.depth)<1e-10,label);
   for(const [a,key] of ['normal_x','normal_y','normal_z'].entries())assert(Math.abs(input[key]-expected.normal[a])<1e-10,label+' normal transport');}
 }
 console.log(JSON.stringify({case:label,depth:r.integral?.penetration??0,normal:r.integral?.normal??null,contactLawInputs:records.length,result:'PASS'}));
}
ctx.colliders=[plate];
check('deck',[0,4.2,0],{normal:[0,1,0],depth:.1});
check('clear-underpass',[0,3,0],null);
check('underside',[0,3.25,0],{normal:[0,-1,0],depth:.13});
check('side',[2.2,3.7,0],{normal:[1,0,0],depth:.1});
check('edge',[2.2,4.1,0],{normal:[2/Math.sqrt(5),1/Math.sqrt(5),0],depth:.3-Math.sqrt(.05)});
ctx.colliders=[{...plate,minimum:[-2,.8,-2],maximum:[2,1,2]}];
check('ground-retained-under-overhead',[0,.2,0],{identity:'world-floor',normal:[0,1,0],depth:.1});
const crossing=ctx.terrainSegmentCrossing([0,.1,0],[0,-.1,0],body,1.6,false);assert(crossing);assert.equal(crossing.height,0);
ctx.colliders=[{...plate,objectIdentity:'car'}];check('excluded-own-solid',[0,4.2,0],null);
ctx.colliders=[{...plate,role:'projectile-body'}];check('projectile-not-tire-support',[0,4.2,0],null);
const field={identity:'field',objectIdentity:'terrain',surface:{kind:'sampled-height-field',domain:{minimum_x:-1,maximum_x:1,minimum_z:-1,maximum_z:1},origin:[-1,0,-1],cell_size:[2,2],resolution:[2,2],heights:[0,0,0,0]}};
ctx.colliders=[field];check('terrain-below-ground-correction',[0,-.1,0],{identity:'field',normal:[0,1,0],depth:.4});
ctx.colliders=[{...field,objectIdentity:'car'}];assert.equal(ctx.contactSurfaceAt(0,0,.2,1.6,'car',false).identity,'world-floor');
// Matching combined body/solid rotation keeps the contact local geometry
// unchanged while proving the world normal reaches all four force input rows.
body.roll=.3;body.pitch=-.2;body.yaw=.7;
const axes=[[1,0,0],[0,1,0],[0,0,1]].map(v=>Array.from(ctx.rotateBodyVector(body,v))),shape={kind:'oriented-box',origin:[0,4,0],minimum:[-2,-.29,-2],maximum:[2,.29,2],axes},oriented={...plate,solid_shape:shape};
const corners=[];for(const x of [-2,2])for(const y of [-.29,.29])for(const z of [-2,2])corners.push(ctx.closedSolidWorldPoint(oriented,[x,y,z]));
oriented.minimum=[0,1,2].map(i=>Math.min(...corners.map(p=>p[i])));oriented.maximum=[0,1,2].map(i=>Math.max(...corners.map(p=>p[i])));
ctx.colliders=[oriented];check('combined-rotation-underside-input',Array.from(ctx.closedSolidWorldPoint(oriented,[0,-.49,0])),{normal:Array.from(ctx.closedSolidWorldVector(oriented,[0,-1,0])),depth:.1});
console.log('PASS: closed-solid deck/underside/side/edge, ground retention, exclusions, and contact-law input transport. Rotation is outside this AABB contract.');
