const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert/strict'),{execFileSync}=require('child_process');
const root=path.resolve(__dirname,'..'),source=fs.readFileSync(path.join(root,'src/compiler/state_loop_deployment.py'),'utf8').replaceAll('\r\n','\n');
function extract(name){const a=source.indexOf('function '+name+'('),b=a+1+source.slice(a+1).search(/\n(?:async )?function /);assert(a>=0&&b>a);return source.slice(a,b).replaceAll('{{','{').replaceAll('}}','}');}
const ctx={colliders:[]};vm.createContext(ctx);vm.runInContext(execFileSync('python',['-c','from src.compiler.closed_solid_geometry import emit_closed_solid_javascript; print(emit_closed_solid_javascript())'],{cwd:root,encoding:'utf8'}),ctx);
vm.runInContext(['rotateBodyVector','sampleSurface','contactSurfaceAt','solidSegmentCrossing','c2Unit','c2Positive','c2Clamp','ensureVehicleDamage','vehicleExteriorContactSamples','vehicleExteriorSurfaceContact','vehicleCageContactWrench'].map(extract).join('\n'),ctx);
const near=(a,b)=>assert(Math.abs(a-b)<1e-10,`${a} != ${b}`),vec=(a,b)=>a.forEach((v,i)=>near(v,b[i]));
function fixture(roll,pitch,yaw){const body={roll,pitch,yaw},axes=[[1,0,0],[0,1,0],[0,0,1]].map(v=>Array.from(ctx.rotateBodyVector(body,v))),shape={kind:'oriented-box',origin:[1,4,-2],minimum:[-2,-.29,-.5],maximum:[2,.29,.5],axes};
 const c={identity:'plate',objectIdentity:'bridge',runtimePartId:17,solid_shape:shape},corners=[];
 for(const x of [-2,2])for(const y of [-.29,.29])for(const z of [-.5,.5])corners.push(ctx.closedSolidWorldPoint(c,[x,y,z]));
 c.minimum=[0,1,2].map(i=>Math.min(...corners.map(p=>p[i])));c.maximum=[0,1,2].map(i=>Math.max(...corners.map(p=>p[i])));return c;
}
let checks=0;
for(const angles of [[0,0,0],[0,0,Math.PI/4],[.3,-.2,.7]]){
 const c=fixture(...angles);ctx.validateClosedSolid(c);const packed=ctx.closedSolidPack(c);assert.equal(packed.length,32);vec(packed.slice(8,11),c.solid_shape.origin);vec(packed.slice(24,27),c.minimum);
 for(const [p,point,normal,distance] of [
  [[0,.49,0],[0,.29,0],[0,1,0],.2],[[0,-.49,0],[0,-.29,0],[0,-1,0],.2],
  [[2.2,0,0],[2,0,0],[1,0,0],.2],[[2.2,.39,0],[2,.29,0],[2/Math.sqrt(5),1/Math.sqrt(5),0],Math.sqrt(.05)],
  [[0,0,0],[0,-.29,0],[0,-1,0],-.29]]){
   const world=ctx.closedSolidWorldPoint(c,p),hit=ctx.closedSolidClosest(c,world);vec(ctx.closedSolidLocalPoint(c,world),p);vec(hit.point,ctx.closedSolidWorldPoint(c,point));vec(hit.normal,ctx.closedSolidWorldVector(c,normal));near(hit.signedDistance,distance);checks++;
 }
 const start=ctx.closedSolidWorldPoint(c,[0,1,0]),end=ctx.closedSolidWorldPoint(c,[0,-1,0]),hit=ctx.closedSolidSegment(c,start,end);assert(hit);near(hit.fraction,.355);vec(hit.normal,c.solid_shape.axes[1]);
 assert.equal(ctx.closedSolidSegment(c,ctx.closedSolidWorldPoint(c,[0,0,0]),end),null,'inside-to-outside is not entering');
 ctx.colliders=[c];const routed=ctx.solidSegmentCrossing(start,end,{identity:'car'});near(routed.fraction,.355);
 // A real cage sample touching the underside must produce a downward/outward
 // wrench through the unchanged existing cage force law, not just geometry.
 const local=[0,-.34,0],position=Array.from(ctx.closedSolidWorldPoint(c,local)),node={identity:'roof-node',kind:'roll-cage-node',reference_position:[0,0,0]};
 const body={identity:'car',bodyShell:'bare-frame',position,velocity:[0,0,0],roll:0,pitch:0,yaw:0,mechanicalGraphPositions:new Map([['roof-node',[0,0,0]]]),
  config:{mass:100,mechanical_graph:{nodes:[node],edges:[],load_audit:{center_of_mass:[0,0,0]}},solid_contact:{cage_contact_radius:.1,cage_contact_stiffness:1000,cage_contact_maximum_force:10000,cage_contact_damping:100,cage_static_friction:.5,cage_kinetic_friction:.3}}};
 const contact=ctx.vehicleExteriorSurfaceContact(body,position,.1,2.5);near(contact.penetration,.05);vec(contact.surface.normal,ctx.closedSolidWorldVector(c,[0,-1,0]));
 const wrench=ctx.vehicleCageContactWrench(body,1/120),expected=ctx.closedSolidWorldVector(c,[0,-1,0]);assert.equal(wrench.count,1);const magnitude=Math.hypot(...wrench.force);assert(magnitude>0);vec(wrench.force.map(v=>v/magnitude),expected);vec(wrench.torque,[0,0,0]);
}
const c=fixture(0,0,Math.PI/4),outside=[c.solid_shape.origin[0]+1.5,4.2,c.solid_shape.origin[2]-1.5];assert(ctx.closedSolidClosest(c,outside).signedDistance>1.6);assert.equal(ctx.closedSolidVerticalSupport(c,outside[0],outside[2],5,3),null);
const tilted=fixture(.2,-.25,.4),center=tilted.solid_shape.origin,support=ctx.closedSolidVerticalSupport(tilted,center[0],center[2],6,4);assert(support);assert(support.normal[1]>0);near(ctx.closedSolidLocalPoint(tilted,support.point)[1],.29);
const legacy={identity:'legacy',minimum:[-1,-1,-1],maximum:[1,1,1]};near(ctx.closedSolidClosest(legacy,[2,0,0]).signedDistance,1);assert.equal(ctx.closedSolidPack(legacy).length,32);
const bad=structuredClone(c);bad.solid_shape.axes[0][0]*=2;assert.throws(()=>ctx.validateClosedSolid(bad),/Non-orthonormal/);
console.log(JSON.stringify({result:'PASS',closestCases:checks,rotations:3,actualCageWrenches:3,segmentAndVerticalQueries:true,rotatedBroadphaseFalseHitRejected:true,packingAndValidation:true}));
