import assert from 'node:assert/strict';
import {ArenaLineDriver,parseArenaProgram,projectCircuit,observeLine,SHAPES} from '../examples/meowdoku/web/arena_line_driver.js';
const weights=Object.fromEntries(Object.entries(SHAPES).map(([k,shape])=>[k,{shape,data:Array(shape.reduce((a,b)=>a*b,1)).fill(k.startsWith('GM')?1:0)}]));
const packet={schema:'turing.arena-line-driver.v1',network:{architecture:'graph-15-5-3-tanh-gates-v1',weights},circuits:[{id:'blue',color:'#00aaff',segments:[{support:'ground',from:[0,0],to:[20,0]}]}]};
let checks=0;const ok=f=>{f();checks++};
ok(()=>assert.equal(parseArenaProgram(JSON.stringify(packet)).circuits[0].id,'blue'));
for(const mutate of [p=>p.schema='other',p=>p.execute='alert(1)',p=>p.network.weights.GW1.data[0]='1',p=>p.network.weights.GW1.shape=[5,15],p=>p.network.weights.GM1.data[0]=.5,p=>p.circuits[0].color='url(evil)',p=>p.circuits[0].segments[0].to=[0,0],p=>p.circuits.push(p.circuits[0]),p=>p.network.weights.GW1.data[0]=1e200]){
 const p=structuredClone(packet);mutate(p);ok(()=>assert.throws(()=>parseArenaProgram(JSON.stringify(p))));
}
ok(()=>assert.throws(()=>parseArenaProgram(' '.repeat(262145))));
const sampler=(id,x,z)=>{if(id==='ground')return .1*x;if(id==='deck')return 5;throw Error('missing support')};
const path=projectCircuit(packet.circuits[0],sampler);
ok(()=>assert.equal(path.points[4].position[1],.1));
ok(()=>assert.throws(()=>projectCircuit({...packet.circuits[0],segments:[{support:'absent',from:[0,0],to:[1,0]}]},sampler)));
ok(()=>assert.throws(()=>projectCircuit({...packet.circuits[0],segments:[{support:'ground',from:[0,0],to:[1,0]},{support:'deck',from:[1,0],to:[2,0]}]},sampler),/Disconnected/));
let state={position:[0,1,0],velocity:[2,0,0],yawVelocity:0};const basis=[[1,0,0],[0,1,0],[0,0,1]];
const o=observeLine(path,state,basis);ok(()=>assert.equal(o.features.length,15));ok(()=>assert.equal(o.features[12],.2));
const turned=observeLine(path,state,[[0,0,1],[0,1,0],[-1,0,0]]);ok(()=>assert.equal(turned.features[12],0));
let runs=0,sets=0,output=[.2,.3,-.7];
const program={setInput(){sets++},run(){runs++},output(){return output}};
const host={sampleSupport:sampler,vehicle:()=>({id:'truck',state}),ready:()=>true,basis:()=>basis};
const d=new ArenaLineDriver(program,host);d.load(JSON.stringify(packet));ok(()=>assert.equal(runs,0));d.start('blue');
const controls=d.controls({});ok(()=>assert.deepEqual(controls,{steering:.2,throttle:.3,brake:0}));ok(()=>assert.deepEqual(state.position,[0,1,0]));
ok(()=>assert.deepEqual(d.controls({steering:-.5}),{steering:-.5}));ok(()=>assert.equal(d.active,false));
d.start('blue');output=[NaN,1,0];ok(()=>assert.equal(d.controls({}).brake,1));ok(()=>assert.equal(d.active,false));ok(()=>assert.equal(d.controls({}).brake,1));
const before=sets;ok(()=>assert.throws(()=>d.load(JSON.stringify({...packet,execute:'bad'}))));ok(()=>assert.equal(sets,before));
state={...state,position:[100,1,0]};ok(()=>assert.throws(()=>d.start('blue'),/too far/));
console.log(`${checks} arena line contract/control checks passed`);
