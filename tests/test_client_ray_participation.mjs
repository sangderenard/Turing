import assert from 'node:assert/strict';
import {ClientRayParticipation} from '../examples/meowdoku/web/client_ray_participation.js';
const work=()=>({token:'lease',revision:'scene',sample_prefix:'sample',geometry_hash:'geometry',kernel:'engine',query_kernel:'query',rays:[[0,0,0,0,0,0,1,100]],triangles:[{v0:[0,0,1],e1:[1,0,0],e2:[0,1,0],owner:3}]});
function harness(){
 const calls=[],messages=[],timers=[];let response={job:'job',state:'pending',remaining_ms:10000,work:work()},count=0,now=0;
 const backend={manifest:{source_sha256:'query'},setScene(s){this.scene=s;},async query(rays,revision){count++;return {revision,kernel:'query',count:1,hits:new Float32Array([1,3,1,0])};},close(){this.closed=true;}};
 const loop=new ClientRayParticipation({job:'job',credential:'credential',createBackend:async()=>backend,
 clock:()=>now,schedule:(f,ms)=>{timers.push({f,ms});return timers.length;},unschedule:()=>{},onStatus:s=>messages.push(s),
 fetcher:async(url,opts)=>{calls.push({url,...opts});return {ok:true,json:async()=>url.endsWith('/result')?{job:'job',state:'submitted',committed:false}:response};}});
 return {loop,calls,messages,timers,backend,get count(){return count;},set response(v){response=v;},set now(v){now=v;}};
}
{
 const h=harness();assert.equal(h.loop.enabled,false);await h.loop.tick();assert.equal(h.calls.length,0);
 h.loop.setEnabled(true);await h.loop.tick();assert.equal(h.count,1);assert.equal(h.calls.length,2);
 const sent=JSON.parse(h.calls[1].body);assert.deepEqual(sent.result.hits,[[1,3,1,0]]);assert.equal(sent.result.kernel,'engine');assert.equal(sent.result.query_kernel,'query');
 assert.equal(h.calls[0].headers.Authorization,'Bearer credential');assert(!h.calls[0].url.includes('credential'));
 await h.loop.tick();assert.equal(h.count,1,'same token must never resubmit');
 h.response={job:'job',state:'committed'};await h.loop.tick();assert.equal(h.loop.enabled,false);assert.match(h.messages.at(-1),/committed/);
}
{
 const h=harness();h.backend.manifest.source_sha256='wrong';h.loop.setEnabled(true);await h.loop.tick();assert.equal(h.count,0);assert.equal(h.loop.enabled,false);assert.match(h.messages.at(-1),/identity/);
}
{
 const h=harness();let release;h.backend.query=(rays,revision)=>new Promise(resolve=>release=()=>resolve({revision,kernel:'query',count:1,hits:new Float32Array([1,3,1,0])}));
 h.loop.setEnabled(true);const active=h.loop.tick();await new Promise(r=>setTimeout(r,0));h.loop.setHidden(true);release();await active;assert.equal(h.calls.length,1,'hidden query must not submit');
}
{
 const h=harness();h.response={job:'job',state:'expired'};h.loop.setEnabled(true);await h.loop.tick();assert.equal(h.count,0);assert.equal(h.loop.enabled,true);assert.match(h.messages.at(-1),/remain/);
}
{
 const h=harness();h.backend.query=async(rays,revision)=>{h.now=20000;return {revision,kernel:'query',count:1,hits:new Float32Array([1,3,1,0])};};h.loop.setEnabled(true);await h.loop.tick();assert.equal(h.calls.length,1);assert.match(h.messages.at(-1),/lease/);
}
{
 const h=harness();h.loop.configure({batch:16,hz:1,duty:10});const w=work();w.rays=Array(17).fill(w.rays[0]);h.response={job:'job',state:'pending',remaining_ms:10000,work:w};
 const sizes=[];h.backend.query=async(rays,revision)=>{const count=rays.length/8;sizes.push(count);h.loop.configure({batch:32,hz:1,duty:10});return {revision,kernel:'query',count,hits:new Float32Array(Array(count).fill([1,3,1,0]).flat())};};
 h.loop.setEnabled(true);await h.loop.tick();assert.deepEqual(sizes,[16,1]);
 const sent=h.calls.filter(c=>c.url.endsWith('/result'));assert.equal(sent.length,1);assert.equal(JSON.parse(sent[0].body).result.hits.length,17);assert.equal(h.loop.enabled,true);h.loop.close();
}
console.log('Client participation: opt-in, credentials, identities, exactly-once submit, hidden discard, expiry and batch caps passed');
{
 const h=harness(),original=h.loop.fetcher;let sends=0;
 h.loop.fetcher=async(url,opts)=>{if(url.endsWith('/result')){sends++;throw Error('response lost');}return original(url,opts);};
 h.loop.setEnabled(true);await h.loop.tick();assert.equal(h.loop.enabled,true);assert.match(h.messages.at(-1),/outcome unknown/);
 await h.loop.tick();assert.equal(sends,1);assert.equal(h.count,1);
 h.response={job:'job',state:'committed'};await h.loop.tick();assert.equal(h.loop.enabled,false);assert.match(h.messages.at(-1),/committed/);
}
{
 const h=harness();let release;h.loop.createBackend=()=>new Promise(resolve=>release=()=>resolve(h.backend));
 h.loop.setEnabled(true);const pending=h.loop.tick();await new Promise(r=>setTimeout(r,0));h.loop.close();release();await pending;
 assert.equal(h.backend.closed,true,'backend created after close must be released');assert.equal(h.count,0);assert.equal(h.loop.backend,null);h.loop.setEnabled(true);assert.equal(h.loop.enabled,false);
}
console.log('Uncertain result delivery confirms without resending; late backend initialization closes safely');
{
 const h=harness();let installed=0;const set=h.backend.setScene.bind(h.backend);h.backend.setScene=s=>{installed++;set(s);};
 const initial=work();initial.query_geometry_hash='immutable-intersections';
 h.response={job:'job',state:'pending',remaining_ms:10000,work:initial,scene_ack_required:true};
 h.loop.setEnabled(true);await h.loop.tick();
 const later={...initial,token:'next',revision:'new-material-revision',geometry_hash:'new-transport-state',sample_prefix:'new-sample'};delete later.triangles;
 h.response={job:'job',state:'pending',remaining_ms:10000,work:later};await h.loop.tick();
 assert.equal(installed,1,'material-only revision must reuse installed exact query geometry');
 assert.equal(h.count,2);
 const submissions=h.calls.filter(c=>c.url.endsWith('/result')).map(c=>JSON.parse(c.body).result);
 assert.equal(submissions[1].revision,'new-material-revision');assert.equal(submissions[1].geometry_hash,'new-transport-state');
 assert.equal(submissions[1].query_geometry_hash,'immutable-intersections');
 h.loop.close();
}
console.log('Query-only geometry residency preserves full transport result binding');

{
 const h=harness(),original=h.loop.fetcher;let limited=true;
 h.loop.fetcher=async(url,opts)=>limited?{ok:false,status:429,json:async()=>({error:'Poll rate exceeds ten per second'})}:original(url,opts);
 h.loop.setEnabled(true);await h.loop.tick();assert.equal(h.loop.enabled,true);assert.equal(h.count,0);assert.equal(h.timers.at(-1).ms,1000);
 limited=false;await h.loop.tick();assert.equal(h.count,1);assert.equal(h.calls.filter(c=>c.url.endsWith('/result')).length,1);h.loop.close();
 assert.throws(()=>h.loop.configure({hz:60}),/10 idle polls/);
}
console.log('Polling rejection backs off without stopping, consuming, or duplicating work; cadence matches server');
