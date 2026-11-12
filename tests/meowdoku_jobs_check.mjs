import assert from 'node:assert/strict';
import {jobLabel, jobFileUrl, collationLabel, mountJobs} from '../examples/meowdoku/web/marble_jobs.js';
const digest = 'a'.repeat(64), other = 'b'.repeat(64);
assert.match(jobLabel({digest,state:'queued',queue_position:2,kind:'photo'}),/queued #2/);
assert.equal(jobFileUrl({digest,image:'frame 1.png'}),`/api/marble_photo/image?digest=${digest}&name=frame%201.png`);
assert.equal(jobFileUrl({digest:'../bad',image:'frame.png'}),null);
assert.equal(jobFileUrl({digest,hdr:'linear.npz'},'hdr'),`/api/marble_photo/hdr?digest=${digest}&name=linear.npz`);
class Element {
  constructor(tag) { this.tag=tag; this.children=[]; this.listeners={}; this.dataset={}; this.style={}; this.textContent=''; this._value=''; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children=[...nodes]; this._value=''; }
  setAttribute() {}
  addEventListener(name,fn) { this.listeners[name]=fn; }
  get value() { return this._value || (this.tag==='select' ? this.children[0]?.value || '' : ''); }
  set value(v) { this._value=v; }
}
const doc = {hidden:false, createElement: tag=>new Element(tag), addEventListener(){},removeEventListener(){}};
const root = new Element('section');
let data={entries:[{digest,kind:'photo',state:'queued',queue_position:1},{digest:other,kind:'photo',state:'complete',image:'frame.png'}],scheduling:'single_gpu',queue_available:false};
const calls=[];
let failed=false;
const fetcher=async (url,options)=> { calls.push([url,options]); if(failed) throw Error('offline'); return {ok:true,json:async()=>data}; };
const jobs=mountJobs(root,{doc,fetcher,interval:999999});
await new Promise(resolve=>setTimeout(resolve,0));
const [label,refresh,library,status,detail]=root.children, select=label.children[0];
assert.equal(select.children.length,2); assert.equal(library.href,'marble_jobs.html');
assert.match(status.textContent,/cannot queue yet/); assert.equal(detail.children[1].textContent,'Cancel this job');
select.value=other;select.listeners.change();assert.equal(detail.children[1].textContent,'View image');
await jobs.refresh();assert.equal(select.value,other,'selection survives polling');
select.value=digest;select.listeners.change();await detail.children[1].listeners.click();
assert.equal(calls.filter(c=>c[0].endsWith('/cancel')).length,1);
assert.deepEqual(JSON.parse(calls.find(c=>c[0].endsWith('/cancel'))[1].body),{digest});
data={entries:[{digest,kind:'photo',state:'exposing',can_cancel:false}],scheduling:'single_gpu',queue_available:false};await jobs.refresh();assert.equal(detail.children.length,1,'server can_cancel=false suppresses cancellation');
failed=true;await jobs.refresh();assert.match(status.textContent,/offline/);failed=false;
data={entries:[],scheduling:'single_gpu',queue_available:false};await jobs.refresh();assert.equal(select.disabled,true);assert.equal(detail.children.length,0);
jobs.dispose();const before=calls.length;await jobs.refresh();assert.equal(calls.length,before);
// A disposed slow response must not resurrect a removed view.
let resolve;const delayed=new Promise(r=>resolve=r);const r2=new Element('section');
const j2=mountJobs(r2,{doc,fetcher:()=>delayed,interval:999999});j2.dispose();resolve({ok:true,json:async()=>({entries:[{digest,state:'live'}]})});
await new Promise(r=>setTimeout(r,0));assert.equal(r2.children[0].children[0].children.length,0);
console.log('jobs UI checks passed: listing, queue labels, selection, file URLs, cancel, failures, empty and disposed responses');
// Rejoin is explicit, owner-authenticated, and never starts just by listing.
const r3=new Element('section');let joined=0,destroyed=0,joinCalls=0;
let active=true;
const j3=mountJobs(r3,{doc,interval:999999,fetcher:async(url,opts)=>{
 if(url==='/api/client_rays/join'){joinCalls++;assert.equal(opts.credentials,'same-origin');assert.equal(opts.headers['X-Meowdoku-Rays'],'1');assert.deepEqual(JSON.parse(opts.body),{job:digest});return {ok:true,json:async()=>({job:digest,credential:'ephemeral'})};}
 return {ok:true,json:async()=>({entries:[{digest,kind:'photo',state:active?'exposing':'complete',can_cancel:active,client_intersections:true}],queue_available:false})};
},mountParticipant:({join,digest:bound})=>{joined++;assert.equal(join.job,bound);return {loop:{enabled:true},destroy(){destroyed++;}};}});
await new Promise(r=>setTimeout(r,0));assert.equal(joinCalls,0);assert.equal(joined,0);
const d3=r3.children[4];assert.equal(d3.children[1].textContent,'Participate / resume on this device');
await d3.children[1].listeners.click();assert.equal(joinCalls,1);assert.equal(joined,1);assert.equal(d3.children[1].disabled,true);
await j3.refresh();assert.equal(joined,1,'refresh does not recreate a participant');
active=false;await j3.refresh();assert.equal(destroyed,1,'finished jobs stop device polling');j3.dispose();
console.log('jobs explicit rejoin: profile-cookie API, no automatic participation, stable refresh, terminal cleanup passed');

// A collection is one owned parent; child links use that parent's authorization.
assert.equal(collationLabel({kind:'tiles',completed:1,total:4}), '1/4 tiles');
assert.equal(collationLabel({kind:'camera_pose_sequence',source:'orbit',completed:2,total:8}), '2/8 orbit frames');
const r4=new Element('section');
let collection={kind:'tiles',state:'assembling',completed:4,total:4,children:Array.from({length:4},(_,index)=>({index,kind:'tile',state:'complete',window:{x:(index%2)*1024,y:Math.floor(index/2)*1024,width:1024,height:1024}}))};
let parentState='assembling';
const j4=mountJobs(r4,{doc,interval:999999,fetcher:async()=>({ok:true,json:async()=>({entries:[{digest,kind:'photo',state:parentState,can_cancel:parentState==='assembling',collation:collection}],queue_available:false})})});
await new Promise(r=>setTimeout(r,0));
const d4=r4.children[4];
assert.match(r4.children[0].children[0].children[0].textContent,/4\/4 tiles/);
assert.equal(d4.children[1].tag,'details');
assert.equal(d4.children[1].children[2].children.length,4);
assert.match(d4.children[1].children[2].children[3].textContent,/1024, 1024; 1024 × 1024/);
assert.equal(d4.children[1].children[2].children[0].children.length,0,'tile contribution has no invented artifact URL');
assert.equal(d4.children[2].textContent,'Cancel this job','assembly remains cancellable');
d4.children[1].open=true;d4.children[1].listeners.toggle();await j4.refresh();
assert.equal(d4.children[1].open,true,'expanded collection survives refresh');
collection={kind:'camera_pose_sequence',source:'explicit',state:'interrupted',completed:1,total:2,children:[{index:0,kind:'camera_pose',state:'complete',image:'preview-000001.png',hdr:'exposure-000001.npz'},{index:1,kind:'camera_pose',state:'interrupted'}]};parentState='interrupted';
await j4.refresh();
const rows=d4.children[1].children[2].children;
assert.match(d4.children[1].children[0].textContent,/1\/2 camera poses/);
assert.equal(rows[0].children[0].href,`/api/marble_photo/image?digest=${digest}&name=preview-000001.png`);
assert.equal(rows[0].children[1].href,`/api/marble_photo/hdr?digest=${digest}&name=exposure-000001.npz`);
assert.equal(rows[1].children.length,0);assert.equal(d4.children.length,2,'interrupted group has no cancel or restart action');
j4.dispose();
console.log('collated Jobs: one parent, real child results, assembling cancel, interrupted sequence and expansion retention passed');
