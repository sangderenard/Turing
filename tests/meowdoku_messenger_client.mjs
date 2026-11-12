import assert from 'node:assert/strict';
import fs from 'node:fs';
const src=fs.readFileSync('examples/meowdoku/web/messenger-widget.js','utf8');
const apiCode=src.slice(src.indexOf('async function api('),src.indexOf('\nasync function start()'));
let requestOptions;
const badApi=new Function('fetch',apiCode+';return api;')(async (p,o)=>{requestOptions=o;return {ok:true,json:async()=>{throw Error('bad JSON');}};});
await assert.rejects(badApi('/api/messages/send',{}),/Invalid JSON/);
assert.equal(requestOptions.redirect,'error');
const start=src.indexOf('    let sending = false;');
const code=src.slice(start,src.indexOf('    input.addEventListener',start))+'\nreturn send;';
const make=new Function('input','location','crypto','pendingSends','peer','api','viewNote','draft','image','updateControls','active','choosing','view',code);
const receipt={ok:true,id:1,created:1};
let nextKey=0;
function pane(map,api,value='hello',image=null) {
 const input={value,disabled:false}, notes=[];
 const send=make(input,{protocol:'https:'},{randomUUID:()=> 'op-'+(++nextKey)},map,'ben',api,m=>notes.push(m),{},image,()=>{},()=>true,false,{imageState:{enabled:true},toggling:false});
 return {input,send,notes};
}
// First response lost: one operation, fresh handles, double-click suppressed.
{
 const map=new Map(), calls=[];let n=0,issued=0;
 const api=async(p,b)=>{calls.push([p,b]);if(p.endsWith('/conversations'))return {contacts:[{username:'ben',conversation_handle:'h'+(++issued)}]};if(++n===1)throw Error('lost');return receipt;};
 const ui=pane(map,api);await Promise.all([ui.send(),ui.send()]);
 const sends=calls.filter(([p])=>p.endsWith('/send')).map(([,b])=>b);
 assert.equal(sends.length,2);assert.equal(sends[0].idempotency_key,sends[1].idempotency_key);
 assert.notEqual(sends[0].conversation_handle,sends[1].conversation_handle);
 assert.deepEqual(Object.keys(sends[0]).sort(),['body','conversation_handle','idempotency_key']);
 assert.equal(ui.input.value,'');assert.equal(map.size,0);
 ui.input.value='hello';await ui.send();
 assert.equal(calls.filter(([p])=>p.endsWith('/send')).length,3); // Intentional identical new message.
}
// Back/Close/reopen while in-flight shares the operation, then does not resend
// the stale composer after the old pane's request has completed.
{
 const map=new Map();let release, sent=0;
 const api=async(p)=>p.endsWith('/conversations')?{contacts:[{username:'ben',conversation_handle:'h'}]}:(sent++,await new Promise(r=>release=r));
 const old=pane(map,api);const running=old.send();await new Promise(r=>setImmediate(r));
 const reopened=pane(map,api,'');assert.equal(reopened.input.value,'hello');await reopened.send();assert.equal(sent,1);
 release(receipt);await running;await reopened.send();assert.equal(sent,1);assert.equal(reopened.input.value,'');
}
// Invalid receipts stay uncertain. Editing must not discard the original key.
{
 const map=new Map();let sent=0;
 const api=async(p)=>p.endsWith('/conversations')?{contacts:[{username:'ben',conversation_handle:'h'}]}:(sent++,null);
 const ui=pane(map,api);await ui.send();assert.equal(sent,2);assert.equal(ui.input.value,'hello');
 const key=map.get('ben').idempotency_key;ui.input.value='different';await ui.send();
 assert.equal(sent,2);assert.equal(map.get('ben').idempotency_key,key);assert.match(ui.notes.at(-1),/previous message/);
 const reopened=pane(map,api,'');assert.equal(reopened.input.value,'hello');
}
console.log('Widget regression checks passed (isolated send/API functions, not full browser DOM).');

// Definitive validation failure allows correction; uncertainty followed by a
// definitive failure retains the original operation identity instead.
for (const uncertainFirst of [false,true]) {
 const map=new Map();let sends=0, reject=true;
 const api=async(p)=>{
  if(p.endsWith('/conversations'))return {contacts:[{username:'ben',conversation_handle:'h'}]};
  sends++;
  if(uncertainFirst&&sends===1)throw Error('lost response');
  if(reject){const e=Error('body invalid');e.status=400;throw e;}
  return receipt;
 };
 const ui=pane(map,api);await ui.send();
 if(uncertainFirst){
  const key=map.get('ben').idempotency_key;const count=sends;
  ui.input.value='corrected';await ui.send();assert.equal(sends,count);assert.equal(map.get('ben').idempotency_key,key);
 } else {
  assert.equal(map.size,0);reject=false;ui.input.value='corrected';await ui.send();assert.equal(sends,2);assert.equal(ui.input.value,'');
 }
}

// Image bytes and operation identity survive retries and reopening.
{
 const map=new Map(), calls=[]; let reject=true;
 const api=async(p,b)=>{if(p.endsWith('/conversations'))return {contacts:[{username:'ben',conversation_handle:'h'}]};calls.push(b);if(reject)throw Error('lost');return receipt;};
 const attachment={data:'iVBORw0KGgo='};
 const ui=pane(map,api,'image caption',attachment);await ui.send();
 assert.equal(calls.length,2);assert.deepEqual(calls[0].image,attachment);
 assert.equal(calls[0].idempotency_key,calls[1].idempotency_key);
 reject=false;const reopened=pane(map,api,'');await reopened.send();
 assert.deepEqual(calls[2].image,attachment);assert.equal(calls[2].idempotency_key,calls[0].idempotency_key);
}
const helpers=src.slice(src.indexOf('const MAX_IMAGE_BYTES'),src.indexOf('function allowed'));
const {safeImageURL,pngAttachment}=new Function(helpers+';return {safeImageURL,pngAttachment};')();
assert.equal(safeImageURL({locked:false,mime:'image/png',url:'/api/messages/image?id=3'},3),'/api/messages/image?id=3');
for(const url of ['https://evil.test/x','//evil.test/x','data:image/png,x','/api/messages/image?id=4','/api/messages/image?id=3&x=1'])assert.equal(safeImageURL({locked:false,mime:'image/png',url},3),null);
assert.equal(safeImageURL({locked:true,mime:'image/png',url:'/api/messages/image?id=3'},3),null);
await assert.rejects(pngAttachment({size:2097153}),/2 MiB/);
await assert.rejects(pngAttachment({size:8,arrayBuffer:async()=>new Uint8Array(8).buffer}),/PNG file/);
assert.deepEqual(await pngAttachment({size:8,arrayBuffer:async()=>new Uint8Array([137,80,78,71,13,10,26,10]).buffer}),{data:'iVBORw0KGgo='});
console.log('Image attachment retry and URL validation checks passed.');

// Unlock replaces the same loaded messages, walking older pages; stale panes do not mutate.
{
 const code=src.slice(src.indexOf('  async function refreshLoaded('),src.indexOf('  function incoming('));
 const replaced=[];
 const node=m=>({m,replaceWith(n){replaced.push([m.id,n.m.image]);}});
 const view={nodes:new Map([[1,node({id:1})],[50,node({id:50})]])};
 const state={open:true,view,peer:'ben'};
 const calls=[];
 const api=async p=>{calls.push(p);return calls.length===1?{messages:[{id:50,image:{locked:false}}],more:true}:{messages:[{id:1,image:{locked:false}}],more:false};};
 const refresh=new Function('state','api','msgNode',code+';return refreshLoaded;')(state,api,node);
 await refresh(view,'ben');
 assert.deepEqual(replaced,[[50,{locked:false}],[1,{locked:false}]]);
 assert.match(calls[1],/before=50/);assert.equal(view.nodes.size,2);
 state.view={};await refresh(view,'ben');assert.equal(replaced.length,2);
}
console.log('Older image reveal and stale-view checks passed.');
