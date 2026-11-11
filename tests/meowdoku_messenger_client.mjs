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
const make=new Function('input','location','crypto','pendingSends','peer','api','note',code);
const receipt={ok:true,id:1,created:1};
let nextKey=0;
function pane(map,api,value='hello') {
 const input={value,disabled:false}, notes=[];
 const send=make(input,{protocol:'https:'},{randomUUID:()=> 'op-'+(++nextKey)},map,'ben',api,m=>notes.push(m));
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
