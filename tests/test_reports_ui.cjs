// DOM-free request/state tests. No browser, canvas, GPU, or rendered output.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
class Node {
  constructor() { this.value=''; this.hidden=false; this.disabled=false; this.checked=false; this.children=[]; this.handlers={}; this.textContent=''; }
  addEventListener(name, fn) { this.handlers[name]=fn; }
  replaceChildren(...children) { this.children=children; }
  append(child) { this.children.push(child); }
  querySelector() { return this.admin ||= new Node(); }
}
(async () => {
  const nodes = new Map(), get = id => { if (!nodes.has(id)) nodes.set(id,new Node()); return nodes.get(id); };
  get('scope').value='public';
  const id='a'.repeat(32), other='b'.repeat(32), events={}, requests=[];
  let revision=0, visibility='personal', releaseWrite, writes=0;
  const report = key => ({id:key,title:key===id?'<script>untrusted</script>':'New navigation',body:'<img onerror=bad>',
    prompt_history:['<b>text</b>'],next_steps:[{text:'Do not execute'}],can_share:true,access:{visibility,revision,group_id:null}});
  const context = {document:{getElementById:get,createElement:()=>new Node()},
    location:{hash:'#'+id}, window:{addEventListener:(name,fn)=>events[name]=fn},
    fetch:async (url,options) => {
      requests.push([url,options]);
      let value;
      if (options.method==='POST') { writes++; await new Promise(resolve=>releaseWrite=resolve); const body=JSON.parse(options.body); visibility=body.visibility; revision++; value={access:{visibility,revision}}; }
      else if(url==='/api/account/me') value={account:{id:1}};
      else if(url==='/api/whoami') value={role:'admin'};
      else if(url==='/api/account/groups') value={groups:[]};
      else if(/\/api\/reports\/[ab]{32}$/.test(url)) value={report:report(url.split('/').pop())};
      else value={reports:[],next_offset:null};
      return {ok:true,json:async()=>value};
    }};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../examples/meowdoku/web/reports.js'),'utf8'),context);
  const settle=async()=>{for(let i=0;i<8;i++) await new Promise(setImmediate);};
  await settle();
  assert(requests.some(([url])=>url==='/api/reports/public?offset=0'));
  assert.equal(get('scope').value,'public');
  assert.equal(get('title').textContent,'<script>untrusted</script>');
  assert.equal(get('body').textContent,'<img onerror=bad>');
  get('visibility').value='public'; get('visibility').handlers.change();
  assert.equal(get('signoff-label').hidden,false);
  await get('sharing').handlers.submit({preventDefault(){}});
  assert.equal(writes,0);
  get('signoff').checked=true;
  get('cancel').handlers.click();
  assert.equal(get('visibility').value,'personal'); assert.equal(get('signoff').checked,false);
  get('visibility').value='public'; get('signoff').checked=true;
  const pending=get('sharing').handlers.submit({preventDefault(){}});
  await settle();
  await get('sharing').handlers.submit({preventDefault(){}});
  assert.equal(writes,1,'repeated click cannot issue another write');
  context.location.hash='#'+other; events.hashchange(); await settle();
  assert.equal(get('title').textContent,'New navigation');
  releaseWrite(); await pending; await settle();
  assert.equal(get('title').textContent,'New navigation','late save must not resurrect an older detail');
  assert.equal(get('save').disabled,false);
  assert(requests.some(([url])=>url==='/api/reports/'+id));
  console.log('Report UI request/state checks passed: public default, inert text, sign-off, cancel, repeated click, navigation race, readback.');
})().catch(error=>{console.error(error);process.exitCode=1;});
