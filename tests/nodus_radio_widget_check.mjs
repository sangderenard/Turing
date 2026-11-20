// Offline lifecycle test: real header module, minimal DOM boundary, no network.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
class Node {
  constructor(tag='node') { this.tag=tag; this.children=[]; this.attrs={}; this.handlers={}; this.hidden=false; this.classes=new Set(); this.classList={toggle:(k,on)=>on?this.classes.add(k):this.classes.delete(k)}; this.style={setProperty(k,v){this[k]=v;}}; }
  setAttribute(k,v) {this.attrs[k]=v;if(k==='hidden')this.hidden=true;}
  getAttribute(k){return this.attrs[k];}
  removeAttribute(k){delete this.attrs[k];}
  replaceChildren(...nodes){this.children=[];this.append(...nodes);}
  append(...nodes){for(const n of nodes){this.children.push(n);n.parent=this;}}
  prepend(...nodes){for(const n of nodes.reverse()){this.children.unshift(n);n.parent=this;}}
  getBoundingClientRect(){return {bottom:48};}
  addEventListener(k,fn){(this.handlers[k] ||= []).push(fn);}
  removeEventListener(k,fn){this.handlers[k]=(this.handlers[k]||[]).filter(f=>f!==fn);}
  fire(k,event={}){for(const fn of [...this.handlers[k]||[]])fn(event);}
  click(){this.fire('click');}
  focus(){focused=this;}
  remove(){this.parent.children=this.parent.children.filter(n=>n!==this);}
  attachShadow(){return this.shadow=new Node('shadow');}
}
let focused=null,nextFrame=0;const frames=new Map();
const document=new Node('document');document.body=new Node('body');document.createElement=t=>{
  const node=new Node(t);
  if(t==='audio'){
    node.canPlayType=()=>process.argv.includes('--audio')?'probably':'';node.paused=true;node.pause=()=>{node.paused=true;};
    node.load=()=>{};node.play=async()=>{node.paused=false;};
  }
  return node;
};
document.querySelectorAll=()=>[{src:'test',getAttribute:()=> 'allow'}];
document.cookie='';
const window=new Node('window');window.dispatchEvent=e=>window.fire(e.type,e);
const context={document,window,console,CustomEvent:class{constructor(type,options){this.type=type;this.detail=options.detail;}},Event:class{constructor(type){this.type=type;}},requestAnimationFrame:fn=>{frames.set(++nextFrame,fn);return nextFrame;},cancelAnimationFrame:id=>frames.delete(id),fetch:async()=>({ok:false,status:503}),Set,Map,Number,TypeError,Error};
const admin = process.argv.includes('--admin'), withAudio = process.argv.includes('--audio'); let commands=[], jukeboxCommands=[];
let polls=0, timerId=0;const timers=new Map();
Object.assign(context,{URLSearchParams, location:{search:'',pathname:'/'}, setTimeout:fn=>{timers.set(++timerId,fn);return timerId;},clearTimeout:id=>timers.delete(id)});
const songId='upload:1:'+'a'.repeat(32);
const levels=new ArrayBuffer(96), levelView=new DataView(levels);
for(const [slot,frame,rms] of [[0,48000,.8],[1,96000,.4]]){
  const at=slot*48;
  levelView.setUint32(at,0x4441524e,true);levelView.setUint32(at+4,1,true);
  levelView.setUint32(at+8,5,true);levelView.setUint32(at+12,48000,true);
  levelView.setBigUint64(at+16,BigInt(frame),true);
  levelView.setFloat32(at+24,rms,true);
  for(let band=0;band<5;band++)levelView.setFloat32(at+28+4*band,rms,true);
}
context.fetch=async (path, options={})=>{
  if(path.startsWith('/api/radio/levels?id='))return {ok:true,
    headers:{get:k=>({'X-Radio-Track-Id':songId,'X-Radio-Analysis-Version':'1',
      'X-Radio-Frame-Hop':'256','X-Radio-Band-Count':'5','X-Radio-Sample-Rate':'48000'})[k]},
    arrayBuffer:async()=>levels};
  return {ok:true,json:async()=>path==='/api/whoami'?{signed_in:true,username:'dot',capabilities:admin?['view_site','account_manage']:['view_site']}:path==='/api/radio/jukebox'?(options.method==='POST'?(jukeboxCommands.push(JSON.parse(options.body)),{paused:true,catalogue:[{id:'song',title:'Test song'}],current:{id:1,song_id:'song'},requests:[],charts:[]}):{paused:true,catalogue:[{id:'song',title:'Test song'}],current:{id:1,song_id:'song'},requests:[],charts:[]}):path==='/api/radio/playback'?(options.method==='POST'?(commands.push(options),{status:JSON.parse(options.body).action==='start'?'playing':'stopped',configured:true,can_stop:JSON.parse(options.body).action==='start'}):{status:'stopped',configured:true,can_stop:false}):path==='/api/radio'?(polls++,{status:'live',title:'Tempest',audio:withAudio,song_id:songId,rms:.25,bands:[.1,.2,.3,.4,.5]}):{} };
};
vm.createContext(context);
const source=fs.readFileSync(new URL('../examples/meowdoku/web/site-nav.js',import.meta.url),'utf8').replace(/^export /gm,'').replace('import.meta.url','"test"');
vm.runInContext(source,context);
for(let i=0;i<12;i++)await Promise.resolve();
const host=document.body.children[0],nav=host.shadow.children[1],actions=nav.children[0],toggle=nav.children[1];
const chip=actions.children.find(n=>n.textContent==='Nodus Radio');assert.ok(chip);
const panel=actions.children.find(n=>n.attrs['aria-label']==='Nodus Radio');assert.ok(panel.hidden);assert.equal(polls,0);
chip.click();for(let i=0;i<12;i++)await Promise.resolve();
assert.equal(polls,1);assert.equal(panel.hidden,false);assert.ok(timers.size>=1);
assert.equal(panel.children.find(n=>n.className==='radio-bands').children.length,0);
assert.equal(panel.children.find(n=>n.tag==='strong').textContent,'Tempest');
if(withAudio){
  const listen=panel.children.find(n=>n.textContent==='Listen in this browser');
  const player=document.body.children.find(n=>n.tag==='audio');
  assert.equal(listen.hidden,false);
  listen.click();for(let i=0;i<12;i++)await Promise.resolve();
  assert.equal(player.src,'/api/radio/hls/radio.m3u8');
  for(let i=0;i<20;i++)await Promise.resolve();
  player.currentTime=1.02;player.fire('playing');player.fire('seeked');
  assert.ok(Math.abs(Number(panel.children.find(n=>n.className==='radio-bands').children[0].attrs.value)-.8)<1e-5);
}
toggle.click();for(let i=0;i<20;i++)await Promise.resolve();assert.ok(panel.hidden);
if(withAudio){
  const player=document.body.children.find(n=>n.tag==='audio');
  assert.equal(player.parent,document.body);assert.equal(player.paused,false);
  assert.equal(timers.size,1);
  const before=polls;
  for(const [id,fn] of [...timers]){timers.delete(id);fn();}for(let i=0;i<20;i++)await Promise.resolve();
  assert.equal(polls,before+1);
  player.currentTime=2.02;player.fire('seeked');
  assert.ok(Math.abs(Number(panel.children.find(n=>n.className==='radio-bands').children[0].attrs.value)-.4)<1e-5);
  toggle.click();chip.click();
  panel.children.find(n=>n.textContent==='Stop listening').click();
  chip.click();assert.equal(timers.size,0);
}else assert.equal(timers.size,0);
toggle.click();chip.click();chip.click();for(let i=0;i<12;i++)await Promise.resolve();assert.equal(timers.size,0);assert.ok(panel.hidden);
chip.click();for(let i=0;i<12;i++)await Promise.resolve();panel.fire('keydown',{key:'Escape',stopPropagation(){}});assert.equal(timers.size,0);assert.equal(focused,chip);
chip.click();document.fire('pointerdown',{composedPath:()=>[document]});for(let i=0;i<12;i++)await Promise.resolve();assert.equal(timers.size,0);
chip.click();window.fire('pagehide');for(let i=0;i<12;i++)await Promise.resolve();assert.equal(timers.size,0);
console.log('PASS: Radio lazy polling, real packet display, collapse, rapid toggles, Escape, outside close, unload, stale-response protection');

if(admin){
  chip.click();for(let i=0;i<12;i++)await Promise.resolve();
  const play=panel.children.find(n=>n.textContent==='Play on server speakers');
  const stop=panel.children.find(n=>n.textContent==='Stop my server playback');
  assert.ok(play);assert.equal(commands.length,0);assert.equal(play.disabled,false);assert.equal(stop.disabled,true);
  play.click();for(let i=0;i<12;i++)await Promise.resolve();
  assert.equal(commands.length,1);assert.equal(commands[0].headers['X-Meowdoku-Radio'],'1');assert.equal(JSON.parse(commands[0].body).action,'start');assert.equal(stop.disabled,false);
  stop.click();for(let i=0;i<12;i++)await Promise.resolve();assert.equal(commands.length,2);assert.equal(JSON.parse(commands[1].body).action,'stop');assert.equal(stop.disabled,true);
  console.log('PASS: Admin PA controls require explicit click, use guarded JSON route, distinguish server speakers, stop follows ownership');
}

chip.click(); for(let i=0;i<20;i++)await Promise.resolve();
if(panel.hidden){chip.click();for(let i=0;i<20;i++)await Promise.resolve();}
const request=panel.children.find(n=>n.textContent==='Request song');
const songs=panel.children.find(n=>n.attrs['aria-label']==='Song to request');songs.value='song';
request.click();request.click();for(let i=0;i<20;i++)await Promise.resolve();
assert.equal(jukeboxCommands.length,1);assert.equal(jukeboxCommands[0].action,'request');
const rating=panel.children.find(n=>n.attrs['aria-label']==='Rate the playing song');rating.value='0';
panel.children.find(n=>n.textContent==='Vote 0–5').click();for(let i=0;i<20;i++)await Promise.resolve();
assert.equal(jukeboxCommands[1].rating,0);assert.equal(jukeboxCommands[1].play_id,1);
const resume=panel.children.find(n=>n.textContent==='Start / resume jukebox');
assert.equal(!!resume,admin);
if(admin){resume.click();for(let i=0;i<20;i++)await Promise.resolve();assert.equal(jukeboxCommands.at(-1).action,'resume');}
panel.children.find(n=>n.textContent==='Close radio').click();assert.equal(timers.size,0);
console.log('PASS: Jukebox request debounce, explicit zero vote, and admin-only channel controls');
