import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const path=process.argv[2]||'examples/meowdoku/web/marble_studio.js';
const text=fs.readFileSync(path,'utf8'),start=text.indexOf('  let last = null;'),end=text.indexOf('  // Initialization dispatches synthetic inputs',start);
assert.ok(start>=0&&end>start,'Existing Studio camera input block');
const input=text.slice(start,end);
function target(){const listeners=new Map();return {hidden:false,listeners,addEventListener(k,fn){const list=listeners.get(k)||[];list.push(fn);listeners.set(k,list);},send(k,e={}){e={type:k,preventDefault(){this.prevented=true;},...e};for(const fn of listeners.get(k)||[])fn(e);return e;},setPointerCapture(){},hasPointerCapture(){return true;},releasePointerCapture(){}};}
function setup(){const canvas=target(),window=target(),document=target(),settings={'preview-engine':{value:'off'},'preview-camera':{value:'orbit'}};
 const c={canvas,window,document,Math,Map,performance:{now:()=>1000},yaw:0,pitch:0,distance:4,followDistance:.5,changed:0,chasing:false,
  $:id=>settings[id],photo:{lensMinimumFocus:()=>.2},studioSubjectTransform:()=>({unitM:1}),isMarbleChase:()=>c.chasing,
  liveRay:{cameraChanged(){c.changed++;}},chaseCamera:{drag(){},orbit(dx,dy){c.yaw+=dx;c.pitch+=dy;},zoomBy(log){c.followDistance*=Math.exp(log);}}};
 vm.createContext(c);vm.runInContext(input,c);return {c,settings,send:(k,id,x,y,more={})=>canvas.send(k,{pointerId:id,clientX:x,clientY:y,pointerType:'touch',button:0,...more})};}
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
{
 const {c,send}=setup();send('pointerdown',1,0,0);send('pointerdown',2,100,0);send('pointermove',2,200,0);
 near(c.distance,2);near(c.yaw,0);near(c.pitch,0);assert.equal(c.changed,1);
 send('pointermove',2,50,0);near(c.distance,8);assert.equal(c.changed,2);
 send('pointerup',2,50,0);send('pointermove',1,50,40);near(c.yaw,0);near(c.pitch,0);
 send('pointerup',1,50,40);send('pointerdown',3,10,10);send('pointermove',3,20,30);near(c.yaw,.08);near(c.pitch,.16);
}
for(const stop of ['pointerup','pointercancel','lostpointercapture']){
 const {c,send}=setup();send('pointerdown',1,0,0);send(stop,1,0,0);send('pointermove',1,20,20);near(c.yaw,0);
 send('pointerdown',2,10,10);send('pointermove',2,20,10);near(c.yaw,.08);
}
for(const stop of ['blur','visibilitychange']){
 const {c,send}=setup();send('pointerdown',1,0,0);send('pointerdown',2,100,0);
 if(stop==='blur')c.window.send(stop);else {c.document.hidden=true;c.document.send(stop);}
 send('pointermove',2,200,0);near(c.distance,4);near(c.yaw,0);
}
{
 const {c,send}=setup();send('pointermove',77,9,9);near(c.yaw,0);
 send('pointerdown',1,0,0);send('pointerdown',2,100,0);send('pointerdown',3,200,0);send('pointermove',2,300,0);near(c.distance,4);
 send('pointerup',3,200,0);send('pointermove',2,400,0);near(c.distance,4);
}
{
 const {c,send}=setup();send('pointerdown',1,0,0);send('pointerdown',2,0,0);send('pointermove',2,0,0);near(c.distance,4);
 send('pointermove',2,100,0);near(c.distance,4);send('pointermove',2,200,0);near(c.distance,2);
}
{
 const {c,send}=setup();send('pointerdown',1,0,0);send('pointerdown',2,100,0);send('pointermove',2,100000,0);near(c.distance,.2);
 send('pointermove',2,.01,0);near(c.distance,100000);near(c.yaw,0);
}
{
 const {c}=setup();const e=c.canvas.send('wheel',{deltaY:100});near(c.distance,4*Math.exp(.1));assert.ok(e.prevented);
}
{
 const {c,settings,send}=setup();c.chasing=true;settings['preview-engine'].value='particle';settings['preview-camera'].value='marble';
 send('pointerdown',1,0,0);send('pointerdown',2,100,0);send('pointermove',2,200,0);near(c.followDistance,.25);near(c.distance,4);
}
console.log(`Studio pointer pinch, orbit, zoom limits and interruption checks passed: ${path}`);
