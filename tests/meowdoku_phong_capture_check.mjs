import assert from 'node:assert/strict';
import {createPhongCapture,phongRecordingMime,phongCaptureFilename} from '../examples/meowdoku/web/marble_phong_capture.js';

const png=new Blob([new Uint8Array([137,80,78,71,13,10,26,10,1])],{type:'image/png'});
class Recorder {
  static isTypeSupported(type){return type==='video/webm;codecs=vp8';}
  constructor(stream,options){this.stream=stream;this.mimeType=options.mimeType;this.state='inactive';Recorder.last=this;}
  start(){this.state='recording';}
  stop(){this.state='inactive';queueMicrotask(()=>{this.ondataavailable?.({data:new Blob(['video'])});this.onstop?.();});}
}
function fixture(options={}){
  const source={width:640,height:480},events=[],artifacts=[],surfaces=[],timers=new Map();let time=0,id=0;
  function makeCanvas(){const track={requestFrame(){events.push('frame');},stop(){events.push('stop-track');}},target={width:0,height:0,
    getContext(){return {drawImage(c){assert.equal(c,source);events.push('copy');}};},
    toBlob(callback){queueMicrotask(()=>callback(png));},captureStream(){return {getTracks:()=>[track],getVideoTracks:()=>[track]};}};
    surfaces.push(target);return target;}
  const capture=createPhongCapture({canvas:source,makeCanvas,Recorder,now:()=>time,setTimer:(fn,ms)=>{timers.set(++id,{fn,ms});return id;},clearTimer:i=>timers.delete(i),
    onState:x=>events.push(x),onArtifact:x=>artifacts.push(x),...options});
  return {capture,source,events,artifacts,surfaces,timers,tick:ms=>time+=ms};
}
assert.equal(phongRecordingMime(Recorder),'video/webm;codecs=vp8');assert.equal(phongRecordingMime(null),null);
assert.match(phongCaptureFilename('video/mp4',new Date('2026-01-01Z')),/\.mp4$/);assert.throws(()=>phongCaptureFilename('text/plain'));
{
 const f=fixture(),p=f.capture.takePhoto();await assert.rejects(f.capture.takePhoto(),/already/);
 assert.equal(f.artifacts.length,0);f.capture.frame();const a=await p;assert.equal(a.blob,png);assert.deepEqual([a.width,a.height],[640,480]);
 assert.equal(f.artifacts.length,1);assert.equal(f.capture.state.photoPending,false);f.capture.dispose();assert.equal(f.timers.size,0);
}
{
 const f=fixture(),p=f.capture.start({fps:30});assert.throws(()=>f.capture.start(),/already/);
 f.capture.frame();assert.equal(f.capture.state.videoState,'recording');f.capture.frame();assert.equal(f.events.filter(x=>x==='frame').length,1);
 f.tick(34);f.capture.frame();assert.equal(f.events.filter(x=>x==='frame').length,2);
 const stopped=f.capture.stop();assert.equal(f.capture.state.videoState,'stopping');const a=await stopped;assert.equal(await p,a);assert.equal(a.blob.type,'video/webm;codecs=vp8');
 assert.equal(a.fps,30);assert.equal(f.events.filter(x=>x==='stop-track').length,1);assert.equal(f.timers.size,0);assert.equal(f.capture.state.videoState,'idle');
 const again=f.capture.start();f.capture.frame();f.tick(34);f.capture.frame();await f.capture.stop();await again;assert.equal(f.artifacts.length,2);f.capture.dispose();
}
{
 const f=fixture(),p=f.capture.start();await assert.rejects(f.capture.stop(),/before/);await assert.rejects(p,/before/);assert.equal(f.capture.state.videoState,'idle');assert.equal(f.timers.size,0);
}
{
 const f=fixture(),p=f.capture.start();f.capture.frame();f.source.width=800;f.capture.frame();const a=await p;assert.match(a.reason,/size changed/);assert.equal(a.width,640);
}
{
 const f=fixture(),p=f.capture.start();f.capture.frame();const timer=[...f.timers.values()].find(t=>t.ms===120000);assert.ok(timer);timer.fn();assert.match((await p).reason,/two-minute/);
}
{
 const f=fixture({limits:{pixels:16777216,seconds:120,bytes:2}}),p=f.capture.start();f.capture.frame();Recorder.last.ondataavailable({data:new Blob(['too large'])});await assert.rejects(p,/memory limit/);assert.equal(f.artifacts.length,0);
}
{
 const f=fixture(),p=f.capture.takePhoto();f.capture.fail(new Error('lost'));await assert.rejects(p,/lost/);assert.equal(f.capture.state.ready,false);f.capture.dispose();
}
{
 const f=fixture(),p=f.capture.start();f.capture.frame();f.capture.fail(new Error('context lost'));assert.match((await p).reason,/context lost/);assert.equal(f.capture.state.videoState,'idle');
}
{
 const f=fixture({Recorder:null});assert.throws(()=>f.capture.start(),/unavailable/);const p=f.capture.takePhoto();f.capture.frame();assert.equal((await p).kind,'photo');
}
{
 const f=fixture(),p=f.capture.start();f.capture.frame();f.capture.dispose();await assert.rejects(p,/closed/);assert.equal(f.artifacts.length,0);assert.equal(f.timers.size,0);
}
{
 const f=fixture();f.source.width=20000;f.source.height=20000;assert.throws(()=>f.capture.start(),/million/);assert.throws(()=>f.capture.start({fps:7}),/Choose/);
}
console.log('Phong capture: PNG, manual video cadence, repeat/stop, resize, timeout, memory, loss and capability checks passed.');
// Failure paths must release admission and allow the next user attempt.
{
 const f=fixture(),p=f.capture.start();[...f.timers.values()][0].fn();await assert.rejects(p,/No completed/);assert.equal(f.capture.state.videoState,'idle');
}
{
 const f=fixture(),p=f.capture.start();f.capture.frame();Recorder.last.onerror({error:new Error('codec error')});await assert.rejects(p,/codec error/);assert.equal(f.artifacts.length,0);assert.equal(f.timers.size,0);
}
{
 const f=fixture({makeCanvas:()=>({getContext:()=>({drawImage(){}}),toBlob:callback=>callback(null)})}),p=f.capture.takePhoto();f.capture.frame();await assert.rejects(p,/encode/);assert.equal(f.capture.state.photoPending,false);assert.equal(f.timers.size,0);
}
{
 const f=fixture({makeCanvas:()=>({getContext:()=>({drawImage(){}}),toBlob(){}})}),p=f.capture.takePhoto();f.capture.frame();assert.equal(f.capture.state.photoPending,true);
 [...f.timers.values()][0].fn();await assert.rejects(p,/encoding timed out/);assert.equal(f.capture.state.photoPending,false);
}
console.log('Phong capture encoder failure and stalled-frame recovery passed.');
