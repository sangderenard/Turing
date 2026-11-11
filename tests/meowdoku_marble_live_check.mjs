import assert from "node:assert/strict";
import {createLiveRayView,liveDimensions,liveHistoryLimit,MAX_LIVE_HISTORY_BYTES,LIVE_HISTORY_WARNING_BYTES,liveEmitterSampling} from "../examples/meowdoku/web/marble_live.js";
import {createPhotoBounceControls} from "../examples/meowdoku/web/marble_photo.js";
const flush=async()=>{for(let i=0;i<100;i++)await Promise.resolve();};
const defer=()=>{let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve};};
const A="a".repeat(64),B="b".repeat(64);
const reply=(value,status=200)=>({ok:status>=200&&status<300,status,json:async()=>value,blob:async()=>({})});
assert.deepEqual(liveDimensions(256,2,16),{width:256,height:128,bytes:7077888});
for(const checkpoint of [false,true])for(const edge of [128,256,512,768,1024,2048,4096])for(const aspect of [.1,.5,1,2,10]){
 const max=liveHistoryLimit(edge,aspect,checkpoint);
 assert.ok(liveDimensions(edge,aspect,max,checkpoint).bytes<=MAX_LIVE_HISTORY_BYTES);
 assert.throws(()=>liveDimensions(edge,aspect,max+1,checkpoint),/2 GiB/);
}
assert.throws(()=>liveHistoryLimit(4096,1,false,128*1024*1024),/GPU cannot fit/);
assert.equal(liveHistoryLimit(128,1),10920);
assert.equal(liveHistoryLimit(128,1,true),10914);
assert.equal(liveHistoryLimit(512,1,true),674);
assert.equal(liveHistoryLimit(512,1,false,128*1024*1024),42,"actual per-buffer device limit wins");
assert.equal(liveDimensions(128,1,2728).bytes,LIVE_HISTORY_WARNING_BYTES-131072);
assert.ok(liveDimensions(128,1,2729).bytes>LIVE_HISTORY_WARNING_BYTES);
function harness(options={}){
 const saved=Object.fromEntries(["fetch","ResizeObserver","setTimeout","clearTimeout","createImageBitmap"].map(k=>[k,globalThis[k]]));
 const elements=new Map(),timers=new Map(),requests=[],paints=[];let tid=0,hook=null,revision=0,poseError=null,pose={eye_m:[0,0,.07],target_m:[0,0,.18],fov_degrees:18,focus_distance_m:.11,aperture_radius_m:.002};
 class Element extends EventTarget{
  constructor(id){super();Object.assign(this,{id,value:"",hidden:false,checked:false,style:{},width:1,height:1,clientWidth:400,clientHeight:400,textContent:"",classList:{toggle(){}}});}
  closest(selector){if(selector===".stage")return stage;return null;}
  getContext(){return {clearRect(){paints.push("clear");},drawImage(){paints.push("draw");}};}
  setAttribute(name,value){this[name]=value;}
  fire(type){this.dispatchEvent(new Event(type));}
 }
 const $=id=>{if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id);};
 const stage=new Element("stage"),doc=new EventTarget();doc.getElementById=$;doc.defaultView=new EventTarget();doc.defaultView.Event=Event;doc.hidden=false;
 $("live-history").value="16";$("live-resolution").value="128";$("preview-engine").value="visual";$("preview-motion").value="off";$("photo-bounces").value="4";$("photo-gamma").value="100";
 $("live-tint").value="0";$("live-normalization").value="auto99";$("live-normalization-white").value="1";$("photo-bounce-mode").value="manual";
 $("live-normalization-percentile").value="99";$("live-ray-batch").value="8192";
 $("live-display-mode").value="overlay";$("live-emitter-ray-budget").value="4096";$("live-emitter-seed").value="42";$("live-emitter-refresh").checked=true;
 for(const [id,value] of Object.entries({exposure:0,whitebalance:0,saturation:100,vibrance:0,"black-point":0,"firefly-threshold":800,"firefly-strength":100}))$("live-"+id).value=String(value);
 $("photo-triangle-budget").value="250000";$("photo-sphere-coverage").value="mesh";
 createPhotoBounceControls(doc);
 globalThis.ResizeObserver=class{observe(){}disconnect(){}};
 globalThis.setTimeout=(fn)=>{timers.set(++tid,fn);return tid;};globalThis.clearTimeout=id=>timers.delete(id);
 globalThis.createImageBitmap=async()=>({width:128,height:128,close(){}});
 globalThis.fetch=async(url,options={})=>{
  const request={url,options};requests.push(request);if(hook){const value=hook(request);if(value)return value;}
  if(url==="/api/marble_photo")return reply({digest:B,state:"preparing"});
  if(url==="/api/marble_photo/live"){const body=JSON.parse(options.body);revision=body.revision??revision;return reply({ok:true,revision});}
  if(url.startsWith("/api/marble_photo/image"))return reply({});
  return reply({digest:B,state:"live",live_revision:revision,image:"preview-000016.png",history_samples:16,history_filled:16,history_bytes:3538944,width:128,height:128});
 };
 const geometry={interior:[[[0,0,0],[1,0,0],[0,1,0],0,[[0,0,1],[0,0,1],[0,0,1]]]],glass:[]};
 const api=createLiveRayView({...options,doc,canvas:$("view"),getGeometry:()=>geometry,getScene:()=>({}),getPose:()=>{if(poseError)throw new Error(poseError);return structuredClone(pose);},getFilmResponse:()=>({version:1,centers_nm:[650,550,450],sigmas_oct:[.06,.06,.06]})});
 return {api,$,doc,requests,paints,set hook(x){hook=x;},set pose(x){pose={...pose,...x};},set poseError(x){poseError=x;},
  photoEdit(id,value){$(id).value=value;const event=new Event("input");Object.defineProperty(event,"target",{value:$(id)});$("photo-controls").dispatchEvent(event);},
  async enable(){api.saved(A);$("live-rays").checked=true;$("live-rays").fire("change");await flush();},
  async tick(){const entry=timers.entries().next().value;assert.ok(entry);timers.delete(entry[0]);entry[1]();await flush();},
  async close(){api.dispose();await flush();Object.assign(globalThis,saved);}};
}
{
 const observations=[],connections=[],ticket={stencil:12,gridSize:64,region:'scene'};
 const fieldView={capture:()=>ticket,connect:(...args)=>connections.push(args),observe:(...args)=>observations.push(args),invalidate(){},stop(){},revise(){}};
 const h=harness({fieldView});try {
  h.$('live-display-mode').value='field-feed';h.$('live-display-mode').fire('change');
  assert(h.$('phong-light-field').checked);assert(h.$('live-emitter-sampling').checked);
  h.$('live-field-internal-media').checked=true;
  h.$('ray-surface-transport').checked=true;
  await h.enable();assert(h.api.enabled);assert(h.api.fieldFeed);assert.equal(connections.length,1);assert.equal(observations.length,1);
  assert.equal(h.$('live-ray-view').hidden,true);assert.equal(h.$('live-ray-view').style.opacity,'0');assert.equal(h.$('view').style.opacity,'');
  assert(!h.requests.some(r=>r.url.startsWith('/api/marble_photo/image')));assert(!h.paints.includes('draw'));
  const body=h.requests.find(r=>r.url==='/api/marble_photo').options.body;
  const config=JSON.parse(new TextDecoder().decode(new Uint8Array(body,16,new DataView(body).getUint32(4,true))));
  assert.deepEqual(config.emitter_sampling,{version:1,ray_budget:4096,seed:42,refresh_each_sweep:true});assert.equal(config.receive_light_field,true);
  assert.equal(config.light_field_internal_media,true);
  assert.equal(config.surface_transport,'studio-pbr-mixture-v1');
  await h.tick();assert.equal(observations.length,2);assert.equal(h.requests.filter(r=>r.url==='/api/marble_photo').length,1);
  assert.match(h.$('live-status').textContent,/worker still computes sensor samples/);
  assert.match(h.$('live-status').textContent,/No receiving categories.*ordinary primary Phong/);
  for(const name of ['subject','stage','room'])assert.equal(h.$(`phong-field-${name}`).checked,false,'field-feed never silently opts categories in');
  h.$('live-display-mode').value='overlay';h.$('live-display-mode').fire('change');await h.tick();assert(h.paints.includes('draw'));
  h.$('live-display-mode').value='field-feed';h.$('live-display-mode').fire('change');await h.tick();
  assert.equal(h.requests.filter(r=>r.url==='/api/marble_photo').length,1,'display-only mode changes keep the same captured worker');
  assert.equal(h.$('live-ray-view').hidden,true);
  h.$('live-emitter-ray-budget').value='127';assert.throws(()=>liveEmitterSampling(h.doc),/budget/);
  h.$('live-emitter-ray-budget').value='4096';h.$('live-emitter-seed').value='2147483648';assert.throws(()=>liveEmitterSampling(h.doc),/seed/);
 }finally{await h.close();}
}
{
 const fieldView={capture:()=>({stencil:12,gridSize:64,region:'scene'}),connect(){},observe(){},invalidate(){},stop(){},revise(){}};
 const h=harness({fieldView}),image=defer();try {
  h.$('live-emitter-sampling').checked=true;
  h.hook=r=>r.url.startsWith('/api/marble_photo/image')?image.promise:null;
  await h.enable();h.$('live-display-mode').value='field-feed';h.$('live-display-mode').fire('change');
  image.resolve(reply({}));await flush();assert(!h.paints.includes('draw'),'late image cannot cover field-feed Phong');
  assert.equal(h.$('live-ray-view').hidden,true);
 }finally{image.resolve(reply({}));await h.close();}
}
{
 const h=harness();try{
  await h.enable();assert.equal(h.api.enabled,true);assert.equal(h.$("view").style.opacity,"");assert.equal(h.$("live-ray-view").style.opacity,"1");assert.ok(h.paints.includes("draw"));
  const upload=h.requests.find(x=>x.url==="/api/marble_photo").options.body;
  const view=new DataView(upload),config=JSON.parse(new TextDecoder().decode(new Uint8Array(upload,16,view.getUint32(4,true))));
  assert.equal(config.live.history_samples,16);assert.equal(config.transport,"fixed:3");assert.equal(config.max_bounces,4);
  h.pose={eye_m:[.1,0,.07]};await h.tick();assert.equal(JSON.parse(h.requests.filter(x=>x.url.endsWith("/live")).at(-1).options.body).revision,1);
  h.$("live-history").value="8";await h.tick();assert.equal(JSON.parse(h.requests.filter(x=>x.url.endsWith("/live")).at(-1).options.body).live.history_samples,8);
  assert.equal(h.requests.filter(x=>x.url==="/api/marble_photo").length,1,"camera and length must not restart scene");
  await h.api.stop();assert.equal(h.$("view").style.opacity,"");assert.equal(h.$("live-ray-view").hidden,true);
  assert.ok(h.requests.some(x=>x.url.endsWith("/live")&&JSON.parse(x.options.body).stop));
  assert.ok(!h.requests.some(x=>x.url.endsWith("/cancel")),"never use photograph cancellation");
 }finally{await h.close();}
}
{
 const h=harness();try{
  h.hook=r=>r.url==="/api/marble_photo"?reply({error:"GPU is busy"},409):null;
  await h.enable();assert.equal(h.api.enabled,false);assert.match(h.$("live-status").textContent,/GPU is busy/);
  assert.equal(h.requests.length,1,"busy photo must not be canceled or controlled");
 }finally{await h.close();}
}
{
 const h=harness(),pending=defer();try{
  h.hook=r=>r.url==="/api/marble_photo"?pending.promise:null;
  await h.enable();h.$("live-rays").checked=false;h.$("live-rays").fire("change");await flush();
  pending.resolve(reply({digest:B,state:"preparing"}));await flush();
  assert.equal(h.api.enabled,false);assert.ok(!h.paints.includes("draw"));
  assert.ok(h.requests.some(x=>x.url.endsWith("/live")&&JSON.parse(x.options.body).digest===B&&JSON.parse(x.options.body).stop));
 }finally{await h.close();}
}
{
 const h=harness();try{
  await h.enable();h.api.stale();await flush();assert.equal(h.api.enabled,false);assert.match(h.$("live-status").textContent,/Save it again/);
  h.$("live-rays").checked=true;h.$("live-rays").fire("change");await flush();assert.equal(h.api.enabled,false);
 }finally{await h.close();}
}
{
 const h=harness();try{
  h.$("preview-engine").value="particle";await h.enable();assert.equal(h.api.enabled,false);assert.equal(h.requests.length,0);
 }finally{await h.close();}
}
{
 const h=harness();try{
  await h.enable();h.doc.hidden=true;h.doc.dispatchEvent(new Event("visibilitychange"));await flush();
  assert.equal(h.api.enabled,false);assert.match(h.$("live-status").textContent,/tab is hidden/);
 }finally{await h.close();}
}
{
 const h=harness(),pending=defer();try{
  await h.enable();const first=h.requests.filter(x=>x.url==="/api/marble_photo")[0].options.body;
  h.hook=r=>r.url.endsWith("/live")&&JSON.parse(r.options.body).stop?pending.promise:null;
  h.$("live-rays").checked=false;h.$("live-rays").fire("change");await flush();
  h.$("live-rays").checked=true;h.$("live-rays").fire("change");await flush();
  assert.equal(h.requests.filter(x=>x.url==="/api/marble_photo").length,1,"start waits for earlier stop");
  pending.resolve(reply({state:"canceled"}));await flush();
  assert.equal(h.api.enabled,true);assert.equal(h.requests.filter(x=>x.url==="/api/marble_photo").length,2);
  const second=h.requests.filter(x=>x.url==="/api/marble_photo")[1].options.body;
  const config=b=>JSON.parse(new TextDecoder().decode(new Uint8Array(b,16,new DataView(b).getUint32(4,true))));
  assert.notEqual(config(first).live.session_id,config(second).live.session_id,"new toggle has distinct owner identity");
  h.hook=null;
 }finally{pending.resolve(reply({}));await h.close();}
}
{
 const h=harness(),originalNow=Date.now;let now=10000;try{
  Date.now=()=>now;
  h.$("photo-bounces").value="12";h.$("photo-bounces").fire("input");
  await h.enable();
  const configs=()=>h.requests.filter(x=>x.url==="/api/marble_photo").map(x=>{
    const bytes=x.options.body;return JSON.parse(new TextDecoder().decode(new Uint8Array(bytes,16,new DataView(bytes).getUint32(4,true))));
  });
  assert.equal(configs()[0].max_bounces,12,"legacy actual count is not a nonlinear slider position");
  h.$("photo-bounces-slider").value="100";h.$("photo-bounces-slider").fire("input");
  const event=new Event("input");Object.defineProperty(event,"target",{value:h.$("photo-bounces-slider")});
  h.$("photo-controls").dispatchEvent(event);
  now+=400;await h.tick();
  assert.equal(configs().length,2,"range input rebuilds the live scene");
  assert.equal(configs()[1].max_bounces,200,"live and still photos use the same actual count");
 }finally{Date.now=originalNow;await h.close();}
}
{
 const h=harness();try{
  h.$("live-history-count").value="2728";h.$("live-history-count").fire("change");assert.equal(h.$("live-history-memory").style.color,"");
  h.$("live-history-count").value="2729";h.$("live-history-count").fire("change");assert.equal(h.$("live-history-memory").style.color,"#ff0000");
  h.$("live-history-count").value="10920";h.$("live-history-count").fire("change");assert.equal(h.$("live-history").value,"10920");
  h.$("live-checkpoint").checked=true;h.$("live-checkpoint").fire("change");assert.equal(h.$("live-history").value,"10914");
  h.$("live-resolution").value="512";h.$("live-resolution").fire("change");assert.equal(h.$("live-history").value,"674");
  assert.match(h.$("live-history-memory").textContent,/2,145,386,496 bytes/);
  assert.equal(h.requests.length,0,"editing bounds does not start GPU work");
 }finally{await h.close();}
}
{
 const h=harness();try{
  await h.enable();const clears=h.paints.filter(x=>x==="clear").length;h.$("live-tint").value="150";h.$("live-tint").fire("input");
  h.$("live-normalization").value="manual";h.$("live-normalization").fire("change");h.$("live-normalization-white").value="2000";
  await h.tick();const control=JSON.parse(h.requests.filter(x=>x.url.endsWith("/live")).at(-1).options.body);
  assert.deepEqual(control.live.display,{display_look:"reference",tint:1.5,normalization:"manual",normalization_white:2000,normalization_percentile:99,tone_curve:"exponential",exposure:0,whitebalance:0,saturation:1,vibrance:0,black_point:0,firefly_threshold:8,firefly_strength:1,firefly_filter:false});
  assert.equal(h.requests.filter(x=>x.url==="/api/marble_photo").length,1,"display controls do not restart scene");
  assert.equal(h.paints.filter(x=>x==="clear").length,clears,"display-only edits do not blank the current image");
  assert.equal(h.$("live-normalization-white").disabled,false);
 }finally{await h.close();}
}
{
 const h=harness();try{
  let effectiveWhite=8,liveRevision=0;
  h.hook=r=>{
    if(r.url==="/api/marble_photo/live"){
      const body=JSON.parse(r.options.body);liveRevision=body.revision??liveRevision;
      if(body.live?.display)effectiveWhite=body.live.display.normalization_percentile/10;
      return reply({ok:true,revision:liveRevision});
    }
    if(r.url.startsWith("/api/marble_photo?digest="))return reply({state:"live",live_revision:liveRevision,
      image:`preview-${liveRevision}.png`,history_samples:16,history_filled:16,history_bytes:3538944,width:128,height:128,
      display_white:999,display_transform:{effective_white:effectiveWhite},
      retouched_image:"different-retouch.png",retouch_display_transform:{effective_white:123}});
    return null;
  };
  h.$("live-checkpoint").checked=true;await h.enable();
  assert.equal(h.$("live-normalization-percentile").disabled,true);
  const clears=h.paints.filter(x=>x==="clear").length;
  h.$("live-normalization").value="percentile";h.$("live-normalization").fire("change");
  assert.equal(h.$("live-normalization-percentile").disabled,false);
  h.$("live-normalization-percentile").value="97.2";h.$("live-normalization-percentile").fire("input");await h.tick();
  let control=JSON.parse(h.requests.filter(x=>x.url.endsWith("/live")).at(-1).options.body);
  assert.equal(control.live.display.normalization_percentile,97.2);assert.equal(control.live.checkpoint,true);
  h.$("live-normalization").value="specular";h.$("live-normalization").fire("change");
  assert.equal(h.$("live-normalization-percentile").value,"95");await h.tick();
  control=JSON.parse(h.requests.filter(x=>x.url.endsWith("/live")).at(-1).options.body);
  assert.equal(control.live.display.normalization,"specular");assert.equal(control.live.display.tone_curve,"linear");
  h.$("live-normalization-percentile").value="91.4";h.$("live-normalization-percentile").fire("input");
  h.$("live-normalization").fire("change");assert.equal(h.$("live-normalization-percentile").value,"91.4");await h.tick();
  h.$("live-normalization").value="manual";h.$("live-normalization").fire("change");
  assert.equal(h.$("live-normalization-white").value,String(91.4/10),"manual uses the current displayed image's effective white, not stale99 or a different retouch");
  await h.tick();
  assert.equal(h.requests.filter(x=>x.url==="/api/marble_photo").length,1,"display mode and percentile edits do not start another scene");
  assert.equal(h.paints.filter(x=>x==="clear").length,clears,"mode and slider edits preserve displayed rays and checkpoint");
  assert.ok(!h.requests.some(x=>x.url.endsWith("/live")&&JSON.parse(x.options.body).stop));
 }finally{await h.close();}
}
for(const value of ["", "NaN", "Infinity", ".99", "100.01"]){
 const h=harness();try{
  h.$("live-normalization").value="percentile";h.$("live-normalization").fire("change");
  h.$("live-normalization-percentile").value=value;await h.enable();
  assert.equal(h.api.enabled,false);assert.equal(h.requests.length,0);assert.match(h.$("live-status").textContent,/percentile/);
 }finally{await h.close();}
}
{
 const h=harness();try{
  h.poseError="Zoom farther out for this saved radius";await h.enable();assert.equal(h.api.enabled,false);assert.equal(h.requests.length,0);
  h.poseError=null;await h.enable();const before=h.requests.length;
  h.poseError="Incomplete axial focus";await h.tick();assert.equal(h.api.enabled,true);
  assert(!h.requests.slice(before).some(r=>r.url.endsWith('/live')&&JSON.parse(r.options.body).stop),'invalid camera edit retains owned worker');
  assert.match(h.$('live-status').textContent,/Previous worker retained/);
  assert.equal(h.$('live-ray-view').style.opacity,'0');
  h.poseError=null;h.pose={focus_distance_m:.005};await h.tick();assert.equal(h.api.enabled,true);
  assert.equal(JSON.parse(h.requests.filter(r=>r.url.endsWith('/live')).at(-1).options.body).camera.focus_distance_m,.005);
  assert.equal(h.requests.filter(r=>r.url==='/api/marble_photo').length,1,'valid recovery sends control, not another scene');
 }finally{await h.close();}
}
{
 const h=harness();try{
  for(const [id,value] of [["photo-triangle-budget",""],["photo-triangle-budget","999"],["photo-sphere-coverage","polygon"]]){
   h.$("photo-triangle-budget").value="250000";h.$("photo-sphere-coverage").value="mesh";h.$(id).value=value;
   await h.enable();assert.equal(h.api.enabled,false);assert.equal(h.requests.length,0,"invalid policy never uploads or cancels");
  }
 }finally{await h.close();}
}
{
 const h=harness(),originalNow=Date.now;let now=10000;try{
  Date.now=()=>now;await h.enable();
  const configs=()=>h.requests.filter(x=>x.url==="/api/marble_photo").map(x=>{
   const bytes=x.options.body;return JSON.parse(new TextDecoder().decode(new Uint8Array(bytes,16,new DataView(bytes).getUint32(4,true))));
  });
  const edits=(id,value)=>{
   h.$(id).value=value;const event=new Event("input");Object.defineProperty(event,"target",{value:h.$(id)});
   h.$("photo-controls").dispatchEvent(event);
  };
  const stops=()=>h.requests.filter(x=>x.url.endsWith("/live")&&JSON.parse(x.options.body).stop).length;
  assert.equal(configs()[0].triangle_budget,250000);assert.equal(configs()[0].sphere_coverage,"mesh");
  for(const [id,value] of [["photo-triangle-budget",""],["photo-triangle-budget","999"],["photo-sphere-coverage","polygon"]]){
   h.$("photo-triangle-budget").value="250000";h.$("photo-sphere-coverage").value="mesh";
   edits(id,value);now+=400;await h.tick();
   assert.equal(h.api.enabled,true);assert.equal(configs().length,1);assert.equal(stops(),0);
   assert.match(h.$("live-status").textContent,/Live scene unchanged/);
  }
  h.$("photo-sphere-coverage").value="mesh";
  edits("photo-triangle-budget","1000");edits("photo-triangle-budget","");now+=400;await h.tick();
  assert.equal(stops(),0,"incomplete edit cancels a pending valid restart before replacing the worker");
  assert.equal(configs().length,1);
  edits("photo-triangle-budget","123456");now+=400;await h.tick();
  assert.equal(stops(),0,"server owns same-session replacement after validation");assert.equal(configs().length,2);
  assert.equal(configs()[1].triangle_budget,123456);assert.equal(configs()[1].sphere_coverage,"mesh");
  edits("photo-triangle-budget","1000");h.$("photo-triangle-budget").value="";now+=400;await h.tick();
  assert.equal(stops(),0,"policy is rechecked after debounce even without another DOM event");
  assert.equal(configs().length,2);assert.equal(h.api.enabled,true);
  assert.ok(!h.requests.some(x=>x.url.endsWith("/cancel")),"never cancels a photograph");
 }finally{Date.now=originalNow;await h.close();}
}
{
 const h=harness(),originalNow=Date.now;let now=10000;try{
  Date.now=()=>now;await h.enable();h.pose={eye_m:[.1,0,.07]};await h.tick();
  const controls=()=>h.requests.filter(x=>x.url.endsWith("/live")).map(x=>JSON.parse(x.options.body));
  assert.equal(controls().at(-1).revision,1);
  h.hook=r=>r.url==="/api/marble_photo"?reply({error:"Scene needs 24450 triangles; exposure budget is 1000."},400):null;
  const draws=h.paints.filter(x=>x==="draw").length;
  h.photoEdit("photo-triangle-budget","1000");now+=400;await h.tick();
  assert.equal(h.api.enabled,true);assert.ok(!controls().some(x=>x.stop),"rejected replacement never pre-cancels the old worker");
  for(let i=0;i<5;i++){now+=1000;await h.tick();}
  assert.equal(h.requests.filter(x=>x.url==="/api/marble_photo").length,2,"a rejected scene is not retried every poll");
  assert.ok(h.requests.at(-1).url.includes(B));
  assert.equal(controls().at(-1).revision,1,"old accepted pose and revision survive a rejected replacement");
  assert.equal(h.paints.filter(x=>x==="draw").length,draws,"rejected scene shows Phong rather than obsolete rays");
  assert.equal(h.$("live-ray-view").style.opacity,"0");
  assert.match(h.$("live-status").textContent,/Live scene unchanged: Scene needs 24450/);
  h.hook=null;h.photoEdit("photo-triangle-budget","300000");now+=400;await h.tick();
  assert.equal(h.requests.filter(x=>x.url==="/api/marble_photo").length,3,"a later relevant edit retries once");
  assert.ok(!controls().some(x=>x.stop));assert.doesNotMatch(h.$("live-status").textContent,/Scene needs/);
 }finally{Date.now=originalNow;await h.close();}
}
for(const rejected of [false,true]){
 const h=harness(),pending=defer(),originalNow=Date.now,C="c".repeat(64);let now=10000;try{
  Date.now=()=>now;await h.enable();
  h.hook=r=>r.url==="/api/marble_photo"?pending.promise:null;
  h.photoEdit("photo-triangle-budget","300000");now+=400;await h.tick();
  assert.ok(!h.requests.some(x=>x.url.endsWith("/live")&&JSON.parse(x.options.body).stop));
  h.$("live-rays").checked=false;h.$("live-rays").fire("change");await flush();
  const draws=h.paints.filter(x=>x==="draw").length;
  pending.resolve(rejected?reply({error:"over budget"},400):reply({digest:C,state:"preparing"}));await flush();
  assert.equal(h.api.enabled,false);assert.equal(h.$("live-ray-view").hidden,true);
  assert.equal(h.paints.filter(x=>x==="draw").length,draws,"Off cannot be undone by a late replacement response");
  const stopped=h.requests.filter(x=>x.url.endsWith("/live")&&JSON.parse(x.options.body).stop).map(x=>JSON.parse(x.options.body).digest);
  assert.deepEqual(stopped,rejected?[B]:[B,C],"Off cleans up both the old and any late successful replacement");
  assert.doesNotMatch(h.$("live-status").textContent,/Live scene unchanged/);
 }finally{pending.resolve(reply({error:"over budget"},400));Date.now=originalNow;await h.close();}
}
{
 const h=harness(),originalNow=Date.now;let now=10000;try{
  Date.now=()=>now;await h.enable();
  h.hook=r=>r.url==="/api/marble_photo"?Promise.reject(new Error("uncertain network outcome")):null;
  h.photoEdit("photo-triangle-budget","300000");now+=400;await h.tick();
  assert.equal(h.api.enabled,false,"uncertain replacement errors must not pretend the old scene is still authoritative");
  assert.ok(h.requests.some(x=>x.url.endsWith("/live")&&JSON.parse(x.options.body).stop));
  assert.match(h.$("live-status").textContent,/uncertain network outcome/);
 }finally{Date.now=originalNow;await h.close();}
}
{
 const h=harness(),originalNow=Date.now;let now=10000;try{
  Date.now=()=>now;await h.enable();h.pose={eye_m:[.1,0,.07]};await h.tick();h.pose={eye_m:[0,0,.07]};await h.tick();
  h.api.sceneChanged();now+=400;await h.tick();now+=3500;await h.tick();
  const controls=h.requests.filter(x=>x.url.endsWith("/live")).map(x=>JSON.parse(x.options.body));
  assert.equal(controls.at(-1).revision,2,"server reuse of the same immutable digest preserves the current revision");
  assert.ok(!controls.some(x=>x.stop));assert.equal(h.api.enabled,true);
 }finally{Date.now=originalNow;await h.close();}
}
for(const reused of [false,true]){
 const h=harness(),pending=defer(),originalNow=Date.now,C="c".repeat(64);let now=10000,uploads=0;try{
  Date.now=()=>now;await h.enable();
  h.hook=r=>r.url==="/api/marble_photo"?(++uploads===1?pending.promise:reply({error:"over budget"},400)):null;
  h.api.sceneChanged();now+=400;await h.tick();
  h.photoEdit("photo-triangle-budget","1000");
  pending.resolve(reply({digest:reused?B:C,state:"preparing"}));await flush();
  now+=400;await h.tick();
  const stopped=h.requests.filter(x=>x.url.endsWith("/live")&&JSON.parse(x.options.body).stop).map(x=>JSON.parse(x.options.body).digest);
  assert.deepEqual(stopped,reused?[]:[C],"a stale newly created job is cleaned up, but reused active state is retained");
  assert.equal(h.api.enabled,reused,"a rejected later edit cannot resume a digest the preceding replacement already superseded");
 }finally{pending.resolve(reply({error:"over budget"},400));Date.now=originalNow;await h.close();}
}
{
 const h=harness(),originalNow=Date.now,C="c".repeat(64);let now=10000,accept=false;try{
  Date.now=()=>now;h.$("live-resolution").value="512";await h.enable();
  h.hook=r=>{
   if(r.url==="/api/marble_photo")return accept?reply({digest:C,state:"preparing"}):reply({error:"Scene needs more triangles"},400);
   if(r.url.endsWith("/live")&&!accept&&JSON.parse(r.options.body).live?.history_samples>16)
    return reply({error:"history would overflow old high-resolution worker"},400);
   return null;
  };
  const draws=h.paints.filter(x=>x==="draw").length;
  h.$("live-resolution").value="128";h.$("live-resolution").fire("change");
  h.$("live-history-count").value="1000";h.$("live-history-count").fire("change");
  h.photoEdit("photo-triangle-budget","1000");now+=400;await h.tick();
  h.pose={eye_m:[.1,0,.07]};h.$("live-tint").value="150";
  now+=3500;await h.tick();
  const controls=h.requests.filter(x=>x.url.endsWith("/live")).map(x=>JSON.parse(x.options.body));
  assert.equal(h.api.enabled,true);assert.ok(!controls.some(x=>x.stop));
  assert.equal(controls.at(-1).live.history_samples,16,"a rejected low-resolution resize cannot leak its larger capacity into the old worker");
  assert.equal(controls.at(-1).revision,0);assert.equal(controls.at(-1).live.display.tint,0);
  assert.deepEqual(controls.at(-1).camera.eye_m,[0,0,.07],"lease renewal uses the last accepted camera and settings");
  assert.equal(h.paints.filter(x=>x==="draw").length,draws,"unaccepted UI scene must not display an old ray image");
  assert.equal(h.$("view").style.opacity,"");
  assert.match(h.$("live-status").textContent,/Live scene unchanged/);
  accept=true;h.photoEdit("photo-triangle-budget","300000");now+=400;await h.tick();
  const bytes=h.requests.filter(x=>x.url==="/api/marble_photo").at(-1).options.body;
  const config=JSON.parse(new TextDecoder().decode(new Uint8Array(bytes,16,new DataView(bytes).getUint32(4,true))));
  assert.equal(config.camera.resolution,128);assert.equal(config.live.history_samples,1000);assert.equal(config.live.display.tint,1.5);
  assert.equal(h.api.enabled,true);assert.doesNotMatch(h.$("live-status").textContent,/Live scene unchanged/);
 }finally{Date.now=originalNow;await h.close();}
}
console.log("Live ray UI lifecycle, percentile/specular controls, and memory boundary matrix PASS");
{
 const h=harness();try{
  await h.enable();
  let image='invalid-rods.png',unresolved=9,revision=0;
  h.hook=r=>r.url.startsWith('/api/marble_photo?digest=')?reply({state:'live',live_revision:revision,image,
    history_samples:16,history_filled:16,history_bytes:3538944,width:128,height:128,
    rod_bundle_diagnostic:{unresolved_samples:unresolved,policy:'reject',warning:'Native rod cut unresolved.'}}):null;
  await h.tick();assert.equal(h.$('live-optical-warning').hidden,false);
  assert.match(h.$('live-optical-warning').textContent,/INVALID IMAGE: 9/);
  unresolved=0;await h.tick();assert.equal(h.$('live-optical-warning').hidden,false,'same image cannot erase an invalid result');
  image='valid-rods.png';await h.tick();assert.equal(h.$('live-optical-warning').hidden,true,'a newly painted explicitly valid image clears warning');
  unresolved=4;image='invalid-again.png';await h.tick();assert.equal(h.$('live-optical-warning').hidden,false);
  await h.api.stop();assert.equal(h.$('live-optical-warning').hidden,false,'stop does not certify the last downloaded image');
 }finally{await h.close();}
}
console.log('Persistent native rod invalid-image warning follows actual live image delivery PASS');
for(const mode of ['glass_rod','injection_dye']) {
 const h=harness({getOpticalMode:()=>mode});try {
  await h.enable();assert.equal(h.api.enabled,true);
  assert.equal(h.requests.filter(x=>x.url==="/api/marble_photo").length,1,'optical mode must use the ordinary live upload path');
 }finally{await h.close();}
}

{
 const h=harness(),pending=defer();try{
  h.hook=r=>r.url==="/api/marble_photo"?pending.promise:null;
  await h.enable();assert.equal(h.$("view").style.opacity,"");assert.equal(h.$("live-ray-view").style.opacity,"0","preparing leaves Phong visible");
  pending.resolve(reply({digest:B}));await flush();assert.equal(h.$("live-ray-view").style.opacity,"1");
  h.api.cameraChanged();assert.equal(h.$("live-ray-view").style.opacity,"0","orbit hides old revision immediately");
  h.pose={eye_m:[.1,0,.07]};await h.tick();assert.equal(h.$("live-ray-view").style.opacity,"1");
  assert.equal(h.$("live-ray-view")["aria-hidden"],"false");await h.api.stop();assert.equal(h.$("live-ray-view").style.opacity,"0");
  assert.equal(h.$("live-ray-view")["aria-hidden"],"true");
 }finally{pending.resolve(reply({digest:B}));await h.close();}
}
{
 const pending=defer(),h=harness({ensureSaved:()=>pending.promise});try{
  await h.enable();assert.equal(h.requests.length,0);h.$("live-rays").checked=false;h.$("live-rays").fire("change");
  pending.resolve();await flush();assert.equal(h.requests.length,0,"Off during autosave cannot later start rays");assert.equal(h.api.enabled,false);
 }finally{pending.resolve();await h.close();}
}
{
 const h=harness(),originalNow=Date.now;let now=10000;try{
  Date.now=()=>now;await h.enable();h.api.pendingSave("Saving latest object");
  h.$("live-resolution").value="512";h.$("live-history").value="4000";
  now+=3500;await h.tick();assert.equal(h.api.enabled,true);assert.equal(h.$("live-ray-view").style.opacity,"0");
  const control=JSON.parse(h.requests.filter(x=>x.url.endsWith("/live")).at(-1).options.body);
  assert.equal(control.live.history_samples,16,"pending object heartbeat retains accepted settings");
  assert.ok(!h.requests.some(x=>x.url.endsWith("/live")&&JSON.parse(x.options.body).stop));
  h.$("live-resolution").value="128";h.$("live-history").value="16";h.api.saved("c".repeat(64));now+=400;await h.tick();
  assert.equal(h.requests.filter(x=>x.url==="/api/marble_photo").length,2);assert.equal(h.$("live-ray-view").style.opacity,"1");
 }finally{Date.now=originalNow;await h.close();}
}
{
 const h=harness();try{
  await h.enable();h.$("live-ray-batch").value="12345";h.$("live-ray-batch").fire("change");assert.equal(h.$("live-ray-view").style.opacity,"0");await h.tick();
  const control=JSON.parse(h.requests.filter(x=>x.url.endsWith("/live")).at(-1).options.body);
  assert.equal(control.live.ray_batch_size,12345);assert.equal(control.revision,1);assert.equal(h.requests.filter(x=>x.url==="/api/marble_photo").length,1);
 }finally{await h.close();}
}
console.log("Transparent Phong fallback, autosave lease, start cancellation, and actual ray-batch UI PASS");

{
 const h=harness();try{
  await h.enable();h.$("live-ray-batch").value="";h.$("live-ray-batch").fire("change");await h.tick();
  assert.equal(h.api.enabled,true);assert.equal(h.$("live-ray-view").style.opacity,"0");
  assert.ok(!h.requests.some(x=>x.url.endsWith("/live")&&JSON.parse(x.options.body).stop));
  assert.match(h.$("live-status").textContent,/previous worker is retained/);
  h.$("live-ray-batch").value="65536";h.$("live-ray-batch").fire("change");await h.tick();
  assert.equal(h.api.enabled,true);assert.equal(h.$("live-ray-view").style.opacity,"1");
  assert.equal(JSON.parse(h.requests.filter(x=>x.url.endsWith("/live")).at(-1).options.body).live.ray_batch_size,65536);
 }finally{await h.close();}
}
console.log("Incomplete batch edits retain the worker; valid edits recover without restart PASS");

{
 const h=harness();try{
  await h.enable();const clears=h.paints.filter(x=>x==="clear").length;
  for(const [id,value] of Object.entries({exposure:-200,whitebalance:75,saturation:120,vibrance:35,"black-point":3,"firefly-threshold":400,"firefly-strength":50})){
   h.$("live-"+id).value=String(value);h.$("live-"+id).fire("input");
  }
  h.$("live-firefly-filter").checked=true;h.$("live-firefly-filter").fire("change");
  await h.tick();const display=JSON.parse(h.requests.filter(x=>x.url.endsWith("/live")).at(-1).options.body).live.display;
  assert.equal(display.exposure,-2);assert.equal(display.whitebalance,.75);assert.equal(display.saturation,1.2);
  assert.equal(display.vibrance,.35);assert.equal(display.black_point,.03);assert.equal(display.firefly_filter,true);
  assert.equal(display.firefly_threshold,4);assert.equal(display.firefly_strength,.5);
  h.$("live-firefly-filter").checked=false;h.$("live-firefly-filter").fire("change");
  await h.tick();
  assert.equal(JSON.parse(h.requests.filter(x=>x.url.endsWith("/live")).at(-1).options.body).live.display.firefly_filter,false,"unchecking sends explicit false");
  assert.equal(h.$("live-firefly-threshold").disabled,true);
  assert.equal(h.$("live-firefly-strength").disabled,true);
  h.$("live-firefly-filter").checked=true;h.$("live-firefly-filter").fire("change");
  await h.tick();
  assert.equal(JSON.parse(h.requests.filter(x=>x.url.endsWith("/live")).at(-1).options.body).live.display.firefly_filter,true,"rechecking restores filter");
  assert.equal(h.$("live-firefly-threshold").disabled,false);
  assert.equal(h.paints.filter(x=>x==="clear").length,clears,"display edits preserve visible image and ray history");
  assert.equal(h.requests.filter(x=>x.url==="/api/marble_photo").length,1,"no scene restart for display controls");
  h.$("live-exposure").value="";await h.tick();assert.equal(h.api.enabled,true);
  assert.match(h.$("live-status").textContent,/Previous display and rays retained/);
  assert.equal(h.paints.filter(x=>x==="clear").length,clears,"invalid display edit preserves visible image");
  h.$("live-exposure").value="-100";await h.tick();assert.equal(h.api.enabled,true);
  assert.equal(h.requests.filter(x=>x.url==="/api/marble_photo").length,1);
  assert.equal(JSON.parse(h.requests.filter(x=>x.url.endsWith("/live")).at(-1).options.body).live.display.exposure,-1);
 }finally{await h.close();}
}
console.log("Live exposure/WB/color/firefly controls preserve rays and reject incomplete edits PASS");

{
 const h=harness();try{
  await h.enable();const clears=h.paints.filter(x=>x==="clear").length;
  h.$("live-display-look").value="color_priority";h.$("live-display-look").fire("change");await h.tick();
  let c=JSON.parse(h.requests.filter(x=>x.url.endsWith("/live")).at(-1).options.body);
  assert.equal(c.live.display.display_look,"color_priority");
  h.$("live-display-look").value="reference";h.$("live-display-look").fire("change");await h.tick();
  c=JSON.parse(h.requests.filter(x=>x.url.endsWith("/live")).at(-1).options.body);
  assert.equal(c.live.display.display_look,"reference");
  assert.equal(h.requests.filter(x=>x.url==="/api/marble_photo").length,1);
  assert.equal(h.paints.filter(x=>x==="clear").length,clears);
 }finally{await h.close();}
}

{
 const h=harness();try {
  h.hook=r=>r.url==="/api/marble_photo"?{ok:false,status:503,json:async()=>{throw new Error("not JSON");}}:null;
  await h.enable();assert.equal(h.api.enabled,false);assert.match(h.$("live-status").textContent,/HTTP 503.*unreadable/);
 }finally{await h.close();}
}
{
 const h=harness({ensureSaved:()=>{throw new Error("Material mode unavailable");}});try {
  await h.enable();assert.equal(h.api.enabled,false);assert.match(h.$("live-status").textContent,/Material mode unavailable/);
 }finally{await h.close();}
}
{
 const fs=await import('node:fs');const html=fs.readFileSync(new URL('../examples/meowdoku/web/marble_studio.html',import.meta.url),'utf8');
 assert(html.indexOf('id="live-status"')>html.indexOf('id="live-rays"'));
 assert(html.indexOf('id="live-status"')<html.indexOf('id="live-history"'),'live reason must be immediately visible beside toggle, before all lengthy settings');
 assert(html.includes('aria-describedby="live-status"'));
}
console.log('Adjacent live refusal, malformed HTTP reason and unexpected start failure PASS');
