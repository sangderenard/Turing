// Real ComputeRuntime hidden preparation lifecycle. Product transport is a
// deterministic double; this does not execute or validate a GPU/native law.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {encodeStateStart,decodeStateStart} from '../examples/meowdoku/web/compute_state.js';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const temp=await fs.mkdtemp(path.join(os.tmpdir(),'meowdoku-prepare-'));
const saved={channel:globalThis.MessageChannel,fetch:globalThis.fetch,
  navigator:Object.getOwnPropertyDescriptor(globalThis,'navigator')};
const watchdog=setTimeout(()=>{console.error('Hidden preparation test exceeded 15 seconds');process.exit(1);},15000);
watchdog.unref();
const names=['x','y','z','vx','vy','vz','wx','wy','wz','age','rs'];
const diagnostics=['n_nan','n_out','n_drop','n_live','n_parked','ke','vmax2'];
const dt=1/360,pad=2,n=1;
const q=[0,Math.SQRT1_2,0,Math.SQRT1_2];
const seed=Object.fromEntries(names.map(name=>[name,new Float32Array(pad)]));
seed.y[0]=.2;seed.rs[0]=.01;seed.vx[0]=.125;seed.wz[0]=.25;
seed.age[0]=1;seed.age[1]=-1e9;
const manifest={law:{dt,state:names,diagnostics,runtime_inputs:[],gravity:9.81},
  cost_model:{min_speedup:1},settle:{sample:30,hold_samples:12,max_seconds:40},
  points:{x:'x',y:'y',z:'z'},sizes:{small:{n,pad,uniforms:{},display_stride:1,
    source_digest:'prepare-fixture',route:'webgpu-compute',verdict:{speedup:2},
    wgsl_abi:{dispatches:1},memory:{arena_bytes:88}}}};
const freshSpec=()=>({manifest:'fixture.json',size:'small',epoch:1,studio_count:1,
  uniforms:{},start:encodeStateStart(seed,pad,names,[q])});
const deferred=()=>{let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve};};
const turn=()=>new Promise(r=>setTimeout(r,0));
let context,checks=0;const failures=[];
async function check(name,fn){try{await fn();checks++;console.log(`PASS ${name}`);}
  catch(error){failures.push({name,error});console.error(`FAIL ${name}: ${error.stack}`);}}
try {
  for(const name of ['compute.js','compute_state.js','case_vibration.js','recording.js',
    'particle_styles.js','world_curve.js','world_zoom.js','noise_filter.js','mobroad.js',
    'case_motion.js','marble_rotation.js','emoji_tumble.js','marble_sim_preview.js'])
    await fs.copyFile(path.join(root,'examples/meowdoku/web',name),path.join(temp,name));
  await fs.copyFile(path.join(root,'examples/mobile_reference_app/web/turing-wasm-shell.js'),path.join(temp,'turing-wasm-shell.js'));
  await fs.writeFile(path.join(temp,'package.json'),'{"type":"module"}');
  await fs.appendFile(path.join(temp,'compute.js'),'\nexport {GpuProduct};\n');
  const {ComputeRuntime,GpuProduct}=await import(pathToFileURL(path.join(temp,'compute.js')));
  const {integrateAngular}=await import(pathToFileURL(path.join(temp,'marble_rotation.js')));
  Object.defineProperty(globalThis,'navigator',{configurable:true,value:{gpu:{}}});
  globalThis.MessageChannel=class {
    constructor(){const c=context;this.port1={close(){c.events.push('close1');}};
      this.port2={postMessage:()=>{c.posts++;},close(){c.events.push('close2');}};}
  };
  globalThis.fetch=async()=>{const c=context;await stage(c,'fetch');return {ok:true,json:async()=>structuredClone(manifest)};};
  async function stage(c,name){c.events.push(name);c.entered[name]?.resolve();await c.gates[name]?.promise;
    if(c.fail===name)throw new Error(`injected ${name} failure`);}
  GpuProduct.limits=async()=>{const c=context;c.limits++;await stage(c,'capabilities');return {ok:true};};
  GpuProduct.prototype.init=async function(){this.c=context;this.c.product=this;this.c.inits++;
    this.kind='gpu';this.frameBytes=pad*20;this.pool=[];this.failure=null;await stage(this.c,'load');};
  GpuProduct.prototype.reset=function({named,uniforms}){this.c.resets++;this.state=Object.fromEntries(names.map(k=>[k,new Float32Array(named[k])]));this.uniforms={...uniforms};};
  GpuProduct.prototype.setUniform=function(k,v){this.uniforms[k]=v;};
  GpuProduct.prototype.canBatch=()=>true;
  GpuProduct.prototype.batch=async function(request){const c=this.c;c.requests.push(request);await stage(c,'batch');
    // Transport fixture progression, not a replacement physics integrator.
    for(let i=0;i<pad;i++)this.state.x[i]=Math.fround(this.state.x[i]+this.state.vx[i]*request.snapEvery*dt);
    this.state.age[0]=Math.fround(this.state.age[0]+request.snapEvery*dt);
    const points=new Float32Array(pad*5);
    for(let i=0;i<pad;i++)for(const [j,key]of ['x','y','z'].entries())points[i*5+j]=this.state[key][i];
    if(c.divergent)points[0]+=1;
    const diag=c.diag===undefined?{n_nan:0,n_out:0,n_drop:0,n_live:n,n_parked:pad-n,ke:0,vmax2:0}:c.diag;
    return {frames:[points],spins:[c.noPose?null:[[0,0,.25]]],diags:[diag],
      keys:[{payload:{fixture:true},bytes:88}],emojiSpins:[null]};};
  GpuProduct.prototype.idle=async function(){await stage(this.c,'idle');if(this.c.deviceFailure)this.failure='injected device validation failure';};
  GpuProduct.prototype.columnsNow=async function(requested,count){this.c.readCount=count;await stage(this.c,'readback');
    if(this.c.nonfinite)this.state.vx[0]=NaN;
    return Object.fromEntries(requested.map(k=>[k,this.state[k].slice(0,count)]));};
  GpuProduct.prototype.freeKey=function(){this.c.freed++;};
  GpuProduct.prototype.destroy=function(){this.c.destroyed++;this.c.events.push('destroy');};
  function fixture(options={}){
    const c={events:[],posts:0,inits:0,limits:0,resets:0,destroyed:0,freed:0,requests:[],
      snapshots:0,samples:0,gates:{},entered:{},...options};context=c;
    const runtime=new ComputeRuntime({onStatus(){},onSupport(){},onSamples(){c.samples++;}});
    runtime.onSnapshot=()=>c.snapshots++;
    return {runtime,c};
  }
  function hold(c,stageName){c.gates[stageName]=deferred();c.entered[stageName]=deferred();return c.gates[stageName];}
  async function rejectedClean(runtime,c,promise,pattern){await assert.rejects(promise,pattern);await runtime.queue;
    assert.equal(runtime.ready,false);assert.equal(runtime.product,null);assert.equal(runtime.running,false);
    assert.equal(runtime.preparing,false);assert.equal(c.posts,0);assert.equal(c.snapshots,0);assert.equal(c.samples,0);}

  await check('one hidden native snapshot preserves pose, padding and paused ownership',async()=>{
    const {runtime,c}=fixture();const spec=freshSpec();
    const pending=runtime.preparePausedFirstFrame(spec);spec.start.state_f32.y='broken';spec.epoch=999;
    const result=await pending;
    assert.deepEqual(c.requests,[{frames:1,snapEvery:6,diagAt:[0],keyAt:[0]}]);
    assert.equal(c.posts,0);assert.equal(c.snapshots,0);assert.equal(c.samples,0);assert.equal(c.resets,1);
    assert.equal(runtime.ready,true);assert.equal(runtime.running,false);assert.equal(runtime.handoffPaused,true);
    assert.equal(result.frame,1);assert.equal(result.step,6);assert.equal(result.clock.epoch,1);
    assert.equal(c.readCount,pad);assert.equal(c.destroyed,0);
    assert.ok(c.events.indexOf('batch')<c.events.indexOf('idle'));
    assert.ok(c.events.indexOf('idle')<c.events.indexOf('readback'));
    const state=decodeStateStart(result.start,pad,names);
    assert.equal(state.x[0],Math.fround(.125*6*dt));assert.equal(state.y[0],seed.y[0]);
    assert.equal(state.age[1],seed.age[1]);assert.equal(state.vx[0],seed.vx[0]);
    assert.deepEqual(result.start.orientations_xyzw,[integrateAngular(q,[0,0,.25],6*dt)]);
    runtime.frame(100);assert.equal(runtime.shown,1);assert.equal(c.posts,0);
    runtime.control('continue');await turn();assert.equal(runtime.running,true);assert.equal(runtime.handoffPaused,false);
    assert.equal(c.resets,1);assert.equal(c.requests.length,1);assert.equal(c.posts,1);
    await runtime.disposeAfterIdle();assert.equal(c.destroyed,1);
  });
  await check('manifest gate runs before capability/device and refuses async approval',async()=>{
    for(const gate of [()=>false,()=>Promise.resolve(true),()=>{throw new Error('rejected scene');}]){
      const {runtime,c}=fixture();await rejectedClean(runtime,c,runtime.preparePausedFirstFrame(freshSpec(),{validateManifest:gate}),/admission|rejected scene/);
      assert.equal(c.limits,0);assert.equal(c.inits,0);
    }
  });
  await check('fresh runtime and bounded complete seed preconditions',async()=>{
    for(const options of [{maxNativeSteps:0},{maxNativeSteps:1.5}]){
      const {runtime,c}=fixture();await assert.rejects(runtime.preparePausedFirstFrame(freshSpec(),options),/bounded/);assert.equal(c.inits,0);
    }
    const {runtime,c}=fixture();await assert.rejects(runtime.preparePausedFirstFrame({...freshSpec(),start:{}}),/full named/);
    const gate=hold(c,'fetch');const pending=runtime.preparePausedFirstFrame(freshSpec());
    await assert.rejects(runtime.preparePausedFirstFrame(freshSpec()),/fresh detached/);gate.resolve();await pending;await runtime.disposeAfterIdle();
    const limited=fixture();await rejectedClean(limited.runtime,limited.c,limited.runtime.preparePausedFirstFrame(freshSpec(),{maxNativeSteps:5}),/needs 6 steps/);
    assert.equal(limited.c.requests.length,0);assert.equal(limited.c.destroyed,1);
  });
  await check('diagnostic, pose and displayed-state failures cannot reveal a candidate',async()=>{
    for(const options of [{diag:null},{diag:{n_nan:1}},{diag:{n_out:1}},{diag:{n_drop:1}},
      {diag:{n_live:0}},{diag:{n_parked:0}},{diag:{ke:NaN}},{noPose:true},{divergent:true},{deviceFailure:true},{nonfinite:true}]){
      const {runtime,c}=fixture(options);
      await rejectedClean(runtime,c,runtime.preparePausedFirstFrame(freshSpec()),/diagnostics|orientations|differs|validation|nonfinite/);
      assert.equal(c.destroyed,1);assert.equal(c.freed,1);
    }
  });
  await check('cancellation waits for load, batch, idle and readback settlement',async()=>{
    for(const phase of ['fetch','capabilities','load','batch','idle','readback']){
      const {runtime,c}=fixture();const gate=hold(c,phase),controller=new AbortController();
      const pending=runtime.preparePausedFirstFrame(freshSpec(),{signal:controller.signal});
      await c.entered[phase].promise;controller.abort();await turn();assert.equal(c.destroyed,0);
      gate.resolve();await rejectedClean(runtime,c,pending,/canceled|superseded/);
      assert.equal(c.destroyed,['fetch','capabilities'].includes(phase)?0:1);
      if(phase==='capabilities')assert.equal(c.inits,0);
    }
    const {runtime,c}=fixture();const controller=new AbortController();controller.abort();
    await rejectedClean(runtime,c,runtime.preparePausedFirstFrame(freshSpec(),{signal:controller.signal}),/canceled/);
    assert.equal(c.inits,0);
  });
  await check('superseding sync and playback cannot publish stale preparation',async()=>{
    for(const cancel of [r=>r.sync(null),r=>r.sync({...freshSpec(),epoch:2}),r=>r.control('continue'),r=>r.control('replay'),r=>r.control('rebirth')]){
      const {runtime,c}=fixture();const gate=hold(c,'readback'),pending=runtime.preparePausedFirstFrame(freshSpec());
      await c.entered.readback.promise;cancel(runtime);gate.resolve();
      await rejectedClean(runtime,c,pending,/canceled|superseded/);assert.equal(c.destroyed,1);
    }
  });
  await check('retirement is idempotent and cannot destroy pending transport',async()=>{
    for(const phase of ['fetch','capabilities','load','batch','idle','readback']){
      const {runtime,c}=fixture();const gate=hold(c,phase),pending=runtime.preparePausedFirstFrame(freshSpec());
      await c.entered[phase].promise;const retirement=runtime.disposeAfterIdle();assert.equal(runtime.disposeAfterIdle(),retirement);
      await turn();assert.equal(c.destroyed,0);gate.resolve();
      await rejectedClean(runtime,c,pending,/canceled|superseded/);await retirement;
      assert.equal(c.destroyed,['fetch','capabilities'].includes(phase)?0:1);assert.equal(runtime.retiring,true);
    }
  });
  await check('transport failures clean up exactly once after pending work',async()=>{
    for(const fail of ['fetch','capabilities','load','batch','idle','readback']){
      const {runtime,c}=fixture({fail});await rejectedClean(runtime,c,runtime.preparePausedFirstFrame(freshSpec()),/injected/);
      assert.equal(c.destroyed,['fetch','capabilities'].includes(fail)?0:1);
      await runtime.disposeAfterIdle();assert.equal(c.destroyed,['fetch','capabilities'].includes(fail)?0:1);
    }
  });
  await check('missing required diagnostic fields refuse admission',async()=>{
    for(const diag of [{},{ke:0,vmax2:0}]){
      const {runtime,c}=fixture({diag});await rejectedClean(runtime,c,runtime.preparePausedFirstFrame(freshSpec()),/diagnostic/);
      assert.equal(c.destroyed,1);
    }
  });
  assert.equal(failures.length,0,failures.map(f=>f.name).join('\n'));
  console.log(`${checks} hidden preparation lifecycle checks passed`);
} finally {
  clearTimeout(watchdog);
  globalThis.MessageChannel=saved.channel;globalThis.fetch=saved.fetch;
  if(saved.navigator)Object.defineProperty(globalThis,'navigator',saved.navigator);else delete globalThis.navigator;
  if(path.dirname(temp)!==path.resolve(os.tmpdir())||!path.basename(temp).startsWith('meowdoku-prepare-'))throw new Error('Unexpected temporary test path');
  await fs.rm(temp,{recursive:true,force:true});
}
