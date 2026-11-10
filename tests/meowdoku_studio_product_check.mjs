// Consume an actually emitted manifest through unchanged ComputeRuntime load
// and reset. Device transport is a double; this does not claim GPU execution.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {encodeStateStart} from '../examples/meowdoku/web/compute_state.js';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const manifest=JSON.parse(await fs.readFile(process.argv[2],'utf8'));
const temp=await fs.mkdtemp(path.join(os.tmpdir(),'studio-native-load-'));
const oldChannel=globalThis.MessageChannel, oldFetch=globalThis.fetch;
const oldNavigator=Object.getOwnPropertyDescriptor(globalThis,'navigator');
try {
  for(const name of ['compute.js','compute_state.js','case_vibration.js','recording.js','particle_styles.js',
    'world_curve.js','world_zoom.js','noise_filter.js','mobroad.js','case_motion.js','marble_rotation.js','emoji_tumble.js'])
    await fs.copyFile(path.join(root,'examples/meowdoku/web',name),path.join(temp,name));
  await fs.copyFile(path.join(root,'examples/mobile_reference_app/web/turing-wasm-shell.js'),path.join(temp,'turing-wasm-shell.js'));
  await fs.writeFile(path.join(temp,'package.json'),'{"type":"module"}');
  await fs.appendFile(path.join(temp,'compute.js'),'\nexport {GpuProduct};\n');
  globalThis.MessageChannel=undefined;
  globalThis.fetch=async()=>({ok:true,json:async()=>manifest});
  Object.defineProperty(globalThis,'navigator',{configurable:true,value:{gpu:{}}});
  const {ComputeRuntime,GpuProduct}=await import(pathToFileURL(path.join(temp,'compute.js')));
  GpuProduct.limits=async()=>({ok:true});
  GpuProduct.prototype.init=async function(rec) {this.kind='webgpu';this.frameBytes=rec.n*20;this.pool=[];};
  GpuProduct.prototype.reset=function(seed) {this.loadedSeed=seed;};
  const runtime=new ComputeRuntime({onStatus(){},onSamples(){},onSupport(){}});
  const rec=manifest.sizes.studio,names=manifest.law.state;
  const columns=Object.fromEntries(names.map(name=>[name,new Float32Array(rec.pad)]));
  columns.y[0]=.209;columns.rs[0]=.01;
  const start=encodeStateStart(columns,rec.pad,names,[[0,0,0,1]]);
  runtime.sync({manifest:'studio.manifest.json',size:'studio',epoch:1,uniforms:rec.uniforms,start,studio_count:1});
  await runtime.queue;
  assert.equal(runtime.error,null);
  assert.equal(runtime.ready,true); assert.equal(runtime.running,true);
  assert.equal(runtime.product.loadedSeed.named.y[0],Math.fround(.209));
  assert.equal(runtime.product.loadedSeed.uniforms.grav_y,rec.uniforms.grav_y);
  for(const name of Object.keys(rec.wgsl_abi.uniform_index).filter(name=>name.startsWith('studio_')))
    assert.equal(runtime.product.loadedSeed.uniforms[name],rec.uniforms[name],name);
  assert.deepEqual(runtime.rotations.get(0)[0].quaternion,[0,0,0,1]);
  console.log('Actual Studio manifest passed ComputeRuntime load/reset ABI smoke');
} finally {
  globalThis.MessageChannel=oldChannel;globalThis.fetch=oldFetch;
  if(oldNavigator) Object.defineProperty(globalThis,'navigator',oldNavigator); else delete globalThis.navigator;
  await fs.rm(temp,{recursive:true,force:true});
}
