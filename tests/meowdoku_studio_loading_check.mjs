import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
for(const path of ['examples/meowdoku/web/marble_studio.js','build/meowdoku/web/marble_studio.js']){
 const source=fs.readFileSync(path,'utf8');
 assert(source.includes('// Reuse the server loading panel;'));
 assert(source.includes('import "./loading_panel.js";'));
 const boot=source.slice(source.indexOf('async function boot() {'),source.indexOf('\nboot();'));
 for(const failure of ['no-gl','fetch','gpu']){
  const calls=[];const message={textContent:''};
  const context={gl:failure==='no-gl'?null:{},window:{meowdokuLoading:{initializing:v=>calls.push('initializing'),done:()=>calls.push('done'),release:v=>calls.push(v)}},message,console:{error(){}},fetch:async()=>{if(failure==='fetch')throw Error('fetch failure');return{ok:true,text:async()=>'',json:async()=>({material_index:{}})}},shopEmissionProfiles:null,ids:null,glyphMeshes:null};
  await vm.runInNewContext(boot+'\nboot()',context);
  assert.deepEqual(calls.slice(-2),['done','boot']);
  if(failure==='gpu')assert(calls.includes('initializing'));
  if(failure==='fetch')assert.match(message.textContent,/fetch failure/);
 }
 assert(boot.indexOf('frame();')<boot.indexOf('finally { loading?.done()'));
 console.log(path+': no-WebGL/download/initialization failure release checked; first-frame release ordering checked');
}
