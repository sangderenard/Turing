import assert from 'node:assert/strict';
import {validateStudioCameraState} from '../examples/meowdoku/web/standalone_studio.js';
import fs from 'node:fs';
const I=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const s={view:I,projection:I,width:320,height:240};
const result=validateStudioCameraState(s);assert.deepEqual(result,s);assert.notEqual(result.view,I);
for(const bad of [{...s,width:0},{...s,height:2049},{...s,view:[NaN]}, {...s,controls:true}])assert.throws(()=>validateStudioCameraState(bad));
const source=fs.readFileSync(new URL('../examples/meowdoku/web/standalone_studio.js',import.meta.url),'utf8');
assert(!/requestAnimationFrame|addEventListener|setInterval|new Worker/.test(source));
console.log('Standalone Studio camera validation and no-control contract passed');

const page=fs.readFileSync(new URL('../examples/meowdoku/web/standalone_studio.html',import.meta.url),'utf8');
assert(page.includes('if(renderer||loading)'));assert(page.includes('finally{loading=false;}'));
