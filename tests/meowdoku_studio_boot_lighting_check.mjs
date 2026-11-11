import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import { LIGHTING_PROFILE_FIELDS } from '../examples/meowdoku/web/marble_studio_lighting.js';
const source=fs.readFileSync(new URL('../examples/meowdoku/web/marble_studio.js',import.meta.url),'utf8');
assert.match(source,/import\s*\{[^}]*\bLIGHTING_PROFILE_FIELDS\b[^}]*\}\s*from\s*["']\.\/marble_studio_lighting\.js["']/);
const start=source.indexOf('function studioBuilderLighting('),end=source.indexOf('\nfunction studioBuilderOptions(',start);
assert(start>=0&&end>start);
const state={controls:{held:42}},fallback={enabled:true};
const context={LIGHTING_PROFILE_FIELDS,studioRenderState:()=>null,studioLighting:null,studioControlValue:(id,channel)=>`${id}:${channel}`,studioTableState:()=>fallback,defaultStudioScale:()=>({unitMm:10}),shopLightState:()=>({lights:[]})};
// Load the helper region once, as the browser module does. Subsequent calls
// exercise changed host state without redeclaring module lexical bindings.
vm.createContext(context);
vm.runInContext(source.slice(start,end),context);
const initial=vm.runInContext('studioBuilderLighting("studio")',context);
assert.deepEqual(Object.keys(initial.controls),LIGHTING_PROFILE_FIELDS);
assert.equal(initial.controls['room-scale'],'room-scale:studio');
assert.equal(initial.table,fallback);
context.studioRenderState=()=>state;
assert.equal(vm.runInContext('studioBuilderLighting("studio")',context),state);
console.log('PASS: initial Studio lighting uses the imported authored field list before controls mount; mounted state retains identity');
