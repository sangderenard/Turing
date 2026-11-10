import assert from 'node:assert/strict';
import fs from 'node:fs';import vm from 'node:vm';
import {defaultFixtureSizes,resolveUnderlightFit,validateFixtureSizes,applyFixtureSizes} from '../examples/meowdoku/web/marble_fixture_sizes.js';
import {tableLightRoles,tableLightModifierFaces} from '../examples/meowdoku/web/marble_table_lights.js';
import {defaultStudioTable} from '../examples/meowdoku/web/marble_studio_table.js';
import {roomSettingLeaves} from '../examples/meowdoku/web/marble_lighting_profiles.js';
const source=fs.readFileSync(new URL('../examples/meowdoku/web/marble_studio.js',import.meta.url),'utf8');
const context=vm.createContext({});vm.runInContext(source.slice(source.indexOf('function boxHousing('),source.indexOf('function tubeHousing(')),context);
for(const [width,length]of [[8,4],[2,2],[20,20],[5.2,3]])for(const [shortMargin,longMargin]of [[0,0],[.05,.1],[.2,.15]]){
 const table={...defaultStudioTable(),width,length};const sizes=resolveUnderlightFit({...defaultFixtureSizes(),underlightFit:{shortMargin,longMargin}},table);
 const role=applyFixtureSizes([tableLightRoles()[2]],sizes)[0];context.role=role;
 const faces=[...tableLightModifierFaces(role,99),...vm.runInContext('boxHousing(role.pos,role.dir,role.halfW*1.25,role.halfH*1.25,role.housingDepth,99)',context)];
 const pts=faces.flatMap(f=>f.slice(0,3)),maxX=Math.max(...pts.map(p=>Math.abs(p[0]))),maxZ=Math.max(...pts.map(p=>Math.abs(p[2])));
 assert.ok(Math.abs(maxX-(width/2-.34-longMargin))<1e-10);assert.ok(Math.abs(maxZ-(length/2-shortMargin))<1e-10);
 assert.deepEqual(validateFixtureSizes(sizes),sizes);
}
const legacy=defaultFixtureSizes(true);legacy.lights[9].width=3.12;legacy.lights[9].height=.77;
assert.equal(resolveUnderlightFit(legacy,defaultStudioTable()),legacy);
assert.deepEqual(validateFixtureSizes(legacy),legacy);
assert.throws(()=>resolveUnderlightFit({...defaultFixtureSizes(),underlightFit:{shortMargin:2,longMargin:0}},defaultStudioTable()));
assert.throws(()=>resolveUnderlightFit({...defaultFixtureSizes(),underlightFit:{shortMargin:NaN,longMargin:0}},defaultStudioTable()));
const paths=roomSettingLeaves({fixtureSizes:defaultFixtureSizes()}).map(v=>v.path);
assert.ok(paths.includes('fixtureSizes/underlightFit/shortMargin'));assert.ok(!paths.includes('fixtureSizes/lights/@underlight/width'));
console.log('PASS actual can/apron mesh bounds at 4 table sizes and 3 margins; legacy identity, roundtrip, impossible margins, margin-only Room leaves.');
