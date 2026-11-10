import {tableLightRoles,tableLightModifierFaces} from '../examples/meowdoku/web/marble_table_lights.js';
import assert from 'node:assert/strict';
import {defaultFixtureSizes,validateFixtureSizes,applyFixtureSizes,SIZED_FIXTURES} from '../examples/meowdoku/web/marble_fixture_sizes.js';
import {STUDIO_LIGHTS} from '../examples/meowdoku/web/marble_light_strengths.js';
assert.deepEqual(SIZED_FIXTURES.map(f=>f.id),STUDIO_LIGHTS.map(f=>f.id));
const legacy=validateFixtureSizes(undefined),fresh=defaultFixtureSizes();
assert.ok(legacy.lights.every(l=>l.width===1&&(l.height===undefined||l.height===1)));
const roles=SIZED_FIXTURES.map((f,i)=>({roleIndex:i,round:!f.rect,...(f.rect?{halfW:.3,halfH:.4}:{radius:.2}),gobo:!!f.snoot}));
const old=applyFixtureSizes(roles,legacy),next=applyFixtureSizes(roles,fresh);
for(const [i,f]of SIZED_FIXTURES.entries()){
 if(f.rect){const factor=f.id==='fill-upper'?8/.6:f.id==='fill-lower'?4/.6:f.id==='underlight'?fresh.lights[i].width:2;assert.equal(old[i].halfW,.3);assert.equal(next[i].halfW,.3*factor);assert.equal(next[i].halfH,.4*(f.id==='fill-lower'?2/.6:f.id==='underlight'?fresh.lights[i].height:factor));}
 else assert.equal(next[i].radius,.2);
 if(f.snoot){assert.equal(old[i].shroudLength,.2*2.2);assert.equal(next[i].gobo,true);}
}
const custom=structuredClone(fresh);custom.lights[0].width=3;custom.lights[0].height=.5;custom.lights[3].snootLength=.8;
assert.ok(Math.abs(applyFixtureSizes(roles,custom)[0].halfW-.9)<1e-12);
assert.equal(applyFixtureSizes(roles,custom)[0].halfH,.2);
assert.equal(applyFixtureSizes(roles,custom)[3].shroudLength,.8);
custom.lights[3].snoot=false;assert.equal(applyFixtureSizes(roles,custom)[3].gobo,false);
assert.deepEqual(validateFixtureSizes(JSON.parse(JSON.stringify(custom))),custom);
for(const bad of [NaN,Infinity,0,33,'2']){const s=structuredClone(fresh);s.lights[0].width=bad;assert.throws(()=>validateFixtureSizes(s));}
assert.throws(()=>validateFixtureSizes({...fresh,lights:[...fresh.lights.slice(1),fresh.lights[1]]}));
console.log('Fixture sizes: new doubled rectangular dimensions, unchanged legacy, anisotropic sizing, kicker physical snoot toggle/length and validation passed');

const under=tableLightRoles()[2],gridState=defaultFixtureSizes(true);
const baselineGrid=tableLightModifierFaces(under,99);
assert.equal(baselineGrid.length,240);
const oldProfile=structuredClone(gridState);for(const key of ['grid','gridCellsX','gridCellsY','gridDepth','gridThickness'])delete oldProfile.lights[9][key];
assert.deepEqual(tableLightModifierFaces(applyFixtureSizes([under],oldProfile)[0],99),baselineGrid,'Old profiles retain exact grid geometry');
gridState.lights[9].gridCellsX=3;gridState.lights[9].gridCellsY=4;gridState.lights[9].gridDepth=.4;gridState.lights[9].gridThickness=.02;
const grid=applyFixtureSizes([under],gridState)[0],faces=tableLightModifierFaces(grid,99);
assert.equal(faces.length,(3+4+2+4)*12);
const along=faces.flatMap(f=>f.slice(0,3).map(v=>v.reduce((sum,x,i)=>sum+(x-grid.pos[i])*grid.dir[i],0)));
assert.ok(Math.abs(Math.max(...along)-.402)<1e-12);assert.ok(Math.abs(Math.min(...along)+.002)<1e-12);
assert.ok(faces.every(f=>f[3]===99));
gridState.lights[9].grid=false;assert.equal(tableLightModifierFaces(applyFixtureSizes([under],gridState)[0],99).length,0);
for(const [key,bad] of [['gridCellsX',0],['gridCellsY',1.5],['gridCellsX',33],['gridDepth',0],['gridThickness',.3]]){const s=defaultFixtureSizes();s.lights[9][key]=bad;assert.throws(()=>validateFixtureSizes(s));}
const blocked=defaultFixtureSizes();blocked.lights[9].width=.1;blocked.lights[9].gridCellsX=32;assert.throws(()=>validateFixtureSizes(blocked),/cell pitch/);
console.log('Underlight grid: legacy settings with sealed mesh, cell count/pitch, physical depth, materials, removal and nonzero openings passed');
