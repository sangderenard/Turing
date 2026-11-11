import * as sceneBuilder from '../examples/meowdoku/web/studio_scene_builder.js';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {lightCeilingSupportLayout,lightCeilingSupportFaces} from '../examples/meowdoku/web/marble_light_ceiling_support.js';
import {LIGHT_STAND_DIMENSIONS} from '../examples/meowdoku/web/marble_light_stands.js';
import {defaultFixtureSizes,applyFixtureSizes} from '../examples/meowdoku/web/marble_fixture_sizes.js';
import {defaultLightTools,applyLightTools} from '../examples/meowdoku/web/marble_light_tools.js';
import {tableLightModifierFaces} from '../examples/meowdoku/web/marble_table_lights.js';
import {STUDIO_LIGHTS} from '../examples/meowdoku/web/marble_light_strengths.js';
import {studioRoomBounds} from '../examples/meowdoku/web/marble_room.js';
import {defaultStudioScale} from '../examples/meowdoku/web/marble_studio_scale.js';

// The current production builder supplies every housing, emitter and tool point.
// No test-only approximate softbox stands in for its actual shared geometry.
const source=fs.readFileSync(new URL('../examples/meowdoku/web/marble_studio.js',import.meta.url),'utf8');
const primitives=vm.createContext({STUDIO_LIGHTS,tableLightModifierFaces,ids:{polished_silver:39,polished_gold:40}});
Object.assign(primitives,sceneBuilder,{buildFixtureFaces:(roles,housing,onPart)=>sceneBuilder.buildStudioFixtureFaces(roles,housing,onPart,primitives.ids??{})});
const builderStart=source.indexOf('function buildFixtureFaces(');
Object.assign(primitives,sceneBuilder,{buildFixtureFaces:(roles,housing,onPart)=>sceneBuilder.buildStudioFixtureFaces(roles,housing,onPart,primitives.ids??{})});

const {resolveLightingPoses}=await import('../examples/meowdoku/web/marble_lighting.js');
const state={version:1,poses:[{id:'fill-upper',pos:[1,12,-2],yaw:0,pitch:-90,aimAtSubject:false}]};
const role=()=>applyLightTools(applyFixtureSizes([{...resolveLightingPoses(state,['fill-upper'])[0],halfW:.3,halfH:.3,round:false,roleIndex:6,mat:8}],defaultFixtureSizes()),defaultLightTools(defaultFixtureSizes()),defaultFixtureSizes())[0];
const mesh=()=>structuredClone(primitives.buildFixtureFaces([role()],99));
const down=mesh();state.poses[0].yaw=65;state.poses[0].pitch=-35;const tilted=mesh(),dir=role().dir;
assert.notDeepEqual(down.filter(f=>f[3]===99),tilted.filter(f=>f[3]===99),'Gimbal turns real housing/grid');
const emitters=tilted.filter(f=>f[3]===8);assert.equal(emitters.length,2);
for(const face of emitters){const[a,b,c]=face,u=b.map((v,i)=>v-a[i]),v=c.map((v,i)=>v-a[i]);const n=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]];assert.ok(n.reduce((s,x,i)=>s+x*dir[i],0)>0,'Ray emitter winding follows aim');assert.ok(face[4][0].reduce((s,x,i)=>s+x*dir[i],0)>.999,'Phong emitter normal follows same aim');}
state.poses[0].pos[1]-=2;const lowered=mesh();
for(let i=0;i<tilted.length;i++)for(let j=0;j<3;j++)for(let k=0;k<3;k++)assert.ok(Math.abs(lowered[i][j][k]-tilted[i][j][k]-(k===1?-2:0))<1e-10,'Drop translates every fixture vertex');
console.log('PASS overhead mesh: real housing/grid rotates, Phong normals and ray winding follow gimbal, drop translates every vertex');
