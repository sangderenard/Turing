import assert from 'node:assert/strict';
import {studioRoomBounds} from '../examples/meowdoku/web/marble_room.js';
import {defaultStudioTable,tableFloorDrop} from '../examples/meowdoku/web/marble_studio_table.js';
import {stageProgrammerFaces,stageProgrammerLayout,pickStageProgrammer,PROGRAMMER_KEYS,PROGRAMMER_FADERS} from '../examples/meowdoku/web/marble_stage_programmer.js';
import {createStudioSceneGeometry} from '../examples/meowdoku/web/studio_scene_builder.js';

const table=defaultStudioTable(),bounds=studioRoomBounds(3,tableFloorDrop(table));
const ids={metal_steel:1,metal_aluminum:2,black_matte_paint:3},parts=[];
const faces=stageProgrammerFaces(bounds,table,ids,p=>parts.push(p));
assert.equal(parts.reduce((n,p)=>n+p.faces.length,0),faces.length);
assert.equal(parts.length,1+4+1+PROGRAMMER_KEYS.length+PROGRAMMER_FADERS.length);
assert(parts.every(p=>p.closed&&p.faces.length===12));
const lighting={controls:{'room-scale':3,'stage-programmer-enabled':true},table};
const geometry=createStudioSceneGeometry({lighting,materials:{ids,fixtureMaterialBase:100},subject:{unitM:.001}});
assert.equal(geometry.programmerFaces().length,faces.length);
lighting.controls['stage-programmer-enabled']=false;
assert.equal(createStudioSceneGeometry({lighting,materials:{ids,fixtureMaterialBase:100},subject:{unitM:.001}}).programmerFaces().length,0);

const layout=stageProgrammerLayout(bounds,table),eyeY=layout.buttonY+2;
const frame={view:[1,0,0,0, 0,0,1,0, 0,-1,0,0, -layout.x,layout.z,-eyeY,1],fovDegrees:60,aspect:1};
const point=(x,z)=>{
  const depth=eyeY-layout.buttonY,extent=Math.tan(Math.PI/6);
  return [(.5+(x-layout.x)/(2*depth*extent)),(.5+(z-layout.z)/(2*depth*extent))];
};
for(let i=0;i<PROGRAMMER_KEYS.length;i++){
  const [u,v]=point(layout.x-.51+i*.34,layout.z+.16);
  assert.equal(pickStageProgrammer(frame,bounds,table,u,v),PROGRAMMER_KEYS[i]);
}
for(let i=0;i<PROGRAMMER_FADERS.length;i++){
  const [u,v]=point(layout.x-.55+i*.275,layout.z-.13);
  const hit=pickStageProgrammer(frame,bounds,table,u,v);
  assert.equal(hit.bus,PROGRAMMER_FADERS[i]);assert(Math.abs(hit.level-.5)<.03);
}
assert.equal(pickStageProgrammer(frame,bounds,table,0,0),null);
console.log('PASS stage programmer geometry, toggle, four keys, and five bus faders');
