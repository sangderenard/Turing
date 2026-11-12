import assert from 'node:assert/strict';
import {backdropFaces,defaultBackdropLayers,validateBackdropLayers} from '../examples/meowdoku/web/marble_backdrop.js';

const ids={stone_onyx:1};
const layers=defaultBackdropLayers();
const render=(layer,options={})=>{
  const parts=[];
  const faces=backdropFaces({count:1,material:'stone_onyx',sweep:true,
    layers:layers.map((value,index)=>index?value:{...value,...layer}),
    wallZ:-6,floorY:-4,table:{enabled:true,length:4,width:8},...options},ids,part=>parts.push(part));
  return {faces,parts};
};
const points=faces=>faces.flatMap(face=>face.slice(0,3));
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-6,`${a} != ${b}`);

const legacy=render({});
close(legacy.faces[0][0][2],-5.988);
close(legacy.faces[0][0][1],-4);
assert.equal(legacy.faces.length,4);

const table=render({lockToTableBack:true,heightOffGround:0,curveRadius:.5,pitch:10});
const sweep=table.parts.find(part=>part.id==='layer-0-sweep');
assert.ok(sweep.faces.length>20);
close(Math.min(...points(sweep.faces).map(point=>point[1])),-.988);
close(Math.min(...points(sweep.faces).map(point=>point[2])),-1.988);
assert.ok(Math.max(...points(sweep.faces).map(point=>point[2]))<0);
const wall=table.parts.find(part=>part.id==='layer-0-wall');
close(wall.faces[0][0][1],sweep.faces[1][2][1]);
close(wall.faces[0][0][2],sweep.faces[1][2][2]);
assert.ok(wall.faces[0][4][0][1]>0);

const free=render({lockToBackWall:false,stageDistance:2,heightOffGround:0,curveRadius:.4});
close(free.faces[0][0][2],-1.988);
const wallLocked=render({lockToBackWall:true,stageDistance:2,heightOffGround:0});
close(wallLocked.faces[0][0][2],-5.988);

const valid=validateBackdropLayers(layers.map((value,index)=>index?value:
  {...value,lockToTableBack:true,heightOffGround:0,curveRadius:.5,pitch:10,aspectLocked:true}),Object.keys(ids));
assert.equal(valid[0].curveRadius,.5);
assert.equal(valid[0].aspectLocked,true);
console.log('Backdrop placement: wall/stage/table locks, tangent sweep, saved fields PASS');
