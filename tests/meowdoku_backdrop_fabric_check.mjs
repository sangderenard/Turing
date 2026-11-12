import assert from 'node:assert/strict';
import {backdropFaces,defaultBackdropLayers,validateBackdropLayers} from '../examples/meowdoku/web/marble_backdrop.js';
import {fabricWallMesh,fabricRowConfiguration} from '../examples/meowdoku/web/marble_backdrop_fabric.js';

const ids={fabric_muslin_white:43,white_plaster:1};
const layers=defaultBackdropLayers();
layers[0]={...layers[0],material:'fabric_muslin_white',surfaceBehavior:'fabric',
  fabricFrozen:true,lockToTableBack:true,heightOffGround:0,pitch:10};
const valid=validateBackdropLayers(layers,Object.keys(ids));
assert.equal(valid[0].fabricFrozen,true);
const parts=[];
const faces=backdropFaces({count:1,material:'white_plaster',sweep:false,layers:valid,
  wallZ:-6,floorY:-4,table:{enabled:true,length:4,width:8}},ids,part=>parts.push(part));
assert.equal(faces.length,128);
assert.equal(parts.length,1);
assert.equal(parts[0].faces.length,128);
assert.ok(faces.every(face=>face[3]===43));
const wall=parts[0].faces;
const sourceCorners=[wall[0][0],wall[0][1],wall.at(-1)[1],wall.at(-1)[2]];
assert.ok(sourceCorners.every(point=>point.length===3&&point.every(Number.isFinite)));
const base=backdropFaces({count:1,material:'white_plaster',sweep:false,
  layers:valid.map((layer,i)=>i?layer:{...layer,surfaceBehavior:'cardstock'}),
  wallZ:-6,floorY:-4,table:{enabled:true,length:4,width:8}},ids);
assert.equal(base.length,2);
const mesh=fabricWallMesh(base);
assert.equal(mesh.positions.length,81);
assert.equal(mesh.faces.length,128);
assert.deepEqual(mesh.faces,faces);
const displaced=mesh.positions.map(point=>point.slice());
displaced[40][2]+=0.25;
const changed=backdropFaces({count:1,material:'white_plaster',sweep:false,layers:valid,
  wallZ:-6,floorY:-4,table:{enabled:true,length:4,width:8},fabricPositions:[displaced]},ids);
assert.ok(changed.some(face=>face.slice(0,3).some(point=>point[2]===displaced[40][2])));
assert.ok(changed.every(face=>face[3]===43));
const row=fabricRowConfiguration(mesh,{network:2,identity:42,frozen:true});
assert.deepEqual(row.shape,[8,256,38]);
assert.deepEqual([...row.control],[1,1]);
const at=(node,field)=>(2*256+node)*38+field;
assert.equal(row.data[at(0,37)],42);
assert.equal(row.data[at(72,7)],1);
assert.equal(row.data[at(80,7)],1);
assert.equal(row.data[at(40,6)],1);
assert.equal(row.data[at(40,17)],39);
assert.equal(row.data[at(40,18)],41);
assert.equal(row.data[at(40,19)],31);
assert.equal(row.data[at(40,20)],49);
assert.equal(row.data[at(81,6)],0);
assert.throws(()=>validateBackdropLayers(valid.map((layer,i)=>i?layer:{...layer,shape:'ellipse'}),Object.keys(ids)));
const noRigidSweep=backdropFaces({count:1,material:'white_plaster',sweep:true,layers:valid,
  wallZ:-6,floorY:-4,table:{enabled:true,length:4,width:8}},ids);
assert.equal(noRigidSweep.length,128);
console.log('Backdrop fabric: source wall triangulated, material preserved, 81-node Nodus row and freeze contract PASS');
