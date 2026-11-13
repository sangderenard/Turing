import assert from 'node:assert/strict';
import {writeFileSync} from 'node:fs';
import {backdropFaces,defaultBackdropLayers,tallMountedFabricLayer,validateBackdropLayers} from '../examples/meowdoku/web/marble_backdrop.js';
import {fabricWallMesh,fabricCycloramaPose,fabricLayerRowConfiguration} from '../examples/meowdoku/web/marble_backdrop_fabric.js';

const ids={fabric_muslin_white:43,white_plaster:1};
const layers=defaultBackdropLayers();
layers[0]=tallMountedFabricLayer(layers[0]);
const valid=validateBackdropLayers(layers,Object.keys(ids));
assert.equal(valid[0].fabricGridSide,16);
assert.equal(valid[0].fabricAttachmentPoints.length,16);
assert.equal(valid[0].height,18);
const table={enabled:true,width:8,length:4};
const base=backdropFaces({count:1,material:'fabric_muslin_white',sweep:false,
  layers:valid.map((layer,i)=>i?layer:{...layer,surfaceBehavior:'cardstock'}),
  wallZ:-18.6,floorY:-1.02,table},ids);
assert.equal(base.length,2);
const mesh=fabricWallMesh(base,{side:16});
assert.equal(mesh.positions.length,256);
assert.equal(mesh.faces.length,450);
assert.ok(mesh.positions[0][1]>1&&mesh.positions[240][1]>19);
assert.ok(mesh.positions.every(point=>Math.abs(point[2]+5)<.1));
const packed=fabricLayerRowConfiguration(mesh,valid[0],{identity:20261112});
assert.deepEqual(packed.shape,[8,256,38]);
const at=(node,field)=>node*38+field;
for(let node=0;node<256;node++)assert.equal(packed.data[at(node,7)],node>=240?1:0);
assert.equal(packed.data[at(255,20)],0);
const faces=backdropFaces({count:1,material:'fabric_muslin_white',sweep:false,
  layers:valid,wallZ:-18.6,floorY:-1.02,table},ids);
assert.deepEqual(faces,mesh.faces);
const swept=fabricCycloramaPose(mesh,{floorY:-1.02});
assert.ok(swept.positions[0][2]>mesh.positions[0][2]+1.4);
assert.equal(swept.positions[0][1],-1.02);
assert.deepEqual(swept.positions.slice(240),mesh.positions.slice(240));
assert.deepEqual(backdropFaces({count:1,material:'fabric_muslin_white',sweep:true,
  layers:valid,wallZ:-18.6,floorY:-1.02,table},ids),swept.faces);
if(process.env.FABRIC_HOST_CONFIG_PATH){
  const bytes=Buffer.alloc(packed.data.length*8);
  packed.data.forEach((value,i)=>bytes.writeDoubleLE(value,i*8));
  writeFileSync(process.env.FABRIC_HOST_CONFIG_PATH,bytes);
}
console.log('Tall mounted Studio fabric: 16×16, 450 triangles, 16 top pins and native row config PASS');
