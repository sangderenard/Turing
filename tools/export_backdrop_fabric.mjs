// Export the actual Studio backdrop wall vertices for the native Nodus demo.
import {readFileSync,writeFileSync} from 'node:fs';
import {backdropFaces,defaultBackdropLayers} from '../examples/meowdoku/web/marble_backdrop.js';
import {fabricWallMesh} from '../examples/meowdoku/web/marble_backdrop_fabric.js';

const [assetsPath,outputPath]=process.argv.slice(2);
if(!assetsPath||!outputPath)throw Error('usage: node tools/export_backdrop_fabric.mjs assets.json output.csv');
const assets=JSON.parse(readFileSync(assetsPath,'utf8'));
const ids=assets.material_index;
if(!Number.isInteger(ids.fabric_muslin_white))throw Error('Export Studio assets with fabric materials first');
const layers=defaultBackdropLayers();
layers[0]={...layers[0],material:'fabric_muslin_white',lockToTableBack:true,
  heightOffGround:0,pitch:10};
const source=backdropFaces({count:1,material:'fabric_muslin_white',sweep:false,layers,
  wallZ:-6,floorY:-4,table:{enabled:true,length:4,width:8}},ids);
const mesh=fabricWallMesh(source);
writeFileSync(outputPath,'x,y,z\n'+mesh.positions.map(point=>point.join(',')).join('\n')+'\n');
console.log(`Studio backdrop ${source.length} source triangles -> ${mesh.positions.length} fabric nodes / ${mesh.faces.length} triangles; material ${mesh.material}; ${outputPath}`);
