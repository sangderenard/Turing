// Export the actual Studio backdrop wall vertices for the native Nodus demo.
import {readFileSync,writeFileSync} from 'node:fs';
import {backdropFaces,defaultBackdropLayers} from '../examples/meowdoku/web/marble_backdrop.js';
import {fabricWallMesh,fabricTableDrapePose,fabricTableContact,fabricLayerRowConfiguration} from '../examples/meowdoku/web/marble_backdrop_fabric.js';
import {defaultStudioTable} from '../examples/meowdoku/web/marble_studio_table.js';

const [assetsPath,outputPath,mode]=process.argv.slice(2);
if(!assetsPath||!outputPath||mode&&!['table-drape'].includes(mode))
  throw Error('usage: node tools/export_backdrop_fabric.mjs assets.json output.csv [table-drape]');
const assets=JSON.parse(readFileSync(assetsPath,'utf8'));
const ids=assets.material_index;
if(!Number.isInteger(ids.fabric_muslin_white))throw Error('Export Studio assets with fabric materials first');
const layers=defaultBackdropLayers();
layers[0]={...layers[0],material:'fabric_muslin_white',lockToTableBack:true,
  heightOffGround:0,pitch:10,surfaceBehavior:'fabric',fabricPose:mode==='table-drape'?'table':'wall'};
const table=defaultStudioTable();
const source=backdropFaces({count:1,material:'fabric_muslin_white',sweep:false,
  layers:layers.map((layer,index)=>index?layer:{...layer,surfaceBehavior:'cardstock'}),
  wallZ:-6,floorY:-4,table},ids);
const wall=fabricWallMesh(source);
const mesh=mode==='table-drape'?fabricTableDrapePose(wall,table):wall;
const attachments=layers[0].fabricAttachmentPoints??(layers[0].fabricPose==='table'?[0,8]:[72,80]);
const packed=fabricLayerRowConfiguration(mesh,layers[0],{identity:20261112});
const cell=(node,field)=>packed.data[node*38+field];
writeFileSync(outputPath,'x,y,z,pinned\n'+mesh.positions.map((point,node)=>
  [...point,cell(node,7)].join(',')).join('\n')+'\n');
if(mode==='table-drape')writeFileSync(outputPath+'.contact.csv',
  'topY,minX,maxX,minZ,maxZ\n'+[...fabricTableContact(table)].slice(1).join(',')+'\n');
console.log(`Studio backdrop ${source.length} source triangles -> ${mesh.positions.length} fabric nodes / ${mesh.faces.length} triangles; material ${mesh.material}; attachments ${attachments.join(',')}; ${outputPath}`);
