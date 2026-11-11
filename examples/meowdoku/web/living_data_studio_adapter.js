// Explicit composition, not a capture or appearance-faithful replay of the map.
// Native Living Data triangles/materials + caller-supplied verified Studio rig.
import {hydrateStudioSceneAsset} from './marble_studio_scene_asset.js';
import {createStandaloneStudio} from './standalone_studio.js';
import {validateStudioLightTransform} from './marble_studio_subject_renderer.js';
const I=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const assert=(ok,message)=>{if(!ok)throw Error(message);};
const hash=async text=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(text))),b=>b.toString(16).padStart(2,'0')).join('');
export async function composeLivingDataStudio({mesh,studioSceneAsset,lightTransform=I}){
 mesh=structuredClone(mesh);
 lightTransform=Array.from(lightTransform);validateStudioLightTransform(lightTransform);
 assert(mesh?.schema==='living-data-realized-mesh.v1'&&mesh.coordinate_space==='world'&&mesh.up_axis==='Y','Native world/Y mesh schema required');
 assert(typeof mesh.renderer_identity==='string'&&mesh.renderer_identity.endsWith('/pluck-phong'),'Native renderer identity required');
 const rig=await hydrateStudioSceneAsset(studioSceneAsset);
 assert(mesh?.stride===9&&Array.isArray(mesh.vertices)&&mesh.vertices.length>0&&mesh.vertices.length%27===0&&mesh.vertices.length<=200000*27&&mesh.vertices.every(Number.isFinite),'Bounded native interleaved triangle mesh required');
 assert(typeof mesh.world_identity==='string'&&/^[a-f0-9]{64}$/.test(mesh.source_sha256),'Native source identity required');
 assert(Array.isArray(mesh.identity_spans)&&Array.isArray(mesh.semantic_part_spans)&&Array.isArray(mesh.overlays),'Native ownership and overlay provenance required');
 const objects=mesh.world?.objects;
 assert(Array.isArray(objects)&&objects.length>0&&objects.every(o=>typeof o.identity==='string'&&o.identity),'Native source object registry required');
 const owners=new Set(objects.map(o=>o.identity));assert(owners.size===objects.length,'Duplicate native object identity');
 const ownedRanges=new Map();
 for(const span of mesh.identity_spans){
  assert(owners.has(span.identity)&&!ownedRanges.has(span.identity),'Native span owner is missing or duplicate');
  ownedRanges.set(span.identity,span);
 }
 for(const span of mesh.semantic_part_spans){
  const owner=ownedRanges.get(span.objectIdentity);
  assert(owner&&typeof span.identity==='string'&&span.identity&&Number.isSafeInteger(span.firstVertex)&&Number.isSafeInteger(span.vertexCount)&&span.vertexCount>0&&span.firstVertex%3===0&&span.vertexCount%3===0&&span.firstVertex>=owner.firstVertex&&span.firstVertex+span.vertexCount<=owner.firstVertex+owner.vertexCount,'Invalid semantic owner/range');
 }
 const overlayKinds=new Set();
 for(const span of mesh.overlays){assert(['navigation-routes','portal-tubes','portal-splats'].includes(span.kind)&&!overlayKinds.has(span.kind),'Unsupported or duplicate overlay provenance');overlayKinds.add(span.kind);}
 let covered=0;
 const ranges=[...mesh.identity_spans,...mesh.overlays].filter(span=>span.vertexCount!==0).sort((a,b)=>a.firstVertex-b.firstVertex);
 for(const span of ranges){
  assert(Number.isSafeInteger(span.firstVertex)&&Number.isSafeInteger(span.vertexCount)&&span.firstVertex===covered&&span.vertexCount>0&&span.firstVertex%3===0&&span.vertexCount%3===0,'Native mesh ownership has gaps or overlapping/non-triangle ranges');
  covered+=span.vertexCount;
 }
 assert(covered===mesh.vertices.length/9,'Native ownership does not account for all triangles');
 const colors=mesh.material_colors;
 assert(Array.isArray(colors)&&colors.length>0&&colors.length<=16&&colors.every(c=>Array.isArray(c)&&c.length===3&&c.every(Number.isFinite)),'Native material palette required');
 const materials={};
 for(const [name,stride] of [['pbr',16],['phong',8],['enamel',8],['texstack',16]]){
  const values=mesh.materials?.[name];assert(Array.isArray(values)&&values.length===colors.length*stride&&values.every(Number.isFinite),'Native material table mismatch: '+name);
  materials[name]={layout:{floats_per_record:stride},data:values.map(Math.fround)};
 }
 const nearest=color=>{let index=0,distance=Infinity;colors.forEach((c,i)=>{
  const d=c.reduce((sum,x,k)=>sum+(Math.fround(x)-color[k])**2,0);if(d<distance){distance=d;index=i;}});return index;};
 const faces=[];
 for(let offset=0;offset<mesh.vertices.length;offset+=27){
  const points=[],normals=[],ids=[];
  for(let k=0;k<3;k++){const at=offset+k*9;points.push(mesh.vertices.slice(at,at+3));normals.push(mesh.vertices.slice(at+3,at+6));ids.push(nearest(mesh.vertices.slice(at+6,at+9)));}
  assert(ids.every(id=>id===ids[0]),'Per-vertex material transition is unsupported');
  faces.push([...points,ids[0],normals,-1]);
 }
 const render=structuredClone(rig.packet.render);render.materials=materials;
 for(const uniform of render.uniforms)if(uniform.name==='uMaterialCount')uniform.value=colors.length;
 const provenance={schema:'turing.living-data-studio-composition.v1',world_identity:mesh.world_identity,
  source_sha256:mesh.source_sha256,mesh_sha256:await hash(JSON.stringify(mesh)),rig_identity:rig.identity,
  native_renderer_identity:mesh.renderer_identity,lighting:'explicit-captured-Studio-rig-not-native-celestial',light_transform:[...lightTransform],
  sampler_difference:'Studio flat color/depth feeds replace native map identity feeds; appearance equivalence is not claimed',
  identity_spans:mesh.identity_spans,semantic_part_spans:mesh.semantic_part_spans,overlays:mesh.overlays};
 const packet={schema:'turing.living-data-studio-composition.v1',provenance,revision:0,render,
  scene:{channel:'studio',ray:{pbr:materials.pbr.data},subject:{id:mesh.world_identity,faces,model:I,
   phong:{interiorCount:faces.length,glassCount:0,bubbles:[]}}}};
 return {packet,samplers:rig.samplers,provenance,lightTransform};
}
export async function createLivingDataStudio({canvas,mesh,studioSceneAsset,lightTransform=I}){
 const composition=await composeLivingDataStudio({mesh,studioSceneAsset,lightTransform});
 const renderer=createStandaloneStudio({canvas,packet:composition.packet,samplers:composition.samplers,lightTransform:composition.lightTransform});
 return Object.freeze({render(camera){return {...renderer.render(camera),composition:structuredClone(composition.provenance)};},dispose:renderer.dispose});
}
