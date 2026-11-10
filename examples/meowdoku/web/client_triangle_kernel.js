import {ClientSphereQuery,verifiedText} from './client_ray_kernel.js';
export const MAX_CLIENT_TRIANGLES=4096;
export function validateTriangleScene(scene) {
 if(!scene||typeof scene.revision!=='string'||!scene.revision||scene.revision.length>256||!Array.isArray(scene.triangles)||!scene.triangles.length||scene.triangles.length>MAX_CLIENT_TRIANGLES)throw Error('Expected versioned scene with 1-4096 explicit triangles.');
 if(scene.unsupportedGeometry?.length)throw Error('Unsupported geometry cannot be silently omitted.');
 const ids=new Set();
 for(const t of scene.triangles){
  if(!Number.isSafeInteger(t.owner)||t.owner<0||t.owner>16777215||ids.has(t.owner))throw Error('Invalid or duplicate triangle owner.');ids.add(t.owner);
  for(const key of ['v0','e1','e2'])if(!Array.isArray(t[key])||t[key].length!==3||!t[key].every(v=>Number.isFinite(v)&&Number.isFinite(Math.fround(v))))throw Error('Invalid triangle coordinates.');
 }
 return scene;
}
export class ClientTriangleQuery extends ClientSphereQuery {
 static async create(base=new URL('./',import.meta.url),canvas=null){
  const r=await fetch(new URL('client_ray_triangle.manifest.json',base),{cache:'no-store'});if(!r.ok)throw Error('Triangle manifest '+r.status);
  const m=await r.json();if(m.schema!=='turing.client-triangle-query.v1'||m.maximum_batch!==4096||m.maximum_triangles!==4096)throw Error('Unsupported triangle ABI.');
  const texts=await Promise.all(['vert','frag'].map(k=>verifiedText(new URL(m.artifacts[k].file,base),m.artifacts[k].sha256)));
  const q=new ClientTriangleQuery(canvas||new OffscreenCanvas(1,1),texts,m);q.geometrySampler='uTriangles';q.geometryCount='uTriangleCount';q.geometryStrideBytes=48;return q;
 }
 setScene(scene){
  if(this.closed||this.busy)throw Error('Query backend closed or busy.');validateTriangleScene(scene);
  const count=scene.triangles.length,gl=this.gl;if(count>gl.getParameter(gl.MAX_TEXTURE_SIZE))throw Error('Triangle scene exceeds device texture capacity.');
  const data=new Float32Array(count*12);
  scene.triangles.forEach((t,i)=>{data.set([...t.v0,t.owner],i*4);data.set(t.e1,count*4+i*4);data.set(t.e2,count*8+i*4);});
  gl.bindTexture(gl.TEXTURE_2D,this.textures[2]);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA32F,count,3,0,gl.RGBA,gl.FLOAT,data);
  if(gl.getError()!==gl.NO_ERROR)throw Error('Triangle upload rejected.');this.scene={revision:scene.revision,count};return {revision:scene.revision,triangles:count,geometryBytes:data.byteLength};
 }
 async query(rays,revision){const {normals,...result}=await super.query(rays,revision);return {...result,barycentrics:normals};}
}
