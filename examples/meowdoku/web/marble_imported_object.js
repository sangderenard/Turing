// Supplied subjects retain the server asset as their immutable geometry source.
// No repair, decimation, invented volume or procedural fallback is performed.
const assets=new Map();
function immutable(value) {
  if(value&&typeof value==='object') {for(const child of Object.values(value))immutable(child);Object.freeze(value);}
  return value;
}
const digestPattern=/^[0-9a-f]{64}$/;
export function importedMechanics() {
  return {massKg:null,massExact:false,validForRoll:false,centerM:null,inertiaKgM2:null,
    components:[],principalMoments:[],mechanicsNote:'Imported mesh mechanics have not been certified'};
}
export function registerObjAsset(asset) {
  if(!digestPattern.test(asset?.digest))throw new Error('Invalid OBJ asset identity');
  const g=asset.geometry,v=g?.vertices,ids=g?.material_ids;
  if(!Array.isArray(v)||!v.length||v.length%9||v.length>4500000||!v.every(Number.isFinite)||
    !Array.isArray(ids)||ids.length!==v.length/9||!ids.every(x=>Number.isInteger(x)&&x>=0))
    throw new Error('Invalid supplied OBJ geometry');
  const saved=immutable(structuredClone(asset));
  assets.set(asset.digest,saved);return saved;
}
export async function ensureObjAsset(digest,{fetcher=globalThis.fetch}={}) {
  if(!digestPattern.test(digest))throw new Error('Invalid OBJ asset identity');
  if(assets.has(digest))return assets.get(digest);
  const response=await fetcher(`/api/obj_assets/${digest}`,{cache:'no-store'});
  if(!response.ok)throw new Error(`OBJ asset unavailable (${response.status})`);
  const asset=await response.json();
  if(asset.digest!==digest)throw new Error('OBJ asset identity mismatch');
  return registerObjAsset(asset);
}
export function importedObjectGeometry(design,shell) {
  if(design.material_mode!=='studio-shell')throw new Error('Choose the explicit Studio shell material override');
  if(!Number.isInteger(shell)||shell<0)throw new Error('Invalid subject material');
  const asset=assets.get(design.source_asset);
  if(!asset)throw new Error('Load the supplied OBJ asset before building its subject');
  const v=asset.geometry.vertices,glass=[];
  for(let i=0;i<v.length;i+=9) {
    const a=v.slice(i,i+3),b=v.slice(i+3,i+6),c=v.slice(i+6,i+9);
    const u=b.map((x,k)=>x-a[k]),w=c.map((x,k)=>x-a[k]);
    const n=[u[1]*w[2]-u[2]*w[1],u[2]*w[0]-u[0]*w[2],u[0]*w[1]-u[1]*w[0]],length=Math.hypot(...n);
    if(!(length>0))throw new Error('Supplied OBJ contains a degenerate triangle');
    const normal=n.map(x=>x/length);
    glass.push([a,b,c,shell,[normal.slice(),normal.slice(),normal.slice()]]);
  }
  return {interior:[],glass,triangleCount:glass.length,ribbons:[],tubes:[],confetti:[],bubbles:[],
    imported:true,sourceAsset:design.source_asset,mechanicsResolved:false,
    mechanicsNote:importedMechanics().mechanicsNote,topology:structuredClone(asset.topology||{})};
}
