// Chosen Studio appearances on the original game's inventory and sphere bodies.
// No movement loop, collider, controller, worker or state tensor is replaced.
import {hydrateStudioGameAsset} from './marble_game_asset.js';
import {adaptMechanicalCreatureProjectiles} from './mechanical_creature_instances.js';
import {mechanicalCreatureStudioCamera} from './mechanical_creature_studio_camera.js';
import {createStudioSubjectRenderer,validateStudioLightTransform} from './marble_studio_subject_renderer.js';
import {renderStudioSubjectThumbnail} from './marble_studio_subject_thumbnail.js';

const multiply=(a,b)=>{const m=new Float64Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++)
  for(let k=0;k<4;k++)m[c*4+r]+=a[k*4+r]*b[c*4+k];return m;};
// Invert a validated uniform, positive affine pose without moving any vertex.
export function studioRigToGame(sourceModel,targetModel) {
  validateStudioLightTransform(sourceModel);validateStudioLightTransform(targetModel);
  const scale2=sourceModel[0]**2+sourceModel[1]**2+sourceModel[2]**2,inv=new Float64Array(16);inv[15]=1;
  for(let c=0;c<3;c++)for(let r=0;r<3;r++)inv[c*4+r]=sourceModel[r*4+c]/scale2;
  for(let r=0;r<3;r++)inv[12+r]=-[0,1,2].reduce((v,k)=>v+inv[k*4+r]*sourceModel[12+k],0);
  return multiply(targetModel,inv);
}

/** Install once on an appearance-hook-enabled original game. Assets are fully
 * validated before upload. Inventory admission happens in its existing render
 * callback, after the exact appearance and real rendered icon are available.
 * Selection is always the user's original ammo picker action. */
export async function installStudioCollection({assets,host=globalThis,onStatus=()=>{}}={}) {
  if(!Array.isArray(assets)||assets.length>8)throw Error('Choose up to eight Studio marble assets.');
  if(host.mechanicalCreatureImportedAppearance)throw Error('The game already has an imported appearance adapter.');
  if(!host.mechanicalCreatureAmmo)throw Error('The original game ammunition hook is unavailable.');
  const entries=[],ids=new Set();
  for(const row of assets) {
    const hydrated=await hydrateStudioGameAsset(row.asset??row);
    if(ids.has(hydrated.identity))throw Error('The chosen set contains a duplicate asset.');ids.add(hydrated.identity);
    const quantity=row.quantity??1;
    if(!Number.isInteger(quantity)||quantity<1||quantity>128)throw Error('Chosen inventory quantity must be 1 to128.');
    entries.push({hydrated,quantity,renderer:null,item:null,error:null});
  }
  let closed=false,admitting=false,frame=[],statusKey='',lastContext=null,lastDraw=null,drawCount=0;
  function report(state,extra={}) {
    const value={state,assets:entries.map(e=>({identity:e.hydrated.identity,name:e.hydrated.name,
      ready:!!e.item,error:e.error,triangles:e.hydrated.packet.scene.subject.faces.length})),...extra};
    const key=JSON.stringify(value);if(key!==statusKey){statusKey=key;
      try{onStatus(value);}catch(error){console.warn('Studio collection status observer failed',error);}}
  }
  const adapter={
    identities(context) {
      if(closed||admitting)return new Set();lastContext=context;
      admitting=true;
      try {
        for(const entry of entries)if(!entry.renderer&&!entry.error) {
          const {packet,samplers,name,identity}=entry.hydrated;
          try {
            entry.renderer=createStudioSubjectRenderer({gl:context.gl,packet,samplers});
            const icon=renderStudioSubjectThumbnail({gl:context.gl,packet,samplers});
            entry.item=host.mechanicalCreatureAmmo.addAsset({assetIdentity:identity,name,iconUrl:icon.dataUrl,quantity:entry.quantity});
          } catch(error) {entry.renderer?.dispose();entry.renderer=null;entry.error=String(error.message||error);}
        }
      } finally {admitting=false;}
      frame=[];
      for(const entry of entries)if(entry.renderer&&entry.item) {
        const result=adaptMechanicalCreatureProjectiles({...context,source:entry.hydrated.source,
          fixedQuaternionXYZW:[0,0,0,1],appearanceAssetIdentity:entry.hydrated.identity});
        for(const body of result.instances)frame.push({entry,body});
      }
      report(entries.every(e=>e.item)?'ready':entries.some(e=>e.item)?'partial':'failed',
        {importedBodies:frame.length,selectedAsset:host.mechanicalCreatureAmmo.selected()?.properties?.appearance_asset_identity??null,
          inventory:host.mechanicalCreatureAmmo.items().map(item=>({identity:item.identity,quantity:item.quantity}))});
      return new Set(frame.map(x=>x.body.identity));
    },
    draw(context) {
      if(closed)return false;
      const camera=mechanicalCreatureStudioCamera(context),draws=frame.filter(x=>context.identities.has(x.body.identity));
      // Existing sphere centers order separate transparent subjects. As with
      // Studio's triangle sort this is not an exact intersecting-glass solver.
      if(context.phase==='transparent')draws.sort((a,b)=>{
        const depth=x=>x.body.positionM.reduce((v,p,k)=>v+(p-camera.eye[k])*camera.forward[k],0);
        return depth(b)-depth(a);
      });
      for(const {entry,body}of draws)entry.renderer.draw({...camera,model:body.model,
        lightTransform:studioRigToGame(entry.hydrated.packet.scene.subject.model,body.model),phase:context.phase});
      if(draws.length){drawCount++;lastDraw={phase:context.phase,identities:draws.map(x=>x.body.identity)};}
      return true;
    },
  };
  // Another chosen-set install may have completed during async verification.
  if(host.mechanicalCreatureImportedAppearance)throw Error('The game already has an imported appearance adapter.');
  host.mechanicalCreatureImportedAppearance=adapter;report('waiting-for-original-render');
  return {
    readStatus:()=>({closed,ready:entries.filter(e=>e.item).length,bodyCount:frame.length,
      bodies:frame.map(({entry,body})=>({identity:body.identity,assetIdentity:entry.hydrated.identity,
        positionM:body.positionM.slice(),radiusM:body.radiusM,sleeping:body.sleeping,pickup:body.pickup})),
      snapshotSequence:lastContext?.stateLoopRuntime.latestSequence??null,drawCount,lastDraw,
      camera:lastContext?{position:lastContext.cameraPosition,facing:lastContext.cameraFacing,width:lastContext.width,height:lastContext.height}:null}),
    dispose(){if(closed)return;closed=true;if(host.mechanicalCreatureImportedAppearance===adapter)host.mechanicalCreatureImportedAppearance=null;
      for(const entry of entries)entry.renderer?.dispose();frame=[];report('closed');},
  };
}
