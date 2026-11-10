// The original game page loads this module only in its optional Studio host.
// Direct same-origin calls share immutable asset values; the original worker
// and existing application loop remain the only producers of body state.
import {installStudioCollection} from './mechanical_creature_studio_collection.js';
import {installStudioRoom} from './mechanical_creature_studio_room.js';

export function createMechanicalCreatureStudioLayerHost({host=globalThis,doc=document}={}) {
  const bridge=host.mechanicalCreatureStudioBridge;
  if(!bridge?.prepareEnvironment||!bridge?.queueEnvironment||!bridge?.setSink)
    throw Error('The original game runtime bridge is not installed.');
  let collection=null,room=null,busy=false,closed=false,lastSequence=0,lastReceipt=null;
  const epoch='studio-game:'+crypto.randomUUID();
  const panel=doc.createElement('details');panel.open=true;panel.dataset.studioLayerAmmo='true';
  panel.style.cssText='position:fixed;left:150px;top:8px;z-index:110;max-width:min(320px,80vw);background:#142a30eb;color:white;padding:8px;border-radius:7px;font:13px system-ui';
  const title=doc.createElement('summary');title.textContent='Studio collection';
  const status=doc.createElement('p');status.id='studio-layer-host-status';status.textContent='Waiting for the current Studio scene.';
  const choices=doc.createElement('div');choices.style.cssText='display:flex;gap:5px;flex-wrap:wrap';
  panel.append(title,status,choices);doc.body.append(panel);
  for(const name of ['pointerdown','pointerup','click','keydown'])panel.addEventListener(name,e=>e.stopPropagation());
  function refresh(){
    choices.replaceChildren();
    for(const item of host.mechanicalCreatureAmmo.items()){
      const button=doc.createElement('button');button.type='button';button.style.minHeight='44px';
      button.setAttribute('aria-label','Use '+item.name);button.setAttribute('aria-pressed',String(host.mechanicalCreatureAmmo.selected()?.identity===item.identity));
      if(item.properties.appearance_icon){const image=doc.createElement('img');image.src=item.properties.appearance_icon;image.width=32;image.height=32;image.alt='';button.append(image);}
      button.append(doc.createTextNode(item.name+' x'+item.quantity));
      button.addEventListener('click',()=>{host.mechanicalCreatureAmmo.select(item.identity);refresh();});choices.append(button);
    }
  }
  async function update({asset,source,marbleAsset=null,sequence}) {
    if(closed||busy)throw Error('A Studio environment update is already in progress or closed.');
    if(!Number.isSafeInteger(sequence)||sequence<=lastSequence||source?.revision!==asset?.revision)
      throw Error('A newer matching Studio environment sequence is required.');
    busy=true;status.textContent='Preparing current Studio assets.';
    try{
      const prepared=bridge.prepareEnvironment(source,{assetIdentity:asset.identity});
      const beforeCommit=({isCurrent,commit})=>bridge.queueEnvironment({assetIdentity:asset.identity,revision:source.revision,
        sequence,definition:prepared.definition,isCurrent,commit});
      if(!collection){
        if(!marbleAsset)throw Error('Initial Studio import needs its actual marble asset.');
        const candidate=await installStudioCollection({host,assets:[{asset:marbleAsset,quantity:12}],onStatus:value=>{
          if(value.state==='ready'||value.state==='partial')refresh();
          if(value.state==='failed')status.textContent=value.assets.map(x=>x.error||x.name).join('; ');
        }});
        // dispose() cannot reach a handle that its installer has not returned.
        // A late handle remains local and is closed before any further work.
        if(closed){candidate.dispose();throw Error('The Studio layer closed while preparing its collection.');}
        collection=candidate;
      }
      let acknowledged;
      if(!room){
        const candidate=await installStudioRoom({host,asset,building:prepared.building,sequence,sourceRevision:source.revision,beforeCommit,
          onStatus:value=>{status.textContent='Studio scene: '+value.state+(value.error?' '+value.error:'');}});
        if(closed){candidate.dispose();throw Error('The Studio layer closed while preparing its room.');}
        room=candidate;
        acknowledged=await room.ready;
      }else acknowledged=await room.update({asset,building:prepared.building,sequence,sourceRevision:source.revision,beforeCommit});
      // Preserve a ready/update rejection, but never admit a success after Stop.
      if(closed)throw Error('The Studio layer closed before its update was acknowledged.');
      lastSequence=sequence;
      lastReceipt={...acknowledged,building:prepared.building,sequence,sourceRevision:source.revision,
        colliderPublication:'posted-to-original-worker',physics:'original-game-worker',camera:'original-game-controls'};
      status.textContent='Studio revision '+source.revision+' displayed. Use original game controls and ammunition.';
      return lastReceipt;
    }catch(error){status.textContent='Studio update unavailable: '+String(error.message||error);throw error;}
    finally{busy=false;}
  }
  return {update,connect(sink){bridge.setSink(sink,{epoch});return ()=>bridge.setSink(null);},readSnapshot:()=>bridge.readSnapshot(),
    read:()=>({busy,closed,lastSequence,lastReceipt,collection:collection?.readStatus(),room:room?.readStatus()}),
    dispose(){closed=true;bridge.setSink(null);room?.dispose();collection?.dispose();panel.remove();}};
}

if(typeof window!=='undefined'&&window.mechanicalCreatureStudioBridge)
  window.mechanicalCreatureStudioLayer=createMechanicalCreatureStudioLayerHost();
