// The original game page loads this module only in its optional Studio host.
// Direct same-origin calls share immutable asset values; the original worker
// and existing application loop remain the only producers of body state.
import {installStudioCollection} from './mechanical_creature_studio_collection.js';
import {installStudioRoom} from './mechanical_creature_studio_room.js';
import {createStudioGameProtocol,validateStudioGameReceipt} from './marble_studio_game_protocol.js';

export function createMechanicalCreatureStudioLayerHost({host=globalThis,doc=document}={}) {
  const bridge=host.mechanicalCreatureStudioBridge;
  if(!['prepareEnvironment','queueEnvironment','setSink','readSnapshot'].every(key=>typeof bridge?.[key]==='function'))
    throw Error('The original game runtime bridge is not installed.');
  const player=host.mechanicalCreatureStudioEntry;
  if(!['enter','setActive','read','whenReady','readQualification'].every(key=>typeof player?.[key]==='function'))
    throw Error('The original player Studio entry hook is not installed.');
  let collection=null,room=null,busy=false,closed=false,lastSequence=0,lastReceipt=null,connection=null;
  let entry=null,active=true;
  player.setActive(false);
  const epoch='studio-game:'+crypto.randomUUID();
  const protocol=createStudioGameProtocol(epoch);
  const panel=doc.createElement('details');panel.open=true;panel.dataset.studioLayerAmmo='true';
  panel.style.cssText='position:fixed;left:150px;top:8px;z-index:110;max-width:min(320px,80vw);background:#142a30eb;color:white;padding:8px;border-radius:7px;font:13px system-ui';
  const title=doc.createElement('summary');title.textContent='Studio collection';
  const status=doc.createElement('p');status.id='studio-layer-host-status';status.textContent='Waiting for the current Studio scene.';
  const choices=doc.createElement('div');choices.style.cssText='display:flex;gap:5px;flex-wrap:wrap';
  const playerStatus=doc.createElement('p');playerStatus.id='studio-layer-player-status';
  playerStatus.textContent='Player entry waits for the current Studio environment.';
  panel.append(title,status,playerStatus,choices);doc.body.append(panel);
  for(const name of ['pointerdown','pointerup','click','keydown'])panel.addEventListener(name,e=>e.stopPropagation());
  function refresh(){
    if(closed)return;
    choices.replaceChildren();
    for(const item of host.mechanicalCreatureAmmo.items()){
      const button=doc.createElement('button');button.type='button';button.style.minHeight='44px';
      button.setAttribute('aria-label','Use '+item.name);button.setAttribute('aria-pressed',String(host.mechanicalCreatureAmmo.selected()?.identity===item.identity));
      if(item.properties.appearance_icon){const image=doc.createElement('img');image.src=item.properties.appearance_icon;image.width=32;image.height=32;image.alt='';button.append(image);}
      button.append(doc.createTextNode(item.name+' x'+item.quantity));
      button.addEventListener('click',()=>{host.mechanicalCreatureAmmo.select(item.identity);refresh();});choices.append(button);
    }
  }
  function refreshPlayer(){
    if(closed||!entry)return;
    const current=player.read(),position=current.positionM;
    playerStatus.dataset.entry=entry.location??'outside-studio-entrance';
    playerStatus.dataset.position=JSON.stringify(position);
    playerStatus.dataset.controls=String(current.active);
    playerStatus.textContent='Original player '+(position?position.map(value=>value.toFixed(3)).join(', '):'waiting')+
      ' m · '+(current.mountedVehicle?'mounted':'walking')+' · '+(current.active?'game controls':'Studio photo view');
  }
  async function update({asset,source,marbleAsset=null,sequence}) {
    if(closed||busy)throw Error('A Studio environment update is already in progress or closed.');
    if(!Number.isSafeInteger(sequence)||sequence<=lastSequence||!Number.isSafeInteger(source?.revision)||source.revision<0||
      source.revision!==asset?.revision||typeof source.classIdentity!=='string'||!source.classIdentity||typeof asset?.identity!=='string'||!asset.identity)
      throw Error('A newer matching Studio environment sequence is required.');
    const assetIdentity=asset.identity,revision=source.revision,classIdentity=source.classIdentity;
    busy=true;status.textContent='Preparing current Studio assets.';
    try{
      const qualification=player.whenReady();
      if(qualification){
        status.textContent='Waiting for original vehicle qualification.';
        await qualification;
        if(closed)throw Error('The Studio layer closed before original qualification completed.');
      }
      const prepared=bridge.prepareEnvironment(source,{assetIdentity});
      const beforeCommit=({isCurrent,commit})=>{
        if(closed)throw Error('The Studio layer closed before its environment could be queued.');
        return bridge.queueEnvironment({assetIdentity,revision,sequence,definition:prepared.definition,
          isCurrent:()=>!closed&&isCurrent(),commit});
      };
      if(!collection){
        const candidate=await installStudioCollection({host,assets:marbleAsset===null?[]:[{asset:marbleAsset,quantity:12}],onStatus:value=>{
          if(closed)return;
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
        const candidate=await installStudioRoom({host,asset,building:prepared.building,sequence,sourceRevision:revision,beforeCommit,
          onStatus:value=>{if(!closed)status.textContent='Studio scene: '+value.state+(value.error?' '+value.error:'');}});
        if(closed){candidate.dispose();throw Error('The Studio layer closed while preparing its room.');}
        room=candidate;
        acknowledged=await room.ready;
      }else acknowledged=await room.update({asset,building:prepared.building,sequence,sourceRevision:revision,beforeCommit});
      // Preserve a ready/update rejection, but never admit a success after Stop.
      if(closed)throw Error('The Studio layer closed before its update was acknowledged.');
      const receipt=validateStudioGameReceipt({...acknowledged,epoch,building:prepared.building,
        colliderPublication:acknowledged?.environment?.colliderPublication,
        physics:'original-game-worker',camera:'original-game-controls'},
        {protocol,assetIdentity,classIdentity,revision,sequence});
      if(!entry){entry=player.enter(receipt);player.setActive(active);refreshPlayer();}
      lastSequence=sequence;lastReceipt=receipt;
      status.textContent='Studio revision '+revision+' displayed. Use original game controls and ammunition.';
      return lastReceipt;
    }catch(error){if(!closed)status.textContent='Studio update unavailable: '+String(error.message||error);throw error;}
    finally{busy=false;}
  }
  return {update,readQualification:()=>player.readQualification(),setActive(value){
    if(closed)return false;
    active=Boolean(value);player.setActive(active&&!!entry);refreshPlayer();return active;
  },previewPose(packet){
    if(closed||!lastReceipt||packet?.epoch!==epoch)return false;
    return room?.previewPose(packet)??false;
  },readProtocol:()=>protocol,connect(sink){
    if(closed)throw Error('The Studio layer is closed.');
    if(typeof sink!=='function')throw TypeError('The Studio layer state sink must be a function.');
    const owner={};
    bridge.setSink(packet=>{if(!closed&&connection===owner){refreshPlayer();sink(packet);}},{epoch});connection=owner;
    return ()=>{if(connection!==owner)return;connection=null;bridge.setSink(null);};
  },readSnapshot:()=>closed?null:bridge.readSnapshot(),
    read:()=>({busy,closed,lastSequence,lastReceipt,entry,player:player.read(),collection:collection?.readStatus(),room:room?.readStatus()}),
    dispose(){if(closed)return;player.setActive(false);closed=true;connection=null;bridge.setSink(null);room?.dispose();collection?.dispose();panel.remove();}};
}

if(typeof window!=='undefined'&&window.mechanicalCreatureStudioBridge)
  window.mechanicalCreatureStudioLayer=createMechanicalCreatureStudioLayerHost();
