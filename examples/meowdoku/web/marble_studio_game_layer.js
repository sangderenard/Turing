// Optional host for the original generated game. Studio owns authored scene
// revisions; the game's existing worker owns its moving bodies and clock.
// No timer, movement controller, solver, camera override or ray job is created.
import {createStudioSceneAsset} from './marble_studio_scene_asset.js';
import {createStudioGameAsset} from './marble_game_asset.js';
import {createMechanicalCreatureStudioReturn} from './mechanical_creature_studio_return.js';

export function createStudioGameLayerController({capture,report=()=>{},createReturn=createMechanicalCreatureStudioReturn,
  sceneAsset=createStudioSceneAsset,gameAsset=createStudioGameAsset}) {
  if(typeof capture!=='function')throw TypeError('The displayed Studio capture function is required.');
  let host=null,receiver=null,marble=null,busy=false,desired=false,seen=-1,sent=-1,sequence=0,generation=0,
    failure=null,failureRevision=null,received=0,lastReceipt=null,disconnect=null,phase='off';
  const publish=()=>{try{report(read());}catch{/* UI cannot change engine ownership. */}};
  function read(){return {enabled:desired,connected:!!host,busy,authoredRevision:seen,sentRevision:sent,
    sequence,phase,receivedFrames:received,returned:receiver?.read()??null,receipt:lastReceipt,error:failure};}
  function connect(value){
    if(!value?.update||!value?.connect||!value?.readSnapshot)throw Error('The original game layer API is unavailable.');
    if(host&&host!==value)throw Error('Stop the existing game layer before replacing its host.');
    host=value;desired=true;failure=null;
    disconnect=host.connect(packet=>{if(!receiver)return;try{if(receiver.receive(packet)?.accepted===true){received++;publish();}}
      catch(error){failure=String(error.message||error);publish();}});publish();
  }
  async function send(revision,token){
    busy=true;failure=null;phase='capturing displayed scene';publish();
    try {
      // capture() runs at the incumbent Studio frame boundary. All inputs are
      // detached before its first await; later edits cause one newer revision.
      const packet=await capture();
      if(!desired||token!==generation)return;
      if(packet.revision!==revision)throw Error('Studio changed before its displayed capture.');
      phase='packing exact scene';publish();const asset=await sceneAsset(packet);
      if(!desired||token!==generation)return;
      const first=!marble;
      phase=first?'packing marble ammunition':'preparing environment update';publish();
      const authoredMarble=first?await gameAsset(packet,{name:'Current Studio marble'}):null;
      if(!desired||token!==generation)return;
      const source={schema:packet.schema,revision:packet.revision,classIdentity:packet.classIdentity,
        scene:{frame:packet.scene.frame,room:packet.scene.room,parts:packet.scene.parts}};
      const nextSequence=++sequence;
      phase='waiting for original game commit';publish();
      const receipt=await host.update({asset,source,marbleAsset:authoredMarble,sequence:nextSequence});
      if(!desired||token!==generation)return;
      const environment={buildingIdentity:receipt.building.identity,environmentAssetIdentity:asset.identity,environmentRevision:revision,environmentSequence:nextSequence,
        buildingPlacement:receipt.building.placement,unitM:packet.scene.frame.unitM};
      if(!receiver){
        const current=host.readSnapshot();
        if(!current?.epoch)throw Error('The original game has not published its first snapshot.');
        phase='preparing returned body view';publish();
        const candidate=await createReturn({epoch:current.epoch,environment,assets:[authoredMarble],fixedQuaternionXYZW:[0,0,0,1]});
        if(!desired||token!==generation){candidate.dispose();return;}
        receiver=candidate;
        if(receiver.receive(current)?.accepted===true)received++;
      }else receiver.setEnvironment(environment);
      if(first)marble=authoredMarble;sent=revision;lastReceipt=receipt;phase='running';
    }catch(error){if(token===generation){failure=String(error.message||error);failureRevision=revision;}}
    finally{if(token===generation){busy=false;publish();}}
  }
  return {connect,read,
    draw({gl,view,projection,phase='all'}){if(receiver&&gl)try{receiver.draw({gl,view,projection,phase});}
      catch(error){failure=String(error.message||error);publish();}},
    frame({revision,gl,view,projection,phase='all'}){
      if(!Number.isSafeInteger(revision)||revision<0)return;
      seen=revision;
      if(failure&&failureRevision!==null&&revision!==failureRevision){failure=null;failureRevision=null;}
      if(receiver&&gl)try{receiver.draw({gl,view,projection,phase});}
      catch(error){failure=String(error.message||error);publish();}
      if(desired&&host&&!busy&&revision!==sent&&!failure)void send(revision,generation);
    },
    retry(){failure=null;publish();},
    // A stopped host may still finish an installer. The new generation owns
    // busy/error state; the old promise cannot clear or poison its replacement.
    stop(){generation++;desired=false;busy=false;disconnect?.();disconnect=null;host?.dispose?.();host=null;receiver?.dispose();receiver=null;
      marble=null;sent=-1;received=0;lastReceipt=null;failure=null;failureRevision=null;phase='off';publish();},
  };
}

export function mountStudioGameLayer({doc=document,canvas,capture,url='./mechanical_creature_studio_layer.html'}) {
  if(!canvas)throw Error('The current Studio canvas is required.');
  const panel=doc.createElement('section');panel.id='studio-game-layer-controls';
  const open=doc.createElement('button');open.type='button';open.textContent='Open original game layer';
  const back=doc.createElement('button');back.type='button';back.textContent='Return to Studio view';back.hidden=true;
  const stop=doc.createElement('button');stop.type='button';stop.textContent='Stop game layer';stop.hidden=true;
  const retry=doc.createElement('button');retry.type='button';retry.textContent='Retry scene update';retry.hidden=true;
  const status=doc.createElement('p');status.id='studio-game-layer-status';status.textContent='Optional. The existing Studio and photo camera stay unchanged until opened.';
  const inspect=doc.createElement('details'),title=doc.createElement('summary'),receipt=doc.createElement('pre');
  title.textContent='Game synchronization';receipt.id='studio-game-layer-receipt';receipt.style.cssText='max-height:180px;overflow:auto;white-space:pre-wrap;font-size:10px';
  inspect.append(title,receipt);panel.append(open,back,stop,retry,status,inspect);
  const engine=doc.getElementById('preview-engine');(engine?.closest('fieldset')??engine?.parentElement?.parentElement??doc.body).append(panel);
  let frame=null,visible=false,loading=false;
  const controller=createStudioGameLayerController({capture,report:value=>{
    status.dataset.state=value.error?'failed':value.sentRevision>=0?'ready':value.enabled?'syncing':'off';
    status.textContent=value.error?'Game update paused: '+value.error:value.sentRevision>=0?
      `Scene revision ${value.sentRevision} sent. ${value.receivedFrames} original game frames received. Photo camera unchanged.`:
      value.connected?'Game layer: '+value.phase+'.':loading?'Loading the original game and controls.':'Game layer off. Studio remains unchanged.';
    retry.hidden=!value.error;receipt.textContent=JSON.stringify(value,null,2);
    if(value.sentRevision>=0&&frame)frame.style.visibility=visible?'visible':'hidden';
  }});
  function show(on){visible=on;back.hidden=!on;open.hidden=on;stop.hidden=!frame;
    Object.assign(panel.style,on?{position:'fixed',top:'0',left:'0',right:'0',height:'60px',overflow:'auto',zIndex:'230',background:'#15232e',padding:'4px'}:
      {position:'relative',top:'',left:'',right:'',height:'',overflow:'',zIndex:'',background:'',padding:''});
    if(frame){frame.style.visibility=on&&controller.read().sentRevision>=0?'visible':'hidden';frame.style.pointerEvents=on?'auto':'none';}
    if(!on)canvas.focus?.();}
  open.addEventListener('click',()=>{
    if(frame){show(true);return;}
    const target=new URL(url,doc.location.href);if(target.origin!==doc.location.origin)throw Error('Game layer must use the same origin as Studio.');
    loading=true;frame=doc.createElement('iframe');frame.title='Original MechanicalCreature game in Studio';
    frame.allow='gamepad';frame.dataset.studioGameLayer='true';
    Object.assign(frame.style,{position:'fixed',inset:'60px 0 0 0',width:'100%',height:'calc(100% - 60px)',border:'0',zIndex:'220',visibility:'hidden'});
    // The close strip belongs to Studio, not a replacement game HUD.
    Object.assign(panel.style,{position:'relative',zIndex:'230',background:'#15232e',padding:'8px'});
    const ownedFrame=frame;
    frame.addEventListener('load',()=>{if(frame!==ownedFrame)return;try{controller.connect(ownedFrame.contentWindow.mechanicalCreatureStudioLayer);loading=false;}
      catch(error){loading=false;status.dataset.state='failed';status.textContent=String(error.message||error);}}, {once:true});
    frame.src=target.href;doc.body.append(frame);show(true);status.textContent='Loading the original game and controls.';
  });
  back.addEventListener('click',()=>show(false));retry.addEventListener('click',()=>controller.retry());
  stop.addEventListener('click',()=>{controller.stop();frame?.remove();frame=null;loading=false;show(false);});
  return {frame:options=>controller.frame(options),draw:options=>controller.draw(options),read:controller.read,dispose(){controller.stop();frame?.remove();panel.remove();}};
}
