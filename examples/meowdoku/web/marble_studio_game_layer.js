// Optional host for the original generated game. Studio owns authored scene
// revisions; the game's existing worker owns its moving bodies and clock.
// No timer, movement controller, solver, camera override or ray job is created.
import {createStudioSceneAsset} from './marble_studio_scene_asset.js';
import {prepareStudioAmmunition} from './marble_game_asset.js';
import {createMechanicalCreatureStudioReturn} from './mechanical_creature_studio_return.js';
import {validateStudioGameProtocol,validateStudioGameReceipt} from './marble_studio_game_protocol.js';

export function createStudioGameLayerController({capture,report=()=>{},createReturn=createMechanicalCreatureStudioReturn,
  sceneAsset=createStudioSceneAsset,ammunition=prepareStudioAmmunition}) {
  if(typeof capture!=='function')throw TypeError('The displayed Studio capture function is required.');
  let host=null,protocol=null,receiver=null,initialized=false,ammunitionStatus=null,busy=false,desired=false,seen=-1,sent=-1,sequence=0,generation=0,
    poseSequence=0,failure=null,failureRevision=null,received=0,lastReceipt=null,disconnect=null,phase='off',active=true,qualificationKey='';
  const publish=()=>{try{report(read());}catch{/* UI cannot change engine ownership. */}};
  function read(){return {enabled:desired,connected:!!host,busy,authoredRevision:seen,sentRevision:sent,
    sequence,phase,protocol,qualification:host?.readQualification?.()??null,ammunition:ammunitionStatus,receivedFrames:received,returned:receiver?.read()??null,receipt:lastReceipt,error:failure};}
  function connect(value){
    if(!['update','connect','readSnapshot','readProtocol'].every(key=>typeof value?.[key]==='function'))throw Error('The original game layer API is unavailable.');
    if(host&&host!==value)throw Error('Stop the existing game layer before replacing its host.');
    if(host===value)return; // Repeated UI activation cannot replace a live sink.
    const admitted=validateStudioGameProtocol(value.readProtocol()),token=generation;
    const release=value.connect(packet=>{if(token!==generation||host!==value||!receiver)return;try{if(receiver.receive(packet)?.accepted===true){received++;publish();}}
      catch(error){failure=String(error.message||error);publish();}});
    if(typeof release!=='function')throw Error('The original game must provide its connection release.');
    host=value;protocol=admitted;desired=true;failure=null;disconnect=release;host.setActive?.(active);publish();
  }
  async function send(revision,token){
    busy=true;failure=null;phase='capturing displayed scene';publish();
    const currentPreparation=()=>{
      if(!desired||token!==generation)return false;
      if(seen!==revision){phase='waiting for latest displayed scene';return false;}
      return true;
    };
    try {
      // capture() runs at the incumbent Studio frame boundary. All inputs are
      // detached before its first await; later edits cause one newer revision.
      const packet=await capture();
      if(!currentPreparation())return;
      if(packet.revision!==revision)throw Error('Studio changed before its displayed capture.');
      phase='packing exact scene';publish();const asset=await sceneAsset(packet);
      if(!currentPreparation())return;
      const first=!initialized;
      phase=first?'checking optional marble ammunition':'preparing environment update';publish();
      const choice=first?await ammunition(packet,{name:'Current Studio marble'}):null;
      const authoredMarble=choice?.supported?choice.asset:null;
      if(!currentPreparation())return;
      const source={schema:packet.schema,revision:packet.revision,classIdentity:packet.classIdentity,
        scene:{frame:packet.scene.frame,room:packet.scene.room,parts:packet.scene.parts}};
      const nextSequence=++sequence;
      phase='waiting for original game commit';publish();
      // Once queued, retain its receipt even if Studio edits again. The game
      // may already own this complete environment; the next frame sends latest.
      const acknowledgment=await host.update({asset,source,marbleAsset:authoredMarble,sequence:nextSequence});
      if(!desired||token!==generation)return;
      const receipt=validateStudioGameReceipt(acknowledgment,{protocol,assetIdentity:asset.identity,classIdentity:source.classIdentity,revision,sequence:nextSequence});
      const environment={buildingIdentity:receipt.building.identity,environmentAssetIdentity:asset.identity,environmentRevision:revision,environmentSequence:nextSequence,
        buildingPlacement:receipt.building.placement,unitM:packet.scene.frame.unitM};
      if(!receiver){
        const current=host.readSnapshot();
        if(current&&current.epoch!==protocol.epoch)throw Error('The original snapshot belongs to a different game session.');
        phase='preparing returned body view';publish();
        const candidate=await createReturn({epoch:protocol.epoch,environment,assets:authoredMarble?[authoredMarble]:[],fixedQuaternionXYZW:[0,0,0,1]});
        if(!desired||token!==generation){candidate.dispose();return;}
        try{if(current&&candidate.receive(current)?.accepted===true)received++;}
        catch(error){candidate.dispose();throw error;}
        receiver=candidate;
      }else receiver.setEnvironment(environment);
      if(first){initialized=true;ammunitionStatus=Object.freeze({supported:choice.supported,identity:authoredMarble?.identity??null,reason:choice.reason});}
      sent=revision;lastReceipt=receipt;phase='running';
    }catch(error){if(token===generation){failure=String(error.message||error);failureRevision=revision;}}
    finally{if(token===generation){busy=false;publish();}}
  }
  return {connect,read,setActive(value){active=Boolean(value);host?.setActive?.(active);},
    draw({gl,view,projection,phase='all'}){if(receiver&&gl)try{receiver.draw({gl,view,projection,phase});}
      catch(error){failure=String(error.message||error);publish();}},
    frame({revision,subjectPose,gl,view,projection,phase='all'}){
      if(!Number.isSafeInteger(revision)||revision<0)return;
      seen=revision;
      // Reuse Studio's incumbent display frame to show the original gate. The
      // game alone decides qualification and starts its own world frame loop.
      const q=host?.readQualification?.(),key=q?`${q.state}:${Math.floor(q.simulatedSeconds)}`:'';
      if(key!==qualificationKey){qualificationKey=key;publish();}
      if(desired&&host&&lastReceipt&&revision===sent&&subjectPose&&typeof host.previewPose==='function')try {
        host.previewPose({epoch:protocol.epoch,assetIdentity:lastReceipt.identity,revision,
          environmentSequence:lastReceipt.sequence,sequence:++poseSequence,model:subjectPose});
      }catch(error){failure=String(error.message||error);publish();}
      if(failure&&failureRevision!==null&&revision!==failureRevision){failure=null;failureRevision=null;}
      if(receiver&&gl)try{receiver.draw({gl,view,projection,phase});}
      catch(error){failure=String(error.message||error);publish();}
      if(desired&&host&&!busy&&revision!==sent&&!failure)void send(revision,generation);
    },
    retry(){failure=null;publish();},
    // A stopped host may still finish an installer. The new generation owns
    // busy/error state; the old promise cannot clear or poison its replacement.
    stop(){generation++;desired=false;busy=false;disconnect?.();disconnect=null;host?.dispose?.();host=null;receiver?.dispose();receiver=null;
      protocol=null;poseSequence=0;initialized=false;ammunitionStatus=null;sent=-1;received=0;lastReceipt=null;failure=null;failureRevision=null;phase='off';publish();},
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
  const ammoNote=doc.createElement('p');ammoNote.id='studio-game-layer-ammunition';ammoNote.hidden=true;
  const inspect=doc.createElement('details'),title=doc.createElement('summary'),receipt=doc.createElement('pre');
  title.textContent='Game synchronization';receipt.id='studio-game-layer-receipt';receipt.style.cssText='max-height:180px;overflow:auto;white-space:pre-wrap;font-size:10px';
  inspect.append(title,receipt);panel.append(open,back,stop,retry,status,ammoNote,inspect);
  const engine=doc.getElementById('preview-engine');(engine?.closest('fieldset')??engine?.parentElement?.parentElement??doc.body).append(panel);
  let frame=null,visible=false,loading=false;
  const controller=createStudioGameLayerController({capture,report:value=>{
    status.dataset.state=value.error?'failed':value.sentRevision>=0?'ready':value.enabled?'syncing':'off';
    status.textContent=value.error?'Game update paused: '+value.error:value.qualification?.state==='failed'?
      'Original vehicle qualification failed. The original game has withheld the world.':value.qualification?.state==='pending'?
      `Original vehicle qualification: ${value.qualification.simulatedSeconds.toFixed(1)} s. Waiting for its result.`:value.sentRevision>=0?
      `Scene revision ${value.sentRevision} sent. ${value.receivedFrames} original game frames received. Photo camera unchanged.`:
      value.connected?'Game layer: '+value.phase+'.':loading?'Loading the original game and controls.':'Game layer off. Studio remains unchanged.';
    retry.hidden=!value.error;receipt.textContent=JSON.stringify(value,null,2);
    ammoNote.hidden=value.ammunition?.supported!==false;
    ammoNote.textContent=ammoNote.hidden?'':'Room imported. Subject is display-only: '+value.ammunition.reason;
    if(frame)frame.style.visibility=visible&&(value.sentRevision>=0||value.qualification)?'visible':'hidden';
  }});
  function show(on){visible=on;controller.setActive(on);back.hidden=!on;open.hidden=on;stop.hidden=!frame;
    Object.assign(panel.style,on?{position:'fixed',top:'0',left:'0',right:'0',height:'60px',overflow:'auto',zIndex:'230',background:'#15232e',padding:'4px'}:
      {position:'relative',top:'',left:'',right:'',height:'',overflow:'',zIndex:'',background:'',padding:''});
    if(frame){const value=controller.read();frame.style.visibility=on&&(value.sentRevision>=0||value.qualification)?'visible':'hidden';frame.style.pointerEvents=on?'auto':'none';}
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
