// Opt-in sample collection on the unchanged playable world's current controls.
import {installStudioCollection} from './mechanical_creature_studio_collection.js';
import {installStudioRoom} from './mechanical_creature_studio_room.js';
const panel=document.createElement('details');panel.open=true;
Object.assign(panel.style,{position:'fixed',left:'160px',top:'12px',zIndex:'110',maxWidth:'min(330px,85vw)',
  padding:'10px',background:'#102023ed',color:'#edf8ef',border:'1px solid #76ad99',borderRadius:'8px',font:'13px/1.4 system-ui'});
const title=document.createElement('summary');title.textContent='Studio marble collection';
const status=document.createElement('p');status.id='studio-transfer-status';status.textContent='Loading the captured Studio marble…';
const roomStatus=document.createElement('p');roomStatus.id='studio-room-status';roomStatus.hidden=true;
const help=document.createElement('p');help.textContent='Choose ammunition below or in slot 8. Use the existing fire / use, pickup and vacuum controls.';
const choices=document.createElement('div');choices.setAttribute('role','group');choices.setAttribute('aria-label','Studio collection ammunition');
Object.assign(choices.style,{display:'flex',gap:'6px',flexWrap:'wrap'});
const note=document.createElement('small');note.textContent='Original game physics and controls. Exact Studio Phong appearance with a captured light rig. No ray-traced bake yet.';
const inspect=document.createElement('button');inspect.type='button';inspect.textContent='Inspect transfer';
const receipt=document.createElement('pre');receipt.id='studio-transfer-receipt';receipt.hidden=true;
Object.assign(receipt.style,{maxHeight:'180px',overflow:'auto',fontSize:'10px',whiteSpace:'pre-wrap'});
panel.append(title,status,roomStatus,help,choices,note,document.createElement('br'),inspect,receipt);document.body.append(panel);
const responsive=document.createElement('style');responsive.textContent='@media(max-width:600px){details[data-studio-collection]{left:12px!important;top:100px!important;max-width:260px!important}}';
panel.dataset.studioCollection='true';document.head.append(responsive);
for(const event of ['pointerdown','pointerup','click','keydown'])panel.addEventListener(event,e=>e.stopPropagation());
let collection,room;
function refreshChoices(){
  choices.replaceChildren();
  for(const item of globalThis.mechanicalCreatureAmmo.items()){
    const button=document.createElement('button');button.type='button';button.setAttribute('aria-label','Use '+item.name);
    button.setAttribute('aria-pressed',String(globalThis.mechanicalCreatureAmmo.selected()?.identity===item.identity));
    Object.assign(button.style,{minHeight:'44px',display:'flex',alignItems:'center',gap:'4px'});
    if(item.properties.appearance_icon){const icon=document.createElement('img');icon.src=item.properties.appearance_icon;icon.alt='';icon.width=40;icon.height=40;button.append(icon);}
    const label=document.createElement('span');label.textContent=item.name+' ×'+item.quantity;button.append(label);
    button.addEventListener('click',()=>{globalThis.mechanicalCreatureAmmo.select(item.identity);refreshChoices();});choices.append(button);
  }
}
inspect.addEventListener('click',()=>{receipt.hidden=!receipt.hidden;receipt.textContent=JSON.stringify({marbles:collection?.readStatus()??{state:'loading'},studio:room?.readStatus()??null},null,2);});
try {
  const script=document.querySelector('script[data-studio-game-asset]');
  const assetUrl=new URL(script?.dataset.studioGameAsset??'./studio-cat-eye.game-asset.json',location.href);
  if(assetUrl.origin!==location.origin)throw Error('The chosen asset must be hosted with this game.');
  const response=await fetch(assetUrl);if(!response.ok)throw Error('Studio asset HTTP '+response.status);
  const asset=await response.json();
  collection=await installStudioCollection({assets:[{asset,quantity:12}],onStatus:value=>{
    if(value.state==='ready'||value.state==='partial')refreshChoices();
    status.dataset.state=value.state;
    status.textContent=value.state==='ready'?`Studio cat-eye ready. ${value.importedBodies} imported ball${value.importedBodies===1?'':'s'} in the original world.`:
      value.state==='waiting-for-original-render'?'Marble verified. Waiting for the original game renderer…':
      value.assets.map(x=>x.error||x.name).join('; ');
  }});
  if(script.dataset.studioSceneAsset){
    roomStatus.hidden=false;roomStatus.textContent='Loading the complete Studio building…';
    const url=new URL(script.dataset.studioSceneAsset,location.href);
    if(url.origin!==location.origin)throw Error('The Studio scene must be hosted with this game.');
    const reply=await fetch(url);if(!reply.ok)throw Error('Studio scene HTTP '+reply.status);
    const scene=await reply.json();
    room=await installStudioRoom({asset:scene,building:globalThis.mechanicalCreatureStudioBuilding,onStatus:value=>{
      roomStatus.dataset.state=value.state;
      roomStatus.textContent=value.state==='ready'?'Complete Marble Studio is in the lot beside the test tracks. Find Marble Studio on the existing map.':
        value.state==='failed'?'Studio materials unavailable: '+value.error:'Studio verified. Waiting for the original game renderer…';
    }});
  }
} catch(error) {status.dataset.state='failed';status.textContent='Studio transfer unavailable: '+String(error.message||error);}
