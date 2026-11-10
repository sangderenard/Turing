// CPU-only gobo geometry, two-sided ray hits and dynamic Advanced editors.
import assert from 'node:assert/strict';
import {MAX_STUDIO_GOBOS,GOBO_SEGMENTS,GOBO_THICKNESS,validateStudioGobos,
  studioGoboFaces,goboRotation,createStudioGoboControls} from '../examples/meowdoku/web/marble_studio_gobos.js';
const names=['black_felt','white_plaster','marble_clear_glass'],ids=Object.fromEntries(names.map((n,i)=>[n,i]));
const clone=v=>JSON.parse(JSON.stringify(v));
const sample={id:1,enabled:true,shape:'rectangle',width:2,height:4,x:0,y:0,z:0,pitch:0,yaw:0,roll:0,front:'black_felt',back:'white_plaster'};
const wrap=gobo=>({version:1,gobos:[gobo]});
const near=(a,b)=>assert(Math.abs(a-b)<1e-10,`${a} != ${b}`);
const sub=(a,b)=>a.map((v,i)=>v-b[i]),dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const hit=(origin,direction,faces)=>faces.map(face=>{
  const [a,b,c]=face,e1=sub(b,a),e2=sub(c,a),p=cross(direction,e2),det=dot(e1,p);
  if(Math.abs(det)<1e-12)return null;
  const tvec=sub(origin,a),u=dot(tvec,p)/det;if(u<0||u>1)return null;
  const q=cross(tvec,e1),v=dot(direction,q)/det;if(v<0||u+v>1)return null;
  const t=dot(e2,q)/det;return t>1e-8?{t,material:face[3]}:null;
}).filter(Boolean).sort((a,b)=>a.t-b.t)[0];
assert.deepEqual(validateStudioGobos(undefined,names),{version:1,gobos:[]});
assert.deepEqual(studioGoboFaces(undefined,ids),[]);
assert.deepEqual(studioGoboFaces(wrap({...sample,enabled:false}),ids),[]);
for(const value of [null,{},wrap({...sample,id:0}),wrap({...sample,enabled:1}),wrap({...sample,shape:'circle'}),
  wrap({...sample,front:'missing'}),wrap({...sample,back:''}),{version:2,gobos:[]},
  {version:1,gobos:[sample,sample]}, {version:1,gobos:Array.from({length:9},(_,i)=>({...sample,id:i+1}))}])
  assert.throws(()=>validateStudioGobos(value,names));
for(const key of ['width','height','x','y','z','pitch','yaw','roll'])
  for(const value of [NaN,Infinity,-Infinity,'1',null,true,1000,-1000])
    assert.throws(()=>validateStudioGobos(wrap({...sample,[key]:value}),names));
for(const key of ['width','height'])assert.throws(()=>validateStudioGobos(wrap({...sample,[key]:0.19}),names));
const state=wrap(sample),valid=validateStudioGobos(state,names);valid.gobos[0].x=5;assert.equal(state.gobos[0].x,0);
for(const shape of ['rectangle','ellipse'])for(const angles of [[0,0,0],[0,90,0],[90,0,0],[0,0,90],[23,-67,121]]) {
  const gobo={...sample,shape,x:4,y:-2,z:7,pitch:angles[0],yaw:angles[1],roll:angles[2]};
  const faces=studioGoboFaces(wrap(gobo),ids),count=shape==='rectangle'?4:GOBO_SEGMENTS;
  assert.equal(faces.length,count*4);
  const edges=new Map();let volume=0;
  for(const [a,b,c,material,ns] of faces) {
    assert([ids.black_felt,ids.white_plaster].includes(material));
    assert(dot(cross(sub(b,a),sub(c,a)),ns[0])>0,'Winding and normal agree');near(Math.hypot(...ns[0]),1);
    volume+=dot(a,cross(b,c))/6;
    for(const [p,q] of [[a,b],[b,c],[c,a]]) {
      const keys=[p.join(','),q.join(',')],key=keys.slice().sort().join('|'),old=edges.get(key)||[0,0];
      edges.set(key,[old[0]+1,old[1]+(keys[0]<keys[1]?1:-1)]);
    }
  }
  assert(volume>0);for(const edge of edges.values())assert.deepEqual(edge,[2,0],'Closed, consistently wound solid');
  const point=p=>goboRotation(gobo,p).map((v,i)=>v+[gobo.x,gobo.y,gobo.z][i]);
  const forward=goboRotation(gobo,[0,0,-1]),backward=goboRotation(gobo,[0,0,1]);
  const frontHit=hit(point([.13,.37,3]),forward,faces),backHit=hit(point([.13,.37,-3]),backward,faces);
  assert.equal(frontHit.material,ids.black_felt,`front ${shape} ${angles}`);assert.equal(backHit.material,ids.white_plaster,`back ${shape} ${angles}: ${JSON.stringify(backHit)}`);
  near(frontHit.t,3-GOBO_THICKNESS/2);near(backHit.t,3-GOBO_THICKNESS/2);
  assert.equal(hit(point([.95,1.9,3]),forward,faces)!==undefined,shape==='rectangle','Ovoid corners are empty, not a rectangular card');
}
assert.deepEqual(goboRotation(sample,[0,0,1]),[0,0,1]);
const yaw=goboRotation({...sample,yaw:90},[0,0,1]);near(yaw[0],1);near(yaw[1],0);near(yaw[2],0);
// All valid panels leave room for the actual fixture/room scene (integration
// check measures its final total, rather than changing the backend limit).
assert.equal(studioGoboFaces({version:1,gobos:Array.from({length:MAX_STUDIO_GOBOS},(_,i)=>({...sample,id:i+1,shape:'ellipse'}))},ids).length,1536);

class Element {
  constructor(tag,doc){this.tagName=tag;this.ownerDocument=doc;this.children=[];this.events={};this.value='';this.type='';}
  append(...nodes){this.children.push(...nodes);}
  replaceChildren(...nodes){this.children=nodes;}
  setAttribute(key,value){this[key]=value;}
  addEventListener(type,fn){(this.events[type]??=[]).push(fn);}
  fire(type){for(const fn of this.events[type]??[])fn();}
  focus(){this.ownerDocument.activeElement=this;}
  setCustomValidity(value){this.validityMessage=value;}
}
const all=node=>[node,...node.children.filter(n=>typeof n==='object').flatMap(all)];
const doc={createElement:tag=>new Element(tag,doc),getElementById:id=>all(container).find(n=>n.id===id)};
const container=new Element('div',doc);let changes=0;
const editor=createStudioGoboControls(container,names,()=>changes++),$=id=>doc.getElementById(id);
assert.equal(changes,0);assert.deepEqual(editor.getState(),{version:1,gobos:[]});
$('studio-gobo-add').fire('click');assert.equal(changes,1);assert.equal(editor.getState().gobos.length,1);
const first=editor.getState().gobos[0].id,prefix=`studio-gobo-${first}`;
assert.equal(editor.ownsControl(`${prefix}-width`),true);
for(const [key,value,event] of [['shape','ellipse','change'],['width','3.2','input'],['height','1.4','input'],
  ['x','1.1','input'],['y','-2.3','input'],['z','4.5','input'],['pitch','11','input'],['yaw','29','input'],['roll','-47','input'],
  ['front','white_plaster','change'],['back','marble_clear_glass','change']]) {
  const node=$(`${prefix}-${key}`);node.value=value;node.fire(event);
  assert.equal(editor.getState().gobos[0][key],node.type==='number'?Number(value):value);
}
const validState=editor.getState(),count=changes;
for(const value of ['','-1','1000','NaN']) {
  $(`${prefix}-width`).value=value;$(`${prefix}-width`).fire('input');
  assert.deepEqual(editor.getState(),validState);assert.equal(changes,count);assert.equal($(`${prefix}-width`)['aria-invalid'],'true');
}
$(`${prefix}-width`).value='2.5';$(`${prefix}-width`).fire('input');assert.equal($(`${prefix}-width`)['aria-invalid'],'false');
$(`${prefix}-enabled`).checked=false;$(`${prefix}-enabled`).fire('change');assert.equal(editor.getState().gobos[0].enabled,false);
assert.deepEqual(studioGoboFaces(editor.getState(),ids),[]);
for(let i=1;i<MAX_STUDIO_GOBOS;i++)$('studio-gobo-add').fire('click');
assert.equal($('studio-gobo-add').disabled,true);$('studio-gobo-add').fire('click');assert.equal(editor.getState().gobos.length,MAX_STUDIO_GOBOS);
const existing=editor.getState(),last=existing.gobos.at(-1);
$(`${prefix}-remove`).fire('click');assert.equal($('studio-gobo-add').disabled,false);
assert.deepEqual(editor.getState().gobos,existing.gobos.slice(1));assert.equal(editor.ownsControl(`${prefix}-width`),false);
$('studio-gobo-add').fire('click');assert(editor.getState().gobos.at(-1).id>last.id);
const save=editor.getState(),before=changes;
assert.throws(()=>editor.setState(wrap({...sample,back:'missing'})));assert.deepEqual(editor.getState(),save);assert.equal(changes,before);
editor.setState(undefined);assert.deepEqual(editor.getState(),{version:1,gobos:[]});assert.equal(changes,before);
editor.setState(save);assert.deepEqual(editor.getState(),save);assert.equal(changes,before);
const cloneState=editor.getState();cloneState.gobos[0].x=99;assert.deepEqual(editor.getState(),save);
console.log('Advanced gobos: validation, closed solids, two-sided hits/orientation, ovoid corners, spawn/edit/hide/remove/limits and atomic state passed');


