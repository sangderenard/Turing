import assert from 'node:assert/strict';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {defaultStudioGrid,validateStudioGrid,studioGridCells,validateStudioArtwork,studioGraphPaperArtwork} from '../examples/meowdoku/web/marble_studio_grid.js';
import {studioGoboFaces,validateStudioGobos,goboRotation,createStudioGoboControls,studioGraphPaperGobo} from '../examples/meowdoku/web/marble_studio_gobos.js';
import {backdropFaces,defaultBackdropLayers,validateBackdropLayers,createBackdropLayerControls} from '../examples/meowdoku/web/marble_backdrop.js';
import {buildStudioAgentProduct} from '../examples/meowdoku/web/marble_studio_agent.js';
const names=['black_felt','white_plaster','marble_clear_glass'],ids=Object.fromEntries(names.map((n,i)=>[n,i]));
const grid={version:1,rows:2,columns:3,gap:0.1,cells:['white_plaster',null,'','black_felt','marble_clear_glass',null]};
const gobo={id:1,enabled:true,shape:'rectangle',width:6,height:4,x:0,y:0,z:-2.5,pitch:0,yaw:0,roll:0,front:'black_felt',back:'white_plaster',grid};
const wrap=g=>({version:1,gobos:[g]});
const clone=structuredClone,near=(a,b)=>assert(Math.abs(a-b)<1e-10,`${a} != ${b}`);
assert.equal(validateStudioGrid(undefined,names),undefined);
assert.deepEqual(validateStudioGrid(grid,names),grid);
const valid=validateStudioGrid(grid,names);valid.cells[0]=null;assert.equal(grid.cells[0],'white_plaster');
for(const change of [{version:2},{rows:0},{columns:65},{rows:1.5},{gap:0},{gap:NaN},{gap:.91},{source:'invented'},
  {rows:64,columns:64,cells:Array(4096).fill('')},{cells:['']},{cells:['missing',null,'','black_felt','marble_clear_glass',null]},
  {cells:[{},null,'','black_felt','marble_clear_glass',null]},{cells:Array(6)}])
  assert.throws(()=>validateStudioGrid({...grid,...change},names));
assert.throws(()=>validateStudioGobos(wrap({...gobo,shape:'ellipse'}),names),/rectangle/);
const layers=defaultBackdropLayers();layers[0]={material:'white_plaster',width:6,height:4,grid};
assert.throws(()=>validateBackdropLayers([{...layers[0],shape:'ellipse'},...layers.slice(1)],names),/rectangle/);
const cells=studioGridCells(grid,6,4);
assert.deepEqual(cells.map(c=>[c.id,c.x,c.y]),[['r0-c0',-2,1],['r0-c2',2,1],['r1-c0',-2,-1],['r1-c1',0,-1]]);
assert(cells.every(c=>c.width===1.8&&c.height===1.8));
const sub=(a,b)=>a.map((x,i)=>x-b[i]),dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
function hit(origin,direction,faces) {
  return faces.flatMap(([a,b,c,material])=>{
    const e1=sub(b,a),e2=sub(c,a),p=cross(direction,e2),det=dot(e1,p);if(Math.abs(det)<1e-12)return [];
    const t=sub(origin,a),u=dot(t,p)/det,q=cross(t,e1),v=dot(direction,q)/det,d=dot(e2,q)/det;
    return u>=0&&v>=0&&u+v<=1&&d>0?[{d,material}]:[];
  }).sort((a,b)=>a.d-b.d)[0];
}
for(const angles of [[0,0,0],[23,-67,121]]) {
  const state={...gobo,pitch:angles[0],yaw:angles[1],roll:angles[2]},parts=[];
  const faces=studioGoboFaces(wrap(state),ids,p=>parts.push(p));assert.equal(faces.length,64);assert.equal(parts.length,4);
  assert.equal(new Set(parts.map(p=>p.bodyId)).size,4);assert.deepEqual(parts.flatMap(p=>p.faces),faces);
  for(const part of parts) {
    assert(part.closed&&part.convex);const edges=new Map();let volume=0;
    for(const [a,b,c,,ns] of part.faces) {
      assert(dot(cross(sub(b,a),sub(c,a)),ns[0])>0);near(Math.hypot(...ns[0]),1);volume+=dot(a,cross(b,c))/6;
      for(const [p,q] of [[a,b],[b,c],[c,a]]){const keys=[p.join(','),q.join(',')],key=keys.slice().sort().join('|'),edge=edges.get(key)??[0,0];edge[0]++;edge[1]+=keys[0]<keys[1]?1:-1;edges.set(key,edge);}
    }
    near(volume,1.8*1.8*.12);for(const edge of edges.values())assert.deepEqual(edge,[2,0]);
  }
  const point=p=>goboRotation(state,p).map((v,i)=>v+[state.x,state.y,state.z][i]),direction=goboRotation(state,[0,0,-1]);
  assert.equal(hit(point([.01,1.01,3]),direction,faces),undefined,'Explicit null leaves a real hole');
  assert.equal(hit(point([-1,1.01,3]),direction,faces),undefined,'Cell-pitch gap leaves a real opening');
  assert.equal(hit(point([2.01,1.01,3]),direction,faces).material,ids.black_felt,'Inherited front');
  assert.equal(hit(point([2.01,1.01,-3]),goboRotation(state,[0,0,1]),faces).material,ids.white_plaster,'Inherited back');
  assert.equal(hit(point([-1.99,1.01,-3]),goboRotation(state,[0,0,1]),faces).material,ids.white_plaster,'Named material on both faces');
}
for(const sweep of [false,true]) {
  const parts=[],faces=backdropFaces({count:1,material:'black_felt',sweep,layers},ids,p=>parts.push(p));
  assert.equal(faces.length,sweep?16:8);assert.deepEqual(parts.flatMap(p=>p.faces),faces);
  assert.equal(new Set(parts.map(p=>p.id)).size,parts.length);
  for(const [a,b,c,,ns] of faces)assert(dot(cross(sub(b,a),sub(c,a)),ns[0])>0);
}
assert.deepEqual(studioGoboFaces(wrap({...gobo,grid:{...grid,cells:Array(6).fill(null)}}),ids),[]);
assert.deepEqual(backdropFaces({count:1,material:'black_felt',layers:[{...layers[0],grid:{...grid,cells:Array(6).fill(null)}},...layers.slice(1)]},ids),[]);

class Element {
  constructor(tag,doc){this.tagName=tag;this.ownerDocument=doc;this.children=[];this.events={};this.value='';this.type='';}
  append(...n){this.children.push(...n);} replaceChildren(...n){this.children=n;} setAttribute(k,v){this[k]=v;}
  addEventListener(t,fn){(this.events[t]??=[]).push(fn);} fire(t){for(const fn of this.events[t]??[])fn();}
  focus(){} setCustomValidity(v){this.validityMessage=v;}
}
const all=n=>[n,...n.children.filter(n=>typeof n==='object').flatMap(all)];
const doc={createElement:t=>new Element(t,doc),getElementById:id=>all(container).find(n=>n.id===id)},container=new Element('div',doc),$=doc.getElementById;
let edits=0;const editor=createStudioGoboControls(container,names,()=>edits++);
$('studio-gobo-add-grid').fire('click');assert.equal(edits,1);
const authored=editor.getState().gobos[0];assert.deepEqual([authored.x,authored.y,authored.z,authored.width,authored.height],[0,0,-2.5,8,8]);
assert.deepEqual(authored.grid,defaultStudioGrid());assert.equal(studioGoboFaces(editor.getState(),ids).length,4096);
const prefix=`studio-gobo-${authored.id}-grid`;
assert(editor.ownsControl(`${prefix}-cells`));
$(`${prefix}-rows`).value='2';$(`${prefix}-columns`).value='3';$(`${prefix}-resize`).fire('click');assert.equal(edits,2);
$(`${prefix}-cells`).value=JSON.stringify(grid.cells);$(`${prefix}-apply`).fire('click');assert.equal(edits,3);
const previous=editor.getState();$(`${prefix}-cells`).value='["missing"]';$(`${prefix}-apply`).fire('click');
assert.deepEqual(editor.getState(),previous);assert.equal(edits,3);
$(`${prefix}-cells`).value='not JSON';$(`${prefix}-apply`).fire('click');assert.deepEqual(editor.getState(),previous);
editor.setState(previous);assert.deepEqual(editor.getState(),previous);
$(`${prefix}-enabled`).checked=false;$(`${prefix}-enabled`).fire('change');assert.equal(editor.getState().gobos[0].grid,undefined);
$(`${prefix}-enabled`).checked=true;$(`${prefix}-enabled`).fire('change');assert.deepEqual(editor.getState(),previous);
const snapshot=editor.getState();snapshot.gobos[0].grid.cells[0]=null;assert.deepEqual(editor.getState(),previous);
$(`studio-gobo-${authored.id}-remove`).fire('click');assert(!editor.ownsControl(`${prefix}-cells`));
const backdropContainer=doc.createElement('div');container.append(backdropContainer);
const backdropEditor=createBackdropLayerControls(backdropContainer,names,()=>edits++);backdropEditor.setState(layers);
const backdropState=backdropEditor.getState();backdropState[0].grid.cells[0]=null;assert.deepEqual(backdropEditor.getState(),layers);
$('backdrop-layer-0-grid-cells').value=JSON.stringify(Array(6).fill(null));$('backdrop-layer-0-grid-apply').fire('click');
assert(backdropEditor.getState()[0].grid.cells.every(c=>c===null));

// Existing validator and scene/MRP1 boundary, using actual captured authoring.
const {assets,authoring}=JSON.parse(await readFile(new URL('fixtures/studio_scene_builder_authoring.json',import.meta.url),'utf8'));
const settings={object:authoring.object,lighting:clone(authoring.lighting),camera:{eye_m:[0,.08,.07],target_m:[0,.08,.18],fov_degrees:18,aperture_radius_m:.002,focus_distance_m:.11}};
settings.lighting.gobos=wrap({...gobo,grid:defaultStudioGrid()});
settings.lighting.roomOverrides={version:1,values:[]};
const request={schema:'turing.studio-agent-request.v1',operation:'ray-payload',settings,photo:{resolution:256,oversampling:4,seconds:1,maxBounces:4}};
const before=JSON.stringify(request),product=await buildStudioAgentProduct(request,{assets});assert.equal(JSON.stringify(request),before);
const view=new DataView(product.bytes.buffer,product.bytes.byteOffset,product.bytes.byteLength),size=view.getUint32(4,true);
const config=JSON.parse(new TextDecoder().decode(product.bytes.slice(16,16+size)));
assert.equal(config.camera.resolution,256);assert.equal(config.oversampling,4);assert.deepEqual(config.tiled,{tile_size:1024});
assert(config.fixtures.material_ids.length>4096);assert(size<=2*1024*1024,'Default grid fits unchanged MRP1 serialized config limit');
const out=new URL('../build/dot-studio-grid-check/',import.meta.url);await mkdir(out,{recursive:true});
await writeFile(new URL('ray.mrp1',out),product.bytes);
console.log(`Structured grid passed: validation, closed cell solids, true holes, normals/materials, backdrop/sweep, atomic UI and real Studio MRP1. ${config.fixtures.material_ids.length} fixture triangles; ${size} config bytes; 256² final, 4×, 1024 tile.`);

const paper=studioGraphPaperArtwork(),catalog=Object.keys(assets.material_index),paperGobo=studioGraphPaperGobo({unitMm:304.8});
const refContainer=doc.createElement('div');container.append(refContainer);
const refEditor=createStudioGoboControls(refContainer,catalog,()=>{},
  {referencePlacement:()=>({unitMm:304.8,center:[0,-1+32/304.8,-.17]})});
const refNode=id=>all(refContainer).find(n=>n.id===id);
refNode('studio-gobo-add-reference').fire('click');
assert.equal(refEditor.getState().gobos.length,1);near(refEditor.getState().gobos[0].width*304.8,64);
assert.equal(refEditor.getState().gobos[0].front,'white_plaster');
const refBefore=refEditor.getState();refNode('studio-gobo-1-grid-enabled').checked=true;refNode('studio-gobo-1-grid-enabled').fire('change');
assert.deepEqual(refEditor.getState(),refBefore,'A perforated grid cannot silently replace reflective surface markings');
for(const name of ['white_plaster','black_matte_paint',...paper.reference.swatches]) {
  const offset=assets.material_index[name]*assets.materials.pbr.floats_per_record;
  assert.deepEqual(assets.materials.pbr.data.slice(offset+8,offset+11),[0,0,0],`${name}: reference card must not introduce a light`);
}
assert.deepEqual(validateStudioArtwork(paper.artwork,catalog),paper.artwork);
assert.deepEqual(validateStudioGobos(wrap(paperGobo),catalog).gobos[0],paperGobo);
for(const r of [{...paper.artwork.rectangles[0],x:.6},{...paper.artwork.rectangles[0],material:'missing'},{...paper.artwork.rectangles[0],width:0}])
  assert.throws(()=>validateStudioArtwork({version:1,rectangles:[r]},catalog));
assert.throws(()=>validateStudioGobos(wrap({...paperGobo,grid}),catalog),/solid rectangular/);
const paperParts=[],paperFaces=studioGoboFaces(wrap(paperGobo),assets.material_index,p=>paperParts.push(p));
assert.equal(paperFaces.length,16+paper.artwork.rectangles.length*2);assert.deepEqual(paperParts.flatMap(p=>p.faces),paperFaces);
assert(paperParts[0].closed);assert(paperParts.slice(1).every(p=>!p.closed));
assert.equal(new Set(paperParts.map(p=>p.id)).size,paperParts.length);
const line0=paper.artwork.rectangles.find(r=>r.id==='grid-v-0'),line1=paper.artwork.rectangles.find(r=>r.id==='grid-v-1');
near((line1.x-line0.x)*paperGobo.width*304.8,2);
assert.equal(paper.reference.qrPayloads.join('|'),'STUDIO;2mm;L|STUDIO;2mm;R');
const paperLayers=defaultBackdropLayers();paperLayers[0]={...paperLayers[0],artwork:paper.artwork};
const paperBackdropContainer=doc.createElement('div');container.append(paperBackdropContainer);
const paperBackdropEditor=createBackdropLayerControls(paperBackdropContainer,catalog,()=>{});paperBackdropEditor.setState(paperLayers);
const paperShape=all(paperBackdropContainer).find(n=>n.id==='backdrop-layer-0-shape');paperShape.value='ellipse';paperShape.fire('change');
assert.deepEqual(paperBackdropEditor.getState(),paperLayers);assert.equal(paperShape.value,'rectangle');
settings.lighting.gobos=wrap(paperGobo);
const graphProduct=await buildStudioAgentProduct(request,{assets}),graphView=new DataView(graphProduct.bytes.buffer,graphProduct.bytes.byteOffset,graphProduct.bytes.byteLength);
assert(graphView.getUint32(4,true)<2*1024*1024);
await writeFile(new URL('graph-paper.json',out),JSON.stringify(paper));
await writeFile(new URL('graph-paper.mrp1',out),graphProduct.bytes);
console.log(`Graph paper passed: ${paper.artwork.rectangles.length} bounded material rectangles, exact2mm pitch at64mm width, canonical swatches and two incumbent-encoder QR codes; ${graphView.getUint32(4,true)} config bytes.`);
