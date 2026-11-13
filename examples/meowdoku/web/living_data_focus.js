// Optical adapter for the original Living Data draw. Reuses Studio's triangle
// law, DOF shader and autofocus objective; no second renderer or simulation.
import {StudioRangefinder} from './marble_rangefinder.js';
import {createStudioDof} from './marble_studio_dof.js';
import {autofocusSamples,autofocusCandidates,autofocusContrast} from './marble_autofocus.js';
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit=a=>{const n=Math.hypot(...a);if(n<1e-9)throw Error('Camera facing is parallel to map up');return a.map(x=>x/n);};
export function livingDataFocusCapture(mesh,camera){
 const f=unit(camera.facing),r=unit(cross(f,[0,1,0])),u=cross(r,f),p=camera.position;
 const view=[r[0],u[0],-f[0],0,r[1],u[1],-f[1],0,r[2],u[2],-f[2],0,-dot(r,p),-dot(u,p),dot(f,p),1];
 const vertices=[],material_ids=[],triangleSources=[],colors=mesh.material_colors;
 for(let i=0;i<mesh.vertices.length;i+=27){
  let material=-1;
  for(let k=0;k<3;k++){
   const at=i+k*9,c=mesh.vertices.slice(at+6,at+9);
   let distance=Infinity,index=-1;
   colors.forEach((row,j)=>{const d=row.reduce((s,x,n)=>s+(Math.fround(x)-c[n])**2,0);if(d<distance){distance=d;index=j;}});
   if(k&&material!==index)throw Error('Mixed native material triangle');material=index;
   vertices.push(...mesh.vertices.slice(at,at+3));
  }
  const vertex=i/9,owner=[...mesh.identity_spans,...mesh.overlays].find(s=>vertex>=s.firstVertex&&vertex<s.firstVertex+s.vertexCount);
  if(!owner)throw Error('Unowned native map triangle');
  material_ids.push(material);triangleSources.push({partId:owner.identity??owner.kind,bodyId:owner.identity??owner.kind,triangle:(vertex-owner.firstVertex)/3});
 }
 return {revision:0,state:{schema:'turing.studio-visible-scene.v1',channel:'studio',
  camera:{view,aspect:camera.width/camera.height,fovDegrees:2*Math.atan(.70)*180/Math.PI},
  frame:{unitM:1,rayOriginM:[0,0,0]},ray:{vertices,material_ids,pbr:mesh.materials.pbr},triangleSources}};
}
export async function renderLivingDataFocus({gl,canvas,mesh,camera,draw,check,base=new URL('./client-ray-plain/',import.meta.url)}){
 const request=camera.optics;
 const optical={fov_degrees:2*Math.atan(.70)*180/Math.PI,
  aperture_radius_m:request.sensor_height_m/(4*.70*request.f_stop),focus_distance_m:request.focus_distance_m};
 const dof=createStudioDof(gl,{samples:0});let rangefinder=null;
 const preview=document.createElement('canvas'),ctx=preview.getContext('2d',{willReadFrequently:true});
 const scale=160/Math.max(camera.width,camera.height);preview.width=Math.max(3,Math.round(camera.width*scale));preview.height=Math.max(3,Math.round(camera.height*scale));
 const started=performance.now();
 function paint(focus){
  check();if(performance.now()-started>30000)throw Error('Autofocus exceeded 30-second budget');
  const optics={...optical,focus_distance_m:focus};
  // The native shadow/depth prepasses bind their own framebuffers. The supplied
  // hook begins Studio DOF only after those passes and before the native draw.
  draw(()=>dof.begin(camera.width,camera.height),()=>dof.finish(optics,{unitM:1,near:.04,far:128}));
  gl.finish();check();if(gl.getError()!==gl.NO_ERROR)throw Error('Native optical frame failed');
 }
 try{
  let winner={focus:optical.focus_distance_m},measured=0;
  if(request.autofocus){
   rangefinder=await StudioRangefinder.create(base);const capture=livingDataFocusCapture(mesh,camera),samples=[];
   for(const s of autofocusSamples()){check();const result=await rangefinder.query(capture,s);const hit=result.hits.find(h=>!h.behindOpaque&&h.axialDistanceM>0);if(hit)samples.push({...s,depth:hit.axialDistanceM});}
   measured=samples.length;
   const candidates=autofocusCandidates(samples,optical,{height:camera.height,aspect:camera.width/camera.height,min:.041,max:128});
   let best=null;
   for(const candidate of candidates){
    paint(candidate.focus);ctx.drawImage(canvas,0,0,preview.width,preview.height);
    const contrast=autofocusContrast(ctx.getImageData(0,0,preview.width,preview.height));
    if(!best||contrast>best.contrast)best={...candidate,contrast};
   }
   if(!(best?.contrast>0))throw Error('No image contrast for autofocus; no photo produced');
   winner=best;
  }
  paint(winner.focus);
  return {focus_distance_m:winner.focus,f_stop:request.f_stop,sensor_height_m:request.sensor_height_m,
   autofocus:request.autofocus,measured_rays:measured,coverage:winner.coverage??null,contrast:winner.contrast??null,
   method:'studio-gaussian-rangefinder-phong-contrast',depth_mode:'opaque',fov_degrees:optical.fov_degrees};
 }finally{rangefinder?.close();dof.dispose();}
}
