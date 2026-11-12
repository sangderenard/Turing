// Diagnostic geometry only. This does not change a fixture, its optics, or any
// render ray. The aperture guide uses emitter-center rays; stage coverage is
// explicitly sampled through the existing compiled original triangle law.
import {studioTableFaces,TABLE_TOP_Y} from './marble_studio_table.js';
import {StudioCompiledLawBatch} from './studio_compiled_law_batch.js';
import {verifiedComputeBytes} from './studio_triangle_compute.js';
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0),sub=(a,b)=>a.map((x,i)=>x-b[i]);
const add=(a,b,s=1)=>a.map((x,i)=>x+s*b[i]),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit=v=>{const n=Math.hypot(...v);if(!(n>0))throw Error('Guide direction must be nonzero');return v.map(x=>x/n);};
const vector=v=>Array.isArray(v)&&v.length===3&&v.every(Number.isFinite);
function basis(dir){const forward=unit(dir),right=unit(cross(forward,Math.abs(forward[1])>.95?[1,0,0]:[0,1,0]));return [right,unit(cross(right,forward)),forward];}
function check(role,subject){if(!vector(role?.pos)||!vector(role?.dir)||!vector(subject?.center)||!Number.isFinite(subject.radius)||subject.radius<=0)throw Error('Guide needs a finite light pose and subject sphere');}
const distance2=(a,b)=>dot(sub(a,b),sub(a,b));
function segmentDistance2(p,a,b){const e=sub(b,a),t=Math.max(0,Math.min(1,dot(sub(p,a),e)/dot(e,e)));return distance2(p,add(a,e,t));}
function faceDistance2(p,a,b){
 // A cone side is {u*a+v*b: u,v>=0, u+v>=1}. Project to its
 // plane, then test each of its three boundary pieces when outside.
 const aa=dot(a,a),bb=dot(b,b),ab=dot(a,b),pa=dot(p,a),pb=dot(p,b),det=aa*bb-ab*ab;
 const u=(pa*bb-pb*ab)/det,v=(pb*aa-pa*ab)/det;
 const candidates=[segmentDistance2(p,a,b),distance2(p,a.map(x=>x*Math.max(1,pa/aa))),distance2(p,b.map(x=>x*Math.max(1,pb/bb)))];
 if(u>=0&&v>=0&&u+v>=1)candidates.push(distance2(p,a.map((x,i)=>u*x+v*b[i])));
 return Math.min(...candidates);
}

export function snootSphereGuide(role,subject){
 check(role,subject);
 const scope='Exact 12-sided emerging aperture cone versus subject sphere, emitter-center rays only. Extended-source illumination and scene occlusion are not certified.';
 if(role.grid)return {state:'unverified',scope:'The physical rectangular grid is traced by the renderer; this circular-snoot guide does not certify it.',message:'Physical grid attached; aperture clearance is not certified by the snoot guide.'};
 if(!role.gobo||!(role.shroudLength>0))return {state:'open',scope,message:'No active snoot aperture restriction.'};
 const offset=role.shroudOffset??0,length=role.shroudLength,total=offset+length;
 const mount=role.shroudRadius??role.radius*1.1,exit=role.shroudExitRadius??mount;
 if(![offset,length,mount,exit].every(Number.isFinite)||offset<0||length<=0||mount<=0||exit<=0)throw Error('Invalid physical snoot dimensions');
 // The real collar and linear taper have homothetic 12-sided cross sections.
 // A center ray is constrained by the smallest radius/axial-distance ratio.
 const slope=Math.min(exit/total,offset>0?mount/offset:Infinity),radius=slope*total;
 const axes=basis(role.dir),delta=sub(subject.center,role.pos),p=axes.map(a=>dot(delta,a));
 const vertices=Array.from({length:12},(_,i)=>[radius*Math.cos(i*Math.PI/6),radius*Math.sin(i*Math.PI/6),total]);
 const normals=vertices.map((v,i)=>unit(cross(v,vertices[(i+1)%12]))),margins=normals.map(n=>dot(p,n));
 const baseMargin=p[2]-total,minMargin=Math.min(baseMargin,...margins);
 let gap=0;
 if(minMargin<0){
  const projection=[p[0],p[1],total];
  const distances=vertices.map((v,i)=>faceDistance2(p,v,vertices[(i+1)%12]));
  if(normals.every(n=>dot(projection,n)>=0))distances.push(baseMargin*baseMargin);
  else vertices.forEach((v,i)=>distances.push(segmentDistance2(p,v,vertices[(i+1)%12])));
  gap=Math.sqrt(Math.min(...distances));
 }
 const state=minMargin>=subject.radius?'clear':gap>=subject.radius?'clipped':'partial';
 return {state,scope,minimumClearance:minMargin-subject.radius,distanceToCone:gap,exitDistance:total,
  axialDistance:p[2],subjectCrossesExit:p[2]-subject.radius<total&&p[2]+subject.radius>total,
  innerHalfAngleDegrees:Math.atan(slope*Math.cos(Math.PI/12))*180/Math.PI,
  outerHalfAngleDegrees:Math.atan(slope)*180/Math.PI,
  message:{clear:'Subject sphere fits the aperture cone.',partial:'Subject sphere is partly outside the aperture cone.',clipped:'Subject sphere does not overlap the emerging aperture cone.'}[state]};
}

export function kickerSubjectRays(role,subject,unitM=1){
 check(role,subject);if(!Number.isFinite(unitM)||unitM<=0)throw Error('Invalid physical Studio unit');
 const delta=sub(subject.center,role.pos),distance=Math.hypot(...delta),r=subject.radius;
 if(distance<=r)throw Error('Kicker is inside the subject clearance sphere');
 const axes=basis(delta),samples=[[0,0]];
 // Center plus four twelve-ray annuli across the apparent disk. These are
 // diagnostic samples, not an area/power integral or proof of total occlusion.
 for(let ring=0;ring<4;ring++)for(let i=0;i<12;i++)samples.push([Math.sqrt((ring+.5)/4),2*Math.PI*(i+(ring%2)/2)/12]);
 return samples.map(([fraction,angle])=>{
  const sin=fraction*r/distance,cos=Math.sqrt(1-sin*sin);
  const direction=axes[2].map((x,i)=>cos*x+sin*(Math.cos(angle)*axes[0][i]+Math.sin(angle)*axes[1][i]));
  const end=distance*cos-Math.sqrt(r*r-distance*distance*sin*sin);
  return {origin:role.pos.map(x=>x*unitM),direction,maximum:end*unitM};
 });
}

export function kickerTableClearanceBound(role,subject,table,floorY){
 check(role,subject);
 if(!table.enabled)return {state:'off',minimumHeight:null,message:'Table is off; no table-height bound.'};
 if(!Number.isFinite(floorY))throw Error('Guide needs the actual emitter dimensions and floor');
 const axes=basis(role.dir),rect=role.round===false||Number.isFinite(role.halfW)&&Number.isFinite(role.halfH);
 const radius=rect?Math.hypot(role.halfW,role.halfH):role.radius;
 if(!Number.isFinite(radius)||radius<=0||rect&&(!(role.halfW>0)||!(role.halfH>0)))throw Error('Guide needs the actual emitter dimensions and floor');
 const vertical=rect?Math.abs(axes[0][1])*role.halfW+Math.abs(axes[1][1])*role.halfH:radius*Math.hypot(axes[2][0],axes[2][2]);
 const bottom=role.pos[1]-vertical;
 const subjectAbove=subject.center[1]-subject.radius>=TABLE_TOP_Y;
 // Both endpoint sets above the common slab top imply every segment stays
 // above it, by convexity. Legs and skirt are wholly below that same plane.
 // Using the full radius for a proposed height is valid at every aiming angle.
 const minimumHeight=TABLE_TOP_Y-floorY+radius;
 return {state:subjectAbove&&bottom>=TABLE_TOP_Y?'clear':'unproven',minimumHeight,emitterBottomY:bottom,
  message:subjectAbove?`Whole-emitter table clearance is sufficient at height ${minimumHeight.toFixed(3)} or higher (conservative bound, any aim). Aperture and other scene objects still apply.`:
   'Subject extends below tabletop; the whole-emitter table-clearance bound does not apply.'};
}

export function tableGuideScene(table,floorY,unitM=1){
 if(!Number.isFinite(unitM)||unitM<=0)throw Error('Invalid physical Studio unit');
 const names=[...new Set([table.material,table.skirt?.material??'black_felt'])],ids=Object.fromEntries(names.map((n,i)=>[n,i]));
 const triangles=[];
 studioTableFaces(table,ids,floorY,part=>{for(const face of part.faces){
  const [a,b,c]=face.slice(0,3).map(v=>v.map(x=>x*unitM));
  triangles.push({v0:a,e1:sub(b,a),e2:sub(c,a),part:part.id,material:names[face[3]]});
 }});
 return {triangles,scope:'Shared table slab, legs and skirt mesh only. Material transmission, room walls, other fixtures and gobos are not evaluated.'};
}

export class KickerTableQuery {
 static async create(base=new URL('./client-ray-plain/',import.meta.url)){
  const contract=JSON.parse(new TextDecoder().decode(await verifiedComputeBytes(base,'studio_bvh_query.json')));
  if(contract.schema!=='studio.bvh-query.v1'||!contract.laws?.hit_tri)throw Error('Original Studio triangle law is unavailable');
  const law=await StudioCompiledLawBatch.create(base,'studio_hit_tri_batch',contract.laws.hit_tri);
  if(law.manifest.input_ids.length!==15||law.manifest.output_ids.length!==2||law.manifest.entry!=='hit_tri_batch')throw Error('Unsupported original triangle batch ABI');
  return new KickerTableQuery(law,contract);
 }
 constructor(law,contract){this.law=law;this.contract=contract;this.busy=false;}
 async query(scene,rays,{signal}={}){
  if(this.busy)throw Error('Kicker table query is busy');
  if(!Array.isArray(scene?.triangles)||!Array.isArray(rays)||rays.length<1||rays.length>4096||scene.triangles.length>4096)throw Error('Invalid bounded table guide query');
  const check=()=>{if(signal?.aborted)throw Error('Kicker guide canceled');};check();
  const hits=rays.map(()=>null),width=this.law.manifest.capacity,feed=new Float32Array(width*15),count=scene.triangles.length;
  this.busy=true;
  try{
   for(let begin=0;begin<count*rays.length;begin+=width){
    check();const size=Math.min(width,count*rays.length-begin);
    for(let j=0;j<size;j++){
     const pair=begin+j,ray=rays[Math.floor(pair/count)],t=scene.triangles[pair%count];
     [...ray.origin,...ray.direction,...t.v0,...t.e1,...t.e2].forEach((x,k)=>feed[k*width+j]=x);
    }
    const [valid,distances]=this.law.run(feed,size,width);
    for(let j=0;j<size;j++){
     const pair=begin+j,index=Math.floor(pair/count),t=distances[j];
     if(![0,1].includes(valid[j])||(valid[j]&&!Number.isFinite(t)))throw Error('Invalid original-law table hit');
     if(valid[j]&&t>0&&t<rays[index].maximum&&(!hits[index]||t<hits[index].distance))hits[index]={...scene.triangles[pair%count],distance:t};
    }
    await new Promise(resolve=>setTimeout(resolve,0));
   }
   check();return {hits,scope:scene.scope,kernel:this.contract.query_kernel_sha256,sourceLaw:this.contract.laws.hit_tri};
  }finally{this.busy=false;}
 }
}

export function summarizeKickerSamples(hits){
 if(!Array.isArray(hits)||!hits.length)throw Error('No kicker samples');
 const crossed=hits.filter(Boolean).length,parts=[...new Set(hits.filter(Boolean).map(hit=>hit.part))];
 return {crossed,total:hits.length,state:crossed===0?'none':crossed===hits.length?'all':'some',parts,
  message:`${crossed}/${hits.length} sampled rays cross table surfaces${parts.length?' ('+parts.join(', ')+')':''}. This is sampled geometric coverage, not a full-cone or optical-transmission guarantee.`};
}
