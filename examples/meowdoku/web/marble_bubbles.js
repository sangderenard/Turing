import {catEyeMarble} from './marble_mesh.js';

export const MAX_BUBBLES = 64;
const finiteVector = value => Array.isArray(value) && value.length === 3 && value.every(Number.isFinite);
const distance = (a,b) => Math.hypot(...a.map((x,i)=>x-b[i]));

// Conservative enclosing spheres keep air out of the existing closed inclusion
// volumes. Call on canonical full geometry, never on view-dependent LODs.
export function inclusionExclusions(mesh) {
  const exclusions = [];
  for (const part of [...(mesh.ribbons || []), ...(mesh.tubes || []), ...(mesh.confetti || [])]) {
    const faces = mesh.interior.slice(part.start, part.start + part.count);
    if (!faces.length || faces.length !== part.count) throw new RangeError('invalid inclusion membership');
    const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
    for (const face of faces) for (const p of face.slice(0,3)) {
      if (!finiteVector(p)) throw new RangeError('nonfinite inclusion vertex');
      for(let k=0;k<3;k++) {lo[k]=Math.min(lo[k],p[k]);hi[k]=Math.max(hi[k],p[k]);}
    }
    const center=lo.map((x,k)=>(x+hi[k])/2);
    let radius=0;
    for (const face of faces) for (const p of face.slice(0,3)) radius=Math.max(radius,distance(p,center));
    exclusions.push({center,radius});
  }
  return exclusions;
}

/** Seeded, prefix-stable disjoint air spheres in marble-radius units.
 * Count 0 is the legacy/default. No controls, materials, storage, jobs or
 * geometry are mutated. Conservative exclusions may reject crowded designs.
 */
export function generateAirBubbles({count=0,seed=0,exclusions=[],interiorRadius=0.94,
  minRadius=0.018,maxRadius=0.08,gap=0.004}={}) {
  if (!Number.isInteger(count) || count<0 || count>MAX_BUBBLES) throw new RangeError('bubble count must be 0..64');
  if (!Number.isInteger(seed) || seed<0 || seed>0x7fffffff) throw new RangeError('invalid bubble seed');
  if (![interiorRadius,minRadius,maxRadius,gap].every(Number.isFinite) ||
      !(interiorRadius>0 && interiorRadius<1 && minRadius>0 && minRadius<=maxRadius &&
        maxRadius+gap<interiorRadius && gap>=0)) throw new RangeError('invalid bubble dimensions');
  if (!Array.isArray(exclusions) || exclusions.some(b=>!b || !finiteVector(b.center) || !Number.isFinite(b.radius) || b.radius<0))
    throw new RangeError('invalid closed inclusion exclusion');
  // Independent stream: increasing bubble count does not reroll existing
  // bubbles, and changing bubbles never consumes the confetti random stream.
  let state=(seed ^ 0x61c88647)>>>0;
  const random=()=>((state=(Math.imul(state,1664525)+1013904223)>>>0)/0x100000000);
  const result=[];
  for(let attempts=0;result.length<count;attempts++) {
    if(attempts>=200000) throw new RangeError('requested air bubbles cannot fit without overlap');
    const radius=minRadius+(maxRadius-minRadius)*random();
    const center=[0,1,2].map(()=>interiorRadius*(2*random()-1));
    if(Math.hypot(...center)+radius+gap>interiorRadius)continue;
    if([...exclusions,...result].some(b=>distance(center,b.center)<radius+b.radius+gap))continue;
    result.push({id:`bubble-${result.length}`,center,radius,positiveMedium:'shell',negativeMedium:'air'});
  }
  return result;
}

/** Use the existing sphere generator for closed air-cavity boundaries.
 * The shell material supplies surface lookup; the saved explicit medium map
 * defines the real glass-to-air transition rather than a solid glass ball.
 */
export function buildBubbleBoundaries(bubbles,{material,rings=12,segments=24}={}) {
  if(!Number.isInteger(material)||material<0)throw new RangeError('boundary material index required');
  const sphere=catEyeMarble({glass:material,eye:material,accent:material},{kind:'clear',rings,segments}).glass;
  const faces=[],boundaries=[];
  for(const bubble of bubbles) {
    const start=faces.length;
    for(const [a,b,c,id,normals] of sphere) {
      const point=p=>p.map((x,k)=>bubble.center[k]+bubble.radius*x);
      faces.push([point(a),point(b),point(c),id,normals.map(n=>n.slice())]);
    }
    boundaries.push({...bubble,center:bubble.center.slice(),start,count:faces.length-start});
  }
  return {faces,boundaries};
}

export function bubbleCount(design) {
  const count=design['bubble-count'] ?? 0;
  if (!Number.isInteger(count)||count<0||count>MAX_BUBBLES) throw new RangeError('Air bubbles must be an integer from 0 to 64.');
  return count;
}
export function validateBubbleHost(design, shell, pbr) {
  if (!bubbleCount(design)) return;
  const row=pbr.data.slice(shell*pbr.stride,(shell+1)*pbr.stride);
  if (row.length<11 || ![row[4],row[5],row[6],row[7],...row.slice(8,11)].every(Number.isFinite) || row[4]!==0 || row[5]<=0 || row[7]>=0.999 || row[6]<=1 || row.slice(8,11).some(x=>x!==0))
    throw new RangeError('Air bubbles require a transparent, nonmetal, nonemissive glass host.');
}
export function appendAirBubbles(mesh,canonical,design,shell) {
  const count=bubbleCount(design);
  if (!count) return mesh;
  if(design['base-shape']!=='marble'||+design.coverage!==0||+design['band-count']!==0)
    throw new RangeError('Air bubbles currently require a spherical marble without enamel or painted bands.');
  const opticalVolumes=design.optical_authoring?.version===1 &&
    ['glass_rod','injection_dye'].includes(design.optical_authoring.mode);
  if(canonical.mechanicsResolved===false && !opticalVolumes)
    throw new RangeError('Air bubbles require disjoint inclusions. Choose separated ribbon lanes instead of overlapping axial ribbons.');
  const bubbles=generateAirBubbles({count,seed:+design.seed,exclusions:opticalVolumes?[]:inclusionExclusions(canonical)});
  const built=buildBubbleBoundaries(bubbles,{material:shell});
  const offset=mesh.interior.length;
  mesh.bubbles=built.boundaries.map(b=>({...b,start:b.start+offset,material:shell,densityRole:'air'}));
  mesh.interior.push(...built.faces);
  const ranges=[...(mesh.ribbons||[]),...(mesh.tubes||[]),...(mesh.confetti||[])].map(p=>({start:p.start,count:p.count,positive:shell,negative:p.material}));
  ranges.push(...mesh.bubbles.map(p=>({start:p.start,count:p.count,positive:shell,negative:-1})));
  ranges.push({start:mesh.interior.length,count:mesh.glass.length,positive:-1,negative:shell});
  mesh.boundaryMedia={version:1,ranges:ranges.sort((a,b)=>a.start-b.start)};
  mesh.triangleCount=mesh.interior.length+mesh.glass.length;
  if(opticalVolumes && ((mesh.ribbons?.length||0)+(mesh.tubes?.length||0)+(mesh.confetti?.length||0))) {
    mesh.mechanicsResolved=false;
    mesh.mechanicsNote='Optical cut volumes are resolved by ray transport; shared-volume mechanical mass integration remains approximate.';
  }
  return mesh;
}
// Phong cannot model refraction at a cavity. Never render a substitute ball.
export function phongInterior(mesh) {
  if(!mesh.bubbles?.length)return mesh.interior;
  return mesh.interior.filter((face,i)=>!mesh.bubbles.some(b=>i>=b.start&&i<b.start+b.count));
}

export function validateBubbleInclusions(mesh,pbr) {
  if(!mesh.bubbles?.length)return;
  const ids=new Set([...(mesh.ribbons||[]),...(mesh.tubes||[]),...(mesh.confetti||[])].map(p=>p.material));
  for(const id of ids) {
    const row=pbr.data.slice(id*pbr.stride,(id+1)*pbr.stride);
    if(row[7]<0.999 && (row.length<11 || !row.slice(4,11).every(Number.isFinite) || !(row[5]>0) || row[4]!==0 || !(row[6]>=1) || row.slice(8,11).some(x=>x!==0)))
      throw new RangeError('Transmissive inclusions with air bubbles must be nonmetal and nonemissive, with IOR at least 1.');
  }
}

export function validateBubbleRadius(design,radiusMM) {
  if(bubbleCount(design)&&(!Number.isFinite(radiusMM)||radiusMM<1))
    throw new RangeError('Air bubbles require a physical marble radius of at least 1 mm.');
}
