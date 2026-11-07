// Exact signed tetrahedral integrals of closed triangle surfaces in unit radii.
// The shell displacement model assumes the separate inclusions do not overlap.
const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const dot = (a,b) => a.reduce((sum,x,i)=>sum+x*b[i],0);
const zero = () => [[0,0,0],[0,0,0],[0,0,0]];
export function principalMoments(tensor) {
  const a=tensor.map((row)=>row.slice()),vectors=[[1,0,0],[0,1,0],[0,0,1]];
  for(let iteration=0;iteration<40;iteration++) {
    let p=0,q=1;
    for(const [i,j] of [[0,1],[0,2],[1,2]]) if(Math.abs(a[i][j])>Math.abs(a[p][q])) [p,q]=[i,j];
    if(Math.abs(a[p][q])<1e-18)break;
    const angle=0.5*Math.atan2(2*a[p][q],a[q][q]-a[p][p]),c=Math.cos(angle),s=Math.sin(angle);
    const app=a[p][p],aqq=a[q][q],apq=a[p][q];
    a[p][p]=c*c*app-2*s*c*apq+s*s*aqq;
    a[q][q]=s*s*app+2*s*c*apq+c*c*aqq;
    a[p][q]=a[q][p]=0;
    for(let k=0;k<3;k++) if(k!==p&&k!==q) {
      const kp=a[k][p],kq=a[k][q];
      a[k][p]=a[p][k]=c*kp-s*kq;
      a[k][q]=a[q][k]=s*kp+c*kq;
    }
    for(let k=0;k<3;k++) {
      const vp=vectors[k][p],vq=vectors[k][q];
      vectors[k][p]=c*vp-s*vq;vectors[k][q]=s*vp+c*vq;
    }
  }
  return [0,1,2].map((i)=>({momentKgM2:a[i][i],axis:[vectors[0][i],vectors[1][i],vectors[2][i]]}))
    .sort((x,y)=>x.momentKgM2-y.momentKgM2);
}
function integrate(faces) {
  let volume=0; const first=[0,0,0], second=zero();
  for (const [a,b,c] of faces) {
    const vertices=[a,b,c], v=dot(a,cross(b,c))/6; volume+=v;
    for (let i=0;i<3;i++) {
      first[i]+=v*(a[i]+b[i]+c[i])/4;
      for (let j=0;j<3;j++) {
        const diagonal=vertices.reduce((s,p)=>s+p[i]*p[j],0);
        const sums=(a[i]+b[i]+c[i])*(a[j]+b[j]+c[j]);
        second[i][j]+=v*(diagonal+sums)/20;
      }
    }
  }
  return {volume,first,second};
}
function add(target, source, factor) {
  target.volume+=factor*source.volume;
  for(let i=0;i<3;i++) { target.first[i]+=factor*source.first[i];
    for(let j=0;j<3;j++) target.second[i][j]+=factor*source.second[i][j]; }
}
export function marbleMass(mesh, {radiusM,densityByRole,enamelMaterial,enamelThicknessM=0}) {
  if (!(radiusM>0) || !Number.isFinite(radiusM)) throw new RangeError("positive radius required");
  const total={volume:0,first:[0,0,0],second:zero()}, components=[];
  const outer=integrate(mesh.glass);
  const glassDensity=densityByRole.shell;
  if (!(glassDensity>0)) throw new RangeError("positive shell density required");
  add(total,outer,glassDensity);
  components.push({name:"shell",volumeM3:outer.volume*radiusM**3,massKg:outer.volume*radiusM**3*glassDensity});
  for (const part of [...mesh.ribbons.map((x)=>({...x,name:"ribbon"})),
    ...mesh.tubes.map((x)=>({...x,name:"tube"})),...mesh.confetti.map((x)=>({...x,name:"confetti"}))]) {
    const integral=integrate(mesh.interior.slice(part.start,part.start+part.count));
    const density=densityByRole[part.densityRole];
    if (!(density>0)) throw new RangeError("positive inclusion density required");
    add(total,integral,density-glassDensity);
    components[0].volumeM3-=integral.volume*radiusM**3;
    components[0].massKg-=integral.volume*radiusM**3*glassDensity;
    components.push({name:part.name,material:part.material,volumeM3:integral.volume*radiusM**3,massKg:integral.volume*radiusM**3*density});
  }
  let approximateCoating=false;
  if (enamelThicknessM>0 && enamelMaterial!==undefined) {
    const density=densityByRole.enamel;
    if (!(density>0)) throw new RangeError("positive enamel density required");
    let area=0;
    for(const [a,b,c,id] of mesh.glass) if(id===enamelMaterial) {
      const u=b.map((x,i)=>x-a[i]),v=c.map((x,i)=>x-a[i]);
      const faceArea=Math.hypot(...cross(u,v))/2; area+=faceArea;
      const center=a.map((x,i)=>(x+b[i]+c[i])/3);
      const massFactor=faceArea*(enamelThicknessM/radiusM)*density;
      total.volume+=massFactor;
      for(let i=0;i<3;i++) {total.first[i]+=massFactor*center[i];
        for(let j=0;j<3;j++) total.second[i][j]+=massFactor*center[i]*center[j]; }
    }
    if(area>0) approximateCoating=true;
    components.push({name:"enamel (thin-film estimate)",volumeM3:area*radiusM**2*enamelThicknessM,massKg:area*radiusM**2*enamelThicknessM*density});
  }
  const mass=total.volume*radiusM**3;
  const center=total.first.map((x)=>x*radiusM**4/mass);
  const inertia=zero();
  for(let i=0;i<3;i++) for(let j=0;j<3;j++) {
    const trace=total.second[0][0]+total.second[1][1]+total.second[2][2];
    inertia[i][j]=radiusM**5*((i===j?trace:0)-total.second[i][j])
      -mass*((i===j?dot(center,center):0)-center[i]*center[j]);
  }
  // The generator bounds ribbon cross-sections by their lane spacing, packs
  // confetti by disjoint enclosing spheres, confines hybrid dye to an inner
  // radial zone, and separates ring radii and spiral turns. Thus component
  // volume subtraction resolves the generated designs without a mesh union.
  const determinant = (a) => a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
    -a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
    +a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]);
  const unresolvedMark=mesh.contactModel==='polyhedron' && mesh.parameters.markingMode!=='flat' &&
    densityByRole.mark!==undefined && densityByRole.mark!==glassDensity;
  const massExact=!approximateCoating&&!unresolvedMark&&
    Number.isFinite(mass)&&mass>0&&center.every(Number.isFinite)&&
    inertia.flat().every(Number.isFinite)&&inertia[0][0]>0&&
    inertia[0][0]*inertia[1][1]-inertia[0][1]**2>0&&determinant(inertia)>0&&
    components.every((c)=>c.volumeM3>=0&&c.massKg>=0)&&
    mesh.interior.every((face)=>face.slice(0,3).every((p)=>Math.hypot(...p)<1));
  const valid=massExact&&mesh.contactModel!=='polyhedron';
  return {massKg:mass,centerM:center,inertiaKgM2:inertia,components,
    principalMoments:principalMoments(inertia),
    massExact,validForRoll:valid,mechanicsNote:unresolvedMark?"Relief uses body substrate; distinct marking bulk density is unresolved":
      mesh.contactModel==='polyhedron'?"Closed die mass and moments calculated for body substrate; rolling needs shape-aware contact dynamics":
      approximateCoating?"Enamel thin film uses an area estimate; rolling waits for closed coating geometry":
      valid?"Closed disjoint volumes integrated":"Final mesh has unresolved mechanical properties"};
}
