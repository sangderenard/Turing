// Exact signed tetrahedral integrals of closed triangle surfaces in unit radii.
// The shell displacement model assumes the separate inclusions do not overlap.
const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const dot = (a,b) => a.reduce((sum,x,i)=>sum+x*b[i],0);
const zero = () => [[0,0,0],[0,0,0],[0,0,0]];
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
    components.push({name:"enamel",volumeM3:area*radiusM**2*enamelThicknessM,massKg:area*radiusM**2*enamelThicknessM*density});
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
  const valid=Number.isFinite(mass)&&mass>0&&center.every(Number.isFinite)&&
    inertia.flat().every(Number.isFinite)&&inertia[0][0]>0&&
    inertia[0][0]*inertia[1][1]-inertia[0][1]**2>0&&determinant(inertia)>0&&
    components.every((c)=>c.volumeM3>=0&&c.massKg>=0);
  return {massKg:mass,centerM:center,inertiaKgM2:inertia,components,validForRoll:valid};
}
