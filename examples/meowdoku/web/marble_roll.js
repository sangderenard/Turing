// Flat-plane no-slip rigid-body step for the final triangulated marble.
// Unknown center acceleration, angular acceleration, contact friction and
// normal force are solved together from COM force, full world inertia tensor,
// angular momentum, and the two tangential contact constraints.
const add=(a,b)=>a.map((x,i)=>x+b[i]);
const sub=(a,b)=>a.map((x,i)=>x-b[i]);
const scale=(a,s)=>a.map((x)=>x*s);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const matvec=(a,b)=>a.map((row)=>row.reduce((sum,x,i)=>sum+x*b[i],0));
function solve(matrix,values) {
  const rows=matrix.map((r,i)=>[...r,values[i]]),n=values.length;
  for(let c=0;c<n;c++) {
    let pivot=c;for(let r=c+1;r<n;r++) if(Math.abs(rows[r][c])>Math.abs(rows[pivot][c]))pivot=r;
    if(Math.abs(rows[pivot][c])<1e-18) throw new RangeError("singular marble contact system");
    [rows[c],rows[pivot]]=[rows[pivot],rows[c]];
    const divisor=rows[c][c];for(let j=c;j<=n;j++)rows[c][j]/=divisor;
    for(let r=0;r<n;r++)if(r!==c) {const factor=rows[r][c];for(let j=c;j<=n;j++)rows[r][j]-=factor*rows[c][j];}
  }
  return rows.map((row)=>row[n]);
}
function quatMultiply(a,b) {return [a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
  a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
  a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
  a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]];}
const unitQuat=(q)=>scale(q,1/Math.hypot(...q));
export function rotationMatrix(q) {
  const [w,x,y,z]=q;
  return [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
    [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
    [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]];
}
export function rollingStep(state,properties,forceN,dt,radiusM) {
  if(!properties.validForRoll || !(dt>0) || !(radiusM>0))throw new RangeError("validated mass, inertia, radius and time required");
  const R=rotationMatrix(state.orientation),c=matvec(R,properties.centerM),m=properties.massKg;
  const I=R.map((row,i)=>row.map((_,j)=>
    R[i].reduce((sum,ri,k)=>sum+ri*properties.inertiaKgM2[k].reduce((s,x,l)=>s+x*R[j][l],0),0)));
  const omega=state.angularVelocity,contact=[0,-radiusM,0],r=sub(contact,c);
  const centripetal=cross(omega,cross(omega,c));
  const contactCentripetal=cross(omega,cross(omega,contact));
  const gyro=cross(omega,matvec(I,omega));
  function equations(q) {
    const [ax,az,alphax,alphay,alphaz,fx,fz,N]=q;
    const alpha=[alphax,alphay,alphaz],aCenter=[ax,0,az],friction=[fx,N,fz];
    const aCom=add(add(aCenter,cross(alpha,c)),centripetal);
    const linear=sub(scale(aCom,m),add(forceN,add(friction,[0,-m*9.80665,0])));
    const angular=sub(add(matvec(I,alpha),gyro),cross(r,friction));
    const rolling=add(add(aCenter,cross(alpha,contact)),contactCentripetal);
    return [...linear,...angular,rolling[0],rolling[2]];
  }
  const zero=Array(8).fill(0),bias=equations(zero);
  const columns=zero.map((_,i)=>{const basis=zero.slice();basis[i]=1;return sub(equations(basis),bias);});
  const matrix=zero.map((_,row)=>columns.map((col)=>col[row]));
  const q=solve(matrix,scale(bias,-1)),alpha=q.slice(2,5);
  const nextOmega=add(omega,scale(alpha,dt));
  const delta=unitQuat([1,...scale(nextOmega,dt/2)]);
  const orientation=unitQuat(quatMultiply(delta,state.orientation));
  const velocity=scale(cross(nextOmega,contact),-1);
  const position=add(state.position,scale(velocity,dt));
  return {position,orientation,angularVelocity:nextOmega,angularAcceleration:alpha,
    contactForce:[q[5],q[7],q[6]]};
}
