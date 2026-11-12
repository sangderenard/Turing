// Position views author existing world-space poses; they are not new saved state.
export const RADIAL_LIGHTS=['key-left','key-right','kicker-left','kicker-right','table-left','table-right'];
const radians=Math.PI/180;
export function radialLightPose(pos,id,center=[0,0,0]) {
 const d=pos.map((v,i)=>v-center[i]),stage=id.startsWith('table-');
 const side=id.endsWith('left')?-1:1,back=id.startsWith('kicker-');
 const horizontal=Math.hypot(d[0],d[2]);
 return {arc:Math.atan2(stage?d[0]:side*d[0],back?-d[2]:d[2])/radians,
  elevation:Math.atan2(d[1],horizontal)/radians,radius:stage?Math.hypot(...d):horizontal};
}
export function radialLightPosition(current,id,values,center=[0,0,0]) {
 const stage=id.startsWith('table-'),side=stage?1:id.endsWith('left')?-1:1,back=id.startsWith('kicker-');
 if(![values.arc,values.radius,...(stage?[values.elevation]:[])].every(Number.isFinite)||Math.abs(values.arc)>180||values.radius<0||(stage&&Math.abs(values.elevation)>90))throw Error('Invalid radial light coordinates');
 const angle=values.arc*radians,elevation=values.elevation*radians;
 const r=values.radius*(stage?Math.cos(elevation):1);
 const pos=[center[0]+side*r*Math.sin(angle),stage?center[1]+values.radius*Math.sin(elevation):current[1],center[2]+(back?-1:1)*r*Math.cos(angle)];
 if(pos.some(v=>Math.abs(v)>50))throw Error('Light position is outside -50 to 50 Studio units');
 return pos;
}
// Only called before a new rig's absolute pose owner is created. Loaded profiles
// never pass through this function. Geometry is fitted, not point centers alone.
export function wallRelativeFixtureDefaults(roles,bounds,vertices,subject=[0,0,0]) {
 const result=roles.map(r=>({...r,pos:[...r.pos],dir:[...r.dir]}));
 const aim=role=>{const d=subject.map((v,i)=>v-role.pos[i]),length=Math.hypot(...d);if(length>1e-9)role.dir=d.map(v=>v/length);};
 for(const index of [0,1,2,3,4,5]) {
  const role=result[index],rear=index>=2&&index<=4;
  const angle=index<=1?25:index===3||index===4?45:0,side=index===0||index===3?-1:1;
  const radius=(rear?subject[2]-bounds.backZ:bounds.frontZ-subject[2])-.05;
  if((index===3||index===4)&&role.round)role.defaultFloorHeight=-1-bounds.floorY+role.radius+.02;
  const originalY=role.defaultFloorHeight===undefined?role.pos[1]:bounds.floorY+role.defaultFloorHeight;
  const place=t=>{role.pos=[subject[0]+side*radius*t*Math.sin(angle*radians),originalY,subject[2]+(rear?-1:1)*radius*t*Math.cos(angle*radians)];
    if(index!==2)aim(role);
    if(index>=2&&role.defaultFloorHeight===undefined)for(let n=0;n<12;n++){const low=Math.min(...vertices(role).map(p=>p[1]));const shift=bounds.floorY+.02-low;role.pos[1]+=shift;if(index!==2)aim(role);if(Math.abs(shift)<1e-10)break;}
  };
  const fits=()=>vertices(role).every(p=>p[0]>=bounds.leftX+.02&&p[0]<=bounds.rightX-.02&&p[2]>=bounds.backZ+.02&&p[2]<=bounds.frontZ-.02);
  place(1);
  if(!fits()) {let low=0,high=1;for(let n=0;n<40;n++){const mid=(low+high)/2;place(mid);if(fits())low=mid;else high=mid;}place(low);if(!fits())throw Error('Fixture does not fit the room at its authored size');}
 }
 return result;
}
