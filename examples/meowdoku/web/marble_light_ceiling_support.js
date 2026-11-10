import {LIGHT_STAND_DIMENSIONS,pipeFaces} from './marble_light_stands.js';

// Derived support only. The existing stand/profile owner supplies the same
// shared catalogue material and the existing fixture builder's actual vertices.
export const LIGHT_CEILING_SUPPORT_DIMENSIONS = Object.freeze({
  cableRadius:.0125, attachmentRadius:.025, ceilingMountRadius:.075,
  ceilingMountDepth:.035, clearance:.01, mountInset:.6,
});
const add=(a,b,s=1)=>a.map((v,i)=>v+s*b[i]);
const dot=(a,b)=>a.reduce((sum,v,i)=>sum+v*b[i],0);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit=v=>{const n=Math.hypot(...v);if(!(n>0))throw new RangeError('Ceiling support direction must be nonzero');return v.map(x=>x/n);};
const vector=v=>Array.isArray(v)&&v.length===3&&v.every(Number.isFinite);
const positive=v=>Number.isFinite(v)&&v>0;

function frameFor(role) {
  if(!role||role.round||!vector(role.pos)||!vector(role.dir)||!positive(role.halfW)||!positive(role.halfH))
    throw new RangeError('Ceiling support requires a rectangular fixture with a finite pose and positive dimensions');
  const dir=unit(role.dir),right=unit(cross(dir,Math.abs(dir[1])>.95?[1,0,0]:[0,1,0]));
  const up=unit(cross(right,dir)),depth=role.housingDepth??Math.max(role.halfW,role.halfH)*1.4;
  if(!positive(depth))throw new RangeError('Invalid ceiling support housing depth');
  return {dir,right,up,depth,halfW:role.halfW*1.25,halfH:role.halfH*1.25};
}
function validateRoom(bounds,ceilingY) {
  if(!Number.isFinite(ceilingY))throw new RangeError('Invalid ceiling support height');
  if(!bounds||!['leftX','rightX','backZ','frontZ','floorY','ceilingY'].every(k=>Number.isFinite(bounds[k]))||
     bounds.ceilingY!==ceilingY||bounds.leftX>=bounds.rightX||bounds.backZ>=bounds.frontZ||bounds.floorY>=bounds.ceilingY)
    throw new RangeError('Ceiling support bounds must describe the same room ceiling');
}
const limits=bounds=>[[0,bounds.leftX,bounds.rightX],[1,bounds.floorY,bounds.ceilingY],[2,bounds.backZ,bounds.frontZ]];
function fitsRoom(segment,bounds) {
  const axis=unit(add(segment.end,segment.start,-1));
  return limits(bounds).every(([i,low,high])=>{
    const extent=segment.radius*Math.sqrt(Math.max(0,1-axis[i]*axis[i]));
    return Math.min(segment.start[i],segment.end[i])-extent>=low&&
      Math.max(segment.start[i],segment.end[i])+extent<=high;
  });
}

/** Four cables anchored to an upward-facing rear/side of the real softbox.
 * Actual fixture vertices include its emitter, housing, grid and any tools.
 * The cable/ceiling-mount cylinders stay outside one support plane of those
 * vertices. Short attachment pins intentionally meet that housing surface and
 * remain behind its emitting plane. This is not a general collision check:
 * other fixtures, table/skirt/props, component unions and mount strength are
 * unverified. An unavailable layout has no segments; the UI must show reason. */
export function lightCeilingSupportLayout(role,ceilingY,{bounds,fixtureVertices}={}) {
  validateRoom(bounds,ceilingY);
  const frame=frameFor(role),{dir,right,up,depth,halfW,halfH}=frame;
  if(!Array.isArray(fixtureVertices)||!fixtureVertices.length||!fixtureVertices.every(vector))
    throw new RangeError('Ceiling support requires the actual fixture vertices as finite XYZ points');
  const vertices=fixtureVertices,footprintSource='fixture-vertices';
  const unavailable=reason=>({available:false,reason,segments:[],ceilingY,footprintSource});
  const fixtureTop=Math.max(...vertices.map(p=>p[1]));
  if(fixtureTop>=ceilingY)
    return unavailable('The softbox or its tools reach the ceiling. Lower the fixture, reduce its size, or enlarge the room.');
  if(vertices.some(p=>limits(bounds).some(([i,low,high])=>p[i]<low||p[i]>high)))
    return unavailable('The softbox or its tools extend outside this room. Move or resize the fixture, or enlarge the room.');

  const dimensions=LIGHT_CEILING_SUPPORT_DIMENSIONS;
  const attachmentRadius=Math.min(dimensions.attachmentRadius,depth/12,halfW/12,halfH/12);
  const ratio=attachmentRadius/dimensions.attachmentRadius;
  const cableRadius=dimensions.cableRadius*ratio,mountRadius=dimensions.ceilingMountRadius*ratio;
  const mountDepth=dimensions.ceilingMountDepth*ratio,clearance=dimensions.clearance*ratio;
  const center=add(role.pos,dir,-depth/2),back=dir.map(x=>-x);
  const surfaces=[
    {id:'back',normal:back,center:add(role.pos,dir,-depth),u:right,v:up,halfU:halfW,halfV:halfH},
    ...[[right,halfW,up,halfH,'right'],[up,halfH,right,halfW,'top']].flatMap(([normal,offset,v,halfV,name])=>
      [1,-1].map(sign=>({id:sign===1?name:`opposite-${name}`,normal:normal.map(x=>x*sign),
        center:add(center,normal,offset*sign),u:dir,v,halfU:depth/2,halfV}))),
  ].filter(surface=>surface.normal[1]>0).sort((a,b)=>b.normal[1]-a.normal[1]);
  if(!surfaces.length)
    return unavailable('No rear or side housing face points toward the ceiling. Aim the softbox downward or sideways to attach its cables behind the aperture.');

  for(const surface of surfaces) {
    const support=Math.max(...vertices.map(p=>dot(p,surface.normal)));
    // The complete plate as well as the thinner cable remains outside the
    // actual fixture's supporting plane, not merely outside its centreline.
    const offset=Math.max(0,support-dot(surface.center,surface.normal))+mountRadius+clearance;
    const segments=[],mounts=[];
    for(const [i,[a,b]]of [[-1,-1],[1,-1],[1,1],[-1,1]].entries()) {
      const mount=add(add(surface.center,surface.u,a*surface.halfU*dimensions.mountInset),
        surface.v,b*surface.halfV*dimensions.mountInset);
      const cableStart=add(mount,surface.normal,offset);
      const ceilingMountStart=[cableStart[0],ceilingY-mountDepth,cableStart[2]];
      const ceilingMountEnd=[cableStart[0],ceilingY,cableStart[2]];
      mounts.push({housing:mount,cableStart,ceiling:ceilingMountEnd});
      segments.push({id:`attachment-${i}`,start:mount,end:cableStart,radius:attachmentRadius},
        {id:`cable-${i}`,start:cableStart,end:ceilingMountStart,radius:cableRadius},
        {id:`ceiling-mount-${i}`,start:ceilingMountStart,end:ceilingMountEnd,radius:mountRadius});
    }
    if(mounts.some(mount=>mount.cableStart[1]>=ceilingY-mountDepth))continue;
    if(!segments.every(segment=>fitsRoom(segment,bounds)))continue;
    return {available:true,segments,mounts,ceilingY,fixtureTop,fixtureCeilingClearance:ceilingY-fixtureTop,
      footprintSource,housingSurface:surface.id,exclusionDirection:[...surface.normal],fixtureSupport:support};
  }
  return unavailable('The ceiling cables and mounts do not fit above this softbox. Lower or move the fixture, reduce its size, or enlarge the room.');
}

/** Standard Studio triangle tuples; material is the caller-resolved shared
 * catalogue ID. Preview and ray geometry must call this same face builder. */
export function lightCeilingSupportFaces(role,ceilingY,material,onComponent,options={}) {
  if(!Number.isSafeInteger(material)||material<0)throw new RangeError('Unknown ceiling support material ID');
  const layout=lightCeilingSupportLayout(role,ceilingY,options);
  if(!layout.available)throw new RangeError(layout.reason);
  const faces=[];
  for(const segment of layout.segments) {
    const firstTriangle=faces.length,part=pipeFaces(segment,material,role.mat??0);
    faces.push(...part);
    onComponent?.({id:segment.id,firstTriangle,triangleCount:part.length,sourceConstruction:'lightCeilingSupportPipe',
      intendedShape:'capped-polygonal-cylinder',constructionArguments:{start:[...segment.start],end:[...segment.end],
        radius:segment.radius,segments:LIGHT_STAND_DIMENSIONS.segments},contactAdmission:'unverified'});
  }
  return faces;
}
