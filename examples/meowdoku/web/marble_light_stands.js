// Derived support geometry only. The caller owns saved enable/material state,
// the room floor, and the shared catalogue-to-photo material remap.
export const DEFAULT_LIGHT_STAND_MATERIAL = 'black_matte_paint';
export const LIGHT_STAND_DIMENSIONS = Object.freeze({radius:.035, footRadius:.025, spread:.45, clearance:.02, segments:12});

const add = (a,b,s=1) => a.map((v,i)=>v+s*b[i]);
const dot = (a,b) => a.reduce((n,v,i)=>n+v*b[i],0);
const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit = v => {const n=Math.hypot(...v); if(!(n>0))throw new RangeError('Stand direction must be nonzero'); return v.map(x=>x/n);};
const vector = v => Array.isArray(v)&&v.length===3&&v.every(Number.isFinite);
const positive = v => Number.isFinite(v)&&v>0;
function fixtureFrame(role) {
  if(!role||!vector(role.pos)||!vector(role.dir)||
     !(role.round?positive(role.radius):positive(role.halfW)&&positive(role.halfH)))
    throw new RangeError('Invalid light stand fixture pose or dimensions');
  const dir=unit(role.dir),right=unit(cross(dir,Math.abs(dir[1])>.95?[1,0,0]:[0,1,0]));
  const up=unit(cross(right,dir)),size=role.round?role.radius:Math.max(role.halfW,role.halfH);
  const depth=role.housingDepth??size*1.4;
  if(!positive(depth))throw new RangeError('Invalid light stand housing depth');
  return {dir,right,up,depth,halfW:(role.round?role.radius:role.halfW)*1.25,
    halfH:(role.round?role.radius:role.halfH)*1.25};
}

// A conservative fallback for callers without actual mesh vertices. Production
// should pass the can/emitter/tool vertices from its existing shared builder.
function fixtureEnvelope(role,frame) {
  let {halfW,halfH}=frame,front=0;
  if(role.gobo) {
    const radius=Math.max(role.shroudRadius??role.radius*1.1,role.shroudExitRadius??role.shroudRadius??role.radius*1.1);
    halfW=Math.max(halfW,radius);halfH=Math.max(halfH,radius);
    front=Math.max(front,(role.shroudOffset??0)+(role.shroudLength??role.radius*2.2));
  }
  if(role.grid) {
    halfW=Math.max(halfW,role.halfW*1.25+.002);halfH=Math.max(halfH,role.halfH*1.25+.002);
    front=Math.max(front,(role.gridOffset??0)+(role.gridDepth??.2)+.002);
  }
  if(![halfW,halfH,front].every(Number.isFinite))throw new RangeError('Invalid light stand attachment bounds');
  const points=[];
  for(const x of [-halfW,halfW])for(const y of [-halfH,halfH])for(const z of [-frame.depth,front])
    points.push(add(add(add(role.pos,frame.right,x),frame.up,y),frame.dir,z));
  return points;
}

function validateBounds(bounds,floorY) {
  if(bounds===undefined)return;
  if(!bounds||!['leftX','rightX','backZ','frontZ','floorY','ceilingY'].every(k=>Number.isFinite(bounds[k]))||
     bounds.floorY!==floorY||bounds.leftX>=bounds.rightX||bounds.backZ>=bounds.frontZ||bounds.floorY>=bounds.ceilingY)
    throw new RangeError('Light stand bounds must describe the same room floor');
}

// Exact analytic cylinder support bounds are conservative for the inscribed
// twelve-sided mesh. Horizontal feet explicitly own their floor contact plane.
function fitsRoom(segment,bounds) {
  if(!bounds)return true;
  const axis=unit(add(segment.end,segment.start,-1));
  for(const [i,low,high]of [[0,bounds.leftX,bounds.rightX],[2,bounds.backZ,bounds.frontZ],[1,bounds.floorY,bounds.ceilingY]]) {
    const extent=segment.radius*Math.sqrt(Math.max(0,1-axis[i]*axis[i]));
    if(Math.max(segment.start[i],segment.end[i])+extent>high)return false;
    if(!(i===1&&segment.id.startsWith('foot-'))&&Math.min(segment.start[i],segment.end[i])-extent<low)return false;
  }
  return true;
}

/** Return a derived layout without editing a fixture/profile. An unavailable
 * layout has no segments and a reason the UI must surface. Room fitting checks
 * only this stand and the provided fixture footprint, not table/skirt/objects,
 * other lights, or the mechanical stability of the combined assembly. */
export function lightStandLayout(role,floorY,{bounds,fixtureVertices,supportDirections}={}) {
  if(!Number.isFinite(floorY))throw new RangeError('Invalid light stand floor');
  validateBounds(bounds,floorY);
  const frame=fixtureFrame(role),{dir,right,up,depth,halfW,halfH}=frame;
  if(fixtureVertices!==undefined&&(!Array.isArray(fixtureVertices)||!fixtureVertices.length||!fixtureVertices.every(vector)))
    throw new RangeError('Light stand fixture vertices must be finite XYZ points');
  const vertices=fixtureVertices??fixtureEnvelope(role,frame);
  const footprintSource=fixtureVertices===undefined?'conservative-fixture-envelope':'fixture-vertices';
  const unavailable=reason=>({available:false,reason,segments:[],floorY,footprintSource});
  // Small authored fixtures keep a pipe thinner than both housing depth and
  // aperture. Never resize the fixture or silently lift it off the saved pose.
  const radius=Math.min(LIGHT_STAND_DIMENSIONS.radius,depth/4,Math.min(halfW,halfH)/5);
  const footRadius=Math.min(LIGHT_STAND_DIMENSIONS.footRadius,radius*.7),spread=LIGHT_STAND_DIMENSIONS.spread;
  const center=add(role.pos,dir,-depth/2);
  if(center[1]-radius<floorY||center[1]<=floorY+footRadius)
    return unavailable('The housing mount is too low for a floor stand. Raise the fixture or disable its stand.');
  const horizontal=Math.hypot(dir[0],dir[2]);
  let directions;
  if(horizontal===0)directions=[[1,0,0],[-1,0,0],[0,0,1],[0,0,-1]];
  else {
    const side=[dir[2]/horizontal,0,-dir[0]/horizontal],back=[-dir[0]/horizontal,0,-dir[2]/horizontal];
    directions=[side,side.map(v=>-v),back,unit(add(side,back)),unit(add(back,side,-1))];
  }
  if(supportDirections!==undefined){
    if(!Array.isArray(supportDirections)||!supportDirections.length||supportDirections.length>8||
       supportDirections.some(d=>!vector(d)||d[1]!==0||Math.hypot(...d)===0||dot(d,dir)>0))
      throw new RangeError('Stand support directions must be horizontal side/rear vectors');
    directions=supportDirections.map(unit);
  }
  const candidates=[];
  for(const direction of directions) {
    // Exit the actual can along a horizontal side/rear ray at its mid-depth.
    // This arm stays behind the emitting plane, including downward/upward aim.
    const axial=dot(direction,dir),across=dot(direction,right),vertical=dot(direction,up);
    const radial=role.round?Math.hypot(across,vertical):0;
    const exit=Math.min(axial<0?depth/(-2*axial):Infinity,
      role.round?(radial>0?halfW/radial:Infinity):Math.min(across===0?Infinity:halfW/Math.abs(across),vertical===0?Infinity:halfH/Math.abs(vertical)));
    const mount=add(center,direction,exit);
    const support=Math.max(...vertices.map(p=>dot(add(p,center,-1),direction)),exit);
    // The complete tripod, not just the mast centre, stays outside the supplied
    // horizontal support plane. Its feet cannot cross this fixture's aperture.
    const distance=support+spread+footRadius+LIGHT_STAND_DIMENSIONS.clearance;
    const mastTop=add(center,direction,distance),mastBottom=[mastTop[0],floorY+footRadius,mastTop[2]];
    for(let rotation=0;rotation<6;rotation++) {
      const segments=[{id:'arm',start:mount,end:mastTop,radius},{id:'mast',start:mastBottom,end:mastTop,radius}];
      const angle=Math.atan2(direction[2],direction[0])+rotation*Math.PI/6;
      for(let i=0;i<3;i++)segments.push({id:`foot-${i}`,start:[...mastBottom],
        end:add(mastBottom,[Math.cos(angle+i*2*Math.PI/3),0,Math.sin(angle+i*2*Math.PI/3)],spread),radius:footRadius});
      if(segments.every(segment=>fitsRoom(segment,bounds)))candidates.push({available:true,segments,mount,mastTop,mastBottom,
        floorY,footprintSource,exclusionDirection:direction,fixtureSupport:support,armLength:distance-exit});
    }
  }
  candidates.sort((a,b)=>a.armLength-b.armLength);
  return candidates[0]??unavailable('No tested side or rear floor-stand placement fits this room. Enlarge the room, move the fixture, or disable its stand.');
}

// Closed capped pipe, with shared ring points at the seam and outward winding.
// The existing fixture helpers are open can/snoot surfaces, not capped supports.
// Explicit smooth side normals and ray winding describe the same outside.
export function pipeFaces(segment,material,group) {
  const {start,end,radius}=segment,axis=unit(add(end,start,-1));
  const right=unit(cross(Math.abs(axis[1])>.95?[1,0,0]:[0,1,0],axis));
  const up=unit(cross(axis,right)),count=LIGHT_STAND_DIMENSIONS.segments;
  const normals=Array.from({length:count},(_,i)=>add(right.map(v=>v*Math.cos(i*2*Math.PI/count)),up,Math.sin(i*2*Math.PI/count)));
  const a=normals.map(n=>add(start,n,radius)),b=normals.map(n=>add(end,n,radius));
  const back=axis.map(v=>-v),faces=[];
  for(let i=0;i<count;i++) {
    const j=(i+1)%count;
    faces.push([a[i],a[j],b[j],material,[normals[i],normals[j],normals[j]],group],
      [a[i],b[j],b[i],material,[normals[i],normals[j],normals[i]],group],
      [start,a[j],a[i],material,[back,back,back],group],
      [end,b[i],b[j],material,[axis,axis,axis],group]);
  }
  return faces;
}

/** Material is a caller-resolved shared catalogue ID, never an emitter slot.
 * Use the same call for preview, ray geometry, and source-component metadata. */
export function lightStandFaces(role,floorY,material,onComponent,options={}) {
  if(!Number.isSafeInteger(material)||material<0)throw new RangeError('Unknown light stand material ID');
  const layout=lightStandLayout(role,floorY,options);
  if(!layout.available)throw new RangeError(layout.reason);
  const faces=[];
  for(const segment of layout.segments) {
    const firstTriangle=faces.length,part=pipeFaces(segment,material,role.mat??0);
    faces.push(...part);
    onComponent?.({id:segment.id,firstTriangle,triangleCount:part.length,sourceConstruction:'lightStandPipe',
      intendedShape:'capped-polygonal-cylinder',constructionArguments:{start:[...segment.start],end:[...segment.end],
        radius:segment.radius,segments:LIGHT_STAND_DIMENSIONS.segments},contactAdmission:'unverified'});
  }
  return faces;
}
