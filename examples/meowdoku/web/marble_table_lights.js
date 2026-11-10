import {TABLE_TOP_Y, TABLE_THICKNESS} from './marble_studio_table.js';

// One footprint definition serves both the real stand and its placement warning.
export const TABLE_STAND = {halfWidth: .14, halfLength: .12, baseHalfHeight: .01, baseOffset: .27};

// These three additions have their own material and pose identities. They are
// appended after the original seven so older profiles retain every old slot.
export function tableLightRoles(materialBase = 0) {
  const aimed = pos => { const n = Math.hypot(...pos); return pos.map(v => -v / n); };
  return [-1, 1].map((side, i) => {
    const pos = [side * 1.85, TABLE_TOP_Y + .28, .10];
    return {pos, dir: aimed(pos), radius: .09, round: true, shape: 'snoot',
      gobo: true, shroudRadius: .1125, shroudLength: .36, tableStand: true, roleIndex: 7 + i, mat: materialBase + 7 + i};
  }).concat([{pos: [0, TABLE_TOP_Y - TABLE_THICKNESS - .48, 0], dir: [0, 1, 0],
    halfW: .9, halfH: 1.2, round: false, shape: 'grid', grid: true,
    housingDepth: .14, roleIndex: 9, mat: materialBase + 9}]);
}

const add = (a, b, s = 1) => a.map((v, i) => v + b[i] * s);
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const unit = v => { const n = Math.hypot(...v); return v.map(x => x / n); };
function basis(dir) {
  const right = unit(cross(dir, Math.abs(dir[1]) > .95 ? [1,0,0] : [0,1,0]));
  return [right, unit(cross(right, dir)), dir];
}

// Advisory only: never repair, clamp or relocate an authored fixture/profile.
// The caller supplies the same scaled subject bound used by preview/photo.
export function studioPlacementWarnings(table, poses, bounds, subject = {center: [0,0,0], radius: 1}) {
  if (!Array.isArray(subject.center) || subject.center.length !== 3 ||
      !subject.center.every(Number.isFinite) || !Number.isFinite(subject.radius) || subject.radius <= 0)
    throw new RangeError('Invalid subject clearance bound');
  const warnings = {};
  const append = (id, message) => {warnings[id] = [warnings[id], message].filter(Boolean).join(' ');};
  if (subject.center[0] - subject.radius < bounds.leftX || subject.center[0] + subject.radius > bounds.rightX ||
      subject.center[2] - subject.radius < bounds.backZ || subject.center[2] + subject.radius > bounds.frontZ ||
      subject.center[1] - subject.radius < bounds.floorY || subject.center[1] + subject.radius > bounds.ceilingY)
    append('table', 'Marble clearance extends beyond the room. Reduce marble size or enlarge the room before rendering.');
  if (table.enabled && (Math.abs(subject.center[0]) + subject.radius > table.width / 2 ||
      Math.abs(subject.center[2]) + subject.radius > table.length / 2))
    append('table', 'Marble clearance overhangs the tabletop. Check the intended scale and fixture clearance.');
  if (table.enabled && (table.width / 2 > bounds.rightX || -table.width / 2 < bounds.leftX ||
      -table.length / 2 < bounds.backZ || table.length / 2 > bounds.frontZ))
    append('table', 'Table extends through the room walls. Reduce its width/length or enlarge the room.');
  for (const id of ['table-left', 'table-right']) {
    const pose = poses.find(p => p.id === id);
    if (!pose) continue;
    if (!table.enabled) append(id, 'Table is off; this stage-light stand has no tabletop support.');
    else {
      if (Math.abs(pose.pos[0]) + TABLE_STAND.halfWidth > table.width / 2 ||
          Math.abs(pose.pos[2]) + TABLE_STAND.halfLength > table.length / 2)
        append(id, 'Stand footprint extends beyond the tabletop. Enlarge the table or adjust arc/radius.');
      const bottom = pose.pos[1] - TABLE_STAND.baseOffset - TABLE_STAND.baseHalfHeight;
      if (Math.abs(bottom - TABLE_TOP_Y) > 1e-9)
        append(id, bottom > TABLE_TOP_Y ? 'Stand floats above the tabletop; adjust Y in Advanced.' :
          'Stand intersects or sits below the tabletop; adjust Y in Advanced.');
    }
    // The snoot extends .36 in front of a .1125-radius housing. This bound
    // deliberately warns conservatively instead of asserting no collision.
    if (Math.hypot(...pose.pos.map((v,i) => v - subject.center[i])) < subject.radius + Math.hypot(.36, .1125))
      append(id, 'Light may overlap the subject. Increase radius or adjust its position in Advanced.');
  }
  const fill = poses.find(p => p.id === 'fill-upper');
  if (fill) {
    const axes = basis(fill.dir), low = [-.375, -.375, -.42], high = [.375, .375, 0];
    const distance2 = axes.reduce((sum, axis, i) => {
      const coordinate = axis.reduce((s, v, j) => s + v * (subject.center[j] - fill.pos[j]), 0);
      const gap = coordinate - Math.max(low[i], Math.min(high[i], coordinate));
      return sum + gap * gap;
    }, 0);
    if (distance2 < subject.radius ** 2)
      append('fill-upper', 'Top fill overlaps the subject clearance. Adjust X/Z in Advanced before rendering; height does not move it sideways.');
  }
  const under = poses.find(p => p.id === 'underlight');
  if (table.enabled && under) {
    const axes = basis(under.dir);
    const verticalRadius = Math.abs(axes[0][1]) * 1.125 + Math.abs(axes[1][1]) * 1.5;
    const rear = -.14 * under.dir[1], front = .202 * under.dir[1];
    const low = under.pos[1] - verticalRadius + Math.min(rear, front, 0);
    const high = under.pos[1] + verticalRadius + Math.max(rear, front, 0);
    if (low < bounds.floorY) append('table', 'Underlight extends below the room floor. Increase table height or adjust underlight Y.');
    if (high > TABLE_TOP_Y - TABLE_THICKNESS) append('table', 'Underlight may intersect or rise above the tabletop. Check its height and aim in Advanced.');
  }
  return warnings;
}

// Closed solid bars, with winding agreeing with normals. The ray camera reads
// winding, while Phong reads explicit normals, so both must describe one body.
function solidBox(center, axes, half, material, group) {
  const faces = [];
  for (let axis = 0; axis < 3; axis++) for (const sign of [-1, 1]) {
    const n = axes[axis].map(v => v * sign), c = add(center, axes[axis], half[axis] * sign);
    const a = (axis + 1) % 3, b = (axis + 2) % 3;
    const corners = [[-1,-1],[1,-1],[1,1],[-1,1]].map(([x,y]) => add(add(c, axes[a], x*half[a]), axes[b], y*half[b]));
    if (cross(axes[a], axes[b]).reduce((s,v,i) => s + v*n[i], 0) < 0) corners.reverse();
    faces.push([corners[0],corners[1],corners[2],material,[n,n,n],group],
      [corners[0],corners[2],corners[3],material,[n,n,n],group]);
  }
  return faces;
}

// Subdivide existing surfaces only: no duplicated coplanar coating or blocked openings.
function accentGridBar(part,role,axes,axis,index,count,otherCount,otherSpan) {
  const a=role.gridAccent;if(!a?.enabled)return part;
  if(!['checker','rows','columns'].includes(a.pattern)||!['interiors','walls','corners'].includes(a.scope)||!Number.isInteger(a.size)||a.size<1||a.size>32||!Number.isFinite(a.width)||a.width<=0||a.width>.5)throw new RangeError('Invalid grid accent');
  const material=role.gridAccentMaterialId;if(!Number.isInteger(material))throw new RangeError('Grid accent needs a resolved material');
  const tangent=axes[1-axis],pitch=2*otherSpan/otherCount;
  const coord=p=>p.reduce((v,n,k)=>v+(n-role.pos[k])*tangent[k],0);
  const cuts=[];for(let j=0;j<=otherCount;j++)cuts.push(-otherSpan+j*pitch);
  if(a.scope==='corners')for(let j=0;j<otherCount;j++)cuts.push(-otherSpan+(j+a.width)*pitch,-otherSpan+(j+1-a.width)*pitch);
  cuts.sort((x,y)=>x-y);
  const selected=(cell,other)=>{
    if(cell<0||cell>=count||other<0||other>=otherCount)return false;
    const x=Math.floor((axis===0?cell:other)/a.size),y=Math.floor((axis===0?other:cell)/a.size);
    return (a.pattern==='rows'?y:a.pattern==='columns'?x:x+y)%2===0;
  };
  const clip=(poly,cut,sign)=>{const out=[];for(let i=0;i<poly.length;i++){
    const p=poly[i],q=poly[(i+1)%poly.length],u=sign*(coord(p)-cut),v=sign*(coord(q)-cut);
    if(u>=0)out.push(p);if((u>0&&v<0)||(u<0&&v>0)){const f=u/(u-v);out.push(p.map((n,k)=>n+f*(q[k]-n)));}
  }return out;};
  const result=[];
  part.forEach((face,triangle)=>{
    const side=Math.floor(triangle/2),interior=side===axis*2&&index>0||side===axis*2+1&&index<count;
    if(a.scope==='walls'?side<4:!interior){result.push(face);return;}
    let polys=[face.slice(0,3)];
    for(const cut of cuts){const next=[];for(const poly of polys){const values=poly.map(coord);if(Math.min(...values)<cut&&Math.max(...values)>cut){for(const sign of [-1,1]){const q=clip(poly,cut,sign);if(q.length>=3)next.push(q);}}else next.push(poly);}polys=next;}
    for(const poly of polys){
      const t=poly.reduce((v,p)=>v+coord(p),0)/poly.length,other=Math.floor((t+otherSpan)/pitch);
      const cell=side===axis*2?index-1:index;
      let active=a.scope==='walls'?selected(index-1,other)||selected(index,other):selected(cell,other);
      if(a.scope==='corners'){const f=(t+otherSpan)/pitch-other;active=active&&(f<a.width||f>1-a.width);}
      for(let k=1;k<poly.length-1;k++){
        const tri=[poly[0],poly[k],poly[k+1]],u=tri[1].map((v,i)=>v-tri[0][i]),v=tri[2].map((v,i)=>v-tri[0][i]);
        if(Math.hypot(...cross(u,v))<1e-15)continue;
        result.push([...tri,active?material:face[3],face[4],face[5]]);
      }
    }
  });return result;
}

export function tableLightModifierFaces(role, housingMaterial, onComponent) {
  const faces=[];
  // Metadata records the actual solidBox call and unchanged output range.
  // It is not a closed/convex/embedding certificate for rounded coordinates.
  const box=(id,center,axes,half,faceMaterials=null,accent=null)=>{
    const firstTriangle=faces.length;let part=solidBox(center,axes,half,housingMaterial,role.mat);
    if(faceMaterials)for(let i=0;i<part.length;i++)part[i][3]=faceMaterials[Math.floor(i/2)];
    if(accent)part=accent(part);
    faces.push(...part);
    onComponent?.({id,firstTriangle,triangleCount:part.length,sourceConstruction:"solidBox",
      intendedShape:"cuboid",constructionArguments:{center,axes,half},contactAdmission:"unverified"});
  };
  if (role.tableStand) {
    const axes = [[1,0,0],[0,1,0],[0,0,1]];
    box("stand-base",add(role.pos,[0,1,0],-TABLE_STAND.baseOffset),axes,
      [TABLE_STAND.halfWidth,TABLE_STAND.baseHalfHeight,TABLE_STAND.halfLength]);
    box("stand-post",add(role.pos,[0,1,0],-.14),axes,[.025,.12,.025]);
    return faces;
  }
  if (!role.grid) return [];
  const axes = basis(role.dir), depth = role.gridDepth??.20, thickness = role.gridThickness??.025, offset=role.gridOffset??0;
  const cellsX=role.gridCellsX??6,cellsY=role.gridCellsY??8;
  if(![cellsX,cellsY].every(n=>Number.isInteger(n)&&n>=1&&n<=32)||
     !Number.isFinite(depth)||depth<.01||depth>2||!Number.isFinite(thickness)||thickness<.002||thickness>.2||
     thickness>=Math.min(2*role.halfW/cellsX,2*role.halfH/cellsY,.5*role.halfW,.5*role.halfH))throw new RangeError('Invalid underlight grid geometry');
  // A real egg-crate: 6 by 8 open cells with opaque walls extending in front
  // of the same rectangular emitter uploaded to the photo/live-ray scene.
  for (const [axis, count, span, other] of [[0,cellsX,role.halfW,role.halfH], [1,cellsY,role.halfH,role.halfW]]) {
    for (let i = 0; i <= count; i++) {
      const center = add(add(role.pos, axes[axis], -span + 2*span*i/count), role.dir, offset + depth/2 + .002);
      const half = axis === 0 ? [thickness/2,other+thickness/2,depth/2] : [other+thickness/2,thickness/2,depth/2];
      // Only cell-facing longitudinal surfaces are reflective tunnel walls.
      // Front/back caps, bar ends and outside perimeter stay the exterior finish.
      const outer=role.gridOuterMaterialId??housingMaterial,inner=role.gridInteriorMaterialId??housingMaterial;
      const materials=Array(6).fill(outer);
      if(i>0)materials[axis*2]=inner;
      if(i<count)materials[axis*2+1]=inner;
      box(`grid-${axis===0?"x":"y"}-${i}`,center,axes,half,materials,part=>accentGridBar(part,role,axes,axis,i,count,axis===0?cellsY:cellsX,other));
    }
  }
  // Seal the emitter-to-can annulus, including the old .002 stand-off crack.
  // Four closed strips meet the grid edge and extend behind the can lip.
  // This is real opaque geometry, not an analytic angular cutoff.
  const innerW=role.halfW+thickness/2,innerH=role.halfH+thickness/2;
  const outerW=role.halfW*1.25+.002,outerH=role.halfH*1.25+.002;
  const sealDepth=(offset+depth+.004)/2,sealCenter=(offset+depth)/2;
  for(const sign of [-1,1]){
    box(`grid-apron-x-${sign}`,add(add(role.pos,axes[0],sign*(innerW+outerW)/2),role.dir,sealCenter),axes,
      [(outerW-innerW)/2,outerH,sealDepth],Array(6).fill(role.gridOuterMaterialId??housingMaterial));
    box(`grid-apron-y-${sign}`,add(add(role.pos,axes[1],sign*(innerH+outerH)/2),role.dir,sealCenter),axes,
      [innerW,(outerH-innerH)/2,sealDepth],Array(6).fill(role.gridOuterMaterialId??housingMaterial));
  }
  return faces;
}
