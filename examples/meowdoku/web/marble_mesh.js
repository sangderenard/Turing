// Layered cat's-eye marble geometry for the page's existing litMesh/Phong path.
// Coordinates are in marble radii. Each dyed ribbon is a closed, thin volume
// inside the separate closed glass shell; neither is a painted texture.

const TAU = 2 * Math.PI;
function rotated([x,y,z], yaw, pitch, roll) {
  const cy=Math.cos(yaw),sy=Math.sin(yaw),cx=Math.cos(pitch),sx=Math.sin(pitch),cz=Math.cos(roll),sz=Math.sin(roll);
  const X=cy*x+sy*z, Z=-sy*x+cy*z;
  const Y=cx*y-sx*Z, ZZ=sx*y+cx*Z;
  return [cz*X-sz*Y,sz*X+cz*Y,ZZ];
}
function orient(faces,start,end,o) {
  // Preserve the exact shared generator frame for opt-in volume authoring.
  // p = origin + scale * (q.x*axes[0] + q.y*axes[1] + q.z*axes[2]).
  const frame={origin:[0,0,0],axes:[[1,0,0],[0,1,0],[0,0,1]],scale:1};
  if (!o.orientationYaw && !o.orientationPitch && !o.orientationRoll) return frame;
  const center=[0,0,0],count=(end-start)*3;
  for(let f=start;f<end;f++) for(const vertex of faces[f].slice(0,3))
    for(let k=0;k<3;k++) center[k]+=vertex[k]/count;
  frame.axes=frame.axes.map(axis=>rotated(axis,o.orientationYaw,o.orientationPitch,o.orientationRoll));
  const rotatedCenter=rotated(center,o.orientationYaw,o.orientationPitch,o.orientationRoll);
  frame.origin=center.map((x,k)=>x-rotatedCenter[k]);
  for(let f=start;f<end;f++) {
    const face=faces[f];
    for(let v=0;v<3;v++) {
      face[v]=rotated(face[v].map((x,k)=>x-center[k]),o.orientationYaw,o.orientationPitch,o.orientationRoll)
        .map((x,k)=>x+center[k]);
      face[4][v]=rotated(face[4][v],o.orientationYaw,o.orientationPitch,o.orientationRoll);
    }
  }
  let furthest=0;
  for(let f=start;f<end;f++) for(const vertex of faces[f].slice(0,3)) furthest=Math.max(furthest,Math.hypot(...vertex));
  if(furthest>0.94) {
    const scale=0.94/furthest;
    frame.scale=scale;
    frame.origin=frame.origin.map(x=>x*scale);
    for(let f=start;f<end;f++) for(let v=0;v<3;v++) faces[f][v]=faces[f][v].map((x)=>x*scale);
  }
  return frame;
}

function unit(x, y, z) {
  const length = Math.hypot(x, y, z) || 1;
  return [x / length, y / length, z / length];
}

function checked(options) {
  const o = { rings: 24, segments: 48, eyeWidth: 0.28, eyeLength: 0.75,
    eyeCurve: 0.18, strandCount: 3, ribbonThickness: 0.035,
    ribbonTwist: 0.9, ribbonTip: 1.35, ribbonFlare: 0.18, ribbonPhase: 0,
    // Omitted layout preserves v1 saved recipes and existing game geometry.
    // New Studio designs explicitly select axial, with a shared centreline.
    ribbonLayout: "separated",
    kind: "cat_eye", confettiCount: 220, enamelCoverage: 0,
    paintBandCount: 0, paintBandWidth: 0.18, paintBandCurve: 1.2,
    tubeCount: 2, tubeRadius: 0.024, tubeTurns: 2.2,
    orientationYaw:0,orientationPitch:0,orientationRoll:0, seed: 0, ...options };
  for (const key of ["rings", "segments", "strandCount", "confettiCount", "seed", "paintBandCount", "tubeCount"])
    if (!Number.isInteger(o[key])) throw new RangeError(`${key} must be an integer`);
  if (o.rings < 8 || o.rings > 128 || o.segments < 12 || o.segments > 256 || o.strandCount < 1 || o.strandCount > 7)
    throw new RangeError("marble tessellation outside supported range");
  if (o.seed < 0 || o.seed > 0x7fffffff) throw new RangeError("seed outside supported range");
  if (!["cat_eye", "confetti", "hybrid", "clear", "spiral", "rings"].includes(o.kind) ||
      !["separated", "axial"].includes(o.ribbonLayout) ||
      o.confettiCount < 0 || o.confettiCount > 500 ||
      o.ribbonPhase < 0 || o.ribbonPhase > TAU ||
      o.enamelCoverage < 0 || o.enamelCoverage > 1 || o.paintBandCount < 0 || o.paintBandCount > 12 ||
      o.paintBandWidth < 0.02 || o.paintBandWidth > 0.48 || Math.abs(o.paintBandCurve) > 4 ||
      o.tubeCount < 1 || o.tubeCount > 3 || o.tubeRadius < 0.008 || o.tubeRadius > 0.05 ||
      o.tubeTurns < 1 || o.tubeTurns > 4 ||
      [o.orientationYaw,o.orientationPitch,o.orientationRoll].some((v)=>!Number.isFinite(v)||Math.abs(v)>Math.PI))
    throw new RangeError("marble kind or coverage outside supported range");
  if (o.confettiEnabled !== undefined && typeof o.confettiEnabled !== "boolean") throw new TypeError("confetti enabled must be boolean");
  if (o.kind === "hybrid" && (o.orientationYaw || o.orientationPitch || o.orientationRoll))
    throw new RangeError("hybrid orientation is fixed to keep dye and confetti volumes disjoint");
  if (!(o.eyeWidth >= 0.08 && o.eyeWidth <= 0.42 && o.eyeLength >= 0.4 && o.eyeLength <= 0.85 &&
        o.eyeCurve >= -0.28 && o.eyeCurve <= 0.28))
    throw new RangeError("cat's-eye dimensions outside the glass interior");
  if (!(o.ribbonThickness >= 0.008 && o.ribbonThickness <= 0.08 && o.ribbonTwist >= -3.2 && o.ribbonTwist <= 3.2 &&
        o.ribbonTip >= 0.6 && o.ribbonTip <= 3.0 && o.ribbonFlare >= 0 && o.ribbonFlare <= 0.6))
    throw new RangeError("ribbon geometry outside the glass interior");
  return o;
}

function surface(faces, point, normal, rings, segments, material) {
  for (let i = 0; i < rings; i++) for (let j = 0; j < segments; j++) {
    const a = point(i, j), b = point(i + 1, j), c = point(i + 1, j + 1), d = point(i, j + 1);
    const na = normal(i, j), nb = normal(i + 1, j), nc = normal(i + 1, j + 1), nd = normal(i, j + 1);
    const id = typeof material === "function" ? material(i, j) : material;
    if (i < rings - 1) faces.push([a, c, b, id, [na, nc, nb]]);
    if (i > 0) faces.push([a, d, c, id, [na, nd, nc]]);
  }
}

/**
 * Build separately drawable opaque interior and transparent glass exterior.
 * `materials` contains the integer IDs in the page's MaterialDatabase table.
 * The renderer must draw `interior` before `glass` with depth writes off for
 * glass, as it already does for translucent scene geometry.
 */
export function catEyeMarble(materials, options = {}) {
  const o = checked(options), interior = [], glass = [], ribbons = [], confetti = [], tubes = [];
  for (const key of ["glass", "eye", "accent"])
    if (!Number.isInteger(materials[key]) || materials[key] < 0) throw new TypeError(`missing material ${key}`);
  for (const key of ["enamel", "confetti", "paintA", "paintB"])
    if (materials[key] !== undefined && (!Number.isInteger(materials[key]) || materials[key] < 0))
      throw new TypeError(`invalid material ${key}`);
  if (materials.ribbons !== undefined && (!Array.isArray(materials.ribbons) || materials.ribbons.length < o.strandCount || materials.ribbons.some(id => !Number.isInteger(id) || id < 0)))
    throw new TypeError("invalid ribbon materials");
  const sphere = (radius) => (i, j) => {
    const theta = Math.PI * i / o.rings, phi = TAU * j / o.segments;
    return [radius * Math.sin(theta) * Math.cos(phi), radius * Math.cos(theta), radius * Math.sin(theta) * Math.sin(phi)];
  };
  const shellPoint = sphere(1);
  const coatedColumns = Math.round(o.enamelCoverage * o.segments);
  surface(glass, shellPoint, (i, j) => unit(...shellPoint(i, j)), o.rings, o.segments,
    (i, j) => {
      if (o.paintBandCount) {
        const latitude = Math.PI * (i+0.5) / o.rings;
        const rawPhase = o.paintBandCount*(j+0.5)/o.segments +
          o.paintBandCurve*Math.cos(latitude)/TAU + o.ribbonPhase/TAU;
        const phase = ((rawPhase % 1) + 1) % 1;
        const distance = Math.min(phase,1-phase);
        if (distance < o.paintBandWidth/2) return materials.paintA ?? materials.glass;
        if (distance < o.paintBandWidth) return materials.paintB ?? materials.glass;
      }
      return j < coatedColumns ? (materials.enamel ?? materials.glass) : materials.glass;
    });

  // Each blade has its own closed cross-section, tapering to two points. The
  // broad faces, narrow bevel-like sides and curled tips therefore present
  // different normals to the real Phong light. The blades twist with the
  // shared curve. Axial ribbons have exactly the same centreline and shape,
  // rotated evenly through pi (a blade is two-sided). The legacy separated
  // layout remains byte-for-byte reproducible for existing saved recipes.
  for (let strand = 0; strand < (o.kind === "cat_eye" || o.kind === "hybrid" ? o.strandCount : 0); strand++) {
    const lane = strand - (o.strandCount - 1) / 2;
    const phase = ((o.seed + strand * 619) % 997) / 997 * TAU;
    const point = (i, j) => {
      const theta = Math.PI * i / o.rings, phi = TAU * j / o.segments;
      const s = Math.max(0, Math.sin(theta)), y = o.eyeLength * Math.cos(theta);
      const t = y / o.eyeLength, centre = o.eyeCurve * (1 - t*t);
      if (o.ribbonLayout === "axial") {
        const commonPhase = (o.seed % 997) / 997 * TAU;
        const angle = o.ribbonPhase + Math.PI * strand / o.strandCount + o.ribbonTwist * t;
        const width = o.eyeWidth * Math.pow(s, o.ribbonTip)
          * (1 + o.ribbonFlare * Math.sin(2 * theta + commonPhase) * s);
        const thick = o.ribbonThickness * Math.pow(s, Math.max(0.6, o.ribbonTip * 0.7));
        const p = Math.sign(Math.cos(phi)) * Math.sqrt(Math.abs(Math.cos(phi)));
        const q = Math.sign(Math.sin(phi)) * Math.sqrt(Math.abs(Math.sin(phi)));
        return [centre + width*p*Math.cos(angle) - thick*q*Math.sin(angle),
          y, width*p*Math.sin(angle) + thick*q*Math.cos(angle)];
      }
      const angle = 0.38 * lane + o.ribbonTwist * t + 0.18 * Math.sin(theta + phase);
      const ux = Math.cos(angle), uz = Math.sin(angle), vx = -uz, vz = ux;
      const rawWidth = o.eyeWidth * (0.46 + 0.08 * Math.sin(phase + strand)) * Math.pow(s, o.ribbonTip)
        * (1 + o.ribbonFlare * Math.sin(2 * theta + phase) * s) * Math.min(1,3/o.strandCount);
      const rawThick = o.ribbonThickness * Math.pow(s, Math.max(0.6, o.ribbonTip * 0.7));
      const spread = Math.min(0.35, 0.32 * 3/o.strandCount);
      const centerSpacing = Math.hypot(spread,Math.min(0.07,spread*0.7));
      const cap = 0.45*(centerSpacing-0.048);
      const fit = Math.min(1,cap/Math.hypot(rawWidth,rawThick));
      const width=rawWidth*fit,thick=rawThick*fit;
      // A superellipse cross-section gives broad, flat-looking faces and
      // narrow rounded edges without open seams at a twist or at a tip.
      const p = Math.sign(Math.cos(phi)) * Math.sqrt(Math.abs(Math.cos(phi)));
      const q = Math.sign(Math.sin(phi)) * Math.sqrt(Math.abs(Math.sin(phi)));
      const radialX = lane * spread, radialZ = lane * Math.min(0.07, spread * 0.7);
      const offsetX = radialX * Math.cos(o.ribbonPhase) - radialZ * Math.sin(o.ribbonPhase);
      const offsetZ = radialX * Math.sin(o.ribbonPhase) + radialZ * Math.cos(o.ribbonPhase);
      const sway = 0.024 * Math.sin(2 * theta + phase) * s*s;
      return [centre + offsetX + sway + width * p * ux + thick * q * vx,
        y, offsetZ + width * p * uz + thick * q * vz];
    };
    const normal = (i, j) => {
      if (i <= 0) return [0, 1, 0];
      if (i >= o.rings) return [0, -1, 0];
      const eps = 0.001, p = (u, v) => point(Math.max(0, Math.min(o.rings, u)), v);
      const a = p(i + eps, j), b = p(i - eps, j), c = p(i, j + eps), d = p(i, j - eps);
      const t = a.map((v, k) => v - b[k]), q = c.map((v, k) => v - d[k]);
      return unit(q[1]*t[2]-q[2]*t[1], q[2]*t[0]-q[0]*t[2], q[0]*t[1]-q[1]*t[0]);
    };
    const start = interior.length, material = materials.ribbons?.[strand] ?? (strand % 2 ? materials.accent : materials.eye);
    surface(interior, point, normal, o.rings, o.segments, material);
    ribbons.push({ start, count: interior.length - start, material, densityRole: strand % 2 ? "accent" : "eye" });
  }
  if (o.kind === "spiral" || o.kind === "rings") {
    for (let strand=0; strand<o.tubeCount; strand++) {
      const start=interior.length, material=strand%2?materials.accent:materials.eye;
      const path=(t)=>{
        const turn=TAU*(o.tubeTurns*t+strand/o.tubeCount)+o.ribbonPhase;
        if (o.kind === "spiral") {
          const radial=0.22+0.10*Math.sin(Math.PI*t);
          return [radial*Math.cos(turn),1.30*(t-0.5),radial*Math.sin(turn)];
        }
        const angle=TAU*t+TAU*strand/o.tubeCount+o.ribbonPhase;
        const radial=0.28+0.16*strand+0.025*Math.sin(2*angle+strand);
        return [radial*Math.cos(angle),0.12*Math.sin(3*angle+strand),radial*Math.sin(angle)];
      };
      const count=o.kind === "rings" ? o.rings*2 : o.rings*2;
      const sides=Math.max(16,Math.round(o.segments/3));
      const tube=(i,j)=>{
        const t=i/count, center=path(t);
        const before=path(Math.max(0,t-0.0001)),after=path(Math.min(1,t+0.0001));
        const tangent=unit(...after.map((x,k)=>x-before[k]));
        const axis=unit(tangent[2],0,-tangent[0]);
        const binormal=unit(tangent[1]*axis[2]-tangent[2]*axis[1],
          tangent[2]*axis[0]-tangent[0]*axis[2],tangent[0]*axis[1]-tangent[1]*axis[0]);
        const phi=TAU*j/sides;
        const radius=o.tubeRadius*(o.kind === "spiral" ? Math.min(1,Math.max(0,Math.min(i,count-i))) : 1);
        return center.map((x,k)=>x+radius*(Math.cos(phi)*axis[k]+Math.sin(phi)*binormal[k]));
      };
      const normal=(i,j)=>unit(...tube(i,j).map((x,k)=>x-path(i/count)[k]));
      for(let i=0;i<count;i++) for(let j=0;j<sides;j++) {
        const a=tube(i,j),b=tube(i+1,j),c=tube(i+1,j+1),d=tube(i,j+1);
        const na=normal(i,j),nb=normal(i+1,j),nc=normal(i+1,j+1),nd=normal(i,j+1);
        interior.push([a,c,b,material,[na,nc,nb]],[a,d,c,material,[na,nd,nc]]);
      }
      if(o.kind==="spiral") {
        // The tapered first and last rings collapse to a point, closing both tips.
        // Degenerate triangles at the tips are omitted.
        interior.splice(start,sides*2);
        interior.splice(interior.length-sides*2,sides*2);
        for(const end of [0,count]) {
          const apex=path(end/count), i=end===0?1:count-1;
          for(let j=0;j<sides;j++) {
            const a=tube(i,j),b=tube(i,j+1),B=end===0?b:a,C=end===0?a:b;
            const u=B.map((x,k)=>x-apex[k]),v=C.map((x,k)=>x-apex[k]);
            const n=unit(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]);
            interior.push([apex,B,C,material,[n,n,n]]);
          }
        }
      }
      tubes.push({start,count:interior.length-start,material,densityRole:strand%2?"accent":"eye"});
    }
  }
  // A common rigid transform preserves separation among distinct closed
  // inclusions. In hybrid designs the ribbons are confined to an inner radial
  // zone, leaving a clear annulus for independently packed confetti chips.
  const inclusionFrame=orient(interior,0,interior.length,o);
  if(o.kind==="hybrid") {
    let furthest=0;
    for(const face of interior) for(const p of face.slice(0,3)) furthest=Math.max(furthest,Math.hypot(p[0],p[2]));
    if(furthest>0.65) {
      const scale=0.65/furthest;
      inclusionFrame.scale*=scale;
      inclusionFrame.origin=inclusionFrame.origin.map(x=>x*scale);
      for(const face of interior) for(let k=0;k<3;k++) face[k]=face[k].map((x)=>x*scale);
    }
  }
  if (o.confettiEnabled ?? (o.kind === "confetti" || o.kind === "hybrid")) {
    // New overlays keep the authored decoration unchanged. Conservative boxes
    // around each closed inclusion exclude chips without guessing a surface
    // intersection or shrinking/rotating the user's ribbons and tubes.
    // The historical hybrid path retains its exact original packing recipe.
    const overlay=o.kind!=="hybrid" && interior.length>0;
    const exclusions=overlay?[...ribbons,...tubes].map(part=>{
      const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
      for(const face of interior.slice(part.start,part.start+part.count))for(const p of face.slice(0,3))
        for(let i=0;i<3;i++){lo[i]=Math.min(lo[i],p[i]);hi[i]=Math.max(hi[i],p[i]);}
      return {lo,hi};
    }):[];
    // Each rolled chip is a closed tetrahedral inclusion. A spherical swirl
    // places the chips through the interior, rather than painting the shell.
    let state = (o.seed ^ 0x9e3779b9) >>> 0;
    const random = () => ((state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 0x100000000);
    const occupied=[];
    for (let k = 0, attempts=0; k < o.confettiCount;) {
      if (++attempts > 30000) throw new RangeError("confetti cannot be packed without overlap");
      const hybrid=o.kind==="hybrid";
      const y = (random() * 2 - 1) * (hybrid ? 0.40 : 0.60);
      const r = hybrid ? 0.72 + 0.06 * random() : overlay ? 0.72 + 0.16 * random() : 0.15 + 0.53 * random();
      const phi = TAU * (attempts * 0.61803398875 + random() * 0.18) + 1.2 * y;
      const center = [r * Math.cos(phi), y, r * Math.sin(phi)];
      const size = 0.012 + random() * 0.018, angle = TAU * random();
      if(overlay && (Math.hypot(...center)+1.02*size+0.003>=0.96 ||
        exclusions.some(({lo,hi})=>center.every((x,i)=>x+1.02*size+0.003>=lo[i] && x-1.02*size-0.003<=hi[i]))))continue;
      if (occupied.some(({center:other,size:otherSize}) =>
        (overlay ? center.every((x,i)=>Math.abs(x-other[i]) < 1.02*(size+otherSize)+0.003) :
          Math.hypot(...center.map((x,i)=>x-other[i])) < 1.02*(size+otherSize)+0.003))) continue;
      occupied.push({center,size});
      const u = [Math.cos(angle), 0, Math.sin(angle)];
      const v = [-Math.sin(angle), 0, Math.cos(angle)];
      const add = (a, b, c) => center.map((x, i) => x + size * (a*u[i] + b*v[i] + c*(i === 1 ? 1 : 0)));
      const vertices = [add(1,0,0), add(-0.55,0.85,0), add(-0.55,-0.85,0), add(0,0,0.45)];
      const material = k % 4 === 0 ? materials.accent : (materials.confetti ?? materials.eye);
      const start = interior.length;
      for (const [a,b,c] of [[0,1,2],[0,3,1],[1,3,2],[2,3,0]]) {
        const A=vertices[a], B=vertices[b], C=vertices[c];
        const p=B.map((x,i)=>x-A[i]), q=C.map((x,i)=>x-A[i]);
        const n=unit(p[1]*q[2]-p[2]*q[1],p[2]*q[0]-p[0]*q[2],p[0]*q[1]-p[1]*q[0]);
        interior.push([A,B,C,material,[n,n,n]]);
      }
      confetti.push({start,count:4,material,center,densityRole:k%4===0?"accent":"confetti"});
      k++;
    }
  }
  const overlappingRibbons = o.ribbonLayout === "axial" && ribbons.length > 1;
  return { interior, glass, triangleCount: interior.length + glass.length,
    mechanicsResolved: !overlappingRibbons,
    mechanicsNote: overlappingRibbons ? "Axial ribbons overlap; exact material-union mass and rolling inertia are unresolved" : undefined,
    ribbons, confetti, tubes, inclusionFrame, parameters: {...o,shellMaterial:materials.glass} };
}
