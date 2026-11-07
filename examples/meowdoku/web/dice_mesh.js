// Convex dice solids for the marble designer. All faces are outward-wound,
// closed triangles in the same tuple format as marble_mesh.js. Numbering is
// defined on the geometric faces, rather than depending on traversal order.
const TAU = 2 * Math.PI;
const sub = (a,b) => a.map((x,i)=>x-b[i]);
const dot = (a,b) => a.reduce((s,x,i)=>s+x*b[i],0);
const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit = (a) => { const r=Math.hypot(...a); return a.map(x=>x/r); };
const mean = (points) => points[0].map((_,k)=>points.reduce((s,p)=>s+p[k],0)/points.length);
function originToSegment(a,b) {
  const ab=sub(b,a), length2=dot(ab,ab);
  const t=length2>0?Math.max(0,Math.min(1,-dot(a,ab)/length2)):0;
  return Math.hypot(...a.map((x,i)=>x+t*ab[i]));
}
function originToTriangle(a,b,c) {
  const n=cross(sub(b,a),sub(c,a)), n2=dot(n,n);
  let nearest=Math.min(originToSegment(a,b),originToSegment(b,c),originToSegment(c,a));
  if(n2<1e-24) return nearest;
  const p=n.map(x=>x*dot(n,a)/n2);
  if([[a,b],[b,c],[c,a]].every(([u,v])=>dot(cross(sub(v,u),sub(p,u)),n)>=-1e-12*n2))
    nearest=Math.min(nearest,Math.hypot(...p));
  return nearest;
}

function hull(vertices) {
  const found=new Map(), epsilon=1e-7;
  for(let i=0;i<vertices.length;i++) for(let j=i+1;j<vertices.length;j++) for(let k=j+1;k<vertices.length;k++) {
    let n=cross(sub(vertices[j],vertices[i]),sub(vertices[k],vertices[i]));
    if(Math.hypot(...n)<epsilon) continue;
    n=unit(n);
    const signed=vertices.map(p=>dot(n,sub(p,vertices[i])));
    if(signed.some(x=>x>epsilon)&&signed.some(x=>x<-epsilon)) continue;
    if(signed.some(x=>x>epsilon)) n=n.map(x=>-x);
    const ids=vertices.map((p,m)=>Math.abs(dot(n,sub(p,vertices[i])))<epsilon?m:-1).filter(m=>m>=0);
    const key=ids.join(',');
    if(found.has(key)) continue;
    const center=mean(ids.map(m=>vertices[m]));
    const axis=unit(sub(vertices[ids[0]],center)), side=cross(n,axis);
    ids.sort((a,b)=>Math.atan2(dot(sub(vertices[a],center),side),dot(sub(vertices[a],center),axis))-
                     Math.atan2(dot(sub(vertices[b],center),side),dot(sub(vertices[b],center),axis)));
    found.set(key,{ids,normal:n});
  }
  return [...found.values()];
}
function polarDual(vertices) {
  const faces=hull(vertices);
  return faces.map(({ids,normal})=>normal.map(x=>x/dot(normal,vertices[ids[0]])));
}
function verticesFor(kind) {
  const phi=(1+Math.sqrt(5))/2;
  if(kind==='d4') return [[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]];
  if(kind==='d6') return [-1,1].flatMap(x=>[-1,1].flatMap(y=>[-1,1].map(z=>[x,y,z])));
  if(kind==='d8') return [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]];
  const ico=[...[-1,1].flatMap(y=>[-1,1].map(z=>[0,y,z*phi])),
    ...[-1,1].flatMap(x=>[-1,1].map(y=>[x,y*phi,0])),
    ...[-1,1].flatMap(x=>[-1,1].map(z=>[x*phi,0,z]))];
  if(kind==='d20') return ico;
  if(kind==='d12') return polarDual(ico);
  if(kind==='d10') {
    const top=Array.from({length:5},(_,i)=>[Math.cos(TAU*i/5),0.48,Math.sin(TAU*i/5)]);
    const bottom=Array.from({length:5},(_,i)=>[Math.cos(TAU*(i+0.5)/5),-0.48,Math.sin(TAU*(i+0.5)/5)]);
    return polarDual([...top,...bottom]);
  }
  throw new RangeError(`unsupported die ${kind}`);
}

// The seven bars are separated by small gaps so their closed material regions
// remain legible after triangulation. 6 and 9 receive an orientation underline.
const SEGMENTS={0:'abcedf',1:'bc',2:'abged',3:'abgcd',4:'fgbc',5:'afgcd',
  6:'afgecd',7:'abc',8:'abcdefg',9:'abfgcd'};
const BARS={a:[0.22,0.85,0.78,0.96],b:[0.78,0.52,0.91,0.85],c:[0.78,0.15,0.91,0.48],
  d:[0.22,0.04,0.78,0.15],e:[0.09,0.15,0.22,0.48],f:[0.09,0.52,0.22,0.85],
  g:[0.22,0.445,0.78,0.555]};
function inText(x,y,label) {
  const str=String(label), step=1.16, total=str.length*step-0.16;
  for(let i=0;i<str.length;i++) {
    const local=x+total/2-i*step, digit=+str[i];
    if(!Number.isInteger(digit)) continue;
    for(const bar of SEGMENTS[digit]) {
      const [x0,y0,x1,y1]=BARS[bar];
      if(local>=x0&&local<=x1&&y>=y0&&y<=y1) return true;
    }
    if((digit===6||digit===9)&&local>=0.25&&local<=0.75&&y>=-0.10&&y<=-0.035) return true;
  }
  return false;
}
function numberFaces(faces,kind) {
  if(kind==='d4') return faces.map((face)=>({face,number:null}));
  const count=faces.length, first=kind==='d10'?0:1, numbers=new Array(count);
  if(kind==='d10') {
    // A common d10 ordering is 0,8,2,6,4 around one pole. The lower ring
    // carries the odd values and each antipodal pair sums to nine.
    const upper=faces.map((f,i)=>i).filter(i=>faces[i].normal[1]>0)
      .sort((a,b)=>Math.atan2(faces[a].normal[2],faces[a].normal[0])-
                   Math.atan2(faces[b].normal[2],faces[b].normal[0]));
    if(upper.length!==5) throw new Error('d10 must have five upper faces');
    const upperValues=[0,8,2,6,4];
    for(let k=0;k<5;k++) {
      const a=upper[k];numbers[a]=upperValues[k];
      const opposite=faces.findIndex((f,i)=>i!==a&&dot(f.normal,faces[a].normal)<-0.999);
      if(opposite<0) throw new Error('d10 antipodal face missing');
      numbers[opposite]=9-upperValues[k];
    }
    return faces.map((face,i)=>({face,number:numbers[i]}));
  }
  const order=faces.map((f,i)=>i).sort((a,b)=>{
    const A=faces[a].normal,B=faces[b].normal;
    return B[1]-A[1]||B[2]-A[2]||B[0]-A[0];
  });
  for(let k=0;k<count/2;k++) {
    const a=order.find(i=>numbers[i]===undefined);
    let opposite=-1,best=Infinity;
    for(let i=0;i<count;i++) if(i!==a&&numbers[i]===undefined) {
      const score=dot(faces[a].normal,faces[i].normal);
      if(score<best) {best=score;opposite=i;}
    }
    if(best> -0.999) throw new Error(`${kind} has no opposed face for ${a}`);
    numbers[a]=first+k;
    numbers[opposite]=kind==='d10'?9-k:count+1-(first+k);
  }
  return faces.map((face,i)=>({face,number:numbers[i]}));
}
function markingsFor(face,number,kind,vertices) {
  if(kind!=='d4') return [{label:number,center:mean(face.ids.map(i=>vertices[i]))}];
  // A classic top-read d4 repeats a vertex value on all three adjacent faces.
  // The value at the upward-pointing corner is read after a roll.
  return face.ids.map(i=>({label:i+1,center:vertices[i]}));
}
function evaluateMark(point,face,marks,vertices,scale) {
  const center=mean(face.ids.map(i=>vertices[i]));
  const n=face.normal;
  // Align ordinary face labels with an actual edge. The old center-to-corner
  // basis put cube numbers diagonally across the square face.
  const tangent=unit(sub(vertices[face.ids[1]],vertices[face.ids[0]]));
  const bitangent=cross(n,tangent);
  const inradius=Math.min(...face.ids.map((id,k)=>{
    const a=vertices[id],b=vertices[face.ids[(k+1)%face.ids.length]];
    return Math.abs(dot(cross(sub(b,a),sub(center,a)),n))/Math.hypot(...sub(b,a));
  }));
  for(const mark of marks) {
    const location=kindLocation(mark,center,inradius,scale);
    const offset=sub(point,location.center);
    // On a top-read d4, each corner numeral is upright toward its own apex.
    // A single face-wide basis would turn two of the three values sideways.
    const up=location.corner?unit(sub(mark.center,center)):bitangent;
    const right=location.corner?cross(up,n):tangent;
    // inText already adds half of the whole word width to center each glyph.
    // Adding another 0.5 here displaced a one-digit mark almost entirely off
    // its face, leaving only a few jagged strokes.
    const x=dot(offset,right)/location.height*1.5;
    const y=dot(offset,up)/location.height+0.5;
    if(inText(x,y,mark.label)) return true;
  }
  return false;
}
function kindLocation(mark,faceCenter,inradius,scale) {
  const isCorner=Math.hypot(...sub(mark.center,faceCenter))>inradius*0.5;
  return {center:isCorner?mark.center.map((x,i)=>faceCenter[i]*0.48+x*0.52):mark.center,
    height:inradius*(isCorner?0.46:scale),corner:isCorner};
}

/**
 * `markingMode` is "flat", "raised", or "recessed". Relief displaces the
 * closed shell surface, so marbleMass integrates its actual triangulated volume.
 * Mark material is the optical Pluck material ID; bulk relief retains the body
 * density until separate material-domain geometry is available.
 */
export function diceSolid(kind,{shellMaterial,markMaterial=shellMaterial,markingMode='flat',
  reliefDepth=0.02,subdivisions=(kind==='d4'||kind==='d6')?64:48,markScale=0.72,
  bevelRadius=0,faceRelief=0,compensateRaised=true}={}) {
  if(!Number.isInteger(shellMaterial)||shellMaterial<0||!Number.isInteger(markMaterial)||markMaterial<0)
    throw new TypeError('shell and marking material IDs are required');
  if(!['flat','raised','recessed'].includes(markingMode)||!(reliefDepth>=0&&reliefDepth<=0.08)||
     !Number.isInteger(subdivisions)||subdivisions<12||subdivisions>128||
     !(markScale>=0.3&&markScale<=0.9)||!(bevelRadius>=0&&bevelRadius<=0.16)||
     !(faceRelief>=0&&faceRelief<=0.08)||typeof compensateRaised!=='boolean')
    throw new RangeError('unsupported marking or face geometry');
  let vertices=verticesFor(kind), radius=Math.max(...vertices.map(v=>Math.hypot(...v)));
  vertices=vertices.map(v=>v.map(x=>x/radius));
  const facets=hull(vertices), expected={d4:4,d6:6,d8:8,d10:10,d12:12,d20:20}[kind];
  if(facets.length!==expected) throw new Error(`${kind} hull has ${facets.length} faces`);
  const supports=facets.map(f=>dot(f.normal,vertices[f.ids[0]]));
  const support=supports.reduce((a,b)=>a+b,0)/supports.length;
  if(supports.some(h=>Math.abs(h-support)>1e-6)) throw new Error('dice bevel requires equal-support faces');
  const inset=supports.map(h=>h-bevelRadius), roundedCache=new Map();
  const rounded=(p)=>{
    if(!bevelRadius) return p;
    const key=p.map(x=>x.toFixed(12)).join(','),cached=roundedCache.get(key);
    if(cached) return cached;
    // Project the original surface point onto the polyhedron inset by r,
    // then offset that closest point by an exact distance r. The boundary of
    // this inset-solid + radius-r ball has planar faces, circular edge
    // fillets, and spherical corner caps. Using the same global map for each
    // face keeps every seam closed.
    const candidates=facets.map((f,i)=>i).filter(i=>supports[i]-dot(facets[i].normal,p)<4*bevelRadius);
    let best=null,bestD=Infinity;
    const consider=(indices)=>{
      const ns=indices.map(i=>facets[i].normal);
      const rhs=indices.map(i=>dot(facets[i].normal,p)-inset[i]);
      const gram=ns.map(a=>ns.map(b=>dot(a,b)));
      const augmented=gram.map((row,i)=>[...row,rhs[i]]);
      for(let col=0;col<indices.length;col++) {
        let pivot=col;
        for(let row=col+1;row<indices.length;row++)
          if(Math.abs(augmented[row][col])>Math.abs(augmented[pivot][col])) pivot=row;
        if(Math.abs(augmented[pivot][col])<1e-10) return;
        [augmented[col],augmented[pivot]]=[augmented[pivot],augmented[col]];
        const divisor=augmented[col][col];
        for(let j=col;j<=indices.length;j++) augmented[col][j]/=divisor;
        for(let row=0;row<indices.length;row++) if(row!==col) {
          const factor=augmented[row][col];
          for(let j=col;j<=indices.length;j++) augmented[row][j]-=factor*augmented[col][j];
        }
      }
      const lambda=augmented.map(row=>row[indices.length]);
      if(lambda.some(x=>x< -1e-8)) return;
      const q=p.map((x,k)=>x-indices.reduce((s,i,j)=>s+lambda[j]*facets[i].normal[k],0));
      if(facets.some((f,i)=>dot(f.normal,q)>inset[i]+1e-8)) return;
      const d=dot(sub(p,q),sub(p,q));
      if(d<bestD) {bestD=d;best=q;}
    };
    for(let a=0;a<candidates.length;a++) consider([candidates[a]]);
    for(let a=0;a<candidates.length;a++) for(let b=a+1;b<candidates.length;b++)
      consider([candidates[a],candidates[b]]);
    for(let a=0;a<candidates.length;a++) for(let b=a+1;b<candidates.length;b++)
      for(let c=b+1;c<candidates.length;c++) consider([candidates[a],candidates[b],candidates[c]]);
    if(!best) throw new Error(`no feasible inset projection for ${kind}`);
    const direction=unit(sub(p,best));
    const result=best.map((x,i)=>x+bevelRadius*direction[i]);
    roundedCache.set(key,result);
    return result;
  };
  const numbered=numberFaces(facets,kind), glass=[], faceInfo=[];
  for(let faceIndex=0;faceIndex<numbered.length;faceIndex++) {
    const {face,number}=numbered[faceIndex], polygon=face.ids.map(i=>vertices[i]);
    const start=glass.length;
    const center=mean(polygon), markings=markingsFor(face,number,kind,vertices);
    const mark=(p)=>evaluateMark(p,face,markings,vertices,markScale);
    const inradius=Math.min(...polygon.map((a,k)=>{
      const b=polygon[(k+1)%polygon.length];
      return Math.abs(dot(cross(sub(b,a),sub(center,a)),face.normal))/Math.hypot(...sub(b,a));
    }));
    const reliefEdges=polygon.map((a,k)=>{
      const b=polygon[(k+1)%polygon.length],idA=face.ids[k],idB=face.ids[(k+1)%polygon.length];
      const neighbor=facets.find(other=>other!==face&&other.ids.includes(idA)&&other.ids.includes(idB));
      if(!neighbor) throw new Error(`${kind} missing bevel neighbor`);
      const cosine=Math.max(-0.999999,Math.min(0.999999,dot(face.normal,neighbor.normal)));
      return {a,b,margin:bevelRadius*Math.sqrt((1-cosine)/(1+cosine))};
    });
    const edgeFade=(p)=>{
      const d=Math.min(...reliefEdges.map(({a,b,margin})=>
        Math.abs(dot(cross(sub(b,a),sub(p,a)),face.normal))/Math.hypot(...sub(b,a))-margin));
      const t=Math.min(1,Math.max(0,d/(0.18*inradius)));
      return t*t*(3-2*t);
    };
    const displace=(p)=>{
      const fade=edgeFade(p), marked=mark(p), base=rounded(p);
      const compensation=markingMode==='raised'&&compensateRaised?reliefDepth:0;
      const inset=(faceRelief+compensation)*fade;
      const glyph=(markingMode==='raised'?reliefDepth:markingMode==='recessed'?-reliefDepth:0)
        *(marked?1:0)*fade;
      return base.map((x,i)=>x+face.normal[i]*(glyph-inset));
    };
    for(let side=0;side<polygon.length;side++) {
      const A=center,B=polygon[side],C=polygon[(side+1)%polygon.length];
      const point=(i,j)=>{
        const u=i/subdivisions,v=j/subdivisions;
        return A.map((x,k)=>x*(1-u-v)+B[k]*u+C[k]*v);
      };
      const emit=(a,b,c)=>{
        const raw=[a,b,c].map(([i,j])=>point(i,j));
        const p=raw.map(displace), normal=unit(cross(sub(p[1],p[0]),sub(p[2],p[0])));
        const centroid=mean(raw), id=mark(centroid)?markMaterial:shellMaterial;
        glass.push([p[0],p[1],p[2],id,[normal,normal,normal]]);
      };
      for(let i=0;i<subdivisions;i++) for(let j=0;j<subdivisions-i;j++) {
        emit([i,j],[i+1,j],[i,j+1]);
        if(i+j<subdivisions-1) emit([i+1,j],[i+1,j+1],[i,j+1]);
      }
    }
    faceInfo.push({faceIndex,number,normal:face.normal,vertexIds:[...face.ids],start,count:glass.length-start,
      markings:markings.map(x=>({number:x.label,anchor:x.center}))});
  }
  // Exact closest distance from the origin to the final piecewise-triangular
  // boundary, including bevel, face inset, and marking relief. Because the
  // origin lies inside every supported die, this is the centered insphere
  // radius guaranteed to fit inside its solid, even with concave face relief.
  let safeInteriorRadius=Infinity;
  for(const [a,b,c] of glass)
    safeInteriorRadius=Math.min(safeInteriorRadius,originToTriangle(a,b,c));
  if(!(safeInteriorRadius>0)) throw new Error(`${kind} has no centered interior`);
  return {glass,interior:[],ribbons:[],tubes:[],confetti:[],triangleCount:glass.length,
    faceInfo,parameters:{kind,shellMaterial,markMaterial,markingMode,reliefDepth,subdivisions,markScale,
      bevelRadius,faceRelief,compensateRaised},
    mechanicsResolved:true,contactModel:'polyhedron',safeInteriorRadius};
}

/** Keep all of an existing catEyeMarble interior inside a finished die.
 * Scaling one whole inclusion assembly preserves every disjoint component's
 * winding, material and triangle ranges. The die's sphere bound is conservative
 * for non-spherical inclusions and is computed from its final triangles.
 */
export function fitInclusionsInsideDie(die,inclusions,{gap=0.012,scale=1}={}) {
  if(!die||die.contactModel!=='polyhedron'||!(die.safeInteriorRadius>0)||
    !Array.isArray(inclusions?.interior)||!(gap>=0&&gap<die.safeInteriorRadius)||
    !(scale>=0&&scale<=1)) throw new RangeError('invalid die inclusion fit');
  const source=inclusions.interior;
  const maxRadius=source.reduce((largest,face)=>Math.max(largest,
    ...face.slice(0,3).map(p=>Math.hypot(...p))),0);
  const factor=maxRadius>0?(die.safeInteriorRadius-gap)/maxRadius*scale:1;
  const interior=scale===0?[]:source.map(([a,b,c,id,normals])=>
    [a.map(x=>x*factor),b.map(x=>x*factor),c.map(x=>x*factor),id,normals]);
  const parts=(array)=>scale===0?[]:(array||[]).map(part=>({
    ...part,...(part.center?{center:part.center.map(x=>x*factor)}:{})}));
  return {...die,interior,ribbons:parts(inclusions.ribbons),tubes:parts(inclusions.tubes),
    confetti:parts(inclusions.confetti),triangleCount:die.glass.length+interior.length,
    mechanicsResolved:die.mechanicsResolved!==false&&inclusions.mechanicsResolved!==false,
    inclusionFit:{gap,scale,factor,sourceMaxRadius:maxRadius,safeInteriorRadius:die.safeInteriorRadius}};
}
