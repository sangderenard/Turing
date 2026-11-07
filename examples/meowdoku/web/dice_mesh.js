// Convex dice solids for the marble designer. All faces are outward-wound,
// closed triangles in the same tuple format as marble_mesh.js. Numbering is
// defined on the geometric faces, rather than depending on traversal order.
const TAU = 2 * Math.PI;
const sub = (a,b) => a.map((x,i)=>x-b[i]);
const dot = (a,b) => a.reduce((s,x,i)=>s+x*b[i],0);
const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit = (a) => { const r=Math.hypot(...a); return a.map(x=>x/r); };
const mean = (points) => points[0].map((_,k)=>points.reduce((s,p)=>s+p[k],0)/points.length);

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
  const n=face.normal, tangent=unit(sub(vertices[face.ids[0]],center)), bitangent=cross(n,tangent);
  const inradius=Math.min(...face.ids.map((id,k)=>{
    const a=vertices[id],b=vertices[face.ids[(k+1)%face.ids.length]];
    return Math.abs(dot(cross(sub(b,a),sub(center,a)),n))/Math.hypot(...sub(b,a));
  }));
  for(const mark of marks) {
    const location=kindLocation(mark,center,inradius,scale);
    const offset=sub(point,location.center);
    const x=dot(offset,tangent)/location.height*1.5+0.5;
    const y=dot(offset,bitangent)/location.height+0.5;
    if(inText(x,y,mark.label)) return true;
  }
  return false;
}
function kindLocation(mark,faceCenter,inradius,scale) {
  const isCorner=Math.hypot(...sub(mark.center,faceCenter))>inradius*0.5;
  return {center:isCorner?mark.center.map((x,i)=>faceCenter[i]*0.48+x*0.52):mark.center,
    height:inradius*(isCorner?0.46:scale)};
}

/**
 * `markingMode` is "flat", "raised", or "recessed". Relief displaces the
 * closed shell surface, so marbleMass integrates its actual triangulated volume.
 * Mark material is the optical Pluck material ID; bulk relief retains the body
 * density until separate material-domain geometry is available.
 */
export function diceSolid(kind,{shellMaterial,markMaterial=shellMaterial,markingMode='flat',
  reliefDepth=0.02,subdivisions=kind==='d4'?48:24,markScale=0.72}={}) {
  if(!Number.isInteger(shellMaterial)||shellMaterial<0||!Number.isInteger(markMaterial)||markMaterial<0)
    throw new TypeError('shell and marking material IDs are required');
  if(!['flat','raised','recessed'].includes(markingMode)||!(reliefDepth>=0&&reliefDepth<=0.08)||
     !Number.isInteger(subdivisions)||subdivisions<12||subdivisions>64||
     !(markScale>=0.3&&markScale<=0.9)) throw new RangeError('unsupported marking geometry');
  let vertices=verticesFor(kind), radius=Math.max(...vertices.map(v=>Math.hypot(...v)));
  vertices=vertices.map(v=>v.map(x=>x/radius));
  const facets=hull(vertices), expected={d4:4,d6:6,d8:8,d10:10,d12:12,d20:20}[kind];
  if(facets.length!==expected) throw new Error(`${kind} hull has ${facets.length} faces`);
  const numbered=numberFaces(facets,kind), glass=[], faceInfo=[];
  for(let faceIndex=0;faceIndex<numbered.length;faceIndex++) {
    const {face,number}=numbered[faceIndex], polygon=face.ids.map(i=>vertices[i]);
    const center=mean(polygon), markings=markingsFor(face,number,kind,vertices);
    const mark=(p)=>evaluateMark(p,face,markings,vertices,markScale);
    const displace=(p)=>p.map((x,i)=>x+face.normal[i]*(markingMode==='flat'?0:
      (markingMode==='raised'?reliefDepth:-reliefDepth)*(mark(p)?1:0)));
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
    faceInfo.push({faceIndex,number,normal:face.normal,vertexIds:[...face.ids],
      markings:markings.map(x=>({number:x.label,anchor:x.center}))});
  }
  return {glass,interior:[],ribbons:[],tubes:[],confetti:[],triangleCount:glass.length,
    faceInfo,parameters:{kind,shellMaterial,markMaterial,markingMode,reliefDepth,subdivisions,markScale},
    mechanicsResolved:true,contactModel:'polyhedron'};
}
