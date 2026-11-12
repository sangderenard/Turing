// The cloth topology is sampled from the actual Studio backdrop wall faces.
// Positions and material IDs therefore use the same source as Phong and photo.
export const FABRIC_GRID_SIDE=9;
export const FABRIC_ROW_NETWORKS=8, FABRIC_ROW_NODES=256, FABRIC_ROW_FIELDS=38;

function normal(a,b,c){
  const u=b.map((v,i)=>v-a[i]),v=c.map((value,i)=>value-a[i]);
  const n=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]];
  const length=Math.hypot(...n);
  return length>0?n.map(value=>value/length):[0,0,1];
}

export function fabricWallMesh(wallFaces,{positions:received}={}){
  if(!Array.isArray(wallFaces)||wallFaces.length!==2||wallFaces[0][3]!==wallFaces[1][3])
    throw new RangeError('Fabric requires one solid rectangular backdrop wall');
  const source=wallFaces.flatMap(face=>face.slice(0,3));
  const lowX=Math.min(...source.map(point=>point[0])),highX=Math.max(...source.map(point=>point[0]));
  const lowY=Math.min(...source.map(point=>point[1])),highY=Math.max(...source.map(point=>point[1]));
  const corner=(x,y)=>{
    const found=source.find(point=>Math.abs(point[0]-x)<1e-8&&Math.abs(point[1]-y)<1e-8);
    if(!found)throw new RangeError('Backdrop wall corners are not rectangular');
    return found;
  };
  const a=corner(lowX,lowY),b=corner(highX,lowY),c=corner(highX,highY),d=corner(lowX,highY);
  const side=FABRIC_GRID_SIDE,count=side*side;
  if(received!==undefined&&(!Array.isArray(received)||received.length!==count||
      !received.every(point=>Array.isArray(point)&&point.length===3&&point.every(Number.isFinite))))
    throw new RangeError('Fabric positions must contain 81 finite xyz vectors');
  const positions=[];
  for(let y=0;y<side;y++)for(let x=0;x<side;x++){
    const s=x/(side-1),t=y/(side-1);
    positions.push(received?.[y*side+x]?.slice()??[0,1,2].map(axis=>
      (1-t)*((1-s)*a[axis]+s*b[axis])+t*((1-s)*d[axis]+s*c[axis])));
  }
  const faces=[],material=wallFaces[0][3];
  for(let y=0;y<side-1;y++)for(let x=0;x<side-1;x++){
    const i=y*side+x,p=positions[i],q=positions[i+1],r=positions[i+side+1],s=positions[i+side];
    const n0=normal(p,q,r),n1=normal(p,r,s);
    faces.push([p,q,r,material,[n0,n0,n0]],[p,r,s,material,[n1,n1,n1]]);
  }
  return {positions,faces,material,columns:side,rows:side};
}

export function fabricRowConfiguration(mesh,{network=0,identity,stiffness=600,drag=.35,
    gravity=-9.8,frozen=false}={}){
  if(!Number.isInteger(network)||network<0||network>=FABRIC_ROW_NETWORKS||
      !Number.isSafeInteger(identity)||identity<=0||
      !Number.isFinite(stiffness)||stiffness<0||!Number.isFinite(drag)||drag<0||
      !Number.isFinite(gravity)||typeof frozen!=='boolean')throw new RangeError('Invalid fabric row settings');
  if(mesh?.columns!==FABRIC_GRID_SIDE||mesh.rows!==FABRIC_GRID_SIDE||mesh.positions?.length!==81)
    throw new RangeError('Fabric mesh must be the 9 by 9 backdrop grid');
  const data=new Float64Array(FABRIC_ROW_NETWORKS*FABRIC_ROW_NODES*FABRIC_ROW_FIELDS);
  const at=(node,field)=>(network*FABRIC_ROW_NODES+node)*FABRIC_ROW_FIELDS+field;
  data[at(0,37)]=identity;
  for(let y=0;y<FABRIC_GRID_SIDE;y++)for(let x=0;x<FABRIC_GRID_SIDE;x++){
    const node=y*FABRIC_GRID_SIDE+x,point=mesh.positions[node];
    if(!Array.isArray(point)||point.length!==3||!point.every(Number.isFinite))throw new RangeError('Invalid fabric node');
    for(let axis=0;axis<3;axis++){data[at(node,axis)]=point[axis];data[at(node,8+axis)]=point[axis];}
    data[at(node,6)]=1;data[at(node,7)]=(y===FABRIC_GRID_SIDE-1&&(x===0||x===FABRIC_GRID_SIDE-1))?1:0;
    data[at(node,11)]=1;data[at(node,12)]=stiffness;data[at(node,13)]=drag;data[at(node,15)]=gravity;
    for(const [edge,nx,ny] of [[0,x-1,y],[1,x+1,y],[2,x,y-1],[3,x,y+1]]){
      if(nx<0||nx>=FABRIC_GRID_SIDE||ny<0||ny>=FABRIC_GRID_SIDE)continue;
      const other=ny*FABRIC_GRID_SIDE+nx,neighbor=mesh.positions[other];
      data[at(node,17+edge)]=other;data[at(node,33+edge)]=1;
      for(let axis=0;axis<3;axis++)data[at(node,21+edge*3+axis)]=neighbor[axis]-point[axis];
    }
  }
  return {data,shape:[FABRIC_ROW_NETWORKS,FABRIC_ROW_NODES,FABRIC_ROW_FIELDS],
    control:new Float64Array([1,frozen?1:0])};
}
