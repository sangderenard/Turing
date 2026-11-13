// The cloth topology is sampled from the actual Studio backdrop wall faces.
// Positions and material IDs therefore use the same source as Phong and photo.
import {TABLE_TOP_Y} from './marble_studio_table.js';
export const FABRIC_GRID_SIDE=9;
export const FABRIC_ROW_NETWORKS=8, FABRIC_ROW_NODES=256, FABRIC_ROW_FIELDS=38;

function normal(a,b,c){
  const u=b.map((v,i)=>v-a[i]),v=c.map((value,i)=>value-a[i]);
  const n=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]];
  const length=Math.hypot(...n);
  return length>0?n.map(value=>value/length):[0,0,1];
}

function triangulate(positions,material){
  const faces=[],side=FABRIC_GRID_SIDE;
  for(let y=0;y<side-1;y++)for(let x=0;x<side-1;x++){
    const i=y*side+x,p=positions[i],q=positions[i+1],r=positions[i+side+1],s=positions[i+side];
    const n0=normal(p,q,r),n1=normal(p,r,s);
    faces.push([p,q,r,material,[n0,n0,n0]],[p,r,s,material,[n1,n1,n1]]);
  }
  return faces;
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
  const side=FABRIC_GRID_SIDE;
  if(received!==undefined)validatePositions(received);
  const positions=[];
  for(let y=0;y<side;y++)for(let x=0;x<side;x++){
    const s=x/(side-1),t=y/(side-1);
    positions.push(received?.[y*side+x]?.slice()??[0,1,2].map(axis=>
      (1-t)*((1-s)*a[axis]+s*b[axis])+t*((1-s)*d[axis]+s*c[axis])));
  }
  const material=wallFaces[0][3],faces=triangulate(positions,material);
  return {positions,faces,material,columns:side,rows:side};
}

function validatePositions(positions){
  if(!Array.isArray(positions)||positions.length!==FABRIC_GRID_SIDE**2||
      !positions.every(point=>Array.isArray(point)&&point.length===3&&point.every(Number.isFinite)))
    throw new RangeError('Fabric positions must contain 81 finite xyz vectors');
}

export function fabricMeshAtPositions(mesh,received){
  validatePositions(received);
  const positions=received.map(point=>point.slice());
  return {...mesh,positions,faces:triangulate(positions,mesh.material)};
}

export function fabricTableDrapePose(mesh,table,{lift=.8,backGap=.15,frontInset=.15}={}){
  if(!table?.enabled||!Number.isFinite(table.width)||!Number.isFinite(table.length)||
     table.width<=0||table.length<=backGap+frontInset||
     ![lift,backGap,frontInset].every(value=>Number.isFinite(value)&&value>=0))
    throw new RangeError('Table drape needs an enabled stage table and valid placement');
  if(mesh?.columns!==FABRIC_GRID_SIDE||mesh.rows!==FABRIC_GRID_SIDE||mesh.positions?.length!==81)
    throw new RangeError('Table drape needs the Studio fabric wall mesh');
  const sourceX=mesh.positions.map(point=>point[0]),lowX=Math.min(...sourceX),highX=Math.max(...sourceX);
  const width=Math.min(highX-lowX,table.width),middle=(lowX+highX)/2;
  const back=-table.length/2-backGap,front=table.length/2-frontInset;
  const positions=mesh.positions.map((_,node)=>{
    const column=node%FABRIC_GRID_SIDE,row=Math.floor(node/FABRIC_GRID_SIDE);
    return [middle+width*(column/(FABRIC_GRID_SIDE-1)-.5),
      TABLE_TOP_Y+lift,back+(front-back)*row/(FABRIC_GRID_SIDE-1)];
  });
  return {...mesh,positions,faces:triangulate(positions,mesh.material)};
}

export function fabricTableContact(table){
  if(!table?.enabled||!Number.isFinite(table.width)||!Number.isFinite(table.length)||
     table.width<=0||table.length<=0)throw new RangeError('Table contact needs an enabled stage table');
  return new Float64Array([2,TABLE_TOP_Y,-table.width/2,table.width/2,
    -table.length/2,table.length/2]);
}

export function fabricRowConfiguration(mesh,{network=0,identity,stiffness=600,drag=.35,
    gravity=-9.8,frozen=false,attachments=[72,80]}={}){
  if(!Number.isInteger(network)||network<0||network>=FABRIC_ROW_NETWORKS||
      !Number.isSafeInteger(identity)||identity<=0||
      !Number.isFinite(stiffness)||stiffness<0||!Number.isFinite(drag)||drag<0||
      !Number.isFinite(gravity)||typeof frozen!=='boolean')throw new RangeError('Invalid fabric row settings');
  if(mesh?.columns!==FABRIC_GRID_SIDE||mesh.rows!==FABRIC_GRID_SIDE||mesh.positions?.length!==81)
    throw new RangeError('Fabric mesh must be the 9 by 9 backdrop grid');
  if(!Array.isArray(attachments)||attachments.length>81)throw new RangeError('Invalid fabric attachment points');
  const anchored=new Map();
  for(const item of attachments){
    const node=typeof item==='number'?item:item?.node,anchor=typeof item==='number'?mesh.positions[node]:item?.anchor;
    if(!Number.isInteger(node)||node<0||node>=81||anchored.has(node)||
        !Array.isArray(anchor)||anchor.length!==3||!anchor.every(Number.isFinite))
      throw new RangeError('Invalid fabric attachment point');
    anchored.set(node,anchor);
  }
  const data=new Float64Array(FABRIC_ROW_NETWORKS*FABRIC_ROW_NODES*FABRIC_ROW_FIELDS);
  const at=(node,field)=>(network*FABRIC_ROW_NODES+node)*FABRIC_ROW_FIELDS+field;
  data[at(0,37)]=identity;
  for(let y=0;y<FABRIC_GRID_SIDE;y++)for(let x=0;x<FABRIC_GRID_SIDE;x++){
    const node=y*FABRIC_GRID_SIDE+x,point=mesh.positions[node];
    if(!Array.isArray(point)||point.length!==3||!point.every(Number.isFinite))throw new RangeError('Invalid fabric node');
    for(let axis=0;axis<3;axis++){data[at(node,axis)]=point[axis];data[at(node,8+axis)]=anchored.get(node)?.[axis]??point[axis];}
    data[at(node,6)]=1;data[at(node,7)]=anchored.has(node)?1:0;
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

export function fabricLayerRowConfiguration(mesh,layer,settings={}){
  if(!layer||layer.surfaceBehavior!=='fabric')throw new RangeError('A fabric backdrop layer is required');
  return fabricRowConfiguration(mesh,{...settings,
    attachments:layer.fabricAttachmentPoints??(layer.fabricPose==='table'?[0,8]:[72,80]),
    frozen:layer.fabricFrozen??false});
}
