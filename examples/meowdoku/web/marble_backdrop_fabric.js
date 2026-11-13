// The cloth topology is sampled from the actual Studio backdrop wall faces.
// Positions and material IDs therefore use the same source as Phong and photo.
import {TABLE_TOP_Y} from './marble_studio_table.js';
export const FABRIC_GRID_SIDE=9;
export const FABRIC_GRID_MAX_SIDE=16;
export const FABRIC_ROW_NETWORKS=8, FABRIC_ROW_NODES=256, FABRIC_ROW_FIELDS=38;

function normal(a,b,c){
  const u=b.map((v,i)=>v-a[i]),v=c.map((value,i)=>value-a[i]);
  const n=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]];
  const length=Math.hypot(...n);
  return length>0?n.map(value=>value/length):[0,0,1];
}

function triangulate(positions,material,side){
  const faces=[];
  for(let y=0;y<side-1;y++)for(let x=0;x<side-1;x++){
    const i=y*side+x,p=positions[i],q=positions[i+1],r=positions[i+side+1],s=positions[i+side];
    const n0=normal(p,q,r),n1=normal(p,r,s);
    faces.push([p,q,r,material,[n0,n0,n0]],[p,r,s,material,[n1,n1,n1]]);
  }
  return faces;
}

export function fabricWallMesh(wallFaces,{positions:received,side=FABRIC_GRID_SIDE}={}){
  if(!Array.isArray(wallFaces)||wallFaces.length!==2||wallFaces[0][3]!==wallFaces[1][3])
    throw new RangeError('Fabric requires one solid rectangular backdrop wall');
  if(!Number.isInteger(side)||side<2||side>FABRIC_GRID_MAX_SIDE)
    throw new RangeError('Fabric grid side must be between 2 and 16');
  const source=wallFaces.flatMap(face=>face.slice(0,3));
  const lowX=Math.min(...source.map(point=>point[0])),highX=Math.max(...source.map(point=>point[0]));
  const lowY=Math.min(...source.map(point=>point[1])),highY=Math.max(...source.map(point=>point[1]));
  const corner=(x,y)=>{
    const found=source.find(point=>Math.abs(point[0]-x)<1e-8&&Math.abs(point[1]-y)<1e-8);
    if(!found)throw new RangeError('Backdrop wall corners are not rectangular');
    return found;
  };
  const a=corner(lowX,lowY),b=corner(highX,lowY),c=corner(highX,highY),d=corner(lowX,highY);
  if(received!==undefined)validatePositions(received,side);
  const positions=[];
  for(let y=0;y<side;y++)for(let x=0;x<side;x++){
    const s=x/(side-1),t=y/(side-1);
    positions.push(received?.[y*side+x]?.slice()??[0,1,2].map(axis=>
      (1-t)*((1-s)*a[axis]+s*b[axis])+t*((1-s)*d[axis]+s*c[axis])));
  }
  const material=wallFaces[0][3],faces=triangulate(positions,material,side);
  return {positions,faces,material,columns:side,rows:side};
}

function validatePositions(positions,side){
  if(!Array.isArray(positions)||positions.length!==side**2||
      !positions.every(point=>Array.isArray(point)&&point.length===3&&point.every(Number.isFinite)))
    throw new RangeError(`Fabric positions must contain ${side**2} finite xyz vectors`);
}

export function fabricMeshAtPositions(mesh,received){
  validatePositions(received,mesh.columns);
  const positions=received.map(point=>point.slice());
  return {...mesh,positions,faces:triangulate(positions,mesh.material,mesh.columns)};
}

export function fabricTableDrapePose(mesh,table,{lift=.8,backGap=.15,frontInset=.15}={}){
  if(!table?.enabled||!Number.isFinite(table.width)||!Number.isFinite(table.length)||
     table.width<=0||table.length<=backGap+frontInset||
     ![lift,backGap,frontInset].every(value=>Number.isFinite(value)&&value>=0))
    throw new RangeError('Table drape needs an enabled stage table and valid placement');
  if(!Number.isInteger(mesh?.columns)||mesh.columns<2||mesh.columns>FABRIC_GRID_MAX_SIDE||
     mesh.rows!==mesh.columns||mesh.positions?.length!==mesh.columns**2)
    throw new RangeError('Table drape needs the Studio fabric wall mesh');
  const sourceX=mesh.positions.map(point=>point[0]),lowX=Math.min(...sourceX),highX=Math.max(...sourceX);
  const width=Math.min(highX-lowX,table.width),middle=(lowX+highX)/2;
  const back=-table.length/2-backGap,front=table.length/2-frontInset;
  const positions=mesh.positions.map((_,node)=>{
    const side=mesh.columns,column=node%side,row=Math.floor(node/side);
    return [middle+width*(column/(side-1)-.5),
      TABLE_TOP_Y+lift,back+(front-back)*row/(side-1)];
  });
  return {...mesh,positions,faces:triangulate(positions,mesh.material,mesh.columns)};
}

export function fabricCycloramaPose(mesh,{floorY,extension=1.25,forward=1.5}={}){
  if(!Number.isFinite(floorY)||!Number.isFinite(extension)||extension<1||
     !Number.isFinite(forward)||forward<0)
    throw new RangeError('Cyclorama pose needs a floor and positive extension');
  const side=mesh?.columns;
  if(!Number.isInteger(side)||side<2||mesh.rows!==side||mesh.positions?.length!==side**2)
    throw new RangeError('Cyclorama pose needs a rectangular fabric mesh');
  const highY=Math.max(...mesh.positions.map(point=>point[1]));
  const lowY=Math.min(...mesh.positions.map(point=>point[1]));
  if(highY<=floorY||highY<=lowY)throw new RangeError('Cyclorama mount must stand above floor');
  const length=Math.max((highY-lowY)*extension,highY-floorY+forward);
  const fall=highY-floorY;
  const positions=mesh.positions.map((point,node)=>{
    if(Math.floor(node/side)===side-1)return point.slice();
    const down=length*(1-Math.floor(node/side)/(side-1));
    return [point[0],Math.max(floorY,highY-down),point[2]+Math.max(0,down-fall)];
  });
  return {...mesh,positions,faces:triangulate(positions,mesh.material,side)};
}

export function fabricTableContact(table){
  if(!table?.enabled||!Number.isFinite(table.width)||!Number.isFinite(table.length)||
     table.width<=0||table.length<=0)throw new RangeError('Table contact needs an enabled stage table');
  return new Float64Array([2,TABLE_TOP_Y,-table.width/2,table.width/2,
    -table.length/2,table.length/2]);
}

export function fabricRowConfiguration(mesh,{network=0,identity,stiffness=600,drag=.35,
    gravity=-9.8,frozen=false,attachments}={}){
  if(!Number.isInteger(network)||network<0||network>=FABRIC_ROW_NETWORKS||
      !Number.isSafeInteger(identity)||identity<=0||
      !Number.isFinite(stiffness)||stiffness<0||!Number.isFinite(drag)||drag<0||
      !Number.isFinite(gravity)||typeof frozen!=='boolean')throw new RangeError('Invalid fabric row settings');
  const side=mesh?.columns,count=side**2;
  if(!Number.isInteger(side)||side<2||side>FABRIC_GRID_MAX_SIDE||mesh.rows!==side||mesh.positions?.length!==count)
    throw new RangeError('Fabric mesh must fit the 256-node backdrop batch');
  attachments??=Array.from({length:side},(_,column)=>(side-1)*side+column);
  if(!Array.isArray(attachments)||attachments.length>count)throw new RangeError('Invalid fabric attachment points');
  const anchored=new Map();
  for(const item of attachments){
    const node=typeof item==='number'?item:item?.node,anchor=typeof item==='number'?mesh.positions[node]:item?.anchor;
    const kind=typeof item==='number'?'fixed':item?.kind??'fixed';
    if(!Number.isInteger(node)||node<0||node>=count||anchored.has(node)||
        !['fixed','ring'].includes(kind)||!Array.isArray(anchor)||anchor.length!==3||!anchor.every(Number.isFinite))
      throw new RangeError('Invalid fabric attachment point');
    anchored.set(node,{anchor,kind});
  }
  const data=new Float64Array(FABRIC_ROW_NETWORKS*FABRIC_ROW_NODES*FABRIC_ROW_FIELDS);
  const at=(node,field)=>(network*FABRIC_ROW_NODES+node)*FABRIC_ROW_FIELDS+field;
  data[at(0,37)]=identity;
  for(let y=0;y<side;y++)for(let x=0;x<side;x++){
    const node=y*side+x,point=mesh.positions[node];
    if(!Array.isArray(point)||point.length!==3||!point.every(Number.isFinite))throw new RangeError('Invalid fabric node');
    for(let axis=0;axis<3;axis++){data[at(node,axis)]=point[axis];data[at(node,8+axis)]=anchored.get(node)?.anchor[axis]??point[axis];}
    data[at(node,6)]=1;data[at(node,7)]=anchored.get(node)?.kind==='ring'?2:anchored.has(node)?1:0;
    data[at(node,11)]=1;data[at(node,12)]=stiffness;data[at(node,13)]=drag;data[at(node,15)]=gravity;
    for(const [edge,nx,ny] of [[0,x-1,y],[1,x+1,y],[2,x,y-1],[3,x,y+1]]){
      if(nx<0||nx>=side||ny<0||ny>=side)continue;
      const other=ny*side+nx,neighbor=mesh.positions[other];
      data[at(node,17+edge)]=other;data[at(node,33+edge)]=1;
      for(let axis=0;axis<3;axis++)data[at(node,21+edge*3+axis)]=neighbor[axis]-point[axis];
    }
  }
  return {data,shape:[FABRIC_ROW_NETWORKS,FABRIC_ROW_NODES,FABRIC_ROW_FIELDS],
    control:new Float64Array([1,frozen?1:0])};
}

export function fabricLayerRowConfiguration(mesh,layer,settings={}){
  if(!layer||layer.surfaceBehavior!=='fabric')throw new RangeError('A fabric backdrop layer is required');
  const rings=layer.fabricRingPoints??[],tiebacks=layer.fabricTiebackPoints??[];
  const fixed=layer.fabricAttachmentPoints??(layer.fabricPose==='table'
    ?Array.from({length:mesh.columns},(_,column)=>column)
    :Array.from({length:mesh.columns},(_,column)=>(mesh.rows-1)*mesh.columns+column));
  const replace=new Set([...rings,...tiebacks.map(item=>item.node)]);
  return fabricRowConfiguration(mesh,{...settings,
    attachments:[...fixed.filter(node=>!replace.has(node)),
      ...rings.map(node=>({node,kind:'ring',anchor:mesh.positions[node]})),
      ...tiebacks.map(item=>({node:item.node,kind:'fixed',anchor:item.anchor}))],
    frozen:layer.fabricFrozen??false});
}
