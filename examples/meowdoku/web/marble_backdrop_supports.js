import {pipeFaces,LIGHT_STAND_DIMENSIONS} from './marble_light_stands.js';
const finiteVector=p=>Array.isArray(p)&&p.length===3&&p.every(Number.isFinite);
const add=(a,b)=>a.map((v,i)=>v+b[i]);
function barBottomAtZ(radius,z){
  // Read the existing pipe's actual polygon edges; its circular bound alone
  // would leave a small gap between the sheet and a twelve-sided crossbar.
  const cap=pipeFaces({start:[0,0,0],end:[1,0,0],radius},0,0).filter(face=>face[4][0][0]===-1);
  const hits=[];
  for(const [,a,b]of cap){
    if(z<Math.min(a[2],b[2])||z>Math.max(a[2],b[2]))continue;
    if(a[2]===b[2]){if(z===a[2])hits.push(a[1],b[1]);}
    else hits.push(a[1]+(b[1]-a[1])*(z-a[2])/(b[2]-a[2]));
  }
  if(!hits.length)throw new Error('Backdrop top is outside the actual pipe cross-section');
  return Math.min(...hits);
}
function fits(segment,bounds){
  const d=segment.end.map((v,i)=>v-segment.start[i]),length=Math.hypot(...d);
  if(!(length>0))return false;
  return [[0,bounds.leftX,bounds.rightX],[1,bounds.floorY,bounds.ceilingY],[2,bounds.backZ,bounds.frontZ]].every(([i,low,high])=>{
    const radius=segment.radius*Math.sqrt(Math.max(0,1-(d[i]/length)**2));
    return Math.min(segment.start[i],segment.end[i])-radius>=low&&Math.max(segment.start[i],segment.end[i])+radius<=high;
  });
}
/** One shared frame for the visible backdrop stack. Its top crossbar touches
 * the top-sheet plane; short arms connect it to floor masts forward of the wall.
 * This checks room/holder bounds, not props, other fixtures or structural loads. */
export function backdropSupportLayout(vertices,floorY,{bounds,unitMm}={}){
  if(!Array.isArray(vertices)||vertices.some(p=>!finiteVector(p)))throw new RangeError('Backdrop holder needs actual finite backdrop vertices');
  if(!vertices.length)return {available:true,segments:[],empty:true};
  if(!bounds||!['leftX','rightX','floorY','ceilingY','backZ','frontZ'].every(k=>Number.isFinite(bounds[k]))||bounds.floorY!==floorY||
      bounds.leftX>=bounds.rightX||bounds.floorY>=bounds.ceilingY||bounds.backZ>=bounds.frontZ||!Number.isFinite(unitMm)||unitMm<=0)
    throw new RangeError('Backdrop holder needs the real room floor/bounds and Studio millimetres per unit');
  const unavailable=reason=>({available:false,segments:[],reason});
  const scale=304.8/unitMm,radius=.035*scale,footRadius=.025*scale,spread=.45*scale,clearance=.02*scale;
  const left=Math.min(...vertices.map(p=>p[0])),right=Math.max(...vertices.map(p=>p[0])),top=Math.max(...vertices.map(p=>p[1]));
  const topPoints=vertices.filter(p=>p[1]===top),topZ=topPoints.reduce((n,p)=>n+p[2],0)/topPoints.length;
  // A sheet can sit only12mm in front of the wall. Shift the round bar within
  // its own radius and solve the actual polygon contact height, rather than put half
  // the bar through the wall or silently move the saved backdrop.
  const barZ=Math.max(topZ,bounds.backZ+radius),offsetZ=barZ-topZ;
  if(offsetZ>=radius)return unavailable('The backdrop top is on or behind the wall; its holder cannot touch it from inside the room. Move the backdrop forward.');
  const barY=top-barBottomAtZ(radius,-offsetZ),mastZ=Math.max(barZ+spread+footRadius+clearance,bounds.backZ+spread+footRadius+clearance);
  const xPositions=[left-spread-footRadius-clearance,right+spread+footRadius+clearance];
  const barStart=[xPositions[0],barY,barZ],barEnd=[xPositions[1],barY,barZ];
  if(barY<=floorY+footRadius)return unavailable('The backdrop top is too low for a holder. Raise or resize the backdrop.');
  const segments=[{id:'crossbar',start:barStart,end:barEnd,radius}];
  for(const [side,x]of xPositions.entries()){
    const base=[x,floorY+footRadius,mastZ],head=[x,barY,mastZ],bar=[x,barY,barZ];
    let chosen=null;
    for(let rotation=0;rotation<12;rotation++){
      const candidate=[{id:`side-${side}/mast`,start:base,end:head,radius},{id:`side-${side}/arm`,start:head,end:bar,radius}];
      for(let i=0;i<3;i++){
        const angle=rotation*Math.PI/6+i*2*Math.PI/3;
        candidate.push({id:`side-${side}/foot-${i}`,start:[...base],end:add(base,[Math.cos(angle)*spread,0,Math.sin(angle)*spread]),radius:footRadius});
      }
      if(candidate.every(segment=>fits(segment,bounds))){chosen=candidate;break;}
    }
    if(!chosen)return unavailable(`Backdrop holder side ${side+1} cannot fit its post and three feet inside the room. Reduce backdrop width/height, move its top, or enlarge the room.`);
    segments.push(...chosen);
  }
  if(!segments.every(segment=>fits(segment,bounds)))return unavailable('The backdrop crossbar extends outside the room. Lower or narrow the backdrop, or enlarge the room.');
  return {available:true,segments,top,topZ,barZ,footprintSource:'actual-backdrop-vertices'};
}
export function backdropSupportFaces(vertices,floorY,material,onPart,{groupId=0,...options}={}){
  if(!Number.isSafeInteger(material)||material<0)throw new RangeError('Backdrop holder needs a shared catalogue material ID');
  const layout=backdropSupportLayout(vertices,floorY,options);
  if(!layout.available)throw new RangeError(`Backdrop holder: ${layout.reason}`);
  if(!layout.segments.length)return [];
  const faces=[],components=[],bodyId='backdrop-holder';
  for(const segment of layout.segments){
    const firstTriangle=faces.length,part=pipeFaces(segment,material,groupId);faces.push(...part);
    components.push({id:segment.id,bodyId,firstTriangle,triangleCount:part.length,sourceConstruction:'backdropSupportPipe',
      intendedShape:'capped-polygonal-cylinder',constructionArguments:{start:[...segment.start],end:[...segment.end],radius:segment.radius,
        segments:LIGHT_STAND_DIMENSIONS.segments},contactAdmission:'unverified'});
  }
  onPart?.({id:bodyId,bodyId,construction:'backdrop-holder',closed:false,convex:false,components,
    contactOwnership:'shared-body-union-required',faces});
  return faces;
}
