import {pipeFaces,LIGHT_STAND_DIMENSIONS} from './marble_light_stands.js';
const vector=p=>Array.isArray(p)&&p.length===3&&p.every(Number.isFinite);
const unsupported=reason=>({attached:false,reason,faces:[],components:[]});
const poseKey=c=>JSON.stringify([c.eye_m,c.target_m,c.aperture_radius_m??.005,c.aperture_blades??0,c.aperture_rotation_radians??0]);
/** Physical support for an already positioned native camera. The caller owns
 * that camera and its authoritative housing geometry. No body is invented and
 * the camera pose is never edited. Native geometry transport is separate. */
export function cameraMountGeometry({camera,unitM,originM,floorY,bounds,material,housingVerticesM,housingSource}={}){
  if(!camera||!vector(camera.eye_m)||!vector(camera.target_m)||!vector(originM)||!Number.isFinite(unitM)||unitM<=0||
      !Number.isFinite(floorY)||!Number.isSafeInteger(material)||material<0)
    throw new RangeError('Camera mount needs the actual camera pose, metre conversion, floor and shared material ID');
  if(!bounds||!['leftX','rightX','floorY','ceilingY','backZ','frontZ'].every(k=>Number.isFinite(bounds[k]))||bounds.floorY!==floorY)
    throw new RangeError('Camera mount needs the actual room bounds');
  if(housingSource){
    if(housingSource.schema!=='native-camera-housing-mount.v1'||!housingSource.camera||poseKey(housingSource.camera)!==poseKey(camera))
      return unsupported('Native camera housing is missing or belongs to another camera pose/settings. Refresh its geometry before attaching the mount.');
    housingVerticesM=housingSource.housingVerticesM;
  }
  if(!Array.isArray(housingVerticesM)||!housingVerticesM.length)
    return unsupported('Native camera housing vertices are unavailable. The camera remains unattached; no substitute body or attachment was inferred.');
  if(housingVerticesM.some(p=>!vector(p)))throw new RangeError('Camera housing must contain finite world-metre vertices');
  let minY=Infinity;for(const p of housingVerticesM)minY=Math.min(minY,p[1]);
  const bottom=[...new Map(housingVerticesM.filter(p=>p[1]===minY).map(p=>[p.join(','),p])).values()];
  const attachmentM=[0,1,2].map(i=>bottom.reduce((n,p)=>n+p[i],0)/bottom.length);
  const attachment=attachmentM.map((v,i)=>(v-originM[i])/unitM);
  // Nominal support hardware in metres, independent of subject/camera size.
  const columnRadius=.012/unitM,footRadius=.008/unitM,spread=.3/unitM,plateRadius=.025/unitM,plateDepth=.012/unitM;
  const plateBottom=[attachment[0],attachment[1]-plateDepth,attachment[2]],base=[attachment[0],floorY+footRadius,attachment[2]];
  if(plateBottom[1]<=base[1])return unsupported('The native camera underside is too close to or below the room floor for this mount. Move the camera explicitly or leave it unattached.');
  const fits=segment=>{
    const axis=segment.end.map((v,i)=>v-segment.start[i]),length=Math.hypot(...axis);
    return [[0,bounds.leftX,bounds.rightX],[1,bounds.floorY,bounds.ceilingY],[2,bounds.backZ,bounds.frontZ]].every(([i,low,high])=>{
      const r=segment.radius*Math.sqrt(Math.max(0,1-(axis[i]/length)**2));
      return Math.min(segment.start[i],segment.end[i])-r>=low&&Math.max(segment.start[i],segment.end[i])+r<=high;
    });
  };
  let segments;
  for(let rotation=0;rotation<12;rotation++){
    const candidate=[{id:'mounting-plate',start:plateBottom,end:attachment,radius:plateRadius},
      {id:'column',start:base,end:plateBottom,radius:columnRadius}];
    for(let i=0;i<3;i++){const angle=rotation*Math.PI/6+i*2*Math.PI/3;candidate.push({id:`foot-${i}`,start:[...base],
      end:[base[0]+spread*Math.cos(angle),base[1],base[2]+spread*Math.sin(angle)],radius:footRadius});}
    if(candidate.every(fits)){segments=candidate;break;}
  }
  if(!segments)return unsupported('The camera mount plate, column and feet cannot fit this room at the current camera pose. Move the camera explicitly or leave it unattached.');
  const faces=[],components=[],bodyId='native-camera-mount';
  for(const segment of segments){
    const firstTriangle=faces.length,part=pipeFaces(segment,material,0);faces.push(...part);
    components.push({id:segment.id,bodyId,firstTriangle,triangleCount:part.length,sourceConstruction:'nativeCameraMountPipe',
      intendedShape:'capped-polygonal-cylinder',constructionArguments:{start:[...segment.start],end:[...segment.end],radius:segment.radius,
        segments:LIGHT_STAND_DIMENSIONS.segments},contactAdmission:'unverified'});
  }
  return {attached:true,faces,components,bodyId,attachmentM,attachment,source:'actual-native-housing-vertices',
    construction:'camera-floor-mount',contactOwnership:'shared-body-union-required',mechanicalAttachment:'support-plane-contact-not-certified',
    camera:structuredClone(camera)};
}
