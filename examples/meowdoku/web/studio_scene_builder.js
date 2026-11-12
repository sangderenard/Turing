// Studio's existing scene construction, shared by the page and non-browser callers.
// Inputs are captured values. No page, controls, asset fetch, renderer or job.
import {studioLightingRackGeometry} from './marble_lighting_racks.js';
import {wallRelativeFixtureDefaults} from './marble_light_placement.js';
import {applyLightTools,defaultLightTools} from './marble_light_tools.js';
import {applyFixtureSizes,defaultFixtureSizes} from './marble_fixture_sizes.js';
import {resolveLightingPoses} from './marble_lighting.js';
import {STUDIO_LIGHTS} from './marble_light_strengths.js';
import {tableLightRoles,tableLightModifierFaces} from './marble_table_lights.js';
import {studioRoomBounds,studioRoomFaces} from './marble_room.js';
import {stageProgrammerFaces} from './marble_stage_programmer.js';
import {defaultStudioTable,tableFloorDrop,studioTableFaces,FLOOR_FIXTURE_INDICES} from './marble_studio_table.js';
import {studioGoboFaces} from './marble_studio_gobos.js';
import {studioLightStandFaces} from './marble_light_stand_settings.js';
import {backdropSupportFaces} from './marble_backdrop_supports.js';
import {backdropFaces,defaultBackdropLayers} from './marble_backdrop.js';
import {shopLightEmissionRecords,shopLightEmitterAllocations,writeShopLightEmitterRows,studioShopLightGeometry} from './marble_shop_lights.js';
import {studioScaleTransform} from './marble_studio_scale.js';
import {STUDIO_FIXED_MATERIAL_ROWS,STUDIO_RUNTIME_MATERIAL_ROWS,studioMaterialExtras,studioMaterialBase,savedMaterialRows} from './marble_materials.js';
import {sceneMaterialEngineContract,validateSceneMaterialEdits} from './marble_scene_materials.js';
import {writeStudioFixtureRows} from './marble_studio_scene.js';
import {validateStudioLighting} from './marble_studio_lighting.js';
import {studioLightingFields} from './studio_lighting_fields.js';
import {RESPONSE_INFORMATION,fixtureResponseLineage} from './marble_response_lineage.js';
const rgb = value => [1,3,5].map(k => parseInt(value.slice(k,k+2),16)/255);

// The three room fixtures that do NOT track the camera: fixed positions in
// Studio's own world units, shared verbatim between the live Phong preview
// (frame()) and the ray-traced "Current Studio orbit and zoom" photo bridge
// below, so both are genuinely the same room instead of two independently
// hand-tuned approximations of it. Key positions are mirrored left/right
// off the same magnitude sliders (ROOM_FIXTURES holds only the fixtures
// whose position isn't already slider-authored).
const ROOM_FIXTURES = {
  // Fresh rigs retreat 20% in XZ; explicit profile poses always override these.
  background: [0, -0.95, -2.22],
  kickerL: [-1.92, -0.95, -3.12],
  kickerR: [ 1.92, -0.95, -3.12],
  // Two independently-controlled fills, both fixed room objects (do NOT
  // track the live camera -- a camera-tracking version put a huge panel
  // directly in front of the lens; a version baked from the camera-eye
  // axis instead floated mid-air next to the subject). Lower: on the
  // floor, out toward the camera side, aimed up. Upper: overhead, aimed
  // straight down -- a real two-point bounce-fill rig, each independently
  // powered via its own slider.
  fillLower: [0, -0.95, 1.6],
  fillUpper: [0, 5.2, 0.3],
};
// A flat quad facing `dir` (unit vector, the direction toward the subject),
// as two triangles for mesh(). This is the actual visible shape of a light
// fixture -- real geometry with a real emissive material, not a point.
// `groupId` matches the real uLightGroupId registered for this fixture (see
// frame()), so the fixture's own fragments skip its own registered light
// (aGroupId == uLightGroupId[i] exclusion in the shader).
export function quadFace(center, dir, halfW, halfH, materialId, groupId=0) {
  let worldUp=[0,1,0];
  if (Math.abs(dir[1])>0.95) worldUp=[1,0,0];
  const cross3=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const norm3=(v)=>{const n=Math.max(1e-9,Math.hypot(...v));return v.map((x)=>x/n);};
  const right=norm3(cross3(dir,worldUp));
  const up=norm3(cross3(right,dir));
  const add=(a,b,s)=>a.map((x,i)=>x+b[i]*s);
  const c00=add(add(center,right,-halfW),up,-halfH), c10=add(add(center,right,halfW),up,-halfH);
  const c11=add(add(center,right,halfW),up,halfH), c01=add(add(center,right,-halfW),up,halfH);
  const n=[dir,dir,dir];
  return [[c00,c10,c11,materialId,n,groupId],[c00,c11,c01,materialId,n,groupId]];
}
// A small round disc facing `dir`, as a triangle fan. Used for the
// background and kicker fixtures, which are small round practicals sitting
// on the floor, not big rectangular panels.
export function circleFace(center, dir, radius, materialId, segments=12, groupId=0) {
  let worldUp=[0,1,0];
  if (Math.abs(dir[1])>0.95) worldUp=[1,0,0];
  const cross3=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const norm3=(v)=>{const n=Math.max(1e-9,Math.hypot(...v));return v.map((x)=>x/n);};
  const right=norm3(cross3(dir,worldUp));
  const up=norm3(cross3(right,dir));
  const add=(a,b,s)=>a.map((x,i)=>x+b[i]*s);
  const n=[dir,dir,dir];
  const faces=[];
  for (let i=0;i<segments;i++) {
    const a0=(i/segments)*Math.PI*2, a1=((i+1)/segments)*Math.PI*2;
    const p0=add(add(center,right,Math.cos(a0)*radius),up,Math.sin(a0)*radius);
    const p1=add(add(center,right,Math.cos(a1)*radius),up,Math.sin(a1)*radius);
    faces.push([center,p0,p1,materialId,n,groupId]);
  }
  return faces;
}
// A real open-fronted box behind a quad light panel: a back wall plus four
// side walls, so the housing reads as an actual 3D container the light
// sits inside (like a softbox/barn-door body) -- not a flat disk floating
// in the middle of the same emission lobe, which never blocks anything.
export function boxHousing(center, dir, halfW, halfH, depth, materialId, groupId=0) {
  let worldUp=[0,1,0];
  if (Math.abs(dir[1])>0.95) worldUp=[1,0,0];
  const cross3=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const norm3=(v)=>{const n=Math.max(1e-9,Math.hypot(...v));return v.map((x)=>x/n);};
  const right=norm3(cross3(dir,worldUp));
  const up=norm3(cross3(right,dir));
  const add=(a,b,s)=>a.map((x,i)=>x+b[i]*s);
  const back=add(center,dir,-depth);
  const f00=add(add(center,right,-halfW),up,-halfH), f10=add(add(center,right,halfW),up,-halfH);
  const f11=add(add(center,right,halfW),up,halfH), f01=add(add(center,right,-halfW),up,halfH);
  const b00=add(add(back,right,-halfW),up,-halfH), b10=add(add(back,right,halfW),up,-halfH);
  const b11=add(add(back,right,halfW),up,halfH), b01=add(add(back,right,-halfW),up,halfH);
  const nDir=[dir,dir,dir], nRight=[right,right,right], nLeft=right.map((v)=>-v), nLeftT=[nLeft,nLeft,nLeft];
  const nUp=[up,up,up], nDown=up.map((v)=>-v), nDownT=[nDown,nDown,nDown];
  return [
    [b00,b10,b11,materialId,nDir,groupId],[b00,b11,b01,materialId,nDir,groupId],             // back wall
    [f10,b10,b11,materialId,nRight,groupId],[f10,b11,f11,materialId,nRight,groupId],         // right wall
    [f01,b01,b00,materialId,nLeftT,groupId],[f01,b00,f00,materialId,nLeftT,groupId],         // left wall
    [f11,b11,b01,materialId,nUp,groupId],[f11,b01,f01,materialId,nUp,groupId],               // top wall
    [f00,b00,b10,materialId,nDownT,groupId],[f00,b10,f10,materialId,nDownT,groupId],         // bottom wall
  ];
}
// A real open-fronted tube behind a round light disc: a circular back cap
// plus a cylindrical side wall, the round-fixture equivalent of boxHousing.
export function tubeHousing(center, dir, radius, depth, materialId, groupId=0, segments=12) {
  let worldUp=[0,1,0];
  if (Math.abs(dir[1])>0.95) worldUp=[1,0,0];
  const cross3=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const norm3=(v)=>{const n=Math.max(1e-9,Math.hypot(...v));return v.map((x)=>x/n);};
  const right=norm3(cross3(dir,worldUp));
  const up=norm3(cross3(right,dir));
  const add=(a,b,s)=>a.map((x,i)=>x+b[i]*s);
  const back=add(center,dir,-depth);
  const faces=[...circleFace(back, dir, radius, materialId, segments, groupId)];
  for (let i=0;i<segments;i++) {
    const a0=(i/segments)*Math.PI*2, a1=((i+1)/segments)*Math.PI*2;
    const radial=(a)=>norm3(right.map((v,k)=>v*Math.cos(a)+up[k]*Math.sin(a)));
    const rim=(c,a)=>add(add(c,right,Math.cos(a)*radius),up,Math.sin(a)*radius);
    const f0=rim(center,a0), f1=rim(center,a1), r0=rim(back,a0), r1=rim(back,a1);
    const n0=radial(a0), n1=radial(a1);
    faces.push([f0,r0,r1,materialId,[n0,n0,n1],groupId],[f0,r1,f1,materialId,[n0,n1,n1],groupId]);
  }
  return faces;
}
// A snoot/gobo tube extending FORWARD from a light disc, open at both
// ends. Physically, it narrows the disc's visible cone: a viewer close to
// the `dir` axis (the subject this fixture is aimed at) still sees straight
// down the open tube to the glowing disc, but a viewer off to the side
// (the camera, watching from a much wider angle) has the opaque tube wall
// between them and the disc -- exactly how a real snoot/barn-door hides a
// kicker from camera while still letting it reach the subject.
export function snootTube(center, dir, radius, length, materialId, groupId=0, segments=12, exitRadius=radius) {
  let worldUp=[0,1,0];
  if (Math.abs(dir[1])>0.95) worldUp=[1,0,0];
  const cross3=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const norm3=(v)=>{const n=Math.max(1e-9,Math.hypot(...v));return v.map((x)=>x/n);};
  const right=norm3(cross3(dir,worldUp));
  const up=norm3(cross3(right,dir));
  const add=(a,b,s)=>a.map((x,i)=>x+b[i]*s);
  const tip=add(center,dir,length);
  const faces=[];
  for (let i=0;i<segments;i++) {
    const a0=(i/segments)*Math.PI*2, a1=((i+1)/segments)*Math.PI*2;
    const radial=(a)=>norm3(right.map((v,k)=>exitRadius===radius?-(v*Math.cos(a)+up[k]*Math.sin(a)):-(v*Math.cos(a)+up[k]*Math.sin(a))+(exitRadius-radius)/length*dir[k])); // inward-facing wall
    const rim=(c,a,r=radius)=>add(add(c,right,Math.cos(a)*r),up,Math.sin(a)*r);
    const f0=rim(center,a0), f1=rim(center,a1), r0=rim(tip,a0,exitRadius), r1=rim(tip,a1,exitRadius);
    const n0=radial(a0), n1=radial(a1);
    faces.push([f0,r0,r1,materialId,[n0,n0,n1],groupId],[f0,r1,f1,materialId,[n0,n1,n1],groupId]);
  }
  return faces;
}
// Debug-only: real extruded glyph geometry (the same digit/letter meshes
// exported for chess/wordle tiles, assets.meshes) placed at a world
// position, for marking fixture coordinates directly in the scene instead
// of guessing from screenshots.
export function computeStudioFixtureRoles({lighting,ids,fixtureMaterialBase,subjectPosition=[0,0,0],solo=null,emissionProfiles=null}) {
  const FIXTURE_MAT_BASE=fixtureMaterialBase,control=id=>lighting.controls[id];
  const keyMagX=Math.abs(+control("light-x")/100), keyY=+control("light-y")/100, keyZ=+control("light-z")/100;
  const keyLeft=[-keyMagX, keyY, keyZ], keyRight=[keyMagX, keyY, keyZ];
  const {background,kickerL,kickerR,fillLower,fillUpper}=ROOM_FIXTURES;
  const keyFilterLeft=rgb(control("key-filter-left"));
  const keyFilterRight=rgb(control("key-filter-right"));
  const bgFilter=rgb(control("background-filter"));
  const kickerLeftFilter=rgb(control("kicker-left-filter"));
  const kickerRightFilter=rgb(control("kicker-right-filter"));
  const fillLowerFilter=rgb(control("fill-lower-filter"));
  const fillUpperFilter=rgb(control("fill-upper-filter"));
  // Roles: 0-1 keys (independently filtered, mirrored), 2 background,
  // 3-4 kickers, 5-6 fills (floor+sky) -- every light has its own live
  // filter colour, not a hardcoded role constant.
  const roleColor=[keyFilterLeft,keyFilterRight,bgFilter,kickerLeftFilter,kickerRightFilter,fillLowerFilter,fillUpperFilter,
    rgb(control("table-left-filter")),rgb(control("table-right-filter")),rgb(control("under-filter"))];
  const filterIds=['key-filter-left','key-filter-right','background-filter','kicker-left-filter',
    'kicker-right-filter','fill-lower-filter','fill-upper-filter','table-left-filter',
    'table-right-filter','under-filter'];
  const bare=emissionProfiles?.records?.find(record=>record.key==='led_cool_white')?.preview_chromaticity;
  for(let i=0;i<roleColor.length;i++)if(control(`${filterIds[i]}-off`))roleColor[i]=bare??[1,1,1];
  const roleStrength=STUDIO_LIGHTS.map(({id,strengthId})=>solo&&solo!==id?0:Number(control(strengthId))/10);
  const keyShape=control("key-shape"), keyIsHard=keyShape==="grid"||keyShape==="snoot";
  const aimAt=(p)=>{const d=p.map((v)=>-v);const n=Math.max(1e-6,Math.hypot(...d));return d.map((v)=>v/n);};
  // Kickers graze the subject's near edge (their own side) rather than
  // hitting it face-on dead-centre -- a real rim/glancing kicker aim, not
  // a second key light.
  const aimGlance=(p,edgeX)=>{const d=[edgeX-p[0],0.1-p[1],-p[2]];const n=Math.max(1e-6,Math.hypot(...d));return d.map((v)=>v/n);};
  // The background practical does NOT light the subject: it sits between
  // the subject and the back wall, tilted up and back so its beam climbs
  // the wall above it (a real uplight splash, not a flat horizontal aim).
  const norm3=(v)=>{const n=Math.max(1e-9,Math.hypot(...v));return v.map((x)=>x/n);};
  const backgroundDir=norm3([0,0.55,-1]);
  const keyHalf=keyIsHard?0.10:0.38;
  let fixtureRoles=[
    {pos:keyLeft,    dir:aimAt(keyLeft),    halfW:keyHalf, halfH:keyHalf, mat:FIXTURE_MAT_BASE+0, roleIndex:0, round:false, shape:keyShape},
    {pos:keyRight,   dir:aimAt(keyRight),   halfW:keyHalf, halfH:keyHalf, mat:FIXTURE_MAT_BASE+1, roleIndex:1, round:false, shape:keyShape},
    {pos:background, dir:backgroundDir,     radius:0.30, mat:FIXTURE_MAT_BASE+2, roleIndex:2, round:true, shape:"bare", gobo:true, shroudLength:1.2, shroudRadius:0.375},
    {pos:kickerL,    dir:aimGlance(kickerL,-0.32), radius:0.20, mat:FIXTURE_MAT_BASE+3, roleIndex:3, round:true, shape:"bare", gobo:true},
    {pos:kickerR,    dir:aimGlance(kickerR, 0.32), radius:0.20, mat:FIXTURE_MAT_BASE+4, roleIndex:4, round:true, shape:"bare", gobo:true},
    {pos:fillLower,  dir:aimAt(fillLower),  halfW:0.30, halfH:0.30, mat:FIXTURE_MAT_BASE+5, roleIndex:5, round:false, shape:"bare"},
    {pos:fillUpper,  dir:aimAt(fillUpper),  halfW:0.30, halfH:0.30, mat:FIXTURE_MAT_BASE+6, roleIndex:6, round:false, shape:"bare"},
    ...tableLightRoles(FIXTURE_MAT_BASE),
  ];
  const sizes=lighting.fixtureSizes??defaultFixtureSizes();
  fixtureRoles=applyFixtureSizes(fixtureRoles,sizes);
  fixtureRoles=applyLightTools(fixtureRoles,lighting.lightTools??defaultLightTools(sizes),sizes,keyShape);
  if (lighting.poses) {
    const poses=resolveLightingPoses(lighting.poses,STUDIO_LIGHTS.map(light=>light.id),subjectPosition);
    for (const [i,role] of fixtureRoles.entries()) {
      role.pos=poses[i].pos; role.dir=poses[i].dir;
    }
  } else if ((lighting.table??defaultStudioTable()).enabled) {
    // New-set defaults rest their real housings above the lowered floor.
    // Loaded profiles always use their exact saved world-space poses instead.
    const roomBounds=studioRoomBounds(Number(control("room-scale")||3),tableFloorDrop(lighting.table??defaultStudioTable()));
    fixtureRoles=wallRelativeFixtureDefaults(fixtureRoles,roomBounds,role=>buildStudioFixtureFaces([role],0,undefined,ids).flatMap(face=>face.slice(0,3)),subjectPosition);
    const floorY=roomBounds.floorY;
    for(const index of FLOOR_FIXTURE_INDICES) {
      const role=fixtureRoles[index];
      if(role.defaultFloorHeight!==undefined){role.pos[1]=floorY+role.defaultFloorHeight;continue;}
      const faces=buildStudioFixtureFaces([role],0,undefined,ids);
      const low=Math.min(...faces.flatMap(face=>face.slice(0,3).map(p=>p[1])));
      role.pos=role.pos.map((v,i)=>v+(i===1?floorY+0.02-low:0));
    }
  }
  return {fixtureRoles, roleColor, roleStrength};
}
export function buildStudioFixtureFaces(fixtureRoles, housingMat, onPart, ids={}) {
  const faces=[];
  for (const role of fixtureRoles) {
    const first=faces.length,components=[],componentGroups=[];
    const roleId=STUDIO_LIGHTS[role.roleIndex].id,bodyId=`fixture-${roleId}`;
    const describe=(id,start,count,sourceConstruction,intendedShape,arguments_)=>{
      if(!onPart)return;
      components.push({id,bodyId,firstTriangle:start-first,triangleCount:count,sourceConstruction,intendedShape,
        ...(arguments_?{constructionArguments:arguments_}:{}),contactAdmission:"unverified"});
    };
    const size=role.round?role.radius:Math.max(role.halfW,role.halfH);
    // A real 3D box/tube housing the light sits inside, slightly larger
    // than the panel and with real depth/side walls -- reads as an
    // actual containing body, not a bigger disk floating in the same
    // emission lobe.
    const depth=role.housingDepth??size*1.4;
    // quadFace/circleFace historically wound opposite their explicit normals.
    // Spectral derives emission direction from winding, not those normals:
    // swap emitter vertices so all ten photo sources face their authored aim.
    // Positions, materials, powers and the non-emissive housing stay unchanged.
    if (role.round) {
      const housingStart=faces.length;
      faces.push(...tubeHousing(role.pos, role.dir, role.radius*1.25, depth, housingMat, role.mat));
      describe("housing/back",housingStart,12,"circleFace","triangle-fan",
        {center:role.pos.map((v,i)=>v+role.dir[i]*(-depth)),dir:role.dir,radius:role.radius*1.25,segments:12});
      for(let i=0;i<12;i++)describe(`housing/wall-${i}`,housingStart+12+i*2,2,"tubeHousing/rim","quad-strip");
      if(onPart)componentGroups.push({id:"housing",firstTriangle:housingStart-first,triangleCount:36,sourceConstruction:"tubeHousing",
        intendedShape:"open-front-polygonal-tube",constructionArguments:{center:role.pos,dir:role.dir,radius:role.radius*1.25,depth,segments:12},
        contactAdmission:"unverified"});
      const emitter=circleFace(role.pos, role.dir, role.radius, role.mat, 12, role.mat);
      const emitterStart=faces.length;
      faces.push(...emitter.map(([a,b,c,m,n,g])=>[a,c,b,m,[n[0],n[2],n[1]],g]));
      describe("emitter",emitterStart,12,"circleFace/reversed","triangle-fan",{center:role.pos,dir:role.dir,radius:role.radius,segments:12});
      if (role.gobo) {
        const shroudStart=faces.length,radius=role.shroudRadius??role.radius*1.1,length=role.shroudLength??size*2.2;
        const offset=role.shroudOffset??0,exitRadius=role.shroudExitRadius??radius;
        // An axial mount extension is a real opaque collar, not an open gap.
        if(offset>0){
          faces.push(...snootTube(role.pos,role.dir,radius,offset,housingMat,role.mat));
          for(let i=0;i<12;i++)describe(`shroud/mount-wall-${i}`,shroudStart+i*2,2,"snootTube/rim","quad-strip");
        }
        const center=offset>0?role.pos.map((v,i)=>v+role.dir[i]*offset):role.pos,taperStart=faces.length;
        faces.push(...snootTube(center,role.dir,radius,length,housingMat,role.mat,12,exitRadius));
        for(let i=0;i<12;i++)describe(`shroud/wall-${i}`,taperStart+i*2,2,"snootTube/rim","quad-strip");
        if(onPart)componentGroups.push({id:"shroud",firstTriangle:shroudStart-first,triangleCount:faces.length-shroudStart,sourceConstruction:"snootTube",
          intendedShape:exitRadius===radius?"open-ended-polygonal-tube":"open-ended-polygonal-frustum",constructionArguments:{center:role.pos,dir:role.dir,radius,
            length,offset,exitRadius,segments:12,toolId:role.shroudToolId},contactAdmission:"unverified"});
      }
    } else {
      const housingStart=faces.length;
      faces.push(...boxHousing(role.pos, role.dir, role.halfW*1.25, role.halfH*1.25, depth, housingMat, role.mat));
      for(const [i,name]of ["back","right","left","top","bottom"].entries())
        describe(`housing/${name}`,housingStart+i*2,2,"boxHousing/face","quad-sheet");
      if(onPart)componentGroups.push({id:"housing",firstTriangle:housingStart-first,triangleCount:10,sourceConstruction:"boxHousing",
        intendedShape:"open-front-box",constructionArguments:{center:role.pos,dir:role.dir,halfW:role.halfW*1.25,halfH:role.halfH*1.25,depth},
        contactAdmission:"unverified"});
      const emitter=quadFace(role.pos, role.dir, role.halfW, role.halfH, role.mat, role.mat);
      const emitterStart=faces.length;
      faces.push(...emitter.map(([a,b,c,m,n,g])=>[a,c,b,m,[n[0],n[2],n[1]],g]));
      describe("emitter",emitterStart,2,"quadFace/reversed","quad-sheet",{center:role.pos,dir:role.dir,halfW:role.halfW,halfH:role.halfH});
    }
    const modifierStart=faces.length;
    const modifierRole={...role};
    for(const [name,key]of [['gridInteriorMaterial','gridInteriorMaterialId'],['gridOuterMaterial','gridOuterMaterialId'],['gridAccentMaterial','gridAccentMaterialId']]){
      const material=role[name]??'housing';if(material!=='housing'&&!Number.isInteger(ids[material]))throw Error('Unavailable grid material '+material);modifierRole[key]=material==='housing'?housingMat:ids[material];
    }
    faces.push(...tableLightModifierFaces(modifierRole,housingMat,onPart?component=>components.push({...component,id:`modifier/${component.id}`,
      bodyId,firstTriangle:modifierStart-first+component.firstTriangle}):undefined));
    if(faces.length>modifierStart)if(onPart)componentGroups.push({id:"modifier",firstTriangle:modifierStart-first,triangleCount:faces.length-modifierStart,
      sourceConstruction:"tableLightModifierFaces",intendedShape:role.tableStand?"base-and-post":"intersecting-grid-bars",contactAdmission:"unverified"});
    onPart?.({id:roleId,bodyId,construction:"fixture-housing-emitter-modifier",closed:false,convex:false,
      components,componentGroups,contactOwnership:"shared-body-union-required",faces:faces.slice(first)});
  }
  return faces;
}

// The same calls previously owned by the page, with explicit captured values.
export function createStudioSceneGeometry({lighting,materials,subject,geometrySubject=subject}) {
  const {ids,fixtureMaterialBase}=materials,control=id=>lighting.controls[id];
  const table=lighting.table??defaultStudioTable();
  const bounds=studioRoomBounds(Number(control('room-scale')||3),tableFloorDrop(table));
  const options={bounds,unitMm:geometrySubject.unitM*1000};
  const housingMat=fixtureMaterialBase+STUDIO_LIGHTS.length;
  const surfaceMaterial=(prefix,a,b)=>{
    const pattern=control(`${prefix}-pattern`)||'solid';
    const matA=ids[control(`${prefix}-material-a`)]??ids.stucco_plaster;
    const matB=ids[control(`${prefix}-material-b`)]??matA;
    return pattern==='checker'?((a+b)&1?matA:matB):matA;
  };
  const roomFaces=onSurface=>studioRoomFaces({scale:Number(control('room-scale')||3),floorDrop:tableFloorDrop(table),
    windowOn:!!control('window-glass'),glassMaterial:ids.marble_clear_glass,material:surfaceMaterial,onSurface});
  const backdropRollFaces=onPart=>backdropFaces({count:Number(control('backdrop-layers')),material:control('backdrop-material'),
    sweep:control('backdrop-sweep'),layers:lighting.backdrops??defaultBackdropLayers(),wallZ:bounds.backZ,floorY:bounds.floorY,table},ids,onPart);
  const tableFaces=onPart=>studioTableFaces(table,ids,bounds.floorY,onPart);
  const programmerFaces=onPart=>control('stage-programmer-enabled')?stageProgrammerFaces(bounds,table,ids,onPart):[];
  const goboFaces=onPart=>studioGoboFaces(lighting.gobos,ids,onPart);
  const lightStandFaces=(fixtureRoles,onPart)=>{
    const state=lighting.lightStands;if(!state?.enabled)return [];
    return studioLightStandFaces(state,fixtureRoles,ids,bounds.floorY,{...options,onPart,
      fixtureVertices:role=>buildStudioFixtureFaces([role],housingMat,undefined,ids).flatMap(face=>face.slice(0,3))});
  };
  const backdropHolderFaces=onPart=>{
    const state=lighting.lightStands;if(!state?.enabled||!state.backdropSupport)return [];
    return backdropSupportFaces(backdropRollFaces().flatMap(face=>face.slice(0,3)),bounds.floorY,ids[state.material],onPart,options);
  };
  const shopLightGeometry=()=>{
    const state=lighting.shopLights;
    const geometry=studioShopLightGeometry(state,ids,shopLightEmitterAllocations(state,fixtureMaterialBase+STUDIO_FIXED_MATERIAL_ROWS),options);
    const rails=studioLightingRackGeometry(lighting.lightingRacks,ids,{...options,topMaterial:state?.topMaterial});
    return {...geometry,faces:[...geometry.faces,...rails.faces],parts:[...geometry.parts,...rails.parts]};
  };
  const shopLightFaces=onPart=>{const geometry=shopLightGeometry();for(const part of geometry.parts)onPart?.(part);return geometry.faces;};
  return {roomFaces,backdropRollFaces,tableFaces,programmerFaces,goboFaces,lightStandFaces,backdropHolderFaces,shopLightGeometry,shopLightFaces};
}
// Canonical ownership shared by photo material-engine groups and raster receivers.
export const STUDIO_SCENE_PART_GROUP=Object.freeze({
  fixture:'stage',room:'room',backdrop:'room',backdropHolder:'room',
  table:'stage',programmer:'stage',gobo:'room',lightStand:'stage',shopLight:'room',
});
export function buildStudioPhotoFixtures({lighting,materials,subject,geometrySubject=subject,rayOriginM,shopEmissionProfiles,solo=null,getOpticalRecord,lightFieldInformation,timedExposure=false}) {
  const {ids,pbr:sourcePbr,fixtureMaterialBase:FIXTURE_MAT_BASE}=materials;
  const FIXTURE_HOUSING_MAT=FIXTURE_MAT_BASE+STUDIO_LIGHTS.length;
  const {fixtureRoles,roleColor,roleStrength}=computeStudioFixtureRoles({lighting,ids,fixtureMaterialBase:FIXTURE_MAT_BASE,subjectPosition:subject.subjectCenter,solo,emissionProfiles:shopEmissionProfiles});
  const geometry=createStudioSceneGeometry({lighting,materials,subject,geometrySubject});
  const parts=[];
  const collect=(group,prefix,builder,sourcePrefix)=>{
    const captured=[];const faces=builder(part=>captured.push({...part,group,label:`${prefix} ${part.id}`,
      ...(lightFieldInformation===RESPONSE_INFORMATION?{responseSourceId:`${sourcePrefix}/${part.id}`}:{})}));
    if(captured.reduce((n,p)=>n+p.faces.length,0)!==faces.length)
      throw new Error(`${prefix}: generated scene parts do not cover the actual triangles.`);
    parts.push(...captured);return faces;
  };
  collect(STUDIO_SCENE_PART_GROUP.fixture,'Light fixture',cb=>buildStudioFixtureFaces(fixtureRoles,FIXTURE_HOUSING_MAT,cb,ids),'fixture');
  collect(STUDIO_SCENE_PART_GROUP.room,'Room',cb=>geometry.roomFaces(cb),'room');
  collect(STUDIO_SCENE_PART_GROUP.backdrop,'Backdrop',cb=>geometry.backdropRollFaces(cb),'backdrop');
  collect(STUDIO_SCENE_PART_GROUP.backdropHolder,'Backdrop holder',cb=>geometry.backdropHolderFaces(cb),'backdrop-holder');
  collect(STUDIO_SCENE_PART_GROUP.table,'Table',cb=>geometry.tableFaces(cb),'table');
  collect(STUDIO_SCENE_PART_GROUP.programmer,'Stage programmer',cb=>geometry.programmerFaces(cb),'programmer');
  collect(STUDIO_SCENE_PART_GROUP.gobo,'Gobo',cb=>geometry.goboFaces(cb),'gobo');
  collect(STUDIO_SCENE_PART_GROUP.lightStand,'Light stand',cb=>geometry.lightStandFaces(fixtureRoles,cb),'stand');
  collect(STUDIO_SCENE_PART_GROUP.shopLight,'Shop light',cb=>geometry.shopLightFaces(cb),'shop');
  return packStudioPhotoParts({parts,materials,subject,rayOriginM,lighting,roleColor,roleStrength,shopEmissionProfiles,getOpticalRecord,lightFieldInformation,timedExposure});
}

// Compact the actual authored parts. This is also the narrow packing seam for
// CPU tests of individual scene owners; geometry remains owned by the builders.
export function packStudioPhotoParts({parts,materials,subject,rayOriginM,lighting,roleColor,roleStrength,shopEmissionProfiles,getOpticalRecord,lightFieldInformation,timedExposure=false}) {
  const {pbr:sourcePbr,fixtureMaterialBase:FIXTURE_MAT_BASE}=materials;
  // Use the canonical buffer and the same per-light emission writer as Phong.
  // The compact upload only rekeys IDs. It never invents another material row.
  const stride=sourcePbr.stride,data=new Float32Array(sourcePbr.data);
  writeStudioFixtureRows(data,stride,FIXTURE_MAT_BASE,roleColor,roleStrength);
  writeShopLightEmitterRows(data,stride,FIXTURE_MAT_BASE+STUDIO_FIXED_MATERIAL_ROWS,lighting.shopLights,{channel:"exposure",profiles:shopEmissionProfiles,timedExposure});
  const used=[...new Set([...Array.from({length:STUDIO_LIGHTS.length+1},(_,i)=>FIXTURE_MAT_BASE+i),
    ...parts.flatMap(p=>p.faces.map(f=>f[3]))])];
  const materialIndex=new Map(used.map((id,i)=>[id,i]));
  const pbr=used.flatMap(id=>Array.from(data.slice(id*stride,id*stride+16)));
  const scale=subject.unitM,toMeters=p=>rayOriginM.map((c,i)=>c+p[i]*scale),vertices=[],materialIds=[];
  for(const part of parts)for(const [a,b,c,id]of part.faces) {
    vertices.push(...toMeters(a),...toMeters(b),...toMeters(c));materialIds.push(materialIndex.get(id));
  }
  const material_engines=sceneMaterialEngineContract({engines:lighting.sceneEngines,
    parts,materialIndex,pbr:{data,stride},getOpticalRecord});
  const emitter_profiles=shopLightEmissionRecords(lighting.shopLights,FIXTURE_MAT_BASE+STUDIO_FIXED_MATERIAL_ROWS,materialIndex,shopEmissionProfiles);
  const baseProfile=(shopEmissionProfiles?.records??[]).find(row=>row.key==='led_cool_white');
  if(!baseProfile)throw Error('The base Studio emitter material profile is unavailable');
  const filterIds=['key-filter-left','key-filter-right','background-filter','kicker-left-filter',
    'kicker-right-filter','fill-lower-filter','fill-upper-filter','table-left-filter',
    'table-right-filter','under-filter'];
  for(let i=0;i<STUDIO_LIGHTS.length;i++){
    const material=materialIndex.get(FIXTURE_MAT_BASE+i);
    if(material===undefined)continue;
    const filterId=filterIds[i];
    emitter_profiles.records.push({material,profile:baseProfile.key,revision:baseProfile.revision,
      gain:roleStrength[i],filter:lighting.controls[`${filterId}-off`]?null:roleColor[i]});
  }
  const markerProfile=(shopEmissionProfiles?.records??[]).find(row=>row.key==='studio_calibration_nir850');
  for(const [name,id]of Object.entries(materials.ids))if(name.endsWith('__nir850')){
    const material=materialIndex.get(id);
    if(material===undefined)continue;
    if(!markerProfile)throw Error('The native 850 nm calibration profile is unavailable');
    emitter_profiles.records.push({material,profile:markerProfile.key,revision:markerProfile.revision,
      gain:1,filter:[1,1,1]});
  }
  const shopRows=timedExposure?shopLightEmitterAllocations(lighting.shopLights,FIXTURE_MAT_BASE+STUDIO_FIXED_MATERIAL_ROWS):[];
  const shop_material_ids=shopRows.map(row=>materialIndex.get(row.materialId)).filter(id=>id!==undefined);
  const trigger_emitters=shopRows.flatMap(row=>{
    const fixture=lighting.shopLights?.fixtures?.find(value=>value.id===row.fixtureId);
    const material=materialIndex.get(row.materialId);
    return fixture?.enabled&&fixture.exposureMode==='burst'&&material!==undefined?
      [{material,receiver:`shop:${fixture.id}`,gain:fixture.sourceLevel*fixture.dimmer}]:[];
  });
  const shop_receiver_candidates=shopRows.flatMap(row=>{
    const fixture=lighting.shopLights?.fixtures?.find(value=>value.id===row.fixtureId);
    const material=materialIndex.get(row.materialId);
    return fixture?.enabled&&material!==undefined?
      [{material,receiver:`shop:${fixture.id}`,gain:fixture.sourceLevel*fixture.dimmer}]:[];
  });
  const response_lineage=lightFieldInformation===RESPONSE_INFORMATION?fixtureResponseLineage(parts):null;
  return {vertices,material_ids:materialIds,pbr,material_engines,emitter_profiles,
    ...(timedExposure?{shop_material_ids,trigger_emitters,shop_receiver_candidates}:{}),...(response_lineage?{response_lineage}:{})};
}
export function initializeStudioFixtureRows(data,pbrStride,FIXTURE_MAT_BASE) {
    const base=(slot)=>(FIXTURE_MAT_BASE+slot)*pbrStride;
    for (let slot=0;slot<STUDIO_LIGHTS.length;slot++) {
      const b=base(slot);
      for (let i=0;i<pbrStride;i++) data[b+i]=0;
      data[b+3]=1; data[b+7]=1; // roughness, opacity
    }
    {
      const b=base(STUDIO_LIGHTS.length);
      for (let i=0;i<pbrStride;i++) data[b+i]=0;
      data[b+0]=0.03; data[b+1]=0.03; data[b+2]=0.035; // near-black albedo
      data[b+3]=0.8; data[b+7]=1; // roughness, opacity; emission stays 0
    }
    // Debug labels retain their reserved row.
    {
      const b=base(STUDIO_LIGHTS.length+1);
      for (let i=0;i<pbrStride;i++) data[b+i]=0;
      data[b+3]=1; data[b+7]=1;
      data[b+8]=4; data[b+9]=0.3; data[b+10]=0.3; // bright red debug label, always visible
    }
  return data;
}

// Asset tables and saved object rows retain their original IDs and Float32
// storage. Scene material edits override the same shared rows as page restore.
export function studioSceneMaterials(assets,objectSnapshot,lighting) {
  const ids=assets.material_index,extra=studioMaterialExtras(assets),fixtureMaterialBase=studioMaterialBase(assets,extra);
  const tables=Object.fromEntries(Object.entries(assets.materials).map(([name,table])=>{
    const data=new Float32Array((fixtureMaterialBase+STUDIO_RUNTIME_MATERIAL_ROWS)*table.floats_per_record);data.set(table.data);
    return [name,{table,data}];
  }));
  if(objectSnapshot) {
    const rows=savedMaterialRows(objectSnapshot.materials,ids,extra,tables);
    for(const [name,values]of Object.entries(rows))for(const row of values)tables[name].data.set(row.data,row.offset);
  }
  const edits=validateSceneMaterialEdits(lighting?.materialEdits,Object.keys(ids));
  for(const material of edits?.materials??[])for(const [name,row]of Object.entries(material.tables)) {
    if(row.stride!==tables[name].table.floats_per_record)throw new Error(`Shared ${name} material stride changed.`);
    tables[name].data.set(row.data,ids[material.name]*row.stride);
  }
  const data=tables.pbr.data,stride=tables.pbr.table.floats_per_record;
  initializeStudioFixtureRows(data,stride,fixtureMaterialBase);
  return {ids,pbr:{data,stride},fixtureMaterialBase};
}

/** Build the exact exposure scene around an already prepared object snapshot.
 * lighting is Studio's resolved exposure profile, not DOM/control objects.
 * camera and rayOriginM come from the existing photographic camera owner.
 * Object hashing/preparation and photo packing/submission remain caller-owned.
 */
export function buildStudioScene({lighting,objectSnapshot,materials,camera,rayOriginM,shopEmissionProfiles,solo=null,timedExposure=false}) {
  if(lighting?.version!==1||!lighting.controls||!lighting.poses||!lighting.table||!lighting.scale||!lighting.shopLights)
    throw new TypeError('A complete resolved Studio lighting profile is required.');
  if(!objectSnapshot?.geometry_sha256)throw new TypeError('A prepared object snapshot is required.');
  if(!Array.isArray(rayOriginM)||rayOriginM.length!==3||!rayOriginM.every(Number.isFinite))throw new TypeError('An explicit ray origin is required.');
  for(const name of ['eye_m','target_m'])if(!Array.isArray(camera?.[name])||camera[name].length!==3||!camera[name].every(Number.isFinite))throw new TypeError('An explicit photographic camera is required.');
  if(!(camera.fov_degrees>0&&camera.fov_degrees<179))throw new TypeError('Invalid photographic field of view.');
  const pbr=materials?.pbr,base=materials?.fixtureMaterialBase;
  if(!materials?.ids||!Number.isSafeInteger(base)||base<0||!Number.isSafeInteger(pbr?.stride)||pbr.stride<16||
      !(Array.isArray(pbr.data)||ArrayBuffer.isView(pbr.data))||pbr.data.length%pbr.stride||
      pbr.data.length<(base+STUDIO_FIXED_MATERIAL_ROWS)*pbr.stride||!Array.from(pbr.data).every(Number.isFinite)||
      !Object.values(materials.ids).every(id=>Number.isSafeInteger(id)&&id>=0&&id<base))
    throw new TypeError('Exact catalogue IDs and finite Studio PBR rows, including reserved fixtures, are required.');
  // Reuse the page's complete validator, with declarative field metadata rather
  // than elements. Unknown materials and out-of-range controls must not reach
  // the legacy geometry builders' fallback behavior through this public entry.
  const names=Object.keys(materials.ids);
  lighting=validateStudioLighting(lighting,studioLightingFields(names),names);
  const subject=studioScaleTransform(lighting.scale,Number(objectSnapshot.physical['radius-mm']),lighting.table.enabled);
  const records=new Map((lighting.materialEdits?.materials??[]).filter(m=>m.optical).map(m=>[materials.ids[m.name],m.optical]));
  const fixtures=buildStudioPhotoFixtures({lighting,materials,subject,rayOriginM,shopEmissionProfiles,solo,timedExposure,getOpticalRecord:id=>records.get(id)});
  return {fixtures,object_center_m:rayOriginM.map((v,i)=>v+subject.subjectCenter[i]*subject.unitM),camera:structuredClone(camera),objectSnapshot};
}
