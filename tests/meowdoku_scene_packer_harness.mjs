import {studioDisplayedSubjectCenter,linkedPhotoOrbit,PhotographicCameraBinding} from '../examples/meowdoku/web/marble_camera_binding.js';
// Boot inputs for tests that execute the real sceneForPhoto without a GL app.
import {sceneMaterialEngineContract} from '../examples/meowdoku/web/marble_scene_materials.js';
import {writeStudioFixtureRows} from '../examples/meowdoku/web/marble_studio_scene.js';
import {shopLightEmitterAllocations,writeShopLightEmitterRows,studioShopLightGeometry} from '../examples/meowdoku/web/marble_shop_lights.js';
import {STUDIO_FIXED_MATERIAL_ROWS,STUDIO_RUNTIME_MATERIAL_ROWS} from '../examples/meowdoku/web/marble_materials.js';
import {backdropSupportFaces} from '../examples/meowdoku/web/marble_backdrop_supports.js';
import * as builder from '../examples/meowdoku/web/studio_scene_builder.js';
import {LIGHTING_PROFILE_FIELDS} from '../examples/meowdoku/web/marble_studio_lighting.js';
import {STUDIO_LIGHTS} from '../examples/meowdoku/web/marble_light_strengths.js';
import {defaultStudioTable} from '../examples/meowdoku/web/marble_studio_table.js';
import {defaultFixtureSizes} from '../examples/meowdoku/web/marble_fixture_sizes.js';
import {defaultLightTools} from '../examples/meowdoku/web/marble_light_tools.js';

// Adapt the older focused unit-test inputs to the production value APIs.
// DOM doubles belong only to the tests of controls/camera event handlers.
// Geometry and material packing execute the imported production owners.
export function installStudioSceneBuilders(context) {
  context.shopEmissionProfiles??=undefined;
  context.studioRenderState??=channel=>context.studioLighting?.renderState?.(channel)??null;
  context.studioControlValue??=(id,channel='studio')=>{
    const state=context.studioRenderState(channel),el=context.$?.(id);
    return state?state.controls[id]:el?.type==='checkbox'?el.checked:el?.value;
  };
  const lighting=(channel='studio')=>{
    const state=context.studioRenderState(channel);
    if(state){const sizes=state.fixtureSizes??context.defaultFixtureSizes?.()??defaultFixtureSizes();
      return {...state,fixtureSizes:sizes,lightTools:state.lightTools??context.defaultLightTools?.(sizes)??defaultLightTools(sizes)};}
    const c=context.studioLighting,sizes=c?.fixtureSizes?.getState()??context.defaultFixtureSizes?.()??defaultFixtureSizes();
    const controls=Object.fromEntries(LIGHTING_PROFILE_FIELDS.map(id=>[id,context.studioControlValue(id,channel)]));
    const strengths=c?.strengths?.getStrengths?.(channel);
    if(strengths)STUDIO_LIGHTS.forEach((light,i)=>controls[light.strengthId]=strengths[i]*10);
    return {controls,fixtureSizes:sizes,lightTools:c?.lightTools?.getState()??context.defaultLightTools?.(sizes)??defaultLightTools(sizes),
      poses:c?.poses?.getState?.(),backdrops:c?.backdrops?.getState(),table:context.studioTableState?.()??c?.table?.getState()??defaultStudioTable(),
      gobos:c?.gobos?.getState(),lightStands:c?.lightStands?.getState(),shopLights:c?.shopLights?.state(),scale:c?.scale?.getState()};
  };
  context.studioBuilderLighting=lighting;
  context.studioBuilderOptions=(channel='studio',saved=false)=>({lighting:lighting(channel),
    subject:context.studioSubjectTransform?.(saved,channel)??{unitM:1,subjectCenter:[0,0,0]},
    geometrySubject:context.studioSubjectTransform?.(false,channel)??{unitM:1,subjectCenter:[0,0,0]},
    materials:{ids:context.ids??{},fixtureMaterialBase:context.FIXTURE_MAT_BASE,pbr:{data:context.tables?.pbr?.data,stride:context.tables?.pbr?.table.floats_per_record}}});
  context.createStudioSceneGeometry=builder.createStudioSceneGeometry;
  context.computeStudioFixtureRoles=builder.computeStudioFixtureRoles;
  context.buildStudioFixtureFaces=builder.buildStudioFixtureFaces;
  context.buildFixtureFaces??=(roles,housing,onPart)=>builder.buildStudioFixtureFaces(roles,housing,onPart,context.ids??{});
  context.computeFixtureRoles??=((subjectPosition=[0,0,0],channel='studio',poses=null,sizes=null,tools=null)=>{
    const state={...lighting(channel)};
    if(poses)state.poses=poses;if(sizes)state.fixtureSizes=sizes;if(tools)state.lightTools=tools;
    if(sizes&&!tools&&!context.studioRenderState(channel)?.lightTools&&!context.studioLighting?.lightTools)
      state.lightTools=(context.defaultLightTools??defaultLightTools)(sizes);
    // Existing controller unit tests supply already-resolved pose vectors.
    const resolved=!poses&&!state.poses?context.studioLighting?.poses?.getPoses?.(subjectPosition):null;
    if(resolved)state.table={...state.table,enabled:false};
    const result=builder.computeStudioFixtureRoles({lighting:state,ids:context.ids??{},fixtureMaterialBase:context.FIXTURE_MAT_BASE,subjectPosition,solo:context.studioLighting?.strengths?.getSolo?.(channel)});
    if(resolved)result.fixtureRoles.forEach((role,i)=>{role.pos=resolved[i].pos;role.dir=resolved[i].dir;});
    return result;
  });
  return context;
}

export function installScenePackerInputs(context) {
  Object.assign(context,{sceneMaterialEngineContract,writeStudioFixtureRows,opticalControls:null,
    STUDIO_FIXED_MATERIAL_ROWS,shopLightEmitterAllocations,writeShopLightEmitterRows,studioShopLightGeometry,backdropSupportFaces});
  context.studioRenderState??=()=>null;
  context.studioDisplayedSubjectCenter??=studioDisplayedSubjectCenter;
  context.linkedPhotoOrbit??=linkedPhotoOrbit;
  context.photographicCameraBinding??=new PhotographicCameraBinding();
  context.rollState??=null;
  context.isMarbleChase??=()=>false;
  context.FIXTURE_MAT_BASE??=context.tables.pbr.data.length/16;
  context.FIXTURE_HOUSING_MAT??=context.FIXTURE_MAT_BASE+context.STUDIO_LIGHTS.length;
  const t=context.tables.pbr,stride=t.table.floats_per_record;
  const rows=Math.max(context.FIXTURE_HOUSING_MAT+1,context.FIXTURE_MAT_BASE+STUDIO_RUNTIME_MATERIAL_ROWS);
  const data=new Float32Array(Math.max(t.data.length,rows*stride));data.set(t.data);t.data=data;
  for(let i=0;i<context.STUDIO_LIGHTS.length;i++){const b=(context.FIXTURE_MAT_BASE+i)*stride;data[b+3]=1;data[b+7]=1;}
  data.set([.03,.03,.035,.8,0,0,0,1,0,0,0,0,0,0,0,0],context.FIXTURE_HOUSING_MAT*stride);
  context.lightStandFacesForStudio??=()=>[];
  context.shopLightState??=()=>undefined; // Explicit legacy profile: no shop fixture geometry.
  context.shopLightFacesForStudio??=()=>[];
  context.backdropSupportFacesForStudio??=()=>[];
  installStudioSceneBuilders(context);
  installScenePacking(context);
}

export function installScenePacking(context) {
  // Packing-unit fixtures choose explicit real face arrays for the owner under
  // test. Full scene assembly/page delegation is covered in scene_builder_check.
  context.sceneForPhoto=(saved=true)=>{
    const {fixtureRoles,roleColor,roleStrength}=context.computeFixtureRoles(context.studioSubjectTransform(saved,'exposure').subjectCenter,'exposure');
    const parts=[];
    for(const [group,label,run]of [
      ['stage','Light fixture',cb=>context.buildFixtureFaces(fixtureRoles,context.FIXTURE_HOUSING_MAT,cb)],
      ['room','Room',cb=>context.checkerboardFaces('exposure',cb)],['room','Backdrop',cb=>context.backdropRollFaces('exposure',cb)],
      ['room','Backdrop holder',cb=>context.backdropSupportFacesForStudio('exposure',cb)],['stage','Table',cb=>context.tableFaces('exposure',cb)],
      ['room','Gobo',cb=>context.goboFaces('exposure',cb)],['stage','Light stand',cb=>context.lightStandFacesForStudio('exposure',fixtureRoles,cb)],
      ['room','Shop light',cb=>context.shopLightFacesForStudio('exposure',cb)],
    ])run(part=>parts.push({...part,group,label:`${label} ${part.id}`}));
    const state=context.studioRenderState('exposure')??{};
    return builder.packStudioPhotoParts({parts,materials:{fixtureMaterialBase:context.FIXTURE_MAT_BASE,pbr:{data:context.tables.pbr.data,stride:context.tables.pbr.table.floats_per_record}},
      subject:{unitM:context.studioPhotoScale(saved)},rayOriginM:context.OBJECT_CENTER,
      lighting:{...state,shopLights:context.shopLightState('exposure')},roleColor,roleStrength,shopEmissionProfiles:context.shopEmissionProfiles,
      getOpticalRecord:id=>context.opticalControls?.sceneRecord(id)});
  };
}
