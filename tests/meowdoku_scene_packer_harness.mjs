import {studioDisplayedSubjectCenter,linkedPhotoOrbit,PhotographicCameraBinding} from '../examples/meowdoku/web/marble_camera_binding.js';
// Boot inputs for tests that execute the real sceneForPhoto without a GL app.
import {sceneMaterialEngineContract} from '../examples/meowdoku/web/marble_scene_materials.js';
import {writeStudioFixtureRows} from '../examples/meowdoku/web/marble_studio_scene.js';
import {shopLightEmitterAllocations,writeShopLightEmitterRows,studioShopLightGeometry} from '../examples/meowdoku/web/marble_shop_lights.js';
import {STUDIO_FIXED_MATERIAL_ROWS,STUDIO_RUNTIME_MATERIAL_ROWS} from '../examples/meowdoku/web/marble_materials.js';
import {backdropSupportFaces} from '../examples/meowdoku/web/marble_backdrop_supports.js';
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
}
