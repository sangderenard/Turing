"""Actual Storybook native mesh/materials composed with verified Studio rig."""
from pathlib import Path
import json
import standalone_studio_browser_check as browser
browser.PAGE='''<!doctype html><meta charset="utf-8"><canvas></canvas><script type="module">
import {createLivingDataStudio} from '/examples/meowdoku/web/living_data_studio_adapter.js';
try{
 const mesh=await fetch('/build/dot-storybook-static-phong-20261110/realized-mesh.json').then(r=>r.json());
 const map={geometry:mesh.geometry};
 if(mesh.world_identity!=='introspection-world:python:meowdoku.storybook_environment.StorybookEnvironment')throw Error('Wrong native map');
 const asset=await fetch('/build/dot-studio-game-transfer-20261110/complete-studio.scene-asset.json').then(r=>r.json());
 const runner=await createLivingDataStudio({canvas:document.querySelector('canvas'),mesh,studioSceneAsset:asset});
 const ids=['python:meowdoku.storybook_environment.LanternCottage','python:meowdoku.storybook_environment.ClockworkTower'];
 const images=[];
 for(const target of ids){
  const box=map.geometry.boxes.find(b=>b.identity===target);if(!box)throw Error('Missing native target');
  const [x,z]=box.center,y=Math.max(1,box.height*.6),eyeZ=z-box.half_extent[1]-3;
  const near=.04,far=128,f=1/.70;
  const camera={view:[-1,0,0,0,0,1,0,0,0,0,-1,0,x,-y,eyeZ,1],
   projection:[f/(4/3),0,0,0,0,f,0,0,0,0,-(far+near)/(far-near),-1,0,0,-2*far*near/(far-near),0],width:320,height:240};
  const baseline={...runner.render(camera),target,lighting:'original captured rig'};
  const lightTransform=[1,0,0,0,0,0,1,0,0,-1,0,0,x,1,eyeZ+4,1];
  const moved=await createLivingDataStudio({canvas:document.querySelector('canvas'),mesh,studioSceneAsset:asset,lightTransform});
  const lit={...moved.render(camera),target,lighting:'explicitly rotated and translated captured rig'};
  if(baseline.data_url===lit.data_url)throw Error('Rig translation did not alter pixels');
  if(baseline.composition.mesh_sha256!==lit.composition.mesh_sha256)throw Error('Rig translation changed geometry/material source');
  images.push(baseline,lit);moved.dispose();
 }
 if(images[1].data_url===images[3].data_url)throw Error('Native target cameras produced identical pixels');
 runner.dispose();
 await fetch('/receipt',{method:'POST',body:JSON.stringify({ok:true,kind:'explicit-native-map-Studio-rig-composition',world_identity:mesh.world_identity,source_sha256:mesh.source_sha256,rig_identity:asset.identity,vertices:mesh.vertices.length/9,images})});
}catch(e){await fetch('/receipt',{method:'POST',body:JSON.stringify({ok:false,error:String(e),stack:e.stack})});}
</script>'''
if __name__=='__main__':print(json.dumps(browser.run(browser.ROOT/'build/dot-storybook-studio-composition-20261111'),indent=2))
