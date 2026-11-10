import {StudioRangefinder} from '../examples/meowdoku/web/marble_rangefinder.js';
import {createRangefinderPanel} from '../examples/meowdoku/web/marble_rangefinder_panel.js';
import {createStudioLensControls} from '../examples/meowdoku/web/marble_lens_controls.js';
import {packStudioScene} from '../examples/meowdoku/web/marble_studio_scene.js';
const $=id=>document.getElementById(id),assert=(test,message)=>{if(!test)throw Error(message);};
const identity=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const face=(z,mat)=>[[-.1,-.1,z],[.1,-.1,z],[0,.1,z],mat,[[0,0,1],[0,0,1],[0,0,1]]];
const glass=[.5,.5,.5,.1,0,1,1.5,0,0,0,0,0,0,0,0,0],opaque=[.5,.5,.5,.8,0,0,1,1,0,0,0,0,0,0,0,0];
let panel;
try {
 const capture={revision:1,state:packStudioScene({channel:'studio',unitM:1,rayOriginM:[0,0,.18],
  pbr:{data:[...glass,...opaque],stride:16},parts:[{id:'glass',faces:[face(-.02,0)]},{id:'wall',faces:[face(-.04,1)]},{id:'rear',faces:[face(-.06,0)]}],
  camera:{view:identity,aspect:1,fovDegrees:50},authoring:{},room:null})};
 const lenses=createStudioLensControls(document,{fetcher:async()=>new Response(JSON.stringify({schema:'studio-lens-catalog-v1',references:[],
  profiles:[{key:'test_macro',source_sha256:'test-profile',controls:{focal_length:{minimum:.05,maximum:.05,default:.05},focus_distance:{minimum:.23},aperture:{minimum:4}}}]}))});
 await lenses.ready;
 panel=createRangefinderPanel({doc:document,photo:{applyLensCamera:lenses.applyCamera,focusLensAt:lenses.focusAt},getScene:()=>capture,
  createBackend:()=>StudioRangefinder.create(new URL('../build/dot-client-ray-20261109/studio-compute/',import.meta.url))});
 document.body.append(panel.root);
 const result=await panel.read({u:.5,v:.5,spread:0});panel.accept(result);
 assert($('camera-range-hit').options.length===3,'all hit options');assert($('camera-range-hit').options[2].textContent.includes('behind opaque'),'occluded hit labeled');
 $('camera-range-hit').value='1';$('camera-range-apply').click();
 assert($('photo-focus-mm').value==='110','no original-mode silent no-op claimed');
 assert($('camera-range-status').textContent.includes('explicitly enable'),'explicit custom permission');
 $('camera-range-custom').checked=true;$('camera-range-apply').click();
 assert($('photo-lens').value==='custom','explicit custom switch');assert(Math.abs(+$('photo-focus-mm').value-40)<.001,'selected second-hit axial focus applied');
 const applied=lenses.applyCamera(result.camera);assert(Math.abs(applied.fov_degrees-50)<1e-10,'FOV retained');assert(Math.abs(applied.aperture_radius_m-.002)<1e-10,'pupil retained');
 panel.accept(result);capture.revision=2;$('camera-range-apply').click();assert($('camera-range-status').textContent.includes('changed'),'stale scene refused');
 capture.revision=1;$('photo-lens').value='test_macro';$('photo-lens').dispatchEvent(new Event('change'));$('camera-range-custom').checked=false;
 panel.accept(result);$('camera-range-apply').click();assert($('photo-lens').value==='test_macro','profile never silently replaced');
 assert(+$('photo-focus-mm').value>=230,'source minimum retained');assert($('camera-range-status').textContent.includes('between'),'profile close-focus refusal');
 panel.invalidate();assert($('camera-range-apply').disabled,'invalidate disables apply');
 $('result').textContent=JSON.stringify({passed:true,actualWasm:true,actualLensControls:true,hitCount:result.hits.length,
  tests:['all-hit choices','hidden surface label','explicit custom','second-hit axial focus','FOV/pupil retained','stale rejection','source minimum focus','invalidation']});
 document.title='PASS';
} catch(error){$('result').textContent=JSON.stringify({passed:false,error:error.stack});document.title='FAIL';}
finally {panel?.dispose();}
